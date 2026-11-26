/*
 * Copyright 2025 Valve Corporation
 * Copyright 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "agx_compiler.h"
#include "shader_enums.h"

/* Liveness of GPRs. Export registers are a separate file */
static void
postra_liveness_ins(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_reg_dest(I, d) {
      if (I->dest[d].reg_class != RA_GPR)
         continue;

      unsigned reg = I->dest[d].value;
      BITSET_CLEAR_COUNT(live, reg, agx_index_size_16(I->dest[d]));
   }

   agx_foreach_reg_src(I, s) {
      if (I->src[s].reg_class != RA_GPR)
         continue;

      unsigned reg = I->src[s].value;
      BITSET_SET_COUNT(live, reg, agx_index_size_16(I->src[s]));
   }
}

/*
 * Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */
void
agx_compute_postra_liveness(agx_context *ctx)
{
   u_worklist worklist;
   agx_worklist_init(ctx, &worklist);

   agx_foreach_block(ctx, block) {
      BITSET_ZERO(block->reg_live_in);
      BITSET_ZERO(block->reg_live_out);

      agx_worklist_push_tail(&worklist, block);
   }

   while (!u_worklist_is_empty(&worklist)) {
      /* Pop off in reverse order since liveness is backwards */
      agx_block *blk = agx_worklist_pop_tail(&worklist);

      /* Calculate liveness locally */
      agx_foreach_successor(blk, succ) {
         BITSET_OR(blk->reg_live_out, blk->reg_live_out, succ->reg_live_in);
      }

      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, blk->reg_live_out, sizeof(live));

      agx_foreach_instr_in_block_rev(blk, ins) {
         postra_liveness_ins(live, ins);
      }

      if (BITSET_EQUAL(blk->reg_live_in, live))
         continue;

      /* We made progress, so we need to reprocess the predecessors */
      memcpy(blk->reg_live_in, live, sizeof(live));
      agx_foreach_predecessor(blk, pred) {
         agx_worklist_push_head(&worklist, *pred);
      }
   }

   u_worklist_fini(&worklist);
}

static bool
writes_reg(const agx_instr *I, unsigned reg)
{
   agx_foreach_reg_dest(I, d) {
      unsigned count = agx_index_size_16(I->dest[d]);

      /* Export registers are another file */
      if (I->dest[d].reg_class != RA_GPR)
         continue;

      if (reg >= I->dest[d].value && (reg - I->dest[d].value) < count)
         return true;
   }

   return false;
}

/* Special functions encode a hint of whether their result is read */
static bool
is_special(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_RCP:
   case AGX_OPCODE_RSQRT:
   case AGX_OPCODE_SRSQRT:
   case AGX_OPCODE_SIN_FACTOR:
   case AGX_OPCODE_EXP2:
   case AGX_OPCODE_LOG2:
   case AGX_OPCODE_FLOOR:
   case AGX_OPCODE_CEIL:
   case AGX_OPCODE_TRUNC:
   case AGX_OPCODE_ROUNDEVEN:
      return true;
   default:
      return false;
   }
}

/* Whether a non-store reads the result before it is overwritten */
static bool
result_read(agx_block *block, agx_instr *I)
{
   agx_index dest = I->dest[0];
   unsigned reg = dest.value, count = agx_index_size_16(dest);

   for (agx_instr *J = list_entry(I->link.next, agx_instr, link);
        &J->link != &block->instructions;
        J = list_entry(J->link.next, agx_instr, link)) {

      agx_foreach_reg_src(J, s) {
         agx_index src = J->src[s];
         if (src.reg_class != RA_GPR)
            continue;

         unsigned end = src.value + agx_index_size_16(src);
         if (src.value < reg + count && reg < end &&
             !(J->op == AGX_OPCODE_DEVICE_STORE && s == 0))
            return true;
      }

      agx_foreach_reg_dest(J, d) {
         if (J->dest[d].reg_class == RA_GPR && J->dest[d].value == reg &&
             agx_index_size_16(J->dest[d]) >= count)
            return false;
      }
   }

   return block->successors[0] != NULL;
}

/*
 * Mark last-use sources to allow the hardware to discard from the register
 * cache. Last use information follows immediately from (post-RA) liveness
 * analysis: a register is dead immediately after its last use.
 *
 * Mark cache hints on sources/destinations to encourage the hardware to make
 * better use of the register cache. This is a simple local analysis.
 */
void
agx_opt_register_cache(agx_context *ctx)
{
   /* Analyze the shader globally */
   agx_compute_postra_liveness(ctx);

   agx_foreach_block(ctx, block) {
      /* Live-set at each point in the program */
      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, block->reg_live_out, sizeof(live));

      /* Set of registers read "soon" by an ALU instruction. These are
       * candidates for the .cache bit.
       */
      BITSET_DECLARE(alu_reads, AGX_NUM_REGS) = {0};

      agx_foreach_instr_in_block_rev(block, I) {
         agx_foreach_reg_dest(I, d) {
            if (I->dest[d].reg_class != RA_GPR)
               continue;

            uint64_t reg = I->dest[d].value;
            unsigned nr = agx_index_size_16(I->dest[d]);

            I->dest[d].cache = BITSET_TEST(alu_reads, I->dest[d].value);
            BITSET_CLEAR_COUNT(alu_reads, reg, nr);
         }

         agx_foreach_reg_src(I, s) {
            if (I->src[s].reg_class == RA_GPR)
               I->src[s].cache = BITSET_TEST(alu_reads, I->src[s].value);
         }

         agx_foreach_reg_src(I, s) {
            if (I->src[s].reg_class != RA_GPR)
               continue;

            uint64_t reg = I->src[s].value;
            unsigned nr = agx_index_size_16(I->src[s]);

            /* If the register dead after this instruction, it's the last use.
             * That includes if the register is overwritten this cycle, but that
             * won't show up in the liveness analysis.
             */
            bool lu = !BITSET_TEST_COUNT(live, reg, nr) || writes_reg(I, reg);

            /* Handling divergent blocks would require physical CFG awareness.
             * Just bail for now, skipping this pass won't affect correctness.
             */
            I->src[s].discard = lu && !block->divergent;

            /* Only the low half is written, so the register stays in use */
            if (I->op == AGX_OPCODE_F32_TO_F16 && writes_reg(I, reg))
               I->src[s].discard = false;

            /* Mark any source read by an ALU instruction in the same block as
             * wanting a .cache hint. This is better than just marking
             * everything, although it overly hints for very long blocks and
             * underhints for registers used across block boundaries. It's
             * probably good enough, though, and it's not clear how to do much
             * better given our limited understanding of the hardware.
             */
            if (agx_is_alu(I))
               BITSET_SET_COUNT(alu_reads, reg, nr);

            assert(!(I->src[s].discard && I->src[s].cache));
         }

         postra_liveness_ins(live, I);
      }
   }
}

/*
 * Hint how results are consumed: whether another device load follows in
 * program order, and whether a special function's result is read.
 */
void
agx_mark_result_hints(agx_context *ctx)
{
   bool load_follows = false;
   agx_foreach_block_rev(ctx, block) {
      agx_foreach_instr_in_block_rev(block, I) {
         if (I->op == AGX_OPCODE_DEVICE_LOAD) {
            I->load_follows = load_follows;
            load_follows = true;
         } else if (is_special(I)) {
            I->result_read = result_read(block, I);
         }
      }
   }
}
