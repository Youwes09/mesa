/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx3_machine.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"

#define AGX_MAX_PENDING (8)

/* Most scoreboard slots of any target */
#define AGX_MAX_SLOTS (6)

/*
 * Returns whether an instruction is asynchronous and needs a scoreboard slot
 */
static bool
instr_is_async(agx_instr *I)
{
   return agx_opcodes_info[I->op].immediates & AGX_IMMEDIATE_SCOREBOARD;
}

/* Async behaviour: fixed slot, completion class, sources read after issue */
struct producer {
   bool async;
   int fixed_slot;
   enum agx3_completion_class completion;
   bool reads_sources;
};

static struct producer
classified_producer(const agx_instr *I)
{
   struct producer p = {
      .async = true,
      .fixed_slot = -1,
      .completion = AGX3_COMPLETION_OTHER,
   };

   switch (I->op) {
   case AGX_OPCODE_DEVICE_LOAD:
      p.completion = AGX3_COMPLETION_MEMORY;
      return p;

   case AGX_OPCODE_ATOMIC:
      /* Only a returning atomic has a result to wait for */
      p.async = !agx_is_null(I->dest[0]);
      p.completion = AGX3_COMPLETION_MEMORY;
      return p;

   case AGX_OPCODE_LOCAL_LOAD:
      p.fixed_slot = 5;
      p.completion = AGX3_COMPLETION_MEMORY;
      return p;

   case AGX_OPCODE_LOCAL_ATOMIC:
      p.fixed_slot = 0;
      p.completion = AGX3_COMPLETION_MEMORY;
      return p;

   case AGX_OPCODE_STACK_STORE:
      /* Indexed stores destroy their data register instead */
      p.async = I->src[1].type == AGX_INDEX_IMMEDIATE;
      p.reads_sources = true;
      return p;

   case AGX_OPCODE_STACK_LOAD:
      /* Indexed loads complete on slot 6 */
      if (I->src[0].type != AGX_INDEX_IMMEDIATE)
         p.fixed_slot = 5;
      return p;

   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
      /* Parameters are read until the result completes */
      p.reads_sources = true;
      return p;

   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_LDCF:
   case AGX_OPCODE_IMAGE_LOAD:
      return p;

   case AGX_OPCODE_GET_SR:
   case AGX_OPCODE_GET_SR_COVERAGE:
   case AGX_OPCODE_GET_SR_BARRIER:
      /* Coverage, vertex and instance IDs are read asynchronously */
      p.async = I->sr == AGX_SR_COVERAGE_MASK || I->sr == AGX_SR_VERTEX_ID ||
                I->sr == AGX_SR_INSTANCE_ID;
      p.fixed_slot = 0;
      return p;

   case AGX_OPCODE_MOV:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_PUBLICATION_PAIR:
      /* Export register writes complete on export group 0 */
      p.async = I->dest[0].reg_class == RA_EXPORT;
      p.fixed_slot = 0;
      p.completion = AGX3_COMPLETION_PUBLICATION;
      return p;

   default:
      p.async = false;
      return p;
   }
}

static struct producer
producer(agx_context *ctx, agx_instr *I)
{
   if (ctx->target.dependency_waits)
      return classified_producer(I);

   return (struct producer){.async = instr_is_async(I), .fixed_slot = -1};
}

struct slot {
   /* Set of registers this slot is currently writing */
   BITSET_DECLARE(writes, AGX_NUM_REGS);

   /* Set of export registers this slot is currently writing */
   BITSET_DECLARE(export_writes, AGX_NUM_EXPORT_REGS);

   /* Set of registers this slot is still reading */
   BITSET_DECLARE(reads, AGX_NUM_REGS);

   /* Set of export registers this slot is still reading */
   BITSET_DECLARE(export_reads, AGX_NUM_EXPORT_REGS);

   /* Private frame words this slot is still storing */
   BITSET_DECLARE(frame_writes, AGX_NUM_MODELED_REGS / 2);
   bool any_frame_writes;

   /* Completion classes of the pending messages */
   uint8_t completion;

   /* Number of pending messages on this slot. Must not exceed
    * AGX_MAX_PENDING for correct results.
    */
   uint8_t nr_pending;
};

/* Slots a barrier waits for: all, or those of the classes it orders */
static uint8_t
prior_completion(agx_context *ctx, agx_instr *I, struct slot *slots)
{
   uint8_t classes = 0;
   uint8_t mask = 0;

   if (ctx->target.dependency_waits) {
      enum agx3_encoding form = agx3_encoding_for(I);
      if (form == AGX3_ENC_PSEUDO)
         return 0;

      const struct agx3_encoding_info *info = agx3_encoding_info(form);
      if (!info->requires_prior_completion)
         return 0;

      classes = ~info->prior_completion_exempt;

      /* The acquire and raster order fences also wait for loads */
      if (I->op == AGX_OPCODE_MEMORY_BARRIER_ACQUIRE ||
          I->op == AGX_OPCODE_RASTER_ORDER_FENCE)
         classes = ~0;
   } else if (I->op == AGX_OPCODE_THREADGROUP_BARRIER ||
              I->op == AGX_OPCODE_MEMORY_BARRIER) {
      classes = ~0;
   } else {
      return 0;
   }

   for (unsigned slot = 0; slot < ctx->target.scoreboard_slots; ++slot) {
      if (slots[slot].nr_pending &&
          (!ctx->target.dependency_waits || (slots[slot].completion & classes)))
         mask |= BITSET_BIT(slot);
   }

   return mask;
}

/* Whether a register dies after I, as the register cache sets release flags */
static bool
dies_at(agx_block *block, agx_instr *I, unsigned reg, unsigned count,
        enum ra_class cls)
{
   if (block->divergent)
      return false;

   agx_foreach_reg_dest(I, d) {
      agx_index dest = I->dest[d];
      if (dest.reg_class == cls && dest.value <= reg &&
          reg < dest.value + agx_index_size_16(dest))
         return true;
   }

   for (agx_instr *J = list_entry(I->link.next, agx_instr, link);
        &J->link != &block->instructions;
        J = list_entry(J->link.next, agx_instr, link)) {

      agx_foreach_reg_src(J, s) {
         agx_index src = J->src[s];
         if (src.reg_class == cls && src.value < reg + count &&
             reg < src.value + agx_index_size_16(src))
            return false;
      }

      agx_foreach_reg_dest(J, d) {
         agx_index dest = J->dest[d];
         if (dest.reg_class == cls && dest.value <= reg &&
             reg + count <= dest.value + agx_index_size_16(dest))
            return true;
      }
   }

   if (cls == RA_EXPORT)
      return true;

   return !BITSET_TEST_COUNT(block->reg_live_out, reg, count);
}

/* Fold a wait into the dependency field. Released operands must die here */
static bool
fold_wait(agx_block *block, agx_instr *I, unsigned slot, struct slot *s)
{
   if (I->wait)
      return false;

   I->wait = true;
   I->wait_slot = slot;

   enum agx3_encoding form = agx3_encoding_for(I);
   bool ok = form != AGX3_ENC_PSEUDO;

   if (ok) {
      const struct agx3_encoding_info *info = agx3_encoding_info(form);
      ok = info->dependency_layout != AGX3_DEPENDENCY_NONE;

      if (ok && !info->dependency_retains_sources) {
         agx_foreach_reg_src(I, i) {
            agx_index src = I->src[i];
            unsigned count = agx_index_size_16(src);

            if (src.reg_class == RA_EXPORT &&
                BITSET_TEST_COUNT(s->export_writes, src.value, count)) {
               ok &= I->op == AGX_OPCODE_ST_VARY &&
                     dies_at(block, I, src.value, count, RA_EXPORT);
            } else if (src.reg_class == RA_GPR &&
                       BITSET_TEST_COUNT(s->writes, src.value, count)) {
               ok &= dies_at(block, I, src.value, count, RA_GPR);
            }
         }
      }
   }

   if (!ok)
      I->wait = false;

   return ok;
}

/* Texture operations and image stores wait for export group 0 themselves */
static bool
waits_for_exports(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
          I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_WRITE ||
          I->op == AGX_OPCODE_BLOCK_IMAGE_STORE;
}

/*
 * Insert waits within a block to stall after every async instruction. Useful
 * for debugging.
 */
static void
agx_insert_waits_trivial(agx_context *ctx, agx_block *block)
{
   agx_foreach_instr_in_block_safe(block, I) {
      struct producer p = producer(ctx, I);
      if (p.async) {
         if (p.fixed_slot >= 0)
            I->scoreboard = p.fixed_slot;

         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         agx_wait(&b, I->scoreboard);
      }
   }
}

/*
 * Insert waits within a block, assuming scoreboard slots have already been
 * assigned. This waits for everything at the end of the block, rather than
 * doing something more intelligent/global. This should be optimized.
 */
static void
agx_insert_waits_local(agx_context *ctx, agx_block *block)
{
   struct slot slots[AGX_MAX_SLOTS] = {0};
   unsigned nr_slots = ctx->target.scoreboard_slots;
   bool dependency = ctx->target.dependency_waits;

   /* Whether the block waits for every slot at its end, see below */
   bool waits_at_end =
      block != agx_exit_block(ctx) || (dependency && ctx->falls_through);

   assert(nr_slots <= ARRAY_SIZE(slots));

   agx_foreach_instr_in_block_safe(block, I) {
      uint8_t wait_mask = 0;
      struct producer p = producer(ctx, I);

      /* Check for read-after-write */
      bool exports_waited = dependency && waits_for_exports(I);

      /* An export reads its register after the block's final wait */
      bool reads_nothing =
         dependency && I->op == AGX_OPCODE_EXPORT && waits_at_end;

      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER || reads_nothing)
            continue;

         bool exp = I->src[s].reg_class == RA_EXPORT;
         unsigned nr_read = agx_index_size_16(I->src[s]);
         for (unsigned slot = 0; slot < nr_slots; ++slot) {
            BITSET_WORD *writes =
               exp ? slots[slot].export_writes : slots[slot].writes;

            if (exp && exports_waited && slot == 0)
               continue;

            if (BITSET_TEST_COUNT(writes, I->src[s].value, nr_read))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      if (exports_waited)
         BITSET_ZERO(slots[0].export_writes);

      /* Check for write-after-write and write-after-read */
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_REGISTER)
            continue;

         unsigned reg = I->dest[d].value;
         unsigned nr_writes = agx_index_size_16(I->dest[d]);
         for (unsigned slot = 0; slot < nr_slots; ++slot) {
            struct slot *s = &slots[slot];

            if (I->dest[d].reg_class == RA_EXPORT
                   ? (BITSET_TEST_COUNT(s->export_writes, reg, nr_writes) ||
                      BITSET_TEST_COUNT(s->export_reads, reg, nr_writes))
                   : (BITSET_TEST_COUNT(s->writes, reg, nr_writes) ||
                      BITSET_TEST_COUNT(s->reads, reg, nr_writes)))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Fills wait for pending saves of their word, or any word if indexed */
      if (dependency && I->op == AGX_OPCODE_STACK_LOAD) {
         bool indexed = I->src[0].type != AGX_INDEX_IMMEDIATE;

         for (unsigned slot = 0; slot < nr_slots; ++slot) {
            struct slot *s = &slots[slot];

            if (s->any_frame_writes &&
                (indexed || BITSET_TEST(s->frame_writes, I->src[0].value / 4)))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* A wait for an operand can fold into the instruction */
      uint8_t operand_mask = wait_mask;

      /* Check for barriers */
      wait_mask |= prior_completion(ctx, I, slots);

      /* Try to assign a free slot */
      if (p.async) {
         if (p.fixed_slot >= 0) {
            I->scoreboard = p.fixed_slot;
         } else {
            for (unsigned slot = 0; slot < nr_slots; ++slot) {
               if (slots[slot].nr_pending == 0) {
                  I->scoreboard = slot;
                  break;
               }
            }
         }
      }

      /* Check for slot overflow */
      if (p.async && slots[I->scoreboard].nr_pending >= AGX_MAX_PENDING)
         wait_mask |= BITSET_BIT(I->scoreboard);

      int folded = -1;
      if (dependency) {
         u_foreach_bit(slot, operand_mask) {
            if (fold_wait(block, I, slot, &slots[slot])) {
               folded = slot;
               break;
            }
         }
      }

      /* Insert the appropriate waits, clearing the slots */
      uint8_t unfolded = wait_mask & ~(folded >= 0 ? BITFIELD_BIT(folded) : 0);
      if (dependency && unfolded) {
         /* One instruction waits for every slot */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_wait_slots(&b, unfolded);
      }

      u_foreach_bit(slot, wait_mask) {
         if (slot != folded && !dependency) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_wait(&b, slot);
         }

         memset(&slots[slot], 0, sizeof(slots[slot]));
      }

      /* Record access */
      if (p.async) {
         struct slot *slot = &slots[I->scoreboard];

         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]))
               continue;

            assert(I->dest[d].type == AGX_INDEX_REGISTER);
            BITSET_WORD *writes = I->dest[d].reg_class == RA_EXPORT
                                     ? slot->export_writes
                                     : slot->writes;

            BITSET_SET_COUNT(writes, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         }

         if (p.reads_sources) {
            agx_foreach_reg_src(I, s) {
               if (I->src[s].reg_class == RA_EXPORT) {
                  BITSET_SET_COUNT(slot->export_reads, I->src[s].value,
                                   agx_index_size_16(I->src[s]));
               } else if (I->src[s].reg_class == RA_GPR &&
                          !waits_for_exports(I)) {
                  BITSET_SET_COUNT(slot->reads, I->src[s].value,
                                   agx_index_size_16(I->src[s]));
               }
            }
         }

         if (dependency && I->op == AGX_OPCODE_STACK_STORE) {
            assert(I->src[1].type == AGX_INDEX_IMMEDIATE && "indexed are sync");
            BITSET_SET(slot->frame_writes, I->src[1].value / 4);
            slot->any_frame_writes = true;
         }

         slot->completion |= p.completion;
         slot->nr_pending++;
      }
   }

   /* If there are outstanding messages, wait for them. We don't do this for the
    * exit block, though, since nothing else will execute in the shader so
    * waiting is pointless. With dependency waits, a part that does not stop is
    * followed by another, which must find its registers and exports written.
    */
   if (waits_at_end) {
      agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));
      uint8_t pending = 0;

      for (unsigned slot = 0; slot < nr_slots; ++slot) {
         if (slots[slot].nr_pending) {
            if (dependency)
               pending |= BITFIELD_BIT(slot);
            else
               agx_wait(&b, slot);
         }
      }

      if (pending)
         agx_wait_slots(&b, pending);
   }
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for the
 * appropriate hazard tracking.
 */
void
agx_insert_waits(agx_context *ctx)
{
   /* Folding a wait can depend on whether a source dies */
   if (ctx->target.dependency_waits)
      agx_compute_postra_liveness(ctx);

   agx_foreach_block(ctx, block) {
      if (agx_compiler_debug & AGX_DBG_WAIT)
         agx_insert_waits_trivial(ctx, block);
      else
         agx_insert_waits_local(ctx, block);
   }
}
