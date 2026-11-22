/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_test.h"

#include <gtest/gtest.h>

class RegisterAllocate : public testing::Test {
 protected:
   RegisterAllocate()
   {
      mem_ctx = ralloc_context(NULL);
      nir = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                           &agx_nir_options, "ra");
      ctx = rzalloc(mem_ctx, agx_context);
      list_inithead(&ctx->blocks);
      ctx->nir = nir.shader;
      ctx->stage = MESA_SHADER_COMPUTE;
      ctx->key = &key;
      ctx->target = *agx_get_isa_target(AGX_SHADER_ISA_APPLE9);
      block = agx_test_block(ctx);
      b = agx_init_builder(ctx, agx_after_block(block));
   }

   ~RegisterAllocate()
   {
      ralloc_free(nir.shader);
      ralloc_free(mem_ctx);
   }

   /* Give every operand an unrestricted constraint, to be adjusted */
   agx_reg_constraint *constrain(agx_instr *I)
   {
      unsigned n = I->nr_dests + I->nr_srcs;
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, n);
      for (unsigned i = 0; i < n; ++i)
         I->reg_constraints[i] = {0, UINT16_MAX, 1};
      return I->reg_constraints;
   }

   unsigned count_moves()
   {
      unsigned n = 0;
      agx_foreach_instr_global(ctx, I)
         n += I->op == AGX_OPCODE_MOV;
      return n;
   }

   agx_index export_temp(enum agx_size size)
   {
      agx_index x = agx_temp(ctx, size);
      x.reg_class = RA_EXPORT;
      return x;
   }

   void *mem_ctx;
   nir_builder nir;
   agx_shader_key key = {};
   agx_context *ctx;
   agx_block *block;
   agx_builder b;
};

/* Export values take no GPRs and may share register numbers with GPRs */
TEST_F(RegisterAllocate, ExportFileIsSeparate)
{
   const unsigned count = ctx->target.export_halves / 2;
   ASSERT_GT(2 * count, ctx->target.gpr_halves);

   agx_index x = agx_mov_imm(&b, 32, 42);
   agx_instr *exports[128];
   ASSERT_LE(count, ARRAY_SIZE(exports));

   for (unsigned i = 0; i < count; ++i)
      exports[i] = agx_mov_to(&b, export_temp(AGX_SIZE_32), x);

   /* Keep every export live until all are defined */
   agx_index sum = x;
   for (unsigned i = 0; i < count; ++i)
      sum = agx_iadd(&b, sum, exports[i]->dest[0], 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   EXPECT_EQ(ctx->export_reg_count, 2 * count);
   EXPECT_LT(ctx->max_reg, 8u);

   BITSET_DECLARE(used, AGX_NUM_EXPORT_REGS) = {0};
   for (unsigned i = 0; i < count; ++i) {
      agx_index e = exports[i]->dest[0];
      ASSERT_EQ(e.type, AGX_INDEX_REGISTER);
      ASSERT_EQ(e.reg_class, RA_EXPORT);
      ASSERT_EQ(e.value % 2, 0u);
      ASSERT_LT(e.value + 1, ctx->target.export_halves);
      ASSERT_FALSE(BITSET_TEST_COUNT(used, e.value, 2));
      BITSET_SET_COUNT(used, e.value, 2);
   }

   /* The first export and its GPR source both start at the bottom */
   EXPECT_EQ(exports[0]->dest[0].value, exports[0]->src[0].value);
}

/* A freed export register is reused, without touching the GPR file */
TEST_F(RegisterAllocate, ExportRegistersAreReused)
{
   agx_index x = agx_mov_imm(&b, 32, 7);
   agx_instr *first = agx_mov_to(&b, export_temp(AGX_SIZE_32), x);
   agx_iadd(&b, first->dest[0], x, 0);
   agx_instr *second = agx_mov_to(&b, export_temp(AGX_SIZE_32), x);
   agx_iadd(&b, second->dest[0], x, 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(first->dest[0].reg_class, RA_EXPORT);
   EXPECT_EQ(second->dest[0].reg_class, RA_EXPORT);
   EXPECT_EQ(first->dest[0].value, second->dest[0].value);
   EXPECT_EQ(ctx->export_reg_count, 2u);
}

/* An export value keeps its register on both edges of a join */
TEST_F(RegisterAllocate, ExportLiveAcrossJoin)
{
   agx_block *then_block = agx_test_block(ctx);
   agx_block *else_block = agx_test_block(ctx);
   agx_block *join = agx_test_block(ctx);
   agx_block_add_successor(block, then_block);
   agx_block_add_successor(block, else_block);
   agx_block_add_successor(then_block, join);
   agx_block_add_successor(else_block, join);

   agx_index x = agx_mov_imm(&b, 32, 1);
   agx_instr *def = agx_mov_to(&b, export_temp(AGX_SIZE_32), x);

   /* Allocate exports differently in the two branches */
   b.cursor = agx_after_block(then_block);
   agx_instr *then_export = agx_mov_to(&b, export_temp(AGX_SIZE_32), x);
   agx_iadd(&b, then_export->dest[0], x, 0);

   b.cursor = agx_after_block(else_block);
   agx_index y = agx_iadd(&b, x, agx_immediate(2), 0);
   agx_iadd(&b, y, x, 0);

   b.cursor = agx_after_block(join);
   agx_instr *use = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), def->dest[0],
                                agx_immediate(3), 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(use->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(use->src[0].reg_class, RA_EXPORT);
   EXPECT_EQ(use->src[0].value, def->dest[0].value);
   EXPECT_NE(then_export->dest[0].value, def->dest[0].value);

   /* No copies were needed on either edge */
   agx_foreach_instr_global(ctx, I) {
      EXPECT_FALSE(I->op == AGX_OPCODE_MOV &&
                   I->dest[0].reg_class == RA_EXPORT &&
                   I->src[0].reg_class == RA_EXPORT);
   }
}

static bool
overlaps(agx_index a, agx_index b)
{
   return a.value < b.value + agx_index_size_16(b) &&
          b.value < a.value + agx_index_size_16(a);
}

/* A clobbered source whose value is read later gets a copy to destroy */
TEST_F(RegisterAllocate, ClobberedSourceLiveAfterIsCopied)
{
   agx_index x = agx_mov_imm(&b, 32, 5);
   agx_instr *I =
      agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), x, agx_immediate(1), 0);
   constrain(I)[1].clobber = true;
   agx_instr *later =
      agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), x, I->dest[0], 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(count_moves(), 1u);
   EXPECT_FALSE(overlaps(I->src[0], later->src[0]));
}

/* A clobbered source at its last use is destroyed in place */
TEST_F(RegisterAllocate, ClobberedSourceDyingIsNotCopied)
{
   agx_instr *def = agx_mov_imm_to(&b, agx_temp(ctx, AGX_SIZE_32), 5);
   agx_instr *I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), def->dest[0],
                              agx_immediate(1), 0);
   constrain(I)[1].clobber = true;
   agx_iadd(&b, I->dest[0], agx_immediate(2), 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(count_moves(), 0u);
   EXPECT_EQ(I->src[0].value, def->dest[0].value);
}

/* Another source of the same instruction still needs the value */
TEST_F(RegisterAllocate, ClobberedSourceReadTwiceIsCopied)
{
   agx_index x = agx_mov_imm(&b, 32, 5);
   agx_instr *I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), x, x, 0);
   constrain(I)[1].clobber = true;
   agx_iadd(&b, I->dest[0], agx_immediate(2), 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(count_moves(), 1u);
   EXPECT_FALSE(overlaps(I->src[0], I->src[1]));
}

/* Only a destination without early clobber may take over a dying source */
TEST_F(RegisterAllocate, EarlyClobberDestinationAvoidsSources)
{
   agx_index x = agx_mov_imm(&b, 32, 1);
   agx_index y = agx_mov_imm(&b, 32, 2);
   agx_instr *plain = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), x, y, 0);

   agx_index u = agx_mov_imm(&b, 32, 3);
   agx_index v = agx_mov_imm(&b, 32, 4);
   agx_instr *early = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), u, v, 0);
   constrain(early)[0].early_clobber = true;

   agx_iadd(&b, plain->dest[0], early->dest[0], 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_TRUE(overlaps(plain->dest[0], plain->src[0]) ||
               overlaps(plain->dest[0], plain->src[1]));
   EXPECT_FALSE(overlaps(early->dest[0], early->src[0]));
   EXPECT_FALSE(overlaps(early->dest[0], early->src[1]));
}

/* In a full file, an early-clobber destination still avoids its sources */
TEST_F(RegisterAllocate, EarlyClobberDestinationInFullFile)
{
   ctx->target.gpr_halves = 16;

   /* Five live values, two dying sources and the destination fill 16 halves */
   agx_index live[5];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_mov_imm(&b, 32, i);

   agx_index x = agx_mov_imm(&b, 32, 10);
   agx_index y = agx_mov_imm(&b, 32, 11);
   agx_instr *I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), x, y, 0);
   constrain(I)[0].early_clobber = true;

   agx_index sum = I->dest[0];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      sum = agx_iadd(&b, sum, live[i], 0);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   EXPECT_LT(ctx->max_reg, 16u);
   EXPECT_FALSE(overlaps(I->dest[0], I->src[0]));
   EXPECT_FALSE(overlaps(I->dest[0], I->src[1]));
}

/* An export collect is placed in its reader's window and becomes moves */
TEST_F(RegisterAllocate, ExportBundleTakesReaderWindow)
{
   agx_index x[4];
   for (unsigned i = 0; i < 4; ++i)
      x[i] = agx_mov_imm(&b, 32, i);

   /* An export value live across the bundle, in the bottom of the file */
   agx_instr *other = agx_mov_to(&b, export_temp(AGX_SIZE_32), x[0]);

   agx_index bundle = agx_vec_temp(ctx, AGX_SIZE_32, 4);
   bundle.reg_class = RA_EXPORT;
   agx_instr *collect = agx_collect_to(&b, bundle, 4);
   for (unsigned i = 0; i < 4; ++i)
      collect->src[i] = x[i];

   /* Words 4 to 24, aligned to 4 words */
   agx_instr *use = agx_unit_test(&b, bundle);
   agx_reg_constraint *c = constrain(use);
   c[0] = {8, 48, 8};

   agx_iadd(&b, other->dest[0], x[1], 0);

   ASSERT_TRUE(agx_ra(ctx));
   ASSERT_EQ(use->src[0].type, AGX_INDEX_REGISTER);
   ASSERT_EQ(use->src[0].reg_class, RA_EXPORT);
   EXPECT_GE(use->src[0].value, 8u);
   EXPECT_LE(use->src[0].value, 48u);
   EXPECT_EQ(use->src[0].value % 8, 0u);
   EXPECT_FALSE(overlaps(use->src[0], other->dest[0]));

   /* The collect became one export move per word */
   unsigned moves = 0;
   agx_foreach_instr_global(ctx, I) {
      EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
      if (I->op == AGX_OPCODE_MOV && I->dest[0].reg_class == RA_EXPORT &&
          I->dest[0].value >= use->src[0].value &&
          I->dest[0].value < use->src[0].value + 8) {
         EXPECT_EQ(I->src[0].reg_class, RA_GPR);
         moves++;
      }
   }
   EXPECT_EQ(moves, 4u);
}

/* Spilling GPRs leaves export values in their registers */
TEST_F(RegisterAllocate, SpillingLeavesExportsAlone)
{
   ctx->target.gpr_halves = 16;

   /* Values that cannot be rematerialized */
   agx_index live[12];
   live[0] = agx_get_sr(&b, 32, AGX_SR_THREAD_POSITION_IN_GRID_X);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_iadd(&b, live[i - 1], agx_immediate(i), 0);

   agx_index bundle = agx_vec_temp(ctx, AGX_SIZE_32, 2);
   bundle.reg_class = RA_EXPORT;
   agx_instr *collect = agx_collect_to(&b, bundle, 2);
   collect->src[0] = live[0];
   collect->src[1] = live[1];
   agx_instr *use = agx_unit_test(&b, bundle);
   constrain(use)[0] = {0, 60, 4};

   agx_index sum = live[0];
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_iadd(&b, sum, live[i], 0);
   agx_unit_test(&b, sum);

   ASSERT_TRUE(agx_ra(ctx));
   EXPECT_GT(ctx->scratch_size_B, 0u);
   EXPECT_EQ(use->src[0].reg_class, RA_EXPORT);
   EXPECT_EQ(ctx->export_reg_count, 4u);

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_dest(I, d) {
         if (I->dest[d].reg_class == RA_MEM) {
            agx_foreach_src(I, s)
               EXPECT_NE(I->src[s].reg_class, RA_EXPORT);
         }
      }
   }
}
