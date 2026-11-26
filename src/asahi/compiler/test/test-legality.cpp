/*
 * Copyright 2026 Youwes09
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

/* Operand legality and register constraints on AGX3 */
class Agx3Legality : public testing::Test {
 protected:
   Agx3Legality()
   {
      mem_ctx = ralloc_context(NULL);
      b = builder(AGX_SHADER_ISA_APPLE9);
      ctx = b->shader;
   }

   ~Agx3Legality()
   {
      ralloc_free(mem_ctx);
   }

   agx_builder *builder(enum agx_shader_isa isa)
   {
      agx_builder *nb = agx_test_builder(mem_ctx);
      nb->shader->target = *agx_get_isa_target(isa);
      return nb;
   }

   agx_index temp(enum agx_size size = AGX_SIZE_32)
   {
      return agx_temp(ctx, size);
   }

   bool src_ok(agx_instr *I, unsigned s, agx_index src)
   {
      return agx_src_is_legal(ctx, I, s, src);
   }

   void *mem_ctx;
   agx_builder *b;
   agx_context *ctx;
};

TEST_F(Agx3Legality, IntegerConstantsAreEightBits)
{
   agx_instr *I = agx_iadd_to(b, temp(), temp(), temp(), 0);
   EXPECT_TRUE(src_ok(I, 1, agx_immediate(255)));
   EXPECT_TRUE(src_ok(I, 0, agx_immediate(255)));
   EXPECT_TRUE(src_ok(I, 1, agx_neg(agx_immediate(5))));
   EXPECT_FALSE(src_ok(I, 1, agx_immediate(256)));
   EXPECT_FALSE(src_ok(I, 1, agx_abs(agx_immediate(300))));

   /* Shifted adds do not exist */
   I->shift = 1;
   EXPECT_FALSE(agx_instr_is_legal(ctx, I));
}

TEST_F(Agx3Legality, MultiplyAddConstants)
{
   agx_instr *I = agx_imad_to(b, temp(), temp(), temp(), temp(), 0);
   EXPECT_TRUE(src_ok(I, 2, agx_immediate(2)));
   EXPECT_TRUE(src_ok(I, 1, agx_immediate(127)));

   /* One factor can be a constant, in operand B */
   EXPECT_TRUE(src_ok(I, 0, agx_immediate(3)));
   I->src[1] = agx_immediate(3);
   EXPECT_FALSE(src_ok(I, 0, agx_immediate(3)));

   /* No subtraction */
   I->src[1] = temp();
   EXPECT_FALSE(src_ok(I, 2, agx_neg(temp())));

   /* Only the first factor sign-extends a 16-bit register */
   EXPECT_TRUE(src_ok(I, 0, temp(AGX_SIZE_16)));
   EXPECT_FALSE(src_ok(I, 1, temp(AGX_SIZE_16)));
   EXPECT_TRUE(src_ok(I, 1, agx_abs(temp(AGX_SIZE_16))));
}

TEST_F(Agx3Legality, Minifloats)
{
   /* 17 needs the fourth mantissa bit, which the FMA multiplier lacks */
   agx_instr *I = agx_ffma_to(b, temp(), temp(), temp(), temp());
   EXPECT_TRUE(src_ok(I, 2, agx_immediate_f(17.0)));
   EXPECT_FALSE(src_ok(I, 1, agx_immediate_f(17.0)));
   EXPECT_TRUE(src_ok(I, 1, agx_immediate_f(2.0)));

   agx_instr *add = agx_fadd_to(b, temp(), temp(), temp());
   EXPECT_TRUE(src_ok(add, 1, agx_immediate_f(17.0)));
   EXPECT_TRUE(src_ok(add, 1, agx_immediate_f(-0.5)));

   /* Odd multiples of 1/64 only exist in the AGX minifloat */
   EXPECT_FALSE(src_ok(add, 1, agx_immediate_f(1.0 / 64)));
}

TEST_F(Agx3Legality, Uniforms)
{
   agx_instr *add = agx_fadd_to(b, temp(), temp(), temp());
   EXPECT_TRUE(src_ok(add, 0, agx_uniform(8, AGX_SIZE_32)));
   EXPECT_FALSE(src_ok(add, 0, agx_uniform(9, AGX_SIZE_32)));
   EXPECT_FALSE(src_ok(add, 0, agx_uniform(8, AGX_SIZE_16)));

   /* Some special functions take a uniform, others do not */
   agx_instr *floor = agx_floor_to(b, temp(), temp());
   EXPECT_TRUE(src_ok(floor, 0, agx_uniform(8, AGX_SIZE_32)));
   agx_instr *rcp = agx_rcp_to(b, temp(), temp());
   EXPECT_FALSE(src_ok(rcp, 0, agx_uniform(8, AGX_SIZE_32)));

   EXPECT_TRUE(agx_instr_accepts_uniform(ctx, add, 1, 8, AGX_SIZE_32));
   EXPECT_FALSE(agx_instr_accepts_uniform(ctx, rcp, 0, 8, AGX_SIZE_32));

   /* Memory bases are uniform pairs, indices are registers */
   agx_instr *load = agx_device_load_to(b, temp(), agx_uniform(8, AGX_SIZE_64),
                                        temp(), AGX_FORMAT_I32, 1, 0, false, 0);
   EXPECT_TRUE(agx_instr_is_legal(ctx, load));
   EXPECT_FALSE(src_ok(load, 1, agx_uniform(8, AGX_SIZE_32)));
   EXPECT_FALSE(src_ok(load, 0, agx_uniform(10, AGX_SIZE_64)));
}

TEST_F(Agx3Legality, Logic)
{
   agx_instr *I = agx_xor_to(b, temp(), temp(), temp());
   EXPECT_TRUE(agx_instr_is_legal(ctx, I));
   EXPECT_TRUE(src_ok(I, 0, agx_immediate(99)));

   I->truth_table = 0x9; /* xnor */
   EXPECT_FALSE(agx_instr_is_legal(ctx, I));
}

TEST_F(Agx3Legality, Selects)
{
   agx_index x = temp(), y = temp();

   /* min, max and selects */
   EXPECT_TRUE(agx_instr_is_legal(
      ctx, agx_icmpsel_to(b, temp(), x, y, x, y, AGX_ICOND_SLT)));
   EXPECT_TRUE(agx_instr_is_legal(
      ctx, agx_fcmpsel_to(b, temp(), x, agx_immediate_f(1.0), x,
                          agx_immediate_f(1.0), AGX_FCOND_GTN)));
   EXPECT_TRUE(agx_instr_is_legal(
      ctx, agx_icmpsel_to(b, temp(), x, y, temp(), temp(), AGX_ICOND_UEQ)));

   /* GTN only exists as max */
   EXPECT_FALSE(agx_instr_is_legal(
      ctx, agx_fcmpsel_to(b, temp(), x, y, temp(), temp(), AGX_FCOND_GTN)));

   /* a <= k swaps the operands, so the constant would be operand A */
   EXPECT_FALSE(
      agx_instr_is_legal(ctx, agx_fcmpsel_to(b, temp(), x, agx_immediate_f(1.0),
                                             temp(), temp(), AGX_FCOND_LE)));

   /* No 16-bit selects */
   EXPECT_FALSE(agx_instr_is_legal(
      ctx, agx_icmpsel_to(b, temp(AGX_SIZE_16), x, y, temp(AGX_SIZE_16),
                          temp(AGX_SIZE_16), AGX_ICOND_ULT)));
}

TEST_F(Agx3Legality, Conversions)
{
   agx_instr *I = agx_convert_to(b, temp(), agx_immediate(AGX_CONVERT_S32_TO_F),
                                 temp(), AGX_ROUND_RTE);
   EXPECT_TRUE(agx_instr_is_legal(ctx, I));
   I->round = AGX_ROUND_RTZ;
   EXPECT_FALSE(agx_instr_is_legal(ctx, I));

   I->src[0] = agx_immediate(AGX_CONVERT_U16_TO_F);
   EXPECT_FALSE(agx_instr_is_legal(ctx, I));

   /* FP32 to FP16 and back are moves */
   EXPECT_TRUE(agx_instr_is_legal(
      ctx, agx_fadd_to(b, temp(AGX_SIZE_16), temp(), agx_negzero())));
   EXPECT_TRUE(agx_instr_is_legal(
      ctx, agx_fadd_to(b, temp(), temp(AGX_SIZE_16), agx_negzero())));
}

TEST_F(Agx3Legality, Agx2IsUnrestrictedHere)
{
   agx_builder *b8 = builder(AGX_SHADER_ISA_APPLE8);
   agx_instr *I = agx_bitop_to(b8, agx_temp(b8->shader, AGX_SIZE_32),
                               agx_temp(b8->shader, AGX_SIZE_32),
                               agx_temp(b8->shader, AGX_SIZE_32), 0x9);
   EXPECT_TRUE(agx_instr_is_legal(b8->shader, I));

   agx_set_reg_constraints(b8->shader);
   agx_foreach_instr_global(b8->shader, J) {
      EXPECT_EQ(J->reg_constraints, nullptr);
   }
}

TEST_F(Agx3Legality, RegisterConstraints)
{
   agx_instr *sr = agx_get_sr_to(b, temp(), AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_instr *store = agx_local_store(b, temp(), agx_immediate(0),
                                      temp(AGX_SIZE_16), AGX_FORMAT_I32, 1);
   agx_instr *atomic =
      agx_atomic_to(b, temp(), temp(), agx_uniform(8, AGX_SIZE_64), temp(),
                    AGX_ATOMIC_OPC_ADD);
   agx_instr *add = agx_iadd_to(b, temp(), temp(), temp(), 0);

   agx_set_reg_constraints(ctx);

   /* r0-r63 */
   ASSERT_NE(sr->reg_constraints, nullptr);
   EXPECT_EQ(sr->reg_constraints[0].max, 126);

   /* The data is destroyed, the index names a low half */
   ASSERT_NE(store->reg_constraints, nullptr);
   EXPECT_TRUE(store->reg_constraints[0].clobber);
   EXPECT_EQ(store->reg_constraints[0].max, 126);
   EXPECT_EQ(store->reg_constraints[2].align, 2);
   EXPECT_EQ(store->reg_constraints[2].max, 126);

   ASSERT_NE(atomic->reg_constraints, nullptr);
   for (unsigned s = 0; s < 3; ++s)
      EXPECT_TRUE(atomic->reg_constraints[1 + s].clobber);

   EXPECT_EQ(add->reg_constraints, nullptr);
}

static void
optimize(agx_context *ctx)
{
   agx_optimizer_backward(ctx);
   agx_optimizer_forward(ctx);
   agx_dce(ctx, true);
}

/* The optimizer only creates operands AGX3 can encode */
TEST_F(Agx3Legality, OptimizerRespectsLegality)
{
   agx_index x = temp(), out = temp();

   /* An 8-bit constant is inlined, a wider one is not */
   agx_index small = agx_mov_imm(b, 32, 7);
   agx_index big = agx_mov_imm(b, 32, 0x1234);
   agx_index sum = agx_iadd(b, x, small, 0);
   agx_iadd_to(b, out, sum, big, 0);
   agx_unit_test(b, out);
   optimize(ctx);

   unsigned movs = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM)
         movs++;
      if (I->op == AGX_OPCODE_IADD && I->dest[0].value != out.value) {
         EXPECT_TRUE(agx_is_equiv(I->src[1], agx_immediate(7)));
      }
   }
   EXPECT_EQ(movs, 1u);

   /* xor(not x, y) would be xnor, which AGX3 lacks */
   agx_builder *b2 = builder(AGX_SHADER_ISA_APPLE9);
   agx_context *ctx2 = b2->shader;
   agx_index a = agx_temp(ctx2, AGX_SIZE_32), c = agx_temp(ctx2, AGX_SIZE_32);
   agx_index r = agx_temp(ctx2, AGX_SIZE_32);
   agx_get_sr_to(b2, a, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b2, c, AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_xor_to(b2, r, agx_not(b2, a), c);
   agx_unit_test(b2, r);
   optimize(ctx2);

   agx_foreach_instr_global(ctx2, I) {
      if (I->op == AGX_OPCODE_BITOP) {
         EXPECT_TRUE(agx_instr_is_legal(ctx2, I));
      }
   }

   /* 17.0 goes in the addend of an FMA, not the multiplier */
   agx_builder *b3 = builder(AGX_SHADER_ISA_APPLE9);
   agx_context *ctx3 = b3->shader;
   agx_index k = agx_mov_imm(b3, 32, fui(17.0));
   agx_index p = agx_temp(ctx3, AGX_SIZE_32);
   agx_index f = agx_temp(ctx3, AGX_SIZE_32);
   agx_ffma_to(b3, f, p, k, k);
   agx_unit_test(b3, f);
   optimize(ctx3);

   agx_foreach_instr_global(ctx3, I) {
      if (I->op == AGX_OPCODE_FFMA) {
         EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
         EXPECT_EQ(I->src[2].type, AGX_INDEX_IMMEDIATE);
      }
   }
}
