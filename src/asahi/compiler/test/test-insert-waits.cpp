/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

namespace {

agx_index
r(unsigned n)
{
   return agx_register(2 * n, AGX_SIZE_32);
}

agx_index
k(agx_index x)
{
   x.discard = true;
   return x;
}

} // namespace

class InsertWaits : public testing::Test {
 protected:
   InsertWaits()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~InsertWaits()
   {
      ralloc_free(mem_ctx);
   }

   agx_builder *builder(enum agx_shader_isa isa)
   {
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->target = *agx_get_isa_target(isa);
      return b;
   }

   /* Waits in program order, as slot numbers */
   std::vector<unsigned> waits(agx_context *ctx)
   {
      std::vector<unsigned> slots;
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_WAIT)
            slots.push_back(I->scoreboard);
      }
      return slots;
   }

   agx_instr *load(agx_builder *b, agx_index dst, agx_index index)
   {
      return agx_device_load_to(b, dst, agx_uniform(0, AGX_SIZE_64), index,
                                AGX_FORMAT_I32, 1, 0, false);
   }

   void *mem_ctx;
};

TEST_F(InsertWaits, Apple8WaitsBeforeTheUse)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE8);
   load(b, r(0), r(1));
   agx_instr *add = agx_fadd_to(b, r(2), r(0), r(1));
   agx_insert_waits(b->shader);

   EXPECT_EQ(waits(b->shader), std::vector<unsigned>{0});
   EXPECT_FALSE(add->wait);
}

/* A retaining form waits in its dependency field, as a six-byte float add */
TEST_F(InsertWaits, Apple9FoldsIntoTheConsumer)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_instr *ld = load(b, r(0), r(1));
   agx_instr *add = agx_fadd_to(b, r(2), r(0), r(1));
   agx_unit_test(b, r(0));
   agx_insert_waits(b->shader);

   EXPECT_TRUE(waits(b->shader).empty());
   EXPECT_TRUE(add->wait);
   EXPECT_EQ(add->wait_slot, ld->scoreboard);
   EXPECT_EQ(agx_apple9_encoding_for(add), AGX_APPLE9_ENC_FLOAT2_COMPACT);
}

/* Non-retaining forms release the operand, so only fold at its last use */
TEST_F(InsertWaits, Apple9FoldsIntoNonRetainingFormsAtLastUse)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(1));
   agx_instr *cvt = agx_convert_to(b, r(2), agx_immediate(AGX_CONVERT_U32_TO_F),
                                   r(0), AGX_ROUND_RTE);
   agx_unit_test(b, r(0));
   agx_insert_waits(b->shader);

   EXPECT_EQ(waits(b->shader).size(), 1u);
   EXPECT_FALSE(cvt->wait);

   agx_builder *b2 = builder(AGX_SHADER_ISA_APPLE9);
   load(b2, r(0), r(1));
   agx_instr *last = agx_convert_to(
      b2, r(2), agx_immediate(AGX_CONVERT_U32_TO_F), k(r(0)), AGX_ROUND_RTE);
   agx_insert_waits(b2->shader);

   EXPECT_TRUE(waits(b2->shader).empty());
   EXPECT_TRUE(last->wait);
}

/* Forms without a dependency field wait separately */
TEST_F(InsertWaits, Apple9WaitsSeparatelyWithoutDependencyField)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_instr *ld = load(b, r(0), r(1));
   agx_instr *mov = agx_mov_imm_to(b, r(0), 5);
   agx_insert_waits(b->shader);

   EXPECT_EQ(waits(b->shader), std::vector<unsigned>{ld->scoreboard});
   EXPECT_FALSE(mov->wait);
}

TEST_F(InsertWaits, Apple9UsesSixSlots)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_instr *loads[6];
   for (unsigned i = 0; i < 6; ++i)
      loads[i] = load(b, r(10 + i), r(1));

   agx_insert_waits(b->shader);

   for (unsigned i = 0; i < 6; ++i)
      EXPECT_EQ(loads[i]->scoreboard, i);
   EXPECT_TRUE(waits(b->shader).empty());
}

/* An instruction names one slot, so a second pending operand waits apart */
TEST_F(InsertWaits, Apple9FoldsOneOfTwoWaits)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(5));
   load(b, r(1), r(5));
   agx_instr *add = agx_fadd_to(b, r(2), r(0), r(1));
   agx_insert_waits(b->shader);

   EXPECT_TRUE(add->wait);
   EXPECT_EQ(waits(b->shader).size(), 1u);
   EXPECT_NE(waits(b->shader)[0], add->wait_slot);
}

TEST_F(InsertWaits, Apple9FixedSlots)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_instr *shared =
      agx_local_load_to(b, r(0), agx_immediate(0), agx_register(2, AGX_SIZE_16),
                        AGX_FORMAT_I32, 1);
   agx_instr *atomic =
      agx_local_atomic_to(b, r(3), r(4), agx_immediate(0),
                          agx_register(10, AGX_SIZE_16), AGX_ATOMIC_OPC_ADD);
   agx_instr *coverage = agx_get_sr_coverage_to(b, r(6), AGX_SR_COVERAGE_MASK);
   agx_insert_waits(b->shader);

   EXPECT_EQ(shared->scoreboard, 5);
   EXPECT_EQ(atomic->scoreboard, 0);
   EXPECT_EQ(coverage->scoreboard, 0);
}

/* Barriers wait for pending results except the classes they exempt */
TEST_F(InsertWaits, Apple9PriorCompletion)
{
   /* The workgroup barrier leaves memory results pending */
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(1));
   agx_threadgroup_barrier(b);
   agx_insert_waits(b->shader);
   EXPECT_TRUE(waits(b->shader).empty());

   /* Spill fills are not left pending */
   b = builder(AGX_SHADER_ISA_APPLE9);
   agx_stack_load_to(b, r(0), agx_immediate(16), AGX_FORMAT_I32, 1);
   agx_threadgroup_barrier(b);
   agx_insert_waits(b->shader);
   EXPECT_EQ(waits(b->shader).size(), 1u);

   /* The acquire fence waits for loads, the plain one does not */
   b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(1));
   agx_memory_barrier(b);
   agx_insert_waits(b->shader);
   EXPECT_TRUE(waits(b->shader).empty());

   b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(1));
   agx_memory_barrier_acquire(b);
   agx_insert_waits(b->shader);
   EXPECT_EQ(waits(b->shader).size(), 1u);

   /* Halt waits for everything */
   b = builder(AGX_SHADER_ISA_APPLE9);
   load(b, r(0), r(1));
   agx_stop(b);
   agx_insert_waits(b->shader);
   EXPECT_EQ(waits(b->shader).size(), 1u);
}

/* A spill save reads its source after issue */
TEST_F(InsertWaits, Apple9WaitsBeforeOverwritingASpilledRegister)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_instr *save =
      agx_stack_store(b, r(4), agx_immediate(16), AGX_FORMAT_I32, 1);
   agx_mov_imm_to(b, r(4), 1);
   agx_insert_waits(b->shader);

   EXPECT_EQ(waits(b->shader), std::vector<unsigned>{save->scoreboard});
}

/* Export registers are a separate file with their own hazards */
TEST_F(InsertWaits, Apple9Exports)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_index e0 = agx_export_register(0, AGX_SIZE_32);
   agx_instr *mul = agx_fmul_to(b, e0, r(1), r(2));
   agx_instr *add = agx_fadd_to(b, r(0), r(1), r(2));
   agx_st_vary(b, agx_immediate(0), e0);
   agx_insert_waits(b->shader);

   /* r0 is not e0 */
   EXPECT_FALSE(add->wait);
   EXPECT_EQ(waits(b->shader), std::vector<unsigned>{mul->scoreboard});
}

TEST_F(InsertWaits, Apple9WaitsAtTheEndOfABlock)
{
   agx_builder *b = builder(AGX_SHADER_ISA_APPLE9);
   agx_block *first = agx_start_block(b->shader);
   agx_test_block(b->shader);

   b->cursor = agx_after_block(first);
   load(b, r(0), r(1));
   agx_insert_waits(b->shader);

   EXPECT_EQ(waits(b->shader).size(), 1u);
}
