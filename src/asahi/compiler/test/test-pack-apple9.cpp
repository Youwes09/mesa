/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Apple9 packing of allocated agx_instr. Tests named Unverified* expect
 * bytes that are unverified on hardware.
 */

#include <vector>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

namespace {

/* 32-bit GPR rN, 16-bit halves, export words and uniform words */
agx_index
r(unsigned n)
{
   return agx_register(2 * n, AGX_SIZE_32);
}

agx_index
rl(unsigned n)
{
   return agx_register(2 * n, AGX_SIZE_16);
}

agx_index
rh(unsigned n)
{
   return agx_register(2 * n + 1, AGX_SIZE_16);
}

agx_index
r64(unsigned n)
{
   return agx_register(2 * n, AGX_SIZE_64);
}

agx_index
e(unsigned n)
{
   return agx_export_register(2 * n, AGX_SIZE_32);
}

agx_index
u(unsigned n)
{
   return agx_uniform(2 * n, AGX_SIZE_32);
}

agx_index
u64(unsigned n)
{
   return agx_uniform(4 * n, AGX_SIZE_64);
}

/* Last use */
agx_index
k(agx_index x)
{
   x.discard = true;
   return x;
}

agx_index
f(float x)
{
   return agx_immediate_f(x);
}

} // namespace

class Apple9Pack : public testing::Test {
 protected:
   Apple9Pack()
   {
      mem_ctx = ralloc_context(NULL);
      b = agx_test_builder(mem_ctx);
      b->shader->target = *agx_get_isa_target(AGX_SHADER_ISA_APPLE9);
   }

   ~Apple9Pack()
   {
      ralloc_free(mem_ctx);
   }

   std::vector<uint8_t> pack(agx_instr *I, bool load_follows = false,
                             bool alu_reads = true)
   {
      uint8_t bytes[32];
      unsigned length =
         agx_apple9_pack_instr(I, load_follows, alu_reads, bytes);
      return std::vector<uint8_t>(bytes, bytes + length);
   }

   /* Compare, ignoring the bits set in mask */
   void check(agx_instr *I, std::vector<uint8_t> expected,
              std::vector<uint8_t> mask = {}, bool load_follows = false,
              bool alu_reads = true)
   {
      std::vector<uint8_t> packed = pack(I, load_follows, alu_reads);
      ASSERT_EQ(packed.size(), expected.size());

      for (unsigned i = 0; i < expected.size(); ++i) {
         uint8_t m = i < mask.size() ? mask[i] : 0;
         EXPECT_EQ(packed[i] & ~m, expected[i] & ~m) << "byte " << i;
      }
   }

   void wait(agx_instr *I, unsigned hw_slot)
   {
      if (hw_slot) {
         I->wait = true;
         I->wait_slot = hw_slot - 1;
      } else {
         I->wait = false;
      }
   }

   enum agx_apple9_encoding form(agx_instr *I)
   {
      return agx_apple9_encoding_for(I);
   }

   void *mem_ctx;
   agx_builder *b;
};

TEST_F(Apple9Pack, PerspectiveMultiplyNamesCoefficient)
{
   agx_instr *I = agx_fmul_perspective_to(b, r(15), k(r(10)), r(6), 2);
   check(I, {0xf9, 0x15, 0x2f, 0x8d, 0x00, 0x02, 0x00, 0x00});

   for (unsigned cf = 0; cf <= 32; ++cf) {
      I->component = cf;
      EXPECT_EQ(pack(I)[5], cf);
   }
}

TEST_F(Apple9Pack, ReciprocalHintsItsReaderAndReleasesItsSource)
{
   agx_instr *I = agx_rcp_to(b, r(18), k(r(7)));

   /* Read only by a device store, released, waiting on slot 6 */
   wait(I, 6);
   check(I, {0xaf, 0x00, 0x56, 0x24, 0x02, 0x1c, 0x10, 0x48, 0x20, 0x00}, {},
         false, false);

   /* Read by an ALU instruction, retained */
   I->src[0] = r(7);
   wait(I, 0);
   check(I, {0xaf, 0x00, 0x54, 0x24, 0x03, 0x1c, 0x00, 0x48, 0x20, 0x00});

   for (unsigned slot : {4, 5}) {
      wait(I, slot);
      std::vector<uint8_t> p = pack(I);
      EXPECT_EQ((p[1] >> 4) | ((p[2] & 3) << 4), 1u << (slot - 1));
   }
}

TEST_F(Apple9Pack, ScalarConversionsAndArithmeticShift)
{
   agx_instr *I = agx_convert_to(b, r(16), agx_immediate(AGX_CONVERT_S32_TO_F),
                                 k(r(17)), AGX_ROUND_RTE);
   check(I, {0xa7, 0x07, 0x54, 0x20, 0x03, 0x44, 0xac, 0x60});

   I->src[0] = agx_immediate(AGX_CONVERT_U32_TO_F);
   check(I, {0xa7, 0x07, 0x54, 0x20, 0x03, 0x44, 0xac, 0x20});

   I->src[1] = r(17);
   check(I, {0xa7, 0x07, 0x54, 0x20, 0x03, 0x44, 0x8c, 0x20});

   I->src[0] = agx_immediate(AGX_CONVERT_S32_TO_F);
   check(I, {0xa7, 0x07, 0x54, 0x20, 0x03, 0x44, 0x8c, 0x60});

   I->src[1] = k(r(17));
   I->round = AGX_ROUND_RTZ;
   I->src[0] = agx_immediate(AGX_CONVERT_F_TO_S32);
   check(I, {0x27, 0x07, 0x54, 0x20, 0x03, 0x44, 0xb4, 0x48, 0x03, 0x00});

   I->src[0] = agx_immediate(AGX_CONVERT_F_TO_U32);
   check(I, {0x27, 0x07, 0x54, 0x20, 0x03, 0x44, 0xb4, 0x08, 0x02, 0x00});

   /* Float to integer keeps a live source */
   I->src[1] = r(17);
   EXPECT_EQ(pack(I)[6], 0x96);

   agx_instr *shift = agx_asr_to(b, r(12), k(r(13)), agx_immediate(7));
   check(shift, {0xa7, 0x01, 0x54, 0x18, 0x02, 0x34, 0x1c, 0x78, 0x62, 0x00});
}

/* 8-bit integer constants. Every register source is a last use. */
TEST_F(Apple9Pack, IntegerImmediates)
{
   std::vector<uint8_t> dep = {0x00, 0xf0, 0x03};
   std::vector<uint8_t> logic_dep = {0, 0, 0, 0, 0, 0xe0, 0, 0xe0};
   std::vector<uint8_t> select_dep = {0, 0, 0, 0, 0, 0, 0, 0xe0};

   /* x + 7 with the constant in operand A */
   check(agx_iadd_to(b, r(0), agx_immediate(7), k(r(0)), 0),
         {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0e, 0x00, 0x88, 0x15, 0x04}, dep);

   check(agx_iadd_to(b, r(0), k(r(0)), agx_neg(agx_immediate(5)), 0),
         {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x14, 0x28, 0x13, 0x01}, dep);

   check(agx_iadd_to(b, r(0), k(r(0)), agx_neg(agx_immediate(255)), 0),
         {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0xfc, 0x2b, 0x13, 0x01}, dep);

   /* 200 - x */
   check(agx_iadd_to(b, r(0), agx_immediate(200), agx_neg(k(r(0))), 0),
         {0x1f, 0x01, 0x56, 0x00, 0x02, 0x90, 0x01, 0x88, 0x15, 0x04}, dep);

   /* x * y + v */
   check(
      agx_imad_to(b, r(0), k(r(2)), k(r(0)), agx_immediate(2), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08, 0x00, 0x10, 0xd0, 0x26, 0x0a, 0x00},
      dep);

   check(
      agx_imad_to(b, r(0), k(r(2)), k(r(0)), agx_immediate(255), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08, 0x00, 0xf8, 0xd7, 0x26, 0x0a, 0x00},
      dep);

   /* x * 127 + y */
   check(
      agx_imad_to(b, r(0), k(r(0)), agx_immediate(127), k(r(2)), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00, 0xfc, 0x21, 0x50, 0x2b, 0x22, 0x00},
      dep);

   check(agx_xor_to(b, r(0), k(r(0)), agx_immediate(99)),
         {0x0b, 0x01, 0x0e, 0xe3, 0x12, 0x0a, 0x00, 0x80, 0x00, 0x00},
         logic_dep);
   check(agx_or_to(b, r(0), k(r(0)), agx_immediate(9)),
         {0x0b, 0x01, 0x0f, 0x89, 0x12, 0x0a, 0x00, 0x80, 0x00, 0x00},
         logic_dep);
   check(agx_and_to(b, r(0), k(r(0)), agx_immediate(200)),
         {0x0b, 0x01, 0x1f, 0xc9, 0x00, 0x02, 0x00, 0x80, 0x00, 0x00},
         logic_dep);
   check(agx_xor_to(b, r(0), k(r(0)), agx_immediate(254)),
         {0x0b, 0x01, 0x1e, 0xff, 0x02, 0x0a, 0x00, 0x80, 0x00, 0x00},
         logic_dep);

   /* x > z ? 200 : y */
   check(agx_icmpsel_to(b, r(0), k(r(0)), k(r(4)), agx_immediate(200), k(r(2)),
                        AGX_ICOND_UGT),
         {0x02, 0x01, 0x1f, 0x09, 0xa2, 0xc8, 0x04, 0xc0, 0x80, 0x04},
         select_dep);

   /* x > z ? 5 : 7 */
   check(agx_icmpsel_to(b, r(0), k(r(0)), k(r(2)), agx_immediate(5),
                        agx_immediate(7), AGX_ICOND_UGT),
         {0x02, 0x01, 0x1f, 0x05, 0x22, 0x85, 0x04, 0xc0, 0x20, 0x87},
         select_dep);
}

/* Minifloat constants, with dependency fields and bit 21 masked */
TEST_F(Apple9Pack, FloatImmediates)
{
   std::vector<uint8_t> m6 = {0, 0, 0x20, 0, 0, 0xe0};
   std::vector<uint8_t> m8 = {0, 0, 0x20, 0, 0, 0, 0, 0xe0};

   auto fmax = [&](float x) {
      return agx_fcmpsel_to(b, r(1), k(r(1)), f(x), k(r(1)), f(x),
                            AGX_FCOND_GTN);
   };

   check(fmax(1.0), {0x12, 0x03, 0x0a, 0xb1, 0x00, 0xc2}, m6);
   check(fmax(0.0), {0x12, 0x03, 0x0a, 0x81, 0x00, 0xc2}, m6);
   check(fmax(-1.0), {0x12, 0x03, 0x1a, 0xb1, 0x00, 0xc2}, m6);
   check(fmax(31.0), {0x12, 0x03, 0x0a, 0xff, 0x10, 0xc2}, m6);
   check(fmax(2.125), {0x12, 0x03, 0x0a, 0xc1, 0x10, 0xc2}, m6);
   check(agx_fcmpsel_to(b, r(1), k(r(1)), f(-0.5), k(r(1)), f(-0.5),
                        AGX_FCOND_LTN),
         {0x12, 0x03, 0x1a, 0xa1, 0x01, 0xc2}, m6);

   check(agx_fmul_to(b, r(0), k(r(0)), f(17.0)),
         {0x09, 0xf1, 0x15, 0x01, 0x88, 0xc0}, m6);
   check(agx_fmul_to(b, r(0), k(r(0)), f(2.125)),
         {0x09, 0xc1, 0x15, 0x01, 0x88, 0xc0}, m6);
   check(agx_fadd_to(b, r(0), k(r(0)), f(31.0)),
         {0x09, 0xff, 0x14, 0x01, 0x88, 0xc0}, m6);

   check(agx_ffma_to(b, r(0), k(r(0)), k(r(2)), f(31.0)),
         {0x09, 0x01, 0x1e, 0x05, 0x21, 0xff, 0x02, 0xc0}, m8);
   check(agx_ffma_to(b, r(0), k(r(0)), k(r(2)), f(2.125)),
         {0x09, 0x01, 0x1e, 0x05, 0x21, 0xc1, 0x02, 0xc0}, m8);
   check(agx_ffma_to(b, r(0), k(r(0)), k(r(2)), f(1.0)),
         {0x09, 0x01, 0x1e, 0x05, 0x21, 0xb0, 0x02, 0xc0}, m8);
}

TEST_F(Apple9Pack, ImmediateFloatCarriesWait)
{
   agx_instr *I = agx_fadd_to(b, r(2), k(r(2)), f(2.0));

   for (unsigned slot = 0; slot <= 6; ++slot) {
      wait(I, slot);
      check(I, {0x29, 0xc1, 0x34, 0x05, 0x80, (uint8_t)(slot << 5)});
   }
}

/* Compare operand B constants of selects and predicate compares */
TEST_F(Apple9Pack, CompareLiterals)
{
   std::vector<uint8_t> select_dep = {0, 0, 0, 0, 0, 0, 0, 0xe0, 0, 0};
   std::vector<uint8_t> pred_dep = {0, 0, 0, 0, 0, 0xe0};
   std::vector<uint8_t> pred_ext_dep = {0, 0, 0, 0, 0, 0, 0, 0xe0, 0, 0};

   auto isel = [&](enum agx_icond cond, unsigned literal) {
      return agx_icmpsel_to(b, r(0), k(r(0)), agx_immediate(literal), k(r(2)),
                            k(r(4)), cond);
   };
   auto fsel = [&](enum agx_fcond cond, float literal) {
      return agx_fcmpsel_to(b, r(0), k(r(0)), f(literal), k(r(2)), k(r(4)),
                            cond);
   };

   check(isel(AGX_ICOND_UGT, 5),
         {0x02, 0x01, 0x0f, 0x84, 0x82, 0x04, 0x14, 0xc2, 0x80, 0x08},
         select_dep);
   check(isel(AGX_ICOND_ULT, 200),
         {0x02, 0x01, 0x1f, 0xc8, 0x82, 0x04, 0x05, 0xc2, 0x80, 0x08},
         select_dep);
   check(isel(AGX_ICOND_UGT, 129),
         {0x02, 0x01, 0x1f, 0x80, 0x82, 0x04, 0x14, 0xc2, 0x80, 0x08},
         select_dep);
   check(isel(AGX_ICOND_ULT, 255),
         {0x02, 0x01, 0x1f, 0xfe, 0x82, 0x04, 0x15, 0xc2, 0x80, 0x08},
         select_dep);
   check(isel(AGX_ICOND_SLT, 7),
         {0x02, 0x01, 0x0f, 0x86, 0x82, 0x04, 0x17, 0xc2, 0x80, 0x08},
         select_dep);
   check(fsel(AGX_FCOND_LT, 1.0),
         {0x02, 0x01, 0x0f, 0xb1, 0x82, 0x04, 0x03, 0xc2, 0x80, 0x08},
         select_dep);
   check(fsel(AGX_FCOND_GT, -1.0),
         {0x02, 0x01, 0x1f, 0xb1, 0x82, 0x04, 0x02, 0xc2, 0x80, 0x08},
         select_dep);
   check(fsel(AGX_FCOND_LT, 2.125),
         {0x02, 0x01, 0x0f, 0xc1, 0x82, 0x04, 0x13, 0xc2, 0x80, 0x08},
         select_dep);

   auto ipred = [&](enum agx_icond cond, unsigned literal) {
      return agx_icmp_pred(b, k(r(2)), agx_immediate(literal), cond, false, 0);
   };

   check(ipred(AGX_ICOND_UGT, 5), {0x0a, 0x05, 0x2a, 0x84, 0x14, 0xc2},
         pred_dep);
   check(ipred(AGX_ICOND_UGT, 129), {0x0a, 0x05, 0x3a, 0x80, 0x14, 0xc2},
         pred_dep);
   check(ipred(AGX_ICOND_SLT, 7), {0x0a, 0x05, 0x2a, 0x86, 0x17, 0xc2},
         pred_dep);
   check(agx_fcmp_pred(b, k(r(2)), f(0.0), AGX_FCOND_GT, false, 0),
         {0x0a, 0x05, 0x2a, 0x81, 0x02, 0xc2}, pred_dep);
   check(agx_fcmp_pred(b, k(r(2)), f(-1.0), AGX_FCOND_GT, false, 0),
         {0x0a, 0x05, 0x3a, 0xb1, 0x02, 0xc2}, pred_dep);
   check(ipred(AGX_ICOND_UEQ, 7),
         {0x0a, 0x05, 0x2b, 0x86, 0x06, 0x00, 0x17, 0xc2, 0x00, 0x00},
         pred_ext_dep);
   check(ipred(AGX_ICOND_UEQ, 0),
         {0x0a, 0x05, 0x2b, 0x80, 0x06, 0x00, 0x07, 0xc2, 0x00, 0x00},
         pred_ext_dep);
}

TEST_F(Apple9Pack, UniformMoves)
{
   static const struct {
      uint8_t dst;
      uint16_t uniform;
      uint8_t bytes[4];
   } cases[] = {
      {1, 4, {0x1b, 0x08, 0x01, 0x08}},   {36, 4, {0x4b, 0x08, 0x81, 0x08}},
      {19, 66, {0x3b, 0x04, 0x49, 0x08}}, {3, 130, {0x3b, 0x04, 0x01, 0x18}},
      {28, 20, {0xcb, 0x28, 0x41, 0x08}},
   };

   for (const auto &test : cases) {
      agx_instr *I =
         agx_bitop_to(b, r(test.dst), u(test.uniform), agx_zero(), 0xa);
      EXPECT_EQ(form(I), AGX_APPLE9_ENC_UNIFORM_MOVE);
      check(I, std::vector<uint8_t>(test.bytes, test.bytes + 4));
   }

   /* r64 and up take the long form */
   agx_instr *I = agx_bitop_to(b, r(64), u(4), agx_zero(), 0xa);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_LOGIC_EXTENDED);
}

/* 16-bit sources of integer add and multiply-add sign-extend without abs */
TEST_F(Apple9Pack, IntegerHalfSources)
{
   std::vector<uint8_t> dep = {0x00, 0xf0, 0x03};

   check(agx_iadd_to(b, r(0), agx_abs(k(rl(2))), k(r(0)), 0),
         {0x9f, 0x01, 0x56, 0x00, 0x02, 0x08, 0x00, 0x88, 0x17, 0x05}, dep);
   check(agx_iadd_to(b, r(0), k(rl(2)), k(r(0)), 0),
         {0x9f, 0x01, 0x56, 0x00, 0x02, 0x08, 0x00, 0x88, 0x17, 0x07}, dep);
   check(agx_iadd_to(b, r(0), agx_abs(k(rh(2))), k(r(0)), 0),
         {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0a, 0x00, 0x88, 0x17, 0x05}, dep);
   check(agx_iadd_to(b, r(0), k(rh(2)), k(r(0)), 0),
         {0x9f, 0x01, 0x56, 0x00, 0x02, 0x0a, 0x00, 0x88, 0x17, 0x07}, dep);

   check(agx_iadd_to(b, r(0), k(r(0)), agx_neg(k(rl(2))), 0),
         {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0x28, 0x17, 0x0d}, dep);
   check(agx_iadd_to(b, r(0), k(r(0)), agx_neg(k(rh(2))), 0),
         {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x14, 0x28, 0x17, 0x0d}, dep);

   check(
      agx_imad_to(b, r(0), agx_abs(k(rh(4))), k(r(0)), k(r(2)), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x12, 0x00, 0x20, 0x90, 0x2f, 0x2a, 0x00},
      dep);
   check(
      agx_imad_to(b, r(0), k(rh(4)), k(r(0)), k(r(2)), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x12, 0x00, 0x20, 0x90, 0x2f, 0x2e, 0x00},
      dep);
   check(
      agx_imad_to(b, r(0), k(r(2)), k(r(0)), agx_abs(k(rh(4))), 0),
      {0x9f, 0x00, 0x56, 0x00, 0x02, 0x08, 0x00, 0x48, 0xd0, 0x2e, 0x2a, 0x00},
      dep);

   /* Only operand A has a sign-extension bit */
   agx_instr *I = agx_imad_to(b, r(0), k(rh(2)), agx_abs(k(rh(4))), k(r(6)), 0);
   EXPECT_EQ((pack(I)[10] >> 2) & 1, 1);
}

TEST_F(Apple9Pack, PackFormat)
{
   check(agx_pack_format_to(b, r(0), k(r(0)), k(r(1)), k(r(2)), agx_null(),
                            AGX_FORMAT_RG11B10F),
         {0xa7, 0x06, 0x54, 0x00, 0x02, 0x00, 0x08, 0x20, 0x10, 0xa0, 0xca,
          0x89, 0x24, 0x08});
   check(agx_pack_format_to(b, r(0), r(1), k(r(0)), k(r(2)), agx_null(),
                            AGX_FORMAT_RG11B10F),
         {0xa7, 0x06, 0x54, 0x00, 0x02, 0x04, 0x02, 0x20, 0x10, 0xa0, 0x8a,
          0x89, 0x24, 0x08});
   check(agx_pack_format_to(b, r(0), k(r(0)), k(r(1)), k(r(2)), k(r(3)),
                            AGX_FORMAT_RGB10A2),
         {0xa7, 0x06, 0x54, 0x00, 0x02, 0x00, 0x08, 0x20, 0x60, 0xa0, 0xea,
          0x8b, 0x24, 0x11});
}

/* FP16 sources converted on read. Retained here, as in Metal's kernels. */
TEST_F(Apple9Pack, Fp16Sources)
{
   std::vector<uint8_t> m6 = {0, 0, 0x20, 0, 0, 0xe0};
   std::vector<uint8_t> m8 = {0, 0, 0x20, 0, 0, 0, 0, 0xe0};

   /* Metal waits here, which takes the six-byte form */
   agx_instr *I = agx_fadd_to(b, r(0), k(r(0)), k(rl(2)));
   wait(I, 6);
   check(I, {0x09, 0x01, 0x1c, 0x04, 0x00, 0xc0}, m6);

   check(agx_fadd_to(b, r(0), k(rh(2)), k(r(0))),
         {0x09, 0x04, 0x1c, 0x01, 0x08, 0x00}, m6);
   check(agx_fmul_to(b, r(0), k(rh(2)), k(r(0))),
         {0x09, 0x04, 0x1d, 0x01, 0x08, 0x00}, m6);

   I = agx_fmul_to(b, r(0), k(rl(0)), k(rh(0)));
   wait(I, 6);
   check(I, {0x09, 0x00, 0x1d, 0x00, 0x10, 0xc0}, m6);

   /* Four-byte form with a low half */
   I = agx_fadd_to(b, r(0), k(rl(2)), k(r(0)));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_FLOAT2_BASE);
   check(I, {0x09, 0x04, 0x18, 0x01}, {0, 0, 0x20, 0});

   check(agx_ffma_to(b, r(0), k(r(0)), k(r(2)), k(rh(4))),
         {0x09, 0x01, 0x1e, 0x05, 0x81, 0x09, 0x00, 0x40}, m8);
   check(agx_ffma_to(b, r(0), k(r(0)), k(rl(4)), k(r(2))),
         {0x09, 0x01, 0x1e, 0x08, 0x81, 0x04, 0x02, 0xc0}, m8);
}

TEST_F(Apple9Pack, RegisterForms)
{
   /* All sources dead */
   uint8_t add[] = {0x9f, 0x01, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x17, 0x05};
   uint64_t word;
   memcpy(&word, add, sizeof(word));
   word &= ~((((1ull << 7) - 1) << 25) | (((1ull << 7) - 1) << 42) |
             (((1ull << 7) - 1) << 51));
   word |= (64ull << 25) | (2ull << 42) | (95ull << 51);
   memcpy(add, &word, sizeof(word));
   check(agx_iadd_to(b, r(64), k(r(2)), k(r(95)), 0),
         std::vector<uint8_t>(add, add + sizeof(add)));

   agx_instr *I = agx_icmpsel_to(b, r(15), k(r(95)), k(r(64)), k(r(63)),
                                 k(r(94)), AGX_ICOND_ULT);
   wait(I, 6);
   check(I, {0xf2, 0x3f, 0x1f, 0x01, 0x82, 0x7e, 0x05, 0xc5, 0xc0, 0x3c});
}

TEST_F(Apple9Pack, MoveImmediate)
{
   agx_instr *I = agx_mov_imm_to(b, r(15), 0x7f);
   check(I, {0xfc, 0x7f});

   /* Bit 7 of the short form is not a value bit */
   I->imm = 0x80;
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_MOV_IMM32);

   static const struct {
      unsigned dst;
      uint32_t value;
      uint8_t bytes[8];
   } cases[] = {
      {0, 0x12345678, {0x0c, 0xf8, 0x02, 0x12, 0x18, 0x08, 0xa2, 0x01}},
      {18, 0x3f800000, {0x2c, 0x80, 0x42, 0x3e, 0x00, 0x00, 0x00, 0x0c}},
      {34, 0x01020304, {0x2c, 0x84, 0x82, 0x00, 0x0c, 0x00, 0x10, 0x08}},
      {63, 0xffffffff, {0xfc, 0xff, 0xc2, 0xfe, 0x1e, 0x0c, 0xff, 0x0f}},
   };

   for (const auto &test : cases) {
      check(agx_mov_imm_to(b, r(test.dst), test.value),
            std::vector<uint8_t>(test.bytes, test.bytes + 8));
   }
}

TEST_F(Apple9Pack, LogicSourceLifetime)
{
   static const struct {
      bool keep_a, keep_b;
      uint8_t bytes[10];
   } cases[] = {
      {0, 0, {0x4b, 0x05, 0x1e, 0x07, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {1, 0, {0x4b, 0x85, 0x16, 0x07, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {0, 1, {0x4b, 0x05, 0x2e, 0x87, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {1, 1, {0x4b, 0x85, 0x26, 0x87, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
   };

   for (const auto &test : cases) {
      agx_index a = test.keep_a ? r(2) : k(r(2));
      agx_index c = test.keep_b ? r(3) : k(r(3));
      check(agx_xor_to(b, r(4), a, c),
            std::vector<uint8_t>(test.bytes, test.bytes + 10));
   }
}

TEST_F(Apple9Pack, LogicWaitIsOneHot)
{
   for (unsigned table : {0x8, 0xe, 0x6}) {
      agx_instr *I = agx_bitop_to(b, r(4), k(r(2)), k(r(3)), table);

      for (unsigned slot = 0; slot <= 6; ++slot) {
         wait(I, slot);
         std::vector<uint8_t> p = pack(I);
         ASSERT_EQ(p.size(), 10u);
         unsigned mask = ((p[5] >> 5) & 7) | (((p[7] >> 5) & 7) << 3);
         EXPECT_EQ(mask, slot ? 1u << (slot - 1) : 0u);
      }
   }
}

TEST_F(Apple9Pack, IntegerFamilyWaitIsOneHotAt12)
{
   agx_instr *instrs[] = {
      agx_iadd_to(b, r(4), k(r(2)), k(r(3)), 0),
      agx_imad_to(b, r(4), k(r(2)), k(r(3)), k(r(5)), 0),
      agx_convert_to(b, r(4), agx_immediate(AGX_CONVERT_U32_TO_F), k(r(2)),
                     AGX_ROUND_RTE),
      agx_convert_to(b, r(4), agx_immediate(AGX_CONVERT_F_TO_U32), k(r(2)),
                     AGX_ROUND_RTZ),
      agx_rcp_to(b, r(4), k(r(2))),
      agx_asr_to(b, r(4), k(r(2)), agx_immediate(3)),
      agx_device_store(b, r(4), u64(0), k(r(2)), AGX_FORMAT_I32, 1, 0, false,
                       0),
   };

   for (agx_instr *I : instrs) {
      for (unsigned slot = 0; slot <= 6; ++slot) {
         wait(I, slot);
         std::vector<uint8_t> p = pack(I);
         EXPECT_EQ((p[1] >> 4) | ((p[2] & 3) << 4),
                   slot ? 1u << (slot - 1) : 0u);
      }
   }
}

TEST_F(Apple9Pack, IntegerMinMax)
{
   static const struct {
      bool keep_a, keep_b;
      uint8_t bytes[6];
   } cases[] = {
      {0, 0, {0x42, 0x05, 0x3e, 0x07, 0x05, 0x00}},
      {1, 0, {0x42, 0x85, 0x36, 0x07, 0x05, 0x00}},
      {0, 1, {0x42, 0x05, 0x2e, 0x87, 0x05, 0x00}},
      {1, 1, {0x42, 0x85, 0x26, 0x87, 0x05, 0x00}},
   };

   for (const auto &test : cases) {
      agx_index a = test.keep_a ? r(2) : k(r(2));
      agx_index c = test.keep_b ? r(3) : k(r(3));
      agx_instr *I = agx_icmpsel_to(b, r(4), a, c, a, c, AGX_ICOND_ULT);
      EXPECT_EQ(form(I), AGX_APPLE9_ENC_MINMAX_COMPACT);
      check(I, std::vector<uint8_t>(test.bytes, test.bytes + 6));
   }

   /* Swapped selection is the other one */
   agx_instr *I = agx_icmpsel_to(b, r(4), k(r(2)), k(r(3)), k(r(3)), k(r(2)),
                                 AGX_ICOND_UGT);
   EXPECT_EQ(pack(I)[4], 5);
}

/* Waits need the six-byte add; without one it shortens to four bytes */
TEST_F(Apple9Pack, CompactFloatStateAndRelease)
{
   static const struct {
      bool keep_a, keep_b;
      uint8_t add[6];
      uint8_t minimum[6];
   } cases[] = {
      {0,
       0,
       {0x49, 0x05, 0x3c, 0x07, 0x00, 0x00},
       {0x42, 0x05, 0x3e, 0x07, 0x01, 0x00}},
      {1,
       0,
       {0x49, 0x85, 0x34, 0x07, 0x00, 0x00},
       {0x42, 0x85, 0x36, 0x07, 0x01, 0x00}},
      {0,
       1,
       {0x49, 0x05, 0x2c, 0x87, 0x00, 0x00},
       {0x42, 0x05, 0x2e, 0x87, 0x01, 0x00}},
      {1,
       1,
       {0x49, 0x85, 0x24, 0x87, 0x00, 0x00},
       {0x42, 0x85, 0x26, 0x87, 0x01, 0x00}},
   };

   for (const auto &test : cases) {
      agx_index a = test.keep_a ? r(2) : k(r(2));
      agx_index c = test.keep_b ? r(3) : k(r(3));

      agx_instr *add = agx_fadd_to(b, r(4), a, c);
      std::vector<uint8_t> base(test.add, test.add + 4);
      base[2] &= ~0x04;
      check(add, base);

      wait(add, 1);
      std::vector<uint8_t> compact(test.add, test.add + 6);
      compact[5] |= 1 << 5;
      check(add, compact);

      check(agx_fcmpsel_to(b, r(4), a, c, a, c, AGX_FCOND_LTN),
            std::vector<uint8_t>(test.minimum, test.minimum + 6));
   }

   agx_instr *add = agx_fadd_to(b, r(4), r(2), r(3));
   wait(add, 6);
   check(add, {0x49, 0x85, 0x24, 0x87, 0x00, 0xc0});

   agx_instr *minimum =
      agx_fcmpsel_to(b, r(4), r(2), r(3), r(2), r(3), AGX_FCOND_LTN);
   wait(minimum, 6);
   check(minimum, {0x42, 0x85, 0x26, 0x87, 0x01, 0xc0});
}

TEST_F(Apple9Pack, CompactBinaryScatteredRegisterBits)
{
   check(agx_fadd_to(b, r(95), k(r(64)), k(r(79))),
         {0xf9, 0x01, 0x7c, 0x1f, 0x00, 0x15});
   check(agx_icmpsel_to(b, r(95), k(r(64)), k(r(79)), k(r(64)), k(r(79)),
                        AGX_ICOND_ULT),
         {0xf2, 0x01, 0x7e, 0x1f, 0x05, 0x15});
}

/* Without a wait, registers below r64 take the six-byte FMA */
TEST_F(Apple9Pack, FmaStateAndRelease)
{
   static const struct {
      bool keep_a, keep_b, keep_c;
      uint8_t bytes[8];
   } cases[] = {
      {0, 0, 0, {0x49, 0x05, 0x3e, 0x07, 0x81, 0x0a, 0x02, 0x00}},
      {1, 0, 0, {0x49, 0x85, 0x36, 0x07, 0x81, 0x0a, 0x02, 0x00}},
      {0, 1, 0, {0x49, 0x05, 0x2e, 0x87, 0x81, 0x0a, 0x02, 0x00}},
      {0, 0, 1, {0x49, 0x05, 0x3e, 0x07, 0x01, 0x8a, 0x02, 0x00}},
      {1, 1, 1, {0x49, 0x85, 0x26, 0x87, 0x01, 0x8a, 0x02, 0x00}},
   };

   for (const auto &test : cases) {
      agx_instr *I = agx_ffma_to(b, r(4), test.keep_a ? r(2) : k(r(2)),
                                 test.keep_b ? r(3) : k(r(3)),
                                 test.keep_c ? r(5) : k(r(5)));
      EXPECT_EQ(form(I), AGX_APPLE9_ENC_FLOAT3_COMPACT);

      wait(I, 6);
      std::vector<uint8_t> expected(test.bytes, test.bytes + 8);
      expected[7] |= 0xc0;
      check(I, expected);
   }
}

TEST_F(Apple9Pack, FmaUniformBanks)
{
   /* Operand B r255, addend u128, operand A r64 */
   agx_instr *I = agx_ffma_to(b, r(95), r(64), u(255), r(65));
   std::vector<uint8_t> p = pack(I);
   auto bit = [&](unsigned n) { return (p[n / 8] >> (n % 8)) & 1; };
   EXPECT_EQ(bit(56), 1);
   EXPECT_EQ(bit(57), 1);
   EXPECT_EQ(((p[3] & 127) >> 1) | bit(20) << 6 | bit(58) << 7, 255);

   I = agx_ffma_to(b, r(95), r(64), r(65), u(128));
   p = pack(I);
   EXPECT_EQ(bit(56), 1);
   EXPECT_EQ(bit(37), 1);
   EXPECT_EQ(((p[5] & 127) >> 1) | bit(39) << 6 | bit(38) << 7, 128);

   /* The six-byte form reaches u0-u63 in operand B */
   I = agx_ffma_to(b, r(4), r(5), u(63), r(6));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_FLOAT3_COMPACT);
   I->src[1] = u(64);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_FLOAT3_EXTENDED);
}

TEST_F(Apple9Pack, LowRegisterSelect)
{
   agx_instr *I = agx_icmpsel_to(b, r(0), k(r(0)), k(r(2)), k(r(4)), k(r(3)),
                                 AGX_ICOND_ULT);
   check(I, {0x02, 0x01, 0x1f, 0x05, 0x82, 0x08, 0x05, 0x00, 0x80, 0x06});

   I->src[0] = r(0);
   I->src[1] = r(2);
   check(I, {0x02, 0x81, 0x07, 0x85, 0x82, 0x08, 0x05, 0x00, 0x80, 0x06});

   I->src[2] = r(4);
   I->src[3] = r(3);
   check(I, {0x02, 0x81, 0x07, 0x85, 0x02, 0x88, 0x05, 0x00, 0x00, 0x86});

   for (unsigned slot = 0; slot <= 6; ++slot) {
      wait(I, slot);
      EXPECT_EQ((pack(I)[7] >> 5) & 7, slot);
   }
}

TEST_F(Apple9Pack, SelectConditions)
{
   static const struct {
      bool floating;
      unsigned cond;
      uint8_t mode, condition;
   } cases[] = {
      {true, AGX_FCOND_GT, 0x82, 0x02},   {true, AGX_FCOND_LT, 0x82, 0x03},
      {false, AGX_ICOND_UGT, 0x82, 0x04}, {false, AGX_ICOND_ULT, 0x82, 0x05},
      {false, AGX_ICOND_SGT, 0x82, 0x06}, {false, AGX_ICOND_SLT, 0x82, 0x07},
      {true, AGX_FCOND_EQ, 0x86, 0x00},   {false, AGX_ICOND_UEQ, 0x86, 0x07},
      {true, AGX_FCOND_GE, 0x86, 0x02},
   };

   for (const auto &test : cases) {
      agx_instr *I = test.floating
                        ? agx_fcmpsel_to(b, r(0), k(r(0)), k(r(2)), k(r(4)),
                                         k(r(3)), (enum agx_fcond)test.cond)
                        : agx_icmpsel_to(b, r(0), k(r(0)), k(r(2)), k(r(4)),
                                         k(r(3)), (enum agx_icond)test.cond);
      std::vector<uint8_t> p = pack(I);
      ASSERT_EQ(p.size(), 10u);
      EXPECT_EQ(p[4], test.mode);
      EXPECT_EQ(p[6], test.condition);
   }

   /* a <= b is b >= a */
   agx_instr *le =
      agx_fcmpsel_to(b, r(0), r(1), r(2), r(4), r(3), AGX_FCOND_LE);
   agx_instr *ge =
      agx_fcmpsel_to(b, r(0), r(2), r(1), r(4), r(3), AGX_FCOND_GE);
   EXPECT_EQ(pack(le), pack(ge));
}

TEST_F(Apple9Pack, PredicatePolarityAndLifetime)
{
   agx_instr *I = agx_icmp_pred(b, k(r(1)), k(r(2)), AGX_ICOND_SLT, false, 0);
   check(I, {0x0a, 0x03, 0x3a, 0x05, 0x07, 0x00});

   /* The 0xc0 tail of Metal's captures is a wait on slot 6 */
   wait(I, 6);
   check(I, {0x0a, 0x03, 0x3a, 0x05, 0x07, 0xc0});

   I->invert_cond = true;
   check(I, {0x1a, 0x03, 0x3a, 0x05, 0x07, 0xc0});
   I->invert_cond = false;

   I->src[0] = r(1);
   check(I, {0x0a, 0x03, 0x32, 0x05, 0x07, 0xc0});
   I->src[0] = k(r(1));
   I->src[1] = r(2);
   check(I, {0x0a, 0x03, 0x2a, 0x05, 0x07, 0xc0});
   I->src[0] = r(1);
   check(I, {0x0a, 0x03, 0x22, 0x05, 0x07, 0xc0});

   I->src[0] = k(r(63));
   I->src[1] = k(r(62));
   I->icond = AGX_ICOND_ULT;
   check(I, {0x0a, 0x7f, 0x3a, 0x7d, 0x05, 0xc0});

   I->src[0] = k(r(1));
   I->src[1] = k(r(2));
   I->icond = AGX_ICOND_UEQ;
   check(I, {0x0a, 0x03, 0x3b, 0x05, 0x06, 0x00, 0x07, 0xc0, 0x00, 0x00});

   agx_instr *F = agx_fcmp_pred(b, k(r(1)), k(r(2)), AGX_FCOND_EQ, false, 0);
   wait(F, 6);
   check(F, {0x0a, 0x03, 0x3b, 0x05, 0x06, 0x00, 0x00, 0xc0, 0x00, 0x00});

   F->fcond = AGX_FCOND_GE;
   F->invert_cond = true;
   check(F, {0x1a, 0x03, 0x3b, 0x05, 0x06, 0x00, 0x02, 0xc0, 0x00, 0x00});

   /* Breaks compare into the loop's predicate, keeping both sources */
   agx_instr *L = agx_icmp_pred(b, r(1), r(2), AGX_ICOND_UEQ, false, 1);
   check(L, {0x2a, 0x03, 0x23, 0x05, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00});
   L->pred = 2;
   check(L, {0x4a, 0x03, 0x23, 0x05, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00});
}

TEST_F(Apple9Pack, ExecutionMaskScope)
{
   check(agx_if_pred(b, 0, false), {0x0f, 0x05, 0x54, 0x01});
   check(agx_else_exec(b), {0x0f, 0x04, 0x04, 0x19});
   check(agx_pop_exec(b, 1), {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00});
   check(agx_if_pred(b, 0, true), {0x0f, 0x05, 0x54, 0x21});
}

TEST_F(Apple9Pack, LoopControl)
{
   check(agx_begin_loop(b), {0x0f, 0x05, 0x54, 0x1a});
   check(agx_while_pred(b, 0, false), {0x8f, 0x04, 0x54, 0x22});
   check(agx_while_pred(b, 2, false), {0x8f, 0x04, 0x54, 0x2a});
   check(agx_break_pred(b, 3, 2), {0x8f, 0x05, 0x54, 0x03, 0x00, 0x02});
   check(agx_end_loop(b), {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00});
}

/* Branches are relative to their own start, 48-bit signed in bytes 3-8 */
TEST_F(Apple9Pack, Branches)
{
   agx_context *ctx = b->shader;
   agx_block *first = agx_start_block(ctx);
   agx_block *second = agx_test_block(ctx);

   b->cursor = agx_after_block(first);
   agx_mov_imm_to(b, r(0), 1);
   agx_jmp_exec_none(b, second);

   b->cursor = agx_after_block(second);
   agx_mov_imm_to(b, r(1), 2);
   agx_jmp_exec_any(b, first);
   agx_stop(b);

   struct util_dynarray binary;
   util_dynarray_init(&binary, mem_ctx);
   agx_pack_binary(ctx, &binary);

   const uint8_t *code = (const uint8_t *)binary.data;
   ASSERT_EQ(binary.size, 2 + 10 + 2 + 10 + 4);

   const uint8_t none[] = {0x0f, 0x01, 0x54, 10, 0, 0, 0, 0, 0, 0x00};
   EXPECT_EQ(memcmp(code + 2, none, sizeof(none)), 0);

   /* Back by 14 bytes, to the start */
   const uint8_t any[] = {0x0f, 0x00, 0x54, 0xf2, 0xff,
                          0xff, 0xff, 0xff, 0xff, 0x00};
   EXPECT_EQ(memcmp(code + 14, any, sizeof(any)), 0);

   const uint8_t halt[] = {0x0e, 0, 0, 0};
   EXPECT_EQ(memcmp(code + 24, halt, sizeof(halt)), 0);
}

TEST_F(Apple9Pack, SubgroupMaxHalves)
{
   agx_index dst = rh(3), src = k(rh(3));
   dst.cache = true;
   check(agx_simd_reduce_to(b, dst, src, AGX_SIMD_OP_UMAX),
         {0xbf, 3, 0x54, 7, 3, 14, 0x10, 3});

   check(agx_simd_reduce_to(b, rl(5), rl(3), AGX_SIMD_OP_UMAX),
         {0xbf, 3, 0x54, 10, 2, 12, 0x10, 1});
   check(agx_simd_reduce_to(b, r(5), r(3), AGX_SIMD_OP_UMAX),
         {0xbf, 3, 0x54, 10, 2, 12, 0x14, 1});
}

TEST_F(Apple9Pack, FloatExport)
{
   agx_instr *I = agx_fmul_to(b, e(7), k(r(16)), k(r(17)));
   check(I, {0x79, 0x21, 0x3d, 0x23, 0x41, 0, 0, 0});

   /* The wait (bits 45-47, 61-63) is independent of the completion slot */
   for (unsigned input = 0; input <= 6; ++input) {
      wait(I, input);
      I->scoreboard = 5 - input % 6;
      std::vector<uint8_t> p = pack(I);
      EXPECT_EQ((p[5] >> 5) | ((p[7] >> 5) << 3),
                input ? 1u << (input - 1) : 0u);
      EXPECT_EQ((p[7] >> 2) & 7, 5u - input % 6);
   }
}

TEST_F(Apple9Pack, LogicExportBounds)
{
   agx_instr *I = agx_xor_to(b, e(127), k(r(95)), k(r(95)));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_LOGIC_EXPORT);
   EXPECT_EQ(pack(I)[4] & 0x40, 0x40);
}

TEST_F(Apple9Pack, TileReadWithMaskRegister)
{
   static const uint8_t tags[][2] = {
      {0x4e, 0}, {0xce, 0}, {0x4e, 1}, {0xce, 1}, {0x4e, 2}, {0xce, 2},
   };

   for (unsigned slot = 0; slot < 6; ++slot) {
      for (bool live : {false, true}) {
         agx_instr *I = agx_ld_tile_to(b, r(9), live ? rl(7) : k(rl(7)),
                                       agx_null(), AGX_FORMAT_I32, 1, 0, false);
         I->scoreboard = slot;
         check(I, {0x67, 0x06, 0x54, 18, 0, 0, 14, tags[slot][0],
                   (uint8_t)(tags[slot][1] | (live ? 0 : 0x20)), 0, 0, 0x20});
      }
   }
}

TEST_F(Apple9Pack, TileReadSlot)
{
   agx_instr *I = agx_ld_tile_to(b, r(0), agx_immediate(1), agx_null(),
                                 AGX_FORMAT_I32, 1, 0, false);
   I->scoreboard = 5;
   check(I, {0x67, 0x0e, 0x54, 0, 0, 0, 1, 0xce, 2, 0, 0, 0});

   I->dest[0] = r(95);
   I->scoreboard = 0;
   std::vector<uint8_t> p = pack(I);
   EXPECT_EQ(p[3], 190);
   EXPECT_EQ(p[7], 0x4e);
   EXPECT_EQ(p[8], 0);
}

TEST_F(Apple9Pack, TileReadAtCoordinates)
{
   for (unsigned sample = 0; sample < 4; ++sample) {
      agx_instr *I = agx_ld_tile_to(b, r(25), agx_immediate(1 << sample),
                                    k(r(8)), AGX_FORMAT_I32, 1, 0, true);
      std::vector<uint8_t> p = pack(I);
      ASSERT_EQ(p.size(), 12u);
      EXPECT_EQ(p[3], 50);
      EXPECT_EQ(p[4], 8);
      EXPECT_EQ(p[6], 1 << sample);
      EXPECT_EQ(p[11], 0x10);
      EXPECT_EQ(p[8], 0);
   }
}

TEST_F(Apple9Pack, DerivativeAxis)
{
   for (bool y : {false, true}) {
      for (bool keep : {false, true}) {
         agx_index src = keep ? r(23) : k(r(23));
         agx_instr *I =
            y ? agx_dfdy_to(b, r(9), src) : agx_dfdx_to(b, r(9), src);
         std::vector<uint8_t> p = pack(I);
         ASSERT_EQ(p.size(), 10u);
         EXPECT_EQ(p[1] & 7, y ? 7 : 5);
         EXPECT_EQ(p[3], 18);
         EXPECT_EQ(p[5], 92);
         EXPECT_EQ(p[6], keep ? 0x92 : 0x90);

         I->src[0] = keep ? r(95) : k(r(95));
         p = pack(I);
         EXPECT_EQ(p[5], 124);
         EXPECT_EQ(p[6], keep ? 0x93 : 0x91);
      }
   }
}

TEST_F(Apple9Pack, SpillsSplitOperandsAndSlots)
{
   agx_instr *save =
      agx_stack_store(b, k(r(82)), agx_immediate(71 * 4), AGX_FORMAT_I32, 1);
   save->scoreboard = 5;
   wait(save, 3);
   check(save, {0x7b, 0x25, 0x0f, 0, 0x22, 0x91, 0, 0x14, 0, 0});

   agx_instr *fill =
      agx_stack_load_to(b, r(18), agx_immediate(71 * 4), AGX_FORMAT_I32, 1);
   fill->scoreboard = 0;
   wait(fill, 6);
   check(fill, {0x2c, 0x8e, 0x41, 0x01, 0x02, 0, 0, 0x80});
}

TEST_F(Apple9Pack, AcquireFenceIsTheDeviceFence)
{
   EXPECT_EQ(pack(agx_memory_barrier(b)), pack(agx_memory_barrier_acquire(b)));
}

TEST_F(Apple9Pack, SystemRegisters)
{
   for (unsigned c = 0; c < 3; ++c) {
      check(agx_get_sr_to(b, r(0),
                          (enum agx_sr)(AGX_SR_THREAD_POSITION_IN_GRID_X + c)),
            {0x0c, (uint8_t)(0xa0 + c), 0x10, 0x06});
   }

   check(agx_get_sr_to(b, r(1), AGX_SR_THREAD_POSITION_IN_THREADGROUP_X),
         {0x14, 0xa4, 0x10, 0x06, 0x13, 0x00, 0x00, 0x01});

   for (unsigned dst = 0; dst < 64; ++dst) {
      std::vector<uint8_t> read =
         pack(agx_get_sr_to(b, r(dst), AGX_SR_THREAD_POSITION_IN_GRID_X));
      EXPECT_EQ(read[0], ((dst & 15) << 4) | 0x0c);
      EXPECT_EQ(read[2], 0x10 | ((dst >> 4) << 6));

      std::vector<uint8_t> pair = pack(
         agx_get_sr_to(b, r(dst), AGX_SR_THREAD_POSITION_IN_THREADGROUP_X));
      EXPECT_EQ(pair[2], read[2]);
      EXPECT_EQ(pair[4], ((dst & 15) << 4) | 3);
      EXPECT_EQ(pair[6], (dst >> 4) << 6);
      EXPECT_EQ(pair[7], 1);
   }
}

TEST_F(Apple9Pack, UniformStore)
{
   check(agx_uniform_store(b, k(r(95)), agx_immediate(2 * 255), 1),
         {0x9f, 0x01, 0x54, 0xfe, 0x01, 0x7c, 0x01, 0xa8, 0x13, 0x01});
}

/* Bytes 8-9 of a device load name its completion slot */
TEST_F(Apple9Pack, DeviceLoadSlotTokens)
{
   static const uint8_t tokens[6][2] = {
      {0x11, 0x00}, {0x51, 0x00}, {0x91, 0x00},
      {0xd1, 0x00}, {0x11, 0x01}, {0x51, 0x01},
   };

   for (unsigned slot = 0; slot < 6; ++slot) {
      agx_instr *I = agx_device_load_to(b, r(2), u64(3), r(1), AGX_FORMAT_I32,
                                        1, 0, false, 0);
      I->scoreboard = slot;
      check(I,
            {0x67, 0x00, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, tokens[slot][0],
             tokens[slot][1], 0x00, 0x40, 0x46, 0x00},
            {}, true);
   }

   /* The last load clears byte 2 bit 4; a dying index sets byte 5 bit 7 */
   agx_instr *I =
      agx_device_load_to(b, r(0), u64(0), r(1), AGX_FORMAT_I32, 1, 0, false, 0);
   I->scoreboard = 5;
   check(I, {0x67, 0x00, 0x44, 0x00, 0x00, 0x01, 0x20, 0x00, 0x51, 0x01, 0x00,
             0x40, 0x46, 0x00});
   I->src[1] = k(r(1));
   EXPECT_EQ(pack(I)[5], 0x81);
}

/* Vector loads and stores: the index shift holds the vector size */
TEST_F(Apple9Pack, VectorMemory)
{
   static const struct {
      unsigned components, dst, index, shift;
      bool follows;
      unsigned wait;
      uint8_t load[14], store_load[14], store_alu[14];
   } cases[] = {
      {2,
       0,
       2,
       1,
       false,
       1,
       {0x67, 0x10, 0x44, 0x00, 0x00, 0x02, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
        0x48, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00}},
      {3,
       4,
       7,
       2,
       true,
       0,
       {0x67, 0x00, 0x54, 0x08, 0x00, 0x07, 0x20, 0x00, 0x5d, 0x01, 0x00, 0x40,
        0x40, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00}},
      {4,
       0,
       4,
       2,
       false,
       1,
       {0x67, 0x10, 0x44, 0x00, 0x00, 0x04, 0x20, 0x00, 0x57, 0x01, 0x00, 0x40,
        0x40, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x04, 0x21, 0x00, 0x17, 0x00, 0x00, 0x10,
        0x10, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x04, 0x21, 0x00, 0x17, 0x00, 0x00, 0x10,
        0x10, 0x00}},
   };

   for (const auto &test : cases) {
      unsigned mask = BITFIELD_MASK(test.components);
      agx_index dst = r(test.dst);
      dst.channels_m1 = test.components - 1;

      agx_instr *load =
         agx_device_load_to(b, dst, u64(0), r(test.index), AGX_FORMAT_I32, mask,
                            test.shift, false, 0);
      load->scoreboard = 5;
      wait(load, test.wait);
      check(load, std::vector<uint8_t>(test.load, test.load + 14), {},
            test.follows);

      agx_index data = r(0);
      data.channels_m1 = test.components - 1;
      agx_instr *store =
         agx_device_store(b, data, u64(1), k(r(test.index)), AGX_FORMAT_I32,
                          mask, test.shift, false, 0);
      wait(store, 6);
      check(store, std::vector<uint8_t>(test.store_load, test.store_load + 14));
      wait(store, 0);
      check(store, std::vector<uint8_t>(test.store_alu, test.store_alu + 14));
   }
}

TEST_F(Apple9Pack, NarrowScalarMemory)
{
   agx_instr *I =
      agx_device_load_to(b, r(1), u64(1), r(0), AGX_FORMAT_I8, 1, 0, false, 0);
   I->scoreboard = 5;
   wait(I, 1);
   check(I, {0x67, 0x10, 0x44, 0x02, 0x01, 0x00, 0x20, 0x00, 0x61, 0x01, 0x00,
             0x40, 0x42, 0x00});

   I =
      agx_device_load_to(b, r(1), u64(2), r(0), AGX_FORMAT_I16, 1, 0, false, 0);
   I->scoreboard = 5;
   wait(I, 1);
   check(I,
         {0x67, 0x10, 0x54, 0x02, 0x02, 0x00, 0x20, 0x00, 0x41, 0x01, 0x00,
          0x40, 0x44, 0x00},
         {}, true);

   check(
      agx_device_store(b, r(0), u64(0), k(r(0)), AGX_FORMAT_I8, 1, 0, false, 0),
      {0xe7, 0x00, 0x54, 0x00, 0x00, 0x00, 0x21, 0x00, 0x21, 0x00, 0x00, 0x90,
       0x10, 0x00});
   check(agx_device_store(b, r(0), u64(0), k(r(0)), AGX_FORMAT_I16, 1, 0, false,
                          0),
         {0xe7, 0x00, 0x54, 0x00, 0x00, 0x00, 0x21, 0x00, 0x01, 0x00, 0x00,
          0x10, 0x11, 0x00});

   /* Registers and the index lifetime */
   I = agx_device_store(b, r(17), u64(3), r(9), AGX_FORMAT_I32, 1, 0, false, 0);
   std::vector<uint8_t> p = pack(I);
   EXPECT_EQ(p[2], 0x54);
   EXPECT_EQ(p[3], 34);
   EXPECT_EQ(p[4], 3);
   EXPECT_EQ(p[5], 9);
   EXPECT_EQ(p[6], 0x20);
   I->src[2] = k(r(9));
   EXPECT_EQ(pack(I)[6], 0x21);
}

/* Register-pair loads with a constant index: byte offset from bit 77 */
TEST_F(Apple9Pack, PointerLoads)
{
   static const struct {
      unsigned index, components;
      enum agx_format format;
      uint8_t bytes[14];
   } cases[] = {
      {0,
       1,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x05, 0x00, 0x40,
        0x22, 0x00}},
      {1,
       1,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x85, 0x00, 0x40,
        0x22, 0x00}},
      {255,
       1,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x85, 0x7f, 0x40,
        0x22, 0x00}},
      {4096,
       1,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x05, 0x00, 0x48,
        0x22, 0x00}},
      {0,
       2,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x05, 0x00, 0x40,
        0x22, 0x00}},
      {0,
       4,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x57, 0x05, 0x00, 0x40,
        0x22, 0x00}},
      {4,
       4,
       AGX_FORMAT_I32,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x57, 0x05, 0x02, 0x40,
        0x22, 0x00}},
      {3,
       1,
       AGX_FORMAT_I8,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61, 0x65, 0x00, 0x40,
        0x22, 0x00}},
      {3,
       1,
       AGX_FORMAT_I16,
       {0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0xc5, 0x00, 0x40,
        0x22, 0x00}},
   };

   for (const auto &test : cases) {
      agx_index dst = r(0);
      dst.channels_m1 = test.components - 1;
      agx_instr *I = agx_device_load_to(
         b, dst, k(r64(0)), agx_immediate(test.index), test.format,
         BITFIELD_MASK(test.components), 0, false, 0);
      I->scoreboard = 5;
      EXPECT_EQ(form(I), AGX_APPLE9_ENC_DEVICE_LOAD_POINTER);
      check(I, std::vector<uint8_t>(test.bytes, test.bytes + 14),
            {0, 0xf0, 0x03});
   }
}

TEST_F(Apple9Pack, DeviceAtomics)
{
   agx_instr *I =
      agx_atomic_to(b, r(30), r(22), u64(5), r(23), AGX_ATOMIC_OPC_ADD);
   std::vector<uint8_t> p = pack(I);
   const uint8_t add[] = {0x67, 0x01, 0x54, 0x00, 0x00, 0x05, 0x8b,
                          0x8b, 0x00, 0x02, 0x00, 0x00, 0x60, 0x02};
   ASSERT_EQ(p.size(), 22u);
   EXPECT_EQ(memcmp(p.data(), add, sizeof(add)), 0);

   agx_index pair = r(23);
   pair.channels_m1 = 1;
   I = agx_atomic_to(b, r(30), pair, u64(5), r(25), AGX_ATOMIC_OPC_CMPXCHG);
   p = pack(I);
   const uint8_t cmpxchg[] = {0x67, 0x01, 0x54, 0x00, 0x00, 0x85, 0x8b,
                              0x8c, 0x00, 0x02, 0x00, 0x00, 0x64, 0x02};
   EXPECT_EQ(memcmp(p.data(), cmpxchg, sizeof(cmpxchg)), 0);

   I = agx_atomic_to(b, agx_null(), r(4), u64(9), r(5), AGX_ATOMIC_OPC_XOR);
   check(I, {0x67, 0x01, 0x54, 0x00, 0x00, 0x09, 0x82, 0x82, 0x00, 0x40, 0x00,
             0x00, 0x7e, 0x02});

   wait(I, 6);
   p = pack(I);
   EXPECT_EQ(p[1], 0x01);
   EXPECT_EQ(p[2], 0x56);
}

/* Result record after a returning atomic: destination and completion slot */
TEST_F(Apple9Pack, DeviceAtomicResult)
{
   for (unsigned dst = 0; dst < 96; ++dst) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         agx_instr *I =
            agx_atomic_to(b, r(dst), r(4), u64(9), r(5), AGX_ATOMIC_OPC_ADD);
         I->scoreboard = slot - 1;
         std::vector<uint8_t> p = pack(I);
         ASSERT_EQ(p.size(), 22u);

         const uint8_t result[] = {
            (uint8_t)(((dst & 0xf) << 4) | 0x0c),
            0x80,
            (uint8_t)(0x09 | (((dst >> 4) & 3) << 6)),
            (uint8_t)(((slot - 1) << 5) | 0x07),
            0x00,
            0x20,
            0x00,
            (uint8_t)((dst >> 6) << 4),
         };
         EXPECT_EQ(memcmp(p.data() + 14, result, sizeof(result)), 0)
            << "dst " << dst << " slot " << slot;
      }
   }
}

/* Unverified: threadgroup memory */
TEST_F(Apple9Pack, UnverifiedThreadgroupMemory)
{
   agx_instr *I =
      agx_local_load_to(b, r(5), agx_immediate(3), k(rl(7)), AGX_FORMAT_I32, 1);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_SHARED_LOAD);
   check(I, {0x67, 0x02, 0x54, 0x0a, 0x0c, 0x1c, 0x80, 0, 0x44, 0x0d, 0, 0xc0,
             0x08, 0});

   I = agx_local_store(b, k(r(5)), agx_immediate(3), rl(7), AGX_FORMAT_I32, 1);
   check(I, {0xe7, 0x02, 0x54, 0x0a, 0x0c, 0x1c, 0x00, 0, 0x44, 0x02, 0, 0x30,
             0x02, 0});

   I = agx_local_atomic_to(b, r(6), k(r(8)), agx_immediate(1), k(rl(9)),
                           AGX_ATOMIC_OPC_ADD);
   check(I, {0x67, 3, 0x54, 0x06, 0x02, 0x08, 0x09, 0, 0x12, 0, 0x00, 0x4c});
}

/* Unverified: subgroup scan, broadcast and ballot */
TEST_F(Apple9Pack, UnverifiedSubgroups)
{
   check(agx_simd_prefix_to(b, r(3), r(4), AGX_SIMD_OP_IADD),
         {0xbf, 0x01, 0x54, 0x06, 0x03, 0x10, 0x16, 0x09});
   check(agx_shuffle_to(b, r(3), r(4), rl(5)),
         {0x47, 0x04, 0x54, 0x06, 0x03, 0x10, 0x14, 0x2c, 0x10, 0});
   check(agx_ballot_to(b, r(3), k(r(4))),
         {0x17, 0x07, 0x54, 0x06, 0x03, 0x10, 0x00, 0x58, 0x22, 0x12});
}

/* Unverified: barriers, halt and the wait instruction */
TEST_F(Apple9Pack, UnverifiedSynchronization)
{
   check(agx_threadgroup_barrier(b), {0x07, 0x04, 0x54, 0x61, 0x09, 0});
   check(agx_memory_barrier(b), {0x07, 0x04, 0x54, 0x84, 0x0a, 0});
   check(agx_raster_order_fence(b), {0x07, 0x04, 0x54, 0xc4, 0x08, 0});
   check(agx_stop(b), {0x0e, 0, 0, 0});

   for (unsigned slot = 0; slot < 6; ++slot)
      check(agx_wait(b, slot), {0x06, (uint8_t)(1u << (slot + 2))});
}

/* Unverified: fragment outputs and tile access */
TEST_F(Apple9Pack, UnverifiedFragmentOutputs)
{
   check(agx_write_coverage(b, e(5)), {0x57, 0x04, 0x54, 0x0a, 0, 1});

   agx_index zs = e(6);
   zs.channels_m1 = 1;
   check(agx_zs_store(b, zs, 1), {0xd7, 0x14, 0x54, 0x0c, 0, 3});
   check(agx_zs_store(b, zs, 2), {0xd7, 0x14, 0x54, 0x0c, 0, 1});
   check(agx_zs_store(b, zs, 3), {0xd7, 0x14, 0x54, 0x0c, 0, 7});

   check(agx_begin_tile(b, 0x700), {0x87, 2, 0x54, 0x00, 0x07, 0});
   check(agx_end_tile(b, 0x300), {0x07, 2, 0x54, 0x00, 0x03, 0});

   check(agx_st_tile(b, r(9), agx_immediate(1), agx_null(), AGX_FORMAT_I32, 1,
                     2, false),
         {0xe7, 0x06, 0x54, 18, 0, 4, 1, 0x4e, 0, 0, 0, 0});
   check(
      agx_st_tile(b, r(9), k(rl(7)), agx_null(), AGX_FORMAT_I32, 1, 2, false),
      {0xe7, 0x06, 0x54, 18, 0, 4, 14, 0x4e, 8, 0, 0, 8});
}

/* Unverified: varyings and interpolation */
TEST_F(Apple9Pack, UnverifiedVaryings)
{
   check(agx_st_vary(b, agx_immediate(9), e(12)),
         {0x57, 0x06, 0x54, 0x18, 0x20, 0x41, 0, 0});

   check(agx_iter_to(b, r(10), agx_immediate(3), agx_null(), 1,
                     AGX_INTERPOLATION_CENTER),
         {0x2f, 0x05, 0x54, 0x14, 0x03, 0x06, 0, 2, 0x10, 0});
   check(agx_iter_to(b, r(10), agx_immediate(3), r(11), 1,
                     AGX_INTERPOLATION_CENTROID),
         {0x2f, 0x05, 0x54, 0x14, 0x03, 0x06, 11, 2, 0x08, 0});

   agx_index tuple = r(20);
   tuple.channels_m1 = 2;
   agx_instr *I = agx_ldcf_to(b, tuple, agx_immediate(5), 3);
   I->scoreboard = 2;
   check(I, {0x1f, 0x03, 0x54, 0x94, 0x02, 0x02});

   check(agx_interpolation_position_to(b, r(4), k(r(70)),
                                       AGX_INTERPOLATION_SAMPLE),
         {0xaf, 0x04, 0x54, 0x08, 0x03, 0x18, 0x0b, 0x03});
   check(agx_interpolation_position_to(b, r(4), k(r(7)),
                                       AGX_INTERPOLATION_CENTROID),
         {0xaf, 0x04, 0x54, 0x08, 0x03, 0x1c, 0x0a, 0x01});
}

/* Unverified: FP32 to FP16 into one half, FP16 to FP32 from either half */
TEST_F(Apple9Pack, UnverifiedHalfConversions)
{
   check(agx_fadd_to(b, rh(4), k(r(70)), agx_negzero()),
         {0x41, 0x0d, 0x1c, 0x81, 0x04, 0x03});
   check(agx_fadd_to(b, r(4), k(rh(3)), agx_negzero()),
         {0x49, 0x06, 0x1c, 0x81, 0x08, 0x02});
}

/* Unverified: register-pointer stores and atomics */
TEST_F(Apple9Pack, UnverifiedPointerStoreAndAtomic)
{
   agx_instr *I = agx_device_store(b, r(3), r64(10), k(r(4)), AGX_FORMAT_I32, 1,
                                   0, false, 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT);
   check(I, {0xe7, 0x00, 0x54, 0x06, 0x0a, 0x04, 0x21, 0x00, 0x11, 0x00, 0x00,
             0x90, 0x19, 0x00});

   I = agx_atomic_to(b, agx_null(), r(4), r64(10), r(5), AGX_ATOMIC_OPC_ADD);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT);
   check(I, {0x67, 0x01, 0x54, 0x00, 0x00, 0x0a, 0x82, 0x82, 0x00, 0x60, 0x00,
             0x00, 0x60, 0x03});
}

/* FP16 arithmetic on the low halves of registers */
TEST_F(Apple9Pack, HalfArithmetic)
{
   agx_instr *I = agx_hmul_to(b, r(6), r(4), r(3));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF2);
   check(I, {0x60, 0x88, 0x05, 0x86, 0x00, 0x00});

   check(agx_hadd_to(b, r(4), k(r(4)), r(3)),
         {0x40, 0x08, 0x0c, 0x86, 0x00, 0x00});

   I = agx_hfma_to(b, r(5), r(4), r(3), r(4));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF3);
   check(I, {0x50, 0x88, 0x06, 0x86, 0x01, 0x88, 0x00, 0x00});
}

/* FP16 to FP32 from the low half of a register, and back */
TEST_F(Apple9Pack, HalfConversionsOfRegisters)
{
   agx_instr *I = agx_f16_to_f32_to(b, r(10), k(r(5)));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF_TO_FLOAT);
   check(I, {0xa9, 0x0a, 0x1c, 0x81, 0x00, 0x02});

   I = agx_f32_to_f16_to(b, r(4), k(r(70)));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT);
   check(I, {0x41, 0x0d, 0x1c, 0x81, 0x00, 0x03});
}

/* PACK_HALF_2X16 narrows into both halves, the high one with bit 34 */
TEST_F(Apple9Pack, PackHalf)
{
   agx_instr *I = agx_pack_half_to(b, r(4), k(r(70)), r(5));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_PACK_HALF_2X16);
   check(I, {0x41, 0x0d, 0x1c, 0x81, 0x00, 0x03, 0x41, 0x8b, 0x14, 0x81, 0x04,
             0x02});

   I = agx_pack_half_to(b, r(4), r(3), k(r(3)));
   check(I, {0x41, 0x87, 0x14, 0x81, 0x00, 0x02, 0x41, 0x07, 0x1c, 0x81, 0x04,
             0x02});
}

/* Private memory at a word index in a register */
TEST_F(Apple9Pack, IndexedPrivateMemory)
{
   agx_instr *I = agx_stack_load_to(b, r(3), r(3), AGX_FORMAT_I32, 1);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_PRIVATE_LOAD);
   check(I, {0x6f, 0x00, 0x54, 0x06, 0x06, 0x02, 0x49, 0x01, 0x00, 0x00, 0x03,
             0x00});

   I = agx_stack_store(b, r(5), r(4), AGX_FORMAT_I32, 1);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_PRIVATE_STORE);
   check(I, {0xef, 0x00, 0x54, 0x0a, 0x08, 0x02, 0x09, 0x03, 0x00, 0x60});
}

/* A 32-bit threadgroup index names the register holding the word index */
TEST_F(Apple9Pack, ThreadgroupIndexRegister)
{
   check(
      agx_local_load_to(b, r(5), agx_immediate(3), k(r(7)), AGX_FORMAT_I32, 1),
      {0x67, 0x02, 0x54, 0x0a, 0x0c, 0x1c, 0x80, 0, 0x44, 0x0d, 0, 0xc0, 0x08,
       0});
}

/* Texture operations. Register textures are r40, register samplers r41 */
struct texture_case {
   enum agx_opcode op;
   enum agx_dim dim;
   enum agx_lod_mode lod;
   bool shadow, query;
   enum agx_gather gather;
   unsigned mask, slot, dst, coord, tex, smp;
   bool reg_tex, reg_smp;
   int offset;
   std::vector<uint8_t> bytes;
};

TEST_F(Apple9Pack, Texture)
{
   const enum agx_opcode S = AGX_OPCODE_TEXTURE_SAMPLE;
   const enum agx_opcode L = AGX_OPCODE_TEXTURE_LOAD;

   /* clang-format off */
   static const texture_case cases[] = {
      /* Implicit LOD, gather, partial result */
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x00, 0x00, 0x80, 0x01, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, false, false, AGX_GATHER_A, 0xf, 3, 59, 16, 31, 1, false, false, -1,
       {0xdd, 0xf8, 0x4c, 0xbe, 0x90, 0x6c, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, false, false, AGX_GATHER_NONE, 0x9, 4, 20, 24, 6, 3, false, false, -1,
       {0xa5, 0x98, 0x0c, 0xa3, 0xb1, 0x90, 0x00, 0x00, 0x00, 0x01, 0x10, 0x00, 0x01, 0x00}},
      /* Explicit LOD, bias with a minimum, LOD query */
      {S, AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x00, 0x01, 0x80, 0x01, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD_BIAS_MIN, false, false, AGX_GATHER_NONE, 0xf, 2, 40, 8, 17, 16, false, false, -1,
       {0x45, 0xc0, 0x4c, 0xb9, 0xb8, 0x48, 0x04, 0x08, 0x80, 0x00, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD_BIAS, false, true, AGX_GATHER_NONE, 0x3, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xa8, 0xb2, 0x24, 0x04, 0x00, 0x80, 0x01, 0x20, 0x00, 0x01, 0x00}},
      /* Cubes, arrays, 3D */
      {S, AGX_DIM_CUBE, AGX_LOD_MODE_LOD_MIN, true, false, AGX_GATHER_NONE, 0x1, 5, 3, 28, 9, 15, false, false, -1,
       {0x1d, 0xa4, 0x0c, 0xa3, 0x97, 0xb4, 0x20, 0x03, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_CUBE_ARRAY, AGX_LOD_MODE_AUTO_LOD_BIAS, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x1c, 0x00, 0x80, 0x01, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_G, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xac, 0x92, 0x24, 0x08, 0x05, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      /* Texel fetches */
      {L, AGX_DIM_3D, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x98, 0x01, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      {L, AGX_DIM_2D_MS_ARRAY, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0xa0, 0x14, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      /* Offsets (1, -1, 0) */
      {S, AGX_DIM_3D, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, 0x0f1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x19, 0x02, 0x80, 0x89, 0x10, 0x00, 0x01, 0x0e}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD_BIAS, false, false, AGX_GATHER_NONE, 0xf, 2, 40, 8, 17, 16, false, false, 0x0f1,
       {0x45, 0xc0, 0x4c, 0xb9, 0xb8, 0x48, 0x81, 0x02, 0x80, 0x88, 0x10, 0x00, 0x01, 0x0e}},
      /* Gradients */
      {S, AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x04, 0x01, 0x80, 0x01, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_GRAD, true, false, AGX_GATHER_NONE, 0x1, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xa0, 0x92, 0x24, 0x2c, 0x05, 0x80, 0x01, 0x00, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 1, 5, false, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x14, 0x01, 0x80, 0x01, 0x10, 0x00, 0x01, 0x00}},
      {S, AGX_DIM_3D, AGX_LOD_MODE_LOD_GRAD, false, false, AGX_GATHER_NONE, 0xf, 4, 20, 16, 6, 3, false, false, -1,
       {0xa5, 0x98, 0x0c, 0xba, 0xb1, 0x90, 0x9c, 0x01, 0x00, 0x01, 0x10, 0x00, 0x01, 0x00}},
      /* Register texture, then register texture and sampler */
      {L, AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, false, false, AGX_GATHER_NONE, 0xf, 1, 13, 4, 0, 5, true, false, -1,
       {0x6d, 0x84, 0x0c, 0xb8, 0xb2, 0x24, 0x80, 0x24, 0x50, 0x03, 0x80, 0x00, 0x01, 0x00, 0x00, 0x20, 0x00, 0x00, 0x40, 0x00}},
      {S, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD_BIAS, false, false, AGX_GATHER_NONE, 0xf, 3, 59, 16, 0, 0, true, true, -1,
       {0xdd, 0x80, 0x4c, 0xba, 0xb2, 0x6c, 0x0c, 0x04, 0x5a, 0x02, 0x90, 0x01, 0x01, 0x00, 0xa0, 0x28, 0x00, 0x00, 0x40, 0x00}},
   };
   /* clang-format on */

   for (const texture_case &c : cases) {
      unsigned comps = c.gather != AGX_GATHER_NONE ? 4 : util_bitcount(c.mask);
      agx_index dst = r(c.dst);
      dst.channels_m1 = comps - 1;

      agx_instr *I = agx_texture_sample_to(
         b, dst, e(c.coord), agx_null(), agx_null(),
         c.reg_tex ? (c.reg_smp ? k(r(40)) : r(40)) : agx_immediate(c.tex),
         c.reg_smp ? k(r(41)) : agx_immediate(c.smp),
         c.offset >= 0 ? agx_immediate(c.offset) : agx_null(), c.dim, c.lod,
         c.mask, c.offset >= 0, c.shadow, c.query, c.gather, false);
      I->op = c.op;
      I->scoreboard = c.slot;
      I->src[0].channels_m1 = agx_apple9_texture_params(I) - 1;

      EXPECT_EQ(form(I), agx_apple9_texture_params(I) == 2
                            ? AGX_APPLE9_ENC_TEXTURE_SAMPLE
                         : c.lod == AGX_LOD_MODE_LOD_GRAD ||
                               c.lod == AGX_LOD_MODE_LOD_GRAD_MIN
                            ? AGX_APPLE9_ENC_TEXTURE_GRAD
                            : AGX_APPLE9_ENC_TEXTURE_LOD);
      check(I, c.bytes);
   }
}

/* Forms without a measured mode are not encodable */
TEST_F(Apple9Pack, TextureForms)
{
   agx_instr *I = agx_texture_sample_to(
      b, r(0), e(0), agx_null(), agx_null(), agx_immediate(0), agx_immediate(0),
      agx_null(), AGX_DIM_CUBE, AGX_LOD_MODE_LOD_MIN, 1, false, false, false,
      AGX_GATHER_NONE, false);
   EXPECT_TRUE(agx_apple9_texture_encodable(I));

   /* Cubes have no texel fetch, 3D no minimum LOD, cubes no offsets */
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   EXPECT_FALSE(agx_apple9_texture_encodable(I));

   I->op = AGX_OPCODE_TEXTURE_SAMPLE;
   I->offset = true;
   EXPECT_FALSE(agx_apple9_texture_encodable(I));

   I->offset = false;
   I->dim = AGX_DIM_3D;
   I->lod_mode = AGX_LOD_MODE_LOD_GRAD_MIN;
   EXPECT_FALSE(agx_apple9_texture_encodable(I));
}

/* Cube face selection of r5, r70 and r33: kept sources, then released ones */
TEST_F(Apple9Pack, Cube)
{
   agx_index major = r(40);
   major.channels_m1 = 1;

   agx_instr *I = agx_cube_to(b, major, r(5), r(70), r(33), 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_CUBE);
   check(I, {0x17, 0x01, 0x54, 0x28, 0x03, 0x0a, 0x19, 0x0b, 0x55, 0x21, 0x92,
             0x00});

   check(
      agx_cube_to(b, r(40), k(r(5)), k(r(70)), k(r(33)), 1),
      {0x17, 0x01, 0x54, 0x28, 0x03, 0x0a, 0x18, 0x09, 0x51, 0x2f, 0x92, 0x04});

   check(
      agx_cube_to(b, r(40), r(5), r(70), r(33), 2),
      {0x17, 0x01, 0x54, 0x28, 0x03, 0x0a, 0x19, 0x0b, 0x55, 0x21, 0x92, 0x08});
}

/* PBE image writes of float and integer data, with their export join */
TEST_F(Apple9Pack, ImageWrite)
{
   agx_index params = e(0);
   params.channels_m1 = 7;
   agx_index image = k(r(2));
   image.channels_m1 = 1;

   agx_instr *I = agx_image_write(b, params, agx_null(), agx_null(), agx_null(),
                                  image, AGX_DIM_2D, false, false);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_IMAGE_WRITE);
   check(I, {0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x04, 0x21, 0x04, 0x00,
             0x00, 0x88, 0x3a, 0x89, 0x00, 0x07, 0x12, 0x54, 0x00, 0x03, 0x00});

   params = e(8);
   params.channels_m1 = 7;
   image = r(6);
   image.channels_m1 = 1;
   check(agx_image_write(b, params, agx_null(), agx_null(), agx_null(), image,
                         AGX_DIM_2D, false, true),
         {0xd7, 0x10, 0x54, 0x10, 0x00, 0x40, 0x00, 0x0c, 0x20, 0x04, 0x00,
          0x00, 0x88, 0x1a, 0x89, 0x00, 0x07, 0x12, 0x54, 0x00, 0x03, 0x00});
}

/* Block image stores of 2D, array and multisampled images */
TEST_F(Apple9Pack, BlockImageStore)
{
   struct {
      unsigned base, image, format;
      enum agx_dim dim;
      std::vector<uint8_t> bytes;
   } cases[] = {
      {0,
       0,
       7,
       AGX_DIM_2D,
       {0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80, 0xa8, 0x75, 0x01, 0x00,
        0x07, 0x12, 0x54, 0x00, 0x03, 0x00}},
      {4,
       3,
       5,
       AGX_DIM_2D,
       {0x57, 0x10, 0x54, 0x08, 0x00, 0x30, 0x00, 0x80, 0xa8, 0x55, 0x01, 0x00,
        0x07, 0x12, 0x54, 0x00, 0x03, 0x00}},
      {8,
       15,
       3,
       AGX_DIM_2D_ARRAY,
       {0x57, 0x10, 0x54, 0x10, 0x00, 0xf0, 0x00, 0x80, 0x28, 0x35, 0x01, 0x00,
        0x07, 0x12, 0x54, 0x00, 0x03, 0x00}},
      {12,
       1,
       1,
       AGX_DIM_2D_MS,
       {0x57, 0x10, 0x54, 0x18, 0x00, 0x10, 0x00, 0x80, 0x28, 0x12, 0x01, 0x00,
        0x07, 0x12, 0x54, 0x00, 0x03, 0x00}},
      {0,
       2,
       7,
       AGX_DIM_2D_MS_ARRAY,
       {0x57, 0x10, 0x54, 0x00, 0x00, 0x20, 0x00, 0x80, 0x28, 0x70, 0x01, 0x00,
        0x07, 0x12, 0x54, 0x00, 0x03, 0x00}},
   };

   for (auto &c : cases) {
      agx_index params = e(c.base);
      params.channels_m1 = 3;
      agx_instr *I = agx_block_image_store(
         b, params, agx_immediate(c.image), agx_null(), agx_null(),
         (enum agx_format)c.format, c.dim, false);
      EXPECT_EQ(form(I), (c.dim == AGX_DIM_2D)
                            ? AGX_APPLE9_ENC_BLOCK_IMAGE_STORE
                            : AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_EXTENDED);
      check(I, c.bytes);
   }
}

/* Vertex and instance IDs are asynchronous; the layer is a system register */
TEST_F(Apple9Pack, GraphicsSystemRegisters)
{
   agx_instr *I = agx_get_sr_to(b, r(5), AGX_SR_VERTEX_ID);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_GET_DRAW_ID);
   check(I, {0x5c, 0xdd, 0x10, 0x06});

   I = agx_get_sr_to(b, r(40), AGX_SR_INSTANCE_ID);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_GET_DRAW_ID);
   check(I, {0x8c, 0xd8, 0x90, 0x06});

   I = agx_get_sr_to(b, r(2), AGX_SR_LAYER_ID);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_GET_SR);
   check(I, {0x2c, 0xc3, 0x10, 0x06});
}

/* Normalized unpacking and packing */
TEST_F(Apple9Pack, NormalizedPacking)
{
   agx_index pair = r(2);
   pair.channels_m1 = 1;

   agx_instr *I = agx_unpack_norm_to(b, pair, r(1), AGX_FORMAT_U8NORM, 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_UNPACK_NORM);
   check(I, {0x17, 0x04, 0x54, 0x04, 0x00, 0x04, 0x06, 0xea});

   pair = r(4);
   pair.channels_m1 = 1;
   check(agx_unpack_norm_to(b, pair, r(1), AGX_FORMAT_S8NORM, 1),
         {0x17, 0x04, 0x54, 0x08, 0x00, 0x06, 0x06, 0xba});
   check(agx_unpack_norm_to(b, pair, k(r(1)), AGX_FORMAT_S16NORM, 0),
         {0x17, 0x04, 0x54, 0x08, 0x00, 0x04, 0x1c, 0xaa});

   pair = r(2);
   pair.channels_m1 = 1;
   check(agx_unpack_norm_to(b, pair, r(1), AGX_FORMAT_U16NORM, 0),
         {0x17, 0x04, 0x54, 0x04, 0x00, 0x04, 0x0e, 0xca});

   I = agx_pack_unorm_4x8_to(b, r(6), r(2), r(3), r(4), r(1));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_PACK_UNORM_4X8);
   check(I, {0x97, 0x04, 0x54, 0x0c, 0x02, 0x08, 0x18, 0x50, 0x44, 0xc2,
             0x97, 0x04, 0x54, 0x0d, 0x02, 0x10, 0x08, 0x50, 0x44, 0xc2});

   I = agx_pack_unorm_2x16_to(b, r(7), r(1), r(2));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_PACK_UNORM_2X16);
   check(I, {0x97, 0x04, 0x56, 0x0e, 0x02, 0x04, 0x10, 0x50, 0x44, 0x82});
}

/* FP16 minimum, maximum and compare-select */
TEST_F(Apple9Pack, HalfMinMaxSelect)
{
   agx_instr *I = agx_hmax_to(b, r(3), k(r(4)), k(r(3)));
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF_MINMAX);
   check(I, {0x32, 0x08, 0x3c, 0x06, 0x00, 0x00});

   check(agx_hmin_to(b, r(3), k(r(3)), k(r(4))),
         {0x32, 0x06, 0x3c, 0x08, 0x01, 0x00});

   I = agx_hcmpsel_to(b, r(10), r(5), r(2), k(r(4)), r(3), AGX_FCOND_LT);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF_COMPARE_SELECT);
   check(I, {0xa2, 0x8a, 0x07, 0x84, 0x82, 0x08, 0x03, 0x00, 0x00, 0x86});
}

/* FP16 compares into predicates of r2 and r70, kept then released */
TEST_F(Apple9Pack, HalfPredicates)
{
   agx_instr *I = agx_hcmp_pred(b, r(2), r(70), AGX_FCOND_LT, false, 1);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF_PREDICATE_SHORT);
   check(I, {0x2a, 0x04, 0x22, 0x0c, 0x03, 0x04});

   I = agx_hcmp_pred(b, k(r(2)), k(r(70)), AGX_FCOND_EQ, false, 2);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_HALF_PREDICATE_EXTENDED);
   check(I, {0x4a, 0x04, 0x3b, 0x0c, 0x06, 0x00, 0x00, 0x04, 0x00, 0x00});
}

/* Integer arithmetic into the uniform file, the STORE_UNIFORM_* forms */
TEST_F(Apple9Pack, UniformDestinations)
{
   agx_instr *I =
      agx_iadd_to(b, agx_uniform(400, AGX_SIZE_32), r(3), k(r(70)), 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_STORE_UNIFORM_ADD);
   check(I, {0x9f, 0x01, 0x54, 0x90, 0x01, 0x0c, 0x32, 0xaa, 0x15, 0x05});

   check(agx_iadd_to(b, agx_uniform(26, AGX_SIZE_32), k(r(3)),
                     agx_neg(k(r(70))), 0),
         {0x1f, 0x01, 0x54, 0x1a, 0x00, 0x0c, 0x30, 0xaa, 0x17, 0x05});

   I = agx_imad_to(b, agx_uniform(34, AGX_SIZE_32), r(3), k(r(70)), r(9), 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_STORE_UNIFORM_MAD);
   check(I, {0x9f, 0x00, 0x54, 0x22, 0x00, 0x0c, 0x32, 0x92, 0xd8, 0x25, 0x2a,
             0x00});

   I = agx_imad_to(b, agx_uniform(80, AGX_SIZE_64), agx_abs(k(r(3))),
                   agx_abs(k(r(70))), agx_zero(), 0);
   EXPECT_EQ(form(I), AGX_APPLE9_ENC_STORE_UNIFORM_MUL_WIDE);
   check(I, {0x9f, 0x00, 0x54, 0x50, 0x00, 0x0c, 0x30, 0x02, 0xe0, 0x26, 0x0a,
             0x00});

   check(
      agx_imad_to(b, agx_uniform(80, AGX_SIZE_64), r(3), r(70), agx_zero(), 0),
      {0x9f, 0x00, 0x54, 0x50, 0x00, 0x0c, 0x32, 0x06, 0xe0, 0x20, 0x1e, 0x00});
}
