/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_encoding.h"
#include "agx_apple9_machine.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* Apple9 (G16/G17) instruction packing. A released source register loses its
 * value, so only last uses (the discard flag) set the release bit.
 */

/* Largest instruction or instruction pair packed for one IR instruction */
#define A9_MAX_BYTES 24

struct a9_packed {
   uint8_t bytes[A9_MAX_BYTES];
   unsigned length;
   enum agx_apple9_encoding form;

   /* Second instruction of a pair, packed without a dependency field */
   unsigned dependency_length;
};

struct a9_state {
   /* Another device load follows in program order */
   bool load_follows;

   /* A special function's result is read by more than device stores */
   bool alu_reads_result;
};

static void
pack_assert_internal(const agx_instr *I, bool condition, const char *msg)
{
   if (!condition) {
      fprintf(stderr, "Apple9 packing assertion failed for instruction:\n\n");
      agx_print_instr(I, stderr);
      fprintf(stderr, "\n%s\n", msg);
      abort();
   }
}

#define pack_assert(I, cond) pack_assert_internal(I, cond, #cond)

static void
set_bits(uint8_t *bytes, unsigned start, unsigned width, uint64_t value)
{
   assert(width < 64 && value < (1ull << width));

   for (unsigned bit = 0; bit < width; ++bit) {
      unsigned byte = (start + bit) / 8;
      unsigned shift = (start + bit) % 8;

      bytes[byte] &= ~(1u << shift);
      bytes[byte] |= ((value >> bit) & 1) << shift;
   }
}

static void
a9_init(struct a9_packed *p, enum agx_apple9_encoding form,
        const uint8_t *bytes, unsigned length)
{
   assert(length <= A9_MAX_BYTES);
   memset(p->bytes, 0, sizeof(p->bytes));
   memcpy(p->bytes, bytes, length);
   p->length = p->dependency_length = length;
   p->form = form;
}

static bool
is_gpr(agx_index idx)
{
   return idx.type == AGX_INDEX_REGISTER && idx.reg_class == RA_GPR;
}

static bool
is_export(agx_index idx)
{
   return idx.type == AGX_INDEX_REGISTER && idx.reg_class == RA_EXPORT;
}

/* 32-bit register number of a GPR operand at least 32 bits wide */
static unsigned
gpr(const agx_instr *I, agx_index idx)
{
   pack_assert(I, is_gpr(idx));
   pack_assert(I, idx.size != AGX_SIZE_16);
   pack_assert(I, (idx.value % agx_size_align_16(idx.size)) == 0);
   pack_assert(I,
               idx.value + agx_index_size_16(idx) <= AGX_APPLE9_GPR_COUNT * 2);
   return idx.value >> 1;
}

/* 32-bit register number holding a 16-bit GPR operand */
static unsigned
gpr_of_half(const agx_instr *I, agx_index idx)
{
   pack_assert(I, is_gpr(idx) && idx.size == AGX_SIZE_16);
   pack_assert(I, idx.value < AGX_APPLE9_GPR_COUNT * 2);
   return idx.value >> 1;
}

static unsigned
export_word(const agx_instr *I, agx_index idx)
{
   pack_assert(I, is_export(idx) && idx.size == AGX_SIZE_32);
   pack_assert(I, idx.value < AGX_APPLE9_PUBLICATION_COUNT * 2);
   return idx.value >> 1;
}

/* 32-bit uniform word */
static unsigned
uniform_word(const agx_instr *I, agx_index idx)
{
   pack_assert(I, idx.type == AGX_INDEX_UNIFORM && idx.size == AGX_SIZE_32);
   pack_assert(I, (idx.value & 1) == 0);
   pack_assert(I, idx.value < AGX_APPLE9_UNIFORM_COUNT * 2);
   return idx.value >> 1;
}

static bool
retain(agx_index idx)
{
   return !idx.discard;
}

static bool
is_imm(agx_index idx)
{
   return idx.type == AGX_INDEX_IMMEDIATE;
}

static bool
is_uniform(agx_index idx)
{
   return idx.type == AGX_INDEX_UNIFORM;
}

/* Hardware slot (1-6) for an IR scoreboard slot (0-5) */
static unsigned
hw_slot(unsigned slot)
{
   assert(slot < 6);
   return slot + 1;
}

/* Completion slot fields take IR slots 0-5 (hardware slots 1-6) */
static unsigned
completion_slot(const agx_instr *I)
{
   pack_assert(I, I->scoreboard < 6);
   return I->scoreboard;
}

/* Inline float constants are eeeemmm1; exponents below 8 name a uniform */
static bool
encode_minifloat(uint32_t bits, uint8_t *encoded)
{
   uint32_t magnitude = bits & 0x7fffffffu;

   for (unsigned exponent = 8; exponent < 16; ++exponent) {
      for (unsigned mantissa = 0; mantissa < 8; ++mantissa) {
         float value = exponent == 8 ? (mantissa / 8.0f) * 0.25f
                                     : (1.0f + mantissa / 8.0f) *
                                          (1u << (exponent - 8)) * 0.125f;
         if (fui(value) == magnitude) {
            *encoded = (exponent << 4) | (mantissa << 1) | 1;
            return true;
         }
      }
   }

   return false;
}

static bool
encode_minifloat_wide(uint32_t bits, uint8_t *encoded, bool *low)
{
   *low = false;
   if (encode_minifloat(bits, encoded))
      return true;

   uint32_t magnitude = bits & 0x7fffffffu;
   for (unsigned exponent = 9; exponent < 16; ++exponent) {
      for (unsigned mantissa = 0; mantissa < 16; mantissa += 2) {
         float value =
            (1.0f + (mantissa + 1) / 16.0f) * (1u << (exponent - 8)) * 0.125f;
         if (fui(value) == magnitude) {
            *encoded = (exponent << 4) | mantissa | 1;
            *low = true;
            return true;
         }
      }
   }

   return false;
}

/* IR float immediates are AGX minifloats */
static uint32_t
imm_float_bits(agx_index idx)
{
   assert(is_imm(idx) && idx.value < 0x100);
   return fui(agx_minifloat_decode(idx.value));
}

bool
agx_apple9_float_immediate_ok(agx_index idx, bool wide)
{
   uint8_t encoded;
   bool low;

   if (!is_imm(idx) || idx.value >= 0x100 || idx.abs)
      return false;

   uint32_t bits = imm_float_bits(idx);
   return wide ? encode_minifloat_wide(bits, &encoded, &low)
               : encode_minifloat(bits, &encoded);
}

/* Integer operand: 7 value bits, retain or uniform bit 7, release, file */
static void
pack_int_source(uint8_t *bytes, unsigned value, bool uniform, bool retained,
                unsigned value_bit, unsigned discard_bit, unsigned file_bit)
{
   set_bits(bytes, value_bit, 7, value & 0x7f);
   set_bits(bytes, value_bit + 7, 1, uniform ? value >> 7 : retained);
   set_bits(bytes, discard_bit, 1, uniform || !retained);
   set_bits(bytes, file_bit, 1, !uniform);
}

/* 8-bit zero-extended constant, from one bit below the register field */
static void
pack_int_immediate(uint8_t *bytes, uint32_t value, unsigned value_bit,
                   unsigned discard_bit, unsigned file_bit,
                   unsigned descriptor_bit)
{
   assert(value <= UINT8_MAX);
   set_bits(bytes, value_bit - 1, 8, value);
   set_bits(bytes, value_bit + 7, 1, 0);
   set_bits(bytes, discard_bit, 1, 0);
   set_bits(bytes, file_bit, 1, 0);
   set_bits(bytes, descriptor_bit, 1, 0);
}

/* 16-bit integer source: a high-half bit, sign-extending unless abs */
static void
pack_int_half(uint8_t *bytes, agx_index src, unsigned hi_bit,
              unsigned width_bit, int sext_bit)
{
   set_bits(bytes, hi_bit, 1, src.value & 1);
   set_bits(bytes, width_bit, 1, 0);
   if (sext_bit >= 0)
      set_bits(bytes, sext_bit, 1, !src.abs);
}

/* Register fields of the compact float, integer min/max and logic forms */
static void
pack_compact_registers(uint8_t *bytes, unsigned dst, unsigned a, unsigned b)
{
   set_bits(bytes, 4, 4, dst & 0xf);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 44, 1, dst >> 6);
   set_bits(bytes, 9, 6, a & 0x3f);
   set_bits(bytes, 40, 1, a >> 6);
   set_bits(bytes, 25, 6, b & 0x3f);
   set_bits(bytes, 42, 1, b >> 6);
}

/* Compact uniform sources, packed last as they overlap the release bits */
static void
pack_compact_uniform(uint8_t *bytes, unsigned s, unsigned u)
{
   bytes[1 + 2 * s] = ((u & 63) << 1) | 1;
   set_bits(bytes, s ? 41 : 39, 1, 1);
   set_bits(bytes, s ? 20 : 19, 1, (u >> 6) & 1);
   set_bits(bytes, s ? 42 : 40, 1, u >> 7);
}

/* A retained compact operand sets its cache bit and clears its release bit */
static void
pack_compact_retain(uint8_t *bytes, bool retain_a, bool retain_b)
{
   if (retain_a) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }

   if (retain_b) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
}

/* Floating-point add and multiply, in the smallest form that encodes them */
static void
pack_float2(const agx_instr *I, struct a9_packed *p)
{
   bool mul = I->op == AGX_OPCODE_FMUL;
   agx_index a = I->src[0], b = I->src[1];

   pack_assert(I, I->dest[0].size == AGX_SIZE_32);

   /* Constants are always operand B */
   if (is_imm(a)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   if (is_imm(b)) {
      pack_assert(I, is_gpr(a) && !a.abs && a.size == AGX_SIZE_32);
      pack_assert(I, !b.abs);

      uint32_t bits = imm_float_bits(b) ^ (b.neg ? 0x80000000u : 0);
      uint8_t encoded;
      bool low;
      pack_assert(I, encode_minifloat_wide(bits, &encoded, &low));

      bool extended = I->saturate;
      uint8_t bytes[8] = {0x09, encoded, 0x34, 0x01, 0x80, 0x00};
      if (mul)
         set_bits(bytes, 16, 3, 5);

      unsigned dst = gpr(I, I->dest[0]), src = gpr(I, a);
      set_bits(bytes, 19, 1, bits >> 31);
      set_bits(bytes, 35, 1, low);
      set_bits(bytes, 4, 4, dst & 15);
      set_bits(bytes, 22, 2, (dst >> 4) & 3);
      set_bits(bytes, 44, 1, dst >> 6);
      set_bits(bytes, 25, 6, src & 63);
      set_bits(bytes, 42, 1, src >> 6);
      set_bits(bytes, 43, 1, a.neg);
      pack_compact_retain(bytes, false, retain(a));

      if (extended) {
         bytes[4] |= 1;
         set_bits(bytes, 57, 1, 1);
      }

      a9_init(p,
              extended ? AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED
                       : AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT,
              bytes, extended ? 8 : 6);
      return;
   }

   /* The uniform may be A or B; only B can be negated in the compact form */
   if (is_uniform(a) && !is_uniform(b) && !a.neg && !a.abs && !b.neg &&
       !b.abs) {
      agx_index t = a;
      a = b;
      b = t;
   }

   bool neg_a = a.neg, neg_b = b.neg;
   bool any_abs = a.abs || b.abs;
   bool hi = (is_gpr(a) && a.size == AGX_SIZE_16 && (a.value & 1)) ||
             (is_gpr(b) && b.size == AGX_SIZE_16 && (b.value & 1));
   bool uniforms = is_uniform(a) || is_uniform(b);
   unsigned dst = gpr(I, I->dest[0]);

   enum agx_apple9_encoding form;
   unsigned length;
   if (any_abs) {
      form = mul ? AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED
                 : AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED;
      length = mul ? 12 : 10;
   } else if (neg_a || I->saturate) {
      form = AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED;
      length = 8;
   } else if (!neg_b && !hi && !uniforms && !I->wait && dst < 64) {
      form = AGX_APPLE9_ENC_FLOAT2_BASE;
      length = 4;
   } else {
      form = AGX_APPLE9_ENC_FLOAT2_COMPACT;
      length = 6;
   }

   uint8_t bytes[12] = {0x09, 0x05, 0x1c, 0x01, 0x00, 0x00};
   set_bits(bytes, 16, 3, mul ? 5 : 4);
   set_bits(bytes, 43, 1, neg_b);
   set_bits(bytes, 21, 1, 1);
   pack_compact_retain(bytes, is_gpr(a) && retain(a), is_gpr(b) && retain(b));
   pack_compact_registers(
      bytes, dst,
      is_gpr(a) ? (a.size == AGX_SIZE_16 ? gpr_of_half(I, a) : gpr(I, a)) : 0,
      is_gpr(b) ? (b.size == AGX_SIZE_16 ? gpr_of_half(I, b) : gpr(I, b)) : 0);

   if (is_uniform(a))
      pack_compact_uniform(bytes, 0, uniform_word(I, a));
   if (is_uniform(b))
      pack_compact_uniform(bytes, 1, uniform_word(I, b));

   /* FP16 sources convert on read; bits 35 and 36 select the high half */
   agx_index srcs[2] = {a, b};
   for (unsigned s = 0; s < 2; ++s) {
      if (!is_gpr(srcs[s]) || srcs[s].size != AGX_SIZE_16)
         continue;

      set_bits(bytes, s ? 24 : 8, 1, 0);
      set_bits(bytes, 35 + s, 1, srcs[s].value & 1);
   }

   if (form == AGX_APPLE9_ENC_FLOAT2_BASE)
      set_bits(bytes, 18, 1, 0);

   if (length >= 8) {
      set_bits(bytes, 32, 2,
               form == AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED ? 1 : 2);
      set_bits(bytes, 49, 1, neg_a);
      if (any_abs)
         set_bits(bytes, 64, 2, a.abs | (b.abs << 1));
   }

   if (I->saturate)
      set_bits(bytes, 57, 1, 1);

   a9_init(p, form, bytes, length);
}

/* fmul into an export register, signalling the slot in bits 58-60 */
static void
pack_float2_export(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b) && !a.abs && !a.neg && !b.abs);

   uint8_t bytes[8] = {0x09, 0x05, 0x1c, 0x01, 0x41, 0x00};
   set_bits(bytes, 16, 3, 5);
   set_bits(bytes, 43, 1, b.neg);
   set_bits(bytes, 21, 1, 1);
   pack_compact_retain(bytes, retain(a), retain(b));
   pack_compact_registers(bytes, export_word(I, I->dest[0]), gpr(I, a),
                          gpr(I, b));
   set_bits(bytes, 58, 3, completion_slot(I));
   a9_init(p, AGX_APPLE9_ENC_FLOAT2_EXPORT, bytes, 8);
}

/* fmul by 1/W, naming the coefficient in byte 5 */
static void
pack_float2_perspective(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b));
   pack_assert(I, !a.abs && !a.neg && !b.abs && !b.neg && !I->saturate);
   pack_assert(I, I->component < AGX_APPLE9_NUM_COEFFICIENTS);

   unsigned dst = gpr(I, I->dest[0]), ra = gpr(I, a), rb = gpr(I, b);
   pack_assert(I, dst < 64 && ra < 64 && rb < 64);

   uint8_t bytes[8] = {0x09, 0x05, 0x1c, 0x01, 0x00, 0x00};
   set_bits(bytes, 16, 3, 7);
   set_bits(bytes, 21, 1, 1);
   pack_compact_retain(bytes, retain(a), retain(b));
   pack_compact_registers(bytes, dst, ra, rb);
   bytes[5] = I->component;
   a9_init(p, AGX_APPLE9_ENC_FLOAT2_PROJECT, bytes, 8);
}

/* FP32/FP16 conversion as fadd x, -0.0; narrowing writes one dest half */
static void
pack_float_convert(const agx_instr *I, struct a9_packed *p)
{
   agx_index dst = I->dest[0], src = I->src[0];
   pack_assert(I, !src.abs && !src.neg && !I->saturate);

   /* f32_to_f16 and f16_to_f32 name the register whose low half is FP16 */
   bool widen = I->op == AGX_OPCODE_F16_TO_F32 ||
                (I->op == AGX_OPCODE_FADD && dst.size == AGX_SIZE_32);
   bool whole =
      I->op == AGX_OPCODE_F32_TO_F16 || I->op == AGX_OPCODE_F16_TO_F32;

   unsigned d = widen || whole ? gpr(I, dst) : gpr_of_half(I, dst);
   unsigned s = !widen || whole ? gpr(I, src) : gpr_of_half(I, src);

   uint8_t bytes[6] = {0x01, 0x01, 0x1c, 0x81, 0x00, 0x02};
   set_bits(bytes, 4, 4, d & 15);
   set_bits(bytes, 22, 2, (d >> 4) & 3);
   set_bits(bytes, 44, 1, d >> 6);
   set_bits(bytes, 9, 6, s & 63);
   set_bits(bytes, 40, 1, s >> 6);

   if (widen) {
      set_bits(bytes, 0, 4, 9);
      set_bits(bytes, 8, 1, 0);
      set_bits(bytes, 35, 1, !whole && (src.value & 1));
   } else {
      set_bits(bytes, 34, 1, !whole && (dst.value & 1));
   }

   pack_compact_retain(bytes, retain(src), false);
   a9_init(
      p,
      widen ? AGX_APPLE9_ENC_HALF_TO_FLOAT : AGX_APPLE9_ENC_FLOAT_TO_HALF,
      bytes, 6);
}

/* Two FP32 to FP16 conversions into the low and high destination halves */
static void
pack_half_pair(const agx_instr *I, struct a9_packed *p)
{
   unsigned d = gpr(I, I->dest[0]);
   uint8_t bytes[12];

   for (unsigned h = 0; h < 2; ++h) {
      agx_index src = I->src[h];
      pack_assert(I, !src.abs && !src.neg);

      unsigned s = gpr(I, src);
      pack_assert(I, s != d);

      uint8_t *part = bytes + 6 * h;
      memcpy(part, (const uint8_t[]){0x01, 0x01, 0x1c, 0x81, 0x00, 0x02}, 6);
      set_bits(part, 4, 4, d & 15);
      set_bits(part, 22, 2, (d >> 4) & 3);
      set_bits(part, 44, 1, d >> 6);
      set_bits(part, 9, 6, s & 63);
      set_bits(part, 40, 1, s >> 6);
      set_bits(part, 34, 1, h);

      bool same = agx_is_equiv(I->src[0], I->src[1]);
      pack_compact_retain(part, retain(src) || (h == 0 && same), false);
   }

   a9_init(p, AGX_APPLE9_ENC_PACK_HALF_2X16, bytes, 12);
   p->dependency_length = 0;
}

/* FP16 add, multiply and FMA on low halves, leaving the high half undefined */
static void
pack_half_alu(const agx_instr *I, struct a9_packed *p)
{
   bool fma = I->op == AGX_OPCODE_HFMA;
   unsigned count = fma ? 3 : 2;

   pack_assert(I, !I->saturate);

   uint8_t bytes[8] = {
      0,
      0,
      fma ? 0x1e : I->op == AGX_OPCODE_HMUL ? 0x1d : 0x1c,
      0,
      fma ? 0x81 : 0,
      0,
      0,
      0,
   };

   unsigned dst = gpr(I, I->dest[0]);
   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, fma ? 60 : 44, 1, dst >> 6);

   for (unsigned s = 0; s < count; ++s) {
      agx_index src = I->src[s];
      unsigned reg = gpr(I, src);
      pack_assert(
         I, !src.abs && (!src.neg || (I->op == AGX_OPCODE_HADD && s == 1)));

      set_bits(bytes, 9 + 16 * s, 6, reg & 63);
      set_bits(bytes, s == 2 ? 38 : (fma ? 56 : 40) + 2 * s, 1, reg >> 6);

      if (retain(src)) {
         set_bits(bytes, 15 + 16 * s, 1, 1);
         set_bits(bytes, s == 2 ? 39 : 19 + s, 1, 0);
      }
   }

   if (I->op == AGX_OPCODE_HADD && I->src[1].neg)
      set_bits(bytes, 43, 1, 1);

   a9_init(p, fma ? AGX_APPLE9_ENC_HALF3 : AGX_APPLE9_ENC_HALF2, bytes,
           fma ? 8 : 6);
}

/* Fused multiply-add */
static void
pack_float3(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1], c = I->src[2];

   /* Operand A is a register or uniform; a constant multiplier goes in B */
   if (is_imm(a)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   pack_assert(I, !is_imm(a) && I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, a.size == AGX_SIZE_32);

   unsigned dst = gpr(I, I->dest[0]);
   bool inline_b = !is_gpr(b), inline_c = !is_gpr(c);
   unsigned ra = is_gpr(a) ? gpr(I, a) : 0;
   unsigned rb =
      !inline_b ? (b.size == AGX_SIZE_16 ? gpr_of_half(I, b) : gpr(I, b)) : 0;
   unsigned rc =
      !inline_c ? (c.size == AGX_SIZE_16 ? gpr_of_half(I, c) : gpr(I, c)) : 0;
   bool half = (is_gpr(b) && b.size == AGX_SIZE_16) ||
               (is_gpr(c) && c.size == AGX_SIZE_16);
   bool neg_a = a.neg, neg_b = b.neg, neg_c = c.neg;

   /* The constant's own sign folds into its negation */
   if (is_imm(b) && !b.abs)
      neg_b ^= (b.value & 0x80) != 0;
   if (is_imm(c) && !c.abs)
      neg_c ^= (c.value & 0x80) != 0;

   bool compact = !I->wait && !I->saturate && !a.abs && !b.abs && !c.abs &&
                  !neg_a && !neg_b && !neg_c && !half && is_gpr(a) &&
                  is_gpr(c) && dst < 64 && ra < 64 && rc < 64 &&
                  (is_gpr(b) ? rb < 64 : true) &&
                  (is_uniform(b) ? uniform_word(I, b) < 64 : true);

   enum agx_apple9_encoding form;
   unsigned length;
   if (compact) {
      form = AGX_APPLE9_ENC_FLOAT3_COMPACT;
      length = 6;
   } else if (a.abs || b.abs) {
      form = AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED;
      length = 12;
   } else if (I->saturate) {
      form = AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED;
      length = 10;
   } else {
      form = AGX_APPLE9_ENC_FLOAT3_EXTENDED;
      length = 8;
   }

   uint8_t bytes[12] = {0x09, 0x01, 0x1e, 0x05, 0x81, 0x08, 0x02, 0x00};
   set_bits(bytes, 21, 1, 1);
   if (is_gpr(a) && retain(a)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (is_gpr(b) && retain(b)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
   if (is_gpr(c) && retain(c)) {
      set_bits(bytes, 47, 1, 1);
      set_bits(bytes, 39, 1, 0);
   }

   set_bits(bytes, 4, 4, dst & 0xf);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 60, 1, dst >> 6);
   set_bits(bytes, 9, 6, ra & 0x3f);
   set_bits(bytes, 56, 1, ra >> 6);
   set_bits(bytes, 25, 6, rb & 0x3f);
   set_bits(bytes, 58, 1, rb >> 6);
   set_bits(bytes, 41, 6, rc & 0x3f);
   set_bits(bytes, 38, 1, rc >> 6);

   if (is_uniform(a)) {
      unsigned u = uniform_word(I, a);
      bytes[1] = ((u & 63) << 1) | 1;
      set_bits(bytes, 55, 1, 1);
      set_bits(bytes, 19, 1, (u >> 6) & 1);
      set_bits(bytes, 56, 1, u >> 7);
   }

   if (inline_b) {
      uint8_t encoded;
      unsigned u = 0;
      if (is_imm(b)) {
         pack_assert(I, encode_minifloat(imm_float_bits(b), &encoded));
      } else {
         u = uniform_word(I, b);
         encoded = ((u & 63) << 1) | 1;
      }

      bytes[3] = encoded;
      set_bits(bytes, 57, 1, 1);
      set_bits(bytes, 20, 1, (u >> 6) & 1);
      set_bits(bytes, 58, 1, u >> 7);
   }

   if (inline_c) {
      uint8_t encoded;
      bool low = false;
      unsigned u = 0;
      if (is_imm(c)) {
         pack_assert(I,
                     encode_minifloat_wide(imm_float_bits(c), &encoded, &low));
      } else {
         u = uniform_word(I, c);
         encoded = ((u & 63) << 1) | 1;
      }

      /* The addend's fourth mantissa bit replaces the byte's low bit */
      bytes[5] = (encoded & ~1) | low;
      set_bits(bytes, 37, 1, 1);
      set_bits(bytes, 39, 1, (u >> 6) & 1);
      set_bits(bytes, 38, 1, u >> 7);
   }

   set_bits(bytes, 35, 1, c.abs);
   set_bits(bytes, 36, 1, neg_c);
   set_bits(bytes, 59, 1, neg_a ^ neg_b);

   /* FP16 sources: operand B in its low half only, the addend in either */
   if (is_gpr(b) && b.size == AGX_SIZE_16) {
      pack_assert(I, !(b.value & 1));
      set_bits(bytes, 24, 1, 0);
   }

   if (is_gpr(c) && c.size == AGX_SIZE_16) {
      set_bits(bytes, 49, 1, 0);
      set_bits(bytes, 40, 1, c.value & 1);
   }

   if (compact) {
      /* B's alternative file selector moves from bit 57 to bit 37 */
      set_bits(bytes, 32, 2, 0);
      set_bits(bytes, 37, 1, inline_b);
      set_bits(bytes, 57, 1, 0);
   } else if (form == AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED) {
      set_bits(bytes, 32, 2, 3);
      set_bits(bytes, 80, 2, a.abs | (b.abs << 1));
   } else if (form == AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED) {
      set_bits(bytes, 32, 2, 2);
   }

   if (I->saturate)
      set_bits(bytes, 73, 1, 1);

   a9_init(p, form, bytes, length);
}

/* Bit 33 clear writes an 8-bit uniform word, even for a 64-bit product */
static unsigned
uniform_destination(const agx_instr *I, uint8_t *bytes)
{
   agx_index dst = I->dest[0];
   pack_assert(I, dst.type == AGX_INDEX_UNIFORM && !(dst.value & 1));
   pack_assert(
      I, dst.value + agx_index_size_16(dst) <= AGX_APPLE9_UNIFORM_COUNT * 2);
   pack_assert(I, dst.size != AGX_SIZE_64 || !(dst.value & 3));

   set_bits(bytes, 33, 1, 0);
   set_bits(bytes, 25, 8, dst.value >> 1);

   for (unsigned s = 0; s < I->nr_srcs; ++s)
      pack_assert(I, is_gpr(I->src[s]) || agx_is_equiv(I->src[s], agx_zero()));

   return dst.value >> 1;
}

/* Integer add. A negated B subtracts; a negated A swaps the operands */
static void
pack_iadd(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && !I->saturate && !I->shift);
   pack_assert(I, !(a.neg && b.neg));

   if (a.neg) {
      agx_index t = a;
      a = b;
      b = t;
   }

   bool sub = b.neg;
   bool uniform = I->dest[0].type == AGX_INDEX_UNIFORM;
   uint8_t bytes[10] = {
      sub ? 0x1f : 0x9f, 0x01, 0x54, 0x00, 0x02, 0x00, 0x00, 0xa8, 0x17, 0x05};

   if (uniform)
      uniform_destination(I, bytes);
   else
      set_bits(bytes, 25, 7, gpr(I, I->dest[0]));

   agx_index srcs[2] = {a, b};
   for (unsigned s = 0; s < 2; ++s) {
      agx_index src = srcs[s];

      if (is_imm(src)) {
         pack_int_immediate(bytes, src.value, 42 + 9 * s, 65 + s, 72 + 2 * s,
                            61 + 2 * s);
      } else if (is_uniform(src)) {
         pack_int_source(bytes, uniform_word(I, src), true, false, 42 + 9 * s,
                         65 + s, 72 + 2 * s);
      } else if (src.size == AGX_SIZE_16) {
         pack_int_source(bytes, gpr_of_half(I, src), false, retain(src),
                         42 + 9 * s, 65 + s, 72 + 2 * s);
         pack_int_half(bytes, src, 41 + 9 * s, 61 + 2 * s, 73 + 2 * s);
      } else {
         pack_int_source(bytes, gpr(I, src), false, retain(src), 42 + 9 * s,
                         65 + s, 72 + 2 * s);
      }
   }

   a9_init(p,
           uniform ? AGX_APPLE9_ENC_STORE_UNIFORM_ADD
                   : AGX_APPLE9_ENC_INT_ADD_EXTENDED,
           bytes, 10);
}

/* 32x32 to 64-bit multiply; abs zero-extends both factors */
static void
pack_imul_wide(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b) && a.abs == b.abs);
   pack_assert(I, agx_is_equiv(I->src[2], agx_zero()) && !I->shift);

   uint8_t bytes[12] = {0x9f, 0x00, 0x54, 0x00, 0x02, 0x00,
                        0x00, 0x00, 0xe0, 0x26, 0x0a, 0x00};
   if (!a.abs)
      bytes[10] = 0x1e;

   bool uniform = I->dest[0].type == AGX_INDEX_UNIFORM;
   if (uniform)
      uniform_destination(I, bytes);
   else
      set_bits(bytes, 25, 7, gpr(I, I->dest[0]));

   set_bits(bytes, 42, 7, gpr(I, a));
   set_bits(bytes, 51, 7, gpr(I, b));
   if (retain(a)) {
      set_bits(bytes, 49, 1, 1);
      set_bits(bytes, 73, 1, 0);
   }
   if (retain(b)) {
      set_bits(bytes, 58, 1, 1);
      set_bits(bytes, 74, 1, 0);
   }

   a9_init(p,
           uniform ? AGX_APPLE9_ENC_STORE_UNIFORM_MUL_WIDE
                   : AGX_APPLE9_ENC_INT_MUL_WIDE,
           bytes, 12);
}

/* Integer multiply-add. Constants go in factor B and the addend */
static void
pack_imad(const agx_instr *I, struct a9_packed *p)
{
   if (I->dest[0].size == AGX_SIZE_64) {
      pack_imul_wide(I, p);
      return;
   }

   agx_index srcs[3] = {I->src[0], I->src[1], I->src[2]};
   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && !I->shift);

   if (is_imm(srcs[0])) {
      agx_index t = srcs[0];
      srcs[0] = srcs[1];
      srcs[1] = t;
   }

   pack_assert(I, !is_imm(srcs[0]));

   uint8_t bytes[12] = {0x9f, 0x00, 0x54, 0x00, 0x02, 0x00,
                        0x00, 0x00, 0xd0, 0x2f, 0x2a, 0x00};
   bool uniform = I->dest[0].type == AGX_INDEX_UNIFORM;

   if (uniform)
      uniform_destination(I, bytes);
   else
      set_bits(bytes, 25, 7, gpr(I, I->dest[0]));

   for (unsigned s = 0; s < 3; ++s) {
      agx_index src = srcs[s];
      pack_assert(I, !src.neg);

      if (is_imm(src)) {
         pack_int_immediate(bytes, src.value, 42 + 9 * s, 73 + s, 81 + 2 * s,
                            70 + s);
      } else if (is_uniform(src)) {
         pack_int_source(bytes, uniform_word(I, src), true, false, 42 + 9 * s,
                         73 + s, 81 + 2 * s);
      } else if (src.size == AGX_SIZE_16) {
         pack_assert(I, s == 0 || src.abs);
         pack_int_source(bytes, gpr_of_half(I, src), false, retain(src),
                         42 + 9 * s, 73 + s, 81 + 2 * s);
         pack_int_half(bytes, src, 41 + 9 * s, 70 + s, s == 0 ? 82 : -1);
      } else {
         pack_int_source(bytes, gpr(I, src), false, retain(src), 42 + 9 * s,
                         73 + s, 81 + 2 * s);
      }
   }

   a9_init(p,
           uniform ? AGX_APPLE9_ENC_STORE_UNIFORM_MAD
                   : AGX_APPLE9_ENC_INT_MAD_EXTENDED,
           bytes, 12);
}

/* ishl is bfi into zero, ushr bfeil from zero, by a constant or register */
static void
pack_shift(const agx_instr *I, struct a9_packed *p)
{
   bool arithmetic = I->op == AGX_OPCODE_ASR;
   bool left = I->op == AGX_OPCODE_BFI;
   agx_index src = arithmetic ? I->src[0] : I->src[1];
   agx_index amount = arithmetic ? I->src[1] : I->src[2];
   bool variable = !is_imm(amount);
   unsigned width = arithmetic ? 0 : I->bfi_mask;

   if (!arithmetic)
      pack_assert(I, agx_is_equiv(I->src[0], agx_zero()));

   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && src.size == AGX_SIZE_32);
   pack_assert(I, !left || width == 0);
   pack_assert(I, !variable || width == 0);
   pack_assert(I, variable || amount.value < 32);
   pack_assert(I, width < 64);

   uint8_t bytes[12] = {0xa7, 0, 0x54, 0, 0x02, 0, 0, 0, 0xf0, 0x11, 0x01, 0};
   if (arithmetic) {
      bytes[1] = 1;
      bytes[7] = variable ? 0xd8 : 0x78;
      bytes[8] = variable ? 0xe2 : 0x62;
      bytes[9] = 0;
   } else {
      if (left)
         bytes[0] = 0x27;

      if (variable) {
         bytes[9] = 0x13;
         bytes[10] = 0x05;
      }
   }

   set_bits(bytes, 25, 7, gpr(I, I->dest[0]));

   unsigned release_bit = arithmetic ? 62 : 72;
   unsigned file_bit = arithmetic ? 69 : 80;

   agx_index srcs[2] = {src, amount};
   for (unsigned s = 0; s < (variable ? 2 : 1); ++s) {
      if (is_uniform(srcs[s])) {
         pack_int_source(bytes, uniform_word(I, srcs[s]), true, false,
                         s ? 51 : 42, release_bit + s, file_bit + 2 * s);
      } else {
         set_bits(bytes, s ? 51 : 42, 7, gpr(I, srcs[s]));
         if (retain(srcs[s]))
            set_bits(bytes, release_bit + s, 1, 0);
      }
   }

   if (!variable)
      set_bits(bytes, 50, 5, amount.value);

   if (!arithmetic && !variable)
      set_bits(bytes, 84, 6, width);

   enum agx_apple9_encoding form =
      arithmetic ? (variable ? AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER
                             : AGX_APPLE9_ENC_SHIFT_EXTENDED)
                 : (variable ? AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER
                             : AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE);
   a9_init(p, form, bytes, arithmetic ? 10 : 12);
}

/* Integer and float conversions between 32-bit types */
static void
pack_convert(const agx_instr *I, struct a9_packed *p)
{
   enum agx_convert mode = I->src[0].value;
   agx_index src = I->src[1];
   unsigned dst = gpr(I, I->dest[0]);
   unsigned value = is_uniform(src) ? uniform_word(I, src) : gpr(I, src);

   pack_assert(I, is_imm(I->src[0]));

   switch (mode) {
   case AGX_CONVERT_U32_TO_F:
   case AGX_CONVERT_S32_TO_F: {
      bool sign = mode == AGX_CONVERT_S32_TO_F;
      pack_assert(I, I->round == AGX_ROUND_RTE);

      /* Byte 6 holds the source lifetime, byte 7 bit 6 the signedness */
      uint8_t bytes[8] = {
         0xa7,
         0x07,
         0x54,
         dst << 1,
         0x03,
         (value & 63) << 2,
         (retain(src) ? 0x8c : 0xac) | ((value >> 6) & 1),
         sign ? 0x60 : 0x20,
      };

      if (is_uniform(src))
         pack_int_source(bytes, value, true, false, 42, 53, 61);

      a9_init(
         p, sign ? AGX_APPLE9_ENC_SINT_TO_FLOAT : AGX_APPLE9_ENC_UINT_TO_FLOAT,
         bytes, 8);
      return;
   }

   case AGX_CONVERT_F_TO_U32:
   case AGX_CONVERT_F_TO_S32: {
      bool sign = mode == AGX_CONVERT_F_TO_S32;
      pack_assert(I, I->round == AGX_ROUND_RTZ);

      /* The signed format with an unsigned conversion saturates at 65535 */
      uint8_t bytes[10] = {
         0x27,
         0x07,
         0x54,
         dst << 1,
         0x03,
         (value & 63) << 2,
         (retain(src) ? 0x96 : 0xb4) | ((value >> 6) & 1),
         sign ? 0x48 : 0x08,
         sign ? 0x03 : 0x02,
         0x00,
      };

      if (is_uniform(src))
         pack_int_source(bytes, value, true, false, 42, 53, 59);

      a9_init(
         p, sign ? AGX_APPLE9_ENC_FLOAT_TO_SINT : AGX_APPLE9_ENC_FLOAT_TO_UINT,
         bytes, 10);
      return;
   }

   default:
      pack_assert(I, !"unsupported conversion");
   }
}

/* Population count, find MSB and bit reverse */
static void
pack_bit_unary(const agx_instr *I, struct a9_packed *p)
{
   agx_index src = I->src[0];
   unsigned dst = gpr(I, I->dest[0]);
   unsigned value = is_uniform(src) ? uniform_word(I, src) : gpr(I, src);
   bool count = I->op == AGX_OPCODE_POPCOUNT;

   uint8_t bytes[8] = {
      count ? 0x27 : 0xa7,
      I->op == AGX_OPCODE_BITREV ? 0x04 : 0x05,
      0x54,
      dst << 1,
      0x02,
      (value & 63) << 2,
      (retain(src) ? 0x4e : 0x5c) | ((value >> 6) & 1),
      0x04,
   };

   if (is_uniform(src)) {
      /* Measured for popcount, which then uses another destination format */
      if (count) {
         bytes[4] = 0x03;
         bytes[6] = 0xcc;
      }

      pack_int_source(bytes, value, true, false, 42, 52, 58);
   }

   a9_init(p, AGX_APPLE9_ENC_BIT_UNARY, bytes, 8);
}

/* FP32 special functions. Byte 4 is 3 if an ALU reads the result, else 2 */
static void
pack_special(const agx_instr *I, const struct a9_state *st, struct a9_packed *p)
{
   unsigned family = 0, function = 0, precision = 0x40, rounding = 0;
   bool reciprocal = false;

   switch (I->op) {
   case AGX_OPCODE_RCP:
      family = 1;
      precision = 0x48;
      rounding = 0x20;
      reciprocal = true;
      break;
   case AGX_OPCODE_RSQRT:
      family = 1;
      function = 1;
      break;
   case AGX_OPCODE_SRSQRT:
      function = 1;
      break;
   case AGX_OPCODE_SIN_FACTOR:
      function = 3;
      break;
   case AGX_OPCODE_EXP2:
      family = 1;
      function = 2;
      break;
   case AGX_OPCODE_LOG2:
      function = 2;
      break;
   case AGX_OPCODE_FLOOR:
      rounding = 2;
      break;
   case AGX_OPCODE_CEIL:
      rounding = 4;
      break;
   case AGX_OPCODE_TRUNC:
      rounding = 6;
      break;
   case AGX_OPCODE_ROUNDEVEN:
      break;
   default:
      UNREACHABLE("not a special function");
   }

   agx_index src = I->src[0];
   unsigned dst = gpr(I, I->dest[0]);
   unsigned value = is_uniform(src) ? uniform_word(I, src) : gpr(I, src);

   uint8_t bytes[10] = {
      0x2f | (family << 7),
      function,
      0x54,
      dst << 1,
      st->alu_reads_result ? 0x03 : 0x02,
      (value & 63) << 2,
      ((value >> 6) & 1) | (reciprocal ? (retain(src) ? 0x00 : 0x10)
                                       : (retain(src) ? 0x90 : 0xb0)),
      precision,
      rounding,
      0,
   };

   if (is_uniform(src))
      pack_int_source(bytes, value, true, false, 42, 53, 62);

   if (I->saturate)
      set_bits(bytes, reciprocal ? 65 : 58, 1, 1);

   set_bits(bytes, reciprocal ? 70 : 63, 1, src.abs);
   set_bits(bytes, reciprocal ? 71 : 64, 1, src.neg);

   a9_init(p,
           is_uniform(src) ? AGX_APPLE9_ENC_FLOAT_SPECIAL_UNIFORM
                           : AGX_APPLE9_ENC_FLOAT_SPECIAL,
           bytes, 10);
}

static void
pack_derivative(const agx_instr *I, struct a9_packed *p)
{
   agx_index src = I->src[0];
   unsigned dst = gpr(I, I->dest[0]), s = gpr(I, src);
   pack_assert(I, !src.abs && !src.neg);

   uint8_t bytes[10] = {
      0x37,
      0x05 | ((I->op == AGX_OPCODE_DFDY) << 1),
      0x54,
      dst << 1,
      0x03,
      (s & 63) << 2,
      0x90 | (retain(src) ? 2 : 0) | (s >> 6),
      0x40 | (I->saturate ? 4 : 0),
      0,
      0,
   };
   a9_init(p, AGX_APPLE9_ENC_DERIVATIVE, bytes, 10);
}

/* Constant compare operand B, in the byte that otherwise names a uniform */
static void
pack_compare_literal(const agx_instr *I, uint8_t *bytes, agx_index src,
                     bool is_float, unsigned cond_bit, unsigned file_bit)
{
   if (is_float) {
      uint32_t bits = imm_float_bits(src);
      uint8_t encoded;
      bool low;
      pack_assert(I, encode_minifloat_wide(bits, &encoded, &low));

      bytes[3] = encoded;
      set_bits(bytes, 20, 1, bits >> 31);
      set_bits(bytes, cond_bit, 1, low);
   } else {
      pack_assert(I, src.value <= UINT8_MAX);
      bytes[3] = 0x80 | (((src.value >> 1) & 0x3f) << 1);
      set_bits(bytes, 20, 1, src.value >> 7);
      set_bits(bytes, cond_bit, 1, src.value & 1);
   }

   set_bits(bytes, file_bit, 1, 1);
   set_bits(bytes, file_bit + 1, 1, 0);
}

/* Select conditions, with bit 8 choosing the equality form */
#define A9_SELECT_FEQ      0x00
#define A9_SELECT_FGT      0x02
#define A9_SELECT_FLT      0x03
#define A9_SELECT_UGT      0x04
#define A9_SELECT_ULT      0x05
#define A9_SELECT_IGT      0x06
#define A9_SELECT_ILT      0x07
#define A9_SELECT_EQUALITY 0x100

/* Select condition of icmpsel/fcmpsel and operand swap, if encodable */
static bool
select_condition(const agx_instr *I, unsigned *cond, bool *swap)
{
   *swap = false;

   if (I->op == AGX_OPCODE_ICMPSEL) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         *cond = A9_SELECT_ILT | A9_SELECT_EQUALITY;
         return true;
      case AGX_ICOND_ULT:
         *cond = A9_SELECT_ULT;
         return true;
      case AGX_ICOND_UGT:
         *cond = A9_SELECT_UGT;
         return true;
      case AGX_ICOND_SLT:
         *cond = A9_SELECT_ILT;
         return true;
      case AGX_ICOND_SGT:
         *cond = A9_SELECT_IGT;
         return true;
      default:
         return false;
      }
   } else {
      switch (I->fcond) {
      case AGX_FCOND_EQ:
         *cond = A9_SELECT_FEQ | A9_SELECT_EQUALITY;
         return true;
      case AGX_FCOND_LT:
         *cond = A9_SELECT_FLT;
         return true;
      case AGX_FCOND_GT:
         *cond = A9_SELECT_FGT;
         return true;
      case AGX_FCOND_GE:
         *cond = A9_SELECT_FGT | A9_SELECT_EQUALITY;
         return true;
      case AGX_FCOND_LE:
         *cond = A9_SELECT_FGT | A9_SELECT_EQUALITY;
         *swap = true;
         return true;
      default:
         return false;
      }
   }
}

/* min and max: a select between the compared operands themselves */
static int
minmax_select(const agx_instr *I)
{
   if (I->dest[0].size != AGX_SIZE_32 || I->op == AGX_OPCODE_HCMPSEL)
      return -1;

   bool same =
      agx_is_equiv(I->src[0], I->src[2]) && agx_is_equiv(I->src[1], I->src[3]);
   bool swapped =
      agx_is_equiv(I->src[0], I->src[3]) && agx_is_equiv(I->src[1], I->src[2]);

   if (!same && !swapped)
      return -1;

   if (I->op == AGX_OPCODE_ICMPSEL) {
      for (unsigned s = 0; s < 4; ++s) {
         /* Only the float form takes a constant */
         if (I->src[s].neg || I->src[s].abs || is_imm(I->src[s]) ||
             I->src[s].size != AGX_SIZE_32)
            return -1;
      }

      switch (I->icond) {
      case AGX_ICOND_UGT:
         return same ? 4 : 5;
      case AGX_ICOND_ULT:
         return same ? 5 : 4;
      case AGX_ICOND_SGT:
         return same ? 6 : 7;
      case AGX_ICOND_SLT:
         return same ? 7 : 6;
      default:
         return -1;
      }
   } else {
      /* Modifiers must match between the compared and selected operands */
      if (!agx_is_equiv_with_modifiers(I->src[0], I->src[same ? 2 : 3]) ||
          !agx_is_equiv_with_modifiers(I->src[1], I->src[same ? 3 : 2]) ||
          I->src[0].neg || I->src[0].abs || I->src[1].abs)
         return -1;

      for (unsigned s = 0; s < 2; ++s) {
         if (!is_imm(I->src[s]) && I->src[s].size != AGX_SIZE_32)
            return -1;
      }

      switch (I->fcond) {
      case AGX_FCOND_GTN:
         return same ? 0 : 1;
      case AGX_FCOND_LTN:
         return same ? 1 : 0;
      default:
         return -1;
      }
   }
}

static void
pack_minmax(const agx_instr *I, unsigned select, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   bool half = I->op == AGX_OPCODE_HMIN || I->op == AGX_OPCODE_HMAX;
   bool floating = I->op == AGX_OPCODE_FCMPSEL || half;

   /* Operand B is the one that can be a constant */
   if (is_imm(a) || (is_uniform(a) && !is_uniform(b) && !b.neg)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   pack_assert(I, !is_imm(a) && !(is_uniform(a) && !is_gpr(b) && !is_imm(b)));
   pack_assert(I, floating || !is_imm(b));

   uint8_t bytes[6] = {0x02, 0x01, 0x1e, 0x05, select, 0x00};
   set_bits(bytes, 21, 1, 1);
   pack_compact_retain(bytes, is_gpr(a) && retain(a), is_gpr(b) && retain(b));
   pack_compact_registers(bytes, gpr(I, I->dest[0]), is_gpr(a) ? gpr(I, a) : 0,
                          is_gpr(b) ? gpr(I, b) : 0);

   if (b.neg) {
      set_bits(bytes, 18, 1, 0);
      set_bits(bytes, 43, 1, 1);
   }

   if (is_uniform(a))
      pack_compact_uniform(bytes, 0, uniform_word(I, a));
   if (is_uniform(b))
      pack_compact_uniform(bytes, 1, uniform_word(I, b));

   if (is_imm(b)) {
      uint32_t bits = imm_float_bits(b);
      uint8_t encoded;
      bool low;
      pack_assert(I, encode_minifloat_wide(bits, &encoded, &low));

      bytes[3] = encoded;
      set_bits(bytes, 18, 1, 0);
      set_bits(bytes, 20, 1, bits >> 31);
      set_bits(bytes, 36, 1, low);
      set_bits(bytes, 41, 1, 1);
      set_bits(bytes, 42, 1, 0);
   }

   /* FP16 operands are the low halves, without uniforms or negation */
   if (half) {
      pack_assert(I, is_gpr(a) && is_gpr(b) && !b.neg);
      set_bits(bytes, 8, 1, 0);
      set_bits(bytes, 17, 1, 0);
      set_bits(bytes, 24, 1, 0);
      a9_init(p, AGX_APPLE9_ENC_HALF_MINMAX, bytes, 6);
      return;
   }

   a9_init(p, AGX_APPLE9_ENC_MINMAX_COMPACT, bytes, 6);
}

/* Compare and select: dest = (A cond B) ? C : D */
static void
pack_select(const agx_instr *I, struct a9_packed *p)
{
   int minmax = minmax_select(I);
   if (minmax >= 0) {
      pack_minmax(I, minmax, p);
      return;
   }

   unsigned cond;
   bool swap;
   pack_assert(I, select_condition(I, &cond, &swap));

   agx_index srcs[4] = {I->src[0], I->src[1], I->src[2], I->src[3]};
   if (swap) {
      agx_index t = srcs[0];
      srcs[0] = srcs[1];
      srcs[1] = t;
   }

   bool half = I->op == AGX_OPCODE_HCMPSEL;
   bool floating = I->op == AGX_OPCODE_FCMPSEL || half;
   bool abs_ab = srcs[0].abs || srcs[1].abs;
   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, !is_imm(srcs[0]) && !srcs[0].neg);
   pack_assert(I, floating || !(srcs[0].abs || srcs[1].abs || srcs[1].neg));

   bool modifiers = false;
   for (unsigned s = 0; s < 4; ++s) {
      pack_assert(I, is_imm(srcs[s]) || srcs[s].size == AGX_SIZE_32);
      modifiers |= srcs[s].abs || srcs[s].neg;
   }

   bool extended = abs_ab;

   uint8_t bytes[14] = {0x02, 0x01, 0x1f, 0x01, 0x82,
                        0x00, 0x05, 0x00, 0x80, 0x00};
   bytes[6] = cond & 0xff;
   if (cond & A9_SELECT_EQUALITY)
      bytes[4] |= 0x04;

   static const unsigned cache[] = {15, 31, 47, 79};
   static const unsigned discard[] = {19, 20, 39, 71};
   static const unsigned high[] = {56, 58, 38, 70};
   static const unsigned file[] = {55, 57, 37, 69};
   static const unsigned field[] = {9, 25, 41, 73};

   for (unsigned s = 0; s < 4; ++s) {
      agx_index src = srcs[s];

      if (is_gpr(src)) {
         unsigned reg = gpr(I, src);
         set_bits(bytes, field[s], 6, reg & 0x3f);
         set_bits(bytes, high[s], 1, reg >> 6);

         if (retain(src)) {
            set_bits(bytes, cache[s], 1, 1);
            set_bits(bytes, discard[s], 1, 0);
         }
      }
   }

   unsigned dst = gpr(I, I->dest[0]);
   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 60, 1, dst >> 6);

   for (unsigned s = 0; s < 4; ++s) {
      agx_index src = srcs[s];

      if (is_uniform(src)) {
         unsigned u = uniform_word(I, src);
         set_bits(bytes, field[s], 6, u & 0x3f);
         set_bits(bytes, cache[s], 1, 0);
         set_bits(bytes, discard[s], 1, (u >> 6) & 1);
         set_bits(bytes, high[s], 1, u >> 7);
         set_bits(bytes, file[s], 1, 1);
      } else if (is_imm(src) && s == 1) {
         set_bits(bytes, cache[s], 1, 1);
         pack_compare_literal(I, bytes, src, floating, 52, 57);
      } else if (is_imm(src)) {
         /* Constant bit 0 below the register field, bit 7 in the release bit */
         pack_assert(I, src.value <= UINT8_MAX);
         set_bits(bytes, field[s] - 1, 1, src.value & 1);
         set_bits(bytes, field[s], 6, (src.value >> 1) & 0x3f);
         set_bits(bytes, cache[s], 1, 1);
         set_bits(bytes, discard[s], 1, src.value >> 7);
         set_bits(bytes, high[s], 1, 0);
         set_bits(bytes, file[s], 1, 1);
      }
   }

   if (extended) {
      set_bits(bytes, 32, 2, 3);
      set_bits(bytes, 96, 2, srcs[0].abs | (srcs[1].abs << 1));
   }

   if (srcs[2].abs || srcs[2].neg || srcs[3].abs || srcs[3].neg)
      set_bits(bytes, 18, 1, 0);

   set_bits(bytes, 35, 1, srcs[2].abs);
   set_bits(bytes, 36, 1, srcs[2].neg);
   set_bits(bytes, 67, 1, srcs[3].abs);
   set_bits(bytes, 68, 1, srcs[3].neg);
   set_bits(bytes, 59, 1, srcs[1].neg);

   pack_assert(I, !modifiers || floating);

   /* The FP16 select compares the low halves of operands A and B */
   if (half) {
      pack_assert(I, !modifiers && is_gpr(srcs[0]) && is_gpr(srcs[1]) &&
                        is_gpr(srcs[2]) && is_gpr(srcs[3]));
      set_bits(bytes, 8, 1, 0);
      set_bits(bytes, 24, 1, 0);
      a9_init(p, AGX_APPLE9_ENC_HALF_COMPARE_SELECT, bytes, 10);
      return;
   }

   a9_init(p,
           extended ? AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED
                    : AGX_APPLE9_ENC_SELECT_GPR_WIDE,
           bytes, extended ? 14 : 10);
}

/* Logic truth tables encodable as and, or, xor or a move (or with zero) */
enum a9_logic {
   A9_LOGIC_AND,
   A9_LOGIC_OR,
   A9_LOGIC_XOR,
   A9_LOGIC_MOV_A,
   A9_LOGIC_MOV_B,
   A9_LOGIC_NONE,
};

static enum a9_logic
logic_op(uint32_t truth_table)
{
   switch (truth_table) {
   case 0x8:
      return A9_LOGIC_AND;
   case 0xe:
      return A9_LOGIC_OR;
   case 0x6:
      return A9_LOGIC_XOR;
   case 0xa:
      return A9_LOGIC_MOV_A;
   case 0xc:
      return A9_LOGIC_MOV_B;
   default:
      return A9_LOGIC_NONE;
   }
}

/* Move of a uniform word into r0-r63 */
static void
pack_uniform_move(const agx_instr *I, agx_index src, struct a9_packed *p)
{
   unsigned dst = gpr(I, I->dest[0]), u = uniform_word(I, src);

   uint8_t bytes[4] = {0x0b, 0x00, 0x01, 0x08};
   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 9, 6, u & 63);
   set_bits(bytes, 19, 1, (u >> 6) & 1);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 28, 1, u >> 7);
   a9_init(p, AGX_APPLE9_ENC_UNIFORM_MOVE, bytes, 4);
}

/* Bitwise operations, into a GPR or an export register */
static void
pack_logic(const agx_instr *I, struct a9_packed *p)
{
   enum a9_logic op = logic_op(I->truth_table);
   agx_index a = I->src[0], b = I->src[1];
   bool to_export = is_export(I->dest[0]);

   pack_assert(I, op != A9_LOGIC_NONE);

   if (op == A9_LOGIC_MOV_B) {
      a = b;
      op = A9_LOGIC_MOV_A;
   }

   if (op == A9_LOGIC_MOV_A) {
      if (is_uniform(a) && !to_export && !I->wait && gpr(I, I->dest[0]) < 64) {
         pack_uniform_move(I, a, p);
         return;
      }

      /* An export of a register is an or with itself */
      b = to_export ? a : agx_zero();
      op = A9_LOGIC_OR;
   }

   if (is_imm(a) || (is_uniform(a) && is_gpr(b))) {
      agx_index t = a;
      a = b;
      b = t;
   }

   pack_assert(I, !is_imm(a) && a.size == AGX_SIZE_32);
   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, !to_export || (is_gpr(a) && is_gpr(b)));

   uint8_t bytes[10] = {0x0b, 0x05, 0x1f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00};
   if (op == A9_LOGIC_OR) {
      bytes[4] = 0x02;
      bytes[5] = 0x08;
   } else if (op == A9_LOGIC_XOR) {
      bytes[2] = 0x1e;
      bytes[4] = 0x02;
      bytes[5] = 0x08;
   }

   bool retain_a = is_gpr(a) && retain(a), retain_b = is_gpr(b) && retain(b);
   pack_compact_retain(bytes, retain_a, retain_b);
   if (retain_b)
      set_bits(bytes, 21, 1, 1);

   pack_compact_registers(
      bytes, to_export ? export_word(I, I->dest[0]) : gpr(I, I->dest[0]),
      is_gpr(a) ? gpr(I, a) : 0, is_gpr(b) ? gpr(I, b) : 0);

   if (to_export) {
      bytes[4] |= 0x40;
      set_bits(bytes, 21, 1, 1);
      set_bits(bytes, 58, 3, completion_slot(I));
   }

   if (is_uniform(a))
      pack_compact_uniform(bytes, 0, uniform_word(I, a));
   if (is_uniform(b))
      pack_compact_uniform(bytes, 1, uniform_word(I, b));

   if (is_imm(b)) {
      /* Bit 0 at 36, bits 1-6 in the B register field, bit 7 at 20 */
      pack_assert(I, b.value <= UINT8_MAX);
      set_bits(bytes, 24, 1, 1);
      set_bits(bytes, 25, 6, (b.value >> 1) & 0x3f);
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, b.value >> 7);
      set_bits(bytes, 21, 1, 0);
      set_bits(bytes, 36, 1, b.value & 1);
      set_bits(bytes, 41, 1, 1);
      set_bits(bytes, 42, 1, 0);
   }

   a9_init(
      p,
      to_export ? AGX_APPLE9_ENC_LOGIC_EXPORT : AGX_APPLE9_ENC_LOGIC_EXTENDED,
      bytes, 10);
}

/* Export moves of the DEPTH_STORE pair; bit 68 gives each lane one word */
static void
pack_publication_pair(const agx_instr *I, struct a9_packed *p)
{
   bool split = I->imm;
   unsigned word = export_word(I, I->dest[0]);
   pack_assert(I, word <= 62 && !(word & 1));

   uint8_t bytes[20];
   unsigned moves = split ? 1 : 2;

   for (unsigned c = 0; c < moves; ++c) {
      agx_index src = I->src[c];
      pack_assert(I, is_gpr(src) && src.size == AGX_SIZE_32);

      /* The move keeps its source if a later move or instruction reads it */
      bool keep =
         retain(src) || (c == 0 && !split && agx_is_equiv(I->src[1], src));

      uint8_t *m = bytes + 10 * c;
      const uint8_t logic_or[10] = {0x0b, 0x05, 0x1f, 0x01, 0x02, 0x08};
      memcpy(m, logic_or, sizeof(logic_or));
      memset(m + 6, 0, 4);

      pack_compact_retain(m, keep, keep);
      if (keep)
         set_bits(m, 21, 1, 1);

      pack_compact_registers(m, word + c, gpr(I, src), gpr(I, src));
      m[4] |= 0x40;
      set_bits(m, 21, 1, 1);
      set_bits(m, 58, 3, completion_slot(I));

      set_bits(m, 25, 6, 0);
      set_bits(m, 42, 1, 0);
      set_bits(m, 31, 1, 0);
      set_bits(m, 20, 1, 0);
      set_bits(m, 24, 1, 0);
      set_bits(m, 43, 1, 0);

      if (!split)
         m[8] |= 0x10;
   }

   a9_init(p, AGX_APPLE9_ENC_PUBLICATION_PAIR, bytes, 10 * moves);
}

/* r0-r15 = 0-0x7f in two bytes; value bit 7 means no write */
static void
pack_mov_imm(const agx_instr *I, struct a9_packed *p)
{
   unsigned dst = gpr(I, I->dest[0]);
   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && I->imm <= UINT32_MAX);

   uint8_t compact[2];
   if (agx_apple9_encode_mov_imm7(compact, dst, I->imm)) {
      a9_init(p, AGX_APPLE9_ENC_MOV_IMM_COMPACT, compact, 2);
      return;
   }

   /* Six destination bits. A seventh at bit 60 hung the GPU. */
   uint32_t v = I->imm;
   pack_assert(I, dst < 64);
   const uint8_t bytes[8] = {
      ((dst & 0xf) << 4) | 0x0c, 0x80 | (v & 0x7f), ((dst >> 4) << 6) | 0x02,
      (v >> 24) & 0xfe,          (v >> 6) & 0x1e,   (v >> 9) & 0x0c,
      (v >> 13) & 0xff,          (v >> 21) & 0x0f,
   };
   a9_init(p, AGX_APPLE9_ENC_MOV_IMM32, bytes, 8);
}

/* System registers. 16-bit reads are followed by an in-place zero extend */
struct a9_sr {
   uint8_t selector;
   bool zext16;
};

static bool
sr_selector(enum agx_sr sr, struct a9_sr *out)
{
   switch (sr) {
   case AGX_SR_THREAD_POSITION_IN_GRID_X:
   case AGX_SR_THREAD_POSITION_IN_GRID_Y:
   case AGX_SR_THREAD_POSITION_IN_GRID_Z:
      *out = (struct a9_sr){0xa0 + (sr - AGX_SR_THREAD_POSITION_IN_GRID_X)};
      return true;
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_X:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Y:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Z:
      *out =
         (struct a9_sr){0x9c + (sr - AGX_SR_THREADGROUP_POSITION_IN_GRID_X)};
      return true;
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
      *out = (struct a9_sr){
         0xa4 + (sr - AGX_SR_THREAD_POSITION_IN_THREADGROUP_X), true};
      return true;
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
      *out =
         (struct a9_sr){0x98 + (sr - AGX_SR_THREADS_PER_THREADGROUP_X), true};
      return true;
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
      *out = (struct a9_sr){0xa7, true};
      return true;
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
      *out = (struct a9_sr){0x82, true};
      return true;
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
      *out = (struct a9_sr){0x85, true};
      return true;
   case AGX_SR_COVERAGE_MASK:
      *out = (struct a9_sr){0xc2};
      return true;
   case AGX_SR_BACKFACING:
      *out = (struct a9_sr){0xc5};
      return true;
   case AGX_SR_LAYER_ID:
   case AGX_SR_INSTANCE_ID:
   case AGX_SR_VERTEX_ID:
      *out = (struct a9_sr){sr};
      return true;
   default:
      return false;
   }
}

bool
agx_apple9_sr_supported(enum agx_sr sr)
{
   struct a9_sr out;
   return sr_selector(sr, &out);
}

static void
pack_get_sr(const agx_instr *I, struct a9_packed *p)
{
   struct a9_sr sr;
   pack_assert(I, sr_selector(I->sr, &sr));

   unsigned dst = gpr(I, I->dest[0]);
   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && dst < 64);

   if (sr.zext16) {
      const uint8_t bytes[8] = {
         ((dst & 15) << 4) | 0x04, sr.selector, 0x10 | ((dst >> 4) << 6), 0x06,
         ((dst & 15) << 4) | 0x03, 0x00,        (dst >> 4) << 6,          0x01,
      };
      a9_init(p, AGX_APPLE9_ENC_GET_SR_ZEXT16, bytes, 8);
      return;
   }

   const uint8_t bytes[4] = {
      ((dst & 0xf) << 4) | 0x0c,
      sr.selector,
      0x10 | ((dst >> 4) << 6),
      0x06,
   };
   /* Coverage, vertex and instance IDs are asynchronous, in one layout */
   a9_init(p,
           I->sr == AGX_SR_COVERAGE_MASK ? AGX_APPLE9_ENC_GET_COVERAGE
           : I->sr == AGX_SR_VERTEX_ID || I->sr == AGX_SR_INSTANCE_ID
              ? AGX_APPLE9_ENC_GET_DRAW_ID
              : AGX_APPLE9_ENC_GET_SR,
           bytes, 4);
}

/* Bytes 8-9 of a device load name the slot its result completes on */
static const uint16_t load_slot_token[6] = {
   0x1100, 0x5100, 0x9100, 0xd100, 0x1101, 0x5101,
};

/* Elements per access and the element shift of a device memory access */
static unsigned
memory_components(const agx_instr *I)
{
   pack_assert(I, I->mask && util_is_power_of_two_nonzero(I->mask + 1));
   return util_bitcount(I->mask);
}

static unsigned
memory_shift(const agx_instr *I)
{
   unsigned element = I->format == AGX_FORMAT_I8    ? 0
                      : I->format == AGX_FORMAT_I16 ? 1
                                                    : 2;
   pack_assert(I, I->format == AGX_FORMAT_I8 || I->format == AGX_FORMAT_I16 ||
                     I->format == AGX_FORMAT_I32);
   return element + I->shift;
}

/* Byte offset and index shift, shared by loads and stores two bits apart */
static void
pack_memory_address(const agx_instr *I, uint8_t *bytes, bool load,
                    unsigned shift, int offset)
{
   unsigned base = load ? 75 : 73;
   pack_assert(I, shift <= 4 && offset >= INT16_MIN && offset <= INT16_MAX);
   set_bits(bytes, base + 2, 16, (uint16_t)offset);
   set_bits(bytes, base + 22, 3, shift == 4 ? 0 : shift + 1);
}

/* Device loads of base + (index << shift) + offset, zero-extended */
static void
pack_device_load(const agx_instr *I, const struct a9_state *st,
                 struct a9_packed *p)
{
   agx_index base = I->src[0], index = I->src[1];
   unsigned components = memory_components(I);
   unsigned shift = memory_shift(I);
   bool pointer = is_imm(index);
   bool indirect = is_gpr(base);
   unsigned dst = gpr(I, I->dest[0]);

   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, components == 1 || I->format == AGX_FORMAT_I32);
   pack_assert(I, base.size == AGX_SIZE_64 && (indirect || is_uniform(base)));
   pack_assert(I, !pointer || indirect);

   uint8_t bytes[14] = {
      0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x20,
      0x00, 0x11, 0x00, 0x00, 0x40, 0x46, 0x00,
   };

   if (st->load_follows)
      bytes[2] = 0x54;

   bytes[3] = dst << 1;

   if (indirect) {
      bytes[4] = gpr(I, base);
   } else {
      pack_assert(
         I, (base.value & 3) == 0 && base.value < AGX_APPLE9_UNIFORM_COUNT * 2);
      bytes[4] = base.value >> 2;
   }

   if (pointer) {
      bytes[5] = 0;
      bytes[6] &= ~0x20;
   } else {
      pack_assert(I, index.size == AGX_SIZE_32);
      bytes[5] = gpr(I, index) | (retain(index) ? 0 : 0x80);
   }

   static const uint8_t width_bits[] = {0x00, 0x08, 0x0c, 0x06};
   static const uint8_t width_tail[] = {0x46, 0x48, 0x40, 0x40};
   uint16_t token = load_slot_token[completion_slot(I)];
   bytes[8] = (token >> 8) | width_bits[components - 1];
   bytes[9] = token & 0xff;
   bytes[12] = width_tail[components - 1];

   if (components == 1) {
      static const uint8_t format[] = {0x21, 0x01, 0x11};
      static const uint8_t tail[] = {0x42, 0x44, 0x46};
      unsigned size = shift - I->shift;
      bytes[8] = (bytes[8] & 0xc0) | format[size];
      bytes[12] = tail[size];
   }

   if (pointer) {
      /* No index: bits 53 and 102 clear. Bit 74 releases the pair. */
      bytes[12] = (bytes[12] & ~0x40) | 0x20;
      if (!retain(base))
         bytes[9] |= 4;
   } else if (indirect) {
      bytes[12] |= 0x20;
      if (!retain(base))
         bytes[9] |= 4;
   }

   int offset = (pointer ? (int)(index.value << shift) : 0) + I->byte_offset;
   pack_memory_address(I, bytes, true, pointer ? 0 : shift, offset);

   a9_init(p,
           pointer    ? AGX_APPLE9_ENC_DEVICE_LOAD_POINTER
           : indirect ? AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT
                      : AGX_APPLE9_ENC_DEVICE_LOAD,
           bytes, 14);
}

/* Device stores. Byte 6 bit 0 releases the index register. */
static void
pack_device_store(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], base = I->src[1], index = I->src[2];
   unsigned components = memory_components(I);
   unsigned shift = memory_shift(I);
   bool indirect = is_gpr(base);
   unsigned data = gpr(I, value);

   pack_assert(I, value.size == AGX_SIZE_32 && index.size == AGX_SIZE_32);
   pack_assert(I, components == 1 || I->format == AGX_FORMAT_I32);
   pack_assert(I, base.size == AGX_SIZE_64 && (indirect || is_uniform(base)));

   uint8_t bytes[14] = {
      0xe7, 0x00, 0x54, data << 1, 0x00, 0x00, 0x20,
      0x00, 0x11, 0x00, 0x00,      0x90, 0x11, 0x00,
   };

   if (indirect) {
      bytes[4] = gpr(I, base);
   } else {
      pack_assert(
         I, (base.value & 3) == 0 && base.value < AGX_APPLE9_UNIFORM_COUNT * 2);
      bytes[4] = base.value >> 2;
   }

   bytes[5] = gpr(I, index);
   bytes[6] |= !retain(index);

   if (components == 1) {
      if (I->format == AGX_FORMAT_I8) {
         bytes[8] = 0x21;
         bytes[11] = 0x90;
         bytes[12] = 0x10;
      } else if (I->format == AGX_FORMAT_I16) {
         bytes[8] = 0x01;
         bytes[11] = 0x10;
         bytes[12] = 0x11;
      }
   } else {
      static const uint8_t width_bits[] = {0x00, 0x08, 0x0c, 0x06};
      static const uint8_t width_tail[] = {0x11, 0x12, 0x10, 0x10};
      bytes[8] = 0x11 | width_bits[components - 1];
      bytes[11] = 0x10;
      bytes[12] = width_tail[components - 1];
   }

   if (indirect)
      bytes[12] |= 0x08;

   pack_memory_address(I, bytes, false, shift, I->byte_offset);
   a9_init(p,
           indirect ? AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT
                    : AGX_APPLE9_ENC_DEVICE_STORE,
           bytes, 14);
}

static unsigned
atomic_op(const agx_instr *I)
{
   switch (I->atomic_opc) {
   case AGX_ATOMIC_OPC_ADD:
      return 0x10;
   case AGX_ATOMIC_OPC_AND:
      return 0x11;
   case AGX_ATOMIC_OPC_CMPXCHG:
      return 0x12;
   case AGX_ATOMIC_OPC_FADD:
      return 0x13;
   case AGX_ATOMIC_OPC_IMAX:
      return 0x14;
   case AGX_ATOMIC_OPC_IMIN:
      return 0x15;
   case AGX_ATOMIC_OPC_OR:
      return 0x16;
   case AGX_ATOMIC_OPC_SUB:
      return 0x1b;
   case AGX_ATOMIC_OPC_UMAX:
      return 0x1c;
   case AGX_ATOMIC_OPC_UMIN:
      return 0x1d;
   case AGX_ATOMIC_OPC_XCHG:
      return 0x1e;
   case AGX_ATOMIC_OPC_XOR:
      return 0x1f;
   }

   UNREACHABLE("invalid atomic");
}

/* Device atomics. A returning one leaves the old value in private word 0 */
static void
pack_device_atomic(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], base = I->src[1], index = I->src[2];
   bool indirect = is_gpr(base);
   bool returning = !agx_is_null(I->dest[0]);
   unsigned data = gpr(I, value), idx = gpr(I, index);
   unsigned binding;

   pack_assert(I, index.size == AGX_SIZE_32 &&
                     agx_channels(value) ==
                        (I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG ? 2 : 1));

   if (indirect) {
      binding = gpr(I, base);
   } else {
      pack_assert(I, is_uniform(base) && (base.value & 3) == 0);
      binding = base.value >> 2;
   }

   pack_assert(I, binding < 128);

   uint8_t bytes[A9_MAX_BYTES] = {
      0x67,
      0x01,
      0x54,
      0x00,
      0x00,
      binding | ((data & 1) << 7),
      ((data >> 1) & 0x3f) | ((idx & 1) << 7),
      0x80 | ((idx >> 1) & 0x3f),
      0x00,
      returning ? 0x02 : 0x40,
      0x00,
      0x00,
      (atomic_op(I) << 1) | 0x40,
      0x02,
   };

   if (indirect) {
      bytes[9] |= 0x20;
      bytes[13] |= 0x01;
   }

   a9_init(p,
           indirect ? AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT
                    : AGX_APPLE9_ENC_DEVICE_ATOMIC,
           bytes, 14);

   if (returning) {
      unsigned dst = gpr(I, I->dest[0]);
      unsigned slot = hw_slot(completion_slot(I));
      const uint8_t result[8] = {
         ((dst & 0x0f) << 4) | 0x0c,
         0x80,
         0x09 | (((dst >> 4) & 3) << 6),
         ((slot - 1) << 5) | 0x07,
         0x00,
         0x20,
         0x00,
         (dst >> 6) << 4,
      };

      memcpy(p->bytes + 14, result, sizeof(result));
      p->length += sizeof(result);
   }
}

/* Register whose low half holds a threadgroup memory word index */
static unsigned
local_index(const agx_instr *I, agx_index index)
{
   return index.size == AGX_SIZE_16 ? gpr_of_half(I, index) : gpr(I, index);
}

/* Threadgroup memory. Stores and atomics destroy their data and index */
static void
pack_local_load_store(const agx_instr *I, struct a9_packed *p)
{
   bool load = I->op == AGX_OPCODE_LOCAL_LOAD;
   agx_index data = load ? I->dest[0] : I->src[0];
   agx_index base = load ? I->src[0] : I->src[1];
   agx_index index = load ? I->src[1] : I->src[2];

   pack_assert(I, I->format == AGX_FORMAT_I32 && I->mask == 1);
   pack_assert(I, is_imm(base) && base.value < 64);
   pack_assert(I, !(index.value & 1));

   unsigned d = gpr(I, data), idx = local_index(I, index);
   pack_assert(I, d < 64 && idx < 64);

   const uint8_t bytes[14] = {
      load ? 0x67 : 0xe7,
      0x02,
      0x54,
      d << 1,
      base.value << 2,
      idx << 2,
      retain(index) ? 0 : 0x80,
      0,
      0x44,
      load ? 0x0d : 0x02,
      0,
      load ? 0xc0 : 0x30,
      load ? 0x08 : 0x02,
      0,
   };
   a9_init(p, load ? AGX_APPLE9_ENC_SHARED_LOAD : AGX_APPLE9_ENC_SHARED_STORE,
           bytes, 14);
}

static void
pack_local_atomic(const agx_instr *I, struct a9_packed *p)
{
   agx_index base = I->src[1], index = I->src[2];
   bool cmpxchg = I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG;

   pack_assert(I, !agx_is_null(I->dest[0]));
   pack_assert(I, agx_channels(I->src[0]) == (cmpxchg ? 2 : 1));
   pack_assert(I, is_imm(base) && base.value < 64 && !(index.value & 1));

   unsigned dst = gpr(I, I->dest[0]), data = gpr(I, I->src[0]);
   unsigned idx = local_index(I, index);
   unsigned root = base.value * 4;
   unsigned operation = 0x4800 | (atomic_op(I) << 6);

   pack_assert(I, dst < 64 && data + (cmpxchg ? 1 : 0) < 64 && idx < 64);

   const uint8_t bytes[12] = {
      0x67, 3,    0x54, dst | ((root & 1) << 7), root >> 1,      data, idx,
      0,    0x12, 0,    operation & 0xff,        operation >> 8,
   };
   a9_init(p, AGX_APPLE9_ENC_SHARED_ATOMIC, bytes, 12);
}

/* Private memory at a word index register. Stores destroy their data */
static void
pack_private(const agx_instr *I, struct a9_packed *p)
{
   bool load = I->op == AGX_OPCODE_STACK_LOAD;
   unsigned data = gpr(I, load ? I->dest[0] : I->src[0]);
   unsigned index = gpr(I, I->src[load ? 0 : 1]);

   pack_assert(I, data < 64 && index < 64);

   if (load) {
      const uint8_t bytes[12] = {0x6f, 0, 0x54, data << 1, index << 1, 2,
                                 0x49, 1, 0,    0,         3,          0};
      a9_init(p, AGX_APPLE9_ENC_PRIVATE_LOAD, bytes, 12);
   } else {
      const uint8_t bytes[10] = {0xef, 0, 0x54, data << 1, index << 1,
                                 2,    9, 3,    0,         0x60};
      a9_init(p, AGX_APPLE9_ENC_PRIVATE_STORE, bytes, 10);
   }
}

/* Spills: fills reach the first 1024 words, saves the first 2048 */
static void
pack_stack(const agx_instr *I, struct a9_packed *p)
{
   bool load = I->op == AGX_OPCODE_STACK_LOAD;
   agx_index offset = I->src[load ? 0 : 1];
   unsigned slot = completion_slot(I);

   pack_assert(I, I->format == AGX_FORMAT_I32 && I->mask == 1);

   if (!is_imm(offset)) {
      pack_private(I, p);
      return;
   }

   pack_assert(I, (offset.value & 3) == 0);

   unsigned word = offset.value / 4;

   if (load) {
      unsigned dst = gpr(I, I->dest[0]);
      pack_assert(I, word < AGX_APPLE9_FILL_WORD_COUNT);

      const uint8_t bytes[8] = {0x0c | ((dst & 15) << 4),
                                0x80 | ((word & 63) << 1),
                                0x01 | (((dst >> 4) & 3) << 6),
                                1 | (slot << 5),
                                (word >> 6) << 1,
                                0,
                                0,
                                (dst >> 6) << 4};
      a9_init(p, AGX_APPLE9_ENC_SPILL_LOAD, bytes, 8);
   } else {
      unsigned src = gpr(I, I->src[0]);
      pack_assert(I, word < AGX_APPLE9_SAVE_WORD_COUNT);

      const uint8_t bytes[10] = {0x0b | ((word & 15) << 4),
                                 1 | ((src & 63) << 1),
                                 0x0f | (((word >> 4) & 3) << 6),
                                 0,
                                 0x22,
                                 (src >> 6) | (((word >> 6) & 1) << 4),
                                 (word >> 7) << 2,
                                 slot << 2,
                                 0,
                                 0};
      a9_init(p, AGX_APPLE9_ENC_SPILL_STORE, bytes, 10);
   }
}

/* uniform[word] = register, an integer add of zero into the uniform file */
static void
pack_uniform_store(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], index = I->src[1];
   pack_assert(I, value.size == AGX_SIZE_32 && agx_channels(value) == 1);
   pack_assert(I, is_imm(index) && !(index.value & 1) &&
                     index.value < AGX_APPLE9_UNIFORM_COUNT * 2);

   uint8_t bytes[10] = {0x9f, 0x01, 0x54, 0, 0, 0, 0, 0xa8, 0x17, 0x01};
   set_bits(bytes, 66, 1, 0);
   set_bits(bytes, 25, 8, index.value >> 1);
   set_bits(bytes, 42, 7, gpr(I, value));
   if (retain(value)) {
      set_bits(bytes, 49, 1, 1);
      set_bits(bytes, 65, 1, 0);
   }

   a9_init(p, AGX_APPLE9_ENC_STORE_UNIFORM, bytes, 10);
}

static void
pack_subgroup(const agx_instr *I, struct a9_packed *p)
{
   agx_index src = I->src[0];
   unsigned dst_reg = I->dest[0].value, src_reg = src.value;

   switch (I->op) {
   case AGX_OPCODE_SIMD_PREFIX: {
      /* Exclusive scan. The source stays live. */
      pack_assert(I, I->simd_op == AGX_SIMD_OP_IADD);
      unsigned d = gpr(I, I->dest[0]), s = gpr(I, src);
      pack_assert(I, d < 64 && s < 64);

      const uint8_t bytes[8] = {0xbf, 0x01,   0x54, d << 1,
                                0x03, s << 2, 0x16, 0x09};
      a9_init(p, AGX_APPLE9_ENC_SUBGROUP_SCAN_IADD, bytes, 8);
      return;
   }

   case AGX_OPCODE_SIMD_REDUCE: {
      /* Operands name 16-bit halves, so 32-bit ones are even */
      bool wide = src.size == AGX_SIZE_32;
      pack_assert(I, I->simd_op == AGX_SIMD_OP_UMAX);
      pack_assert(I, is_gpr(src) && is_gpr(I->dest[0]));
      pack_assert(I, src.size == I->dest[0].size && src.size <= AGX_SIZE_32);
      pack_assert(I, dst_reg < 128 && src_reg < 128);
      pack_assert(I, !wide || !((dst_reg | src_reg) & 1));

      const uint8_t bytes[8] = {
         0xbf,
         0x03,
         0x54,
         dst_reg,
         0x02 | I->dest[0].cache,
         src_reg << 1,
         0x10 | (src_reg >> 7) | (src.cache << 1) | (wide << 2),
         0x01 | (src.discard << 1),
      };
      a9_init(p, AGX_APPLE9_ENC_SUBGROUP_REDUCE_UMAX, bytes, 8);
      return;
   }

   case AGX_OPCODE_SHUFFLE: {
      /* Read the lane in the index's low half. Both sources stay live */
      agx_index lane = I->src[1];
      unsigned d = gpr(I, I->dest[0]), s = gpr(I, src);
      unsigned l =
         lane.size == AGX_SIZE_16 ? gpr_of_half(I, lane) : gpr(I, lane);
      pack_assert(I, lane.size == AGX_SIZE_32 || !(lane.value & 1));
      pack_assert(I, d < 64 && s < 64 && l < 64);

      const uint8_t bytes[10] = {0x47,   0x04,   0x54, d << 1, 0x03,
                                 s << 2, l << 2, 0x2c, 0x10,   0};
      a9_init(p, AGX_APPLE9_ENC_SUBGROUP_BROADCAST, bytes, 10);
      return;
   }

   case AGX_OPCODE_BALLOT: {
      /* Mask of active lanes with a nonzero source. Destroys the source. */
      unsigned d = gpr(I, I->dest[0]), s = gpr(I, src);
      pack_assert(I, d < 64 && s < 64);

      const uint8_t bytes[10] = {0x17,   0x07, 0x54, d << 1, 0x03,
                                 s << 2, 0x00, 0x58, 0x22,   0x12};
      a9_init(p, AGX_APPLE9_ENC_SUBGROUP_BALLOT, bytes, 10);
      return;
   }

   default:
      UNREACHABLE("not a subgroup operation");
   }
}

/* Predicate compares */
#define A9_PRED_FGT 0x02
#define A9_PRED_FLT 0x03
#define A9_PRED_UGT 0x04
#define A9_PRED_ULT 0x05
#define A9_PRED_IGT 0x06
#define A9_PRED_ILT 0x07

/* Extended form conditions */
#define A9_PRED_FEQ 0x00
#define A9_PRED_FGE 0x02
#define A9_PRED_FLE 0x03
#define A9_PRED_IEQ 0x07

static bool
pred_condition(const agx_instr *I, unsigned *cond, bool *extended)
{
   *extended = false;

   if (I->op == AGX_OPCODE_ICMP_PRED) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         *cond = A9_PRED_IEQ;
         *extended = true;
         return true;
      case AGX_ICOND_ULT:
         *cond = A9_PRED_ULT;
         return true;
      case AGX_ICOND_UGT:
         *cond = A9_PRED_UGT;
         return true;
      case AGX_ICOND_SLT:
         *cond = A9_PRED_ILT;
         return true;
      case AGX_ICOND_SGT:
         *cond = A9_PRED_IGT;
         return true;
      default:
         return false;
      }
   } else {
      switch (I->fcond) {
      case AGX_FCOND_LT:
         *cond = A9_PRED_FLT;
         return true;
      case AGX_FCOND_GT:
         *cond = A9_PRED_FGT;
         return true;
      case AGX_FCOND_EQ:
         *cond = A9_PRED_FEQ;
         *extended = true;
         return true;
      case AGX_FCOND_GE:
         *cond = A9_PRED_FGE;
         *extended = true;
         return true;
      case AGX_FCOND_LE:
         *cond = A9_PRED_FLE;
         *extended = true;
         return true;
      default:
         return false;
      }
   }
}

/* Compare into predicate pred, named by byte 0 bits 5-7; bit 4 inverts */
static void
pack_compare_pred(const agx_instr *I, struct a9_packed *p)
{
   unsigned cond;
   bool extended;
   pack_assert(I, pred_condition(I, &cond, &extended));
   pack_assert(I, I->pred < 6);

   agx_index a = I->src[0], b = I->src[1];
   bool half = I->op == AGX_OPCODE_HCMP_PRED;
   bool floating = I->op == AGX_OPCODE_FCMP_PRED || half;
   pack_assert(I, !half || (is_gpr(a) && is_gpr(b)));
   pack_assert(I, !is_imm(a) && a.size == AGX_SIZE_32);
   pack_assert(I, is_imm(b) || b.size == AGX_SIZE_32);
   pack_assert(I, !a.abs && !a.neg && !b.abs && !b.neg);

   unsigned ra = is_gpr(a) ? gpr(I, a) : uniform_word(I, a);
   unsigned rb = is_gpr(b) ? gpr(I, b) : is_uniform(b) ? uniform_word(I, b) : 0;

   uint8_t opcode = 0x0a | (I->pred << 5) | (I->invert_cond ? 0x10 : 0);
   uint8_t control = 0x22 | (extended ? 0x01 : 0);
   if (is_gpr(a) && !retain(a))
      control |= 0x08;
   if (is_gpr(b) && !retain(b))
      control |= 0x10;

   /* Bit 0 of bytes 1 and 3 marks FP32 operands, clear for FP16 */
   uint8_t bytes[10] = {opcode, ((ra & 63) << 1) | !half, control,
                        ((rb & 63) << 1) | !half};

   if (!extended) {
      bytes[4] = cond;
      set_bits(bytes, 40, 1, (ra >> 6) & 1);
      set_bits(bytes, 42, 1, (rb >> 6) & 1);

      if (is_uniform(a))
         pack_compact_uniform(bytes, 0, ra);
      if (is_uniform(b))
         pack_compact_uniform(bytes, 1, rb);
      if (is_imm(b))
         pack_compare_literal(I, bytes, b, floating, 36, 41);

      a9_init(p,
              half ? AGX_APPLE9_ENC_HALF_PREDICATE_SHORT
                   : AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT,
              bytes, 6);
   } else {
      bytes[4] = 0x06;
      bytes[6] = cond;
      set_bits(bytes, 56, 1, (ra >> 6) & 1);
      set_bits(bytes, 58, 1, (rb >> 6) & 1);

      agx_index srcs[2] = {a, b};
      for (unsigned s = 0; s < 2; ++s) {
         if (!is_uniform(srcs[s]))
            continue;

         unsigned u = s ? rb : ra;
         set_bits(bytes, s ? 31 : 15, 1, 0);
         set_bits(bytes, s ? 20 : 19, 1, (u >> 6) & 1);
         set_bits(bytes, s ? 58 : 56, 1, u >> 7);
         set_bits(bytes, s ? 57 : 55, 1, 1);
      }

      if (is_imm(b))
         pack_compare_literal(I, bytes, b, floating, 52, 57);

      a9_init(p,
              half ? AGX_APPLE9_ENC_HALF_PREDICATE_EXTENDED
                   : AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED,
              bytes, 10);
   }
}

/* Signed, instruction-relative branch, patched in at bytes 3-8 */
static void
pack_jump(const agx_instr *I, struct a9_packed *p)
{
   const uint8_t bytes[10] = {0x0f, I->op == AGX_OPCODE_JMP_EXEC_ANY ? 0 : 1,
                              0x54};
   a9_init(p,
           I->op == AGX_OPCODE_JMP_EXEC_ANY ? AGX_APPLE9_ENC_JMP_EXEC_ANY
                                            : AGX_APPLE9_ENC_JMP_EXEC_NONE,
           bytes, 10);
}

static void
pack_control(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_IF_PRED: {
      /* Predicates 6 and 7 are the constants true and false */
      pack_assert(I, I->pred < 8);
      const uint8_t bytes[4] = {
         0x0f, 0x05, 0x54, (0x01 + 4 * I->pred) | (I->invert_cond ? 0x20 : 0)};
      a9_init(p, AGX_APPLE9_ENC_EXEC_MASK_PUSH, bytes, 4);
      return;
   }

   case AGX_OPCODE_ELSE_EXEC: {
      const uint8_t bytes[4] = {0x0f, 0x04, 0x04, 0x19};
      a9_init(p, AGX_APPLE9_ENC_EXEC_MASK_ELSE, bytes, 4);
      return;
   }

   case AGX_OPCODE_POP_EXEC: {
      pack_assert(I, I->nest == 1);
      const uint8_t bytes[6] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
      a9_init(p, AGX_APPLE9_ENC_EXEC_MASK_POP, bytes, 6);
      return;
   }

   case AGX_OPCODE_BEGIN_LOOP: {
      const uint8_t bytes[4] = {0x0f, 0x05, 0x54, 0x1a};
      a9_init(p, AGX_APPLE9_ENC_LOOP_MASK_PUSH, bytes, 4);
      return;
   }

   case AGX_OPCODE_WHILE_PRED: {
      /* Metal's loops set bit 5 */
      pack_assert(I, I->pred < 6);
      const uint8_t bytes[4] = {
         0x8f, 0x04, 0x54, (0x02 + 4 * I->pred) | (I->invert_cond ? 0 : 0x20)};
      a9_init(p, AGX_APPLE9_ENC_LOOP_MASK_UPDATE, bytes, 4);
      return;
   }

   case AGX_OPCODE_END_LOOP: {
      const uint8_t bytes[6] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
      a9_init(p, AGX_APPLE9_ENC_LOOP_MASK_POP, bytes, 6);
      return;
   }

   case AGX_OPCODE_BREAK_PRED: {
      /* A loop break unwinds two scopes plus one per enclosing if */
      pack_assert(I, I->nest >= 2 && I->nest <= UINT8_MAX);
      pack_assert(I, I->pred >= 1 && I->pred < 6);
      const uint8_t bytes[6] = {0x8f, 0x05, 0x54, I->nest, 0x00, I->pred};
      a9_init(p, AGX_APPLE9_ENC_BREAK_MASK_UNWIND, bytes, 6);
      return;
   }

   default:
      UNREACHABLE("not a control flow operation");
   }
}

static void
pack_sync(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_THREADGROUP_BARRIER: {
      const uint8_t bytes[6] = {0x07, 0x04, 0x54, 0x61, 0x09, 0};
      a9_init(p, AGX_APPLE9_ENC_WORKGROUP_BARRIER, bytes, 6);
      return;
   }

   case AGX_OPCODE_MEMORY_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER_ACQUIRE: {
      /* Acquire only differs in what it waits for beforehand */
      const uint8_t bytes[6] = {0x07, 0x04, 0x54, 0x84, 0x0a, 0};
      a9_init(p, AGX_APPLE9_ENC_DEVICE_FENCE, bytes, 6);
      return;
   }

   case AGX_OPCODE_RASTER_ORDER_FENCE: {
      const uint8_t bytes[6] = {0x07, 0x04, 0x54, 0xc4, 0x08, 0};
      a9_init(p, AGX_APPLE9_ENC_DEVICE_FENCE, bytes, 6);
      return;
   }

   case AGX_OPCODE_STOP: {
      const uint8_t bytes[4] = {0x0e, 0, 0, 0};
      a9_init(p, AGX_APPLE9_ENC_HALT, bytes, 4);
      return;
   }

   case AGX_OPCODE_WAIT:
   case AGX_OPCODE_WAIT_SLOTS: {
      /* Bits 10-15 are a mask of slots */
      unsigned mask =
         I->op == AGX_OPCODE_WAIT ? BITFIELD_BIT(I->scoreboard) : I->imm;
      pack_assert(I, mask && mask < 64);
      const uint8_t bytes[2] = {0x06, mask << 2};
      a9_init(p, AGX_APPLE9_ENC_WAIT, bytes, 2);
      return;
   }

   default:
      UNREACHABLE("not a synchronization operation");
   }
}

/* Packs three or four FP32 channels into RG11B10F or RGB10A2 */
static void
pack_pack_format(const agx_instr *I, struct a9_packed *p)
{
   bool rgba = I->format == AGX_FORMAT_RGB10A2;
   pack_assert(I, rgba || I->format == AGX_FORMAT_RG11B10F);
   pack_assert(I, rgba != agx_is_null(I->src[3]));

   uint8_t bytes[14] = {0xa7,
                        0x06,
                        0x54,
                        0,
                        0x02,
                        0,
                        0,
                        0,
                        rgba ? 0x00 : 0x10,
                        0xa0,
                        rgba ? 0x2a : 0x0a,
                        0x88,
                        0x24,
                        rgba ? 0x11 : 0x08};

   /* A bit after each channel keeps it cached; bits 86-89 release */
   set_bits(bytes, 25, 7, gpr(I, I->dest[0]));
   for (unsigned s = 0; s < (rgba ? 4 : 3); ++s) {
      set_bits(bytes, 42 + 9 * s, 7, gpr(I, I->src[s]));
      set_bits(bytes, 49 + 9 * s, 1, retain(I->src[s]));
      set_bits(bytes, 86 + s, 1, !retain(I->src[s]));
   }

   a9_init(p,
           rgba ? AGX_APPLE9_ENC_PACK_FORMAT : AGX_APPLE9_ENC_PACK_FORMAT_RGB,
           bytes, 14);
}

/* Textures read parameter bundles from exports, waiting on export group 0 */

/* Texture dimension classes of the sample modes */
enum a9_tex_dim {
   A9_TEX_2D = 0,
   A9_TEX_CUBE = 1,
   A9_TEX_2D_ARRAY = 2,
   A9_TEX_3D = 3,
   A9_TEX_CUBE_ARRAY = 4,
};

static enum a9_tex_dim
a9_tex_dim(const agx_instr *I)
{
   switch (I->dim) {
   case AGX_DIM_2D:
   case AGX_DIM_2D_MS:
      return A9_TEX_2D;
   case AGX_DIM_2D_ARRAY:
   case AGX_DIM_2D_MS_ARRAY:
      return A9_TEX_2D_ARRAY;
   case AGX_DIM_3D:
      return A9_TEX_3D;
   case AGX_DIM_CUBE:
      return A9_TEX_CUBE;
   case AGX_DIM_CUBE_ARRAY:
      return A9_TEX_CUBE_ARRAY;
   default:
      UNREACHABLE("1D textures are lowered to 2D");
   }
}

/* The offset variant of a sample mode */
static bool
texture_offset_mode(uint16_t mode, uint16_t *offset_mode)
{
   static const uint16_t modes[][2] = {
      {0x0000, 0x0201}, /* implicit LOD, gather */
      {0x0004, 0x0281}, /* bias */
      {0x0100, 0x0211}, /* explicit LOD */
      {0x0508, 0x0651}, /* 2D array, explicit LOD */
      {0x0104, 0x0209}, /* gradient */
      {0x0114, 0x0249}, /* gradient, minimum LOD */
      {0x0020, 0x0221}, /* compare, implicit LOD */
      {0x0024, 0x02a1}, /* compare, bias */
      {0x0120, 0x0231}, /* compare, explicit LOD */
      {0x0124, 0x0229}, /* compare, gradient */
      {0x0180, 0x0291}, /* 3D, implicit LOD */
      {0x0190, 0x0299}, /* 3D, bias */
      {0x2188, 0x0219}, /* 3D, explicit LOD */
      {0x019c, 0x0319}, /* 3D, gradient */
   };

   for (unsigned m = 0; m < ARRAY_SIZE(modes); ++m) {
      if (modes[m][0] == mode) {
         *offset_mode = modes[m][1];
         return true;
      }
   }

   return false;
}

static bool
texture_gradient(const agx_instr *I)
{
   return I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
          I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN;
}

/* Words of the parameter bundle a texture operation reads */
unsigned
agx_apple9_texture_params(const agx_instr *I)
{
   if (texture_gradient(I))
      return I->dim == AGX_DIM_3D ? 16 : 8;
   else if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
            I->lod_mode == AGX_LOD_MODE_AUTO_LOD && (I->dim == AGX_DIM_2D) &&
            !I->shadow)
      return 2;
   else
      return 4;
}

/* Properties of a texture operation that select its mode */
struct a9_texture {
   bool gradient, lod, fetch, bias, min_lod, ms, gather;
   enum a9_tex_dim dim;
   unsigned words;
};

static struct a9_texture
a9_texture(const agx_instr *I)
{
   struct a9_texture t = {
      .gradient = texture_gradient(I),
      .words = agx_apple9_texture_params(I),
      .ms = I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY,
      .bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
              I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN,
      .min_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN ||
                 I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN,
      .gather = I->gather != AGX_GATHER_NONE,
      .dim = a9_tex_dim(I),
   };

   t.lod = !t.gradient && t.words == 4;
   t.fetch = I->op == AGX_OPCODE_TEXTURE_LOAD && !t.ms;
   return t;
}

/* Sample mode in bytes 6 and 7 without an offset, if measured */
static bool
texture_mode(const agx_instr *I, const struct a9_texture *t, uint16_t *out)
{
   uint16_t mode = 0;

   if (t->gradient)
      mode = 0x0104;
   else if (t->lod)
      mode = t->bias ? 0x0004 : 0x0100;

   /* Arrays share word 3 between the layer and the LOD */
   if (!t->gradient && t->dim == A9_TEX_2D_ARRAY)
      mode = t->bias ? 0x040c : 0x0508;

   if (!t->gradient && t->dim == A9_TEX_3D)
      mode = t->bias ? 0x0190 : 0x2188;
   else if (!t->gradient && t->dim == A9_TEX_CUBE)
      mode = t->bias ? 0x0014 : 0x0300;
   else if (!t->gradient && t->dim == A9_TEX_CUBE_ARRAY)
      mode = t->bias ? 0x001c : 0x0308;

   if (t->fetch) {
      if (!t->lod || I->shadow || t->bias || t->dim == A9_TEX_CUBE)
         return false;

      mode = t->dim == A9_TEX_2D_ARRAY ? 0x24a0
             : t->dim == A9_TEX_3D     ? 0x0198
                                       : 0x2480;
   }

   if (t->gradient && t->dim == A9_TEX_2D_ARRAY)
      mode = 0x050c;
   else if (t->gradient && t->dim == A9_TEX_3D)
      mode = 0x019c;

   if (t->min_lod) {
      if (t->dim == A9_TEX_3D)
         return false;

      if (t->gradient) {
         mode |= 0x10;
      } else {
         /* Bias with a minimum LOD, both in parameter word 3 */
         if (!t->bias || t->dim != A9_TEX_2D)
            return false;

         mode = 0x0804;
      }
   }

   if (I->shadow)
      mode |= 0x20;

   if (t->ms) {
      /* Integer X and Y, then a packed layer and sample word */
      if (I->op != AGX_OPCODE_TEXTURE_LOAD || !t->lod || t->bias)
         return false;

      mode = t->dim == A9_TEX_2D_ARRAY ? 0x14a0 : 0x1480;
   }

   if (I->offset && !texture_offset_mode(mode, &mode))
      return false;

   *out = mode;
   return true;
}

/* Whether the texture operation has a measured encoding */
bool
agx_apple9_texture_encodable(const agx_instr *I)
{
   struct a9_texture t = a9_texture(I);
   uint16_t mode;

   if (I->query_lod && (t.gradient || t.ms || I->shadow || t.fetch ||
                        (I->mask & ~3u) || I->offset))
      return false;

   if (t.gather && (t.gradient || t.ms || t.fetch || I->query_lod))
      return false;

   if (I->offset && (t.fetch || t.ms))
      return false;

   return texture_mode(I, &t, &mode);
}

static void
pack_texture(const agx_instr *I, struct a9_packed *p)
{
   agx_index params = I->src[0], texture = I->src[3], sampler = I->src[4];
   struct a9_texture t = a9_texture(I);
   bool register_texture = is_gpr(texture);
   bool register_sampler = is_gpr(sampler);
   unsigned mask = t.gather ? 0xf : I->mask;
   unsigned comps = util_bitcount(mask);
   uint16_t mode;

   pack_assert(I, agx_apple9_texture_encodable(I));
   pack_assert(I, texture_mode(I, &t, &mode));
   pack_assert(I, is_export(params) && agx_channels(params) == t.words);
   pack_assert(I, !I->sparse && mask != 0);
   pack_assert(I, agx_channels(I->dest[0]) == comps);
   pack_assert(I, register_texture || (is_imm(texture) && texture.value < 256));
   pack_assert(I, register_sampler || (is_imm(sampler) && sampler.value < 32));
   pack_assert(I, !register_sampler || register_texture);
   pack_assert(I, t.bias || !t.lod || I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                     I->lod_mode == AGX_LOD_MODE_AUTO_LOD);

   unsigned dst = gpr(I, I->dest[0]);
   unsigned coord = export_word(I, params);
   pack_assert(I,
               dst + comps <= (register_texture ? AGX_APPLE9_GPR_COUNT : 64));
   pack_assert(I, coord <= (t.gradient ? 32 - t.words : t.lod ? 28 : 30));
   pack_assert(I, (coord & (t.words == 2 ? 1 : 3)) == 0);

   unsigned tex = register_texture ? 0 : texture.value;
   unsigned smp = register_sampler ? 0 : sampler.value;

   /* Texture index: bit 0 at 71, bits 1-4 at 11-14, bits 5-7 at 68-70 */
   const uint8_t texture_hi = ((tex & 1) << 7) | ((tex >> 5) << 4);
   uint8_t bytes[20] = {
      5 | ((dst & 31) << 3),
      0x80 | (coord & 7) | (((tex >> 1) & 0xf) << 3),
      0x0c | (((dst >> 5) & 1) << 6),
      0xb8 | (coord >> 3),
      0xb0 | (smp >> 1),
      (completion_slot(I) << 2) | (completion_slot(I) << 5),
      mode & 0xff,
      mode >> 8,
      texture_hi,
      smp & 1,
      /* Filtered samples */
      t.fetch || t.ms || I->shadow || t.gather ? 0 : 0x10,
      0,
      1,
      0,
   };

   if (I->query_lod)
      bytes[10] = 0x20;

   /* Gather names its component as a one-component mask would */
   if (t.gather) {
      mask = 1u << (I->gather >> 1);
      bytes[3] |= 4;
   }

   /* Texel offsets are signed 4-bit constants */
   if (I->offset) {
      agx_index offset = I->src[5];
      pack_assert(I, is_imm(offset) && offset.value < (1 << 12));

      unsigned x = offset.value & 15;
      unsigned y = (offset.value >> 4) & 15;
      unsigned z = (offset.value >> 8) & 15;
      set_bits(bytes, 75, 4, x);
      set_bits(bytes, 79, 1, y & 1);
      set_bits(bytes, 105, 3, y >> 1);
      set_bits(bytes, 108, 3, z & 7);
      set_bits(bytes, 21, 1, z >> 3);
   }

   /* Permuted result selector; results pack in component order */
   static const uint8_t selectors[16] = {
      0, 0, 1, 5, 2, 8, 9, 6, 3, 4, 10, 13, 11, 12, 14, 7,
   };
   unsigned selector = selectors[mask];
   set_bits(bytes, 27, 2, selector & 3);
   set_bits(bytes, 37, 1, (selector >> 2) & 1);
   set_bits(bytes, 86, 1, selector >> 3);

   enum agx_apple9_encoding form = t.gradient ? AGX_APPLE9_ENC_TEXTURE_GRAD
                                   : t.lod    ? AGX_APPLE9_ENC_TEXTURE_LOD
                                              : AGX_APPLE9_ENC_TEXTURE_SAMPLE;

   if (!register_texture) {
      a9_init(p, form, bytes, 14);
      return;
   }

   /* Bit 73 selects the 20-byte form, reading indices from registers */
   bytes[9] |= 0x02;
   bytes[15] = 0x20 | (retain(texture) ? 0 : 0x08);
   bytes[18] = 0x40;
   set_bits(bytes, 153, 1, dst >> 6);

   unsigned tr = gpr(I, texture) * 2;
   set_bits(bytes, 11, 4, tr & 0xf);
   set_bits(bytes, 68, 3, (tr >> 4) & 7);
   set_bits(bytes, 122, 1, tr >> 7);
   bytes[10] |= 0x80;

   if (register_sampler) {
      unsigned sr = gpr(I, sampler) * 2;
      set_bits(bytes, 32, 4, sr & 0xf);
      set_bits(bytes, 65, 3, (sr >> 4) & 7);
      bytes[9] &= ~0x01;
      bytes[11] |= 0x01;
      bytes[14] = 0x80 | (retain(sampler) ? 0 : 0x20);
      set_bits(bytes, 116, 1, sr >> 7);
   }

   a9_init(p, form, bytes, 20);
}

/* Normalized unpacking of two unorm/snorm 8/16-bit values into a pair */
static void
pack_unpack_norm(const agx_instr *I, struct a9_packed *p)
{
   static const uint8_t modes[] = {
      [AGX_FORMAT_S16NORM] = 2,
      [AGX_FORMAT_S8NORM] = 3,
      [AGX_FORMAT_U16NORM] = 4,
      [AGX_FORMAT_U8NORM] = 6,
   };
   bool narrow =
      I->format == AGX_FORMAT_U8NORM || I->format == AGX_FORMAT_S8NORM;
   unsigned dst = gpr(I, I->dest[0]);
   agx_index src = I->src[0];

   pack_assert(I, I->format < ARRAY_SIZE(modes) && modes[I->format]);
   pack_assert(I, agx_channels(I->dest[0]) == 2 && !(dst & 1));
   pack_assert(I, narrow || !I->imm);

   uint8_t bytes[8] = {0x17, 0x04, 0x54, 0, 0, 0, 0x04, 0x8a};
   set_bits(bytes, 25, 7, dst);
   set_bits(bytes, 41, 1, I->imm);
   set_bits(bytes, 42, 7, gpr(I, src));
   set_bits(bytes, 49, 1, retain(src));
   set_bits(bytes, 51, 1, !narrow);
   set_bits(bytes, 52, 1, !retain(src));
   set_bits(bytes, 60, 3, modes[I->format]);
   a9_init(p, AGX_APPLE9_ENC_UNPACK_NORM, bytes, 8);
}

/* unorm packing of FP32 pairs; 8-bit writes one half per pair */
static void
pack_pack_unorm(const agx_instr *I, struct a9_packed *p)
{
   bool bytes = I->op == AGX_OPCODE_PACK_UNORM_4X8;
   unsigned dst = gpr(I, I->dest[0]);
   uint8_t code[20] = {0};

   for (unsigned pair = 0; pair < (bytes ? 2 : 1); ++pair) {
      unsigned a = gpr(I, I->src[2 * pair]);
      unsigned b = gpr(I, I->src[2 * pair + 1]);
      pack_assert(I, !bytes || (a != dst && b != dst));

      uint8_t *part = code + 10 * pair;
      memcpy(
         part,
         (const uint8_t[]){0x97, 0x04, 0x56, 0, 0x02, 0, 0, 0x50, 0x44, 0x82},
         10);
      set_bits(part, 25, 7, dst);
      set_bits(part, 42, 7, a);
      set_bits(part, 51, 7, b);

      if (bytes) {
         part[2] = 0x54;
         set_bits(part, 24, 1, pair);
         part[9] = 0xc2;
      }
   }

   a9_init(
      p, bytes ? AGX_APPLE9_ENC_PACK_UNORM_4X8 : AGX_APPLE9_ENC_PACK_UNORM_2X16,
      code, bytes ? 20 : 10);
}

/* Cube face selection: major axis, then the face in the next low half */
static void
pack_cube(const agx_instr *I, struct a9_packed *p)
{
   pack_assert(I, I->imm <= 2);
   pack_assert(I, agx_channels(I->dest[0]) == (I->imm == 0 ? 2 : 1));

   uint8_t bytes[12] = {0x17, 1, 0x54, 0, 3, 0, 0, 0, 0x50, 0x2f, 0x92, 0};
   set_bits(bytes, 24, 7, gpr(I, I->dest[0]));
   set_bits(bytes, 90, 2, I->imm);

   for (unsigned i = 0; i < 3; ++i) {
      set_bits(bytes, 41 + 9 * i, 7, gpr(I, I->src[i]));

      if (retain(I->src[i])) {
         set_bits(bytes, 48 + 9 * i, 1, 1);
         set_bits(bytes, 73 + i, 1, 0);
      }
   }

   a9_init(p, AGX_APPLE9_ENC_CUBE, bytes, 12);
}

/* PBE write of 32-bit data, then the join of its parameter export */
static void
pack_image_write(const agx_instr *I, struct a9_packed *p)
{
   agx_index params = I->src[0], image = I->src[4];
   unsigned base = export_word(I, params);
   unsigned index = gpr(I, image);

   pack_assert(I, agx_channels(params) == 8 && base <= 24 && !(base & 7));
   pack_assert(I, agx_channels(image) == 2 && !(index & 1));
   pack_assert(I, I->dim == AGX_DIM_2D);

   const uint8_t bytes[22] = {
      0xd7,
      0x10,
      0x54,
      base << 1,
      0,
      0x40,
      0,
      index << 1,
      0x20 | (retain(image) ? 0 : 1),
      0x04,
      0,
      0,
      0x88,
      I->integer ? 0x1a : 0x3a,
      0x89,
      0,
      0x07,
      0x12,
      0x54,
      0,
      3,
      0,
   };
   a9_init(p, AGX_APPLE9_ENC_IMAGE_WRITE, bytes, 22);
}

/* Block image store of the tile, then the join of its parameter export */
static void
pack_block_image_store(const agx_instr *I, struct a9_packed *p)
{
   unsigned base = export_word(I, I->src[0]);
   bool ms = I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY;
   bool array = I->dim == AGX_DIM_2D_ARRAY || I->dim == AGX_DIM_2D_MS_ARRAY;

   pack_assert(I, agx_channels(I->src[0]) == 4 && base <= 12 && !(base & 3));
   pack_assert(I, I->src[1].value < 16 && I->format <= 15);

   unsigned mode = array ? (ms ? 0x028 : 0x528) : ms ? 0x228 : 0x5a8;

   const uint8_t bytes[18] = {
      0x57,        0x10,
      0x54,        base << 1,
      0,           I->src[1].value << 4,
      0,           0x80,
      mode & 0xff, (I->format << 4) | (mode >> 8),
      1,           0,
      0x07,        0x12,
      0x54,        0,
      3,           0,
   };
   a9_init(p,
           (array || ms) ? AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_EXTENDED
                         : AGX_APPLE9_ENC_BLOCK_IMAGE_STORE,
           bytes, 18);
}

/* Fragment outputs and tile access */
static void
pack_fragment(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_WRITE_COVERAGE: {
      unsigned mask = export_word(I, I->src[0]);
      pack_assert(I, mask < 64);

      const uint8_t bytes[6] = {0x57, 0x04, 0x54, mask << 1, 0, 1};
      a9_init(p, AGX_APPLE9_ENC_COVERAGE, bytes, 6);
      return;
   }

   case AGX_OPCODE_ZS_STORE: {
      /* Mode 3 writes depth, 1 tests stencil in word 1's high half, 7 both */
      static const uint8_t modes[4] = {0, 3, 1, 7};
      agx_index zs = I->src[0];
      pack_assert(I, is_export(zs) && agx_channels(zs) == 2);
      pack_assert(I, I->zs >= 1 && I->zs <= 3 && !(zs.value & 3));

      unsigned word = export_word(I, zs);
      pack_assert(I, word < 63);

      const uint8_t bytes[6] = {0xd7, 0x14, 0x54, word << 1, 0, modes[I->zs]};
      a9_init(p, AGX_APPLE9_ENC_DEPTH_STORE, bytes, 6);
      return;
   }

   case AGX_OPCODE_BEGIN_TILE:
   case AGX_OPCODE_END_TILE: {
      /* Only the measured controls are known to work */
      bool begin = I->op == AGX_OPCODE_BEGIN_TILE;
      uint64_t c = I->imm;
      pack_assert(I, begin ? (c == 0x700 || c == 0xf00 || c == 0x600 ||
                              (c & ~0xfcull) == 0x800 || c == 1)
                           : (c == 0x300 || (c & ~0xfcull) == 0 ||
                              (c & ~0xfcull) == 0x200 || c == 1));

      uint8_t bytes[6];
      agx_apple9_encode_tile(bytes, begin, c);
      a9_init(p, begin ? AGX_APPLE9_ENC_TILE_ACCESS : AGX_APPLE9_ENC_TILE_FENCE,
              bytes, 6);
      return;
   }

   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ST_TILE: {
      /* Raw tile words at word pixel_offset of the pixel */
      bool load = I->op == AGX_OPCODE_LD_TILE;
      agx_index mask = I->src[load ? 0 : 1];
      agx_index data = load ? I->dest[0] : I->src[0];
      bool coords = I->explicit_coords;
      bool dynamic = !is_imm(mask);

      pack_assert(I, I->format == AGX_FORMAT_I32 && I->mask == 1);
      pack_assert(I, I->pixel_offset < 32);
      pack_assert(I, !coords || load);
      pack_assert(I, coords ? (!dynamic && mask.value >= 1 && mask.value <= 15)
                            : (dynamic || mask.value == 1));
      pack_assert(I, !dynamic || (mask.size <= AGX_SIZE_32 && is_gpr(mask) &&
                                  mask.value < 256));

      unsigned d = gpr(I, data);
      /* A register mask is named by its 16-bit half */
      uint8_t mask_byte = mask.value;

      if (load) {
         unsigned tag = 0x4e | (completion_slot(I) << 7);
         const uint8_t bytes[12] = {
            0x67,
            coords    ? 0x16
            : dynamic ? 0x06
                      : 0x0e,
            0x54,
            d << 1,
            coords ? gpr(I, I->src[1]) : 0,
            I->pixel_offset << 1,
            mask_byte,
            tag & 0xff,
            (tag >> 8) | (dynamic && !retain(mask) ? 0x20 : 0),
            0,
            0,
            coords    ? 0x10
            : dynamic ? 0x20
                      : 0,
         };
         a9_init(p,
                 coords    ? AGX_APPLE9_ENC_TILE_LOAD_COORDS
                 : dynamic ? AGX_APPLE9_ENC_TILE_LOAD_MASK
                           : AGX_APPLE9_ENC_TILE_LOAD,
                 bytes, 12);
      } else {
         const uint8_t bytes[12] = {
            0xe7,      0x06, 0x54,
            d << 1,    0,    I->pixel_offset << 1,
            mask_byte, 0x4e, dynamic && !retain(mask) ? 8 : 0,
            0,         0,    dynamic ? 8 : 0,
         };
         a9_init(p,
                 dynamic ? AGX_APPLE9_ENC_TILE_STORE_MASK
                         : AGX_APPLE9_ENC_TILE_STORE,
                 bytes, 12);
      }
      return;
   }

   default:
      UNREACHABLE("not a fragment operation");
   }
}

/* Varyings and interpolation */
static void
pack_varying(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_ST_VARY: {
      /* Commits an export register to a varying slot */
      agx_index slot = I->src[0];
      uint8_t bytes[8];
      pack_assert(
         I, is_imm(slot) && agx_apple9_encode_vary_store(
                               bytes, export_word(I, I->src[1]), slot.value));
      a9_init(p, AGX_APPLE9_ENC_VARY_STORE, bytes, 8);
      return;
   }

   case AGX_OPCODE_ITER: {
      /* One coefficient, at the centre or a position. Coefficient 0 is 1/W */
      agx_index cf = I->src[0];
      bool coord = I->interpolation != AGX_INTERPOLATION_CENTER;
      pack_assert(I, is_imm(cf) && cf.value < AGX_APPLE9_NUM_COEFFICIENTS &&
                        I->channels == 1);
      pack_assert(I, !coord || I->interpolation == AGX_INTERPOLATION_SAMPLE ||
                        I->interpolation == AGX_INTERPOLATION_CENTROID ||
                        I->interpolation == AGX_INTERPOLATION_SAMPLE_REGISTER);

      unsigned position = coord ? gpr(I, I->src[1]) : 0;
      const uint8_t bytes[10] = {
         0x2f,
         0x05,
         0x54,
         gpr(I, I->dest[0]) << 1,
         0x03,
         cf.value << 1,
         position,
         2,
         coord ? (retain(I->src[1]) ? 0x08 : 0x20) : 0x10,
         0,
      };
      a9_init(p, coord ? AGX_APPLE9_ENC_ITER_COORD : AGX_APPLE9_ENC_ITER, bytes,
              10);
      return;
   }

   case AGX_OPCODE_LDCF: {
      /* Reads the three-word coefficient tuple, completing on its slot */
      agx_index cf = I->src[0];
      unsigned dst = gpr(I, I->dest[0]);
      pack_assert(I, is_imm(cf) && cf.value < 512 && I->channels == 3);
      pack_assert(I, agx_channels(I->dest[0]) == 3);

      const uint8_t bytes[6] = {0x1f,          0x03,
                                0x54,          dst | ((cf.value & 1) << 7),
                                cf.value >> 1, completion_slot(I)};
      a9_init(p, AGX_APPLE9_ENC_ITER_FLAT, bytes, 6);
      return;
   }

   case AGX_OPCODE_INTERPOLATION_POSITION: {
      /* Sample position or centroid; destroys the source register */
      agx_index src = I->src[0];
      bool sample = I->interpolation == AGX_INTERPOLATION_SAMPLE ||
                    I->interpolation == AGX_INTERPOLATION_SAMPLE_REGISTER;
      pack_assert(I, sample || I->interpolation == AGX_INTERPOLATION_CENTROID);

      unsigned s = gpr(I, src);
      const uint8_t bytes[8] = {0xaf,
                                0x04,
                                0x54,
                                gpr(I, I->dest[0]) << 1,
                                0x03,
                                (s & 63) << 2,
                                0x0a | (s >> 6),
                                sample ? 0x03 : 0x01};
      a9_init(p, AGX_APPLE9_ENC_INTERPOLATION_POSITION, bytes, 8);
      return;
   }

   default:
      UNREACHABLE("not a varying operation");
   }
}

/* fmov from FP32 to FP16 or back */
static bool
is_float_convert(const agx_instr *I)
{
   return I->op == AGX_OPCODE_FADD && agx_is_equiv(I->src[1], agx_negzero()) &&
          !I->src[1].abs && !I->src[1].neg && I->dest[0].size != I->src[0].size;
}

/* Smallest form of an instruction, with a dependency field if it waits */
static void
pack_body(const agx_instr *I, const struct a9_state *st, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_FADD:
      if (is_float_convert(I))
         pack_float_convert(I, p);
      else
         pack_float2(I, p);
      return;

   case AGX_OPCODE_FMUL:
      if (is_export(I->dest[0]))
         pack_float2_export(I, p);
      else
         pack_float2(I, p);
      return;

   case AGX_OPCODE_FMUL_PERSPECTIVE:
      pack_float2_perspective(I, p);
      return;

   case AGX_OPCODE_F32_TO_F16:
   case AGX_OPCODE_F16_TO_F32:
      pack_float_convert(I, p);
      return;

   case AGX_OPCODE_PACK_HALF:
      pack_half_pair(I, p);
      return;

   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
      pack_texture(I, p);
      return;

   case AGX_OPCODE_CUBE:
      pack_cube(I, p);
      return;

   case AGX_OPCODE_UNPACK_NORM:
      pack_unpack_norm(I, p);
      return;

   case AGX_OPCODE_PACK_UNORM_2X16:
   case AGX_OPCODE_PACK_UNORM_4X8:
      pack_pack_unorm(I, p);
      return;

   case AGX_OPCODE_IMAGE_WRITE:
      pack_image_write(I, p);
      return;

   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      pack_block_image_store(I, p);
      return;

   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
      pack_half_alu(I, p);
      return;

   case AGX_OPCODE_FFMA:
      pack_float3(I, p);
      return;

   case AGX_OPCODE_IADD:
      pack_iadd(I, p);
      return;

   case AGX_OPCODE_IMAD:
      pack_imad(I, p);
      return;

   case AGX_OPCODE_ASR:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
      pack_shift(I, p);
      return;

   case AGX_OPCODE_CONVERT:
      pack_convert(I, p);
      return;

   case AGX_OPCODE_BITREV:
   case AGX_OPCODE_POPCOUNT:
   case AGX_OPCODE_FFS:
      pack_bit_unary(I, p);
      return;

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
      pack_special(I, st, p);
      return;

   case AGX_OPCODE_DFDX:
   case AGX_OPCODE_DFDY:
      pack_derivative(I, p);
      return;

   case AGX_OPCODE_HMIN:
   case AGX_OPCODE_HMAX:
      pack_minmax(I, I->op == AGX_OPCODE_HMIN ? 1 : 0, p);
      return;

   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_FCMPSEL:
   case AGX_OPCODE_HCMPSEL:
      pack_select(I, p);
      return;

   case AGX_OPCODE_PUBLICATION_PAIR:
      pack_publication_pair(I, p);
      return;

   case AGX_OPCODE_BITOP:
      pack_logic(I, p);
      return;

   case AGX_OPCODE_MOV_IMM:
      pack_mov_imm(I, p);
      return;

   case AGX_OPCODE_GET_SR:
   case AGX_OPCODE_GET_SR_COVERAGE:
   case AGX_OPCODE_GET_SR_BARRIER:
      pack_get_sr(I, p);
      return;

   case AGX_OPCODE_DEVICE_LOAD:
      pack_device_load(I, st, p);
      return;

   case AGX_OPCODE_DEVICE_STORE:
      pack_device_store(I, p);
      return;

   case AGX_OPCODE_ATOMIC:
      pack_device_atomic(I, p);
      return;

   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
      pack_local_load_store(I, p);
      return;

   case AGX_OPCODE_LOCAL_ATOMIC:
      pack_local_atomic(I, p);
      return;

   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
      pack_stack(I, p);
      return;

   case AGX_OPCODE_UNIFORM_STORE:
      pack_uniform_store(I, p);
      return;

   case AGX_OPCODE_SIMD_PREFIX:
   case AGX_OPCODE_SIMD_REDUCE:
   case AGX_OPCODE_SHUFFLE:
   case AGX_OPCODE_BALLOT:
      pack_subgroup(I, p);
      return;

   case AGX_OPCODE_ICMP_PRED:
   case AGX_OPCODE_FCMP_PRED:
   case AGX_OPCODE_HCMP_PRED:
      pack_compare_pred(I, p);
      return;

   case AGX_OPCODE_IF_PRED:
   case AGX_OPCODE_ELSE_EXEC:
   case AGX_OPCODE_POP_EXEC:
   case AGX_OPCODE_BEGIN_LOOP:
   case AGX_OPCODE_WHILE_PRED:
   case AGX_OPCODE_END_LOOP:
   case AGX_OPCODE_BREAK_PRED:
      pack_control(I, p);
      return;

   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      pack_jump(I, p);
      return;

   case AGX_OPCODE_THREADGROUP_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER_ACQUIRE:
   case AGX_OPCODE_RASTER_ORDER_FENCE:
   case AGX_OPCODE_STOP:
   case AGX_OPCODE_WAIT:
   case AGX_OPCODE_WAIT_SLOTS:
      pack_sync(I, p);
      return;

   case AGX_OPCODE_PACK_FORMAT:
      pack_pack_format(I, p);
      return;

   case AGX_OPCODE_WRITE_COVERAGE:
   case AGX_OPCODE_ZS_STORE:
   case AGX_OPCODE_BEGIN_TILE:
   case AGX_OPCODE_END_TILE:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ST_TILE:
      pack_fragment(I, p);
      return;

   case AGX_OPCODE_ST_VARY:
   case AGX_OPCODE_ITER:
   case AGX_OPCODE_LDCF:
   case AGX_OPCODE_INTERPOLATION_POSITION:
      pack_varying(I, p);
      return;

   default:
      /* Pseudo-instructions and unencoded operations */
      p->form = AGX_APPLE9_ENC_PSEUDO;
      p->length = p->dependency_length = 0;
      return;
   }
}

/* Folds the wait into one six-bit slot mask or two three-bit ones */
static void
pack_dependency(const agx_instr *I, struct a9_packed *p)
{
   enum agx_apple9_dependency_layout layout =
      agx_apple9_encoding_info(p->form)->dependency_layout;

   if (!I->wait)
      return;

   unsigned slot = hw_slot(I->wait_slot);
   pack_assert(I, layout != AGX_APPLE9_DEPENDENCY_NONE);

   switch (layout) {
   case AGX_APPLE9_DEPENDENCY_INDEX_45_47:
      pack_assert(I, p->dependency_length * 8 >= 48);
      set_bits(p->bytes, 45, 3, slot);
      return;

   case AGX_APPLE9_DEPENDENCY_INDEX_61_63:
      pack_assert(I, p->dependency_length * 8 >= 64);
      set_bits(p->bytes, 61, 3, slot);
      return;

   case AGX_APPLE9_DEPENDENCY_MASK_12_17:
      set_bits(p->bytes, 12, 6, 1u << (slot - 1));
      return;

   case AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63:
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79:
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95: {
      unsigned first =
         layout == AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63 ? 45 : 61;
      unsigned second =
         first + (layout == AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95 ? 32 : 16);
      pack_assert(I, p->dependency_length * 8 >= second + 3);
      set_bits(p->bytes, first, 3, slot <= 3 ? 1u << (slot - 1) : 0);
      set_bits(p->bytes, second, 3, slot >= 4 ? 1u << (slot - 4) : 0);
      return;
   }

   case AGX_APPLE9_DEPENDENCY_NONE:
      UNREACHABLE("checked above");
   }
}

enum agx_apple9_encoding
agx_apple9_encoding_for(const agx_instr *I)
{
   struct a9_state st = {0};
   struct a9_packed p;
   pack_body(I, &st, &p);
   return p.form;
}

/* Zero-byte instructions */
static bool
is_empty(const agx_instr *I)
{
   return I->op == AGX_OPCODE_EXPORT;
}

unsigned
agx_apple9_pack_instr(const agx_instr *I, bool load_follows,
                      bool alu_reads_result, uint8_t *bytes)
{
   if (is_empty(I))
      return 0;

   struct a9_state st = {
      .load_follows = load_follows,
      .alu_reads_result = alu_reads_result,
   };
   struct a9_packed p;

   pack_body(I, &st, &p);
   pack_assert(I, p.form != AGX_APPLE9_ENC_PSEUDO && "no Apple9 encoding");
   pack_dependency(I, &p);
   memcpy(bytes, p.bytes, p.length);
   return p.length;
}

/* Special functions, which hint who reads their result */
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
alu_reads_result(agx_block *block, agx_instr *I)
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

struct a9_fixup {
   off_t offset;
   agx_block *block;
   bool skip_to_end;
};

void
agx_pack_binary_apple9(agx_context *ctx, struct util_dynarray *emission)
{
   struct util_dynarray fixups;
   util_dynarray_init(&fixups, ctx);

   unsigned loads = 0;
   agx_foreach_instr_global(ctx, I) {
      loads += I->op == AGX_OPCODE_DEVICE_LOAD;
   }

   agx_foreach_block(ctx, block) {
      block->offset = emission->size;

      agx_foreach_instr_in_block(block, I) {
         block->last_offset = emission->size;

         bool load_follows = false;
         if (I->op == AGX_OPCODE_DEVICE_LOAD)
            load_follows = --loads > 0;

         bool reads = is_special(I) && alu_reads_result(block, I);

         if (I->op == AGX_OPCODE_JMP_EXEC_ANY ||
             I->op == AGX_OPCODE_JMP_EXEC_NONE ||
             I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER) {
            struct a9_fixup fixup = {
               .offset = emission->size,
               .block = I->target,
               .skip_to_end = I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER,
            };
            util_dynarray_append(&fixups, fixup);
         }

         /* The linker relocates the slots of varying commits */
         if (I->op == AGX_OPCODE_ST_VARY) {
            assert(I->src[0].value < AGX_APPLE9_MAX_UVS_SLOTS &&
                   !ctx->out->apple9_vary_stores[I->src[0].value]);
            ctx->out->apple9_vary_stores[I->src[0].value] = emission->size + 1;
         }

         uint8_t bytes[A9_MAX_BYTES];
         unsigned length = agx_apple9_pack_instr(I, load_follows, reads, bytes);
         memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
      }
   }

   /* Branch displacements are relative to the branch, in bytes 3-8 */
   util_dynarray_foreach(&fixups, struct a9_fixup, fixup) {
      off_t target =
         fixup->skip_to_end ? fixup->block->last_offset : fixup->block->offset;
      int64_t displacement = (int64_t)target - (int64_t)fixup->offset;
      uint8_t *location = (uint8_t *)emission->data + fixup->offset;

      for (unsigned i = 0; i < 6; ++i)
         location[3 + i] = (uint64_t)displacement >> (8 * i);
   }

   util_dynarray_fini(&fixups);
}
