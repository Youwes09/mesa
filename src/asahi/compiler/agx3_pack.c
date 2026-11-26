/*
 * Copyright 2026 Youwes09
 * SPDX-License-Identifier: MIT
 */

#include "agx3_pack.h"
#include "agx3_machine.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/* AGX3 instruction packing: the instructions of AGX3.xml that
 * encode an IR instruction, and their fields. A released source register
 * loses its value, so only last uses (the discard flag) set the release bit.
 */

/* Largest instruction or instruction pair packed for one IR instruction */
#define A9_MAX_BYTES 24

struct a9_packed {
   uint8_t bytes[A9_MAX_BYTES + AGX3_MAX_LENGTH];
   unsigned length;
   enum agx3_encoding form;

   /* Only select the form, without packing */
   bool form_only;
};

static void
pack_assert_internal(const agx_instr *I, bool condition, const char *msg)
{
   if (!condition) {
      fprintf(stderr, "AGX3 packing assertion failed for instruction:\n\n");
      agx_print_instr(I, stderr);
      fprintf(stderr, "\n%s\n", msg);
      abort();
   }
}

#define pack_assert(I, cond) pack_assert_internal(I, cond, #cond)

/* Hardware slot (1-6) for an IR scoreboard slot (0-5) */
static unsigned
hw_slot(unsigned slot)
{
   assert(slot < 6);
   return slot + 1;
}

/* Fields an instruction waits through, by the layout of its form */
static void
set_wait(struct agx3_fields *f, enum agx3_dependency_layout layout,
         unsigned slot)
{
   switch (layout) {
   case AGX3_DEPENDENCY_INDEX_45_47:
      f->index_45_47 = slot;
      return;
   case AGX3_DEPENDENCY_INDEX_61_63:
      f->index_61_63 = slot;
      return;
   case AGX3_DEPENDENCY_MASK_12_17:
      f->mask_12_17 = BITFIELD_BIT(slot - 1);
      return;
   case AGX3_DEPENDENCY_MASK_45_47_61_63:
      f->mask_45_47_61_63 = BITFIELD_BIT(slot - 1);
      return;
   case AGX3_DEPENDENCY_MASK_61_63_77_79:
      f->mask_61_63_77_79 = BITFIELD_BIT(slot - 1);
      return;
   case AGX3_DEPENDENCY_MASK_61_63_93_95:
      f->mask_61_63_93_95 = BITFIELD_BIT(slot - 1);
      return;
   case AGX3_DEPENDENCY_NONE:
      UNREACHABLE("no dependency field");
   }
}

/* Appends an instruction of the form. The first one folds the wait. */
static void
emit(const agx_instr *I, struct a9_packed *p, enum agx3_encoding form,
     agx3_pack_fn pack, struct agx3_fields f)
{
   p->form = form;
   if (p->form_only)
      return;

   if (p->length == 0 && I->wait) {
      enum agx3_dependency_layout layout =
         agx3_encoding_info(form)->dependency_layout;

      pack_assert(I, layout != AGX3_DEPENDENCY_NONE);
      set_wait(&f, layout, hw_slot(I->wait_slot));
   }

   unsigned length = pack(p->bytes + p->length, &f);
   pack_assert(I, length != 0 && "fields the instruction cannot encode");
   p->length += length;
   pack_assert(I, p->length <= A9_MAX_BYTES);
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

/* A register, uniform or export operand */
static struct agx3_operand
plain(const agx_instr *I, agx_index idx)
{
   enum agx3_kind kind;
   unsigned count;

   if (is_gpr(idx)) {
      kind = AGX3_KIND_REG;
      count = AGX3_GPR_COUNT;
   } else if (is_export(idx)) {
      kind = AGX3_KIND_EXPORT;
      count = AGX3_PUBLICATION_COUNT;
   } else {
      pack_assert(I, is_uniform(idx));
      kind = AGX3_KIND_UNIFORM;
      count = AGX3_UNIFORM_COUNT;
   }

   pack_assert(I, idx.value + agx_index_size_16(idx) <= count * 2);

   return (struct agx3_operand){
      .kind = kind,
      .value = idx.value,
      .size32 = idx.size == AGX_SIZE_32,
      .size64 = idx.size == AGX_SIZE_64,
   };
}

/* A source, a register keeping its cache bit until its release */
static struct agx3_operand
src(const agx_instr *I, agx_index idx)
{
   struct agx3_operand op = plain(I, idx);
   op.cache = is_gpr(idx) && !idx.discard;
   op.lu = is_gpr(idx) && idx.discard;
   return op;
}

/* A source released at its last use, without a cache bit */
static struct agx3_operand
released(const agx_instr *I, agx_index idx)
{
   struct agx3_operand op = plain(I, idx);
   op.lu = is_gpr(idx) && idx.discard;
   return op;
}

/* The low 16-bit half of a register */
static struct agx3_operand
low_half(struct agx3_operand op)
{
   op.size32 = false;
   return op;
}

/* Integer constant */
static struct agx3_operand
constant(uint32_t value, bool size32)
{
   return (struct agx3_operand){
      .kind = AGX3_KIND_IMM,
      .value = value,
      .size32 = size32,
   };
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
agx3_float_immediate_ok(agx_index idx, bool wide)
{
   uint8_t encoded;
   bool low;

   if (!is_imm(idx) || idx.value >= 0x100 || idx.abs)
      return false;

   uint32_t bits = imm_float_bits(idx);
   return wide ? encode_minifloat_wide(bits, &encoded, &low)
               : encode_minifloat(bits, &encoded);
}

/* Float constant operand, the wide form with a fourth mantissa bit */
static struct agx3_operand
float_constant(const agx_instr *I, uint32_t bits, bool wide)
{
   uint8_t encoded;
   bool low = false;

   if (wide)
      pack_assert(I, encode_minifloat_wide(bits, &encoded, &low));
   else
      pack_assert(I, encode_minifloat(bits, &encoded));

   return (struct agx3_operand){
      .kind = AGX3_KIND_A9FIMM,
      .value = (encoded >> 1) | (low << 7),
      .neg = bits >> 31,
   };
}

/* The -0.0 operand of conversions, which add it */
static const struct agx3_operand negzero = {
   .kind = AGX3_KIND_A9FIMM,
   .value = 0x40,
   .neg = true,
};

/* Floating-point add and multiply, in the smallest form that encodes them */
static void
pack_float2(const agx_instr *I, struct a9_packed *p)
{
   bool mul = I->op == AGX_OPCODE_FMUL;
   agx_index a = I->src[0], b = I->src[1];

   pack_assert(I, I->dest[0].size == AGX_SIZE_32);

   /* Constants are operand A of the instruction */
   if (is_imm(a)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   uint64_t dest = agx3_encode_dst(plain(I, I->dest[0]));

   if (is_imm(b)) {
      pack_assert(I, is_gpr(a) && a.size == AGX_SIZE_32 && !a.abs && !b.abs);

      struct agx3_operand x = src(I, a);
      x.neg = a.neg;

      uint32_t bits = imm_float_bits(b) ^ (b.neg ? 0x80000000u : 0);
      struct agx3_fields f = {
         .dest = dest,
         .src = {agx3_encode_fsrc(float_constant(I, bits, true)),
                 agx3_encode_fsrc(x)},
         .dstate = 1,
         .saturate = I->saturate,
      };

      if (I->saturate)
         emit(I, p, AGX3_ENC_FLOAT2_IMMEDIATE_EXTENDED,
              mul ? agx3_pack_fmul_mod : agx3_pack_fadd_mod, f);
      else
         emit(I, p, AGX3_ENC_FLOAT2_IMMEDIATE_COMPACT,
              mul ? agx3_pack_fmul : agx3_pack_fadd, f);
      return;
   }

   /* The uniform may be A or B; only B can be negated in the compact form */
   if (is_uniform(a) && !is_uniform(b) && !a.neg && !a.abs && !b.neg &&
       !b.abs) {
      agx_index t = a;
      a = b;
      b = t;
   }

   struct agx3_operand x = src(I, a), y = src(I, b);
   x.neg = a.neg;
   x.abs = a.abs;
   y.neg = b.neg;
   y.abs = b.abs;

   /* FP16 sources convert on read */
   if (is_gpr(a) && a.size == AGX_SIZE_16)
      x = low_half(x);
   if (is_gpr(b) && b.size == AGX_SIZE_16)
      y = low_half(y);

   bool hi = (is_gpr(a) && a.size == AGX_SIZE_16 && (a.value & 1)) ||
             (is_gpr(b) && b.size == AGX_SIZE_16 && (b.value & 1));
   bool low = I->dest[0].value < 128 && (!is_gpr(a) || a.value < 128) &&
              (!is_gpr(b) || b.value < 128);

   struct agx3_fields f = {
      .dest = dest,
      .src = {agx3_encode_fsrc(x), agx3_encode_fsrc(y)},
      .dstate = 1,
      .saturate = I->saturate,
   };

   if (a.abs || b.abs) {
      emit(
         I, p,
         mul ? AGX3_ENC_FLOAT2_MUL_ABS_EXTENDED : AGX3_ENC_FLOAT2_ABS_EXTENDED,
         mul ? agx3_pack_fmul_abs : agx3_pack_fadd_abs, f);
   } else if (a.neg || I->saturate) {
      emit(I, p, AGX3_ENC_FLOAT2_MODIFIER_EXTENDED,
           mul ? agx3_pack_fmul_mod : agx3_pack_fadd_mod, f);
   } else if (!b.neg && !hi && !is_uniform(a) && !is_uniform(b) && !I->wait &&
              low) {
      f.dstate = 0;
      emit(I, p, AGX3_ENC_FLOAT2_BASE,
           mul ? agx3_pack_fmul_base : agx3_pack_fadd_base, f);
   } else {
      emit(I, p, AGX3_ENC_FLOAT2_COMPACT, mul ? agx3_pack_fmul : agx3_pack_fadd,
           f);
   }
}

/* fmul into an export register, completing on its slot */
static void
pack_float2_export(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b) && !a.abs && !a.neg && !b.abs);

   struct agx3_operand y = src(I, b);
   y.neg = b.neg;

   emit(I, p, AGX3_ENC_FLOAT2_EXPORT, agx3_pack_fmul_export,
        (struct agx3_fields){
           .dest = agx3_encode_export(plain(I, I->dest[0])),
           .src = {agx3_encode_fsrc(src(I, a)), agx3_encode_fsrc(y)},
           .slot = completion_slot(I),
           .dstate = 1,
        });
}

/* fmul by 1/W, naming the coefficient */
static void
pack_float2_perspective(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b));
   pack_assert(I, !a.abs && !a.neg && !b.abs && !b.neg && !I->saturate);
   pack_assert(I, I->component < AGX3_NUM_COEFFICIENTS);

   emit(I, p, AGX3_ENC_FLOAT2_PROJECT, agx3_pack_fmul_cf,
        (struct agx3_fields){
           .dest = agx3_encode_dst(plain(I, I->dest[0])),
           .src = {agx3_encode_fsrc(src(I, a)), agx3_encode_fsrc(src(I, b))},
           .cf = I->component,
           .dstate = 1,
        });
}

/* FP32/FP16 conversion as fadd x, -0.0; narrowing writes one dest half */
static void
pack_float_convert(const agx_instr *I, struct a9_packed *p)
{
   agx_index dst = I->dest[0], s = I->src[0];
   pack_assert(I, !s.abs && !s.neg && !I->saturate);

   /* f32_to_f16 and f16_to_f32 name the register whose low half is FP16 */
   bool widen = I->op == AGX_OPCODE_F16_TO_F32 ||
                (I->op == AGX_OPCODE_FADD && dst.size == AGX_SIZE_32);

   struct agx3_operand d = plain(I, dst), x = src(I, s);
   if (widen)
      x = low_half(x);
   else
      d = low_half(d);

   struct agx3_fields f = {
      .dest = agx3_encode_dst(d),
      .src = {agx3_encode_fsrc(x), agx3_encode_fsrc(negzero)},
   };

   if (widen) {
      emit(I, p, AGX3_ENC_HALF_TO_FLOAT, agx3_pack_fadd, f);
   } else {
      emit(I, p, AGX3_ENC_FLOAT_TO_HALF, agx3_pack_fadd_f16, f);
   }
}

/* Two FP32 to FP16 conversions into the low and high destination halves */
static void
pack_half_pair(const agx_instr *I, struct a9_packed *p)
{
   struct agx3_operand d = low_half(plain(I, I->dest[0]));

   for (unsigned h = 0; h < 2; ++h) {
      agx_index s = I->src[h];
      pack_assert(I, is_gpr(s) && s.size == AGX_SIZE_32 && !s.abs && !s.neg);
      pack_assert(I, s.value != I->dest[0].value);

      /* The first conversion keeps a source the second reads */
      struct agx3_operand x = src(I, s);
      x.cache |= h == 0 && agx_is_equiv(I->src[0], I->src[1]);
      x.lu = !x.cache;

      d.value = I->dest[0].value + h;
      emit(I, p, AGX3_ENC_PACK_HALF_2X16, agx3_pack_fadd_f16,
           (struct agx3_fields){
              .dest = agx3_encode_dst(d),
              .src = {agx3_encode_fsrc(x), agx3_encode_fsrc(negzero)},
           });
   }
}

/* FP16 add, multiply and FMA on low halves, leaving the high half undefined */
static void
pack_half_alu(const agx_instr *I, struct a9_packed *p)
{
   bool fma = I->op == AGX_OPCODE_HFMA;
   struct agx3_operand s[3];

   pack_assert(I, !I->saturate && I->dest[0].size == AGX_SIZE_32);

   for (unsigned i = 0; i < (fma ? 3 : 2); ++i) {
      agx_index x = I->src[i];
      pack_assert(I, is_gpr(x) && x.size == AGX_SIZE_32 && !x.abs &&
                        (!x.neg || (I->op == AGX_OPCODE_HADD && i == 1)));

      s[i] = low_half(src(I, x));
      s[i].neg = x.neg;
   }

   struct agx3_fields f = {
      .dest = agx3_encode_dst(low_half(plain(I, I->dest[0]))),
      .src = {agx3_encode_fsrc(s[0]), agx3_encode_fsrc(s[1])},
   };

   if (fma) {
      f.src[0] = agx3_encode_fsrc3(s[0]);
      f.src[1] = agx3_encode_fsrc3(s[1]);
      f.src[2] = agx3_encode_fsrc3(s[2]);
      emit(I, p, AGX3_ENC_HALF3, agx3_pack_hfma, f);
   } else {
      emit(I, p, AGX3_ENC_HALF2,
           I->op == AGX_OPCODE_HMUL ? agx3_pack_hmul : agx3_pack_hadd, f);
   }
}

/* Operand of a fused multiply-add: a register, uniform or constant */
static struct agx3_operand
float3_source(const agx_instr *I, agx_index x, bool wide)
{
   struct agx3_operand op =
      is_imm(x) ? float_constant(I, imm_float_bits(x), wide) : src(I, x);
   op.abs = x.abs;

   /* FP16 sources: operand B in its low half only, the addend in either */
   if (is_gpr(x) && x.size == AGX_SIZE_16)
      op.size32 = false;

   return op;
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
   pack_assert(I, !is_gpr(b) || b.size != AGX_SIZE_16 || !(b.value & 1));

   bool half = (is_gpr(b) && b.size == AGX_SIZE_16) ||
               (is_gpr(c) && c.size == AGX_SIZE_16);
   bool neg_a = a.neg, neg_b = b.neg, neg_c = c.neg;

   /* The constant's own sign folds into its negation */
   if (is_imm(b) && !b.abs)
      neg_b ^= (b.value & 0x80) != 0;
   if (is_imm(c) && !c.abs)
      neg_c ^= (c.value & 0x80) != 0;

   struct agx3_operand x = float3_source(I, a, false);
   struct agx3_operand y = float3_source(I, b, false);
   struct agx3_operand z = float3_source(I, c, true);

   /* Bit 59 negates the product */
   x.neg = neg_a ^ neg_b;
   y.neg = false;
   z.neg = neg_c;

   struct agx3_fields f = {
      .dest = agx3_encode_dst(plain(I, I->dest[0])),
      .src = {agx3_encode_fsrc3(x), agx3_encode_fsrc3(y), agx3_encode_fsrc3(z)},
      .dstate = 1,
      .saturate = I->saturate,
   };

   bool low = I->dest[0].value < 128 && a.value < 128 && c.value < 128 &&
              (!is_gpr(b) || b.value < 128) &&
              (!is_uniform(b) || b.value < 128);

   if (!I->wait && !I->saturate && !a.abs && !b.abs && !c.abs && !neg_a &&
       !neg_b && !neg_c && !half && is_gpr(a) && is_gpr(c) && low) {
      emit(I, p, AGX3_ENC_FLOAT3_COMPACT, agx3_pack_ffma_compact, f);
   } else if (a.abs || b.abs) {
      emit(I, p, AGX3_ENC_FLOAT3_MODIFIER_EXTENDED, agx3_pack_ffma_mod, f);
   } else if (I->saturate) {
      emit(I, p, AGX3_ENC_FLOAT3_SATURATE_EXTENDED, agx3_pack_ffma_sat, f);
   } else {
      emit(I, p, AGX3_ENC_FLOAT3_EXTENDED, agx3_pack_ffma, f);
   }
}

/* Bit 33 clear writes the uniform file, sources being registers or zero */
static void
check_uniform_destination(const agx_instr *I)
{
   agx_index dst = I->dest[0];
   pack_assert(I, dst.size != AGX_SIZE_64 || !(dst.value & 3));

   for (unsigned s = 0; s < I->nr_srcs; ++s)
      pack_assert(I, is_gpr(I->src[s]) || agx_is_equiv(I->src[s], agx_zero()));
}

/* Integer add and multiply-add source: a register, uniform or 8-bit
 * constant. 16-bit registers sign-extend unless abs zero-extends them.
 */
static struct agx3_operand
int_source(const agx_instr *I, agx_index x)
{
   if (is_imm(x))
      return constant(x.value, false);

   struct agx3_operand op = src(I, x);
   op.sx = is_gpr(x) && x.size == AGX_SIZE_16 && !x.abs;
   return op;
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

   bool uniform = is_uniform(I->dest[0]);
   if (uniform)
      check_uniform_destination(I);

   emit(I, p, uniform ? AGX3_ENC_STORE_UNIFORM_ADD : AGX3_ENC_INT_ADD_EXTENDED,
        b.neg ? agx3_pack_isub : agx3_pack_iadd,
        (struct agx3_fields){
           .dest = agx3_encode_idst(plain(I, I->dest[0])),
           .src = {agx3_encode_isrc_desc(int_source(I, a)),
                   agx3_encode_isrc_desc(int_source(I, b))},
        });
}

/* 32x32 to 64-bit multiply; abs zero-extends both factors */
static void
pack_imul_wide(const agx_instr *I, struct a9_packed *p)
{
   agx_index a = I->src[0], b = I->src[1];
   pack_assert(I, is_gpr(a) && is_gpr(b) && a.abs == b.abs);
   pack_assert(I, agx_is_equiv(I->src[2], agx_zero()) && !I->shift);

   bool uniform = is_uniform(I->dest[0]);
   if (uniform)
      check_uniform_destination(I);

   emit(I, p, uniform ? AGX3_ENC_STORE_UNIFORM_MUL_WIDE : AGX3_ENC_INT_MUL_WIDE,
        a.abs ? agx3_pack_umul_wide : agx3_pack_imul_wide,
        (struct agx3_fields){
           .dest = agx3_encode_idst64(plain(I, I->dest[0])),
           .src = {agx3_encode_reg(src(I, a)), agx3_encode_reg(src(I, b))},
        });
}

/* Integer multiply-add. Constants go in factor B and the addend */
static void
pack_imad(const agx_instr *I, struct a9_packed *p)
{
   if (I->dest[0].size == AGX_SIZE_64) {
      pack_imul_wide(I, p);
      return;
   }

   agx_index s[3] = {I->src[0], I->src[1], I->src[2]};
   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && !I->shift);

   if (is_imm(s[0])) {
      agx_index t = s[0];
      s[0] = s[1];
      s[1] = t;
   }

   pack_assert(I, !is_imm(s[0]));

   for (unsigned i = 0; i < 3; ++i) {
      pack_assert(I, !s[i].neg);
      pack_assert(
         I, i == 0 || !is_gpr(s[i]) || s[i].size != AGX_SIZE_16 || s[i].abs);
   }

   bool uniform = is_uniform(I->dest[0]);
   if (uniform)
      check_uniform_destination(I);

   emit(I, p, uniform ? AGX3_ENC_STORE_UNIFORM_MAD : AGX3_ENC_INT_MAD_EXTENDED,
        agx3_pack_imad,
        (struct agx3_fields){
           .dest = agx3_encode_idst(plain(I, I->dest[0])),
           .src = {agx3_encode_isrc_desc(int_source(I, s[0])),
                   agx3_encode_isrc_desc(int_source(I, s[1])),
                   agx3_encode_isrc_desc(int_source(I, s[2]))},
        });
}

/* ishl is bfi into zero, ushr bfeil from zero, by a constant or register */
static void
pack_shift(const agx_instr *I, struct a9_packed *p)
{
   bool arithmetic = I->op == AGX_OPCODE_ASR;
   bool left = I->op == AGX_OPCODE_BFI;
   agx_index x = arithmetic ? I->src[0] : I->src[1];
   agx_index amount = arithmetic ? I->src[1] : I->src[2];
   bool variable = !is_imm(amount);
   unsigned width = arithmetic ? 0 : I->bfi_mask;

   if (!arithmetic)
      pack_assert(I, agx_is_equiv(I->src[0], agx_zero()));

   pack_assert(I, I->dest[0].size == AGX_SIZE_32 && x.size == AGX_SIZE_32);
   pack_assert(I, !left || width == 0);
   pack_assert(I, !variable || width == 0);

   struct agx3_fields f = {
      .dest = agx3_encode_reg(plain(I, I->dest[0])),
      .src = {agx3_encode_usrc(released(I, x))},
   };

   if (variable)
      f.src[1] = agx3_encode_usrc(released(I, amount));
   else
      f.amount = amount.value;

   if (arithmetic) {
      emit(I, p,
           variable ? AGX3_ENC_SHIFT_ARITH_REGISTER : AGX3_ENC_SHIFT_EXTENDED,
           variable ? agx3_pack_asr_reg : agx3_pack_asr, f);
   } else if (variable) {
      emit(I, p, AGX3_ENC_SHIFT_LOGICAL_REGISTER,
           left ? agx3_pack_shl_reg : agx3_pack_shr_reg, f);
   } else {
      f.width = width;
      emit(I, p, AGX3_ENC_SHIFT_LOGICAL_IMMEDIATE,
           left ? agx3_pack_shl : agx3_pack_shr, f);
   }
}

/* Integer and float conversions between 32-bit types */
static void
pack_convert(const agx_instr *I, struct a9_packed *p)
{
   enum agx_convert mode = I->src[0].value;
   agx_index x = I->src[1];
   uint64_t dest = agx3_encode_reg(plain(I, I->dest[0]));

   pack_assert(I, is_imm(I->src[0]));

   switch (mode) {
   case AGX_CONVERT_U32_TO_F:
   case AGX_CONVERT_S32_TO_F: {
      bool sign = mode == AGX_CONVERT_S32_TO_F;
      pack_assert(I, I->round == AGX_ROUND_RTE);

      emit(I, p, sign ? AGX3_ENC_SINT_TO_FLOAT : AGX3_ENC_UINT_TO_FLOAT,
           sign ? agx3_pack_s32_to_f : agx3_pack_u32_to_f,
           (struct agx3_fields){
              .dest = dest,
              .src = {agx3_encode_usrc(released(I, x))},
           });
      return;
   }

   case AGX_CONVERT_F_TO_U32:
   case AGX_CONVERT_F_TO_S32: {
      /* The signed format with an unsigned conversion saturates at 65535 */
      bool sign = mode == AGX_CONVERT_F_TO_S32;
      pack_assert(I, I->round == AGX_ROUND_RTZ);

      emit(I, p, sign ? AGX3_ENC_FLOAT_TO_SINT : AGX3_ENC_FLOAT_TO_UINT,
           sign ? agx3_pack_f_to_s32 : agx3_pack_f_to_u32,
           (struct agx3_fields){
              .dest = dest,
              .src = {agx3_encode_usrc(src(I, x))},
           });
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
   agx_index x = I->src[0];
   agx3_pack_fn pack;

   switch (I->op) {
   case AGX_OPCODE_POPCOUNT:
      /* A uniform source changes the destination format */
      pack = is_uniform(x) ? agx3_pack_popcount_uniform : agx3_pack_popcount;
      break;
   case AGX_OPCODE_FFS:
      pack = agx3_pack_ffs;
      break;
   case AGX_OPCODE_BITREV:
      pack = agx3_pack_bitrev;
      break;
   default:
      UNREACHABLE("not a bit operation");
   }

   emit(I, p, AGX3_ENC_BIT_UNARY, pack,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_usrc(src(I, x))},
        });
}

/* FP32 and FP16 special functions. The FP32 reciprocal has its own form for
 * a uniform source. Some functions have no FP16 form.
 */
static void
pack_special(const agx_instr *I, struct a9_packed *p)
{
   agx_index x = I->src[0];
   bool half = I->format == AGX_FORMAT_F16;
   agx3_pack_fn f32, f16;

   switch (I->op) {
   case AGX_OPCODE_RCP:
      f32 = is_uniform(x) ? agx3_pack_rcp_uniform : agx3_pack_rcp;
      f16 = agx3_pack_rcp_f16;
      break;
   case AGX_OPCODE_RSQRT:
      f32 = agx3_pack_rsqrt;
      f16 = agx3_pack_rsqrt_f16;
      break;
   case AGX_OPCODE_SRSQRT:
      f32 = agx3_pack_srsqrt;
      f16 = NULL;
      break;
   case AGX_OPCODE_SIN_FACTOR:
      f32 = agx3_pack_sin_factor;
      f16 = NULL;
      break;
   case AGX_OPCODE_EXP2:
      f32 = agx3_pack_exp2;
      f16 = agx3_pack_exp2_f16;
      break;
   case AGX_OPCODE_LOG2:
      f32 = agx3_pack_log2;
      f16 = agx3_pack_log2_f16;
      break;
   case AGX_OPCODE_FLOOR:
      f32 = agx3_pack_floor;
      f16 = agx3_pack_floor_f16;
      break;
   case AGX_OPCODE_CEIL:
      f32 = agx3_pack_ceil;
      f16 = agx3_pack_ceil_f16;
      break;
   case AGX_OPCODE_TRUNC:
      f32 = agx3_pack_trunc;
      f16 = agx3_pack_trunc_f16;
      break;
   case AGX_OPCODE_ROUNDEVEN:
      f32 = agx3_pack_rint;
      f16 = agx3_pack_rint_f16;
      break;
   default:
      UNREACHABLE("not a special function");
   }

   /* FP16 reads and writes low halves */
   pack_assert(I, !half || (is_gpr(x) && f16));

   struct agx3_operand op = released(I, x);
   op.neg = x.neg;
   op.abs = x.abs;

   emit(I, p,
        half            ? AGX3_ENC_HALF_SPECIAL
        : is_uniform(x) ? AGX3_ENC_FLOAT_SPECIAL_UNIFORM
                        : AGX3_ENC_FLOAT_SPECIAL,
        half ? f16 : f32,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_usrc(op)},
           .alu = I->result_read,
           .saturate = I->saturate,
        });
}

static void
pack_derivative(const agx_instr *I, struct a9_packed *p)
{
   agx_index x = I->src[0];
   pack_assert(I, !x.abs && !x.neg);

   /* The source is cached, never released */
   struct agx3_operand op = plain(I, x);
   op.cache = !x.discard;

   emit(I, p, AGX3_ENC_DERIVATIVE,
        I->op == AGX_OPCODE_DFDY ? agx3_pack_dfdy : agx3_pack_dfdx,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_reg(op)},
           .saturate = I->saturate,
        });
}

/* Condition of a compare-select, and whether its compared operands swap */
static bool
select_condition(const agx_instr *I, unsigned *cond, bool *swap)
{
   *swap = false;

   if (I->op == AGX_OPCODE_ICMPSEL) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         *cond = AGX3_ICOND_IEQ;
         return true;
      case AGX_ICOND_ULT:
         *cond = AGX3_ICOND_ULT;
         return true;
      case AGX_ICOND_UGT:
         *cond = AGX3_ICOND_UGT;
         return true;
      case AGX_ICOND_SLT:
         *cond = AGX3_ICOND_ILT;
         return true;
      case AGX_ICOND_SGT:
         *cond = AGX3_ICOND_IGT;
         return true;
      default:
         return false;
      }
   } else {
      switch (I->fcond) {
      case AGX_FCOND_EQ:
         *cond = AGX3_FCOND_FEQ;
         return true;
      case AGX_FCOND_LT:
         *cond = AGX3_FCOND_FLT;
         return true;
      case AGX_FCOND_GT:
         *cond = AGX3_FCOND_FGT;
         return true;
      case AGX_FCOND_GE:
         *cond = AGX3_FCOND_FGE;
         return true;
      case AGX_FCOND_LE:
         *cond = AGX3_FCOND_FGE;
         *swap = true;
         return true;
      default:
         return false;
      }
   }
}

/* min and max: a select between the compared operands themselves */
static agx3_pack_fn
minmax_select(const agx_instr *I)
{
   if (I->dest[0].size != AGX_SIZE_32 || I->op == AGX_OPCODE_HCMPSEL)
      return NULL;

   bool same =
      agx_is_equiv(I->src[0], I->src[2]) && agx_is_equiv(I->src[1], I->src[3]);
   bool swapped =
      agx_is_equiv(I->src[0], I->src[3]) && agx_is_equiv(I->src[1], I->src[2]);

   if (!same && !swapped)
      return NULL;

   if (I->op == AGX_OPCODE_ICMPSEL) {
      for (unsigned s = 0; s < 4; ++s) {
         /* Only the float form takes a constant */
         if (I->src[s].neg || I->src[s].abs || is_imm(I->src[s]) ||
             I->src[s].size != AGX_SIZE_32)
            return NULL;
      }

      switch (I->icond) {
      case AGX_ICOND_UGT:
         return same ? agx3_pack_umax : agx3_pack_umin;
      case AGX_ICOND_ULT:
         return same ? agx3_pack_umin : agx3_pack_umax;
      case AGX_ICOND_SGT:
         return same ? agx3_pack_imax : agx3_pack_imin;
      case AGX_ICOND_SLT:
         return same ? agx3_pack_imin : agx3_pack_imax;
      default:
         return NULL;
      }
   } else {
      /* Modifiers must match between the compared and selected operands */
      if (!agx_is_equiv_with_modifiers(I->src[0], I->src[same ? 2 : 3]) ||
          !agx_is_equiv_with_modifiers(I->src[1], I->src[same ? 3 : 2]) ||
          I->src[0].neg || I->src[0].abs || I->src[1].abs)
         return NULL;

      for (unsigned s = 0; s < 2; ++s) {
         if (!is_imm(I->src[s]) && I->src[s].size != AGX_SIZE_32)
            return NULL;
      }

      switch (I->fcond) {
      case AGX_FCOND_GTN:
         return same ? agx3_pack_fmax : agx3_pack_fmin;
      case AGX_FCOND_LTN:
         return same ? agx3_pack_fmin : agx3_pack_fmax;
      default:
         return NULL;
      }
   }
}

static void
pack_minmax(const agx_instr *I, agx3_pack_fn pack, struct a9_packed *p)
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
   pack_assert(I, !is_imm(b) || !b.neg);

   struct agx3_operand x = src(I, a);
   struct agx3_operand y =
      is_imm(b) ? float_constant(I, imm_float_bits(b), true) : src(I, b);
   if (!is_imm(b))
      y.neg = b.neg;

   /* FP16 operands are the low halves, without uniforms or negation */
   if (half) {
      pack_assert(I, is_gpr(a) && is_gpr(b) && !b.neg);
      x = low_half(x);
      y = low_half(y);
   }

   uint64_t (*encode)(struct agx3_operand) =
      floating ? agx3_encode_fsrc : agx3_encode_isrc;

   emit(I, p, half ? AGX3_ENC_HALF_MINMAX : AGX3_ENC_MINMAX_COMPACT, pack,
        (struct agx3_fields){
           .dest = agx3_encode_dst(plain(I, I->dest[0])),
           .src = {encode(x), encode(y)},
           .bit18 = !b.neg && !is_imm(b),
        });
}

/* Compare and select: dest = (A cond B) ? C : D */
static void
pack_select(const agx_instr *I, struct a9_packed *p)
{
   agx3_pack_fn minmax = minmax_select(I);
   if (minmax) {
      pack_minmax(I, minmax, p);
      return;
   }

   unsigned cond;
   bool swap;
   pack_assert(I, select_condition(I, &cond, &swap));

   agx_index s[4] = {I->src[0], I->src[1], I->src[2], I->src[3]};
   if (swap) {
      agx_index t = s[0];
      s[0] = s[1];
      s[1] = t;
   }

   bool half = I->op == AGX_OPCODE_HCMPSEL;
   bool floating = I->op == AGX_OPCODE_FCMPSEL || half;
   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, !is_imm(s[0]) && !s[0].neg);
   pack_assert(I, floating || !(s[0].abs || s[1].abs || s[1].neg));
   pack_assert(I, !(floating && is_imm(s[1]) && s[1].neg));

   bool modifiers = false;
   struct agx3_operand op[4];

   for (unsigned i = 0; i < 4; ++i) {
      pack_assert(I, is_imm(s[i]) || s[i].size == AGX_SIZE_32);
      modifiers |= s[i].abs || s[i].neg;

      if (!is_imm(s[i]))
         op[i] = src(I, s[i]);
      else if (i == 1 && floating)
         op[i] = float_constant(I, imm_float_bits(s[i]), true);
      else
         op[i] = constant(s[i].value, i != 1);

      /* The sign of a float constant is its own */
      if (!(i == 1 && floating && is_imm(s[i])))
         op[i].neg = s[i].neg;

      op[i].abs = s[i].abs;
   }

   pack_assert(I, !modifiers || floating);

   uint64_t (*encode)(struct agx3_operand) =
      floating ? agx3_encode_fsrc : agx3_encode_isrc;

   /* The FP16 select compares the low halves of operands A and B */
   if (half) {
      pack_assert(I, !modifiers && is_gpr(s[0]) && is_gpr(s[1]) &&
                        is_gpr(s[2]) && is_gpr(s[3]));
      op[0] = low_half(op[0]);
      op[1] = low_half(op[1]);
   }

   struct agx3_fields f = {
      .dest = agx3_encode_dst(plain(I, I->dest[0])),
      .src = {encode(op[0]), encode(op[1]), agx3_encode_isrc(op[2]),
              agx3_encode_isrc(op[3])},
      .cond = cond,
      .bit18 = !(s[2].abs || s[2].neg || s[3].abs || s[3].neg),
   };

   if (half)
      emit(I, p, AGX3_ENC_HALF_COMPARE_SELECT, agx3_pack_fcmpsel, f);
   else if (s[0].abs || s[1].abs)
      emit(I, p, AGX3_ENC_SELECT_MODIFIER_EXTENDED, agx3_pack_fcmpsel_mod, f);
   else
      emit(I, p, AGX3_ENC_SELECT_GPR_WIDE,
           floating ? agx3_pack_fcmpsel : agx3_pack_icmpsel, f);
}

/* Logic operations of truth tables, a move being an or with zero */
static unsigned
logic_op(uint32_t truth_table)
{
   switch (truth_table) {
   case 0x8:
      return AGX3_BITOP_AND;
   case 0xe:
   case 0xa:
   case 0xc:
      return AGX3_BITOP_OR;
   case 0x6:
      return AGX3_BITOP_XOR;
   default:
      return 0;
   }
}

/* Bitwise operations, into a GPR or an export register */
static void
pack_logic(const agx_instr *I, struct a9_packed *p)
{
   unsigned op = logic_op(I->truth_table);
   agx_index a = I->src[0], b = I->src[1];
   bool to_export = is_export(I->dest[0]);

   pack_assert(I, op != 0);

   /* Moves of operand A or B */
   if (I->truth_table == 0xa || I->truth_table == 0xc) {
      if (I->truth_table == 0xc)
         a = b;

      if (is_uniform(a) && !to_export && !I->wait && I->dest[0].value < 128) {
         emit(I, p, AGX3_ENC_UNIFORM_MOVE, agx3_pack_mov_uniform,
              (struct agx3_fields){
                 .dest = agx3_encode_dst(plain(I, I->dest[0])),
                 .src = {agx3_encode_uniform(plain(I, a))},
              });
         return;
      }

      /* An export of a register is an or with itself */
      b = to_export ? a : agx_zero();
   }

   if (is_imm(a) || (is_uniform(a) && is_gpr(b))) {
      agx_index t = a;
      a = b;
      b = t;
   }

   pack_assert(I, !is_imm(a) && a.size == AGX_SIZE_32);
   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, !to_export || (is_gpr(a) && is_gpr(b)));

   struct agx3_fields f = {
      .src = {agx3_encode_isrc(src(I, a)),
              agx3_encode_isrc(is_imm(b) ? constant(b.value, true)
                                         : src(I, b))},
      .op = op,
      .dstate = to_export || (is_gpr(b) && !b.discard),
   };

   if (to_export) {
      f.dest = agx3_encode_export(plain(I, I->dest[0]));
      f.slot = completion_slot(I);
      emit(I, p, AGX3_ENC_LOGIC_EXPORT, agx3_pack_bitop_export, f);
   } else {
      f.dest = agx3_encode_dst(plain(I, I->dest[0]));
      emit(I, p, AGX3_ENC_LOGIC_EXTENDED, agx3_pack_bitop, f);
   }
}

/* Export moves of the DEPTH_STORE pair, giving each lane one word */
static void
pack_publication_pair(const agx_instr *I, struct a9_packed *p)
{
   bool split = I->imm;
   struct agx3_operand dest = plain(I, I->dest[0]);
   pack_assert(I, is_export(I->dest[0]) && I->dest[0].value <= 124 &&
                     !(I->dest[0].value & 2));

   for (unsigned c = 0; c < (split ? 1 : 2); ++c) {
      agx_index s = I->src[c];
      pack_assert(I, is_gpr(s) && s.size == AGX_SIZE_32);

      /* The move keeps its source if a later move or instruction reads it */
      struct agx3_operand x = src(I, s);
      x.cache |= c == 0 && !split && agx_is_equiv(I->src[1], s);
      x.lu = !x.cache;

      dest.value = I->dest[0].value + 2 * c;
      emit(I, p, split ? AGX3_ENC_PUBLICATION_SPLIT : AGX3_ENC_PUBLICATION_PAIR,
           agx3_pack_bitop_export,
           (struct agx3_fields){
              .dest = agx3_encode_export(dest),
              .src = {agx3_encode_isrc(x), 0},
              .op = AGX3_BITOP_MOV,
              .slot = completion_slot(I),
              .per_lane = !split,
              .dstate = 1,
           });
   }
}

/* r0-r15 = 0-0x7f in two bytes, otherwise 32 bits into r0-r63 */
static void
pack_mov_imm(const agx_instr *I, struct a9_packed *p)
{
   agx_index dst = I->dest[0];
   pack_assert(I, dst.size == AGX_SIZE_32 && I->imm <= UINT32_MAX);

   struct agx3_fields f = {
      .dest = agx3_encode_dst(plain(I, dst)),
      .value = I->imm,
   };

   if (dst.value < 32 && I->imm <= 0x7f)
      emit(I, p, AGX3_ENC_MOV_IMM_COMPACT, agx3_pack_mov_imm, f);
   else
      emit(I, p, AGX3_ENC_MOV_IMM32, agx3_pack_mov_imm32, f);
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
      *out = (struct a9_sr){AGX3_SR_THREAD_POSITION_IN_GRID_X +
                            (sr - AGX_SR_THREAD_POSITION_IN_GRID_X)};
      return true;
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_X:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Y:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Z:
      *out = (struct a9_sr){AGX3_SR_THREADGROUP_POSITION_IN_GRID_X +
                            (sr - AGX_SR_THREADGROUP_POSITION_IN_GRID_X)};
      return true;
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
      *out = (struct a9_sr){AGX3_SR_THREAD_POSITION_IN_THREADGROUP_X +
                               (sr - AGX_SR_THREAD_POSITION_IN_THREADGROUP_X),
                            true};
      return true;
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
      *out = (struct a9_sr){AGX3_SR_THREADS_PER_THREADGROUP_X +
                               (sr - AGX_SR_THREADS_PER_THREADGROUP_X),
                            true};
      return true;
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
      *out = (struct a9_sr){AGX3_SR_THREAD_INDEX_IN_THREADGROUP, true};
      return true;
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
      *out = (struct a9_sr){AGX3_SR_THREAD_INDEX_IN_SUBGROUP, true};
      return true;
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
      *out = (struct a9_sr){AGX3_SR_SUBGROUP_INDEX_IN_THREADGROUP, true};
      return true;
   case AGX_SR_COVERAGE_MASK:
      *out = (struct a9_sr){AGX3_SR_COVERAGE_MASK};
      return true;
   case AGX_SR_BACKFACING:
      *out = (struct a9_sr){AGX3_SR_BACKFACING};
      return true;
   case AGX_SR_LAYER_ID:
      *out = (struct a9_sr){AGX3_SR_LAYER_ID};
      return true;
   case AGX_SR_INSTANCE_ID:
      *out = (struct a9_sr){AGX3_SR_INSTANCE_ID};
      return true;
   case AGX_SR_VERTEX_ID:
      *out = (struct a9_sr){AGX3_SR_VERTEX_ID};
      return true;
   default:
      return false;
   }
}

bool
agx3_sr_supported(enum agx_sr sr)
{
   struct a9_sr out;
   return sr_selector(sr, &out);
}

static void
pack_get_sr(const agx_instr *I, struct a9_packed *p)
{
   struct a9_sr sr;
   pack_assert(I, sr_selector(I->sr, &sr));
   pack_assert(I, I->dest[0].size == AGX_SIZE_32);

   struct agx3_fields f = {
      .dest = agx3_encode_dst(plain(I, I->dest[0])),
      .sr = sr.selector,
   };

   if (sr.zext16) {
      emit(I, p, AGX3_ENC_GET_SR_ZEXT16, agx3_pack_get_sr16, f);
      emit(I, p, AGX3_ENC_GET_SR_ZEXT16, agx3_pack_zext16,
           (struct agx3_fields){.dest = f.dest});
      return;
   }

   /* Coverage, vertex and instance IDs are asynchronous, in one layout */
   emit(I, p,
        I->sr == AGX_SR_COVERAGE_MASK ? AGX3_ENC_GET_COVERAGE
        : I->sr == AGX_SR_VERTEX_ID || I->sr == AGX_SR_INSTANCE_ID
           ? AGX3_ENC_GET_DRAW_ID
           : AGX3_ENC_GET_SR,
        agx3_pack_get_sr, f);
}

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

/* Format of a device memory access: elements of 8, 16 or 32 bits, or
 * vectors of 32-bit elements
 */
static unsigned
memory_format(const agx_instr *I, unsigned components)
{
   static const uint8_t vectors[] = {
      AGX3_MEMFORMAT_I32,
      AGX3_MEMFORMAT_I32X2,
      AGX3_MEMFORMAT_I32X3,
      AGX3_MEMFORMAT_I32X4,
   };

   pack_assert(I, components == 1 || I->format == AGX_FORMAT_I32);

   if (I->format == AGX_FORMAT_I8)
      return AGX3_MEMFORMAT_I8;
   else if (I->format == AGX_FORMAT_I16)
      return AGX3_MEMFORMAT_I16;
   else
      return vectors[components - 1];
}

/* Shift field: the element shift plus one, or 0 for 4 */
static unsigned
memory_shift_field(const agx_instr *I, unsigned shift)
{
   pack_assert(I, shift <= 4);
   return shift == 4 ? 0 : shift + 1;
}

/* Base of a device memory access: a register pair or a 64-bit uniform */
static struct agx3_operand
memory_base(const agx_instr *I, agx_index base, bool release)
{
   pack_assert(I,
               base.size == AGX_SIZE_64 && (is_gpr(base) || is_uniform(base)));
   return release ? released(I, base) : plain(I, base);
}

/* Device loads of base + (index << shift) + offset, zero-extended */
static void
pack_device_load(const agx_instr *I, struct a9_packed *p)
{
   agx_index base = I->src[0], index = I->src[1];
   unsigned components = memory_components(I);
   unsigned shift = memory_shift(I);
   bool pointer = is_imm(index);
   bool indirect = is_gpr(base);

   pack_assert(I, I->dest[0].size == AGX_SIZE_32);
   pack_assert(I, !pointer || indirect);
   pack_assert(I, pointer || index.size == AGX_SIZE_32);

   /* Without an index, a register base is a pointer */
   struct agx3_operand none = {.kind = AGX3_KIND_NONE};
   int offset = (pointer ? (int)(index.value << shift) : 0) + I->byte_offset;
   pack_assert(I, offset >= INT16_MIN && offset <= INT16_MAX);

   emit(I, p,
        pointer    ? AGX3_ENC_DEVICE_LOAD_POINTER
        : indirect ? AGX3_ENC_DEVICE_LOAD_INDIRECT
                   : AGX3_ENC_DEVICE_LOAD,
        agx3_pack_device_load,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_membase(memory_base(I, base, true)),
                   agx3_encode_memindex(pointer ? none : released(I, index))},
           .offset = offset,
           .shift = memory_shift_field(I, pointer ? 0 : shift),
           .format = memory_format(I, components),
           .slot = completion_slot(I),
           .next = I->load_follows,
        });
}

/* Device stores. The index register is released, the base is not. */
static void
pack_device_store(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], base = I->src[1], index = I->src[2];
   unsigned components = memory_components(I);
   unsigned shift = memory_shift(I);

   pack_assert(I, value.size == AGX_SIZE_32 && index.size == AGX_SIZE_32);
   pack_assert(I, I->byte_offset >= INT16_MIN && I->byte_offset <= INT16_MAX);

   emit(I, p,
        is_gpr(base) ? AGX3_ENC_DEVICE_STORE_INDIRECT : AGX3_ENC_DEVICE_STORE,
        agx3_pack_device_store,
        (struct agx3_fields){
           .src = {agx3_encode_reg(plain(I, value)),
                   agx3_encode_membase(memory_base(I, base, false)),
                   agx3_encode_reg(released(I, index))},
           .offset = I->byte_offset,
           .shift = memory_shift_field(I, shift),
           .format = memory_format(I, components),
        });
}

static unsigned
atomic_op(const agx_instr *I)
{
   switch (I->atomic_opc) {
   case AGX_ATOMIC_OPC_ADD:
      return AGX3_ATOMIC_ADD;
   case AGX_ATOMIC_OPC_AND:
      return AGX3_ATOMIC_AND;
   case AGX_ATOMIC_OPC_CMPXCHG:
      return AGX3_ATOMIC_CMPXCHG;
   case AGX_ATOMIC_OPC_FADD:
      return AGX3_ATOMIC_FADD;
   case AGX_ATOMIC_OPC_IMAX:
      return AGX3_ATOMIC_IMAX;
   case AGX_ATOMIC_OPC_IMIN:
      return AGX3_ATOMIC_IMIN;
   case AGX_ATOMIC_OPC_OR:
      return AGX3_ATOMIC_OR;
   case AGX_ATOMIC_OPC_SUB:
      return AGX3_ATOMIC_SUB;
   case AGX_ATOMIC_OPC_UMAX:
      return AGX3_ATOMIC_UMAX;
   case AGX_ATOMIC_OPC_UMIN:
      return AGX3_ATOMIC_UMIN;
   case AGX_ATOMIC_OPC_XCHG:
      return AGX3_ATOMIC_XCHG;
   case AGX_ATOMIC_OPC_XOR:
      return AGX3_ATOMIC_XOR;
   }

   UNREACHABLE("invalid atomic");
}

/* Device atomics. A returning one leaves the old value in private word 0,
 * which the next instruction loads.
 */
static void
pack_device_atomic(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], base = I->src[1], index = I->src[2];
   bool indirect = is_gpr(base);
   bool returning = !agx_is_null(I->dest[0]);

   pack_assert(I, index.size == AGX_SIZE_32 &&
                     agx_channels(value) ==
                        (I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG ? 2 : 1));

   struct agx3_operand b = memory_base(I, base, false);

   emit(I, p,
        returning ? (indirect ? AGX3_ENC_DEVICE_ATOMIC_INDIRECT_RETURN
                              : AGX3_ENC_DEVICE_ATOMIC_RETURN)
                  : (indirect ? AGX3_ENC_DEVICE_ATOMIC_INDIRECT
                              : AGX3_ENC_DEVICE_ATOMIC),
        indirect ? agx3_pack_atomic_indirect : agx3_pack_atomic,
        (struct agx3_fields){
           .src = {agx3_encode_reg(plain(I, value)),
                   indirect ? agx3_encode_reg64(b) : agx3_encode_uniform64(b),
                   agx3_encode_reg(plain(I, index))},
           .op = atomic_op(I),
           .ret = returning ? AGX3_ATOMIC_RETURN_RET : AGX3_ATOMIC_RETURN_NORET,
        });

   if (returning) {
      emit(I, p, p->form, agx3_pack_atomic_result,
           (struct agx3_fields){
              .dest = agx3_encode_dst(plain(I, I->dest[0])),
              .slot = completion_slot(I),
           });
   }
}

/* Register whose low half holds a threadgroup memory word index */
static struct agx3_operand
local_index(const agx_instr *I, agx_index index)
{
   pack_assert(I, !(index.value & 1));
   struct agx3_operand op = plain(I, index);
   op.size32 = true;
   return op;
}

/* Threadgroup memory: an immediate root in words and a word index */
static unsigned
local_root(const agx_instr *I, agx_index base)
{
   pack_assert(I, is_imm(base) && base.value < 64);
   return base.value * 4;
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
   pack_assert(I, data.value < 128);

   /* The index register is named by its low half */
   struct agx3_operand idx = low_half(local_index(I, index));
   idx.lu = index.discard;

   struct agx3_fields f = {
      .src = {agx3_encode_half(idx)},
      .root = local_root(I, base),
   };

   if (load) {
      f.dest = agx3_encode_reg(plain(I, data));
      emit(I, p, AGX3_ENC_SHARED_LOAD, agx3_pack_local_load, f);
   } else {
      f.src[1] = f.src[0];
      f.src[0] = agx3_encode_reg(plain(I, data));
      emit(I, p, AGX3_ENC_SHARED_STORE, agx3_pack_local_store, f);
   }
}

static void
pack_local_atomic(const agx_instr *I, struct a9_packed *p)
{
   bool cmpxchg = I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG;

   pack_assert(I, !agx_is_null(I->dest[0]));
   pack_assert(I, agx_channels(I->src[0]) == (cmpxchg ? 2 : 1));

   struct agx3_operand data = plain(I, I->src[0]);
   pack_assert(I, I->dest[0].value < 128 && I->src[2].value < 128 &&
                     data.value + (cmpxchg ? 2 : 0) < 128);

   emit(I, p, AGX3_ENC_SHARED_ATOMIC, agx3_pack_local_atomic,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_reg(data),
                   agx3_encode_reg(local_index(I, I->src[2]))},
           .root = local_root(I, I->src[1]),
           .op = atomic_op(I),
        });
}

/* Private memory at a word index register. Stores destroy their data */
static void
pack_private(const agx_instr *I, struct a9_packed *p)
{
   pack_assert(I, I->dest[0].value < 128 && I->src[0].value < 128 &&
                     (I->nr_srcs < 2 || I->src[1].value < 128));

   if (I->op == AGX_OPCODE_STACK_LOAD) {
      emit(I, p, AGX3_ENC_PRIVATE_LOAD, agx3_pack_stack_load_indexed,
           (struct agx3_fields){
              .dest = agx3_encode_reg(plain(I, I->dest[0])),
              .src = {agx3_encode_reg(plain(I, I->src[0]))},
           });
   } else {
      emit(I, p, AGX3_ENC_PRIVATE_STORE, agx3_pack_stack_store_indexed,
           (struct agx3_fields){
              .src = {agx3_encode_reg(plain(I, I->src[0])),
                      agx3_encode_reg(plain(I, I->src[1]))},
           });
   }
}

/* Spills: fills reach the first 1024 words, saves the first 2048 */
static void
pack_stack(const agx_instr *I, struct a9_packed *p)
{
   bool load = I->op == AGX_OPCODE_STACK_LOAD;
   agx_index offset = I->src[load ? 0 : 1];

   pack_assert(I, I->format == AGX_FORMAT_I32 && I->mask == 1);

   if (!is_imm(offset)) {
      pack_private(I, p);
      return;
   }

   pack_assert(I, (offset.value & 3) == 0);
   unsigned word = offset.value / 4;

   if (load) {
      pack_assert(I, word < AGX3_FILL_WORD_COUNT);
      emit(I, p, AGX3_ENC_SPILL_LOAD, agx3_pack_stack_load,
           (struct agx3_fields){
              .dest = agx3_encode_dst(plain(I, I->dest[0])),
              .word = word,
              .slot = completion_slot(I),
           });
   } else {
      pack_assert(I, word < AGX3_SAVE_WORD_COUNT);
      emit(I, p, AGX3_ENC_SPILL_STORE, agx3_pack_stack_store,
           (struct agx3_fields){
              .src = {agx3_encode_reg(plain(I, I->src[0]))},
              .word = word,
              .slot = completion_slot(I),
           });
   }
}

/* uniform[word] = register, an integer add of zero into the uniform file */
static void
pack_uniform_store(const agx_instr *I, struct a9_packed *p)
{
   agx_index value = I->src[0], index = I->src[1];
   pack_assert(I, value.size == AGX_SIZE_32 && agx_channels(value) == 1);
   pack_assert(I, is_imm(index) && index.value < AGX3_UNIFORM_COUNT * 2);

   struct agx3_operand dest = {
      .kind = AGX3_KIND_UNIFORM,
      .value = index.value,
      .size32 = true,
   };

   emit(I, p, AGX3_ENC_STORE_UNIFORM, agx3_pack_iadd,
        (struct agx3_fields){
           .dest = agx3_encode_idst(dest),
           .src = {agx3_encode_isrc_desc(src(I, value)),
                   agx3_encode_isrc_desc(constant(0, true))},
        });
}

/* Destination of the subgroup operations, which reach r63 */
static uint64_t
subgroup_dest(const agx_instr *I)
{
   pack_assert(I, I->dest[0].value < 128);
   return agx3_encode_reg(plain(I, I->dest[0]));
}

static void
pack_subgroup(const agx_instr *I, struct a9_packed *p)
{
   agx_index x = I->src[0];

   switch (I->op) {
   case AGX_OPCODE_SIMD_PREFIX:
      /* Exclusive scan. The source stays live. */
      pack_assert(I, I->simd_op == AGX_SIMD_OP_IADD);
      emit(I, p, AGX3_ENC_SUBGROUP_SCAN_IADD, agx3_pack_simd_prefix_iadd,
           (struct agx3_fields){
              .dest = subgroup_dest(I),
              .src = {agx3_encode_reg(plain(I, x))},
           });
      return;

   case AGX_OPCODE_SIMD_REDUCE: {
      /* Operands name 16-bit halves, so 32-bit ones are even */
      bool wide = x.size == AGX_SIZE_32;
      pack_assert(I, I->simd_op == AGX_SIMD_OP_UMAX);
      pack_assert(I, is_gpr(x) && is_gpr(I->dest[0]));
      pack_assert(I, x.size == I->dest[0].size && x.size <= AGX_SIZE_32);
      pack_assert(I, I->dest[0].value < 128 && x.value < 128);

      struct agx3_operand d = plain(I, I->dest[0]), s = plain(I, x);
      d.cache = I->dest[0].cache;
      s.cache = x.cache;
      s.lu = x.discard;

      emit(I, p, AGX3_ENC_SUBGROUP_REDUCE_UMAX,
           wide ? agx3_pack_simd_reduce_umax32 : agx3_pack_simd_reduce_umax,
           (struct agx3_fields){
              .dest = wide ? agx3_encode_reg_half(d) : agx3_encode_half(d),
              .src = {wide ? agx3_encode_reg_half(s) : agx3_encode_half(s)},
           });
      return;
   }

   case AGX_OPCODE_SHUFFLE: {
      /* Read the lane in the index's low half. Both sources stay live */
      agx_index lane = I->src[1];
      struct agx3_operand l = plain(I, lane);
      pack_assert(I, !(lane.value & 1));
      l.size32 = true;

      emit(I, p, AGX3_ENC_SUBGROUP_BROADCAST, agx3_pack_shuffle,
           (struct agx3_fields){
              .dest = subgroup_dest(I),
              .src = {agx3_encode_reg(plain(I, x)), agx3_encode_reg(l)},
           });
      return;
   }

   case AGX_OPCODE_BALLOT:
      /* Mask of active lanes with a nonzero source. Destroys the source. */
      emit(I, p, AGX3_ENC_SUBGROUP_BALLOT, agx3_pack_ballot,
           (struct agx3_fields){
              .dest = subgroup_dest(I),
              .src = {agx3_encode_reg(plain(I, x))},
           });
      return;

   default:
      UNREACHABLE("not a subgroup operation");
   }
}

/* Condition of a predicate compare, and whether it needs the extended form */
static bool
pred_condition(const agx_instr *I, unsigned *cond, bool *extended)
{
   *extended = false;

   if (I->op == AGX_OPCODE_ICMP_PRED) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         *cond = AGX3_IPRED_EXT_IEQ;
         *extended = true;
         return true;
      case AGX_ICOND_ULT:
         *cond = AGX3_IPRED_ULT;
         return true;
      case AGX_ICOND_UGT:
         *cond = AGX3_IPRED_UGT;
         return true;
      case AGX_ICOND_SLT:
         *cond = AGX3_IPRED_ILT;
         return true;
      case AGX_ICOND_SGT:
         *cond = AGX3_IPRED_IGT;
         return true;
      default:
         return false;
      }
   } else {
      switch (I->fcond) {
      case AGX_FCOND_LT:
         *cond = AGX3_FPRED_FLT;
         return true;
      case AGX_FCOND_GT:
         *cond = AGX3_FPRED_FGT;
         return true;
      case AGX_FCOND_EQ:
         *cond = AGX3_FPRED_EXT_FEQ;
         *extended = true;
         return true;
      case AGX_FCOND_GE:
         *cond = AGX3_FPRED_EXT_FGE;
         *extended = true;
         return true;
      case AGX_FCOND_LE:
         *cond = AGX3_FPRED_EXT_FLE;
         *extended = true;
         return true;
      default:
         return false;
      }
   }
}

/* Compare into a predicate, released registers keeping no cache bit */
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

   struct agx3_operand x = released(I, a), y;

   if (!is_imm(b))
      y = released(I, b);
   else if (floating)
      y = float_constant(I, imm_float_bits(b), true);
   else
      y = constant(b.value, false);

   /* FP16 operands are the low halves */
   if (half) {
      x = low_half(x);
      y = low_half(y);
   }

   uint64_t (*encode)(struct agx3_operand) =
      floating ? agx3_encode_fsrc : agx3_encode_isrc;

   struct agx3_fields f = {
      .src = {encode(x), encode(y)},
      .pred = I->pred,
      .cond = cond,
      .invert = I->invert_cond,
   };

   if (extended) {
      emit(I, p,
           half ? AGX3_ENC_HALF_PREDICATE_EXTENDED
                : AGX3_ENC_PREDICATE_COMPARE_EXTENDED,
           floating ? agx3_pack_fcmp_pred_ext : agx3_pack_icmp_pred_ext, f);
   } else {
      emit(I, p,
           half ? AGX3_ENC_HALF_PREDICATE_SHORT
                : AGX3_ENC_PREDICATE_COMPARE_SHORT,
           floating ? agx3_pack_fcmp_pred : agx3_pack_icmp_pred, f);
   }
}

/* Signed, instruction-relative branch, packed again once its target is
 * known
 */
static void
pack_jump(const agx_instr *I, struct a9_packed *p)
{
   if (I->op == AGX_OPCODE_JMP_EXEC_ANY)
      emit(I, p, AGX3_ENC_JMP_EXEC_ANY, agx3_pack_jmp_exec_any,
           (struct agx3_fields){0});
   else
      emit(I, p, AGX3_ENC_JMP_EXEC_NONE, agx3_pack_jmp_exec_none,
           (struct agx3_fields){0});
}

static void
pack_control(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_IF_PRED:
      /* Predicates 6 and 7 are the constants true and false */
      emit(I, p, AGX3_ENC_EXEC_MASK_PUSH, agx3_pack_if_pred,
           (struct agx3_fields){
              .pred = I->pred,
              .invert = I->invert_cond,
           });
      return;

   case AGX_OPCODE_ELSE_EXEC:
      emit(I, p, AGX3_ENC_EXEC_MASK_ELSE, agx3_pack_else,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_POP_EXEC:
      pack_assert(I, I->nest == 1);
      emit(I, p, AGX3_ENC_EXEC_MASK_POP, agx3_pack_pop_exec,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_BEGIN_LOOP:
      emit(I, p, AGX3_ENC_LOOP_MASK_PUSH, agx3_pack_begin_loop,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_WHILE_PRED:
      /* Metal's loops set bit 29 */
      pack_assert(I, I->pred < 6);
      emit(I, p, AGX3_ENC_LOOP_MASK_UPDATE, agx3_pack_while_pred,
           (struct agx3_fields){
              .pred = I->pred,
              .bit29 = !I->invert_cond,
           });
      return;

   case AGX_OPCODE_END_LOOP:
      emit(I, p, AGX3_ENC_LOOP_MASK_POP, agx3_pack_end_loop,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_BREAK_PRED:
      /* A loop break unwinds two scopes plus one per enclosing if */
      pack_assert(I, I->nest >= 2);
      pack_assert(I, I->pred >= 1 && I->pred < 6);
      emit(I, p, AGX3_ENC_BREAK_MASK_UNWIND, agx3_pack_break_pred,
           (struct agx3_fields){
              .nest = I->nest,
              .pred = I->pred,
           });
      return;

   default:
      UNREACHABLE("not a control flow operation");
   }
}

static void
pack_sync(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_THREADGROUP_BARRIER:
      emit(I, p, AGX3_ENC_WORKGROUP_BARRIER, agx3_pack_threadgroup_barrier,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_MEMORY_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER_ACQUIRE:
      /* Acquire only differs in what it waits for beforehand */
      emit(I, p, AGX3_ENC_DEVICE_FENCE, agx3_pack_memory_barrier,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_RASTER_ORDER_FENCE:
      emit(I, p, AGX3_ENC_DEVICE_FENCE, agx3_pack_raster_order_fence,
           (struct agx3_fields){0});
      return;

   case AGX_OPCODE_STOP:
      emit(I, p, AGX3_ENC_HALT, agx3_pack_stop, (struct agx3_fields){0});
      return;

   case AGX_OPCODE_WAIT:
   case AGX_OPCODE_WAIT_SLOTS: {
      unsigned mask =
         I->op == AGX_OPCODE_WAIT ? BITFIELD_BIT(I->scoreboard) : I->imm;
      pack_assert(I, mask != 0);
      emit(I, p, AGX3_ENC_WAIT, agx3_pack_wait,
           (struct agx3_fields){.slots = mask});
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

   struct agx3_fields f = {
      .dest = agx3_encode_reg(plain(I, I->dest[0])),
   };

   for (unsigned s = 0; s < (rgba ? 4 : 3); ++s)
      f.src[s] = agx3_encode_reg(src(I, I->src[s]));

   emit(I, p, rgba ? AGX3_ENC_PACK_FORMAT : AGX3_ENC_PACK_FORMAT_RGB,
        rgba ? agx3_pack_pack_rgb10a2 : agx3_pack_pack_rg11b10f, f);
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
a9_tex_dim(enum agx_dim dim)
{
   switch (dim) {
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
texture_gradient(const struct agx3_texture *d)
{
   return d->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
          d->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN;
}

static unsigned
texture_params(const struct agx3_texture *d)
{
   if (texture_gradient(d))
      return d->dim == AGX_DIM_3D ? 16 : 8;
   else if (!d->load && d->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
            (d->dim == AGX_DIM_2D) && !d->shadow)
      return 2;
   else
      return 4;
}

/* Words of the parameter bundle a texture operation reads */
unsigned
agx3_texture_params(const agx_instr *I)
{
   struct agx3_texture d = agx3_texture_of(I);
   return texture_params(&d);
}

/* Properties of a texture operation that select its mode */
struct a9_texture {
   bool gradient, lod, fetch, bias, min_lod, ms, gather;
   enum a9_tex_dim dim;
   unsigned words;
};

static struct a9_texture
a9_texture(const struct agx3_texture *d)
{
   struct a9_texture t = {
      .gradient = texture_gradient(d),
      .words = texture_params(d),
      .ms = d->dim == AGX_DIM_2D_MS || d->dim == AGX_DIM_2D_MS_ARRAY,
      .bias = d->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
              d->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN,
      .min_lod = d->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS_MIN ||
                 d->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN,
      .gather = d->gather != AGX_GATHER_NONE,
      .dim = a9_tex_dim(d->dim),
   };

   t.lod = !t.gradient && t.words == 4;
   t.fetch = d->load && !t.ms;
   return t;
}

/* Sample mode without an offset, if measured */
static bool
texture_mode(const struct agx3_texture *d, const struct a9_texture *t,
             uint16_t *out)
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
      if (!t->lod || d->shadow || t->bias || t->dim == A9_TEX_CUBE)
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

   if (d->shadow)
      mode |= 0x20;

   if (t->ms) {
      /* Integer X and Y, then a packed layer and sample word */
      if (!d->load || !t->lod || t->bias)
         return false;

      mode = t->dim == A9_TEX_2D_ARRAY ? 0x14a0 : 0x1480;
   }

   if (d->offset && !texture_offset_mode(mode, &mode))
      return false;

   *out = mode;
   return true;
}

/* Whether a texture operation has a measured encoding */
bool
agx3_texture_desc_encodable(const struct agx3_texture *d)
{
   struct a9_texture t = a9_texture(d);
   uint16_t mode;

   if (d->query_lod && (t.gradient || t.ms || d->shadow || t.fetch ||
                        (d->mask & ~3u) || d->offset))
      return false;

   if (t.gather && (t.gradient || t.ms || t.fetch || d->query_lod))
      return false;

   if (d->offset && (t.fetch || t.ms))
      return false;

   return texture_mode(d, &t, &mode);
}

bool
agx3_texture_encodable(const agx_instr *I)
{
   struct agx3_texture d = agx3_texture_of(I);
   return agx3_texture_desc_encodable(&d);
}

/* Texture or sampler index: a constant or a register */
static struct agx3_operand
texture_index(const agx_instr *I, agx_index x)
{
   if (is_imm(x))
      return constant(x.value, false);

   return released(I, x);
}

static void
pack_texture(const agx_instr *I, struct a9_packed *p)
{
   agx_index params = I->src[0], texture = I->src[3], sampler = I->src[4];
   struct agx3_texture d = agx3_texture_of(I);
   struct a9_texture t = a9_texture(&d);
   bool register_texture = is_gpr(texture);
   bool register_sampler = is_gpr(sampler);
   unsigned mask = t.gather ? 0xf : I->mask;
   unsigned comps = util_bitcount(mask);
   uint16_t mode;

   pack_assert(I, agx3_texture_encodable(I));
   pack_assert(I, texture_mode(&d, &t, &mode));
   pack_assert(I, is_export(params) && agx_channels(params) == t.words);
   pack_assert(I, !I->sparse && mask != 0);
   pack_assert(I, agx_channels(I->dest[0]) == comps);
   pack_assert(I, register_texture || is_imm(texture));
   pack_assert(I, register_sampler || is_imm(sampler));
   pack_assert(I, !register_sampler || register_texture);
   pack_assert(I, t.bias || !t.lod || I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
                     I->lod_mode == AGX_LOD_MODE_AUTO_LOD);

   pack_assert(I, I->dest[0].value + 2 * comps <=
                     (register_texture ? AGX3_GPR_COUNT * 2 : 128));

   unsigned coord = params.value / 2;
   pack_assert(I, coord <= (t.gradient ? 32 - t.words : t.lod ? 28 : 30));
   pack_assert(I, (coord & (t.words == 2 ? 1 : 3)) == 0);

   /* Gather names its component as a one-component mask would */
   if (t.gather)
      mask = 1u << (I->gather >> 1);

   /* Permuted result selector; results pack in component order */
   static const uint8_t selectors[16] = {
      0, 0, 1, 5, 2, 8, 9, 6, 3, 4, 10, 13, 11, 12, 14, 7,
   };

   struct agx3_fields f = {
      .dest = agx3_encode_reg(plain(I, I->dest[0])),
      .src = {agx3_encode_export(plain(I, params)),
              agx3_encode_table(texture_index(I, texture)),
              agx3_encode_table(texture_index(I, sampler))},
      .mode = mode,
      .select = selectors[mask],
      .slot = completion_slot(I),
      .slot2 = completion_slot(I),
      .gather = t.gather,
      .filter = !(t.fetch || t.ms || I->shadow || t.gather || I->query_lod),
      .query_lod = I->query_lod,
   };

   /* Texel offsets are signed 4-bit constants */
   if (I->offset) {
      agx_index offset = I->src[5];
      pack_assert(I, is_imm(offset) && offset.value < (1 << 12));

      f.x = util_sign_extend(offset.value & 15, 4);
      f.y = util_sign_extend((offset.value >> 4) & 15, 4);
      f.z = util_sign_extend((offset.value >> 8) & 15, 4);
   }

   if (!register_texture) {
      emit(I, p,
           t.gradient ? AGX3_ENC_TEXTURE_GRAD
           : t.lod    ? AGX3_ENC_TEXTURE_LOD
                      : AGX3_ENC_TEXTURE_SAMPLE,
           agx3_pack_texture, f);
      return;
   }

   /* The 20-byte forms read the texture, and the sampler, from registers */
   emit(I, p,
        t.gradient ? AGX3_ENC_TEXTURE_GRAD_INDEXED
        : t.lod    ? AGX3_ENC_TEXTURE_LOD_INDEXED
                   : AGX3_ENC_TEXTURE_SAMPLE_INDEXED,
        register_sampler ? agx3_pack_texture_reg2 : agx3_pack_texture_reg, f);
}

/* Normalized unpacking of two unorm/snorm 8/16-bit values into a pair */
static void
pack_unpack_norm(const agx_instr *I, struct a9_packed *p)
{
   static const uint8_t modes[] = {
      [AGX_FORMAT_S16NORM] = AGX3_UNPACK_S16NORM,
      [AGX_FORMAT_S8NORM] = AGX3_UNPACK_S8NORM,
      [AGX_FORMAT_U16NORM] = AGX3_UNPACK_U16NORM,
      [AGX_FORMAT_U8NORM] = AGX3_UNPACK_U8NORM,
   };
   bool narrow =
      I->format == AGX_FORMAT_U8NORM || I->format == AGX_FORMAT_S8NORM;

   pack_assert(I, I->format < ARRAY_SIZE(modes) && modes[I->format]);
   pack_assert(I, agx_channels(I->dest[0]) == 2 && !(I->dest[0].value & 2));
   pack_assert(I, narrow || !I->imm);

   /* The destination is a register pair */
   struct agx3_operand d = plain(I, I->dest[0]);
   d.size32 = false;
   d.size64 = true;

   emit(I, p, AGX3_ENC_UNPACK_NORM, agx3_pack_unpack_norm,
        (struct agx3_fields){
           .dest = agx3_encode_reg64(d),
           .src = {agx3_encode_reg(src(I, I->src[0]))},
           .mode = modes[I->format],
           .hi = I->imm,
           .wide = !narrow,
        });
}

/* unorm packing of FP32 pairs; 8-bit writes one half per pair */
static void
pack_pack_unorm(const agx_instr *I, struct a9_packed *p)
{
   bool bytes = I->op == AGX_OPCODE_PACK_UNORM_4X8;
   uint64_t dest = agx3_encode_reg(plain(I, I->dest[0]));

   for (unsigned pair = 0; pair < (bytes ? 2 : 1); ++pair) {
      agx_index a = I->src[2 * pair], b = I->src[2 * pair + 1];
      pack_assert(I, !bytes || (a.value != I->dest[0].value &&
                                b.value != I->dest[0].value));

      struct agx3_fields f = {
         .dest = dest,
         .src = {agx3_encode_reg(plain(I, a)), agx3_encode_reg(plain(I, b))},
      };

      if (bytes) {
         f.hi = pair;
         emit(I, p, AGX3_ENC_PACK_UNORM_4X8, agx3_pack_pack_unorm_4x8, f);
      } else {
         emit(I, p, AGX3_ENC_PACK_UNORM_2X16, agx3_pack_pack_unorm_2x16, f);
      }
   }
}

/* Cube face selection: major axis, then the face in the next low half */
static void
pack_cube(const agx_instr *I, struct a9_packed *p)
{
   pack_assert(I, I->imm <= 2);
   pack_assert(I, agx_channels(I->dest[0]) == (I->imm == 0 ? 2 : 1));

   emit(I, p, AGX3_ENC_CUBE, agx3_pack_cube,
        (struct agx3_fields){
           .dest = agx3_encode_reg(plain(I, I->dest[0])),
           .src = {agx3_encode_reg(src(I, I->src[0])),
                   agx3_encode_reg(src(I, I->src[1])),
                   agx3_encode_reg(src(I, I->src[2]))},
           .mode = I->imm,
        });
}

/* PBE write of 32-bit data, then the join of its parameter export */
static void
pack_image_write(const agx_instr *I, struct a9_packed *p)
{
   agx_index params = I->src[0], image = I->src[4];
   pack_assert(I, is_export(params) && agx_channels(params) == 8 &&
                     params.value <= 48 && !(params.value & 15));
   pack_assert(I, agx_channels(image) == 2 && !(image.value & 2));
   pack_assert(I, I->dim == AGX_DIM_2D);

   /* The image descriptor is a register pair */
   struct agx3_operand descriptor = released(I, image);
   descriptor.size32 = false;
   descriptor.size64 = true;

   emit(I, p, AGX3_ENC_IMAGE_WRITE, agx3_pack_image_write,
        (struct agx3_fields){
           .src = {agx3_encode_export(plain(I, params)),
                   agx3_encode_reg64(descriptor)},
           .floating = !I->integer,
        });
   emit(I, p, AGX3_ENC_IMAGE_WRITE, agx3_pack_join_export,
        (struct agx3_fields){0});
}

/* Block image store of the tile, then the join of its parameter export */
static void
pack_block_image_store(const agx_instr *I, struct a9_packed *p)
{
   agx_index params = I->src[0];
   bool ms = I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY;
   bool array = I->dim == AGX_DIM_2D_ARRAY || I->dim == AGX_DIM_2D_MS_ARRAY;
   enum agx3_encoding form = (array || ms) ? AGX3_ENC_BLOCK_IMAGE_STORE_EXTENDED
                                           : AGX3_ENC_BLOCK_IMAGE_STORE;

   pack_assert(I, is_export(params) && agx_channels(params) == 4 &&
                     params.value <= 24 && !(params.value & 7));

   emit(I, p, form, agx3_pack_block_image_store,
        (struct agx3_fields){
           .src = {agx3_encode_export(plain(I, params))},
           .image = I->src[1].value,
           .mode = array ? (ms ? 0x028 : 0x528)
                   : ms  ? 0x228
                         : 0x5a8,
           .format = I->format,
        });
   emit(I, p, form, agx3_pack_join_export, (struct agx3_fields){0});
}

/* Fragment outputs and tile access */
static void
pack_fragment(const agx_instr *I, struct a9_packed *p)
{
   switch (I->op) {
   case AGX_OPCODE_WRITE_COVERAGE:
      pack_assert(I, I->src[0].value < 128);
      emit(I, p, AGX3_ENC_COVERAGE, agx3_pack_write_coverage,
           (struct agx3_fields){
              .src = {agx3_encode_export(plain(I, I->src[0]))},
           });
      return;

   case AGX_OPCODE_ZS_STORE: {
      static const uint8_t modes[4] = {
         [1] = AGX3_DEPTH_DEPTH,
         [2] = AGX3_DEPTH_STENCIL,
         [3] = AGX3_DEPTH_DEPTH_STENCIL,
      };
      agx_index zs = I->src[0];
      pack_assert(I, is_export(zs) && agx_channels(zs) == 2);
      pack_assert(
         I, I->zs >= 1 && I->zs <= 3 && !(zs.value & 3) && zs.value < 126);

      emit(I, p, AGX3_ENC_DEPTH_STORE, agx3_pack_zs_store,
           (struct agx3_fields){
              .src = {agx3_encode_export(plain(I, zs))},
              .mode = modes[I->zs],
           });
      return;
   }

   case AGX_OPCODE_BEGIN_TILE:
   case AGX_OPCODE_END_TILE: {
      /* Only the measured controls are known to work */
      bool begin = I->op == AGX_OPCODE_BEGIN_TILE;
      uint64_t c = I->imm;
      pack_assert(I, begin ? (c == AGX_TILE_DOMAIN_PIXELS ||
                              c == AGX_TILE_DOMAIN_IMAGES ||
                              c == AGX_TILE_DOMAIN_COORDS ||
                              (c & ~0xfcull) == 0x800 || c == AGX_TILE_ZS)
                           : (c == 0x300 || (c & ~0xfcull) == 0 ||
                              (c & ~0xfcull) == 0x200 || c == AGX_TILE_ZS));

      emit(I, p, begin ? AGX3_ENC_TILE_ACCESS : AGX3_ENC_TILE_FENCE,
           begin ? agx3_pack_begin_tile : agx3_pack_end_tile,
           (struct agx3_fields){.control = c});
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
      pack_assert(I, !coords || load);
      pack_assert(I, coords ? (!dynamic && mask.value >= 1 && mask.value <= 15)
                            : (dynamic || mask.value == 1));
      pack_assert(I, !dynamic || (mask.size <= AGX_SIZE_32 && is_gpr(mask)));

      /* A register mask is named by its 16-bit half */
      uint64_t value = agx3_encode_reg(plain(I, data));
      uint64_t dynamic_mask =
         dynamic ? agx3_encode_half(low_half(released(I, mask))) : 0;
      struct agx3_fields f = {.offset = I->pixel_offset};

      if (!load) {
         f.src[0] = value;
         f.src[1] = dynamic_mask;
         emit(I, p, dynamic ? AGX3_ENC_TILE_STORE_MASK : AGX3_ENC_TILE_STORE,
              dynamic ? agx3_pack_st_tile_mask : agx3_pack_st_tile, f);
         return;
      }

      f.dest = value;
      f.slot = completion_slot(I);

      if (coords) {
         f.src[0] = agx3_encode_reg(plain(I, I->src[1]));
         f.mask = mask.value;
         emit(I, p, AGX3_ENC_TILE_LOAD_COORDS, agx3_pack_ld_tile_coords, f);
      } else if (dynamic) {
         f.src[0] = dynamic_mask;
         emit(I, p, AGX3_ENC_TILE_LOAD_MASK, agx3_pack_ld_tile_mask, f);
      } else {
         emit(I, p, AGX3_ENC_TILE_LOAD, agx3_pack_ld_tile, f);
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
   case AGX_OPCODE_ST_VARY:
      /* Commits an export register to a varying slot */
      pack_assert(I, is_imm(I->src[0]) && I->src[0].value < 138);
      emit(I, p, AGX3_ENC_VARY_STORE, agx3_pack_st_vary,
           (struct agx3_fields){
              .src = {agx3_encode_export(plain(I, I->src[1]))},
              .slot = I->src[0].value,
           });
      return;

   case AGX_OPCODE_ITER: {
      /* One coefficient, at the centre or a position. Coefficient 0 is 1/W */
      agx_index cf = I->src[0];
      bool coord = I->interpolation != AGX_INTERPOLATION_CENTER;
      pack_assert(I, is_imm(cf) && I->channels == 1);
      pack_assert(I, !coord || I->interpolation == AGX_INTERPOLATION_SAMPLE ||
                        I->interpolation == AGX_INTERPOLATION_CENTROID ||
                        I->interpolation == AGX_INTERPOLATION_SAMPLE_REGISTER);

      struct agx3_fields f = {
         .dest = agx3_encode_reg(plain(I, I->dest[0])),
         .cf = cf.value,
      };

      if (coord) {
         f.src[0] = agx3_encode_reg(src(I, I->src[1]));
         emit(I, p, AGX3_ENC_ITER_COORD, agx3_pack_iter_coord, f);
      } else {
         emit(I, p, AGX3_ENC_ITER, agx3_pack_iter, f);
      }
      return;
   }

   case AGX_OPCODE_LDCF:
      /* Reads the three-word coefficient tuple, completing on its slot */
      pack_assert(I, is_imm(I->src[0]) && I->channels == 3);
      pack_assert(I, agx_channels(I->dest[0]) == 3);

      emit(I, p, AGX3_ENC_ITER_FLAT, agx3_pack_ldcf,
           (struct agx3_fields){
              .dest = agx3_encode_reg(plain(I, I->dest[0])),
              .cf = I->src[0].value,
              .slot = completion_slot(I),
           });
      return;

   case AGX_OPCODE_INTERPOLATION_POSITION: {
      /* Sample position or centroid; destroys the source register */
      bool sample = I->interpolation == AGX_INTERPOLATION_SAMPLE ||
                    I->interpolation == AGX_INTERPOLATION_SAMPLE_REGISTER;
      pack_assert(I, sample || I->interpolation == AGX_INTERPOLATION_CENTROID);

      emit(I, p, AGX3_ENC_INTERPOLATION_POSITION,
           agx3_pack_interpolation_position,
           (struct agx3_fields){
              .dest = agx3_encode_reg(plain(I, I->dest[0])),
              .src = {agx3_encode_reg(plain(I, I->src[0]))},
              .sample = sample,
           });
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
pack_body(const agx_instr *I, struct a9_packed *p)
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
      pack_special(I, p);
      return;

   case AGX_OPCODE_DFDX:
   case AGX_OPCODE_DFDY:
      pack_derivative(I, p);
      return;

   case AGX_OPCODE_HMIN:
   case AGX_OPCODE_HMAX:
      pack_minmax(I, I->op == AGX_OPCODE_HMIN ? agx3_pack_hmin : agx3_pack_hmax,
                  p);
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
      pack_device_load(I, p);
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
      return;
   }
}

/* Zero-byte instructions */
static bool
is_empty(const agx_instr *I)
{
   return I->op == AGX_OPCODE_EXPORT;
}

enum agx3_encoding
agx3_encoding_for(const agx_instr *I)
{
   struct a9_packed p = {.form = AGX3_ENC_PSEUDO, .form_only = true};
   pack_body(I, &p);
   return p.form;
}

unsigned
agx3_pack_instr(const agx_instr *I, uint8_t *bytes)
{
   if (is_empty(I))
      return 0;

   struct a9_packed p = {.form = AGX3_ENC_PSEUDO};

   pack_body(I, &p);
   pack_assert(I, p.form != AGX3_ENC_PSEUDO && "no AGX3 encoding");
   memcpy(bytes, p.bytes, p.length);
   return p.length;
}

struct a9_fixup {
   off_t offset;
   agx_block *block;
   bool skip_to_end;
   agx3_pack_fn pack;
};

void
agx3_pack_binary(agx_context *ctx, struct util_dynarray *emission)
{
   struct util_dynarray fixups;
   util_dynarray_init(&fixups, ctx);

   agx_foreach_block(ctx, block) {
      block->offset = emission->size;

      agx_foreach_instr_in_block(block, I) {
         block->last_offset = emission->size;

         if (I->op == AGX_OPCODE_JMP_EXEC_ANY ||
             I->op == AGX_OPCODE_JMP_EXEC_NONE ||
             I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER) {
            struct a9_fixup fixup = {
               .offset = emission->size,
               .block = I->target,
               .skip_to_end = I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER,
               .pack = I->op == AGX_OPCODE_JMP_EXEC_ANY
                          ? agx3_pack_jmp_exec_any
                          : agx3_pack_jmp_exec_none,
            };
            util_dynarray_append(&fixups, fixup);
         }

         /* The linker relocates the slots of varying commits */
         if (I->op == AGX_OPCODE_ST_VARY) {
            assert(I->src[0].value < AGX_MAX_UVS_SLOTS &&
                   !ctx->out->vary_stores[I->src[0].value]);
            ctx->out->vary_stores[I->src[0].value] = emission->size + 1;
         }

         uint8_t bytes[A9_MAX_BYTES];
         unsigned length = agx3_pack_instr(I, bytes);
         memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
      }
   }

   /* Branch displacements are relative to the branch */
   util_dynarray_foreach(&fixups, struct a9_fixup, fixup) {
      off_t target =
         fixup->skip_to_end ? fixup->block->last_offset : fixup->block->offset;
      struct agx3_fields f = {
         .offset = (int64_t)target - (int64_t)fixup->offset,
      };

      ASSERTED unsigned length =
         fixup->pack((uint8_t *)emission->data + fixup->offset, &f);
      assert(length != 0);
   }

   util_dynarray_fini(&fixups);
}
