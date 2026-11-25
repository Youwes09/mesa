/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"

/* Lowering to word registers and word memory */

/* The word ALU has no 8/16-bit integer arithmetic, and only FP16 add,
 * multiply, fma and special functions. Narrow values have undefined high bits;
 * other operations run at 32 bits.
 */
static unsigned
alu_width(const nir_instr *instr, UNUSED void *data)
{
   if (instr->type == nir_instr_type_intrinsic) {
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      /* Cross-lane arithmetic and comparisons need defined high bits */
      switch (intr->intrinsic) {
      case nir_intrinsic_reduce:
      case nir_intrinsic_inclusive_scan:
      case nir_intrinsic_exclusive_scan:
         return intr->def.bit_size > 1 && intr->def.bit_size < 32 ? 32 : 0;
      case nir_intrinsic_vote_feq:
      case nir_intrinsic_vote_ieq: {
         unsigned bits = intr->src[0].ssa->bit_size;
         return bits > 1 && bits < 32 ? 32 : 0;
      }
      default:
         return 0;
      }
   }

   if (instr->type != nir_instr_type_alu)
      return 0;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];

   if (info->is_conversion)
      return 0;

   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fmin:
   case nir_op_fmax:
      return 0;

   /* FP16 special functions */
   case nir_op_frcp:
   case nir_op_frsq:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
      return 0;

   /* FP16 compares select whole registers */
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fneu:
      return 0;

   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_bcsel:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_inot:
   case nir_op_iadd:
   case nir_op_isub:
   case nir_op_ineg:
   case nir_op_imul:
      return 0;

   /* Splitting and packing are handled at selection */
   case nir_op_pack_32_2x16_split:
   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y:
   case nir_op_pack_64_2x32_split:
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y:
      return 0;

   default:
      break;
   }

   if (alu->def.bit_size > 1 && alu->def.bit_size < 32)
      return 32;

   for (unsigned i = 0; i < info->num_inputs; ++i) {
      unsigned bits = alu->src[i].src.ssa->bit_size;
      if (bits > 1 && bits < 32)
         return 32;
   }

   return 0;
}

/* Spread the low 16 bits into alternate bits, as word registers have no interleave */
static nir_def *
spread_bits(nir_builder *b, nir_def *x)
{
   static const unsigned shifts[] = {8, 4, 2, 1};
   static const uint32_t masks[] = {0x00ff00ff, 0x0f0f0f0f, 0x33333333,
                                    0x55555555};

   x = nir_iand_imm(b, nir_u2u32(b, x), 0xffff);

   for (unsigned i = 0; i < ARRAY_SIZE(shifts); ++i)
      x = nir_iand_imm(b, nir_ior(b, x, nir_ishl_imm(b, x, shifts[i])),
                       masks[i]);

   return x;
}

static bool
lower_interleave(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_interleave)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = spread_bits(b, nir_ssa_for_alu_src(b, alu, 0));
   nir_def *y = spread_bits(b, nir_ssa_for_alu_src(b, alu, 1));
   nir_def_replace(&alu->def, nir_ior(b, x, nir_ishl_imm(b, y, 1)));
   return true;
}

/* Narrow integer to FP16 conversions go through FP32 */
static bool
lower_half_conversion(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_u2f16 && alu->op != nir_op_i2f16)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *wide = alu->op == nir_op_u2f16 ? nir_u2f32(b, x) : nir_i2f32(b, x);
   nir_def_replace(&alu->def, nir_f2f16(b, wide));
   return true;
}

/* Threadgroup and private memory are word-addressed. Narrow threadgroup stores
 * use atomic AND and OR so as not to race, private ones read-modify-write.
 */
static bool
lower_narrow_word_memory(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   bool scratch, store;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
      scratch = false, store = false;
      break;
   case nir_intrinsic_store_shared:
      scratch = false, store = true;
      break;
   case nir_intrinsic_load_scratch:
      scratch = true, store = false;
      break;
   case nir_intrinsic_store_scratch:
      scratch = true, store = true;
      break;
   default:
      return false;
   }

   unsigned bits = store ? intr->src[0].ssa->bit_size : intr->def.bit_size;
   if (bits >= 32)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *base = nir_u2u32(b, nir_get_io_offset_src(intr)->ssa);
   if (!scratch)
      base = nir_iadd_imm(b, base, nir_intrinsic_base(intr));

   unsigned comps =
      store ? intr->src[0].ssa->num_components : intr->def.num_components;
   nir_def *values[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0; c < comps; ++c) {
      if (store && !(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
         continue;

      nir_def *offset = nir_iadd_imm(b, base, c * bits / 8);
      nir_def *word = nir_iand_imm(b, offset, ~3u);
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, offset, 3), 8);

      if (store) {
         nir_def *mask =
            nir_ishl(b, nir_imm_int(b, BITFIELD_MASK(bits)), shift);
         nir_def *x = nir_u2u32(b, nir_channel(b, intr->src[0].ssa, c));
         nir_def *value =
            nir_ishl(b, nir_iand_imm(b, x, BITFIELD_MASK(bits)), shift);

         if (scratch) {
            nir_def *old = nir_load_scratch(b, 1, 32, word, .align_mul = 4);
            nir_def *new =
               nir_ior(b, nir_iand(b, old, nir_inot(b, mask)), value);
            nir_store_scratch(b, new, word, .align_mul = 4);
         } else {
            nir_shared_atomic(b, 32, word, nir_inot(b, mask),
                              .atomic_op = nir_atomic_op_iand);
            nir_shared_atomic(b, 32, word, value,
                              .atomic_op = nir_atomic_op_ior);
         }
      } else {
         nir_def *loaded = scratch
                              ? nir_load_scratch(b, 1, 32, word, .align_mul = 4)
                              : nir_load_shared(b, 1, 32, word, .align_mul = 4);

         values[c] = nir_u2uN(b, nir_ushr(b, loaded, shift), bits);
      }
   }

   if (store)
      nir_instr_remove(&intr->instr);
   else
      nir_def_replace(&intr->def, nir_vec(b, values, comps));

   return true;
}

bool
agx_nir_lower_word_memory(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_narrow_word_memory,
                                     nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_word_alu(nir_shader *nir)
{
   bool progress = nir_shader_alu_pass(nir, lower_half_conversion,
                                       nir_metadata_control_flow, NULL);
   progress |= nir_shader_alu_pass(nir, lower_interleave,
                                   nir_metadata_control_flow, NULL);
   progress |= nir_lower_bit_size(nir, alu_width, NULL);
   return progress;
}

/* Bounded-accuracy floating-point reduction, like the existing AGX backend.
 * Vulkan requires 2^-11 absolute error only inside [-pi, pi], in every stage,
 * so phase accuracy may degrade as |x| grows. The sine factor consumes
 * quarter-turn units, so fold four quadrants into [-1, 1] before evaluating
 * sin(pi*p/2)/p. Work on the magnitude to avoid cancellation for small
 * negative sine inputs. */
nir_def *
agx_nir_sincos_fp32(nir_builder *b, nir_def *x, bool cosine)
{
   nir_def *turns = nir_fmul_imm(b, nir_fabs(b, x), 0x1.45f306p-3f);
   nir_def *q = nir_fmul_imm(b, nir_fsub(b, turns, nir_ffloor(b, turns)), 4.0f);
   if (cosine) {
      q = nir_fadd_imm(b, q, 1.0f);
      q = nir_bcsel(b, nir_fge_imm(b, q, 4.0f), nir_fadd_imm(b, q, -4.0f), q);
   }
   nir_def *p = nir_bcsel(b, nir_flt_imm(b, q, 1.0f), q,
                          nir_bcsel(b, nir_flt_imm(b, q, 3.0f),
                                    nir_fsub(b, nir_imm_float(b, 2.0f), q),
                                    nir_fadd_imm(b, q, -4.0f)));
   nir_def *result = nir_fmul(b, p, nir_fsin_factor_agx(b, p));
   return cosine ? result : nir_ixor(b, result, nir_iand_imm(b, x, 0x80000000));
}

static bool
math_filter(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op == nir_op_f2u8 || alu->op == nir_op_f2u16 ||
       alu->op == nir_op_f2i8 || alu->op == nir_op_f2i16)
      return true;
   if (alu->op == nir_op_b2f16 || alu->op == nir_op_f2f16_rtz ||
       (alu->op == nir_op_ldexp && alu->def.bit_size == 16))
      return true;
   if ((alu->op == nir_op_fsin || alu->op == nir_op_fcos) &&
       alu->def.bit_size == 16)
      return true;
   return alu->def.bit_size == 32 &&
          (alu->op == nir_op_iabs || alu->op == nir_op_fsin ||
           alu->op == nir_op_fcos);
}

/* Round to nearest, then step one ulp toward zero if that rounded away. */
static nir_def *
lower_f2f16_rtz(nir_builder *b, nir_def *x)
{
   /* Keep algebraic optimization from folding the round trip. */
   uint32_t ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   x = nir_f2f32(b, x);
   nir_def *h = nir_f2f16(b, x);
   nir_def *away = nir_flt(b, nir_fabs(b, x), nir_fabs(b, nir_f2f32(b, h)));
   nir_def *bits = nir_u2u16(b, nir_iadd_imm(b, nir_u2u32(b, h), -1));
   nir_def *result = nir_bcsel(b, away, bits, h);
   b->fp_math_ctrl = ctrl;
   return result;
}

static nir_def *
lower_math(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *source = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; ++i) {
      nir_def *x = nir_channel(b, source, i);
      if (alu->op == nir_op_f2u8 || alu->op == nir_op_f2u16) {
         /* Narrow integer results come from the FP32 conversion. */
         components[i] =
            nir_u2uN(b, nir_f2u32(b, nir_f2fN(b, x, 32)), alu->def.bit_size);
      } else if (alu->op == nir_op_f2i8 || alu->op == nir_op_f2i16) {
         components[i] =
            nir_i2iN(b, nir_f2i32(b, nir_f2fN(b, x, 32)), alu->def.bit_size);
      } else if (alu->op == nir_op_b2f16) {
         components[i] =
            nir_bcsel(b, x, nir_imm_float16(b, 1.0f), nir_imm_float16(b, 0.0f));
      } else if (alu->op == nir_op_ldexp) {
         /* The generic binary16 expansion builds narrow exponent fields. */
         nir_def *exp = nir_channel(
            b, nir_mov_alu(b, alu->src[1], alu->def.num_components), i);
         components[i] = nir_f2f16(b, nir_ldexp(b, nir_f2f32(b, x), exp));
      } else if (alu->op == nir_op_f2f16_rtz) {
         components[i] = lower_f2f16_rtz(b, x);
      } else if (alu->def.bit_size == 16) {
         /* Binary16 sine and cosine use the FP32 reduction. */
         nir_def *wide = nir_f2f32(b, x);
         wide = agx_nir_sincos_fp32(b, wide, alu->op == nir_op_fcos);
         components[i] = nir_f2f16(b, wide);
      } else if (alu->op == nir_op_iabs) {
         components[i] = nir_imax(b, x, nir_ineg(b, x));
      } else {
         components[i] = agx_nir_sincos_fp32(b, x, alu->op == nir_op_fcos);
      }
   }
   return nir_vec(b, components, alu->def.num_components);
}

/* T8132 F2I32 and F2U32 saturate to the destination range, convert NaN to
 * zero, and F2U32 converts negative values to zero (exact-bit probes with
 * +-2^31, +-2^32, +-1e30, +-inf and NaNs). Range checks that rebuild these
 * results around a conversion, such as DXVK's saturating ftoi/ftou, are
 * therefore redundant. NIR leaves out-of-range conversions undefined, so
 * fold them only here, at the backend boundary, and only for non-constant
 * inputs that constant folding cannot evaluate differently. */
static bool
float_const(nir_scalar s, float *value)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_const(s) || s.def->bit_size != 32)
      return false;
   *value = uif(nir_scalar_as_uint(s));
   return true;
}

/* cond selects k exactly when the saturating conversion of x yields k. */
static bool
conversion_select_redundant(nir_scalar cond, nir_scalar x, uint32_t k,
                            bool sign)
{
   if (!nir_scalar_is_alu(cond))
      return false;
   nir_op op = nir_scalar_alu_op(cond);
   /* Only these per-component comparisons qualify; other conditions (such
    * as vector comparisons with sized sources) cannot be chased per scalar. */
   if (op != nir_op_fneu && op != nir_op_flt && op != nir_op_fge)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   float c;
   if (op == nir_op_fneu && nir_scalar_equal(a, x) && nir_scalar_equal(b, x))
      return k == 0;
   /* x > c, with every float above c at or past the saturation bound. */
   if ((op == nir_op_flt && nir_scalar_equal(b, x) && float_const(a, &c)) ||
       (op == nir_op_fge && nir_scalar_equal(a, x) && float_const(b, &c) &&
        (c = nextafterf(c, -INFINITY), true))) {
      return sign ? (k == 0x7fffffff && c >= 2147483520.0f)
                  : (k == 0xffffffff && c >= 4294967040.0f);
   }
   /* x < c, with every float below c at or past the lower bound. */
   if ((op == nir_op_flt && nir_scalar_equal(a, x) && float_const(b, &c)) ||
       (op == nir_op_fge && nir_scalar_equal(b, x) && float_const(a, &c) &&
        (c = nextafterf(c, INFINITY), true))) {
      return sign ? (k == 0x80000000 && c <= -2147483648.0f)
                  : (k == 0 && c <= 0.0f);
   }
   return false;
}

/* The conversion at the core of a chain of redundant range selects. */
static bool
saturating_conversion(nir_scalar s, nir_scalar *x, bool *sign)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_f2i32 || op == nir_op_f2u32) {
      if (nir_scalar_chase_alu_src(s, 0).def->bit_size != 32)
         return false;
      *x = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      *sign = op == nir_op_f2i32;
      return !nir_scalar_is_const(*x);
   }
   if (op != nir_op_bcsel)
      return false;
   uint32_t k;
   nir_scalar sel = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (!nir_scalar_is_const(sel))
      return false;
   k = nir_scalar_as_uint(sel);
   return saturating_conversion(nir_scalar_chase_alu_src(s, 2), x, sign) &&
          conversion_select_redundant(nir_scalar_chase_alu_src(s, 0), *x, k,
                                      *sign);
}

static bool
fold_saturating_conversion(nir_builder *b, nir_alu_instr *alu,
                           UNUSED void *data)
{
   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;
   if (alu->op == nir_op_f2u32) {
      /* F2U32 already maps negative values and NaN to zero. */
      nir_scalar src =
         nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
      if (!nir_scalar_is_alu(src) || nir_scalar_alu_op(src) != nir_op_fmax ||
          src.def->bit_size != 32)
         return false;
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar zero =
            nir_scalar_chase_movs(nir_scalar_chase_alu_src(src, i));
         nir_scalar other = nir_scalar_chase_alu_src(src, 1 - i);
         if (nir_scalar_is_const(zero) &&
             (nir_scalar_as_uint(zero) & 0x7fffffff) == 0 &&
             !nir_scalar_is_const(nir_scalar_chase_movs(other))) {
            b->cursor = nir_before_instr(&alu->instr);
            nir_src_rewrite(&alu->src[0].src, nir_mov_scalar(b, other));
            alu->src[0].swizzle[0] = 0;
            return true;
         }
      }
      return false;
   }
   if (alu->op != nir_op_bcsel)
      return false;
   nir_scalar x;
   bool sign;
   nir_scalar self = nir_get_scalar(&alu->def, 0);
   if (!saturating_conversion(self, &x, &sign))
      return false;
   nir_scalar inner = nir_scalar_chase_alu_src(self, 2);
   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, nir_mov_scalar(b, inner));
   return true;
}

bool
agx_nir_lower_word_math(nir_shader *shader)
{
   bool progress = false;
   if (agx_nir_target(shader)->flushes_denorms)
      progress |= agx_nir_flush_subgroup_denorms(shader);

   progress |= nir_shader_alu_pass(shader, fold_saturating_conversion,
                                   nir_metadata_control_flow, NULL);
   progress |=
      nir_shader_lower_instructions(shader, math_filter, lower_math, NULL);
   /* There is multiply/add/FMA but no interpolation instruction. Use NIR's
    * precision-aware lowering for FP32 mix. */
   progress |= nir_lower_flrp(shader, 16 | 32, false);
   if (progress) {
      nir_opt_constant_folding(shader);
      nir_opt_copy_prop(shader);
      nir_opt_dce(shader);
   }
   return progress;
}
