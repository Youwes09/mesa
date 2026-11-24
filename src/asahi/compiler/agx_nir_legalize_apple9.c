/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"

/* Apple9 has no 8/16-bit integer ALU, and only FP16 add, multiply, fma and
 * special functions. Narrow values have undefined high bits; other operations
 * run at 32 bits.
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

/* Spread the low 16 bits into alternate bits, as Apple9 has no interleave */
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
agx_nir_legalize_apple9_memory(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_narrow_word_memory,
                                     nir_metadata_control_flow, NULL);
}

bool
agx_nir_legalize_apple9_alu(nir_shader *nir)
{
   bool progress = nir_shader_alu_pass(nir, lower_half_conversion,
                                       nir_metadata_control_flow, NULL);
   progress |= nir_shader_alu_pass(nir, lower_interleave,
                                   nir_metadata_control_flow, NULL);
   progress |= nir_lower_bit_size(nir, alu_width, NULL);
   return progress;
}
