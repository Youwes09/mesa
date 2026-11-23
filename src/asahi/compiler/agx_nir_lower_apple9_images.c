/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "asahi/layout/layout.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include "util/set.h"
#include "agx_apple9_storage.h"
#include "agx_nir_texture.h"
#include "libagx.h"

/* Fetch through the view's linear 2D entry, but load RGB32 as three words */
static bool
apple9_lower_texel_buffer(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF ||
       (tex->op != nir_texop_txf && tex->op != nir_texop_txs))
      return false;
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle_idx < 0)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *ptr =
      nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
   ptr = nir_iadd_imm(b, ptr,
                      offsetof(struct agx_apple9_texel_descriptor, metadata));
   nir_def *count = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, ptr,
                   offsetof(struct agx_apple9_texel_metadata, elements)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   if (tex->op == nir_texop_txs) {
      nir_def_replace(&tex->def, count);
      return true;
   }

   int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   nir_push_if(b, nir_ult(b, coord, count));
   nir_def *base = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, ptr, offsetof(struct agx_apple9_texel_metadata, base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *size = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, ptr, offsetof(struct agx_apple9_texel_metadata, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_push_if(b, nir_ieq_imm(b, size, 12));
   nir_def *xyz = nir_load_global(
      b, 3, 32, nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, coord, 12))),
      .align_mul = 4);
   nir_def *rgb32 = nir_vec4(
      b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1), nir_channel(b, xyz, 2),
      floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1));
   nir_push_else(b, NULL);
   /* The view starts at base rounded down to 16 bytes; sizes are powers of 2 */
   nir_def *tail = nir_ushr(b, nir_iand_imm(b, nir_u2u32(b, base), 15),
                            nir_ufind_msb(b, size));
   nir_def *index = nir_iadd(b, coord, tail);
   nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 3);
   fetch->op = nir_texop_txf;
   fetch->sampler_dim = GLSL_SAMPLER_DIM_2D;
   fetch->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   fetch->coord_components = 2;
   fetch->texture_index = tex->texture_index;
   fetch->texture_non_uniform = tex->texture_non_uniform;
   /* Bounds checked above, and the level is always zero */
   fetch->backend_flags = tex->backend_flags | AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   fetch->src[0] =
      (nir_tex_src){.src_type = nir_tex_src_texture_handle,
                    .src = nir_src_for_ssa(tex->src[handle_idx].src.ssa)};
   fetch->src[1] = (nir_tex_src){
      .src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec2(
         b, nir_iand_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH - 1),
         nir_ushr_imm(b, index, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH))))};
   fetch->src[2] = (nir_tex_src){.src_type = nir_tex_src_lod,
                                 .src = nir_src_for_ssa(nir_imm_int(b, 0))};
   nir_def_init(&fetch->instr, &fetch->def, 4, 32);
   nir_builder_instr_insert(b, &fetch->instr);
   nir_pop_if(b, NULL);
   nir_def *result = nir_trim_vector(b, nir_if_phi(b, rgb32, &fetch->def),
                                     tex->def.num_components);
   nir_push_else(b, NULL);
   /* Out-of-bounds reads return zero, with a one in formats lacking alpha */
   nir_def *zero = nir_imm_zero(b, tex->def.num_components, 32);
   if (tex->def.num_components == 4) {
      nir_def *alpha_kind = nir_ushr_imm(
         b,
         nir_load_global_constant(
            b, 1, 32,
            nir_iadd_imm(
               b, ptr,
               offsetof(struct agx_apple9_texel_metadata, component[3])),
            .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX),
         18);
      nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      zero = nir_vector_insert_imm(
         b, zero,
         nir_bcsel(b, nir_ieq_imm(b, alpha_kind, AGX_APPLE9_TEXEL_ONE), one,
                   nir_imm_int(b, 0)),
         3);
   }
   nir_pop_if(b, NULL);
   nir_def_replace(&tex->def, nir_if_phi(b, result, zero));
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Loads or stores a texel of 1 to 16 bytes as zero-padded 32-bit words */
static nir_def *
apple9_load_texel_words(nir_builder *b, nir_def *addr, nir_def *size_B,
                        enum gl_access_qualifier access)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_push_if(b, nir_ult_imm(b, size_B, 4));
   nir_push_if(b, nir_ieq_imm(b, size_B, 1));
   nir_def *r8 = nir_u2u32(
      b, nir_load_global(b, 1, 8, addr, .align_mul = 1, .access = access));
   nir_push_else(b, NULL);
   nir_def *r16 = nir_u2u32(
      b, nir_load_global(b, 1, 16, addr, .align_mul = 2, .access = access));
   nir_pop_if(b, NULL);
   nir_def *small = nir_vec4(b, nir_if_phi(b, r8, r16), zero, zero, zero);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 4));
   nir_def *r32 = nir_vec4(
      b, nir_load_global(b, 1, 32, addr, .align_mul = 4, .access = access),
      zero, zero, zero);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 8));
   nir_def *r64 = nir_pad_vector_imm_int(
      b, nir_load_global(b, 2, 32, addr, .align_mul = 8, .access = access), 0,
      4);
   nir_push_else(b, NULL);
   nir_def *r128 =
      nir_load_global(b, 4, 32, addr, .align_mul = 16, .access = access);
   nir_pop_if(b, NULL);
   nir_def *wide = nir_if_phi(b, r64, r128);
   nir_pop_if(b, NULL);
   nir_def *large = nir_if_phi(b, r32, wide);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, small, large);
}

static void
apple9_store_texel_words(nir_builder *b, nir_def *addr, nir_def *size_B,
                         nir_def *words, enum gl_access_qualifier access)
{
   nir_def *w0 = nir_channel(b, words, 0);
   nir_push_if(b, nir_ieq_imm(b, size_B, 1));
   nir_store_global(b, nir_u2u8(b, w0), addr, .align_mul = 1, .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 2));
   nir_store_global(b, nir_u2u16(b, w0), addr, .align_mul = 2,
                    .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 4));
   nir_store_global(b, w0, addr, .align_mul = 4, .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 8));
   nir_store_global(b, nir_trim_vector(b, words, 2), addr, .align_mul = 8,
                    .write_mask = 0x3, .access = access);
   nir_push_else(b, NULL);
   nir_store_global(b, words, addr, .align_mul = 16, .write_mask = 0xf,
                    .access = access);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static nir_def *
apple9_texel_word(nir_builder *b, nir_def *words, nir_def *index)
{
   nir_def *w = nir_channel(b, words, 0);
   for (unsigned i = 1; i < 4; ++i)
      w = nir_bcsel(b, nir_ieq_imm(b, index, i), nir_channel(b, words, i), w);
   return w;
}

/* Decode component c of a texel of class cls: its kind, or constant 0 or 1 */
static nir_def *
apple9_decode_component(nir_builder *b, nir_def *words, nir_def *fmt,
                        unsigned c, unsigned cls, bool floating)
{
   nir_def *kind = nir_ushr_imm(b, fmt, 18);
   nir_def *value;
   if (cls == AGX_APPLE9_TEXEL_FLOAT11 || cls == AGX_APPLE9_TEXEL_RGB9E5) {
      nir_def *w0 = nir_channel(b, words, 0);
      value = c < 3 ? nir_channel(b,
                                  cls == AGX_APPLE9_TEXEL_FLOAT11
                                     ? nir_format_unpack_11f11f10f(b, w0)
                                     : nir_format_unpack_r9g9b9e5(b, w0),
                                  c)
                    : nir_imm_float(b, 1.0);
   } else {
      nir_def *raw = apple9_texel_word(
         b, words, nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2));
      if (cls == AGX_APPLE9_TEXEL_FLOAT32) {
         value = raw;
      } else {
         nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
         nir_def *width =
            nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
         nir_def *extend = nir_isub(b, nir_imm_int(b, 32), width);
         nir_def *mask = nir_ushr(b, nir_imm_int(b, -1), extend);
         nir_def *u = nir_iand(b, nir_ushr(b, raw, shift), mask);
         nir_def *i = nir_ishr(b, nir_ishl(b, u, extend), extend);
         switch (cls) {
         case AGX_APPLE9_TEXEL_SINT:
            value = i;
            break;
         case AGX_APPLE9_TEXEL_UNORM:
            value = nir_fdiv(b, nir_u2f32(b, u), nir_u2f32(b, mask));
            break;
         case AGX_APPLE9_TEXEL_SNORM:
            value = nir_fmax(b, nir_imm_float(b, -1.0),
                             nir_fdiv(b, nir_i2f32(b, i),
                                      nir_u2f32(b, nir_ushr_imm(b, mask, 1))));
            break;
         case AGX_APPLE9_TEXEL_FLOAT16:
            value = nir_channel(b, nir_unpack_half_2x16(b, u), 0);
            break;
         default:
            value = u;
            break;
         }
      }
   }
   value = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ZERO),
                     nir_imm_int(b, 0), value);
   return nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ONE),
                    floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1),
                    value);
}

static nir_def *apple9_decode_texel_select(nir_builder *b, nir_def *words,
                                           nir_def *layout, bool floating);

/* Branch on the uniform texel class, evaluating every kind for mixed layouts */
static nir_def *
apple9_decode_texel_classes(nir_builder *b, nir_def *words, nir_def *layout,
                            bool floating, const unsigned *classes,
                            unsigned count)
{
   if (!count)
      return apple9_decode_texel_select(b, words, layout, floating);
   nir_def *in_class = nir_imm_true(b);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *k = nir_ushr_imm(b, nir_channel(b, layout, c), 18);
      nir_def *ok = nir_ieq_imm(b, k, classes[0]);
      if (c)
         ok = nir_ior(b, ok,
                      nir_ior(b, nir_ieq_imm(b, k, AGX_APPLE9_TEXEL_ZERO),
                              nir_ieq_imm(b, k, AGX_APPLE9_TEXEL_ONE)));
      in_class = nir_iand(b, in_class, ok);
   }
   nir_push_if(b, in_class);
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c)
      rgba[c] = apple9_decode_component(b, words, nir_channel(b, layout, c), c,
                                        classes[0], floating);
   nir_def *then_value = nir_vec(b, rgba, 4);
   nir_push_else(b, NULL);
   nir_def *else_value = apple9_decode_texel_classes(b, words, layout, floating,
                                                     classes + 1, count - 1);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, then_value, else_value);
}

static nir_def *
apple9_decode_texel(nir_builder *b, nir_def *words, nir_def *layout,
                    bool floating)
{
   static const unsigned float_classes[] = {
      AGX_APPLE9_TEXEL_UNORM,   AGX_APPLE9_TEXEL_FLOAT32,
      AGX_APPLE9_TEXEL_FLOAT16, AGX_APPLE9_TEXEL_SNORM,
      AGX_APPLE9_TEXEL_FLOAT11, AGX_APPLE9_TEXEL_RGB9E5,
   };
   static const unsigned int_classes[] = {AGX_APPLE9_TEXEL_UINT,
                                          AGX_APPLE9_TEXEL_SINT};
   return floating
             ? apple9_decode_texel_classes(b, words, layout, true,
                                           float_classes,
                                           ARRAY_SIZE(float_classes))
             : apple9_decode_texel_classes(b, words, layout, false, int_classes,
                                           ARRAY_SIZE(int_classes));
}

/* Decode a texel by evaluating every kind (agx_apple9_texel_layout) */
static nir_def *
apple9_decode_texel_select(nir_builder *b, nir_def *words, nir_def *layout,
                           bool floating)
{
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *fmt = nir_channel(b, layout, c);
      nir_def *kind = nir_ushr_imm(b, fmt, 18);
      nir_def *raw = apple9_texel_word(
         b, words, nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2));
      nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
      nir_def *width =
         nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
      nir_def *extend = nir_isub(b, nir_imm_int(b, 32), width);
      nir_def *mask = nir_ushr(b, nir_imm_int(b, -1), extend);
      nir_def *u = nir_iand(b, nir_ushr(b, raw, shift), mask);
      nir_def *i = nir_ishr(b, nir_ishl(b, u, extend), extend);
      nir_def *value;
      if (floating) {
         nir_def *unorm = nir_fdiv(b, nir_u2f32(b, u), nir_u2f32(b, mask));
         nir_def *snorm =
            nir_fmax(b, nir_imm_float(b, -1.0),
                     nir_fdiv(b, nir_i2f32(b, i),
                              nir_u2f32(b, nir_ushr_imm(b, mask, 1))));
         value = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_SNORM),
                           snorm, unorm);
         value =
            nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT16),
                      nir_channel(b, nir_unpack_half_2x16(b, u), 0), value);
         value = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT32),
                           raw, value);
         if (c < 3) {
            nir_def *w0 = nir_channel(b, words, 0);
            value = nir_bcsel(
               b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT11),
               nir_channel(b, nir_format_unpack_11f11f10f(b, w0), c), value);
            value = nir_bcsel(
               b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_RGB9E5),
               nir_channel(b, nir_format_unpack_r9g9b9e5(b, w0), c), value);
         }
      } else {
         value =
            nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_SINT), i, u);
      }
      value = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ZERO),
                        nir_imm_int(b, 0), value);
      rgba[c] =
         nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ONE),
                   floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1), value);
   }
   return nir_vec(b, rgba, 4);
}

/* Like the common packer, but preserve NaNs, which its clamp turns into zero */
static nir_def *
apple9_pack_float11(nir_builder *b, nir_def *rgb)
{
   nir_def *packed = nir_format_pack_11f11f10f(b, rgb);
   const unsigned shifts[] = {0, 11, 22};
   const uint32_t masks[] = {0x7ff, 0x7ff, 0x3ff};
   const uint32_t nans[] = {0x7e0, 0x7e0, 0x3f0};
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *is_nan =
         nir_ult(b, nir_imm_int(b, 0x7f800000),
                 nir_iand_imm(b, nir_channel(b, rgb, c), 0x7fffffff));
      nir_def *repaired =
         nir_ior_imm(b, nir_iand_imm(b, packed, ~(masks[c] << shifts[c])),
                     nans[c] << shifts[c]);
      packed = nir_bcsel(b, is_nan, repaired, packed);
   }
   return packed;
}

/* Encode a texel into zero-padded 32-bit words for its descriptor layout */
static nir_def *
apple9_encode_texel_kind(nir_builder *b, nir_def *value, nir_def *layout,
                         bool floating, int cls)
{
   nir_def *words[4] = {
      nir_imm_int(b, 0),
      nir_imm_int(b, 0),
      nir_imm_int(b, 0),
      nir_imm_int(b, 0),
   };
   nir_def *rgb = nir_trim_vector(b, value, 3);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *fmt = nir_channel(b, layout, c);
      nir_def *kind = nir_ushr_imm(b, fmt, 18);
      nir_def *index = nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2);
      nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
      nir_def *width =
         nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
      nir_def *mask = nir_ushr(b, nir_imm_int(b, -1),
                               nir_isub(b, nir_imm_int(b, 32), width));
      nir_def *v = nir_channel(b, value, c);
      nir_def *bits = v;
      if (floating && cls < 0) {
         nir_def *unorm = nir_f2u32(
            b, nir_fround_even(
                  b, nir_fmul(b, nir_fsat(b, v), nir_u2f32(b, mask))));
         nir_def *snorm = nir_f2i32(
            b,
            nir_fround_even(
               b, nir_fmul(b,
                           nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, -1.0)),
                                    nir_imm_float(b, 1.0)),
                           nir_u2f32(b, nir_ushr_imm(b, mask, 1)))));
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_SNORM),
                          snorm, unorm);
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT16),
                          nir_pack_half_2x16_split(b, v, nir_imm_float(b, 0)),
                          bits);
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT32), v,
                          bits);
      } else if (floating && cls == AGX_APPLE9_TEXEL_UNORM) {
         bits = nir_f2u32(b, nir_fround_even(b, nir_fmul(b, nir_fsat(b, v),
                                                         nir_u2f32(b, mask))));
      } else if (floating && cls == AGX_APPLE9_TEXEL_SNORM) {
         bits = nir_f2i32(
            b,
            nir_fround_even(
               b, nir_fmul(b,
                           nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, -1.0)),
                                    nir_imm_float(b, 1.0)),
                           nir_u2f32(b, nir_ushr_imm(b, mask, 1)))));
      } else if (floating && cls == AGX_APPLE9_TEXEL_FLOAT16) {
         bits = nir_pack_half_2x16_split(b, v, nir_imm_float(b, 0));
      }
      bits = nir_ishl(b, nir_iand(b, bits, mask), shift);
      /* Packed float formats encode all three components at once */
      if (floating && c == 0 &&
          (cls < 0 || cls == AGX_APPLE9_TEXEL_FLOAT11 ||
           cls == AGX_APPLE9_TEXEL_RGB9E5)) {
         if (cls < 0 || cls == AGX_APPLE9_TEXEL_FLOAT11)
            bits = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT11),
                             apple9_pack_float11(b, rgb), bits);
         if (cls < 0 || cls == AGX_APPLE9_TEXEL_RGB9E5)
            bits = nir_bcsel(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_RGB9E5),
                             nir_format_pack_r9g9b9e5(b, rgb), bits);
      } else if (floating && c < 3 &&
                 (cls < 0 || cls == AGX_APPLE9_TEXEL_FLOAT11 ||
                  cls == AGX_APPLE9_TEXEL_RGB9E5)) {
         bits =
            nir_bcsel(b,
                      nir_ior(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_FLOAT11),
                              nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_RGB9E5)),
                      nir_imm_int(b, 0), bits);
      }
      bits = nir_bcsel(b,
                       nir_ior(b, nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ZERO),
                               nir_ieq_imm(b, kind, AGX_APPLE9_TEXEL_ONE)),
                       nir_imm_int(b, 0), bits);
      for (unsigned w = 0; w < 4; ++w) {
         words[w] = nir_bcsel(b, nir_ieq_imm(b, index, w),
                              nir_ior(b, words[w], bits), words[w]);
      }
   }
   return nir_vec(b, words, 4);
}

static nir_def *
apple9_encode_texel_classes(nir_builder *b, nir_def *value, nir_def *layout,
                            bool floating, const unsigned *classes,
                            unsigned count)
{
   if (!count)
      return apple9_encode_texel_kind(b, value, layout, floating, -1);
   nir_def *in_class = nir_imm_true(b);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *k = nir_ushr_imm(b, nir_channel(b, layout, c), 18);
      nir_def *ok = nir_ieq_imm(b, k, classes[0]);
      if (c)
         ok = nir_ior(b, ok,
                      nir_ior(b, nir_ieq_imm(b, k, AGX_APPLE9_TEXEL_ZERO),
                              nir_ieq_imm(b, k, AGX_APPLE9_TEXEL_ONE)));
      in_class = nir_iand(b, in_class, ok);
   }
   nir_push_if(b, in_class);
   nir_def *then_words =
      apple9_encode_texel_kind(b, value, layout, floating, classes[0]);
   nir_push_else(b, NULL);
   nir_def *else_words = apple9_encode_texel_classes(b, value, layout, floating,
                                                     classes + 1, count - 1);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, then_words, else_words);
}

/* Branch on the uniform texel class, as for decoding */
static nir_def *
apple9_encode_texel(nir_builder *b, nir_def *value, nir_def *layout,
                    bool floating)
{
   static const unsigned float_classes[] = {
      AGX_APPLE9_TEXEL_UNORM,   AGX_APPLE9_TEXEL_FLOAT32,
      AGX_APPLE9_TEXEL_FLOAT16, AGX_APPLE9_TEXEL_SNORM,
      AGX_APPLE9_TEXEL_FLOAT11, AGX_APPLE9_TEXEL_RGB9E5,
   };
   return floating ? apple9_encode_texel_classes(b, value, layout, true,
                                                 float_classes,
                                                 ARRAY_SIZE(float_classes))
                   : apple9_encode_texel_kind(b, value, layout, false, -1);
}

/* Fast paths for canonical store encodings, equal to apple9_encode_texel */

nir_def *
agx_apple9_encode_texel_fast(nir_builder *b, nir_def *value, nir_def *layout,
                             bool floating, nir_def *encode)
{
   nir_push_if(b, nir_ieq_imm(b, encode, AGX_APPLE9_STORE_ENCODE_RAW32));
   nir_def *raw = value;
   nir_push_else(b, NULL);
   nir_def *rest;
   if (floating) {
      nir_push_if(b, nir_ieq_imm(b, encode, AGX_APPLE9_STORE_ENCODE_UNORM8));
      nir_def *word = nir_imm_int(b, 0);
      for (unsigned c = 0; c < 4; ++c) {
         nir_def *u = nir_f2u32(
            b, nir_fround_even(
                  b, nir_fmul_imm(b, nir_fsat(b, nir_channel(b, value, c)),
                                  255.0)));
         word = nir_ior(b, word, nir_ishl_imm(b, u, 8 * c));
      }
      nir_def *unorm8 = nir_vec4(b, word, nir_imm_int(b, 0), nir_imm_int(b, 0),
                                 nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_push_if(b, nir_ieq_imm(b, encode, AGX_APPLE9_STORE_ENCODE_FLOAT16));
      nir_def *float16 =
         nir_vec4(b,
                  nir_pack_half_2x16_split(b, nir_channel(b, value, 0),
                                           nir_channel(b, value, 1)),
                  nir_pack_half_2x16_split(b, nir_channel(b, value, 2),
                                           nir_channel(b, value, 3)),
                  nir_imm_int(b, 0), nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_push_if(b, nir_ieq_imm(b, encode, AGX_APPLE9_STORE_ENCODE_FLOAT11));
      nir_def *float11 =
         nir_vec4(b, apple9_pack_float11(b, nir_trim_vector(b, value, 3)),
                  nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_def *generic = apple9_encode_texel(b, value, layout, true);
      nir_pop_if(b, NULL);
      nir_def *float11_or_generic = nir_if_phi(b, float11, generic);
      nir_pop_if(b, NULL);
      nir_def *half_or_generic = nir_if_phi(b, float16, float11_or_generic);
      nir_pop_if(b, NULL);
      rest = nir_if_phi(b, unorm8, half_or_generic);
   } else {
      rest = apple9_encode_texel(b, value, layout, false);
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, raw, rest);
}

/* The metadata base includes the view's byte offset */
static nir_def *
apple9_buffer_texel_address(nir_builder *b, nir_def *meta, nir_def *coord,
                            nir_def *size_B)
{
   nir_def *base = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, meta, offsetof(struct agx_apple9_texel_metadata, base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *offset = nir_imul(b, nir_channel(b, coord, 0), size_B);
   return nir_iadd(b, base, nir_u2u64(b, offset));
}

static bool
apple9_lower_buffer_address(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   bool atomic = intr->intrinsic == nir_intrinsic_bindless_image_atomic;
   bool swap = intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   if (!atomic && !swap &&
       intr->intrinsic != nir_intrinsic_bindless_image_texel_address)
      return false;
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *pbe = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *meta =
      nir_iadd_imm(b, pbe,
                   offsetof(struct agx_apple9_storage_descriptor, metadata) -
                      offsetof(struct agx_apple9_storage_words, pbe));
   unsigned size_B = atomic || swap
                        ? intr->def.bit_size / 8
                        : util_format_get_blocksize(nir_intrinsic_format(intr));
   nir_def *addr = apple9_buffer_texel_address(b, meta, intr->src[1].ssa,
                                               nir_imm_int(b, size_B));
   nir_def *result = addr;
   if (atomic || swap) {
      result =
         swap ? nir_global_atomic_swap(
                   b, intr->def.bit_size, addr, intr->src[3].ssa,
                   intr->src[4].ssa, .atomic_op = nir_intrinsic_atomic_op(intr))
              : nir_global_atomic(b, intr->def.bit_size, addr, intr->src[3].ssa,
                                  .atomic_op = nir_intrinsic_atomic_op(intr));
      /* Keep the memory-to-texture fence of the common image lowering */
      if (!(nir_intrinsic_access(intr) & ACCESS_NON_READABLE))
         nir_fence_mem_to_tex_agx(b);
   }
   nir_def_replace(&intr->def, result);
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* The texture unit misses shader writes: route loads no write can precede */
static bool
apple9_image_write(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return true;
   default:
      return false;
   }
}

static nir_loop *
apple9_outermost_loop(nir_block *block)
{
   nir_loop *loop = NULL;
   for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         loop = nir_cf_node_as_loop(n);
   }
   return loop;
}

static void
apple9_texture_route_loads(nir_shader *nir, struct set *safe)
{
   if (nir->info.internal)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct set *write_loops = _mesa_pointer_set_create(NULL);
   bool writes = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             apple9_image_write(nir_instr_as_intrinsic(instr))) {
            writes = true;
            nir_loop *loop = apple9_outermost_loop(block);
            if (loop)
               _mesa_set_add(write_loops, loop);
         }
      }
   }
   bool written = false;
   nir_foreach_block(block, impl) {
      nir_loop *loop = apple9_outermost_loop(block);
      bool loop_writes = loop && _mesa_set_search(write_loops, loop);
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (apple9_image_write(intr)) {
            written = true;
         } else if (intr->intrinsic == nir_intrinsic_barrier) {
            if (writes && (nir_intrinsic_memory_modes(intr) & nir_var_image))
               written = true;
         } else if (intr->intrinsic == nir_intrinsic_bindless_image_load) {
            enum gl_access_qualifier access = nir_intrinsic_access(intr);
            enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
            if (!written && !loop_writes && !(access & ACCESS_VOLATILE) &&
                !(writes && (access & ACCESS_COHERENT)) &&
                dim != GLSL_SAMPLER_DIM_BUF && dim != GLSL_SAMPLER_DIM_MS &&
                dim != GLSL_SAMPLER_DIM_SUBPASS &&
                dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
               _mesa_set_add(safe, intr);
         }
      }
   }
   _mesa_set_destroy(write_loops, NULL);
}

/* Fetch through the native entry of the storage descriptor at `tex` */
static nir_def *
apple9_texture_load(nir_builder *b, nir_intrinsic_instr *intr,
                    nir_intrinsic_instr *handle, nir_def *coord)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool array = nir_intrinsic_image_array(intr) || cube;
   unsigned coords = (dim == GLSL_SAMPLER_DIM_1D   ? 1
                      : dim == GLSL_SAMPLER_DIM_3D ? 3
                                                   : 2) +
                     (array && dim != GLSL_SAMPLER_DIM_3D);
   nir_def *native = nir_bindless_image_agx(
      b,
      nir_iadd_imm(b, handle->src[0].ssa,
                   offsetof(struct agx_apple9_storage_descriptor, texture) -
                      AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                         AGX_APPLE9_TEXTURE_TABLE_STRIDE),
      .desc_set = nir_intrinsic_desc_set(handle));
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_2D : dim;
   tex->is_array = array;
   tex->coord_components = coords;
   tex->dest_type = nir_intrinsic_dest_type(intr);
   /* Bounds checked by the caller, as no texture state is beside the entry */
   tex->backend_flags |=
      AGX_TEXTURE_FLAG_NO_CLAMP | AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_trim_vector(b, nir_u2u32(b, coord), coords));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, native);
   nir_def_init(&tex->instr, &tex->def, 4, intr->def.bit_size);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* Whether the descriptor behind load handle `tex` has a native entry */
static nir_def *
apple9_native_ok(nir_builder *b, nir_def *tex)
{
   return nir_ine_imm(
      b,
      nir_iand_imm(
         b,
         nir_load_global_constant(
            b, 1, 32,
            nir_iadd_imm(
               b, nir_load_from_texture_handle_agx(b, tex),
               offsetof(struct agx_apple9_storage_descriptor, native_ok)),
            .align_mul = 4),
         AGX_APPLE9_STORAGE_READ_VALID),
      0);
}

/* Native stores need every image read to precede every store, with no loop
 * holding both and no calls. Decide before image loads are lowered.
 */
bool
agx_apple9_native_store_shader(const nir_shader *nir)
{
   unsigned impls = 0;
   nir_foreach_function_impl(impl, nir)
      ++impls;
   if (impls != 1)
      return false;
   bool stored = false, ok = true;
   struct set *read_loops = _mesa_pointer_set_create(NULL);
   struct set *store_loops = _mesa_pointer_set_create(NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               ok = false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_loop *loop = apple9_outermost_loop(block);
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
               ok &= !stored;
               if (loop)
                  _mesa_set_add(read_loops, loop);
               break;
            case nir_intrinsic_bindless_image_store:
               stored = true;
               if (loop)
                  _mesa_set_add(store_loops, loop);
               break;
            default:
               break;
            }
         }
      }
   }
   set_foreach(store_loops, entry)
      ok &= !_mesa_set_search(read_loops, entry->key);
   _mesa_set_destroy(read_loops, NULL);
   _mesa_set_destroy(store_loops, NULL);
   return ok;
}

struct apple9_storage_lowering {
   struct set *texture_loads;
   bool native_stores;
};

/* Untyped stores, such as meta copies of raw texels, would encode as float */
static bool
apple9_native_store_type(nir_alu_type type)
{
   nir_alu_type base = nir_alu_type_get_base_type(type);
   return base == nir_type_float || base == nir_type_int ||
          base == nir_type_uint;
}

/* 3D views have one 2D entry per depth slice, addressed like array layers */
static bool
apple9_can_native_store(const nir_builder *b, const nir_intrinsic_instr *intr,
                        bool native_stores)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   enum pipe_format format = nir_intrinsic_format(intr);
   return native_stores &&
          (format == PIPE_FORMAT_NONE ||
           agx_apple9_native_store_format(format)) &&
          b->shader->info.stage == MESA_SHADER_COMPUTE &&
          (dim == GLSL_SAMPLER_DIM_2D || dim == GLSL_SAMPLER_DIM_CUBE ||
           dim == GLSL_SAMPLER_DIM_3D) &&
          intr->src[3].ssa->bit_size == 32 &&
          apple9_native_store_type(nir_intrinsic_src_type(intr)) &&
          !(nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE));
}

static nir_def *
apple9_native_write_ok(nir_builder *b, nir_def *tex, bool array)
{
   nir_def *flags = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
                   offsetof(struct agx_apple9_storage_descriptor, native_ok)),
      .align_mul = 4);
   unsigned mask = array ? AGX_APPLE9_STORAGE_LAYER_WRITE_VALID
                         : AGX_APPLE9_STORAGE_WRITE_VALID;
   return nir_ine_imm(b, nir_iand_imm(b, flags, mask), 0);
}

static void
apple9_native_store(nir_builder *b, nir_intrinsic_instr *intr,
                    nir_def *pbe_addr, nir_def *coord, bool array)
{
   nir_def *packed = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, pbe_addr,
                   offsetof(struct agx_apple9_storage_descriptor, native_ok) -
                      offsetof(struct agx_apple9_storage_words, pbe)),
      .align_mul = 4);
   nir_def *index =
      nir_iand_imm(b, nir_ushr_imm(b, packed, AGX_APPLE9_STORAGE_PBE_SHIFT),
                   AGX_APPLE9_STORAGE_PBE_MASK);
   /* Cube coordinates already hold the face layer, 3D ones the slice */
   if (array)
      index = nir_iadd(b, index, nir_channel(b, coord, 2));
   nir_image_store_apple9_agx(b, index,
                              nir_trim_vector(b, nir_u2u32(b, coord), 2),
                              nir_pad_vector_imm_int(b, intr->src[3].ssa, 0, 4),
                              .access = nir_intrinsic_access(intr),
                              .src_type = nir_intrinsic_src_type(intr));
}

static bool
apple9_lower_formatless_storage(nir_builder *b, nir_intrinsic_instr *intr,
                                bool load, bool route, bool native_stores)
{
   if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   nir_def *offset = handle->src[0].ssa;
   /* Load descriptors point at tex, store descriptors at pbe */
   nir_def *tex =
      load ? intr->src[0].ssa
           : nir_bindless_image_agx(
                b, nir_iadd_imm(b, offset, -sizeof(struct agx_texture_packed)),
                .desc_set = set);
   nir_def *pbe =
      load ? nir_bindless_image_agx(
                b, nir_iadd_imm(b, offset, sizeof(struct agx_texture_packed)),
                .desc_set = set)
           : intr->src[0].ssa;
   bool array = nir_intrinsic_image_array(intr);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   unsigned dimensions;
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
      dimensions = 1;
      break;
   case GLSL_SAMPLER_DIM_1D:
      dimensions = 1 + array;
      break;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS:
      dimensions = 2 + array;
      break;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      dimensions = 3;
      break;
   default:
      return false;
   }
   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   nir_def *meta =
      nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
                   offsetof(struct agx_apple9_storage_descriptor, metadata));
   nir_def *size_B = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, meta, offsetof(struct agx_apple9_texel_metadata, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *layout = nir_load_global_constant(
      b, 4, 32,
      nir_iadd_imm(b, meta,
                   offsetof(struct agx_apple9_texel_metadata, component)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   /* Storage image descriptors also name a canonical store encoding */
   nir_def *encode = NULL;
   if (!load && dim != GLSL_SAMPLER_DIM_BUF) {
      encode = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(
            b, nir_load_from_texture_handle_agx(b, tex),
            offsetof(struct agx_apple9_storage_descriptor, store_encode)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   }

   /* Honeykrisp's storage cube descriptors are 2D arrays of faces */
   nir_def *size =
      nir_bindless_image_size(b, dimensions, 32, tex, nir_imm_int(b, 0),
                              .image_dim = cube ? GLSL_SAMPLER_DIM_2D : dim,
                              .image_array = cube || array);
   nir_def *coord = intr->src[1].ssa;
   nir_def *inside = nir_imm_true(b);
   for (unsigned d = 0; d < dimensions; ++d)
      inside = nir_iand(
         b, inside,
         nir_ult(b, nir_channel(b, coord, d), nir_channel(b, size, d)));
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_bindless_image_samples(
         b, 32, tex, .image_dim = dim, .image_array = array);
      inside = nir_iand(b, inside,
                        nir_ult(b, nir_u2u32(b, intr->src[2].ssa), samples));
   }
   /* Address ahead of the bounds check, so its uniform part can be hoisted */
   bool native = !load && apple9_can_native_store(b, intr, native_stores);
   bool native_array = array || cube || dim == GLSL_SAMPLER_DIM_3D;
   nir_def *ok = route    ? apple9_native_ok(b, tex)
                 : native ? apple9_native_write_ok(b, tex, native_array)
                          : NULL;
   nir_def *pbe_addr = nir_load_from_texture_handle_agx(b, pbe);
   nir_def *addr = NULL;
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      addr = apple9_buffer_texel_address(b, meta, coord, size_B);
   } else if (!route && !native) {
      bool layered = array || cube || dim == GLSL_SAMPLER_DIM_3D;
      addr = libagx_image_texel_address(
         b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
         nir_u2u32(b, intr->src[2].ssa), size_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, false));
   }
   nir_push_if(b, inside);
   nir_def *value = NULL;
   if (load) {
      bool floating = nir_alu_type_get_base_type(
                         nir_intrinsic_dest_type(intr)) == nir_type_float;
      nir_def *fetched = NULL;
      if (route) {
         nir_push_if(b, ok);
         fetched = apple9_texture_load(b, intr, handle, coord);
         nir_push_else(b, NULL);
         bool layered = array || cube || dim == GLSL_SAMPLER_DIM_3D;
         addr = libagx_image_texel_address(
            b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
            nir_u2u32(b, intr->src[2].ssa), size_B,
            nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D), nir_imm_bool(b, false),
            nir_imm_bool(b, layered), nir_imm_bool(b, false));
      }
      nir_def *words = apple9_load_texel_words(b, addr, size_B, access);
      value = apple9_decode_texel(b, words, layout, floating);
      if (route) {
         nir_pop_if(b, NULL);
         value = nir_if_phi(b, fetched, value);
      }
   } else {
      if (native) {
         nir_push_if(b, ok);
         apple9_native_store(b, intr, pbe_addr, coord, native_array);
         nir_push_else(b, NULL);
         addr = libagx_image_texel_address(
            b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
            nir_u2u32(b, intr->src[2].ssa), size_B, nir_imm_bool(b, false),
            nir_imm_bool(b, false), nir_imm_bool(b, native_array),
            nir_imm_bool(b, false));
      }
      bool floating = nir_alu_type_get_base_type(
                         nir_intrinsic_src_type(intr)) == nir_type_float;
      nir_def *texel = nir_pad_vector_imm_int(b, intr->src[3].ssa, 0, 4);
      nir_def *words = encode ? agx_apple9_encode_texel_fast(b, texel, layout,
                                                             floating, encode)
                              : apple9_encode_texel(b, texel, layout, floating);
      apple9_store_texel_words(b, addr, size_B, words, access);
      if (native)
         nir_pop_if(b, NULL);
   }
   nir_def *zero = NULL;
   if (load) {
      /* Out-of-bounds reads return zero, with a one in formats lacking alpha */
      nir_push_else(b, NULL);
      bool floating = nir_alu_type_get_base_type(
                         nir_intrinsic_dest_type(intr)) == nir_type_float;
      nir_def *alpha_kind = nir_ushr_imm(b, nir_channel(b, layout, 3), 18);
      nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      zero =
         nir_vec4(b, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0),
                  nir_bcsel(b, nir_ieq_imm(b, alpha_kind, AGX_APPLE9_TEXEL_ONE),
                            one, nir_imm_int(b, 0)));
   }
   nir_pop_if(b, NULL);
   if (load) {
      value = nir_if_phi(b, value, zero);
      nir_def_replace(&intr->def,
                      nir_trim_vector(b, value, intr->def.num_components));
   } else {
      nir_instr_remove(&intr->instr);
   }
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Residency code of the nearest texel of the nearest level, zero if resident */
static nir_def *
apple9_residency_code(nir_builder *b, nir_def *table, nir_def *base,
                      nir_def *level, nir_def *x, nir_def *y, nir_def *layer)
{
   nir_def *header = nir_load_global_constant(b, 2, 32, table, .align_mul = 16);
   nir_def *abs_level =
      nir_umin_imm(b, nir_iadd(b, nir_iand_imm(b, base, 0xffff), level),
                   AIL_MAX_MIP_LEVELS - 1);
   nir_def *entry = nir_load_global_constant(
      b, 2, 32,
      nir_iadd(b, table,
               nir_u2u64(b, nir_iadd_imm(
                               b, nir_imul_imm(b, abs_level, 8),
                               offsetof(struct agx_apple9_residency, level)))),
      .align_mul = 8);
   nir_def *packed = nir_channel(b, entry, 1);
   nir_def *shift_x = nir_iand_imm(b, nir_ushr_imm(b, packed, 16), 0xff);
   nir_def *shift_y = nir_ushr_imm(b, packed, 24);
   nir_def *tile = nir_iadd(
      b, nir_imul(b, nir_ushr(b, y, shift_y), nir_iand_imm(b, packed, 0xffff)),
      nir_ushr(b, x, shift_x));
   nir_def *abs_layer = nir_iadd(b, nir_ushr_imm(b, base, 16), layer);
   nir_def *page =
      nir_iadd(b, nir_imul(b, abs_layer, nir_channel(b, header, 0)),
               nir_iadd(b, nir_channel(b, entry, 0), tile));
   page = nir_umin(b, page, nir_iadd_imm(b, nir_channel(b, header, 1), -1));
   nir_def *word = nir_load_global_constant(
      b, 1, 32,
      nir_iadd(b, table,
               nir_u2u64(b, nir_iadd_imm(b, nir_iand_imm(b, page, ~3),
                                         offsetof(struct agx_apple9_residency,
                                                  resident)))),
      .align_mul = 4);
   nir_def *byte = nir_iand_imm(
      b, nir_ushr(b, word, nir_imul_imm(b, nir_iand_imm(b, page, 3), 8)), 0xff);
   return nir_b2i32(b, nir_ieq_imm(b, byte, 0));
}

/* Residency code of buffer texel x; buffers without a table are resident */
static nir_def *
apple9_buffer_residency_code(nir_builder *b, nir_def *meta, nir_def *x)
{
   nir_def *table = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, meta,
                   offsetof(struct agx_apple9_texel_metadata, residency)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *code;
   nir_push_if(b, nir_ine_imm(b, table, 0));
   {
      nir_def *base = nir_load_global_constant(b, 1, 64, meta, .align_mul = 8);
      nir_def *fields = nir_load_global_constant(
         b, 2, 32,
         nir_iadd_imm(b, meta,
                      offsetof(struct agx_apple9_texel_metadata, elements)),
         .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *elements = nir_channel(b, fields, 0);
      nir_def *size_B = nir_channel(b, fields, 1);
      x = nir_umin(b, x, nir_iadd_imm(b, nir_umax_imm(b, elements, 1), -1));
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, x, size_B)));
      nir_def *entry =
         nir_iadd(b, table, nir_ushr_imm(b, addr, util_logbase2(AIL_PAGESIZE)));
      nir_def *word = nir_load_global_constant(
         b, 1, 32, nir_iand_imm(b, entry, ~3ull), .align_mul = 4);
      nir_def *byte = nir_iand_imm(
         b,
         nir_ushr(b, word,
                  nir_imul_imm(b, nir_u2u32(b, nir_iand_imm(b, entry, 3)), 8)),
         0xff);
      code = nir_b2i32(b, nir_ieq_imm(b, byte, 0));
   }
   nir_push_else(b, NULL);
   nir_def *resident = nir_imm_int(b, 0);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, code, resident);
}

/* The LOD a sparse sample selects, before the nearest level is chosen */
static nir_def *
apple9_sparse_lod(nir_builder *b, nir_tex_instr *tex, nir_def *xy,
                  nir_def *size)
{
   nir_def *lod;
   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (tex->op == nir_texop_tg4) {
      lod = nir_imm_float(b, 0.0);
   } else if (tex->op == nir_texop_txl) {
      lod = nir_f2f32(b, tex->src[lod_idx].src.ssa);
   } else if (tex->op == nir_texop_txd ||
              b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = nir_f2f32(
            b, tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa);
         dy = nir_f2f32(
            b, tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa);
      } else {
         dx = nir_ddx(b, xy);
         dy = nir_ddy(b, xy);
      }
      dx = nir_fmul(b, nir_trim_vector(b, dx, 2), size);
      dy = nir_fmul(b, nir_trim_vector(b, dy, 2), size);
      lod = nir_flog2(b, nir_fsqrt(b, nir_fmax(b, nir_fdot2(b, dx, dx),
                                               nir_fdot2(b, dy, dy))));
      int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      if (bias >= 0)
         lod = nir_fadd(b, lod, nir_f2f32(b, tex->src[bias].src.ssa));
   } else {
      lod = nir_imm_float(b, 0.0);
   }
   int min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (min_lod >= 0)
      lod = nir_fmax(b, lod, nir_f2f32(b, tex->src[min_lod].src.ssa));
   return lod;
}

static bool
apple9_lower_sparse_tex(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (!tex->is_sparse)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *code = nir_imm_int(b, 0);
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);

   /* No sparse residency for cube maps */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF && handle >= 0) {
      nir_def *meta = nir_iadd_imm(
         b, nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa),
         offsetof(struct agx_apple9_texel_descriptor, metadata));
      nir_def *coord =
         tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
      code = apple9_buffer_residency_code(
         b, meta, nir_u2u32(b, nir_channel(b, coord, 0)));
   } else if (tex->sampler_dim == GLSL_SAMPLER_DIM_2D && handle >= 0) {
      nir_def *desc =
         nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa);
      nir_def *fields = nir_load_global_constant(
         b, 3, 32, nir_iadd_imm(b, desc, AGX_APPLE9_DESCRIPTOR_RESIDENCY),
         .align_mul = 4);
      nir_def *table = nir_pack_64_2x32_split(b, nir_channel(b, fields, 0),
                                              nir_channel(b, fields, 1));
      nir_def *coord =
         tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
      nir_def *level, *x, *y, *layer = nir_imm_int(b, 0);

      if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms) {
         coord = nir_u2u32(b, coord);
         int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
         level =
            lod >= 0 ? nir_u2u32(b, tex->src[lod].src.ssa) : nir_imm_int(b, 0);
         x = nir_channel(b, coord, 0);
         y = nir_channel(b, coord, 1);
         if (tex->is_array)
            layer = nir_channel(b, coord, 2);
      } else {
         coord = nir_f2f32(b, coord);
         nir_def *xy = nir_trim_vector(b, coord, 2);
         /* Inserted after descriptor lowering, so flag the Apple9 metadata */
         nir_def *size_query = nir_get_texture_size(b, tex);
         nir_def_as_tex(size_query)->backend_flags |=
            AGX_TEXTURE_FLAG_METADATA_QUERY;
         nir_def *size0 = nir_trim_vector(b, size_query, 2);
         nir_def *lod = apple9_sparse_lod(b, tex, xy, nir_u2f32(b, size0));
         /* Nearest mipmapping selects ceil(lod + 1/2) - 1 */
         level =
            nir_f2u32(b, nir_fmax(b, nir_fceil(b, nir_fadd_imm(b, lod, -0.5)),
                                  nir_imm_float(b, 0.0)));
         nir_def *size = nir_umax_imm(b, nir_ushr(b, size0, level), 1);
         nir_def *texel =
            nir_fmin(b,
                     nir_fmax(b, nir_fmul(b, xy, nir_u2f32(b, size)),
                              nir_imm_zero(b, 2, 32)),
                     nir_u2f32(b, nir_iadd_imm(b, size, -1)));
         texel = nir_f2u32(b, texel);
         x = nir_channel(b, texel, 0);
         y = nir_channel(b, texel, 1);
         if (tex->is_array) {
            layer = nir_f2u32(
               b, nir_fmax(b, nir_fround_even(b, nir_channel(b, coord, 2)),
                           nir_imm_float(b, 0.0)));
         }
      }
      code = apple9_residency_code(b, table, nir_channel(b, fields, 2), level,
                                   x, y, layer);
   }

   tex->is_sparse = false;
   tex->def.num_components--;
   b->cursor = nir_after_instr(&tex->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < tex->def.num_components; ++c)
      comps[c] = nir_get_scalar(&tex->def, c);
   comps[tex->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, tex->def.num_components + 1);
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

static bool
apple9_lower_sparse_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_is_sparse_texels_resident:
      nir_def_replace(&intr->def, nir_ieq_imm(b, intr->src[0].ssa, 0));
      return true;
   case nir_intrinsic_sparse_residency_code_and:
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;
   case nir_intrinsic_bindless_image_sparse_load:
      break;
   default:
      return false;
   }

   /* Storage views have a single level, and cube faces are array layers */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *code = nir_imm_int(b, 0);
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      nir_def *meta =
         nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
                      offsetof(struct agx_apple9_storage_descriptor, metadata));
      code = apple9_buffer_residency_code(
         b, meta, nir_u2u32(b, nir_channel(b, intr->src[1].ssa, 0)));
   } else if (dim == GLSL_SAMPLER_DIM_2D || dim == GLSL_SAMPLER_DIM_CUBE) {
      nir_def *meta =
         nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
                      offsetof(struct agx_apple9_storage_descriptor, metadata));
      nir_def *table = nir_load_global_constant(b, 1, 64, meta, .align_mul = 8);
      nir_def *base = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, meta,
                      offsetof(struct agx_apple9_texel_metadata, elements)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *coord = nir_u2u32(b, intr->src[1].ssa);
      bool layered =
         nir_intrinsic_image_array(intr) || dim == GLSL_SAMPLER_DIM_CUBE;
      code = apple9_residency_code(
         b, table, base, nir_imm_int(b, 0), nir_channel(b, coord, 0),
         nir_channel(b, coord, 1),
         layered ? nir_channel(b, coord, 2) : nir_imm_int(b, 0));
   }

   intr->intrinsic = nir_intrinsic_bindless_image_load;
   intr->num_components--;
   intr->def.num_components--;
   b->cursor = nir_after_instr(&intr->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < intr->def.num_components; ++c)
      comps[c] = nir_get_scalar(&intr->def, c);
   comps[intr->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, intr->def.num_components + 1);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

bool
agx_nir_lower_apple9_sparse_residency(nir_shader *nir)
{
   bool progress = nir_shader_tex_pass(nir, apple9_lower_sparse_tex,
                                       nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(nir, apple9_lower_sparse_intrinsic,
                                          nir_metadata_none, NULL);
   return progress;
}

/* Uncompressed storage images through memory, addressed as image atomics */
static bool
apple9_lower_storage_image(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   const struct apple9_storage_lowering *lower = data;
   bool load = intr->intrinsic == nir_intrinsic_bindless_image_load;
   if (!load && intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;
   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE)
      return apple9_lower_formatless_storage(
         b, intr, load, load && _mesa_set_search(lower->texture_loads, intr),
         lower->native_stores);
   const struct util_format_description *desc = util_format_description(format);
   unsigned size_B = util_format_get_blocksize(format);
   if ((desc->layout != UTIL_FORMAT_LAYOUT_PLAIN &&
        format != PIPE_FORMAT_R11G11B10_FLOAT &&
        format != PIPE_FORMAT_R9G9B9E5_FLOAT) ||
       !util_is_power_of_two_nonzero(size_B) || size_B > 16 ||
       util_format_is_depth_or_stencil(format))
      return false;

   if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   nir_def *offset = handle->src[0].ssa;
   /* Load descriptors point at tex, store descriptors at pbe */
   nir_def *tex =
      load ? intr->src[0].ssa
           : nir_bindless_image_agx(
                b, nir_iadd_imm(b, offset, -sizeof(struct agx_texture_packed)),
                .desc_set = set);
   nir_def *pbe =
      load ? nir_bindless_image_agx(
                b, nir_iadd_imm(b, offset, sizeof(struct agx_texture_packed)),
                .desc_set = set)
           : intr->src[0].ssa;
   bool array = nir_intrinsic_image_array(intr);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   unsigned dimensions;
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF:
      dimensions = 1;
      break;
   case GLSL_SAMPLER_DIM_1D:
      dimensions = 1 + array;
      break;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS:
      dimensions = 2 + array;
      break;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      dimensions = 3;
      break;
   default:
      return false;
   }
   /* Storage cubes are 2D arrays, so bound the face without cube scaling */
   nir_def *size =
      nir_bindless_image_size(b, dimensions, 32, tex, nir_imm_int(b, 0),
                              .image_dim = cube ? GLSL_SAMPLER_DIM_2D : dim,
                              .image_array = cube || array);
   nir_def *coord = intr->src[1].ssa;
   nir_def *inside = nir_imm_true(b);
   for (unsigned d = 0; d < dimensions; ++d)
      inside = nir_iand(
         b, inside,
         nir_ult(b, nir_channel(b, coord, d), nir_channel(b, size, d)));
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_bindless_image_samples(
         b, 32, tex, .image_dim = dim, .image_array = array);
      inside = nir_iand(b, inside,
                        nir_ult(b, nir_u2u32(b, intr->src[2].ssa), samples));
   }
   /* Typed loads decode with a known format through global memory */
   bool native =
      !load && apple9_can_native_store(b, intr, lower->native_stores);
   bool native_array = array || cube || dim == GLSL_SAMPLER_DIM_3D;
   nir_def *ok = native ? apple9_native_write_ok(b, tex, native_array) : NULL;
   /* See apple9_lower_formatless_storage: address ahead of the check. */
   nir_def *addr =
      native ? NULL
             : nir_bindless_image_texel_address(
                  b, 64, pbe, coord, intr->src[2].ssa, .image_dim = dim,
                  .image_array = array, .format = format,
                  .access = nir_intrinsic_access(intr));
   nir_push_if(b, inside);
   nir_def *value = NULL;
   if (load) {
      nir_def *raw = nir_load_global(
         b, DIV_ROUND_UP(size_B, 4), MIN2(32, size_B * 8), addr,
         .align_mul = size_B, .access = nir_intrinsic_access(intr));
      value = nir_format_unpack_rgba(b, raw, format);
   } else {
      if (native) {
         nir_push_if(b, ok);
         apple9_native_store(b, intr, nir_load_from_texture_handle_agx(b, pbe),
                             coord, native_array);
         nir_push_else(b, NULL);
         addr = nir_bindless_image_texel_address(
            b, 64, pbe, coord, intr->src[2].ssa, .image_dim = dim,
            .image_array = array, .format = format,
            .access = nir_intrinsic_access(intr));
      }
      nir_def *raw =
         format == PIPE_FORMAT_R11G11B10_FLOAT
            ? apple9_pack_float11(b, nir_trim_vector(b, intr->src[3].ssa, 3))
            : nir_format_pack_rgba(b, format, intr->src[3].ssa);
      raw = size_B <= 4 ? nir_u2uN(b, raw, size_B * 8)
                        : nir_trim_vector(b, raw, size_B / 4);
      nir_store_global(b, raw, addr, .align_mul = size_B,
                       .write_mask = BITFIELD_MASK(raw->num_components),
                       .access = nir_intrinsic_access(intr));
      if (native)
         nir_pop_if(b, NULL);
   }
   nir_def *zero = NULL;
   if (load) {
      /* Out-of-bounds reads return zero, with a one in formats lacking alpha */
      nir_push_else(b, NULL);
      bool alpha = desc->swizzle[3] <= PIPE_SWIZZLE_W;
      nir_alu_type type = nir_intrinsic_dest_type(intr);
      nir_def *one = nir_alu_type_get_base_type(type) == nir_type_float
                        ? nir_imm_float(b, 1.0)
                        : nir_imm_int(b, 1);
      zero = nir_vec4(b, nir_imm_int(b, 0), nir_imm_int(b, 0),
                      nir_imm_int(b, 0), alpha ? nir_imm_int(b, 0) : one);
   }
   nir_pop_if(b, NULL);
   if (load) {
      value = nir_if_phi(b, value, zero);
      nir_def_replace(&intr->def,
                      nir_trim_vector(b, value, intr->def.num_components));
   } else {
      nir_instr_remove(&intr->instr);
   }
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Fences stay only while some storage image access uses the texture unit */
bool
agx_nir_lower_apple9_image_barriers(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (nir_intrinsic_has_image_dim(intr) &&
             intr->intrinsic != nir_intrinsic_image_size &&
             intr->intrinsic != nir_intrinsic_bindless_image_size &&
             intr->intrinsic != nir_intrinsic_image_samples &&
             intr->intrinsic != nir_intrinsic_bindless_image_samples &&
             intr->intrinsic != nir_intrinsic_image_levels &&
             intr->intrinsic != nir_intrinsic_bindless_image_levels)
            return false;
      }
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_barrier) {
            unsigned modes = nir_intrinsic_memory_modes(intr);
            if (modes & nir_var_image)
               nir_intrinsic_set_memory_modes(
                  intr, (modes & ~nir_var_image) | nir_var_mem_global);
         } else if (intr->intrinsic == nir_intrinsic_fence_pbe_to_tex_agx ||
                    intr->intrinsic == nir_intrinsic_fence_mem_to_tex_agx) {
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_barrier(
               &b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global);
            nir_instr_remove(instr);
         }
      }
   }
   return nir_progress(true, impl, nir_metadata_control_flow);
}

/* Texel buffers by handle */
bool
agx_nir_lower_apple9_texel_buffers(nir_shader *nir)
{
   /* Bounds and RGB32 checks add control flow */
   return nir_shader_tex_pass(nir, apple9_lower_texel_buffer, nir_metadata_none,
                              NULL);
}

/* Rectangle images (OpenGL) are 2D images with a single level */
static bool
apple9_lower_rect_image(nir_builder *b, nir_intrinsic_instr *intr, void *_)
{
   if (!nir_intrinsic_has_image_dim(intr) ||
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_RECT)
      return false;

   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   return true;
}

/* Storage, texel buffers and sparse residency, before agx_nir_lower_images */
bool
agx_nir_lower_apple9_storage(nir_shader *nir)
{
   bool progress = false;

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, apple9_lower_rect_image,
            nir_metadata_all, NULL);

   /* Buffer residency lookups add control flow */
   NIR_PASS(progress, nir, agx_nir_lower_apple9_sparse_residency);

   NIR_PASS(progress, nir, agx_nir_lower_apple9_texel_buffers);

   struct set *texture_loads = _mesa_pointer_set_create(NULL);
   apple9_texture_route_loads(nir, texture_loads);
   struct apple9_storage_lowering lower = {
      .texture_loads = texture_loads,
      .native_stores = agx_apple9_native_store_shader(nir),
   };
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass,
            apple9_lower_storage_image, nir_metadata_none, &lower);
   _mesa_set_destroy(texture_loads, NULL);

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass,
            apple9_lower_buffer_address, nir_metadata_control_flow, NULL);
   return progress;
}
