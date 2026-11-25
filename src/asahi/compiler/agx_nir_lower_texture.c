/*
 * Copyright 2023 Valve Corporation
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_resource_layout.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "gallium/include/pipe/p_defines.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "glsl_types.h"
#include "libagx.h"
#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* Residency flags are inverted from NIR */
#define AGX_RESIDENT (0)

static bool
fence_image(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_after_instr(&intr->instr);

   /* If the image is write-only, there is no fencing needed */
   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
      return false;
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      nir_fence_pbe_to_tex_agx(b);
      return true;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic_swap:
      nir_fence_mem_to_tex_agx(b);
      return true;

   default:
      return false;
   }
}

static nir_def *
texture_descriptor_ptr(nir_builder *b, nir_tex_instr *tex)
{
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   assert(handle_idx >= 0 && "must be bindless");
   return nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
}

static bool
has_nonzero_lod(nir_tex_instr *tex)
{
   int idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (idx < 0)
      return false;

   nir_src src = tex->src[idx].src;
   return !(nir_src_is_const(src) && nir_src_as_uint(src) == 0);
}

static bool
lower_tex_crawl(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if ((tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples &&
        tex->op != nir_texop_query_levels) ||
       (tex->backend_flags & AGX_TEXTURE_FLAG_METADATA_QUERY))
      return false;

   nir_def *ptr = texture_descriptor_ptr(b, tex);
   unsigned nr_comps = tex->def.num_components;
   assert(nr_comps <= 3);

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod = lod_idx >= 0 ? nir_u2u16(b, tex->src[lod_idx].src.ssa)
                               : nir_imm_intN_t(b, 0, 16);

   nir_def *res;
   if (tex->op == nir_texop_txs) {
      res =
         libagx_txs(b, ptr, lod, nir_imm_int(b, nr_comps),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_BUF),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
                    nir_imm_bool(b, tex->is_array));
   } else if (tex->op == nir_texop_query_levels) {
      res = libagx_texture_levels(b, ptr);
   } else {
      res = libagx_texture_samples(b, ptr);
   }

   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, res, nr_comps));
   nir_instr_remove(instr);
   return true;
}

/*
 * Given a 1D buffer texture coordinate, calculate the 2D coordinate vector that
 * will be used to access the linear 2D texture bound to the buffer.
 */
static nir_def *
coords_for_buffer_texture(nir_builder *b, nir_def *coord)
{
   return nir_vec2(b, nir_umod_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH),
                   nir_udiv_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH));
}

/*
 * Buffer textures are lowered to 2D (1024xN) textures in the driver to access
 * more storage. When lowering, we need to fix up the coordinate accordingly.
 *
 * Furthermore, RGB32 formats are emulated by lowering to global memory access,
 * so to read a buffer texture we generate code that looks like:
 *
 *    if (descriptor->format == RGB32)
 *       return ((uint32_t *) descriptor->address)[x];
 *    else
 *       return txf(texture_as_2d, vec2(x % 1024, x / 1024));
 */
static bool
lower_buffer_texture(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *size = nir_get_texture_size(b, tex);
   nir_def *oob = nir_uge(b, coord, size);

   /* Apply the buffer offset after calculating oob but before remapping */
   nir_def *desc = texture_descriptor_ptr(b, tex);
   coord = libagx_buffer_texture_offset(b, desc, coord);

   /* Map out-of-bounds indices to out-of-bounds coordinates for robustness2
    * semantics from the hardware.
    */
   coord = nir_bcsel(b, oob, nir_imm_int(b, -1), coord);

   bool is_float = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;

   /* Lower RGB32 reads if the format requires. If we are out-of-bounds, we use
    * the hardware path so we get a zero texel.
    */
   nir_if *nif = nir_push_if(
      b, nir_iand(b, libagx_texture_is_rgb32(b, desc), nir_inot(b, oob)));

   nir_def *rgb32 = nir_trim_vector(
      b, libagx_texture_load_rgb32(b, desc, coord, nir_imm_bool(b, is_float)),
      nir_tex_instr_result_size(tex));

   /* Raw loads do not return residency information, but residency queries are
    * supported on buffer textures. Fortunately, we do not need to support
    * sparse RGB32 buffers, so we simply claim all RGB32 loads were resident.
    * Nothing should hit this in practice, but if we don't do *something* here
    * we'll get vector size mismatches which blow up in vkd3d-proton.
    */
   if (tex->is_sparse) {
      rgb32 = nir_pad_vector_imm_int(b, rgb32, AGX_RESIDENT,
                                     rgb32->num_components + 1);
   }

   nir_push_else(b, nif);

   /* Otherwise, lower the texture instruction to read from 2D */
   assert(coord->num_components == 1 && "buffer textures are 1D");
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;

   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_instr_remove(&tex->instr);
   nir_builder_instr_insert(b, &tex->instr);
   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord2d);
   nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
   nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   nir_block *else_block = nir_cursor_current_block(b->cursor);
   nir_pop_if(b, nif);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, rgb32, &tex->def);
   nir_def_rewrite_uses(&tex->def, phi);
   nir_phi_instr *phi_instr = nir_def_as_phi(phi);
   nir_phi_src *else_src = nir_phi_get_src_from_block(phi_instr, else_block);
   nir_src_rewrite(&else_src->src, &tex->def);
   return true;
}

/* The 32-byte texture info of the handle's descriptor or the texture index */
static nir_def *
export_texture_info(nir_builder *b, nir_tex_instr *tex, unsigned field)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);

   if (handle >= 0) {
      nir_def *address = agx_nir_descriptor_address(
         b, tex->src[handle].src.ssa, AGX_DESCRIPTOR_TEXTURE_INFO + field * 4);

      return nir_load_global_constant(
         b, 1, 32, address, .align_mul = 4,
         .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   }

   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_TEXTURE_INFO_OFFSET +
                        tex->texture_index * AGX_TEXTURE_INFO_STRIDE +
                        field * 4),
      .align_mul = 4, .range = ~0);
}

/* Texture info fields */
enum texture_info_field {
   TEXTURE_INFO_WIDTH = 0,
   TEXTURE_INFO_HEIGHT = 1,
   TEXTURE_INFO_LAYERS = 2,
   TEXTURE_INFO_LEVELS = 3,
   TEXTURE_INFO_SAMPLES = 4,

   /* Texel buffers by index: RGB32 views, and swizzle at 3 bits per channel */
   TEXTURE_INFO_RGB32 = 5,
   TEXTURE_INFO_BUFFER_SWIZZLE = 6,
};

/* Size, level and sample queries, all zero for a null view */
static bool
lower_query_export(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *result;

   if (tex->op == nir_texop_txs) {
      int src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *lod =
         src < 0 ? nir_imm_int(b, 0) : nir_u2u32(b, tex->src[src].src.ssa);
      lod = nir_umin(b, lod, nir_imm_int(b, 31));

      nir_def *size[3];
      for (unsigned c = 0; c < tex->def.num_components; ++c) {
         bool layer = tex->is_array && c + 1 == tex->def.num_components;
         size[c] = export_texture_info(b, tex, layer ? TEXTURE_INFO_LAYERS : c);

         if (!layer && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
            size[c] = nir_umax(b, nir_ushr(b, size[c], lod), nir_imm_int(b, 1));
      }

      result = nir_vec(b, size, tex->def.num_components);
   } else {
      result = export_texture_info(b, tex,
                                   tex->op == nir_texop_query_levels
                                      ? TEXTURE_INFO_LEVELS
                                      : TEXTURE_INFO_SAMPLES);
   }

   if (tex->op != nir_texop_query_levels &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      nir_def *null =
         nir_ieq_imm(b, export_texture_info(b, tex, TEXTURE_INFO_LEVELS), 0);
      result = nir_bcsel(b, null, nir_imm_zero(b, result->num_components, 32),
                         result);
   }

   nir_def_rewrite_uses(&tex->def, nir_u2uN(b, result, tex->def.bit_size));
   nir_instr_remove(&tex->instr);
   return true;
}

/* LOD or bias as a signed 6.6 fixed-point value in bits 16-27, floored */
static nir_def *
pack_lod_word(nir_builder *b, nir_def *lod)
{
   nir_def *clamped =
      nir_fmin(b, nir_fmax(b, nir_f2f32(b, lod), nir_imm_float(b, -32.0f)),
               nir_imm_float(b, 2047.0f / 64.0f));

   /* The floor of the exact product, also for negative fractions */
   uint32_t fp_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *fixed = nir_f2i32(b, nir_ffloor(b, nir_fmul_imm(b, clamped, 64.0)));

   b->fp_math_ctrl = fp_math_ctrl;
   return nir_ishl_imm(b, nir_iand_imm(b, fixed, 0xfff), 16);
}

/* Minimum LOD as an unsigned 6.6 fixed-point value in bits 0-11 */
static nir_def *
pack_min_lod_word(nir_builder *b, nir_def *min_lod)
{
   nir_def *clamped =
      nir_fmin(b, nir_fmax(b, nir_f2f32(b, min_lod), nir_imm_float(b, 0.0f)),
               nir_imm_float(b, 2047.0f / 64.0f));

   uint32_t fp_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *fixed = nir_f2u32(b, nir_ffloor(b, nir_fmul_imm(b, clamped, 64.0)));

   b->fp_math_ctrl = fp_math_ctrl;
   return fixed;
}

/* The hardware clamps level, layer and sample, so bound fetches explicitly */
static nir_def *
bound_fetch(nir_builder *b, nir_tex_instr *tex, nir_def *coord, nir_def **lod,
            nir_def *layer, nir_def *sample)
{
   nir_def *valid = nir_imm_true(b);

   if (*lod) {
      valid = nir_iand(
         b, valid,
         nir_ult(b, *lod, export_texture_info(b, tex, TEXTURE_INFO_LEVELS)));
      *lod = nir_bcsel(b, valid, *lod, nir_imm_int(b, 0));
   }

   if (layer) {
      valid = nir_iand(
         b, valid,
         nir_ult(b, layer, export_texture_info(b, tex, TEXTURE_INFO_LAYERS)));
   }

   if (sample) {
      valid = nir_iand(
         b, valid,
         nir_ult(b, sample, export_texture_info(b, tex, TEXTURE_INFO_SAMPLES)));
   }

   nir_def *x =
      nir_bcsel(b, valid, nir_channel(b, coord, 0), nir_imm_int(b, -1));
   return nir_vector_insert_imm(b, coord, x, 0);
}

/* Exported samples do not apply their LOD bias themselves */
static bool
lower_lod_bias_export(nir_builder *b, nir_tex_instr *tex)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   nir_def *bias;

   if (handle >= 0) {
      nir_def *address = agx_nir_descriptor_address(
         b, tex->src[handle].src.ssa,
         AGX_DESCRIPTOR_RUNTIME_SAMPLER + AGX_RUNTIME_SAMPLER_STRIDE);

      bias = nir_load_global_constant(
         b, 1, 16, address, .align_mul = 2,
         .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   } else {
      bias = nir_load_ubo(
         b, 1, 32, nir_imm_int(b, AGX_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
         .align_mul = 4, .range = ~0);
   }

   nir_def_replace(&tex->def, nir_f2fN(b, bias, tex->def.bit_size));
   return true;
}

/* Native min LOD: 2D biased and gradient samples, no compare or offset */
static bool
native_min_lod(const nir_tex_instr *tex)
{
   return tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_shadow &&
          nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 &&
          (tex->op == nir_texop_txd || tex->op == nir_texop_txb ||
           tex->op == nir_texop_tex);
}

static bool
lower_regular_texture_export(nir_builder *b, nir_tex_instr *tex)
{
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_texture_samples)
      return lower_query_export(b, tex);

   if (tex->op == nir_texop_lod_bias)
      return lower_lod_bias_export(b, tex);

   if (nir_tex_instr_is_query(tex) && tex->op != nir_texop_lod)
      return false;

   assert(tex->sampler_dim != GLSL_SAMPLER_DIM_BUF && !tex->is_sparse &&
          "lowered by lower_texture_exports");

   /* Don't lower twice */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;

   /* Texture results are 32-bit, so narrow 16-bit destinations afterwards */
   if (tex->def.bit_size == 16) {
      nir_alu_type base = nir_alu_type_get_base_type(tex->dest_type);
      tex->def.bit_size = 32;
      tex->dest_type = base | 32;

      b->cursor = nir_after_instr(&tex->instr);
      nir_def *narrow = base == nir_type_float ? nir_f2f16(b, &tex->def)
                                               : nir_u2u16(b, &tex->def);
      nir_def_rewrite_uses_after(&tex->def, narrow);
      b->cursor = nir_before_instr(&tex->instr);
   }

   bool fetch = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms;
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   bool volume = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   bool array = tex->is_array && tex->op != nir_texop_lod;

   /* LOD queries are biased samples of the spatial coordinates, layer zero */
   if (tex->op == nir_texop_lod)
      tex->is_shadow = false;

   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *ms = nir_steal_tex_src(tex, nir_tex_src_ms_index);
   nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   nir_def *comparator = nir_steal_tex_src(tex, nir_tex_src_comparator);
   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   nir_def *ddx = nir_steal_tex_src(tex, nir_tex_src_ddx);
   nir_def *ddy = nir_steal_tex_src(tex, nir_tex_src_ddy);
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);

   coord = nir_u2u32(b, coord);

   /* Drop the layer of LOD queries built for biases, or cubes take it as z */
   unsigned spatial =
      glsl_get_sampler_dim_coordinate_components(tex->sampler_dim);
   if (tex->is_array && !array && coord->num_components > spatial)
      coord = nir_trim_vector(b, coord, spatial);

   if (lod)
      lod = fetch ? nir_u2u32(b, lod) : nir_f2f32(b, lod);
   if (ms)
      ms = nir_u2u32(b, ms);

   /* The layer is the last coordinate, as an integer clamped to the view */
   nir_def *layer = NULL;
   if (array) {
      unsigned l = coord->num_components - 1;
      layer = nir_channel(b, coord, l);
      coord = nir_trim_vector(b, coord, l);

      if (!fetch) {
         layer = nir_f2u32(
            b, nir_fmax(b, nir_fround_even(b, layer), nir_imm_float(b, 0.0f)));
         layer = nir_umin(
            b, layer,
            nir_iadd_imm(b, export_texture_info(b, tex, TEXTURE_INFO_LAYERS),
                         -1));
      }
   }

   if (fetch && !(tex->backend_flags &
                  (AGX_TEXTURE_FLAG_FETCH_BOUNDED | AGX_TEXTURE_FLAG_NO_CLAMP)))
      coord = bound_fetch(b, tex, coord, &lod, layer, ms);

   /* Native min LOD needs a bias, so implicit samples become zero-biased */
   if (min_lod && tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      bias = nir_imm_float(b, 0.0f);
   }

   /* The min LOD is in word 2 for biased arrays, else in the LOD word */
   bool array_min_lod = min_lod && array && tex->op == nir_texop_txb;

   if (min_lod) {
      assert(native_min_lod(tex) && "other minimum LODs are lowered");

      if (!array_min_lod)
         tex->backend_flags |= AGX_TEXTURE_FLAG_MIN_LOD;
   }

   nir_def *undef = nir_undef(b, 1, 32);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *words[16];
   unsigned nr = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(words); ++i)
      words[i] = undef;

   words[0] = nir_channel(b, coord, 0);
   words[1] = coord->num_components > 1 ? nir_channel(b, coord, 1) : zero;

   nir_def *z = cube || volume   ? nir_channel(b, coord, 2)
                : tex->is_shadow ? nir_f2f32(b, comparator)
                : ms ? nir_ior(b, layer ? layer : zero, nir_ishl_imm(b, ms, 12))
                : layer         ? layer
                : tex->is_array ? zero
                                : NULL;

   /* The LOD word, with the layer or cube index below */
   nir_def *lod_word = zero;
   if (tex->op == nir_texop_txf)
      lod_word = nir_ishl_imm(b, lod ? lod : zero, 22);
   else if (lod)
      lod_word = pack_lod_word(b, lod);
   else if (bias)
      lod_word = pack_lod_word(b, bias);

   nir_def *lod_base = lod_word;
   if (layer && !ms)
      lod_word = nir_ior(b, lod_word, layer);

   if (tex->op == nir_texop_txd) {
      bool extended = volume || array || tex->is_shadow || min_lod;

      if (extended) {
         nir_def *w3 = layer ? layer : zero;
         if (min_lod)
            w3 = nir_ior(b, w3,
                         nir_ishl_imm(b, pack_min_lod_word(b, min_lod), 16));

         /* The layer is in word 3, word 2 is the comparator or zero */
         words[2] = (z && z != layer) ? z : zero;
         words[3] = volume ? undef : w3;
      }

      unsigned axes = volume ? 3 : 2;
      for (unsigned c = 0; c < axes; ++c) {
         words[4 + 2 * c] = nir_f2f32(b, nir_channel(b, ddx, c));
         words[5 + 2 * c] = nir_f2f32(b, nir_channel(b, ddy, c));
      }

      nr = volume ? 16 : 8;
   } else if ((tex->op == nir_texop_tex || tex->op == nir_texop_tg4) && !z) {
      /* Implicit LOD 2D, including gathers */
      nr = 2;
   } else if (!z && !min_lod) {
      /* Explicit LOD or bias 2D */
      words[3] = lod_word;
      nr = 4;
   } else {
      if (array_min_lod) {
         z = nir_ishl_imm(b, pack_min_lod_word(b, min_lod), 16);
      } else if (min_lod) {
         /* The bias and minimum LOD share the LOD word */
         z = zero;
         lod_word = nir_ior(b, lod_word, pack_min_lod_word(b, min_lod));
      }

      words[2] = z;
      words[3] = ms ? zero : lod_word;
      nr = 4;
   }

   /* Implicit volume samples are zero-biased samples */
   if (tex->op == nir_texop_tex && nr == 4) {
      tex->op = nir_texop_txb;
   }

   /* CUBE selects the face, sampled at u / major + 0.5 and v / major + 0.5 */
   nir_def *face = NULL, *alt[4];
   bool seamful = false;
   if (cube && !fetch) {
      nir_def *f = nir_cube_face_agx(b, coord);
      nir_def *rcp = nir_frcp(b, nir_channel(b, f, 0));
      face = nir_channel(b, f, 1);

      for (unsigned i = 0; i < 2; ++i) {
         words[i] =
            nir_ffma(b, nir_channel(b, f, 2 + i), rcp, nir_imm_float(b, 0.5f));
      }

      /* The comparator goes in word 2, as the face is in the LOD word */
      nir_def *w3 = words[3] == undef ? zero : words[3];
      words[2] = tex->is_shadow ? nir_f2f32(b, comparator) : face;
      words[3] = nir_ior(b, w3, array ? nir_ishl_imm(b, face, 12) : face);

      /* Seamful samplers wrap on the face, a layer of the 2D array alias.
       * Implicit-LOD, gradient and compare samples stay native.
       */
      seamful = !tex->is_shadow &&
                (tex->op == nir_texop_txl || tex->op == nir_texop_tg4) &&
                !(tex->backend_flags & AGX_TEXTURE_FLAG_CUBE_AS_ARRAY) &&
                nr == 4;

      if (seamful) {
         nir_def *as_layer =
            array ? nir_iadd(b, nir_imul_imm(b, layer, 6), face) : face;

         alt[0] = words[0];
         alt[1] = words[1];
         alt[2] = as_layer;
         alt[3] = nir_ior(b, lod_base, as_layer);
      }
   }

   nir_tex_instr_add_src(tex, nir_tex_src_backend1, nir_vec(b, words, nr));

   /* Offsets are constants packed in 4-bit fields */
   if (offset) {
      assert(nir_src_is_const(nir_src_for_ssa(offset)) &&
             "non-constant offsets are lowered");

      uint32_t packed = 0;
      for (unsigned c = 0; c < offset->num_components; ++c) {
         packed |= (nir_scalar_as_uint(nir_get_scalar(offset, c)) & 0xf)
                   << (4 * c);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_backend2, nir_imm_int(b, packed));
   }

   if (seamful) {
      nir_def *is_seamful =
         nir_ine_imm(b, agx_sampler_field(b, tex, AGX_SAMP_SEAMFUL_CUBE), 0);

      nir_tex_instr *as_array =
         nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
      as_array->backend_flags |= AGX_TEXTURE_FLAG_CUBE_AS_ARRAY;
      nir_def *alt_words = nir_vec(b, alt, 4);

      /* The cube view's alias; clone sources are rewritten once inserted */
      int handle =
         nir_tex_instr_src_index(as_array, nir_tex_src_texture_handle);
      nir_def *alias = NULL;
      if (handle >= 0) {
         /* A bindless handle stays one, for the driver's handle lowering */
         nir_def *h = as_array->src[handle].src.ssa;
         nir_instr *hi = nir_def_instr(h);
         nir_intrinsic_instr *bindless =
            hi->type == nir_instr_type_intrinsic ? nir_instr_as_intrinsic(hi)
                                                 : NULL;

         if (bindless &&
             bindless->intrinsic == nir_intrinsic_bindless_image_agx) {
            alias = nir_bindless_image_agx(
               b,
               nir_iadd_imm(b, bindless->src[0].ssa,
                            AGX_CUBE_ALIAS_HANDLE_OFFSET),
               .desc_set = nir_intrinsic_desc_set(bindless));
         } else {
            alias = nir_iadd_imm(b, h, AGX_CUBE_ALIAS_HANDLE_OFFSET);
         }
      } else {
         assert(nir_tex_instr_src_index(as_array, nir_tex_src_texture_offset) <
                   0 &&
                tex->texture_index < AGX_MAX_TEXTURE_INDICES);
         as_array->texture_index = AGX_CUBE_ALIAS_INDEX(tex->texture_index);
      }

      nir_push_if(b, is_seamful);
      nir_builder_instr_insert(b, &as_array->instr);
      nir_src_rewrite(
         &as_array->src[nir_tex_instr_src_index(as_array, nir_tex_src_backend1)]
             .src,
         alt_words);
      if (alias)
         nir_src_rewrite(&as_array->src[handle].src, alias);
      nir_push_else(b, NULL);
      nir_instr_remove(&tex->instr);
      nir_builder_instr_insert(b, &tex->instr);
      nir_pop_if(b, NULL);

      nir_def *phi = nir_if_phi(b, &as_array->def, &tex->def);
      nir_foreach_use_including_if_safe(use, &tex->def) {
         if (nir_src_is_if(use) || nir_src_use_instr(use) != nir_def_instr(phi))
            nir_src_rewrite(use, phi);
      }
   }

   return true;
}

/*
 * NIR indexes into array textures with unclamped floats (integer for txf). AGX
 * requires the index to be a clamped integer. Lower tex_src_coord into
 * tex_src_backend1 for array textures by type-converting and clamping.
 */
static bool
lower_regular_texture(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if (agx_nir_target(b->shader)->texture_exports)
      return lower_regular_texture_export(b, tex);

   if (nir_tex_instr_is_query(tex) && tex->op != nir_texop_lod)
      return false;

   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return lower_buffer_texture(b, tex);

   /* Don't lower twice */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;

   /* Get the coordinates */
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *ms_idx = nir_steal_tex_src(tex, nir_tex_src_ms_index);

   /* Apply txf workaround, see libagx_lower_txf_robustness */
   bool is_txf = ((tex->op == nir_texop_txf) || (tex->op == nir_texop_txf_ms));

   if (is_txf &&
       (has_nonzero_lod(tex) || tex->is_array ||
        nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) &&
       !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {

      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *lod =
         lod_idx >= 0 ? tex->src[lod_idx].src.ssa : nir_undef(b, 1, 16);

      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

      unsigned lidx = coord->num_components - 1;
      nir_def *layer = nir_channel(b, coord, lidx);

      nir_def *replaced = libagx_lower_txf_robustness(
         b, texture_descriptor_ptr(b, tex),
         nir_imm_bool(b, has_nonzero_lod(tex)), lod,
         nir_imm_bool(b, min_lod != NULL), min_lod ?: nir_undef(b, 1, 16),
         nir_imm_bool(b, tex->is_array), layer, nir_channel(b, coord, 0));

      coord = nir_vector_insert_imm(b, coord, replaced, 0);
   }

   /* The layer is always the last component of the NIR coordinate, split it off
    * because we'll need to swizzle.
    */
   nir_def *layer = NULL;

   if (tex->is_array && tex->op != nir_texop_lod) {
      unsigned lidx = coord->num_components - 1;
      nir_def *unclamped_layer = nir_channel(b, coord, lidx);
      coord = nir_trim_vector(b, coord, lidx);

      /* Round layer to nearest even */
      if (!is_txf) {
         unclamped_layer = nir_fround_even(b, unclamped_layer);

         /* Explicitly round negative to avoid undefined behaviour when constant
          * folding. This is load bearing on x86 builds.
          */
         unclamped_layer =
            nir_f2u32(b, nir_fmax(b, unclamped_layer, nir_imm_float(b, 0.0f)));
      }

      /* For a cube array, the layer is zero-indexed component 3 of the
       * coordinate but the number of layers is component 2 of the txs result.
       */
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         assert(lidx == 3 && "4 components");
         lidx = 2;
      }

      /* Clamp to max layer = (# of layers - 1) for out-of-bounds handling.
       * Layer must be 16-bits for the hardware, drop top bits after clamping.
       *
       * For txf, we drop out-of-bounds components rather than clamp, see the
       * above txf robustness workaround.
       */
      if (!(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP) && !is_txf) {
         nir_def *txs = nir_get_texture_size(b, tex);
         nir_def *nr_layers = nir_channel(b, txs, lidx);
         nir_def *max_layer = nir_iadd_imm(b, nr_layers, -1);
         layer = nir_umin(b, unclamped_layer, max_layer);
      } else {
         layer = unclamped_layer;
      }

      layer = nir_u2u16(b, layer);
   }

   /* Combine layer and multisample index into 32-bit so we don't need a vec5 or
    * vec6 16-bit coordinate tuple, which would be inconvenient in NIR for
    * little benefit (a minor optimization, I guess).
    */
   nir_def *sample_array = (ms_idx && layer)
                              ? nir_pack_32_2x16_split(b, ms_idx, layer)
                           : ms_idx ? nir_u2u32(b, ms_idx)
                           : layer  ? nir_u2u32(b, layer)
                                    : NULL;

   /* Combine into the final 32-bit tuple */
   if (sample_array != NULL) {
      unsigned end = coord->num_components;
      coord = nir_pad_vector(b, coord, end + 1);
      coord = nir_vector_insert_imm(b, coord, sample_array, end);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord);

   /* Furthermore, if there is an offset vector, it must be packed */
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);

   if (offset != NULL) {
      nir_def *packed = NULL;

      for (unsigned c = 0; c < offset->num_components; ++c) {
         nir_def *nibble = nir_iand_imm(b, nir_channel(b, offset, c), 0xF);
         nir_def *shifted = nir_ishl_imm(b, nibble, 4 * c);

         /* We pack with iadd instead of ior to let us fuse in the shift with an
          * iadd-lsl instruction.
          */
         if (packed != NULL)
            packed = nir_iadd(b, packed, shifted);
         else
            packed = shifted;
      }

      nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
   }

   if (nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 &&
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) {

      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      nir_def *packed = nir_pack_32_2x16_split(b, bias, min_lod);
      nir_tex_instr_add_src(tex, nir_tex_src_lod_bias_min_agx, packed);
   }

   /* We reserve bound sampler #0, so we offset bound samplers by 1 and
    * otherwise map bound samplers as-is.
    */
   nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!sampler)
      sampler = nir_imm_intN_t(b, tex->sampler_index, 16);

   if (!is_txf &&
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0) {

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                            nir_iadd_imm(b, nir_u2u16(b, sampler), 1));
   }

   return true;
}

static bool
legalize_image_lod(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_src *src;

#define CASE(op, idx)                                                          \
   case nir_intrinsic_##op:                                                    \
   case nir_intrinsic_bindless_##op:                                           \
      src = &intr->src[idx];                                                   \
      break;

   switch (intr->intrinsic) {
      CASE(image_load, 3)
      CASE(image_sparse_load, 3)
      CASE(image_store, 4)
      CASE(image_size, 1)
   default:
      return false;
   }

#undef CASE

   if (src->ssa->bit_size == 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(src, nir_i2i16(b, src->ssa));
   return true;
}

static nir_def *
txs_for_image(nir_builder *b, nir_intrinsic_instr *intr,
              unsigned num_components, unsigned bit_size, bool query_samples)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *lod = query_samples ? NULL : intr->src[1].ssa;
   nir_texop op = query_samples ? nir_texop_texture_samples : nir_texop_txs;

   nir_def *res =
      nir_build_tex(b, op, .texture_handle = intr->src[0].ssa, .lod = lod,
                    .dim = dim, .is_array = nir_intrinsic_image_array(intr),
                    .can_speculate = nir_instr_can_speculate(&intr->instr));

   /* Cube images are implemented as 2D arrays, so we need to divide here. */
   if (dim == GLSL_SAMPLER_DIM_CUBE && res->num_components > 2 &&
       !query_samples) {
      nir_def *divided = nir_udiv_imm(b, nir_channel(b, res, 2), 6);
      res = nir_vector_insert_imm(b, res, divided, 2);
   }

   return res;
}

static nir_def *
image_texel_address(nir_builder *b, nir_intrinsic_instr *intr,
                    bool return_index)
{
   /* First, calculate the address of the PBE descriptor */
   nir_def *desc_address =
      nir_load_from_texture_handle_agx(b, intr->src[0].ssa);

   nir_def *coord = intr->src[1].ssa;

   /* For atomics, we always infer the format. We only go down this path with
    * formatless intrinsics when lowering multisampled image stores, but that
    * uses the return_index path that ignores the block size.
    */
   enum pipe_format format = nir_intrinsic_format(intr);
   assert(return_index || format != PIPE_FORMAT_NONE);

   nir_def *blocksize_B = nir_imm_int(b, util_format_get_blocksize(format));

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool layered = nir_intrinsic_image_array(intr) ||
                  (dim == GLSL_SAMPLER_DIM_CUBE) ||
                  (dim == GLSL_SAMPLER_DIM_3D);

   if (dim == GLSL_SAMPLER_DIM_BUF && return_index) {
      return nir_channel(b, coord, 0);
   } else if (dim == GLSL_SAMPLER_DIM_BUF) {
      return libagx_buffer_texel_address(b, desc_address, coord, blocksize_B);
   } else {
      return libagx_image_texel_address(
         b, desc_address, coord, nir_u2u32(b, intr->src[2].ssa), blocksize_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index));
   }
}

static void
lower_buffer_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord_vector = intr->src[1].ssa;
   nir_def *coord = nir_channel(b, coord_vector, 0);

   /* If we're not bindless, assume we don't need an offset (GL driver) */
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_texture_offset(b, desc, coord);
   } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_image_offset(b, desc, coord);
   }

   /* Lower the buffer load/store to a 2D image load/store, matching the 2D
    * texture/PBE descriptor the driver supplies for buffer images.
    */
   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

static void
lower_1d_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord = intr->src[1].ssa;
   bool is_array = nir_intrinsic_image_array(intr);
   nir_def *zero = nir_imm_intN_t(b, 0, coord->bit_size);

   if (is_array) {
      assert(coord->num_components >= 2);
      coord =
         nir_vec3(b, nir_channel(b, coord, 0), zero, nir_channel(b, coord, 1));
   } else {
      assert(coord->num_components >= 1);
      coord = nir_vec2(b, coord, zero);
   }

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

/*
 * Just like for txf, we need special handling around layers (and LODs, but we
 * don't support mipmapped images yet) for robust image_loads. See
 * libagx_lower_txf_robustness for more info.
 */
static bool
lower_image_load_robustness(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS)
      return false;

   /* We only need to worry about array-like loads */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if (!nir_intrinsic_image_array(intr) && dim != GLSL_SAMPLER_DIM_CUBE)
      return false;

   /* Determine the coordinate component of the layer. Cubes and cube arrays
    * keep their array in their last non-array coordinate component, other
    * arrays are immediately after.
    */
   unsigned lidx = glsl_get_sampler_dim_coordinate_components(dim);
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      lidx--;

   nir_def *coord = intr->src[1].ssa;
   nir_def *lod = nir_undef(b, 1, 16);
   nir_def *layer = nir_channel(b, coord, lidx);

   /* image_load is effectively the same as txf, reuse the txf lower */
   nir_def *replaced = libagx_lower_txf_robustness(
      b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
      nir_imm_bool(b, false /* lower LOD */), lod,
      nir_imm_bool(b, false /* lower min LOD */), lod,
      nir_imm_bool(b, true /* lower layer */), layer, nir_channel(b, coord, 0));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, replaced, 0));
   return true;
}

static bool
lower_images(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store: {
      /* Legalize MSAA index */
      nir_src_rewrite(&intr->src[2], nir_u2u16(b, intr->src[2].ssa));

      if (intr->intrinsic == nir_intrinsic_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {
         lower_image_load_robustness(b, intr);
      }

      switch (nir_intrinsic_image_dim(intr)) {
      case GLSL_SAMPLER_DIM_1D:
         lower_1d_image(b, intr);
         return true;

      case GLSL_SAMPLER_DIM_BUF:
         lower_buffer_image(b, intr);
         return true;

      default:
         return true;
      }
   }

   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
      nir_def_rewrite_uses(
         &intr->def,
         txs_for_image(
            b, intr, intr->def.num_components, intr->def.bit_size,
            intr->intrinsic == nir_intrinsic_bindless_image_samples));
      return true;

   case nir_intrinsic_bindless_image_texel_address:
      nir_def_rewrite_uses(&intr->def, image_texel_address(b, intr, false));
      return true;

   case nir_intrinsic_is_sparse_texels_resident:
      /* Residency information is in bit 0, so we need to mask. Unclear what's
       * in the upper bits. For now, let's match the blob.
       */
      nir_def_replace(
         &intr->def,
         nir_ieq_imm(b, nir_iand_imm(b, intr->src[0].ssa, 1), AGX_RESIDENT));
      return true;

   case nir_intrinsic_sparse_residency_code_and:
      /* ior because residency codes are inverted from NIR */
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;

   case nir_intrinsic_image_size:
   case nir_intrinsic_image_texel_address:
      UNREACHABLE("should've been lowered");

   default:
      return false;
   }
}

/*
 * Map out-of-bounds storage texel buffer accesses and multisampled image stores
 * to -1 indices, which will become an out-of-bounds hardware access. This gives
 * cheap robustness2.
 */
static bool
lower_robustness(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
      break;
   default:
      return false;
   }

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool array = nir_intrinsic_image_array(intr);
   unsigned size_components = nir_image_intrinsic_coord_components(intr);

   nir_def *deref = intr->src[0].ssa;
   nir_def *coord = intr->src[1].ssa;

   if (dim != GLSL_SAMPLER_DIM_BUF &&
       !(dim == GLSL_SAMPLER_DIM_MS &&
         intr->intrinsic == nir_intrinsic_image_deref_store))
      return false;

   /* Bounds check the coordinate */
   nir_def *size =
      nir_image_deref_size(b, size_components, 32, deref, nir_imm_int(b, 0),
                           .image_dim = dim, .image_array = array);
   nir_def *oob = nir_bany(b, nir_uge(b, coord, size));

   /* Bounds check the sample */
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, deref, .image_dim = dim,
                                                 .image_array = array);

      oob = nir_ior(b, oob, nir_uge(b, intr->src[2].ssa, samples));
   }

   /* Replace the last coordinate component with a large coordinate for
    * out-of-bounds. We pick 0xFFF0 as it fits in 16-bit, and it is not signed
    * as 32-bit so we won't get in-bounds coordinates for arrays due to two's
    * complement wraparound. Additionally it still meets this requirement after
    * adding 0xF, the maximum tail offset.
    *
    * This ensures the resulting hardware coordinate is definitely
    * out-of-bounds, giving hardware-level robustness2 behaviour.
    */
   /* A buffer coordinate indexes elements, and a view can have more than
    * 0xFFF0 of them. Where buffer images are bounds-checked in software
    * against the element count, as with exported textures, use the largest
    * index instead.
    */
   const bool *software_buffer_bounds = data;
   unsigned c = size_components - 1;
   unsigned oob_coord = dim == GLSL_SAMPLER_DIM_BUF && *software_buffer_bounds
                           ? 0xFFFFFFFF
                           : 0xFFF0;
   nir_def *r =
      nir_bcsel(b, oob, nir_imm_int(b, oob_coord), nir_channel(b, coord, c));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, r, c));
   return true;
}

/*
 * Early texture lowering passes, called by the driver before lowering
 * descriptor bindings. That means these passes operate on texture derefs. The
 * purpose is to make descriptor crawls explicit in the NIR, so that the driver
 * can accurately lower descriptors after this pass but before calling
 * the full agx_nir_lower_texture.
 */
/* lower_sampler_lod_bias in FP32, which the fixed-point LOD word needs */
static bool
lower_sampler_lod_bias_export(nir_builder *b, nir_tex_instr *tex,
                              UNUSED void *data)
{
   if (!nir_tex_instr_need_sampler(tex) || tex->op == nir_texop_tg4 ||
       tex->op == nir_texop_lod_bias)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *bias = nir_build_texture_query(b, tex, nir_texop_lod_bias, 1,
                                           nir_type_float32, false, false);

   if (tex->op == nir_texop_txd) {
      /* Scale the derivatives by exp2(bias) */
      nir_def *ddx = nir_f2f32(b, nir_steal_tex_src(tex, nir_tex_src_ddx));
      nir_def *ddy = nir_f2f32(b, nir_steal_tex_src(tex, nir_tex_src_ddy));
      nir_def *scale = nir_fexp2(b, bias);

      nir_tex_instr_add_src(tex, nir_tex_src_ddx, nir_fmul(b, ddx, scale));
      nir_tex_instr_add_src(tex, nir_tex_src_ddy, nir_fmul(b, ddy, scale));
   } else if (tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, bias);
   } else {
      nir_tex_src_type src =
         tex->op == nir_texop_txl ? nir_tex_src_lod : nir_tex_src_bias;

      nir_def *orig = nir_steal_tex_src(tex, src);
      if (orig)
         bias = nir_fadd(b, bias, nir_f2f32(b, orig));

      nir_tex_instr_add_src(tex, src, bias);
   }

   return true;
}

/* Implicit-LOD compares of 2D array layers return the stored depth and of cube
 * array faces misread, so sample with derivatives scaled by the bias.
 */
static bool
lower_implicit_array_compare(nir_builder *b, nir_tex_instr *tex,
                             UNUSED void *data)
{
   bool derivatives =
      b->shader->info.stage == MESA_SHADER_FRAGMENT ||
      (b->shader->info.stage == MESA_SHADER_COMPUTE &&
       b->shader->info.derivative_group != DERIVATIVE_GROUP_NONE);

   if (!derivatives || !tex->is_shadow || !tex->is_array ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   nir_def *position = nir_trim_vector(b, coord, coord->num_components - 1);
   nir_def *ddx = nir_ddx(b, position);
   nir_def *ddy = nir_ddy(b, position);

   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias) {
      nir_def *scale = nir_fexp2(b, nir_f2fN(b, bias, ddx->bit_size));
      ddx = nir_fmul(b, ddx, scale);
      ddy = nir_fmul(b, ddy, scale);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   tex->op = nir_texop_txd;
   return true;
}

bool
agx_nir_lower_texture_early(nir_shader *s, bool support_lod_bias,
                            bool software_buffer_bounds)
{
   bool progress = false;
   bool exports = agx_nir_target(s)->texture_exports;

   if (exports && support_lod_bias) {
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_sampler_lod_bias_export,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_robustness,
            nir_metadata_control_flow, &software_buffer_bounds);

   nir_lower_tex_options lower_tex_options = {
      .lower_txp = ~0,
      .lower_invalid_implicit_lod = true,
      .lower_tg4_offsets = true,
      .lower_index_to_offset = true,
      .lower_sampler_lod_bias = support_lod_bias && !exports,

      /* Exported texel fetches have no offset form */
      .lower_txf_offset = exports,

      /* Unclear if/how mipmapped 1D textures work in the hardware. */
      .lower_1d = true,

      /* XXX: Metal seems to handle just like 3D txd, so why doesn't it work?
       * TODO: Stop using this lowering
       */
      .lower_txd_cube_map = true,
   };

   NIR_PASS(progress, s, nir_lower_tex, &lower_tex_options);

   if (exports) {
      bool gradients = false;
      NIR_PASS(gradients, s, nir_shader_tex_pass, lower_implicit_array_compare,
               nir_metadata_control_flow, NULL);

      /* Cube gradients become explicit LODs */
      if (gradients) {
         NIR_PASS(_, s, nir_lower_tex,
                  &(nir_lower_tex_options){.lower_txd_cube_map = true});
      }

      progress |= gradients;
   }

   return progress;
}

bool
agx_nir_lower_texture_queries(nir_shader *s)
{
   return nir_shader_instructions_pass(s, lower_tex_crawl,
                                       nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_images(nir_shader *s)
{
   bool progress = false;

   /* Insert fences before lowering image atomics, since image atomics need
    * different fencing than other image operations.
    */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, fence_image,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_image_atomics_to_global, NULL, NULL);

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, legalize_image_lod,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_images,
            nir_metadata_control_flow, NULL);
   return progress;
}

static bool
lower_txd_min_lod(const nir_tex_instr *tex, UNUSED const void *data)
{
   return nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 &&
          !native_min_lod(tex);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0 ||
       tex->op == nir_texop_txd || native_min_lod(tex))
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   if (tex->op == nir_texop_txf_ms) {
      /* Multisample views have one level */
      nir_def *min_lod =
         nir_f2f32(b, nir_steal_tex_src(tex, nir_tex_src_min_lod));
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *x = nir_bcsel(b, nir_fge_imm(b, min_lod, 1.0),
                             nir_imm_intN_t(b, -1, xyz->bit_size),
                             nir_channel(b, xyz, 0));
      nir_src_rewrite(&tex->src[coord].src,
                      nir_vector_insert_imm(b, xyz, x, 0));
      return true;
   } else if (tex->op == nir_texop_txf) {
      /* An out-of-range level is out of bounds (bound_fetch) */
      nir_def *min_lod =
         nir_f2f32(b, nir_steal_tex_src(tex, nir_tex_src_min_lod));
      int lod_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *level = lod_src >= 0 ? nir_u2u32(b, tex->src[lod_src].src.ssa)
                                    : nir_imm_int(b, 0);
      nir_def *min_level = nir_f2u32(b, nir_ffloor(b, min_lod));
      level =
         nir_bcsel(b, nir_uge(b, level, min_level), level, nir_imm_int(b, ~0));

      if (lod_src >= 0)
         nir_src_rewrite(&tex->src[lod_src].src, level);
      else
         nir_tex_instr_add_src(tex, nir_tex_src_lod, level);
      return true;
   }

   nir_def *lod;
   if (tex->op == nir_texop_txl) {
      lod = nir_steal_tex_src(tex, nir_tex_src_lod);
   } else if (tex->op == nir_texop_tex || tex->op == nir_texop_txb) {
      lod = nir_get_texture_lod(b, tex);
      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      if (bias)
         lod = nir_fadd(b, lod, nir_f2fN(b, bias, lod->bit_size));
   } else {
      return false;
   }

   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   lod = nir_fmax(b, nir_f2f32(b, lod), nir_f2f32(b, min_lod));
   nir_tex_instr_add_src(tex, nir_tex_src_lod, lod);
   tex->op = nir_texop_txl;
   return true;
}

/* Flag offsets the sample cannot encode, for the software path */
static bool
flag_offsets(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset < 0 || tex->op == nir_texop_txf ||
       (tex->backend_flags & AGX_TEXTURE_FLAG_OFFSET_FALLBACK))
      return false;

   bool volume = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   bool implicit_2d = !tex->is_array && !tex->is_shadow && !volume &&
                      tex->sampler_dim == GLSL_SAMPLER_DIM_2D;
   bool min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0;

   /* The final form, as lower_regular_texture_export and the backend pick it */
   agx_instr I = {
      .op = AGX_OPCODE_TEXTURE_SAMPLE,
      .dim = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? AGX_DIM_3D
             : tex->is_array                         ? AGX_DIM_2D_ARRAY
                                                     : AGX_DIM_2D,
      .shadow = tex->is_shadow,
      .offset = true,
      .gather = tex->op == nir_texop_tg4 ? AGX_GATHER_R : AGX_GATHER_NONE,
      .mask = 0xf,
   };

   switch (tex->op) {
   case nir_texop_txd:
      I.lod_mode = AGX_LOD_MODE_LOD_GRAD;
      break;
   case nir_texop_txb:
      I.lod_mode = AGX_LOD_MODE_AUTO_LOD_BIAS;
      break;
   case nir_texop_tex:
      I.lod_mode =
         implicit_2d ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_AUTO_LOD_BIAS;
      break;
   case nir_texop_tg4:
      I.lod_mode = implicit_2d ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN;
      break;
   case nir_texop_txl:
      I.lod_mode = AGX_LOD_MODE_LOD_MIN;
      break;
   default:
      return false;
   }

   if (nir_src_is_const(tex->src[offset].src) && !min_lod &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE && agx3_texture_encodable(&I))
      return false;

   tex->backend_flags |= AGX_TEXTURE_FLAG_OFFSET_FALLBACK;
   return true;
}

/* A texel of a buffer fetched by texture index, from its linear 2D view */
static nir_def *
fetch_buffer(nir_builder *b, const nir_tex_instr *buf, nir_def *index)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->texture_index = buf->texture_index;
   tex->dest_type = buf->dest_type;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_FETCH_BOUNDED;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec2(b, nir_umod_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH),
               nir_udiv_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* Texel buffers by index; RGB32 is fetched as three R32 words and swizzled */
static bool
lower_texel_buffer_index(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->op != nir_texop_txf ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *index = nir_u2u32(b, nir_get_tex_src(tex, nir_tex_src_coord));
   nir_def *size = export_texture_info(b, tex, TEXTURE_INFO_WIDTH);
   nir_push_if(b, nir_ult(b, index, size));
   nir_push_if(
      b, nir_ine_imm(b, export_texture_info(b, tex, TEXTURE_INFO_RGB32), 0));

   nir_def *components[4];
   nir_def *scalar_index = nir_imul_imm(b, index, 3);
   for (unsigned c = 0; c < 3; ++c) {
      components[c] = nir_channel(
         b, fetch_buffer(b, tex, nir_iadd_imm(b, scalar_index, c)), 0);
   }

   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   components[3] = floating ? nir_imm_float(b, 1) : nir_imm_int(b, 1);

   nir_def *swizzle = export_texture_info(b, tex, TEXTURE_INFO_BUFFER_SWIZZLE);
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *channel = nir_iand_imm(b, nir_ushr_imm(b, swizzle, c * 3), 7);
      rgba[c] = nir_imm_int(b, 0);
      for (unsigned j = 0; j < 4; ++j)
         rgba[c] =
            nir_bcsel(b, nir_ieq_imm(b, channel, j), components[j], rgba[c]);
      rgba[c] = nir_bcsel(b, nir_ieq_imm(b, channel, PIPE_SWIZZLE_1),
                          components[3], rgba[c]);
   }

   nir_def *rgb32 = nir_vec(b, rgba, 4);
   nir_push_else(b, NULL);
   nir_def *native = fetch_buffer(b, tex, index);
   nir_pop_if(b, NULL);
   nir_def *texel = nir_if_phi(b, rgb32, native);
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_ivec4(b, 0, 0, 0, 0);
   nir_pop_if(b, NULL);

   nir_def *result = nir_if_phi(b, texel, zero);
   nir_def_replace(&tex->def,
                   nir_trim_vector(b, result, tex->def.num_components));
   return true;
}

static void
legalize_texture_sources(nir_shader *nir)
{
   /* Frontends and common image lowering can supply narrow LODs/coordinates.
    * Native lowering performs arithmetic in 32 bits before packing operands. */
   nir_tex_src_type_constraints constraints = {
      [nir_tex_src_coord] = {true, 32},
      [nir_tex_src_lod] = {true, 32},
      [nir_tex_src_bias] = {true, 32},
      [nir_tex_src_ms_index] = {true, 32},
      [nir_tex_src_comparator] = {true, 32},
      [nir_tex_src_ddx] = {true, 32},
      [nir_tex_src_ddy] = {true, 32},
      [nir_tex_src_offset] = {true, 32},
      [nir_tex_src_min_lod] = {true, 32},
   };
   nir_legalize_16bit_sampler_srcs(nir, constraints);
}

/* A texel offset is applied separately at each selected mip level, before
 * wrapping and filtering. Express that contract with ordinary NIR fetches;
 * the backend supplies its private nearest sampler for those fetches. */
/* A handle-addressed descriptor carries the records otherwise published in
 * the per-draw sysvals. */
static nir_def *
descriptor_word(nir_builder *b, nir_def *handle, unsigned offset)
{
   nir_def *address = agx_nir_descriptor_address(b, handle, offset);
   /* EXP: descriptor records are constant for the draw; speculating them
    * lets the preamble hoist uniform sampler work out of control flow. */
   return nir_load_global_constant(
      b, 1, 32, address, .align_mul = 4,
      .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
}

static nir_def *
descriptor_texture_info(nir_builder *b, const nir_tex_instr *tex,
                        unsigned field)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle >= 0)
      return descriptor_word(b, tex->src[handle].src.ssa,
                             AGX_DESCRIPTOR_TEXTURE_INFO + field * 4);
   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_TEXTURE_INFO_OFFSET +
                        tex->texture_index * AGX_TEXTURE_INFO_STRIDE +
                        field * 4),
      .align_mul = 4, .range = 4);
}

static nir_def *
wrap_texel(nir_builder *b, nir_def *i, nir_def *size, unsigned wrap)
{
   if (wrap == PIPE_TEX_WRAP_REPEAT)
      return nir_imod(b, i, size);
   if (wrap == PIPE_TEX_WRAP_MIRROR_REPEAT) {
      nir_def *period = nir_ishl_imm(b, size, 1);
      nir_def *j = nir_imod(b, i, period);
      return nir_imin(b, j, nir_isub(b, nir_iadd_imm(b, period, -1), j));
   }
   return nir_iclamp(b, i, nir_imm_int(b, 0), nir_iadd_imm(b, size, -1));
}

static nir_def *
compare_depth(nir_builder *b, nir_def *ref, nir_def *depth, unsigned func)
{
   nir_def *test;
   switch (func) {
   case PIPE_FUNC_NEVER:
      return nir_imm_float(b, 0);
   case PIPE_FUNC_ALWAYS:
      return nir_imm_float(b, 1);
   case PIPE_FUNC_LESS:
      test = nir_flt(b, ref, depth);
      break;
   case PIPE_FUNC_LEQUAL:
      test = nir_fge(b, depth, ref);
      break;
   case PIPE_FUNC_GREATER:
      test = nir_flt(b, depth, ref);
      break;
   case PIPE_FUNC_GEQUAL:
      test = nir_fge(b, ref, depth);
      break;
   case PIPE_FUNC_EQUAL:
      test = nir_feq(b, ref, depth);
      break;
   default:
      test = nir_fneu(b, ref, depth);
      break;
   }
   return nir_b2f32(b, test);
}

struct offset_sample {
   nir_tex_instr *tex;
   unsigned dims;
   nir_def *coord, *offset, *comparator, *layer;
   nir_def *base_size[3];
   nir_def *wrap[3], *min, *mag, *mip, *compare, *min_lod, *max_lod, *bias,
      *border;
};

nir_def *
agx_sampler_field(nir_builder *b, const nir_tex_instr *tex, unsigned field)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (handle >= 0)
      return descriptor_word(b, tex->src[handle].src.ssa,
                             AGX_DESCRIPTOR_RUNTIME_SAMPLER + field * 4);
   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_RUNTIME_SAMPLER_OFFSET +
                        tex->sampler_index * AGX_RUNTIME_SAMPLER_STRIDE +
                        4 * field),
      .align_mul = 4, .range = 4);
}

static bool
normalize_runtime_coordinates(nir_builder *b, nir_tex_instr *tex,
                              UNUSED void *data)
{
   /* Unnormalized samplers are restricted to single-level 1D/2D views.
    * The frontend has already expanded 1D to 2D with unit height, and may
    * have rewritten 2D as layer 0 of a one-layer array. Vulkan also forbids
    * them with ImplicitLod, Dref and Proj sampling and with any bias or
    * offset (VUID-vkCmdDraw-None-02703/02704), so those samples never take
    * pixel coordinates. */
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       (tex->is_array && !(tex->backend_flags & AGX_TEXTURE_FLAG_ARRAYED_2D)) ||
       nir_tex_instr_is_query(tex) || tex->op == nir_texop_txf ||
       tex->op == nir_texop_tex || tex->op == nir_texop_txb || tex->is_shadow ||
       nir_tex_instr_src_index(tex, nir_tex_src_projector) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0)
      return false;
   int src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (src < 0 || tex->texture_index >= 32 || tex->sampler_index >= 32)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *pixel = agx_sampler_field(b, tex, AGX_SAMP_PIXEL_COORDS);
   nir_def *coord = tex->src[src].src.ssa;
   nir_def *normalized[NIR_MAX_VEC_COMPONENTS];

   /* Scale by the reciprocal size or one, hoisted to the preamble */
   for (unsigned c = 0; c < coord->num_components; ++c) {
      if (c < 2) {
         nir_def *size = nir_u2f32(b, descriptor_texture_info(b, tex, c));
         nir_def *scale = nir_bcsel(b, nir_ine_imm(b, pixel, 0),
                                    nir_frcp(b, size), nir_imm_float(b, 1.0f));
         normalized[c] = nir_fmul(b, nir_channel(b, coord, c), scale);
      } else {
         normalized[c] = nir_channel(b, coord, c);
      }
   }

   nir_src_rewrite(&tex->src[src].src,
                   nir_vec(b, normalized, coord->num_components));
   return true;
}

/* Explicit-LOD sample of tex's view and sampler outside the image on every
 * axis (keeping an array layer), as a 32-bit vec4. */
static nir_def *
sample_border(nir_builder *b, nir_tex_instr *tex, nir_def *coord, unsigned dims)
{
   static const nir_tex_src_type copied[] = {
      nir_tex_src_texture_deref,  nir_tex_src_sampler_deref,
      nir_tex_src_texture_offset, nir_tex_src_sampler_offset,
      nir_tex_src_texture_handle, nir_tex_src_sampler_handle,
   };
   unsigned count = 2;
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i)
      count += nir_tex_instr_src_index(tex, copied[i]) >= 0;
   nir_tex_instr *bt = nir_tex_instr_create(b->shader, count);
   bt->op = nir_texop_txl;
   bt->sampler_dim = tex->sampler_dim;
   bt->is_array = tex->is_array;
   bt->coord_components = coord->num_components;
   bt->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   bt->texture_index = tex->texture_index;
   bt->sampler_index = tex->sampler_index;
   bt->backend_flags = tex->backend_flags | AGX_TEXTURE_FLAG_BIASED_LEVEL;
   bt->can_speculate = tex->can_speculate;
   nir_def *comps[4];
   for (unsigned c = 0; c < coord->num_components; ++c)
      comps[c] = c < dims ? nir_imm_float(b, -16.0f) : nir_channel(b, coord, c);
   unsigned n = 0;
   bt->src[n++] = (nir_tex_src){
      .src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec(b, comps, coord->num_components))};
   bt->src[n++] = (nir_tex_src){.src_type = nir_tex_src_lod,
                                .src = nir_src_for_ssa(nir_imm_float(b, 0))};
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i) {
      int src = nir_tex_instr_src_index(tex, copied[i]);
      if (src >= 0)
         bt->src[n++] =
            (nir_tex_src){.src_type = copied[i],
                          .src = nir_src_for_ssa(tex->src[src].src.ssa)};
   }
   nir_def_init(&bt->instr, &bt->def, 4, 32);
   nir_builder_instr_insert(b, &bt->instr);
   return &bt->def;
}

static nir_def *
wrap_dynamic(nir_builder *b, nir_def *i, nir_def *size, nir_def *wrap)
{
   nir_def *edge = wrap_texel(b, i, size, PIPE_TEX_WRAP_CLAMP_TO_EDGE);
   nir_def *repeat = wrap_texel(b, i, size, PIPE_TEX_WRAP_REPEAT);
   nir_def *mirror = wrap_texel(b, i, size, PIPE_TEX_WRAP_MIRROR_REPEAT);
   nir_def *mirror_edge =
      nir_imin(b,
               nir_bcsel(b, nir_ilt_imm(b, i, 0),
                         nir_iadd_imm(b, nir_ineg(b, i), -1), i),
               nir_iadd_imm(b, size, -1));
   return nir_bcsel(
      b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_REPEAT), repeat,
      nir_bcsel(
         b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_MIRROR_REPEAT), mirror,
         nir_bcsel(b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE),
                   mirror_edge, edge)));
}

static nir_def *
fetch(nir_builder *b, const struct offset_sample *s, nir_def *coord,
      nir_def *lod, nir_def *border)
{
   int handle = nir_tex_instr_src_index(s->tex, nir_tex_src_texture_handle);
   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 2 + (handle >= 0));
   txf->op = nir_texop_txf;
   txf->sampler_dim = s->tex->sampler_dim;
   txf->is_array = s->tex->is_array;
   txf->texture_index = s->tex->texture_index;
   txf->sampler_index = s->tex->sampler_index;
   txf->coord_components = coord->num_components;
   txf->dest_type = s->tex->dest_type;
   txf->src[0] = (nir_tex_src){.src_type = nir_tex_src_coord,
                               .src = nir_src_for_ssa(coord)};
   txf->src[1] =
      (nir_tex_src){.src_type = nir_tex_src_lod, .src = nir_src_for_ssa(lod)};
   if (handle >= 0) {
      txf->src[2] =
         (nir_tex_src){.src_type = nir_tex_src_texture_handle,
                       .src = nir_src_for_ssa(s->tex->src[handle].src.ssa)};
   }
   nir_def_init(&txf->instr, &txf->def,
                s->tex->is_shadow ? 4 : s->tex->def.num_components, 32);
   nir_builder_instr_insert(b, &txf->instr);
   nir_def *value = nir_bcsel(
      b, border, nir_trim_vector(b, s->border, txf->def.num_components),
      &txf->def);
   if (!s->tex->is_shadow)
      return value;
   nir_def *result = nir_imm_float(b, 0);
   for (unsigned f = PIPE_FUNC_NEVER; f <= PIPE_FUNC_ALWAYS; ++f)
      result = nir_bcsel(
         b, nir_ieq_imm(b, s->compare, f),
         compare_depth(b, s->comparator, nir_channel(b, value, 0), f), result);
   return result;
}

static nir_def *
filter_level(nir_builder *b, const struct offset_sample *s, nir_def *lod,
             bool linear)
{
   nir_def *size[3], *base[3], *weight[3];
   for (unsigned c = 0; c < s->dims; ++c) {
      size[c] =
         nir_imax(b, nir_ushr(b, s->base_size[c], lod), nir_imm_int(b, 1));
      nir_def *p =
         nir_fmul(b, nir_channel(b, s->coord, c), nir_u2f32(b, size[c]));
      if (linear)
         p = nir_fadd_imm(b, p, -0.5);
      nir_def *floor = nir_ffloor(b, p);
      base[c] = nir_iadd(b, nir_f2i32(b, floor), nir_channel(b, s->offset, c));
      weight[c] = nir_fsub(b, p, floor);
   }
   unsigned count = linear ? 1u << s->dims : 1;
   nir_def *taps[8];
   for (unsigned tap = 0; tap < count; ++tap) {
      nir_def *coord[4];
      nir_def *border = nir_imm_false(b);
      for (unsigned c = 0; c < s->dims; ++c) {
         nir_def *i = nir_iadd_imm(b, base[c], (tap >> c) & 1);
         coord[c] = wrap_dynamic(b, i, size[c], s->wrap[c]);
         border = nir_ior(
            b, border,
            nir_iand(b,
                     nir_ieq_imm(b, s->wrap[c], PIPE_TEX_WRAP_CLAMP_TO_BORDER),
                     nir_uge(b, i, size[c])));
      }
      if (s->tex->is_array)
         coord[s->dims] = s->layer;
      taps[tap] = fetch(b, s, nir_vec(b, coord, s->dims + s->tex->is_array),
                        lod, border);
   }
   /* Pair adjacent X taps, then Y and Z; no fixed registers or sampled values
    * escape this ordinary SSA filtering expression. */
   for (unsigned c = 0; linear && c < s->dims; ++c) {
      count /= 2;
      for (unsigned i = 0; i < count; ++i)
         taps[i] = nir_flrp(b, taps[2 * i], taps[2 * i + 1], weight[c]);
   }
   return taps[0];
}

static nir_def *
filter_dynamic(nir_builder *b, const struct offset_sample *s, nir_def *lod,
               nir_def *filter)
{
   /* Integer textures cannot be linearly filtered. Avoid emitting dead float
    * interpolation of integer texels even before constant folding. */
   if (!s->tex->is_shadow &&
       nir_alu_type_get_base_type(s->tex->dest_type) != nir_type_float)
      return filter_level(b, s, lod, false);
   nir_push_if(b, nir_ieq_imm(b, filter, PIPE_TEX_FILTER_LINEAR));
   nir_def *linear = filter_level(b, s, lod, true);
   nir_push_else(b, NULL);
   nir_def *nearest = filter_level(b, s, lod, false);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, linear, nearest);
}

static nir_def *
first_level(nir_builder *b, const struct offset_sample *s, nir_def *lod,
            nir_def *lambda)
{
   nir_def *filter =
      nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0), lambda), s->mag, s->min);
   return filter_dynamic(b, s, lod, filter);
}

/* textureGather returns one component of each texel in the level-zero 2x2
 * bilinear footprint, ordered (i0, j1), (i1, j1), (i1, j0), (i0, j0). Depth
 * comparisons apply to each texel.
 */
static nir_def *
gather(nir_builder *b, const struct offset_sample *s)
{
   nir_def *lod = nir_imm_int(b, 0);
   nir_def *base[2];
   for (unsigned c = 0; c < 2; ++c) {
      nir_def *p = nir_fadd_imm(b,
                                nir_fmul(b, nir_channel(b, s->coord, c),
                                         nir_u2f32(b, s->base_size[c])),
                                -0.5);
      base[c] = nir_f2i32(b, nir_ffloor(b, p));
      if (s->offset)
         base[c] = nir_iadd(b, base[c], nir_channel(b, s->offset, c));
   }
   static const uint8_t taps[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
   nir_def *result[4];
   for (unsigned tap = 0; tap < 4; ++tap) {
      nir_def *coord[3];
      nir_def *border = nir_imm_false(b);
      for (unsigned c = 0; c < 2; ++c) {
         nir_def *i = nir_iadd_imm(b, base[c], taps[tap][c]);
         coord[c] = wrap_dynamic(b, i, s->base_size[c], s->wrap[c]);
         border = nir_ior(
            b, border,
            nir_iand(b,
                     nir_ieq_imm(b, s->wrap[c], PIPE_TEX_WRAP_CLAMP_TO_BORDER),
                     nir_uge(b, i, s->base_size[c])));
      }
      if (s->tex->is_array)
         coord[2] = s->layer;
      nir_def *value =
         fetch(b, s, nir_vec(b, coord, 2 + s->tex->is_array), lod, border);
      result[tap] =
         s->tex->is_shadow ? value : nir_channel(b, value, s->tex->component);
   }
   return nir_vec(b, result, 4);
}

static nir_def *
sample_lod(nir_builder *b, const struct offset_sample *s)
{
   nir_tex_instr *tex = s->tex;
   int explicit_lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod;
   if (explicit_lod >= 0) {
      lod = tex->src[explicit_lod].src.ssa;
   } else if (tex->op != nir_texop_txd &&
              b->shader->info.stage != MESA_SHADER_FRAGMENT) {
      lod = nir_imm_float(b, 0);
   } else {
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa;
         dy = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa;
      } else {
         nir_def *spatial = nir_trim_vector(b, s->coord, s->dims);
         dx = nir_ddx(b, spatial);
         dy = nir_ddy(b, spatial);
      }
      nir_def *xx = nir_imm_float(b, 0), *yy = nir_imm_float(b, 0);
      for (unsigned c = 0; c < s->dims; ++c) {
         nir_def *scale = nir_u2f32(b, s->base_size[c]);
         nir_def *x = nir_fmul(b, nir_channel(b, dx, c), scale);
         nir_def *y = nir_fmul(b, nir_channel(b, dy, c), scale);
         xx = nir_fadd(b, xx, nir_fmul(b, x, x));
         yy = nir_fadd(b, yy, nir_fmul(b, y, y));
      }
      lod = nir_fmul_imm(b, nir_flog2(b, nir_fmax(b, xx, yy)), 0.5);
   }
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0)
      lod = nir_fadd(b, lod, tex->src[bias].src.ssa);
   lod = nir_fadd(b, lod, s->bias);
   /* A shader LOD clamp raises the sampler's minimum. */
   nir_def *min_lod = s->min_lod;
   int shader_min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (shader_min_lod >= 0)
      min_lod = nir_fmax(b, min_lod, tex->src[shader_min_lod].src.ssa);
   return nir_fclamp(b, lod, min_lod, s->max_lod);
}

/* A native explicit-LOD sample of one mip level, with the texel offset applied
 * as a coordinate shift. Vulkan adds offsets to integer texel coordinates
 * before wrapping and filtering, so at a single level the offset equals
 * offset / level_size in normalised coordinates; the shift's rounding error
 * (about one ulp of the coordinate) is far below the sub-texel precision.
 */
static nir_def *
native_offset_level(nir_builder *b, const struct offset_sample *s,
                    nir_def *level)
{
   nir_tex_instr *tex = s->tex;
   static const nir_tex_src_type copied[] = {
      nir_tex_src_texture_deref,  nir_tex_src_sampler_deref,
      nir_tex_src_texture_offset, nir_tex_src_sampler_offset,
      nir_tex_src_texture_handle, nir_tex_src_sampler_handle,
      nir_tex_src_comparator,
   };
   unsigned count = 2;
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i)
      count += nir_tex_instr_src_index(tex, copied[i]) >= 0;
   nir_def *comps[4];
   for (unsigned c = 0; c < s->coord->num_components; ++c) {
      comps[c] = nir_channel(b, s->coord, c);
      if (c < s->dims) {
         nir_def *size =
            nir_umax(b, nir_ushr(b, s->base_size[c], level), nir_imm_int(b, 1));
         nir_def *shift =
            nir_fmul(b, nir_i2f32(b, nir_channel(b, s->offset, c)),
                     nir_frcp(b, nir_u2f32(b, size)));
         comps[c] = nir_fadd(b, comps[c], shift);
      }
   }
   nir_tex_instr *nt = nir_tex_instr_create(b->shader, count);
   nt->op = nir_texop_txl;
   nt->sampler_dim = tex->sampler_dim;
   nt->is_array = tex->is_array;
   nt->is_shadow = tex->is_shadow;
   nt->is_new_style_shadow = tex->is_new_style_shadow;
   nt->coord_components = s->coord->num_components;
   nt->dest_type = tex->dest_type;
   nt->texture_index = tex->texture_index;
   nt->sampler_index = tex->sampler_index;
   nt->backend_flags = tex->backend_flags | AGX_TEXTURE_FLAG_BIASED_LEVEL;
   nt->can_speculate = tex->can_speculate;
   unsigned n = 0;
   nt->src[n++] = (nir_tex_src){
      .src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec(b, comps, s->coord->num_components))};
   nt->src[n++] = (nir_tex_src){.src_type = nir_tex_src_lod,
                                .src = nir_src_for_ssa(nir_i2f32(b, level))};
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i) {
      int src = nir_tex_instr_src_index(tex, copied[i]);
      if (src >= 0)
         nt->src[n++] =
            (nir_tex_src){.src_type = copied[i],
                          .src = nir_src_for_ssa(tex->src[src].src.ssa)};
   }
   nir_def_init(&nt->instr, &nt->def, tex->def.num_components,
                tex->def.bit_size);
   nir_builder_instr_insert(b, &nt->instr);
   return &nt->def;
}

static bool
agx_texture_native_offset(const nir_tex_instr *tex, uint16_t *bits)
{
   int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   /* The shape of an encodable offset. Whether the sample's final form has
    * a measured offset variant is decided by the backend, which lowers the
    * offset otherwise (AGX_TEXTURE_FLAG_OFFSET_FALLBACK). */
   if (offset < 0 || !nir_src_is_const(tex->src[offset].src) ||
       tex->op == nir_texop_txf ||
       (tex->backend_flags & AGX_TEXTURE_FLAG_OFFSET_FALLBACK) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
        tex->op != nir_texop_tg4) ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_3D) ||
       /* The minimum-LOD forms are native only without an offset. */
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0)
      return false;
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2;
   nir_src src = tex->src[offset].src;
   if (src.ssa->num_components < dims)
      return false;
   uint16_t packed = 0;
   for (unsigned c = 0; c < dims; ++c) {
      int64_t v = nir_src_comp_as_int(src, c);
      if (v < -8 || v > 7)
         return false;
      packed |= (uint16_t)(v & 15) << (4 * c);
   }
   if (bits)
      *bits = packed;
   return true;
}

/* With bias_applied, the sampler LOD bias is already in the LOD or bias */
static bool
lower_texture_offsets(nir_shader *nir, bool bias_applied, const char **reason)
{
   legalize_texture_sources(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
         /* Gathers without offsets are native. */
         bool is_gather = tex->op == nir_texop_tg4;
         if (offset < 0 || tex->op == nir_texop_txf)
            continue;
         /* The sample instruction encodes these offsets itself unless the
          * backend finds no offset form for the sample's final form. It then
          * lowers the offset with runtime sampler state, which the driver
          * must supply for any stage that samples with an offset, static
          * key or not. */
         if (agx_texture_native_offset(tex, NULL))
            continue;
         if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
              tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
              !is_gather) ||
             tex->texture_index >= 32 || tex->sampler_index >= 32 ||
             (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
              (tex->sampler_dim != GLSL_SAMPLER_DIM_3D || is_gather))) {
            *reason =
               "software sampling requires a bound 2D, array, or 3D texture";
            return false;
         }
         unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2;
         struct offset_sample s = {
            .tex = tex,
            .dims = dims,
            .coord =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa,
            .offset = offset >= 0 ? tex->src[offset].src.ssa : NULL};
         b.cursor = nir_before_instr(instr);
         {
            for (unsigned c = 0; c < s.dims; ++c)
               s.wrap[c] = agx_sampler_field(&b, tex, AGX_SAMP_WRAP_S + c);
            s.min = agx_sampler_field(&b, tex, AGX_SAMP_MIN);
            s.mag = agx_sampler_field(&b, tex, AGX_SAMP_MAG);
            s.mip = agx_sampler_field(&b, tex, AGX_SAMP_MIP);
            s.compare = agx_sampler_field(&b, tex, AGX_SAMP_COMPARE);
            s.min_lod = agx_sampler_field(&b, tex, AGX_SAMP_MIN_LOD);
            s.max_lod = agx_sampler_field(&b, tex, AGX_SAMP_MAX_LOD);
            s.bias = bias_applied ? nir_imm_float(&b, 0)
                                  : agx_sampler_field(&b, tex, AGX_SAMP_BIAS);
            nir_def *preset = agx_sampler_field(&b, tex, AGX_SAMP_BORDER);
            bool floating =
               tex->is_shadow ||
               nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
            nir_def *one = floating ? nir_imm_float(&b, 1) : nir_imm_int(&b, 1);
            nir_def *rgb = nir_bcsel(&b, nir_ieq_imm(&b, preset, 2), one,
                                     nir_imm_int(&b, 0));
            nir_def *alpha = nir_bcsel(&b, nir_ine_imm(&b, preset, 0), one,
                                       nir_imm_int(&b, 0));
            /* Fetches use the view's composed format/component swizzle. Apply
             * that same mapping to software border taps before selecting a
             * gather component, including constant zero/one channels.
             */
            nir_def *swizzle = descriptor_texture_info(&b, tex, 7);
            nir_def *rgba[4];
            for (unsigned c = 0; c < 4; ++c) {
               nir_def *channel =
                  nir_iand_imm(&b, nir_ushr_imm(&b, swizzle, 3 * c), 7);
               nir_def *value = nir_bcsel(
                  &b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_W), alpha, rgb);
               value = nir_bcsel(&b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_1),
                                 one, value);
               rgba[c] = nir_bcsel(&b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_0),
                                   nir_imm_int(&b, 0), value);
            }
            s.border = nir_vec(&b, rgba, 4);
            /* A custom border colour (preset 3) is known only to the native
             * sampler record, in the view's encoding. Take it from the
             * hardware: an explicit-LOD sample outside the image on every
             * axis returns the swizzled border. */
            if (!tex->is_shadow) {
               nir_push_if(&b, nir_ieq_imm(&b, preset, 3));
               nir_def *custom = sample_border(&b, tex, s.coord, s.dims);
               nir_pop_if(&b, NULL);
               s.border = nir_if_phi(&b, custom, s.border);
            }
         }
         for (unsigned c = 0; c < s.dims; ++c)
            s.base_size[c] = descriptor_texture_info(&b, tex, c);
         if (tex->is_array) {
            s.layer = nir_f2i32(
               &b,
               nir_ffloor(
                  &b, nir_fadd_imm(&b, nir_channel(&b, s.coord, s.dims), 0.5)));
            s.layer = nir_iclamp(
               &b, s.layer, nir_imm_int(&b, 0),
               nir_iadd_imm(&b, descriptor_texture_info(&b, tex, 2), -1));
         }
         if (tex->is_shadow)
            s.comparator =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_comparator)]
                  .src.ssa;
         if (is_gather) {
            nir_def_replace(&tex->def, gather(&b, &s));
            progress = true;
            continue;
         }
         nir_def *lambda = sample_lod(&b, &s);
         nir_def *last =
            nir_iadd_imm(&b, descriptor_texture_info(&b, tex, 3), -1);
         nir_def *level =
            nir_fclamp(&b, lambda, nir_imm_float(&b, 0), nir_i2f32(&b, last));
         nir_def *nearest_lod = nir_imax(
            &b,
            nir_iadd_imm(
               &b, nir_f2i32(&b, nir_fceil(&b, nir_fadd_imm(&b, level, 0.5))),
               -1),
            nir_imm_int(&b, 0));
         nir_def *lod = nir_bcsel(
            &b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_NONE),
            nir_imm_int(&b, 0),
            nir_bcsel(&b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_NEAREST),
                      nearest_lod, nir_f2i32(&b, nir_ffloor(&b, level))));
         bool floating = tex->is_shadow || nir_alu_type_get_base_type(
                                              tex->dest_type) == nir_type_float;
         /* Runtime sampler state: let the hardware filter each level when
          * an explicit integer LOD selects exactly the level(s) the
          * software path would filter. That holds unless level 0 is
          * minified with differing min/mag filters (an explicit LOD of 0
          * selects the magnification filter), or the sampler's LOD clamp
          * would move the explicit LOD off the integer level.
          */
         nir_def *native, *fast;
         {
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *linear = nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_LINEAR);
            fast = nir_iand(
               &b,
               nir_ior(&b, nir_ieq(&b, s.min, s.mag),
                       nir_ior(&b, nir_fge(&b, nir_imm_float(&b, 0), lambda),
                               nir_ige_imm(&b, lod, 1))),
               nir_iand(&b, nir_fge(&b, nir_i2f32(&b, lod), s.min_lod),
                        nir_ior(&b, nir_inot(&b, linear),
                                nir_fge(&b, s.max_lod, nir_i2f32(&b, next)))));
            nir_push_if(&b, fast);
            native = native_offset_level(&b, &s, lod);
            if (floating) {
               nir_push_if(&b, linear);
               nir_def *second = native_offset_level(&b, &s, next);
               nir_def *mixed =
                  nir_flrp(&b, native, second, nir_ffract(&b, level));
               nir_pop_if(&b, NULL);
               native = nir_if_phi(&b, mixed, native);
            }
            nir_push_else(&b, NULL);
         }
         nir_def *value = first_level(&b, &s, lod, lambda);
         if (floating) {
            nir_push_if(&b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_LINEAR));
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *second = filter_dynamic(&b, &s, next, s.min);
            nir_def *mixed = nir_flrp(&b, value, second, nir_ffract(&b, level));
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, mixed, value);
         }
         nir_pop_if(&b, NULL);
         value = nir_if_phi(&b, native, value);
         nir_def_rewrite_uses(&tex->def, value);
         nir_instr_remove(instr);
         progress = true;
      }
   }
   nir->info.flrp_lowered = false;
   nir_progress(progress, impl, nir_metadata_none);
   return true;
}

/* Runs after agx_nir_lower_texture, which applied the sampler LOD bias */
static bool
lower_sampler_state_runtime(nir_shader *nir)
{
   const char *reason = NULL;
   legalize_texture_sources(nir);
   bool progress = nir_shader_tex_pass(nir, normalize_runtime_coordinates,
                                       nir_metadata_control_flow, NULL);
   progress |= lower_texture_offsets(nir, true, &reason);
   assert(reason == NULL && "offsets on cube maps or buffers are invalid");
   return progress;
}

static bool
lower_texture_exports(nir_shader *s)
{
   bool progress = false;

   /* Residency first: it reads the buffer and image dimensions */
   NIR_PASS(progress, s, agx_nir_lower_sparse_residency);
   NIR_PASS(progress, s, agx_nir_lower_texel_buffers);
   NIR_PASS(progress, s, nir_shader_tex_pass, lower_texel_buffer_index,
            nir_metadata_none, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, flag_offsets, nir_metadata_all,
            NULL);
   NIR_PASS(progress, s, lower_sampler_state_runtime);

   /* Lower the divisions and interpolations of the software path */
   nir_lower_idiv_options idiv_options = {.allow_fp16 = true};
   NIR_PASS(progress, s, nir_lower_idiv, &idiv_options);
   NIR_PASS(progress, s, nir_lower_flrp, 16 | 32 | 64, false);
   NIR_PASS(progress, s, nir_shader_tex_pass, lower_min_lod,
            nir_metadata_control_flow, NULL);

   nir_lower_tex_options options = {
      .lower_txd_cb = lower_txd_min_lod,
   };
   NIR_PASS(progress, s, nir_lower_tex, &options);

   NIR_PASS(progress, s, nir_opt_constant_folding);
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_regular_texture,
            nir_metadata_none, NULL);
   return progress;
}

bool
agx_nir_lower_texture(nir_shader *s)
{
   bool progress = false;

   if (agx_nir_target(s)->texture_exports)
      return lower_texture_exports(s);

   nir_tex_src_type_constraints tex_constraints = {
      [nir_tex_src_lod] = {true, 16},
      [nir_tex_src_bias] = {true, 16},
      [nir_tex_src_ms_index] = {true, 16},
      [nir_tex_src_min_lod] = {true, 16},
      [nir_tex_src_texture_offset] = {true, 16},
      [nir_tex_src_sampler_offset] = {true, 16},
   };

   NIR_PASS(progress, s, agx_nir_lower_images);
   NIR_PASS(progress, s, nir_legalize_16bit_sampler_srcs, tex_constraints);

   /* Fold constants after nir_legalize_16bit_sampler_srcs so we can detect 0 in
    * lower_regular_texture. This is required for correctness.
    */
   NIR_PASS(progress, s, nir_opt_constant_folding);

   /* Lower texture sources after legalizing types (as the lowering depends on
    * 16-bit multisample indices) but before lowering queries (as the lowering
    * generates txs for array textures).
    */
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_regular_texture,
            nir_metadata_none, NULL);
   NIR_PASS(progress, s, agx_nir_lower_texture_queries);

   return progress;
}

static bool
lower_multisampled_store(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
      return false;

   nir_def *index_px = nir_u2u32(b, image_texel_address(b, intr, true));
   nir_def *coord2d = coords_for_buffer_texture(b, index_px);

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_src_rewrite(&intr->src[2], nir_imm_int(b, 0));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(intr, false);
   return true;
}

bool
agx_nir_lower_multisampled_image_store(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_multisampled_store,
                                     nir_metadata_control_flow, NULL);
}

/*
 * Given a non-bindless instruction, return whether agx_nir_lower_texture will
 * lower it to something involving a descriptor crawl. This requires the driver
 * to lower the instruction to bindless before calling agx_nir_lower_texture.
 * The implementation just enumerates the cases handled in this file.
 */
bool
agx_nir_needs_texture_crawl(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      /* Queries, atomics always become a crawl */
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_deref_atomic_swap:
         return true;

      /* Multisampled stores need a crawl, others do not */
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_deref_store:
         return nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS;

      /* Array loads need a crawl, other load do not */
      case nir_intrinsic_image_load:
         return nir_intrinsic_image_array(intr) ||
                nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE;

      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      /* Array textures get clamped to their size via txs */
      if (tex->is_array && !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP))
         return true;

      switch (tex->op) {
      /* Queries always become a crawl */
      case nir_texop_txs:
      case nir_texop_texture_samples:
      case nir_texop_query_levels:
         return true;

      /* Buffer textures need their format read and txf needs its LOD/layer
       * clamped.  Buffer textures are only read through txf.
       */
      case nir_texop_txf:
      case nir_texop_txf_ms:
         return has_nonzero_lod(tex) || tex->is_array ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;

      default:
         return false;
      }
   }

   return false;
}
