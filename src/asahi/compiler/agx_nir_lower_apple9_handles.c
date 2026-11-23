/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_apple9_layout.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_compile_apple9.h"
#include "agx_nir.h"

/* Lower handles to USC aperture offsets, texture handles to table indices. Run
 * before the argument lowering, so set uniforms are numbered as on Apple8.
 */

static nir_def *
descriptor_offset(nir_builder *b, nir_intrinsic_instr *handle)
{
   nir_def *set =
      nir_load_preamble(b, 1, 64, .base = nir_intrinsic_desc_set(handle));
   return nir_iadd(b, nir_unpack_64_2x32_split_x(b, set), handle->src[0].ssa);
}

static nir_intrinsic_instr *
handle_intrinsic(nir_src src)
{
   nir_intrinsic_instr *intr = nir_src_as_intrinsic(src);

   if (intr && (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                intr->intrinsic == nir_intrinsic_bindless_sampler_agx))
      return intr;
   else
      return NULL;
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex)
{
   int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (texture < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   for (unsigned i = 0; i < 2; ++i) {
      int s = i ? sampler : texture;
      nir_intrinsic_instr *handle =
         s >= 0 ? handle_intrinsic(tex->src[s].src) : NULL;
      if (!handle)
         continue;

      unsigned entry = i ? AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY
                         : AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY;
      nir_def *index = nir_iadd_imm(
         b, nir_ushr_imm(b, descriptor_offset(b, handle), 5), entry);

      static_assert(AGX_APPLE9_TEXTURE_TABLE_STRIDE == 32, "shift");
      nir_src_rewrite(&tex->src[s].src, index);
   }

   if (sampler < 0 &&
       (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms)) {
      nir_def *fetch = nir_load_ubo(
         b, 1, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET), .align_mul = 4,
         .range_base = AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET, .range = 4);

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, fetch);
   }

   return true;
}

static bool
lower(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type == nir_instr_type_tex)
      return lower_tex(b, nir_instr_as_tex(instr));

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);

   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      nir_intrinsic_instr *handle = handle_intrinsic(intr->src[0]);
      assert(handle && "handles come from descriptors");

      nir_def *set =
         nir_load_preamble(b, 1, 64, .base = nir_intrinsic_desc_set(handle));
      nir_def_replace(&intr->def,
                      nir_iadd(b, set, nir_u2u64(b, handle->src[0].ssa)));
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_bindless_image_agx &&
       intr->intrinsic != nir_intrinsic_bindless_sampler_agx)
      return false;

   /* Other users take the offset; the handle is removed once unused */
   nir_def *offset = descriptor_offset(b, intr);
   nir_foreach_use_safe(use, &intr->def) {
      nir_instr *parent = nir_src_use_instr(use);

      if (parent->type == nir_instr_type_tex)
         continue;

      if (parent->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(parent)->intrinsic ==
             nir_intrinsic_load_from_texture_handle_agx)
         continue;

      nir_src_rewrite(use, nir_u2uN(b, offset, intr->def.bit_size));
   }

   return true;
}

bool
agx_nir_lower_apple9_handles(nir_shader *nir)
{
   /* Texture and address loads first, while their handles are intrinsics */
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_tex ||
                (instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(instr)->intrinsic ==
                    nir_intrinsic_load_from_texture_handle_agx))
               impl_progress |= lower(&b, instr, NULL);
         }
      }

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic)
               impl_progress |= lower(&b, instr, NULL);
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   if (progress)
      NIR_PASS(_, nir, nir_opt_dce);

   return progress;
}
