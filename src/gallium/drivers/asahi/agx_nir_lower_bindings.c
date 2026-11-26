/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "agx_state.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Lower binding table textures and images to texture state registers and (if
 * necessary) bindless access into an internal table mapped like additional
 * texture state registers. The following layout is used:
 *
 *    1. Textures
 *    2. Images (read/write interleaved)
 */

static bool
lower_sampler(nir_builder *b, nir_tex_instr *tex)
{
   if (!nir_tex_instr_need_sampler(tex))
      return false;

   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!index)
      index = nir_imm_int(b, tex->sampler_index);

   nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                         nir_load_sampler_handle_agx(b, index));
   return true;
}

struct lower_state {
   const struct agx_isa_target *target;
   bool *uses_bindless_samplers;
};

static bool
lower(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_state *state = data;
   bool *uses_bindless_samplers = state->uses_bindless_samplers;
   bool progress = false;
   bool exports = state->target->texture_exports;

   /* Texture exports read dimensions from memory, not the descriptor */
   bool force_bindless = !exports && agx_nir_needs_texture_crawl(instr);
   b->cursor = nir_before_instr(instr);

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
         break;
      default:
         return false;
      }

      nir_def *index = intr->src[0].ssa;
      nir_scalar index_scalar = nir_scalar_resolved(index, 0);

      /* Images are then storage descriptors, writes and atomics use PBEs */
      if (exports) {
         index = nir_umin(b, nir_u2u32(b, index),
                          nir_imm_int(b, b->shader->info.num_images - 1));

         bool pbe = intr->intrinsic != nir_intrinsic_image_load &&
                    intr->intrinsic != nir_intrinsic_image_size &&
                    intr->intrinsic != nir_intrinsic_image_samples;
         nir_def *offset = nir_iadd_imm(
            b, nir_imul_imm(b, index, sizeof(struct agx_storage_descriptor)),
            pbe ? sizeof(struct agx_texture_packed) : 0);

         nir_rewrite_image_intrinsic(
            intr,
            nir_bindless_image_agx(b, offset,
                                   .desc_set = AGX_IMAGE_TABLE_UNIFORM),
            nir_image_intrinsic_type_bindless);
         return true;
      }

      /* Remap according to the driver layout */
      unsigned offset = BITSET_LAST_BIT(b->shader->info.textures_used);

      /* For reads and queries, we use the texture descriptor which is first.
       * Writes and atomics use the PBE descriptor.
       */
      if (intr->intrinsic != nir_intrinsic_image_load &&
          intr->intrinsic != nir_intrinsic_image_size &&
          intr->intrinsic != nir_intrinsic_image_samples)
         offset++;

      /* If we can determine statically that the image fits in texture state
       * registers, avoid lowering to bindless access.
       */
      if (nir_scalar_is_const(index_scalar) && !force_bindless) {
         unsigned idx = (nir_scalar_as_uint(index_scalar) * 2) + offset;

         if (idx < AGX_NUM_TEXTURE_STATE_REGS) {
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(b, idx, 16));
            return true;
         }
      }

      /* Otherwise, lower to bindless...
       *
       * The driver uploads enough null texture/PBE descriptors for robustness
       * given the shader limit, but we still need to clamp since we're lowering
       * to bindless so the hardware doesn't know the limit.
       *
       * The GL spec says out-of-bounds image indexing is undefined, but
       * faulting is not acceptable for robustness.
       */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_images - 1, index->bit_size));

      index = nir_iadd_imm(b, nir_imul_imm(b, index, 2), offset);

      nir_rewrite_image_intrinsic(intr, nir_load_texture_handle_agx(b, index),
                                  nir_image_intrinsic_type_bindless);
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (!exports &&
          ((BITSET_COUNT(b->shader->info.samplers_used) > 16) &&
           (nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0 ||
            tex->sampler_index >= 16)) &&
          lower_sampler(b, tex)) {
         progress = true;
         *uses_bindless_samplers = true;
      }

      /* Nothing to do for "real" bindless */
      if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
         return progress;

      /* Texture exports index the whole texture table */
      if (exports &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1)
         return progress;

      /* Textures are mapped 1:1, so if we can prove it fits in a texture state
       * register, use the texture state register.
       */
      if (tex->texture_index < AGX_NUM_TEXTURE_STATE_REGS &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1 &&
          !force_bindless)
         return progress;

      /* Otherwise, lower to bindless. Could be optimized. */
      nir_def *index = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
      if (!index)
         index = nir_imm_int(b, tex->texture_index);

      /* As above */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_textures - 1, index->bit_size));

      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle,
                            nir_load_texture_handle_agx(b, index));
   }

   return true;
}

/* Apple9 texture indices are immediates, so select among the textures */
static nir_def *
select_texture(nir_builder *b, nir_tex_instr *tex, nir_def *index, bool sampler,
               unsigned i, unsigned count)
{
   nir_tex_instr *copy =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   copy->texture_index += i;
   if (sampler)
      copy->sampler_index += i;

   /* Out-of-bounds indices are undefined, take the last texture */
   if (i + 1 == count) {
      nir_builder_instr_insert(b, &copy->instr);
      return &copy->def;
   }

   nir_push_if(b, nir_ieq_imm(b, index, i));
   nir_builder_instr_insert(b, &copy->instr);
   nir_push_else(b, NULL);
   nir_def *other = select_texture(b, tex, index, sampler, i + 1, count);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, &copy->def, other);
}

static bool
lower_dynamic_texture_index(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      struct util_dynarray texs;
      util_dynarray_init(&texs, NULL);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex &&
                nir_tex_instr_src_index(nir_instr_as_tex(instr),
                                        nir_tex_src_texture_offset) >= 0)
               util_dynarray_append(&texs, nir_instr_as_tex(instr));
         }
      }

      util_dynarray_foreach(&texs, nir_tex_instr *, it) {
         nir_tex_instr *tex = *it;
         nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));

         nir_def *index = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
         nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
         progress = true;

         if (nir_scalar_is_const(nir_scalar_resolved(index, 0))) {
            unsigned offset = nir_scalar_as_uint(nir_scalar_resolved(index, 0));
            tex->texture_index += offset;
            if (sampler)
               tex->sampler_index += offset;
            continue;
         }

         unsigned count =
            MAX2(shader->info.num_textures, tex->texture_index + 1) -
            tex->texture_index;

         nir_def *res = select_texture(&b, tex, nir_u2u32(&b, index),
                                       sampler != NULL, 0, count);
         nir_def_replace(&tex->def, res);
      }

      nir_progress(progress, impl, nir_metadata_none);
      util_dynarray_fini(&texs);
   }

   return progress;
}

bool
agx_nir_lower_bindings(nir_shader *shader, const struct agx_isa_target *target,
                       bool *uses_bindless_samplers)
{
   /* First lower index to offset so we can lower more naturally */
   bool progress = nir_lower_tex(
      shader, &(nir_lower_tex_options){.lower_index_to_offset = true});

   /* Next run constant folding so the constant optimizations above have a
    * chance.
    */
   progress |= nir_opt_constant_folding(shader);

   if (target->texture_exports)
      progress |= lower_dynamic_texture_index(shader);

   struct lower_state state = {target, uses_bindless_samplers};
   progress |= nir_shader_instructions_pass(shader, lower,
                                            nir_metadata_control_flow, &state);
   return progress;
}
