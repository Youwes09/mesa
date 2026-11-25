/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include <unistd.h>

#include "util/u_queue.h"
#include "hk_shader.h"

#include "asahi/lib/agx_apple9_layout.h"
#include "pipe/p_defines.h"
#include "poly/geometry.h"
#include "poly/nir/poly_nir.h"
#include "agx_debug.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_nir_lower_vbo.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_format_convert.h"

#include "asahi/libagx/mesh.h"
#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_serialize.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_util.h"
#include "vk_ycbcr_conversion.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_uvs.h"
#include "compiler/spirv/nir_spirv.h"

#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

struct hk_fs_key {
   bool zs_self_dep;

   /** True if sample shading is forced on via an API knob such as
    * VkPipelineMultisampleStateCreateInfo::minSampleShading
    */
   bool force_sample_shading;

   /* Whether a depth/stencil feedback loop can be enabled at draw time */
   bool dynamic_zs_self_dep;

   uint8_t pad[5];
};
static_assert(sizeof(struct hk_fs_key) == 8, "packed");

struct hk_vs_key {
   struct agx_velem_key attribs[32];
   bool skip_prolog;
   bool static_strides;
   bool kill_psiz;
   bool pad[1];
};
static_assert(sizeof(struct hk_vs_key) == 260, "packed");

union hk_key {
   struct hk_vs_key vs;
   struct hk_fs_key fs;
};

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev)
{
   /* Soft faults follow a kernel parameter, so they can change between
    * boots with the same cache. */
   return pdev->dev.debug | ((uint64_t)pdev->apple9_layered_2d << 63) |
          ((uint64_t)agx_has_soft_fault((struct agx_device *)&pdev->dev)
           << 62) |
          ((uint64_t)pdev->apple9_d3d_mip_rounding << 61);
}

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   return agx_nir_options_for_isa(agx_gather_device_key(&pdev->dev).shader_isa);
}

static struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   const struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   return (struct spirv_to_nir_options){
      .ssbo_addr_format = hk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = hk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = hk_min_ssbo_alignment(&pdev->dev),
      .min_ubo_alignment = HK_MIN_UBO_ALIGNMENT,
   };
}

void
hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev, nir_shader *nir)
{
   /* Must lower before io to temps */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   } else if (nir->info.stage == MESA_SHADER_TASK) {
      /* Launching mesh workgroups ends the task workgroup. */
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }

   /* Unroll loops before lowering indirects via
    * nir_lower_io_vars_to_temporaries */
   UNUSED bool progress = false;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);

   /* Treat position, point size, clip/cull distances and tessellation levels
    * as invariant, as RADV does. Whether NIR fuses a multiply into an add
    * depends on the multiply's other users, so the same position math rounds
    * differently in shaders that also write other outputs from it. Blender
    * EEVEE's motion-vector prepass and its material pass then disagree on
    * depth by a few ULPs, and the material pass's EQUAL test drops most of
    * its pixels.
    */
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_propagate_invariant, true);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };

      nir_lower_sysvals_to_varyings(nir, &sysvals_opts);
   }

   NIR_PASS(_, nir, nir_lower_system_values);

   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Mesh and tessellation control outputs are shared by the workgroup or
    * patch: each invocation may write a different element, so copying whole
    * temporaries back at the end would overwrite the other invocations'
    * writes. Without temporaries, lower component accesses before I/O
    * lowering.
    */
   if (nir->info.stage == MESA_SHADER_MESH ||
       nir->info.stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   } else {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_shader_out);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);

   /* Extract immutable tables before agx_preprocess_nir lowers large local
    * arrays to scratch. The analysis requires explicit loads/stores rather
    * than copies. Keep the data attached to NIR through variant cloning.
    */
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);
   if (agx_is_apple9(&pdev->dev))
      NIR_PASS(_, nir, agx_nir_lower_word_math);
   agx_preprocess_nir(nir);

   /* Constant-offset reads can fold away after extraction. Do not upload
    * or bind an unused blob: besides wasting storage, that would extend
    * every draw's root for a table the program never reads.
    */
   if (nir->constant_data_size) {
      bool used = false;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               used |= instr->type == nir_instr_type_intrinsic &&
                       nir_instr_as_intrinsic(instr)->intrinsic ==
                          nir_intrinsic_load_constant;
            }
         }
      }
      if (!used) {
         ralloc_free(nir->constant_data);
         nir->constant_data = NULL;
         nir->constant_data_size = 0;
      }
   }

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
}

static void
hk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   hk_preprocess_nir_internal(vk_pdev, nir);
   /* Only compute dispatches have a base workgroup. */
   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = nir->info.stage == MESA_SHADER_COMPUTE,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}

static void
hk_populate_vs_key(struct hk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state && state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {

      key->kill_psiz = state->ia->primitive_topology != MESA_PRIM_POINTS;
   }

   if (state && state->vi && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDINGS_VALID)) {

      agx_fill_velem_keys(state->vi, ~0 /* compacted on use */, key->attribs);
      key->skip_prolog = true;
      key->static_strides =
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);

      if (!key->static_strides) {
         for (unsigned i = 0; i < ARRAY_SIZE(key->attribs); ++i) {
            key->attribs[i].stride = 0;
         }
      }
   }
}

static void
hk_populate_fs_key(struct hk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features)
{
   memset(key, 0, sizeof(*key));

   /* Pipelines allow draw-time feedback loops only with the dynamic state */
   if (state == NULL) {
      key->dynamic_zs_self_dep =
         features && features->attachmentFeedbackLoopDynamicState;
      return;
   }

   key->dynamic_zs_self_dep = BITSET_TEST(
      state->dynamic, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);

   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
      key->zs_self_dep = true;

   /* We force per-sample interpolation whenever sampleShadingEnable is set
    * regardless of minSampleShading or rasterizationSamples.
    *
    * When sampleShadingEnable is set, few guarantees are made about the
    * location of interpolation of the inputs.  The only real guarantees are
    * that the inputs are interpolated within the pixel and that you get at
    * least `rasterizationSamples * minSampleShading` unique positions.
    * Importantly, it does not require that when `rasterizationSamples *
    * minSampleShading <= 1.0` that those positions are at the fragment
    * center.  Therefore, it's valid to just always do per-sample all the time.
    *
    * The one caveat here is that we have to be careful about gl_SampleMaskIn.
    * When `hk_fs_key::force_sample_shading = true` we also turn any reads of
    * gl_SampleMaskIn into `1 << gl_SampleID` because the hardware sample mask
    * is actually per-fragment, not per-pass.  We handle this by smashing
    * minSampleShading to 1.0 whenever gl_SampleMaskIn is read.
    */
   const struct vk_multisample_state *ms = state->ms;
   if (ms != NULL && ms->sample_shading_enable)
      key->force_sample_shading = true;
}

enum hk_feature_key {
   HK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   /* Apple9 lowers the image view min LOD clamp only when it is enabled. */
   HK_FEAT_MIN_LOD = BITFIELD_BIT(1),
};

static enum hk_feature_key
hk_make_feature_key(const struct vk_features *feats)
{
   return ((!feats || feats->customBorderColors) ? HK_FEAT_CUSTOM_BORDER : 0) |
          ((!feats || feats->minLod) ? HK_FEAT_MIN_LOD : 0);
}

/* Pipeline sample shading shades each sample, so interpolate every non-flat
 * input at its sample, as FragCoord already is through its sample position.
 * Apple9 interpolates explicitly and does not move pixel or centroid
 * barycentrics when shading per sample.
 */
static bool
hk_apple9_lower_bary_to_sample(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   nir_def *sample;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
      sample = nir_load_barycentric_sample(
         b, intr->def.bit_size, .interp_mode = nir_intrinsic_interp_mode(intr));
      break;
   default:
      return false;
   }

   nir_def_replace(&intr->def, sample);
   return true;
}

/* Triangle merging shades a pixel's samples from adjacent strip or fan
 * triangles together, interpolated on one triangle's plane. That is exact at
 * the pixel centre but not at the other triangle's samples, so per-sample
 * shading and centroid, sample or offset interpolation keep the triangles
 * apart.
 */
static bool
hk_apple9_interpolates_off_center(nir_shader *nir)
{
   if (nir->info.fs.uses_sample_shading)
      return true;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_barycentric_centroid:
            case nir_intrinsic_load_barycentric_sample:
            case nir_intrinsic_load_barycentric_at_sample:
            case nir_intrinsic_load_barycentric_at_offset:
               return true;
            default:
               break;
            }
         }
      }
   }
   return false;
}

static void
hk_apple9_write_primitive_id(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(
      &b, nir_load_primitive_id(&b), nir_imm_int(&b, 0), .write_mask = 0x1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
   s->info.outputs_written |= VARYING_BIT_PRIMITIVE_ID;
   nir_progress(true, impl, nir_metadata_control_flow);
}

static void
hk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT)) {
      struct hk_vs_key key;
      hk_populate_vs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   /* A partition of a linked pipeline holds both stages */
   if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      struct hk_fs_key key;
      hk_populate_fs_key(&key, state, features);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state &&
       (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT))) {
      const bool is_multiview = state->mv->view_mask != 0;
      _mesa_blake3_update(&blake3_ctx, &is_multiview, sizeof(is_multiview));
   }

   enum hk_feature_key feature_key = hk_make_feature_key(features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

static nir_def *
bounds_check(nir_builder *b, nir_def *data, nir_def *offs, nir_def *bound)
{
   if (data->bit_size == 32 && data->num_components == 1) {
      return nir_bounds_agx(b, data, offs, bound);
   } else {
      /* TODO: Optimize */
      return nir_bcsel(b, nir_uge(b, bound, offs), data,
                       nir_imm_zero(b, data->num_components, data->bit_size));
   }
}

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

/* Whether the first load_size bytes of a bounded access are in bounds. */
static nir_def *
check_in_bounds_size(nir_builder *b, nir_intrinsic_instr *intr,
                     unsigned load_size)
{
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;

   /* Try to bounds check in terms of elements */
   nir_scalar offset_s = nir_scalar_resolved(offset, 0);
   if (is_op(offset_s, nir_op_amul)) {
      nir_scalar srcs[] = {nir_scalar_chase_alu_src(offset_s, 0),
                           nir_scalar_chase_alu_src(offset_s, 1)};
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      if (nir_scalar_is_const(srcs[1 - i]) &&
          nir_scalar_as_uint(srcs[1 - i]) == load_size) {

         nir_def *index = nir_mov_scalar(b, srcs[i]);
         return nir_ult(b, index, nir_udiv_imm(b, bound, load_size));
      }
   }

   /* TODO: handle also the iadd(amul) pattern, this is important */

   /* Otherwise bounds check in bytes */
   nir_def *sat_offset = nir_umin_imm(b, offset, UINT32_MAX - (load_size - 1));
   return nir_ult(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);
}

static nir_def *
check_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   unsigned bit_size = intr->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   return check_in_bounds_size(b, intr, bit_size / 8 * intr->num_components);
}

/* An amul whose scale is not a shift of 0..4. */
static bool
hk_unencodable_scale(nir_scalar s)
{
   if (!is_op(s, nir_op_amul))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar k = nir_scalar_chase_alu_src(s, i);
      if (nir_scalar_is_const(k))
         return !util_is_power_of_two_nonzero(nir_scalar_as_uint(k)) ||
                nir_scalar_as_uint(k) > 16;
   }
   return true;
}

static nir_def *
bound_offset(nir_builder *b, nir_def *valid, nir_scalar offset,
             bool shift_only_scale)
{
   /* If we can, clamp the source of an amul instead of the result, to remain
    * compatible with the hardware address mode.
    */
   if (is_op(offset, nir_op_amul)) {
      nir_scalar srcs[] = {
         nir_scalar_chase_alu_src(offset, 0),
         nir_scalar_chase_alu_src(offset, 1),
      };
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      /* Apple9 scales an index only by a shift of 0..4. Any other scale
       * would multiply again after the select; selecting the product, which
       * the bounds check already computed, saves that multiply. */
      if (shift_only_scale &&
          !(nir_scalar_is_const(srcs[1 - i]) &&
            util_is_power_of_two_nonzero(nir_scalar_as_uint(srcs[1 - i])) &&
            nir_scalar_as_uint(srcs[1 - i]) <= 16))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset),
                          nir_imm_int(b, 0));
      nir_def *x = nir_mov_scalar(b, srcs[i]);
      nir_def *y = nir_mov_scalar(b, srcs[1 - i]);

      return nir_amul(b, nir_bcsel(b, valid, x, nir_imm_int(b, 0)), y);
   }

   /* Similar case for when there's an addition (chain) in the way */
   if (is_op(offset, nir_op_iadd)) {
      nir_scalar x = nir_scalar_chase_alu_src(offset, 0);
      nir_scalar y = nir_scalar_chase_alu_src(offset, 1);

      /* With an Apple9 scale the index cannot encode, select the complete
       * offset the bounds check computed rather than each term. */
      if (shift_only_scale &&
          (hk_unencodable_scale(x) || hk_unencodable_scale(y)))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset),
                          nir_imm_int(b, 0));

      if (is_op(x, nir_op_amul) || is_op(y, nir_op_amul)) {
         return nir_iadd(b, bound_offset(b, valid, x, shift_only_scale),
                         bound_offset(b, valid, y, shift_only_scale));
      }
   }

   nir_def *def = nir_mov_scalar(b, offset);

   /* If the offset fits within the zero page, clamping is pointless */
   if (nir_scalar_is_const(offset) &&
       (nir_scalar_as_uint(offset) + 16) < AGX_ZERO_PAGE_SIZE)
      return def;

   /* Otherwise, fallback on clamping the offset */
   return nir_bcsel(b, valid, def, nir_imm_int(b, 0));
}

static void
lower_load_global_bounded(nir_builder *b, nir_intrinsic_instr *intr,
                          bool per_component_bounds)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   /* A vector merged from separate accesses (bounds_checked_modes) keeps
    * each component's own check: robustness2 returns in-bounds components
    * even when a later one is out of bounds. Address the load by its first
    * component; the vectorizer only merges naturally aligned accesses, so
    * the read stays within one aligned block of the same page. */
   unsigned byte_size = intr->def.bit_size / 8;
   bool per_component = per_component_bounds && intr->def.num_components > 1;

   /* Bounds check compatibly with the hardware address mode */
   nir_def *valid = per_component ? check_in_bounds_size(b, intr, byte_size)
                                  : check_in_bounds(b, intr);
   base = nir_bcsel(b, valid, base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   offset = bound_offset(b, valid, nir_scalar_resolved(offset, 0),
                         per_component_bounds);

   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   }

   nir_def *val = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, base, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr), .access = access);

   if (per_component) {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      comps[0] = nir_channel(b, val, 0);
      for (unsigned c = 1; c < intr->def.num_components; ++c) {
         comps[c] = nir_bcsel(
            b, check_in_bounds_size(b, intr, (c + 1) * byte_size),
            nir_channel(b, val, c), nir_imm_zero(b, 1, intr->def.bit_size));
      }
      val = nir_vec(b, comps, intr->def.num_components);
   }

   nir_def_replace(&intr->def, val);
}

/* Natural alignment keeps a partly in-bounds merged load within one page */
static bool
hk_apple9_vectorize_cb(unsigned align_mul, unsigned align_offset,
                       unsigned bit_size, unsigned num_components,
                       int64_t hole_size, nir_intrinsic_instr *low,
                       nir_intrinsic_instr *high, void *data)
{
   if (!agx_mem_vectorize_cb(align_mul, align_offset, bit_size, num_components,
                             hole_size, low, high, NULL))
      return false;

   if (low->intrinsic == nir_intrinsic_load_global_bounded ||
       low->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      unsigned bytes = util_next_power_of_two(num_components * bit_size / 8);
      if (hole_size > 0 || align_mul < bytes || align_offset % bytes)
         return false;
   }
   return true;
}

struct hk_bounded_lowering {
   bool soft_fault;
   /* Merged vector loads are checked per component (bounds_checked_modes). */
   bool per_component;
};

static bool
lower_load_global_constant_offset_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin, void *data)
{
   const struct hk_bounded_lowering *opts = data;
   const bool *has_soft_fault = &opts->soft_fault;

   if (intrin->intrinsic == nir_intrinsic_load_global_bounded ||
       (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
        !(*has_soft_fault))) {

      lower_load_global_bounded(b, intrin, opts->per_component);
      return true;
   }

   if (intrin->intrinsic != nir_intrinsic_load_global_constant_offset &&
       intrin->intrinsic != nir_intrinsic_load_global_constant_bounded)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *base_addr = intrin->src[0].ssa;
   nir_def *offset = intrin->src[1].ssa;
   nir_def *bound = NULL;

   unsigned bit_size = intrin->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intrin->num_components;

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      bound = intrin->src[2].ssa;
   }

   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);

   /* Without robustness an access outside the buffer is undefined, so the
    * load itself is in bounds. The Apple9 backend relies on this to fold
    * constant offsets past an amul: the hardware adds the displacement in
    * 64 bits, which differs from the 32-bit offset only outside the buffer.
    */
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);
   if (intrin->intrinsic == nir_intrinsic_load_global_constant_offset)
      access |= ACCESS_IN_BOUNDS;

   nir_def *val = nir_load_global_constant(
      b, intrin->def.num_components, intrin->def.bit_size,
      nir_iadd(b, base_addr, nir_u2u64(b, offset)), .align_mul = align_mul,
      .align_offset = align_offset, .access = access);

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
       opts->per_component && intrin->def.num_components > 1) {
      /* A vector merged from separate robust fields (bounds_checked_modes)
       * returns each in-bounds component, as in lower_load_global_bounded:
       * checking only its last byte would zero the in-bounds fields too. A
       * field split over two merged loads then reads (x, y, 0, 0) where
       * robustness2 allows only (x, y, z, 0) or zero (uniform_buffer
       * len_12/31/252).
       */
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < intrin->def.num_components; ++c) {
         nir_def *end =
            nir_scalar_is_const(offs)
               ? nir_imm_int(b, nir_scalar_as_uint(offs) + (c + 1) * byte_size)
               : nir_iadd_imm(b, offset, (c + 1) * byte_size);
         comps[c] = bounds_check(b, nir_channel(b, val, c), end, bound);
      }
      val = nir_vec(b, comps, intrin->def.num_components);
   } else if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      if (nir_scalar_is_const(offs)) {
         /* Calculate last byte loaded */
         unsigned offs_imm = nir_scalar_as_uint(offs) + load_size;

         /* Compare the last byte loaded against the bound exactly. Bounds
          * are the descriptor's exact range, not rounded to 64 B, so rounding
          * the offset down would let loads past a short range through
          * (robustness2 uniform_buffer len_4..len_60).
          */
         val = bounds_check(b, val, nir_imm_int(b, offs_imm), bound);
      } else {
         offset = nir_iadd_imm(b, offset, load_size);
         val = bounds_check(b, val, offset, bound);
      }
   }

   nir_def_replace(&intrin->def, val);
   return true;
}

/* Lower bounded and offset global loads. Apple9 checks each component of a
 * merged vector separately. */
bool
hk_lower_bounded_global_loads(nir_shader *nir, bool soft_fault, bool apple9)
{
   struct hk_bounded_lowering bounded = {
      .soft_fault = soft_fault,
      .per_component = apple9,
   };
   return nir_shader_intrinsics_pass(nir,
                                     lower_load_global_constant_offset_instr,
                                     nir_metadata_none, &bounded);
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct hk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct hk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion
             ? &sampler->vk.ycbcr_conversion->state
             : NULL;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * This is the world's worst multiview implementation. We simply duplicate each
 * draw on the CPU side, changing a uniform in between, and then plumb the view
 * index into the layer ID here. Whatever, it works.
 *
 * The "proper" implementation on AGX would use vertex amplification, but a
 * MacBook is not a VR headset.
 */
static void
hk_lower_multiview(nir_shader *nir)
{
   /* If there's an existing layer ID write, ignore it. This avoids validation
    * splat with vk_meta.
    */
   nir_variable *existing = nir_find_variable_with_location(
      nir, nir_var_shader_out, VARYING_SLOT_LAYER);

   if (existing) {
      existing->data.mode = nir_var_shader_temp;
      existing->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   /* Now write the view index as the layer */
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_variable *layer =
      nir_variable_create(nir, nir_var_shader_out, glsl_uint_type(), NULL);

   layer->data.location = VARYING_SLOT_LAYER;

   nir_store_var(&b, layer, nir_load_view_index(&b), nir_component_mask(1));
   b.shader->info.outputs_written |= VARYING_BIT_LAYER;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   /* XXX: this is a really weird edge case, is this even well-defined? */
   if (tex->is_shadow)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);

      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      nir_def *clamp_to_0 = &tex_0->def;

      tex_0->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex_0);

      if (tex->op == nir_texop_tg4) {
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);
      }

      /* Combine together with the border */
      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {

         /* For floats, lerp together:
          *
          * For border texels:  (1 * border) + (0 * border      ) = border
          * For regular texels: (x * border) + (x * (1 - border)) = x.
          *
          * Linear filtering is linear (duh), so lerping is compatible.
          */
         replaced = nir_flrp(b, clamp_to_0, clamp_to_1, border);
         b->shader->info.flrp_lowered = false;
      } else {
         /* For integers, just select componentwise since there is no linear
          * filtering. Gathers also use this path since they are unfiltered in
          * each component.
          */
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1), clamp_to_0,
                              border);
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   nir_def_replace(&tex->def, phi);
   return true;
}

static bool
agx_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex, bool int_coords)
{
   nir_alu_type T = int_coords ? nir_type_uint16 : nir_type_float16;
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1, T,
                                  false, false);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const bool *float_fetch = data;
   if (*float_fetch && tex->op == nir_texop_lod &&
       b->shader->info.stage == MESA_SHADER_COMPUTE &&
       b->shader->info.derivative_group == DERIVATIVE_GROUP_LINEAR) {
      /* The compute query consumes the view clamp after sampler/view level
       * clamping, before mip rounding. It is not a sampler minimum LOD. */
      b->cursor = nir_before_instr(&tex->instr);
      nir_tex_instr_add_src(tex, nir_tex_src_min_lod,
                            query_min_lod(b, tex, false));
      return true;
   }
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   /* Apple9 legalizes the minimum-LOD source as a float, so fetches take
    * the float minimum and compare against its floor. */
   nir_def *min_lod = query_min_lod(b, tex, int_coords && !*float_fetch);
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (tex->op == nir_texop_tg4) {
      b->cursor = nir_after_instr(&tex->instr);

      /* The Vulkan spec section "Texel Gathering" says:
       *
       *    If levelbase < minLodIntegerimageView, then any values fetched are
       *    zero if the robustImageAccess2 feature is enabled.
       *
       * We currently always enable robustImageAccess2, so implement that
       * semantic here.
       *
       * We could probably optimize this with a special descriptor for this case
       * but tg4 is rare enough I'm not bothered.
       */
      nir_def *old = &tex->def;
      nir_def *oob = *float_fetch ? nir_fge_imm(b, min_lod, 1.0)
                                  : nir_ine_imm(b, min_lod, 0);
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

static bool
agx_nir_lower_image_view_min_lod(nir_shader *nir, bool float_fetch)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none,
                              &float_fetch);
}

/*
 * In Vulkan, the VIEWPORT should read 0 in the fragment shader if it is not
 * written by the vertex shader, but in our implementation, the varying would
 * otherwise be undefined. This small pass predicates VIEWPORT reads based on
 * whether the hardware vertex shader writes the VIEWPORT (nonzero UVS index).
 */
static bool
lower_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_VIEWPORT)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *orig = &intr->def;

   nir_def *uvs = nir_load_uvs_index_agx(b, .io_semantics = sem);
   nir_def *def = nir_bcsel(b, nir_ine_imm(b, uvs, 0), orig, nir_imm_int(b, 0));

   nir_def_rewrite_uses_after(orig, def);
   return true;
}

/* The Apple9 rasterizer delivers the layer to the fragment layer system
 * value. The viewport index is an ordinary flat varying.
 */
static bool
hk_apple9_lower_layer_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr,
                                  UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *layer = nir_iand_imm(b, nir_load_layer_id(b), 0xffff);
   nir_def_replace(&intr->def, nir_u2uN(b, layer, intr->def.bit_size));
   return true;
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static bool
should_lower_robust(const nir_intrinsic_instr *intr, const void *_)
{
   /* The hardware is robust, but our software image atomics are not. Unlike the
    * GL driver, we don't use the common buffer image lowering, using the
    * agx_nir_lower_texture lowering for robustImageAccess2 semantics.
    */
   return intr->intrinsic == nir_intrinsic_image_deref_atomic ||
          intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
}

/* storageInputOutput16: 16-bit stage inputs and outputs are carried as 32-bit
 * values at every interface, converting exactly at the access (f16 <-> f32,
 * 16-bit integers truncated on load and extended on store), as both sides of an
 * interface are widened alike. Slots and components are unchanged.
 */
static bool
hk_lower_io_16bit_to_32bit(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_primitive_input:
   case nir_intrinsic_load_input_vertex:
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output: {
      if (intr->def.bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_dest_type(intr);
      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, nir_alu_type_get_base_type(type) | 32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *narrow = nir_alu_type_get_base_type(type) == nir_type_float
                           ? nir_f2f16(b, &intr->def)
                           : nir_u2u16(b, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, narrow);
      return true;
   }
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output: {
      nir_def *value = intr->src[0].ssa;
      if (value->bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_src_type(intr);
      nir_alu_type base = nir_alu_type_get_base_type(type);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *wide = base == nir_type_float ? nir_f2f32(b, value)
                      : base == nir_type_int ? nir_i2i32(b, value)
                                             : nir_u2u32(b, value);
      nir_src_rewrite(&intr->src[0], wide);
      nir_intrinsic_set_src_type(intr, base | 32);
      return true;
   }
   default:
      return false;
   }
}

/* Mesh primitives rasterize with their per-primitive builtins as flat varyings
 * of each vertex. User per-primitive varyings are read from the records.
 */
static bool
hk_lower_per_primitive_input(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (hk_mesh_varying_in_memory(sem.location)) {
      nir_def *record = nir_load_input(
         b, 1, 32, nir_imm_int(b, 0), .range = 1, .dest_type = nir_type_uint32,
         .io_semantics.location = HK_MESH_PRIMITIVE_SLOT,
         .io_semantics.num_slots = 1);
      b->shader->info.inputs_read |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);

      nir_def *params = hk_nir_load_root(
         b, 1, 64, hk_root_descriptor_offset(draw.mesh_params));
      nir_def *outputs = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, params, offsetof(struct libagx_mesh_params, outputs)),
         .align_mul = 8);
      nir_def *varying = nir_load_global(
         b, 1, 16,
         nir_iadd_imm(b, params,
                      offsetof(struct libagx_mesh_params, primitive_varying_B) +
                         (sem.location - VARYING_SLOT_VAR0) * 2),
         .align_mul = 2);
      nir_def *offset =
         nir_iadd(b, nir_iadd(b, record, nir_u2u32(b, varying)),
                  nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, 16),
                               nir_intrinsic_component(intr) * 4));
      nir_def *value = nir_load_global(
         b, intr->def.num_components, 32,
         nir_iadd(b, outputs, nir_u2u64(b, offset)), .align_mul = 4);
      if (intr->def.bit_size != 32) {
         value = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
                       nir_type_float
                    ? nir_f2fN(b, value, intr->def.bit_size)
                    : nir_u2uN(b, value, intr->def.bit_size);
      }
      nir_def_replace(&intr->def, value);
      return true;
   }

   nir_def *value = nir_load_input(
      b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr), .range = sem.num_slots,
      .component = nir_intrinsic_component(intr),
      .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
   nir_def_replace(&intr->def, value);
   return true;
}

/* With hk_apple9_d3d_mip_rounding (driconf, set for vkd3d-proton), add the
 * sampler's nudge to explicit LODs, so a NEAREST mip selection exactly
 * halfway between levels rounds up as in D3D. The nudge is zero for other
 * mipmap modes, see hk_CreateSampler.
 */
static bool
hk_lower_d3d_txl_rounding(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_txl || !nir_tex_instr_need_sampler(tex))
      return false;

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (lod_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *nudge = nir_build_texture_query(b, tex, nir_texop_lod_bias, 1,
                                            nir_type_float16, false, false);
   nir_def_as_tex(nudge)->backend_flags |= HK_TEX_TXL_NUDGE;

   nir_def *lod = tex->src[lod_idx].src.ssa;
   nir_src_rewrite(&tex->src[lod_idx].src,
                   nir_fadd(b, lod, nir_f2fN(b, nudge, lod->bit_size)));
   return true;
}

/* With hk_apple9_arrayed_2d_views (driconf, set for vkd3d-proton), sampled
 * 1D and 2D views are one-layer 2D arrays on Apple9 (see pack_texture), since
 * its texture instructions must match the descriptor's arrayness. Sample
 * every non-array 2D texture, 1D having been lowered to 2D, as layer 0 of an
 * array. Layer 0 of a one-layer view is the view itself, and an array
 * declaration bound to a 2D view now honours its layer and level.
 */
static bool
hk_apple9_lower_tex_2d_as_array(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->is_array || (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                         tex->sampler_dim != GLSL_SAMPLER_DIM_MS))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   int c = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (c >= 0) {
      nir_def *coord = tex->src[c].src.ssa;
      nir_def *layer = nir_tex_instr_src_type(tex, c) == nir_type_float
                          ? nir_imm_floatN_t(b, 0.0, coord->bit_size)
                          : nir_imm_intN_t(b, 0, coord->bit_size);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < coord->num_components; ++i)
         comps[i] = nir_channel(b, coord, i);
      comps[coord->num_components] = layer;
      nir_src_rewrite(&tex->src[c].src,
                      nir_vec(b, comps, coord->num_components + 1));
      tex->coord_components++;
   }
   tex->is_array = true;
   tex->backend_flags |= AGX_TEXTURE_FLAG_ARRAYED_2D;

   if (tex->op == nir_texop_txs) {
      /* Array sizes report the layer count too; return the 2D size. */
      unsigned old = tex->def.num_components;
      tex->def.num_components++;
      b->cursor = nir_after_instr(&tex->instr);
      nir_def *size = nir_trim_vector(b, &tex->def, old);
      nir_def_rewrite_uses_after(&tex->def, size);
   }
   return true;
}

static void
hk_lower_nir(struct hk_device *dev, nir_shader *nir,
             const struct vk_pipeline_robustness_state *rs, bool is_multiview,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             enum hk_feature_key features)
{
   if (HK_PERF(dev, NOROBUST)) {
      rs = &vk_robustness_disabled;
   }

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
               &(nir_input_attachment_options){
                  .use_view_id_for_layer = is_multiview,
               });

      NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
               NULL);
      NIR_PASS(_, nir, nir_lower_wpos_center);
   }

   /* XXX: should be last geometry stage, how do I get to that? */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (is_multiview)
         hk_lower_multiview(nir);
   }

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_patch_vertices,
               nir->info.tess.tcs_vertices_out, NULL);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   /* Turn cache flushes into image coherency bits while we still have derefs */
   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_robust_access, should_lower_robust, NULL);

   /* We must do early lowering before hk_nir_lower_descriptors, since this will
    * create lod_bias instructions.
    */
   /* Apple9 bounds-checks buffer images in software against their element
    * count (hk_apple9_lower_storage_image). */
   NIR_PASS(_, nir, agx_nir_lower_texture_early, true /* support_lod_bias */,
            agx_is_apple9(&dev->dev));
   if (hk_device_physical(dev)->apple9_d3d_mip_rounding)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_lower_d3d_txl_rounding,
               nir_metadata_control_flow, NULL);
   if (hk_device_physical(dev)->apple9_layered_2d)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_apple9_lower_tex_2d_as_array,
               nir_metadata_control_flow, NULL);

   struct hk_instance *instance = hk_physical_device_instance(
      (struct hk_physical_device *)dev->vk.physical);

   /* Apple9 pays for the clamp only when the application enables the
    * extension. */
   if (agx_is_apple9(&dev->dev) ? dev->vk.enabled_features.minLod
                                : instance->drirc.misc.image_view_min_lod) {
      NIR_PASS(_, nir, agx_nir_lower_image_view_min_lod,
               agx_is_apple9(&dev->dev));
   }

   /* Apple9 exposes no custom border colors. */
   if ((features & HK_FEAT_CUSTOM_BORDER) && !HK_PERF(dev, NOBORDER) &&
       !agx_is_apple9(&dev->dev)) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border);
   }

   NIR_PASS(_, nir, hk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_buffer_addr_format(rs->uniform_buffers));

   /* Before inserting bounds checks, we want to do a fair bit of optimization.
    * lower_load_global_constant_offset_instr has special optimizations for
    * constant offsets, so we want as many offsets to be constant as possible.
    */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   /* Apple9 vectorizes before the buffer base and offset become one 64-bit
    * address. Afterwards the offset is hidden inside u2u64, where adjacent
    * fields at x + c are no longer provably adjacent, and every field of a
    * dynamically indexed struct becomes a separate scalar load.
    */
   if (agx_is_apple9(&dev->dev)) {
      /* Explicit I/O builds each access's base and offset afresh. The
       * vectorizer only merges accesses sharing those definitions. */
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, nir_opt_load_store_vectorize,
               &(const nir_load_store_vectorize_options){
                  .modes = nir_var_mem_global,
                  .bounds_checked_modes = nir_var_mem_global,
                  .callback = hk_apple9_vectorize_cb,
               });
   }

   nir_move_options opts = nir_move_load_ssbo | nir_move_load_ubo;
   NIR_PASS(progress, nir, nir_opt_sink, opts);
   NIR_PASS(progress, nir, nir_opt_move, opts);

   NIR_PASS(_, nir, hk_lower_bounded_global_loads,
            agx_has_soft_fault(&dev->dev), agx_is_apple9(&dev->dev));

   assert(nir->info.shared_size == 0);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.stage == MESA_SHADER_TASK ||
       nir->info.stage == MESA_SHADER_MESH) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_mem_task_payload, shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16B so we can write whole vec4s. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* TODO: we can do indirect VS output */
   nir_variable_mode lower_indirect_modes = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      lower_indirect_modes |= nir_var_shader_out;
   else if (nir->info.stage == MESA_SHADER_VERTEX)
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;

   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            lower_indirect_modes, UINT32_MAX);

   /* nir_lower_io needs constant indices into compact arrays. Tess levels are
    * indexed dynamically (gl_TessLevelOuter[gl_InvocationID]), so turn them
    * into vectors and lower the component indexing, as RADV does.
    */
   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_tess_level_array_vars_to_vec);
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec,
               nir_var_shader_in | nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   }

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);
   if (agx_is_apple9(&dev->dev))
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_lower_io_16bit_to_32bit,
               nir_metadata_control_flow, NULL);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_lower_per_primitive_input,
               nir_metadata_control_flow, NULL);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      if (agx_is_apple9(&dev->dev)) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  hk_apple9_lower_layer_viewport_fs, nir_metadata_control_flow,
                  NULL);
      }

      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_viewport_fs,
               nir_metadata_control_flow, NULL);
   }

   if (agx_is_apple9(&dev->dev)) {
      NIR_PASS(_, nir, agx_nir_lower_storage, false, false);
      NIR_PASS(_, nir, agx_nir_lower_images);
      /* Storage-image queries read the descriptor, sampled ones metadata */
      NIR_PASS(_, nir, agx_nir_lower_texture_queries);

      NIR_PASS(_, nir, agx_nir_lower_texture);
   } else {
      NIR_PASS(_, nir, agx_nir_lower_texture);
   }
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   /* Keep radians until Apple9 range reduction. The common preprocessor
    * otherwise replaces sine/cosine with the older fsin_agx operation. */
   if (agx_is_apple9(&dev->dev))
      NIR_PASS(_, nir, agx_nir_lower_word_math);
   agx_preprocess_nir(nir);
   if (agx_is_apple9(&dev->dev))
      NIR_PASS(_, nir, agx_nir_lower_image_barriers);

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);
}

/* Byte-identical Apple9 bodies share one code allocation and shader entry.
 * Pipelines and draw-time variants often compile to the same code: in the
 * CS2 capture, three in four bodies duplicate another. The table keeps the
 * most recently uploaded bodies referenced, so a body may outlive its last
 * shader until evicted.
 */
#define HK_APPLE9_BODY_CACHE_SIZE 4096
/* Match the old count limit for minimum-size M4 BOs, but do not retain an
 * unbounded number of bytes when shader bodies span several pages. */
#define HK_APPLE9_BODY_CACHE_BYTES (64ull * 1024 * 1024)

#define HK_APPLE9_BODY_SLAB_SIZE (64 * 1024)
#define HK_APPLE9_BODY_ALIGNMENT 64
/* USC may prefetch beyond the last instruction. Keep the end of every slab
 * mapped but unallocated so its last body cannot prefetch into the VA guard. */
#define HK_APPLE9_BODY_TAIL 1024
#define HK_APPLE9_BODY_SLAB_USABLE                                             \
   (HK_APPLE9_BODY_SLAB_SIZE - HK_APPLE9_BODY_TAIL)

static VkResult
hk_upload_shader(struct hk_device *dev, struct hk_shader *shader)
{
   /* Apple9 preambles are in the linked program; upload constant data only */
   if (agx_is_apple9(&dev->dev)) {
      assert(!shader->b.info.rodata.size_16 && "no uniform preloads");

      if (shader->data_size) {
         shader->bo = agx_bo_create(&dev->dev, shader->data_size, 0,
                                    AGX_BO_LOW_VA, "Constant data");
         if (!shader->bo)
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

         void *map = agx_bo_map(shader->bo);
         if (!map)
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

         memcpy(map, shader->data_ptr, shader->data_size);
         shader->data_addr = shader->bo->va->addr;
      }

      if (!shader->linked.ht) {
         /* Hardware VS read IDs from a prolog; API ones link theirs per draw */
         struct agx_shader_part *prolog = NULL;
         if (shader->b.info.stage == MESA_SHADER_VERTEX) {
            struct agx_vs_prolog_key key = {.hw = true, .static_vi = true};
            prolog = hk_get_prolog_epilog(dev, &key, sizeof(key),
                                          agx_nir_vs_prolog, false, false, 0);
         }

         shader->only_linked =
            hk_fast_link(dev, false, shader, prolog, NULL, 0);
         if (!shader->only_linked->b.bo)
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      if (shader->info.stage == MESA_SHADER_COMPUTE) {
         const struct agx_apple9_program *p = &shader->only_linked->b.apple9;
         struct agx_apple9_launch_template *t = malloc(sizeof(*t));
         if (t && agx_apple9_compute_launch_template(
                     t, &p->profile, dev->dev.shader_base,
                     p->preamble_size ? shader->only_linked->b.bo->va->addr +
                                           p->preamble_offset
                                      : 0))
            shader->apple9_launch = t;
         else
            free(t);
      }

      return VK_SUCCESS;
   }

   if (shader->b.info.has_preamble || shader->b.info.rodata.size_16 ||
       shader->data_size) {
      /* TODO: Do we wnat to compact? Revisit when we rework prolog/epilogs. */
      uint64_t data_offset =
         ALIGN_POT((uint64_t)shader->b.info.binary_size, HK_MIN_UBO_ALIGNMENT);
      uint64_t size = data_offset + shader->data_size;
      if (size > SIZE_MAX)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      assert(size > 0);

      shader->bo = agx_bo_create(&dev->dev, size, 0,
                                 AGX_BO_EXEC | AGX_BO_LOW_VA, "Preamble");
      if (!shader->bo)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      void *map = agx_bo_map(shader->bo);
      if (!map)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      memset(map, 0, size);
      memcpy(map, shader->b.binary, shader->b.info.binary_size);
      if (shader->data_size) {
         memcpy((char *)map + data_offset, shader->data_ptr, shader->data_size);
         shader->data_addr = shader->bo->va->addr + data_offset;
      }
      shader->preamble_addr =
         shader->bo->va->addr + shader->b.info.preamble_offset;
   }

   if (!shader->linked.ht) {
      /* If we only have a single variant, link now. */
      shader->only_linked = hk_fast_link(dev, false, shader, NULL, NULL, 0);
   }

   if (shader->info.stage == MESA_SHADER_FRAGMENT) {
      agx_pack_fragment_face_2(&shader->frag_face, 0, &shader->b.info);
   }

   agx_pack(&shader->counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->b.info.push_count;
      cfg.preshader_register_count = shader->b.info.nr_preamble_gprs;
      cfg.sampler_state_register_count = agx_translate_sampler_state_count(
         shader->b.info.sampler_state_count, false);
      cfg.texture_state_register_count = shader->b.info.texture_state_count;
   }
   return VK_SUCCESS;
}

DERIVE_HASH_TABLE(hk_fast_link_key_vs);
DERIVE_HASH_TABLE(hk_fast_link_key_fs);

static VkResult
hk_init_link_ht(struct hk_shader *shader, mesa_shader_stage sw_stage)
{
   simple_mtx_init(&shader->linked.lock, mtx_plain);

   bool multiple_variants =
      sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_FRAGMENT;

   if (!multiple_variants)
      return VK_SUCCESS;

   if (sw_stage == MESA_SHADER_VERTEX)
      shader->linked.ht = hk_fast_link_key_vs_table_create(NULL);
   else
      shader->linked.ht = hk_fast_link_key_fs_table_create(NULL);

   return (shader->linked.ht == NULL) ? VK_ERROR_OUT_OF_HOST_MEMORY
                                      : VK_SUCCESS;
}

/* Which memory a shader reads, from its intrinsics: texture and image
 * operations read images; loads from any buffer, global or descriptor
 * memory and vertex attributes read buffers. Queries and stores do not
 * read. Unknown memory loads count as buffers. */
bool
hk_nir_reads_draw_id(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_draw_id)
               return true;
         }
      }
   }
   return false;
}

/* Recognize the address expressions produced by descriptor lowering. Stop at
 * memory loads: a descriptor pointer loaded from a root is not root memory.
 * Unknown arithmetic/phis conservatively keep the full push region. */
enum hk_root_address_kind {
   HK_ROOT_ADDRESS_OTHER,
   HK_ROOT_ADDRESS_CONSTANT,
   HK_ROOT_ADDRESS_ROOT,
   HK_ROOT_ADDRESS_UNKNOWN,
};

struct hk_root_address {
   enum hk_root_address_kind kind;
   uint64_t offset;
};

static struct hk_root_address
hk_root_address(nir_def *def, unsigned depth)
{
   if (depth > 32)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return (struct hk_root_address){HK_ROOT_ADDRESS_CONSTANT,
                                      nir_src_as_uint(nir_src_for_ssa(def))};

   if (instr->type == nir_instr_type_intrinsic) {
      return (struct hk_root_address){
         nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_root_agx
            ? HK_ROOT_ADDRESS_ROOT
            : HK_ROOT_ADDRESS_OTHER};
   }

   if (instr->type != nir_instr_type_alu)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   struct hk_root_address a = hk_root_address(alu->src[0].src.ssa, depth + 1);
   if (alu->op == nir_op_mov || alu->op == nir_op_u2u64)
      return a;

   unsigned n = nir_op_infos[alu->op].num_inputs;
   if (alu->op == nir_op_iadd) {
      struct hk_root_address b =
         hk_root_address(alu->src[1].src.ssa, depth + 1);
      if ((a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_ROOT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_ROOT)) {
         uint64_t offset =
            (a.offset + b.offset) & BITFIELD64_MASK(def->bit_size);
         return (struct hk_root_address){MAX2(a.kind, b.kind), offset};
      }
      return (struct hk_root_address){a.kind >= HK_ROOT_ADDRESS_ROOT ||
                                            b.kind >= HK_ROOT_ADDRESS_ROOT
                                         ? HK_ROOT_ADDRESS_UNKNOWN
                                         : HK_ROOT_ADDRESS_OTHER};
   }

   bool root = a.kind >= HK_ROOT_ADDRESS_ROOT;
   for (unsigned i = 1; i < n; ++i)
      root |= hk_root_address(alu->src[i].src.ssa, depth + 1).kind >=
              HK_ROOT_ADDRESS_ROOT;
   return (struct hk_root_address){root ? HK_ROOT_ADDRESS_UNKNOWN
                                        : HK_ROOT_ADDRESS_OTHER};
}

static uint64_t
hk_push_range_mask(uint64_t start, uint64_t size)
{
   static_assert(HK_MAX_PUSH_SIZE == 64 * sizeof(uint32_t));
   if (start >= HK_MAX_PUSH_SIZE || size > HK_MAX_PUSH_SIZE - start)
      return UINT64_MAX;
   unsigned first = start / 4;
   unsigned end = DIV_ROUND_UP(start + size, 4);
   return BITFIELD64_RANGE(first, end - first);
}

uint64_t
hk_nir_push_constant_mask(const nir_shader *nir)
{
   uint64_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            unsigned bytes = in->num_components * in->def.bit_size / 8;
            if (in->intrinsic == nir_intrinsic_load_push_constant) {
               if (!nir_src_is_const(in->src[0]))
                  return UINT64_MAX;
               mask |= hk_push_range_mask((uint64_t)nir_intrinsic_base(in) +
                                             nir_src_as_uint(in->src[0]),
                                          bytes);
            } else if (in->intrinsic == nir_intrinsic_load_deref &&
                       nir_deref_mode_is(nir_src_as_deref(in->src[0]),
                                         nir_var_mem_push_const)) {
               return UINT64_MAX;
            } else if (in->intrinsic == nir_intrinsic_load_global_constant ||
                       in->intrinsic == nir_intrinsic_load_global ||
                       in->intrinsic ==
                          nir_intrinsic_load_global_constant_offset) {
               struct hk_root_address a = hk_root_address(in->src[0].ssa, 0);
               if (in->intrinsic == nir_intrinsic_load_global_constant_offset &&
                   a.kind >= HK_ROOT_ADDRESS_ROOT) {
                  if (!nir_src_is_const(in->src[1]))
                     return UINT64_MAX;
                  a.offset += nir_src_as_uint(in->src[1]);
               }
               if (a.kind == HK_ROOT_ADDRESS_UNKNOWN)
                  return UINT64_MAX;
               if (a.kind != HK_ROOT_ADDRESS_ROOT)
                  continue;
               const uint64_t start = hk_root_descriptor_offset(push);
               const uint64_t end = start + HK_MAX_PUSH_SIZE;
               if (a.offset < end && bytes > start - MIN2(a.offset, start)) {
                  uint64_t lo = MAX2(a.offset, start);
                  uint64_t hi = MIN2(a.offset + bytes, end);
                  mask |= hk_push_range_mask(lo - start, hi - lo);
               }
            }
         }
      }
   }
   return mask;
}

#ifndef NDEBUG

/* Geometry lowering introduces application-memory address phis and vector
 * expressions which the early root walk conservatively calls unknown. They
 * cannot refer to our root unless they depend on load_root_agx. Follow phis
 * with a visited set, but stop at memory loads just like the early analysis.
 */
struct hk_debug_root_walk {
   struct set *visited;
   bool found;
};

static bool
hk_debug_root_source(nir_src *src, void *data)
{
   struct hk_debug_root_walk *walk = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   bool found;
   _mesa_set_search_or_add(walk->visited, instr, &found);
   if (found)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      walk->found |= nir_instr_as_intrinsic(instr)->intrinsic ==
                     nir_intrinsic_load_root_agx;
   } else {
      nir_foreach_src(instr, hk_debug_root_source, walk);
   }
   return !walk->found;
}

static bool
hk_debug_exclude_non_root_load(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global &&
       intr->intrinsic != nir_intrinsic_load_global_constant &&
       intr->intrinsic != nir_intrinsic_load_global_constant_offset)
      return false;
   struct hk_debug_root_walk walk = {
      .visited = _mesa_pointer_set_create(NULL),
   };
   assert(walk.visited);
   hk_debug_root_source(&intr->src[0], &walk);
   _mesa_set_destroy(walk.visited, NULL);
   if (walk.found)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0], nir_imm_int64(b, 0));
   return true;
}

uint64_t
hk_nir_final_push_constant_mask(const nir_shader *nir)
{
   nir_shader *copy = nir_shader_clone(NULL, nir);
   assert(copy);
   NIR_PASS(_, copy, nir_shader_intrinsics_pass, hk_debug_exclude_non_root_load,
            nir_metadata_control_flow, NULL);
   uint64_t mask = hk_nir_push_constant_mask(copy);
   ralloc_free(copy);
   return mask;
}

static void
hk_assert_final_push_mask(const nir_shader *nir, const struct hk_shader *shader)
{
   const struct hk_api_shader *obj = shader->debug_owner;
   assert(obj);
   uint64_t final = hk_nir_final_push_constant_mask(nir);
   assert((final & ~obj->push_constant_mask) == 0 ||
          obj->push_constant_mask == UINT64_MAX);
}
#else
#define hk_assert_final_push_mask(nir, shader) ((void)0)
#endif

uint8_t
hk_nir_pre_raster_reads(const nir_shader *nir)
{
   uint8_t reads = 0;
   if (nir->info.stage == MESA_SHADER_VERTEX && nir->info.inputs_read)
      reads |= HK_READS_BUFFERS;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (!nir_tex_instr_is_query(nir_instr_as_tex(instr)))
                  reads |= HK_READS_IMAGES;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_load:
            case nir_intrinsic_image_sparse_load:
            case nir_intrinsic_image_deref_load:
            case nir_intrinsic_image_deref_sparse_load:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_load_input_attachment_coord:
               reads |= HK_READS_IMAGES;
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_ubo_vec4:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_2x32:
            case nir_intrinsic_load_global_bounded:
            case nir_intrinsic_load_global_constant_bounded:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_input:
            case nir_intrinsic_load_input_vertex:
            case nir_intrinsic_load_per_vertex_input:
               reads |= HK_READS_BUFFERS;
               break;
            case nir_intrinsic_load_deref: {
               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               if (nir_deref_mode_may_be(
                      deref, nir_var_mem_ubo | nir_var_mem_ssbo |
                                nir_var_mem_global | nir_var_shader_in))
                  reads |= HK_READS_BUFFERS;
               if (nir_deref_mode_may_be(deref,
                                         nir_var_image | nir_var_uniform))
                  reads |= HK_READS_IMAGES;
               break;
            }
            default:
               break;
            }
         }
      }
   }
   return reads;
}

/* An early depth/stencil test updates the attachment before the fragment
 * shader runs, and the shader's input-attachment reads then observe this
 * fragment's own write. With a depth/stencil feedback loop, export the
 * interpolated depth at the end instead, so the test follows those reads. */
static void
hk_apple9_late_zs(nir_shader *nir)
{
   if (nir->info.fs.early_fragment_tests ||
       (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_frag_coord_z(&b), nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .src_type = nir_type_float32);
   nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* Hardware bounds shader-exported depth only to [0, 1]. Clamp depth exports,
 * including the side-effect pass's synthetic export, to the draw's clamp
 * range from the root table, as native Metal fragment shaders do with their
 * near/far uniforms.
 *
 * A NEVER depth test skips the fragment shader, which would drop side
 * effects, and a NaN export passes ALWAYS. Such draws therefore test LESS
 * (hk_cmd_apple9.c) against an exported depth above the storable range, so
 * every sample fails depth and still takes its stencil depth-fail op. */
static bool
hk_apple9_clamp_depth_export(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_DEPTH)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *range =
      hk_nir_load_root(b, 2, 32, hk_root_descriptor_offset(draw.depth_clamp));
   nir_def *z = intr->src[0].ssa;
   nir_def *clamped = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                               nir_channel(b, range, 1));
   nir_def *fail = hk_nir_load_root(
      b, 1, 16, hk_root_descriptor_offset(draw.force_never_in_shader));
   nir_src_rewrite(&intr->src[0], nir_bcsel(b, nir_ine_imm(b, fail, 0),
                                            nir_imm_float(b, 2.0), clamped));
   return true;
}

/* Load a field of the root table, see hk_root_descriptor_offset */
nir_def *
hk_nir_load_root(nir_builder *b, unsigned num_components, unsigned bit_size,
                 unsigned offset)
{
   return nir_load_global_constant(
      b, num_components, bit_size,
      nir_iadd_imm(b, nir_load_root_agx(b), offset), .align_mul = bit_size / 8,
      .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE);
}

static bool
lower_uniforms(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Root is first, descriptor sets follow immediately. */
   unsigned *root_ = data;
   unsigned root = *root_;
   unsigned sets = root + 4;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
      /* Change of units from sets to uniforms */
      nir_intrinsic_set_desc_set(intr,
                                 sets + (nir_intrinsic_desc_set(intr) * 4));
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_root_agx &&
       intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep;

   if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
      unsigned s = nir_intrinsic_desc_set(intr);
      rep = nir_load_preamble(b, 1, 64, .base = sets + (4 * s));
   } else {
      rep = nir_load_preamble(b, 1, 64, .base = root);
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

static bool
kill_psiz_write(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;

   if (nir_remove_sysval_output(intr, MESA_SHADER_FRAGMENT)) {
      b->shader->info.outputs_written &= ~VARYING_BIT_PSIZ;
   }

   return true;
}

static bool
hk_apple9_lower_clip_z_coeff(nir_builder *b, nir_intrinsic_instr *intr,
                             UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_clip_z_coeff)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(
      &intr->def,
      hk_nir_load_root(b, 1, 32, hk_root_descriptor_offset(draw.clip_z_coeff)));
   return true;
}

/* VK_EXT_depth_clip_control: one final position store with z += c (w - z) */
static bool
hk_apple9_lower_clip_z(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_variable *pos = NULL;
   nir_intrinsic_instr *first = NULL;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS)
            continue;

         assert(nir_src_bit_size(intr->src[0]) == 32 && "position is FP32");
         if (!pos) {
            pos = nir_local_variable_create(impl, glsl_vec4_type(), "pos");
            first = intr;
         }

         b.cursor = nir_before_instr(instr);
         unsigned c = nir_intrinsic_component(intr);
         nir_def *value = intr->src[0].ssa;
         nir_def *comps[4];
         for (unsigned i = 0; i < 4; ++i) {
            comps[i] = i >= c && i - c < value->num_components
                          ? nir_channel(&b, value, i - c)
                          : nir_undef(&b, 1, 32);
         }

         nir_store_var(&b, pos, nir_vec(&b, comps, 4),
                       nir_intrinsic_write_mask(intr) << c);

         if (intr != first)
            nir_instr_remove(instr);
      }
   }

   if (!pos)
      return nir_no_progress(impl);

   /* The shader's stores are complete at its end */
   nir_io_semantics sem = nir_intrinsic_io_semantics(first);
   unsigned base = nir_intrinsic_base(first);
   nir_instr_remove(&first->instr);

   b.cursor = nir_after_impl(impl);
   nir_def *value = nir_load_var(&b, pos);
   nir_def *z = nir_channel(&b, value, 2);
   nir_def *w = nir_channel(&b, value, 3);
   nir_def *c = nir_load_clip_z_coeff(&b);
   nir_def *new_z =
      nir_ffma_weak(&b, nir_fneg(&b, z), c, nir_ffma_weak(&b, w, c, z));
   nir_store_output(&b, nir_vector_insert_imm(&b, value, new_z, 2),
                    nir_imm_int(&b, 0), .base = base, .write_mask = 0xf,
                    .src_type = nir_type_float32, .io_semantics = sem);

   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   return true;
}

static void
hk_lower_hw_vs(nir_shader *nir, struct hk_shader *shader, bool kill_psiz)
{
   /* VK_EXT_depth_clip_control: Apple9 clips depth to [0, w] */
   if (agx_nir_shader_isa(nir) == AGX_SHADER_ISA_APPLE9) {
      NIR_PASS(_, nir, hk_apple9_lower_clip_z);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_clip_z_coeff,
               nir_metadata_control_flow, NULL);
   }

   /* Point size must be clamped, excessively large points don't render
    * properly on G13.
    *
    * Must be synced with pointSizeRange.
    */
   NIR_PASS(_, nir, nir_lower_point_size, 1.0f, 511.95f);

   if (kill_psiz) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, kill_psiz_write,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);

   NIR_PASS(_, nir, agx_nir_lower_uvs, &shader->info.uvs);
}

static VkResult
hk_copy_shader_constant_data(struct hk_device *dev, struct hk_shader *shader,
                             const nir_shader *nir)
{
   if (nir->constant_data_size > 0) {
      uint64_t data_size =
         ALIGN_POT((uint64_t)nir->constant_data_size, HK_MIN_UBO_ALIGNMENT);
      /* The shader cache stores data lengths in 32 bits. */
      if (data_size > UINT32_MAX)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      void *data = malloc(data_size);
      if (data == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      memcpy(data, nir->constant_data, nir->constant_data_size);

      assert(nir->constant_data_size <= data_size);
      memset(data + nir->constant_data_size, 0,
             data_size - nir->constant_data_size);

      shader->data_ptr = data;
      shader->data_size = data_size;
   }

   return VK_SUCCESS;
}

/* Software stages run as compute, at their compiled workgroup size */
static void
hk_lower_to_compute(struct hk_device *dev, nir_shader *nir, unsigned local_size)
{
   if (agx_is_apple9(&dev->dev)) {
      nir->info.workgroup_size_variable = false;
      nir->info.workgroup_size[0] = local_size;
      nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   } else if (nir->info.stage != MESA_SHADER_VERTEX &&
              nir->info.stage != MESA_SHADER_TESS_EVAL) {
      /* Apple8 compiles the other software stages as they are */
      return;
   }

   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;
}

static VkResult
hk_compile_nir(struct hk_device *dev, const VkAllocationCallbacks *pAllocator,
               nir_shader *nir, VkShaderCreateFlagsEXT shader_flags,
               const struct vk_pipeline_robustness_state *rs,
               const union hk_key *key, enum hk_feature_key features,
               struct hk_shader *shader, mesa_shader_stage sw_stage, bool hw,
               nir_xfb_info *xfb_info, unsigned set_count)
{
   unsigned nr_vbos = 0;
   bool kill_psiz = false;

   /* For now, only shader objects are supported */
   if (sw_stage == MESA_SHADER_VERTEX) {
      nr_vbos = DIV_ROUND_UP(
         BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
      kill_psiz = key->vs.kill_psiz;
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      shader->info.fs.writes_memory = nir->info.writes_memory;
      shader->info.fs.interp.linear = nir->info.linear_varyings;
      shader->info.fs.interp.flat =
         ~(nir->info.linear_varyings | nir->info.perspective_varyings);

      /* Discards must be lowering before lowering MSAA to handle discards */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog,
               &shader->info.fs.epilog_key);

      /* Export depth for draw-time Z/S feedback loops, see hk_apple9_late_zs */
      if (key && key->fs.dynamic_zs_self_dep && !key->fs.zs_self_dep &&
          agx_is_apple9(&dev->dev) && !nir->info.fs.early_fragment_tests &&
          !shader->info.fs.epilog_key.write_z) {
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
         nir_export_agx(&b, nir_load_frag_coord_z(&b), .base = AGX_ABI_FOUT_Z);
         shader->info.fs.dynamic_late_zs = true;
      }

      NIR_PASS(_, nir, agx_nir_lower_sample_mask);

      if (nir->info.fs.uses_sample_shading) {
         /* Ensure the sample mask is preserved in register */
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

         nir_def *mask =
            nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

         nir_export_agx(&b, mask, .base = AGX_ABI_FOUT_SAMPLE_MASK);

         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
      }

      NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);
      NIR_PASS(_, nir, agx_nir_lower_interpolation);
   } else if (sw_stage == MESA_SHADER_TESS_EVAL ||
              sw_stage == MESA_SHADER_TESS_CTRL) {

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      if (sw_stage == MESA_SHADER_TESS_CTRL) {
         shader->info.tess.tcs_output_patch_size =
            nir->info.tess.tcs_vertices_out;
         shader->info.tess.tcs_per_vertex_outputs =
            poly_tcs_per_vertex_outputs(nir);
         shader->info.tess.tcs_nr_patch_outputs =
            util_last_bit(nir->info.patch_outputs_written);
         shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
      } else {
         /* Apple9 counts tessellated primitives per draw: publish patch ID */
         if (agx_is_apple9(&dev->dev) && hw &&
             !(nir->info.outputs_written & VARYING_BIT_PRIMITIVE_ID))
            hk_apple9_write_primitive_id(nir);

         /* This destroys info so it needs to happen after the gather */
         NIR_PASS(_, nir, poly_nir_lower_tes, hw);
      }
   }

   /* Normally, vertex shaders need to write a default point size. However, if
    * we have a geometry/tessellation shader, the hardware vertex (software
    * GS/TES) will handle this itself instead.
    */
   if (sw_stage == MESA_SHADER_VERTEX && hw && !kill_psiz) {
      NIR_PASS(_, nir, nir_lower_default_point_size);
   }

   uint64_t outputs = nir->info.outputs_written;
   if (sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_TESS_EVAL) {
      if (hw) {
         hk_lower_hw_vs(nir, shader, kill_psiz);
      } else {
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         hk_lower_to_compute(dev, nir, 64);
      }
   } else if (sw_stage == MESA_SHADER_TESS_CTRL) {
      hk_lower_to_compute(dev, nir, nir->info.tess.tcs_vertices_out);
   }

   unsigned root = 0;
   if (sw_stage == MESA_SHADER_FRAGMENT)
      root = AGX_ABI_FUNI_ROOT;
   else if (sw_stage == MESA_SHADER_VERTEX)
      root = AGX_ABI_VUNI_COUNT_VK(nr_vbos);

   shader->info.set_count = set_count;

   /* XXX: rename */
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, nr_vbos);
   if (agx_is_apple9(&dev->dev))
      hk_assert_final_push_mask(nir, shader);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_uniforms,
            nir_metadata_control_flow, &root);

#if 0
   /* TODO */
   nir_variable_mode robust2_modes = 0;
   if (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ubo;
   if (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ssbo;
#endif

   struct agx_shader_key backend_key = {
      .reserved_preamble = root + (4 * (1 + set_count)),
      .dev = agx_gather_device_key(&dev->dev),

      /* Fast linking adds the stop, for FS and Apple9 VS */
      .no_stop =
         nir->info.stage == MESA_SHADER_FRAGMENT ||
         (agx_is_apple9(&dev->dev) && nir->info.stage == MESA_SHADER_VERTEX),
      .has_scratch = !nir->info.internal,
      .promote_constants = true,
      .promote_textures = true,

      /* Compute stages run API shaders, which read push words */
      .apple9_graphics_abi = true,
   };

   /* For now, sample shading is always dynamic. Indicate that. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.uses_sample_shading)
      backend_key.fs.inside_sample_loop = true;

   simple_mtx_t *lock = NULL;
   if (agx_get_compiler_debug())
      lock = &hk_device_physical(dev)->debug_compile_lock;

   if (lock)
      simple_mtx_lock(lock);

   assert(nir->info.io_lowered);
   agx_compile_shader_nir(nir, &backend_key, &shader->b);

   if (lock)
      simple_mtx_unlock(lock);

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;

   shader->info.stage = sw_stage;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   shader->b.info.outputs = outputs;

   if (xfb_info) {
      assert(xfb_info->output_count < ARRAY_SIZE(shader->info.xfb_outputs));

      memcpy(&shader->info.xfb_info, xfb_info,
             nir_xfb_info_size(xfb_info->output_count));

      typed_memcpy(shader->info.xfb_stride, nir->info.xfb_stride, 4);
   }

   VkResult data_result = hk_copy_shader_constant_data(dev, shader, nir);
   if (data_result != VK_SUCCESS) {
      ralloc_free(nir);
      return data_result;
   }

   ralloc_free(nir);

   VkResult result = hk_init_link_ht(shader, sw_stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);

   return hk_upload_shader(dev, shader);
}

static const struct vk_shader_ops hk_shader_ops;

static void
hk_destroy_linked_shader(struct hk_device *dev, struct hk_linked_shader *linked)
{
   if (agx_is_apple9(&dev->dev))
      agx_apple9_stage_launch_cache_finish(&linked->apple9_launch);

   agx_bo_unreference(&dev->dev, linked->b.bo);
   ralloc_free(linked);
}

static void
hk_shader_destroy(struct hk_device *dev, struct hk_shader *s)
{
   free((void *)s->code_ptr);
   free((void *)s->data_ptr);
   agx_bo_unreference(&dev->dev, s->bo);
   free(s->apple9_launch);
   s->apple9_launch = NULL;

   simple_mtx_destroy(&s->linked.lock);

   if (s->only_linked)
      hk_destroy_linked_shader(dev, s->only_linked);

   if (s->linked.ht) {
      hash_table_foreach(s->linked.ht, entry) {
         hk_destroy_linked_shader(dev, entry->data);
      }
      _mesa_hash_table_destroy(s->linked.ht, NULL);
   }
}

void
hk_api_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                      const VkAllocationCallbacks *pAllocator)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   hk_foreach_variant(obj, shader) {
      hk_shader_destroy(dev, shader);
   }

   vk_shader_free(&dev->vk, pAllocator, &obj->vk);
}

/* Task/mesh lowering divides after agx_preprocess_nir */
static void
hk_lower_late_division(nir_shader *nir)
{
   nir_lower_idiv_options idiv_options = {.allow_fp16 = true};

   NIR_PASS(_, nir, nir_opt_idiv_const, 16);
   NIR_PASS(_, nir, nir_lower_idiv, &idiv_options);
   NIR_PASS(_, nir, agx_nir_lower_accurate_frcp);
}

/* Apple9 FS lowering for hardware quirks */
static void
hk_apple9_lower_fs(nir_shader *nir, struct hk_shader *shader)
{
   /* Shading per sample interpolates at the sample */
   if (nir->info.fs.uses_sample_shading) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_apple9_lower_bary_to_sample, nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_clamp_depth_export,
            nir_metadata_control_flow, NULL);

   shader->info.fs.pixel_interlock = nir->info.fs.pixel_interlock_ordered ||
                                     nir->info.fs.pixel_interlock_unordered;

   /* Merged triangles share interpolation, wrong away from the pixel centre */
   shader->info.fs.disable_tri_merging = nir->info.fs.uses_sample_shading ||
                                         hk_apple9_interpolates_off_center(nir);
}

VkResult
hk_compile_shader(struct hk_device *dev, struct vk_shader_compile_info *info,
                  const struct vk_graphics_pipeline_state *state,
                  const struct vk_features *vk_features,
                  const VkAllocationCallbacks *pAllocator,
                  struct hk_api_shader **shader_out)
{
   VkResult result;
   enum hk_feature_key features = hk_make_feature_key(vk_features);
   bool apple9 = agx_is_apple9(&dev->dev);

   /* We consume the NIR, regardless of success or failure */
   nir_shader *nir = info->nir;

   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(info->stage);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, info->stage, pAllocator, size);

   if (obj == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   obj->pre_raster_reads = HK_READS_UNKNOWN;
   obj->reads_draw_id = hk_nir_reads_draw_id(nir);
   obj->push_constant_mask = hk_nir_push_constant_mask(nir);
#ifndef NDEBUG
   hk_foreach_variant(obj, variant)
      variant->debug_owner = obj;
#endif

   if (!nir->info.io_lowered) {
      hk_lower_nir(dev, nir, info->robustness, false, info->set_layout_count,
                   info->set_layouts, features);
   }

   /* Apple9 renders track what their stages read and write, see hk_queue */
   if (apple9) {
      obj->writes_memory = nir->info.writes_memory;
      obj->pre_raster_reads = hk_nir_pre_raster_reads(nir);
   }

   mesa_shader_stage sw_stage = nir->info.stage;

   union hk_key key_tmp, *key = NULL;
   if (sw_stage == MESA_SHADER_FRAGMENT) {
      hk_populate_fs_key(&key_tmp.fs, state, vk_features);
      key = &key_tmp;

      nir->info.fs.uses_sample_shading |= key->fs.force_sample_shading;

      /* Force late-Z for Z/S self-deps; Apple9 exports depth for that */
      if (key->fs.zs_self_dep && apple9) {
         hk_apple9_late_zs(nir);
      } else if (key->fs.zs_self_dep) {
         /* TODO: There's probably a less silly way to do this. */
         nir_builder b =
            nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_discard_if(&b, nir_imm_false(&b));
         nir->info.fs.uses_discard = true;
      }

      if (apple9)
         hk_apple9_lower_fs(nir, hk_only_variant(obj));

      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
   } else if (sw_stage == MESA_SHADER_VERTEX) {
      hk_populate_vs_key(&key_tmp.vs, state);
      key = &key_tmp;
   } else if (sw_stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   } else if (sw_stage == MESA_SHADER_MESH) {
      /* Mesh shaders run as compute; hk_nir_mesh_rast rasterizes records */
      struct hk_shader *shader = hk_only_variant(obj);
      hk_mesh_gather_info(nir, &shader->info.mesh);
      shader->info.clip_distance_array_size =
         nir->info.clip_distance_array_size;
      shader->info.cull_distance_array_size =
         nir->info.cull_distance_array_size;

      hk_nir_lower_mesh_main(nir, &shader->info.mesh);
      hk_lower_late_division(nir);
      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->xfb_info = NULL;
   } else if (sw_stage == MESA_SHADER_TASK) {
      struct hk_shader *shader = hk_only_variant(obj);
      shader->info.task.payload_B = nir->info.task_payload_size;
      shader->info.task.max_mesh_groups = hk_nir_task_mesh_bound(nir);

      hk_nir_lower_task_main(nir);
      hk_lower_late_division(nir);
      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   }

   /* Compile all variants up front */
   if (sw_stage == MESA_SHADER_GEOMETRY) {
      struct hk_shader *main_variant = hk_main_gs_variant(obj);
      struct hk_shader *count_variant = hk_count_gs_variant(obj);

      nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;

      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre_gs,
               &count_variant->info.gs);

      agx_preprocess_nir(count);
      agx_preprocess_nir(rast);
      agx_preprocess_nir(pre_gs);

      struct hk_shader *shader = &obj->variants[HK_GS_VARIANT_RAST];
      hk_lower_hw_vs(rast, shader, false);
      shader->info.gs = count_variant->info.gs;
      main_variant->info.gs = count_variant->info.gs;

      struct {
         nir_shader *in;
         struct hk_shader *out;
      } variants[] = {
         {nir, hk_main_gs_variant(obj)},
         {pre_gs, hk_pre_gs_variant(obj)},
         {count, count_variant},
         {rast, &obj->variants[HK_GS_VARIANT_RAST]},
      };

      for (unsigned v = 0; v < ARRAY_SIZE(variants); ++v) {
         if (variants[v].in) {
            if (variants[v].in != rast) {
               hk_lower_to_compute(dev, variants[v].in,
                                   variants[v].in == pre_gs ? 1 : 64);
            }

            result =
               hk_compile_nir(dev, pAllocator, variants[v].in, info->flags,
                              info->robustness, NULL, features, variants[v].out,
                              sw_stage, true, NULL, info->set_layout_count);

            if (result != VK_SUCCESS) {
               hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
               ralloc_free(nir);
               ralloc_free(pre_gs);
               ralloc_free(count);
               ralloc_free(rast);
               return result;
            }
         }
      }
   } else if (sw_stage == MESA_SHADER_VERTEX ||
              sw_stage == MESA_SHADER_TESS_EVAL) {

      VkShaderStageFlags next_stage = info->next_stage_mask;

      /* Transform feedback is layered on top of geometry shaders. If there is
       * not a geometry shader in the pipeline, we will compile a geometry
       * shader for the purpose. Update the next_stage mask accordingly.
       */
      if (nir->xfb_info != NULL) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      /* Apple9 API shaders may need it for these features at draw time */
      if (apple9 && vk_features &&
          dev->vk.enabled_features.primitivesGeneratedQuery) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      if (sw_stage == MESA_SHADER_VERTEX) {
         assert(
            !(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
            "Fixed-function attributes not used in Vulkan");

         NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
      }

      /* Fold constant offset srcs for IO. */
      NIR_PASS(_, nir, nir_opt_constant_folding);

      for (enum hk_vs_variant v = 0; v < HK_VS_VARIANTS; ++v) {
         /* Only compile the software variant if we might use this shader with
          * geometry/tessellation. We need to compile the hardware variant
          * unconditionally to handle the VS -> null FS case, which does not
          * require setting the FRAGMENT bit.
          */
         if (v == HK_VS_VARIANT_SW &&
             !(next_stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                             VK_SHADER_STAGE_GEOMETRY_BIT)))
            continue;

         struct hk_shader *shader = &obj->variants[v];
         bool hw = v == HK_VS_VARIANT_HW;
         bool last = (v + 1) == HK_VS_VARIANTS;

         /* Each variant gets its own NIR. To save an extra clone, we use the
          * original NIR for the last stage.
          */
         nir_shader *clone = last ? nir : nir_shader_clone(NULL, nir);

         NIR_PASS(_, clone, agx_nir_gather_vs_inputs,
                  shader->info.vs.attrib_components_read);

         if (sw_stage == MESA_SHADER_VERTEX) {
            /* Apple9 VS read vertex and instance IDs from the prolog */
            shader->info.vs.use_prolog =
               apple9 || !(key && key->vs.skip_prolog);
            shader->info.vs.attribs_read =
               nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;

            if (shader->info.vs.use_prolog) {
               NIR_PASS(_, clone, agx_nir_lower_vs_input_to_prolog);
            } else {
               struct agx_velem_key attribs[AGX_MAX_ATTRIBS];
               for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
                  if (key->vs.attribs[a].format) {
                     unsigned slot = util_bitcount64(
                        shader->info.vs.attribs_read & BITFIELD_MASK(a));

                     attribs[slot] = key->vs.attribs[a];
                  }
               }

               struct agx_robustness agx_rs = {
                  .soft_fault = agx_has_soft_fault(&dev->dev),

                  /* Correctly handling GPL + pipeline-robustness requires
                   * runtime changes, and I don't care enough to optimize this.
                   */
                  .level = AGX_ROBUSTNESS_D3D,
               };

               agx_nir_lower_vbo(clone, attribs, agx_rs,
                                 !key->vs.static_strides);

               unsigned nr = DIV_ROUND_UP(
                  BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
               agx_nir_lower_non_monolithic_uniforms(clone, nr);
            }
         }

         /* hk_compile_nir takes ownership of the clone */
         result =
            hk_compile_nir(dev, pAllocator, clone, info->flags,
                           info->robustness, key, features, shader, sw_stage,
                           hw, nir->xfb_info, info->set_layout_count);
         if (result != VK_SUCCESS) {
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            ralloc_free(nir);
            return result;
         }
      }
   } else {
      struct hk_shader *shader = hk_only_variant(obj);

      /* hk_compile_nir takes ownership of nir */
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                              info->robustness, key, features, shader, sw_stage,
                              true, NULL, info->set_layout_count);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = obj;
   return VK_SUCCESS;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

static VkResult
hk_compile_shaders_timed(struct vk_device *vk_dev, uint32_t shader_count,
                         struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *pAllocator,
                         struct vk_shader **shaders_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   nir_shader *shaders[shader_count];

   /* Lower shaders, notably lowering I/O. This is a prerequisite for
    * intershader optimization.
    */
   for (uint32_t i = 0; i < shader_count; i++) {
      /* Shader objects have no pipeline robustness, but still follow the
       * device's robustness features. The runtime passes them disabled.
       */
      if (infos[i].robustness == &vk_robustness_disabled)
         infos[i].robustness = &dev->vk.robustness_state;

      const struct vk_shader_compile_info *info = &infos[i];
      /* TODO: Multiview with ESO */
      const bool is_multiview = state && state->mv->view_mask != 0;
      enum hk_feature_key hk_features = hk_make_feature_key(features);
      nir_shader *nir = info->nir;

      hk_lower_nir(dev, nir, info->robustness, is_multiview,
                   info->set_layout_count, info->set_layouts, hk_features);

      if (nir->xfb_info) {
         /* Fold constant offset srcs for IO. */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         nir_io_add_intrinsic_xfb_info(nir);
      }

      shaders[i] = nir;
   }

   /* Mesh primitive outputs link to plain flat inputs. */
   bool mesh = false;
   for (uint32_t i = 0; i < shader_count; i++)
      mesh |= infos[i].stage == MESA_SHADER_MESH;
   if (!mesh) {
      nir_opt_varyings_bulk(shaders, shader_count, true, UINT32_MAX, UINT32_MAX,
                            nir_opts, NULL);
   } else {
      /* Interpolation masks are otherwise gathered by the linker. */
      for (uint32_t i = 0; i < shader_count; i++)
         nir_shader_gather_info(shaders[i],
                                nir_shader_get_entrypoint(shaders[i]));
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      VkResult result =
         hk_compile_shader(dev, &infos[i], state, features, pAllocator,
                           (struct hk_api_shader **)&shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            hk_api_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(infos[j].nir);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }
   }

   return VK_SUCCESS;
}

static VkResult
hk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   VkResult result = hk_compile_shaders_timed(
      vk_dev, shader_count, infos, state, features, pAllocator, shaders_out);
   return result;
}

static VkResult
hk_deserialize_shader(struct hk_device *dev, struct blob_reader *blob,
                      struct hk_shader *shader)
{
   struct hk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));

   struct agx_shader_info b_info;
   blob_copy_bytes(blob, &b_info, sizeof(b_info));

   const uint32_t code_size = blob_read_uint32(blob);
   const uint32_t data_size = blob_read_uint32(blob);
   /* Validate payload lengths before any code/data allocation, including
    * shader binaries.
    * Subtract successively so even a 32-bit host cannot overflow the sum. */
   if (blob->overrun || (unsigned)info.stage >= MESA_SHADER_MESH_STAGES ||
       code_size > blob->end - blob->current ||
       data_size > (blob->end - blob->current) - code_size)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   VkResult result = hk_init_link_ht(shader, info.stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_init(&shader->linked.lock, mtx_plain);

   shader->b.info = b_info;
   shader->info = info;
   shader->code_size = code_size;
   shader->data_size = data_size;
   shader->b.info.binary_size = code_size;

   shader->code_ptr = malloc(MAX2(code_size, 1));
   if (shader->code_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->data_ptr = malloc(MAX2(data_size, 1));
   if (shader->data_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   blob_copy_bytes(blob, (void *)shader->code_ptr, shader->code_size);
   blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.binary = (void *)shader->code_ptr;
   return hk_upload_shader(dev, shader);
}

/* Version the container independently of the hardware programs it owns. */
#define HK_SHADER_CONTAINER_VERSION 0x484b000cu

static VkResult
hk_deserialize_api_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   if (blob_read_uint32(blob) != HK_SHADER_CONTAINER_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   mesa_shader_stage stage = blob_read_uint8(blob);
   if (blob->overrun || stage >= MESA_SHADER_MESH_STAGES)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(stage);

   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, pAllocator, size);

   if (obj == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   hk_foreach_variant(obj, shader) {
      VkResult result = hk_deserialize_shader(dev, blob, shader);

      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }
   obj->writes_memory = blob_read_uint8(blob);
   obj->pre_raster_reads = blob_read_uint8(blob);
   obj->reads_draw_id = blob_read_uint8(blob);
   blob_copy_bytes(blob, &obj->push_constant_mask,
                   sizeof(obj->push_constant_mask));
   if (blob->overrun) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

static void
hk_shader_serialize(struct vk_device *vk_dev, const struct hk_shader *shader,
                    struct blob *blob)
{
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   blob_write_bytes(blob, &shader->b.info, sizeof(shader->b.info));

   blob_write_uint32(blob, shader->code_size);
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->code_ptr, shader->code_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
}

static bool
hk_api_shader_serialize(struct vk_device *vk_dev,
                        const struct vk_shader *vk_shader, struct blob *blob)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   blob_write_uint32(blob, HK_SHADER_CONTAINER_VERSION);
   blob_write_uint8(blob, vk_shader->stage);

   hk_foreach_variant(obj, shader) {
      hk_shader_serialize(vk_dev, shader, blob);
   }
   /* Preserve the API shader's memory effects on a cache hit. In particular,
    * a read-only vertex shader must not turn its render into a memory writer
    * and prevent later renders from overlapping tiling with fragment work.
    */
   blob_write_uint8(blob, obj->writes_memory);
   blob_write_uint8(blob, obj->pre_raster_reads);
   blob_write_uint8(blob, obj->reads_draw_id);
   blob_write_bytes(blob, &obj->push_constant_mask,
                    sizeof(obj->push_constant_mask));

   return !blob->out_of_memory;
}

static VkResult
hk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(obj->vk.stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(obj->vk.stage));
      VK_PRINT_STR(props->description, "%s shader",
                   _mesa_shader_stage_to_string(obj->vk.stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   assert(executable_index == 0);

   /* TODO: find a sane way to report multiple variants and have that play nice
    * with zink.
    */
   struct hk_shader *shader = hk_any_variant(obj);
   vk_add_agx2_stats(out, &shader->b.info.stats);
   return vk_outarray_status(&out);
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
hk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);
   bool incomplete_text = false;

   assert(executable_index == 0);

   /* TODO */
#if 0
   vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR, &out, ir) {
      VK_COPY_STR(ir->name, "AGX assembly");
      VK_COPY_STR(ir->description, "AGX assembly");
      if (!write_ir_text(ir, TODO))
         incomplete_text = true;
   }
#endif

   return incomplete_text ? VK_INCOMPLETE : vk_outarray_status(&out);
}

static const struct vk_shader_ops hk_shader_ops = {
   .destroy = hk_api_shader_destroy,
   .serialize = hk_api_shader_serialize,
   .get_executable_properties = hk_shader_get_executable_properties,
   .get_executable_statistics = hk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      hk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops hk_device_shader_ops = {
   .get_nir_options = hk_get_nir_options,
   .get_spirv_options = hk_get_spirv_options,
   .preprocess_nir = hk_preprocess_nir,
   .hash_state = hk_hash_graphics_state,
   .compile = hk_compile_shaders,
   .deserialize = hk_deserialize_api_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = hk_cmd_bind_shaders,
};

/* Apple9 launch state; varyings and coefficients are filled in per draw */
static void
hk_apple9_init_linked(struct hk_device *dev, struct hk_linked_shader *s,
                      bool fragment, struct hk_shader *main,
                      struct agx_shader_part *prolog,
                      struct agx_shader_part *epilog)
{
   const struct agx_apple9_program *p = &s->b.apple9;
   agx_apple9_stage_launch_cache_init(&s->apple9_launch);

   s->apple9_stage = (struct agx_apple9_render_stage){
      .program_id = p_atomic_inc_return(&dev->apple9_last_program_id),
      .bo = s->b.bo,
      .binary = s->b.bo ? agx_bo_map(s->b.bo) : NULL,
      .binary_size = p->size,
      .preamble_offset = p->preamble_offset,
      .preamble_size = p->preamble_size,
      .scratch_size = p->profile.scratch_size,
      .atomic_frame_size = p->profile.atomic_frame_size,
      .publication_count = p->profile.publication_count,
      .publication_count_valid = true,
      .resource_count = p->profile.resource_binding_count,
      .launch_cache = &s->apple9_launch,
      .position_components = fragment ? 0 : 4,
   };

   for (unsigned i = 0; i < p->profile.resource_binding_count; ++i) {
      assert(p->profile.resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_SSBO &&
             "API shaders reach buffers through descriptors");
      s->apple9_stage.resource_binding[i] = p->profile.resource_binding[i];
   }

   if (fragment) {
      struct agx_shader_part *parts[] = {prolog, &main->b, epilog};
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         if (parts[i]) {
            s->apple9_stage.reads_tile |= parts[i]->info.apple9_tile_reads;
            s->apple9_stage.uses_discard |= parts[i]->info.writes_sample_mask;
         }
      }

      s->apple9_stage.writes_depth = main->info.fs.epilog_key.write_z;
      s->apple9_stage.disable_tri_merging =
         main->b.info.disable_tri_merging || main->info.fs.disable_tri_merging;
      s->apple9_stage.pixel_interlock = main->info.fs.pixel_interlock;
   }
}

struct hk_linked_shader *
hk_fast_link(struct hk_device *dev, bool fragment, struct hk_shader *main,
             struct agx_shader_part *prolog, struct agx_shader_part *epilog,
             unsigned nr_samples_shaded)
{
   struct hk_linked_shader *s = rzalloc(NULL, struct hk_linked_shader);
   agx_fast_link(&s->b, &dev->dev, fragment, &main->b, prolog, epilog,
                 nr_samples_shaded, NULL);

   /* Apple9 programs are launched from their profile, see hk_cmd_apple9.c */
   if (agx_is_apple9(&dev->dev)) {
      hk_apple9_init_linked(dev, s, fragment, main, prolog, epilog);
      return s;
   }

   if (fragment) {
      unsigned samplers = main->b.info.sampler_state_count;
      if (s->b.uses_txf)
         samplers = MAX2(samplers, 1);

      agx_pack(&s->fs_counts, FRAGMENT_SHADER_WORD_0, cfg) {
         cfg.cf_binding_count = s->b.cf.nr_bindings;
         cfg.uniform_register_count = main->b.info.push_count;
         cfg.preshader_register_count = main->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count = main->b.info.texture_state_count;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(samplers, false);
      }
   }

   /* Now that we've linked, bake the USC words to bind this program */
   struct agx_usc_builder b = agx_usc_builder(s->usc.data, sizeof(s->usc.data));

   if (main && main->b.info.rodata.size_16) {
      agx_usc_immediates(&b, &main->b.info.rodata, main->bo->va->addr);
   }

   if (s->b.uses_txf)
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);

   agx_usc_shared_non_fragment(&b, &main->b.info, 0);
   agx_usc_push_packed(&b, SHADER, s->b.shader);
   agx_usc_push_packed(&b, REGISTERS, s->b.regs);

   if (fragment)
      agx_usc_push_packed(&b, FRAGMENT_PROPERTIES, s->b.fragment_props);

   if (main && main->b.info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&dev->dev, main->preamble_addr);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   s->usc.size = b.head - s->usc.data;
   return s;
}
