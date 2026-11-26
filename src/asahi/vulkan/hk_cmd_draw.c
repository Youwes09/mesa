/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include "asahi/compiler/agx_nir.h"
#include "asahi/lib/agx_cdm.h"
#include "asahi/lib/agx_draw_table.h"
#include "asahi/lib/agx_launch_program.h"
#include "asahi/lib/agx_ppp_record.h"
#include "asahi/lib/agx_resource_layout.h"
#include "agx_abi.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_ppp.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_feedback.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/mesh.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "pipe/p_defines.h"
#include "poly/geometry.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

/* CTS coverage of indirect draws is pretty bad, so it's helpful to be able to
 * get some extra smoke testing.
 */
#define HK_TEST_INDIRECTS (0)

UNUSED static inline void
print_draw(struct agx_draw d, FILE *fp)
{
   if (agx_is_indirect(d.b))
      fprintf(fp, "indirect (buffer %" PRIx64 "):", d.b.ptr);
   else
      fprintf(fp, "direct (%ux%u):", d.b.count[0], d.b.count[1]);

   if (d.index_size)
      fprintf(fp, " index_size=%u", agx_index_size_to_B(d.index_size));
   else
      fprintf(fp, " non-indexed");

   if (d.restart)
      fprintf(fp, " restart");

   if (d.index_bias)
      fprintf(fp, " index_bias=%u", d.index_bias);

   if (d.start)
      fprintf(fp, " start=%u", d.start);

   if (d.start_instance)
      fprintf(fp, " start_instance=%u", d.start_instance);

   fprintf(fp, "\n");
}

static bool
hk_rast_discard(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* A non-zero rasterization stream acts as a rasterizer discard unless
    * there's a multistream geometry shader bound.
    */
   if (dyn->rs.rasterization_stream != 0) {
      struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      return !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }

   return dyn->rs.rasterizer_discard_enable;
}

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
}

static void
hk_attachment_use_implicit_sync(struct hk_cmd_buffer *cmd,
                                const struct hk_attachment *att)
{
   if (att->iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
   if (att->resolve_iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->resolve_iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }

   const VkRenderingAttachmentFlagsInfoKHR *flags =
      vk_find_struct_const(info->pNext, RENDERING_ATTACHMENT_FLAGS_INFO_KHR);
   if (flags)
      att->flags = flags->flags;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

static struct agx_ptr
hk_usc_alloc_cb(void *data, size_t size, size_t alignment)
{
   return hk_pool_usc_alloc(data, size, alignment);
}

/* Bytes of the tile allocation of a layout, before its alignment */
static unsigned
hk_raw_tile_bytes(const struct agx_tilebuffer_layout *tib)
{
   unsigned bytes = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib->logical_format); ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE && !tib->spilled[rt])
         bytes += 4 * agx_raw_tile_words(tib->logical_format[rt]);
   }
   return bytes;
}

/* Attachment textures for the background program, rt at texture 2 * rt */
static uint64_t
hk_bg_sysvals(struct hk_cmd_buffer *cmd, bool layered)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   unsigned samples = MAX2(render->tilebuffer.nr_samples, 1);

   alignas(64) uint8_t sysvals[AGX_RUNTIME_SYSVAL_SIZE];
   hk_init_compute_sysvals(dev, sysvals);

   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view)
         continue;

      uint32_t info[5];
      memcpy(info, view->render_descriptors[layered].dimensions, 16);
      info[4] = samples;
      memcpy(
         sysvals + AGX_TEXTURE_INFO_OFFSET + (2 * rt) * AGX_TEXTURE_INFO_STRIDE,
         info, sizeof(info));
   }

   return hk_pool_upload(cmd, sysvals, sizeof(sysvals), 64);
}

/* Target rt: texture 2 * rt, image rt, clear colour in push words 4 + 8 * rt */
static struct hk_bg_eot
hk_launch_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                 struct agx_bg_eot_key *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   bool layered = key->tib.layered;
   VkResult error = VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct agx_ptr textures =
      hk_pool_alloc(cmd, 2 * HK_MAX_RTS * AGX_TEXTURE_TABLE_STRIDE, 64);
   if (!textures.cpu)
      goto fail;

   memset(textures.cpu, 0, 2 * HK_MAX_RTS * AGX_TEXTURE_TABLE_STRIDE);
   uint8_t push[2 * (4 + 8 * HK_MAX_RTS)] = {0};
   unsigned uniforms = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const struct hk_image_view *view =
         rt < render->color_att_count ? render->color_att[rt].iview : NULL;
      uint8_t *entry = textures.cpu;

      if (key->op[rt] == AGX_BG_LOAD) {
         if (!view || !view->render_descriptors[layered].texture_valid)
            goto unsupported;

         memcpy(entry + (2 * rt) * AGX_TEXTURE_TABLE_STRIDE,
                view->render_descriptors[layered].texture,
                AGX_TEXTURE_TABLE_STRIDE);
      } else if (key->op[rt] == AGX_BG_CLEAR) {
         memcpy(push + 2 * (4 + 8 * rt),
                &info->pColorAttachments[rt].clearValue, 16);
         uniforms = MAX2(uniforms, 4 + 8 * rt + 8);
      } else if (key->op[rt] == AGX_EOT_STORE) {
         void *store = entry + rt * AGX_TEXTURE_TABLE_STRIDE;

         /* Feedback-raw attachments store uncompressed, unlike the prepacked */
         if (view && (render->raw_color_mask & BITFIELD_BIT(rt))) {
            if (!hk_image_view_eot(view, layered, true, store))
               goto unsupported;
         } else if (view && view->render_descriptors[layered].store_valid) {
            memcpy(store, view->render_descriptors[layered].store,
                   AGX_TEXTURE_TABLE_STRIDE);
         } else {
            goto unsupported;
         }
      }
   }

   /* Attachmentless renders get their sample count at draw time */
   if (key->tib.nr_samples < 1)
      key->tib.nr_samples = 1;

   key->reserved_preamble = uniforms;
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, key);
   if (!shader)
      goto fail;

   const struct agx_compute_profile *p = &shader->info.compute_profile;
   struct agx_draw_stage stage = {
      .bo = shader->bo,
      .binary = agx_bo_map(shader->bo),
      .binary_size = shader->info.binary_size,
      .preamble_offset = p->preamble_offset,
      .preamble_size = p->preamble_size,
      .publication_count = p->publication_count,
      .publication_count_valid = true,
      .scratch_size = p->scratch_size,
      .atomic_frame_size = p->atomic_frame_size,
   };

   uint64_t resources[AGX_COMPUTE_MAX_RESOURCES];
   for (unsigned i = 0; i < p->resource_binding_count; ++i) {
      if (p->resource_kind[i] != AGX_COMPUTE_RESOURCE_UBO ||
          p->resource_binding[i] != AGX_GRAPHICS_SYSVAL_BINDING)
         goto unsupported;

      resources[i] = hk_bg_sysvals(cmd, layered);
      if (!resources[i])
         goto fail;
   }

   uint64_t table = 0;
   if (p->resource_binding_count) {
      table = hk_pool_upload(cmd, resources,
                             p->resource_binding_count * sizeof(uint64_t), 64);
      if (!table)
         goto fail;
   }

   uint64_t push_addr = hk_pool_upload(cmd, push, sizeof(push), 16);
   if (!push_addr)
      goto fail;

   unsigned samples = key->tib.nr_samples;
   uint32_t launch;
   if (!agx_prepare_stage_launch(
          &dev->dev, cmd, hk_usc_alloc_cb, &stage, true, table, textures.gpu,
          dev->fetch_sampler->va->addr, push_addr,
          agx_launch_tile_bytes(hk_raw_tile_bytes(&key->tib), samples), samples,
          &launch))
      goto fail;

   return (struct hk_bg_eot){.usc = launch};

unsupported:
   error = VK_ERROR_FEATURE_NOT_PRESENT;
fail:
   vk_command_buffer_set_error(&cmd->vk, error);
   return (struct hk_bg_eot){0};
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Construct the key */
   struct agx_bg_eot_key key = {.tib = render->tilebuffer};
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] && (partial_render || store))
         continue;

      if (store) {
         bool should_store = is_attachment_stored(att_info);

         /* Partial renders always need to flush to memory. */
         should_store |= partial_render;

         if (should_store)
            key.op[i] = AGX_EOT_STORE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         /* This is a very frustrating corner case. From the spec:
          *
          *     VK_ATTACHMENT_STORE_OP_NONE specifies the contents within the
          *     render area are not accessed by the store operation as long as
          *     no values are written to the attachment during the render pass.
          *
          * With VK_ATTACHMENT_STORE_OP_NONE, we suppress stores on the main
          * end-of-tile program. Unfortunately, that's not enough: we also need
          * to preserve the contents throughout partial renders. The easiest way
          * to do that is forcing a load in the background program, so that
          * partial stores for unused attachments will be no-op'd by writing
          * existing contents.
          *
          * Optimizing this would require nontrivial tracking. Fortunately,
          * this is all Android gunk and we don't have to care too much for
          * dekstop games. So do the simple thing.
          */
         bool no_store = (att_info->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
         bool no_store_wa = no_store && !load && !clear;
         if (no_store_wa) {
            perf_debug(cmd, "STORE_OP_NONE workaround");
         }

         load |= no_store_wa;

         /* Don't apply clears for spilled render targets when we clear the
          * render area explicitly after.
          */
         if (key.tib.spilled[i] && incomplete_render_area)
            continue;

         if (load)
            key.op[i] = AGX_BG_LOAD;
         else if (clear)
            key.op[i] = AGX_BG_CLEAR;
      }
   }

   if (agx_has_launch_programs(&dev->dev))
      return hk_launch_bg_eot(cmd, info, &key);

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         const struct agx_texture_packed *desc =
            key.tib.layered ? &iview->planes[0].layered_background
                            : &iview->planes[0].background;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(cmd, desc, sizeof(*desc), 8);
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         const struct agx_pbe_packed *desc = key.tib.layered
                                                ? &iview->planes[0].layered_eot
                                                : &iview->planes[0].eot;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(cmd, desc, sizeof(*desc), 8);
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&dev->dev, shader->ptr);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&dev->dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

/* A continued control stream loads what the previous one stored. Only the
 * last control stream of a render pass may drop stores, and only Apple9
 * honours that: its firmware preserves depth/stencil across partial renders
 * independently of these stores.
 */
static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render, bool last,
                    const bool *chain_load)
{
   struct agx_zls zls = {0};

   if (z) {
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && z->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(z->imageLayout,
                                       VK_IMAGE_ASPECT_DEPTH_BIT))
         zls.z_store = false;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && s->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(s->imageLayout,
                                       VK_IMAGE_ASPECT_STENCIL_BIT))
         zls.s_store = false;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   /* A merged or suspend/resume chain loads as its first render pass did. */
   if (chain_load && !partial_render) {
      zls.z_load = z && chain_load[0];
      zls.s_load = s && chain_load[1];
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

/* Raw depth/stencil planes keep their compressed layout and metadata, which
 * marks every tile uncompressed, but ZLS must neither decode nor re-encode
 * them while the render may sample them.
 */
static void
hk_zls_control_make_raw(struct agx_zls_control_packed *packed)
{
   struct agx_zls_control_packed compress;
   agx_pack(&compress, ZLS_CONTROL, cfg) {
      cfg.z_load_compress = cfg.z_store_compress = true;
      cfg.s_load_compress = cfg.s_store_compress = true;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(packed->opaque); ++i)
      packed->opaque[i] &= ~compress.opaque[i];
}

/* Render merging: continue the hardware
 * render of a render pass in the next render pass when nothing but attachment
 * barriers separates them and both use the same colour attachments. The render
 * covers the whole attachment whatever the render area, and tiles keep their
 * contents between the passes, which satisfies the second pass's load.
 * Depth/stencil views and render areas must also match. As with suspend/resume,
 * keep the first pass's ZLS loads and the final pass's stores.
 */
/* Whether the render pass's draws may read this attachment other than as an
 * attachment (render->feedback). A written attachment may be read so only
 * when feedback loop is enabled for it: the feedback loop layout, or
 * VkAttachmentFeedbackLoopInfoEXT under unified image layouts (as RADV
 * reads it). An attachment nothing has written yet, with LOAD_OP_NONE, may
 * also be sampled in GENERAL. GENERAL alone is not a feedback loop, so
 * renders in GENERAL keep barrier hoisting.
 */
static bool
hk_attachment_may_be_sampled(const VkRenderingAttachmentInfo *att)
{
   if (!att || !att->imageView)
      return false;
   if (att->imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
      return true;
   const VkAttachmentFeedbackLoopInfoEXT *loop =
      vk_find_struct_const(att->pNext, ATTACHMENT_FEEDBACK_LOOP_INFO_EXT);
   if (loop && loop->feedbackLoopEnable)
      return true;
   return att->imageLayout == VK_IMAGE_LAYOUT_GENERAL &&
          att->loadOp == VK_ATTACHMENT_LOAD_OP_NONE;
}

static bool
hk_render_merge_enabled(struct hk_cmd_buffer *cmd)
{
   return agx_has_launch_programs(&hk_cmd_buffer_device(cmd)->dev) &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta;
}

/* Key a render pass that may merge, from its attachment views, or return false.
 * The caller checks load and store operations.
 */
bool
hk_render_merge_key_init(struct hk_render_merge_key *key,
                         VkRenderingFlags flags, uint32_t layer_count,
                         uint32_t view_mask, uint32_t color_count,
                         struct hk_image_view *const *views,
                         struct hk_image_view *depth,
                         struct hk_image_view *stencil, uint8_t ds_read_only,
                         VkRect2D area)
{
   if ((flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT |
                 VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) ||
       color_count > HK_MAX_RTS)
      return false;
   /* A depth/stencil-only pass has the same fixed framebuffer and ZLS state
    * as one with colour attachments. The depth/stencil key, loads, stores
    * and barriers below already constrain its continuation. Attachmentless
    * passes still have no sample count until draw time, so keep them out.
    */
   if (!color_count && !depth && !stencil)
      return false;
   /* Merged multiview renders fail multiview.readback_implicit_clear 8_1_1_8
    * (cause not understood yet); keep multiview renders separate. */
   if (view_mask)
      return false;

   memset(key, 0, sizeof(*key));
   key->color_count = color_count;
   key->layer_count = layer_count;
   key->view_mask = view_mask;
   key->ds_read_only = ds_read_only;
   if (depth || stencil)
      key->ds_area = area;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      const struct hk_image_view *view = i == HK_MAX_RTS       ? depth
                                         : i == HK_MAX_RTS + 1 ? stencil
                                         : i < color_count     ? views[i]
                                                               : NULL;
      if (view) {
         key->image[i] = view->vk.image;
         key->level[i] = view->vk.base_mip_level;
         key->layer[i] = view->vk.base_array_layer;
         key->format[i] = view->vk.format;
      }
   }
   return true;
}

static bool
hk_render_merge_begin(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};
   struct hk_render_merge_key key;
   bool empty = cmd->empty_barriers.size != 0;
   if (empty && memcmp(&info->renderArea, &cmd->merge_empty_render->area,
                       sizeof(info->renderArea)))
      return false;
   const VkRenderingAttachmentInfo *ds[2] = {info->pDepthAttachment,
                                             info->pStencilAttachment};
   struct hk_image_view *ds_views[2] = {NULL};
   uint8_t ds_store_mask = 0, ds_read_only = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (!ds[i] || !ds[i]->imageView)
         continue;
      if (ds[i]->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
          ds[i]->resolveMode != VK_RESOLVE_MODE_NONE ||
          hk_attachment_may_be_sampled(ds[i]) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops &
           (i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT)))
         return false;
      if (empty && (ds[i]->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
                    (ds[i]->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout !=
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout !=
                        (i ? VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)) ||
                    hk_attachment_may_be_sampled(ds[i])))
         return false;
      ds_views[i] = hk_image_view_from_handle(ds[i]->imageView);
      if (vk_image_layout_is_read_only(
             ds[i]->imageLayout,
             i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
         ds_read_only |= BITFIELD_BIT(i);
      if (is_attachment_stored(ds[i]))
         ds_store_mask |= BITFIELD_BIT(i);
   }
   if (cmd->merge_ds_store_mask & ~ds_store_mask)
      return false;

   uint32_t store_mask = 0;
   for (unsigned i = 0; i < MIN2(info->colorAttachmentCount, HK_MAX_RTS); ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      views[i] = hk_image_view_from_handle(att->imageView);
      if (empty && att->imageView &&
          (att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
           (att->imageLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
            att->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) ||
           hk_attachment_may_be_sampled(att)))
         return false;
      /* Clears would need an in-pass clear over the render area. */
      if (att->imageView && (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
                             hk_attachment_may_be_sampled(att) ||
                             (cmd->vk.dynamic_graphics_state.feedback_loops &
                              VK_IMAGE_ASPECT_COLOR_BIT)))
         return false;
      if (att->imageView && is_attachment_stored(att))
         store_mask |= BITFIELD_BIT(i);
   }
   /* The merged render ends with this pass's store operations. An attachment
    * an earlier pass stored but this one does not would lose the earlier
    * pass's results outside this pass's render area. */
   if (cmd->merge_store_mask & ~store_mask) {
      return false;
   }

   return !cmd->merge_feedback && hk_render_merge_enabled(cmd) &&
          hk_render_merge_key_init(
             &key, info->flags & ~(empty ? VK_RENDERING_SUSPENDING_BIT : 0),
             info->layerCount, info->viewMask, info->colorAttachmentCount,
             views, ds_views[0], ds_views[1], ds_read_only, info->renderArea) &&
          !memcmp(&key, &cmd->merge_key, sizeof(key));
}

/* Resumed render merging: a suspended render pass keeps its
 * render open for the resuming one. Only state commands may come between
 * them, so the resuming pass continues the same render pass instance in one
 * hardware render: loads from the first pass, stores from the last. */
static bool
hk_render_suspend_keeps_open(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   return cs && !render->feedback && !render->raw_color_mask &&
          !render->raw_zs && (render->flags & VK_RENDERING_SUSPENDING_BIT) &&
          agx_has_launch_programs(&hk_cmd_buffer_device(cmd)->dev) &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta &&
          !cmd->sync_fallback && !cmd->current_cs.pre_gfx &&
          !cmd->current_cs.post_gfx && !cmd->tess_batch.draws;
}

/* At the end of a render pass, whether its render can stay open. */
static bool
hk_render_merge_end(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};

   if (!cs || render->feedback || render->raw_color_mask || render->raw_zs ||
       !hk_render_merge_enabled(cmd) || cmd->sync_fallback ||
       cmd->current_cs.pre_gfx || cmd->current_cs.post_gfx ||
       cmd->tess_batch.draws || cmd->state.gfx.occlusion.active_count)
      return false;

   for (unsigned i = 0; i < MIN2(render->color_att_count, HK_MAX_RTS); ++i) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         return false;
      views[i] = render->color_att[i].iview;
   }

   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE ||
       render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      return false;

   /* A render pass that resumed in its render ends it like any other. */
   return hk_render_merge_key_init(
      &cmd->merge_key,
      render->flags & ~(render->resumed ? VK_RENDERING_RESUMING_BIT : 0),
      render->layer_count, render->view_mask, render->color_att_count, views,
      render->depth_att.iview, render->stencil_att.iview, render->ds_read_only,
      render->area);
}

/* A merged empty chain can still end without any draws. Retain the first
 * pass's clear values so empty-render conversion does not lose them when a
 * later LOAD pass, including suspend/resume, supplies the final render state.
 */
static void
hk_render_inherit_empty_clears(struct hk_cmd_buffer *cmd)
{
   if (!cmd->merge_empty)
      return;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const struct hk_rendering_state *first = cmd->merge_empty_render;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      struct hk_attachment *att = i == HK_MAX_RTS       ? &render->depth_att
                                  : i == HK_MAX_RTS + 1 ? &render->stencil_att
                                                        : &render->color_att[i];
      const struct hk_attachment *old = i == HK_MAX_RTS ? &first->depth_att
                                        : i == HK_MAX_RTS + 1
                                           ? &first->stencil_att
                                           : &first->color_att[i];
      att->clear = old->clear;
      memcpy(att->clear_colour, old->clear_colour, sizeof(att->clear_colour));
   }
}

/* A compressed attachment must use raw stores as soon as feedback is possible.
 * GENERAL alone does not imply sampling, but the layout/enable bit or explicit
 * dynamic feedback state does. Keep raw stores sticky within the render.
 */
static uint8_t
hk_feedback_color_mask(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint8_t mask = 0;
   for (unsigned i = 0; i < info->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *a = &info->pColorAttachments[i];
      if (!a->imageView)
         continue;
      const struct hk_image_view *view =
         hk_image_view_from_handle(a->imageView);
      const struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);
      /* Decompression copies view depth/stencil planes as colour. */
      const struct ail_layout *layout =
         &image->planes[view->planes[0].image_plane].layout;
      if (!hk_feedback_can_compress(&dev->dev, image->vk.image_type,
                                    image->vk.format, image->vk.samples) ||
          !ail_is_level_logically_compressed(layout, view->vk.base_mip_level))
         continue;
      if (cmd->feedback_decompress || hk_attachment_may_be_sampled(a) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops &
           VK_IMAGE_ASPECT_COLOR_BIT))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/* The depth/stencil view whose compressed planes must be raw in this render,
 * if any. Sampling either aspect in a feedback loop makes the whole image raw:
 * its raw state is tracked per image, not per plane.
 */
static const struct hk_image_view *
hk_feedback_zs_view(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* Meta renders inherit the application's dynamic feedback state. */
   if (cmd->in_meta)
      return NULL;
   const VkRenderingAttachmentInfo *ds[2] = {info->pDepthAttachment,
                                             info->pStencilAttachment};
   const struct hk_image_view *view = NULL;
   bool sampled = cmd->vk.dynamic_graphics_state.feedback_loops &
                  (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   for (unsigned i = 0; i < ARRAY_SIZE(ds); ++i) {
      if (ds[i] && ds[i]->imageView) {
         view = hk_image_view_from_handle(ds[i]->imageView);
         sampled |= hk_attachment_may_be_sampled(ds[i]);
      }
   }
   if (!view || !sampled)
      return NULL;
   const struct hk_image *image =
      container_of(view->vk.image, struct hk_image, vk);
   if (!hk_feedback_can_compress(&dev->dev, image->vk.image_type,
                                 image->vk.format, image->vk.samples))
      return NULL;
   for (unsigned p = 0; p < image->plane_count; ++p) {
      if (ail_is_level_logically_compressed(&image->planes[p].layout,
                                            view->vk.base_mip_level))
         return view;
   }
   return NULL;
}

static bool
hk_feedback_decompress_layers(struct hk_cmd_buffer *cmd,
                              const VkRenderingInfo *info,
                              const struct hk_image_view *view)
{
   if (info->viewMask) {
      u_foreach_bit(v, info->viewMask) {
         if (!hk_meta_decompress_view(cmd, view, view->vk.base_array_layer + v))
            return false;
      }
   } else {
      for (unsigned v = 0; v < info->layerCount; v++) {
         if (!hk_meta_decompress_view(cmd, view, view->vk.base_array_layer + v))
            return false;
      }
   }
   return true;
}

static bool
hk_feedback_decompress(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                       uint8_t mask, const struct hk_image_view *zs)
{
   if (cmd->feedback_decompress)
      return true;
   u_foreach_bit(i, mask) {
      const struct hk_image_view *view =
         hk_image_view_from_handle(info->pColorAttachments[i].imageView);
      if (!hk_feedback_decompress_layers(cmd, info, view))
         return false;
   }
   return !zs || hk_feedback_decompress_layers(cmd, info, zs);
}

static void
hk_feedback_make_zs_raw(struct hk_rendering_state *render)
{
   render->raw_zs = true;
   hk_zls_control_make_raw(&render->cr.zls_control);
   hk_zls_control_make_raw(&render->cr.zls_control_partial);
   hk_zls_control_make_raw(&render->zls_last);
   hk_zls_control_make_raw(&render->zls_partial_last);
}

/* Splitting a live render keeps the partial EOT's stores and the partial BG's
 * loads. Rebuild only its final/partial colour stores with the new raw mask.
 * This also handles feedback enabled by binding a static-state pipeline.
 */
static bool
hk_feedback_prepare_draw(struct hk_cmd_buffer *cmd)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!cmd->rendering || cmd->in_meta ||
       !cmd->vk.dynamic_graphics_state.feedback_loops)
      return true;
   VkRenderingAttachmentInfo attachments[HK_MAX_RTS];
   for (unsigned i = 0; i < render->color_att_count; i++) {
      attachments[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = render->store_mask & BITFIELD_BIT(i)
                       ? VK_ATTACHMENT_STORE_OP_STORE
                       : VK_ATTACHMENT_STORE_OP_DONT_CARE,
      };
   }
   VkRenderingAttachmentInfo zs[2];
   for (unsigned i = 0; i < 2; i++) {
      zs[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(i ? render->stencil_att.iview
                                                : render->depth_att.iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      };
   }
   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = attachments,
      .pDepthAttachment = &zs[0],
      .pStencilAttachment = &zs[1],
   };
   uint8_t mask = hk_feedback_color_mask(cmd, &info) & ~render->raw_color_mask;
   const struct hk_image_view *zs_view =
      render->raw_zs ? NULL : hk_feedback_zs_view(cmd, &info);
   if (!mask && !zs_view)
      return true;
   hk_cmd_buffer_end_graphics(cmd);
   struct hk_rendering_state saved = *render;
   if (!hk_feedback_decompress(cmd, &info, mask, zs_view))
      return false;
   *render = saved;
   cmd->rendering = true;
   render->feedback = true;
   render->raw_color_mask |= mask;
   if (zs_view)
      hk_feedback_make_zs_raw(render);
   render->cr.bg.main = render->cr.bg.partial;
   for (unsigned i = 0; i < render->color_att_count; i++)
      render->color_att[i].clear = false;
   render->cr.eot.main = hk_build_bg_eot(cmd, &info, true, false, true);
   render->cr.eot.partial = hk_build_bg_eot(cmd, &info, true, true, true);
   cmd->sync_render_continuation = false;
   cmd->state.gfx.render_gen++;
   hk_cmd_buffer_dirty_all(cmd);
   return !vk_command_buffer_has_error(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetAttachmentFeedbackLoopEnableEXT(VkCommandBuffer commandBuffer,
                                         VkImageAspectFlags aspectMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!BITSET_TEST(dyn->set,
                    MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) ||
       dyn->feedback_loops != aspectMask) {
      dyn->feedback_loops = aspectMask;
      BITSET_SET(dyn->set, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
   }
   if (cmd->rendering && aspectMask)
      cmd->state.gfx.render.feedback = true;
   if (agx_has_launch_programs(&hk_cmd_buffer_device(cmd)->dev))
      hk_feedback_prepare_draw(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   uint8_t raw_mask = hk_feedback_color_mask(cmd, pRenderingInfo);
   const struct hk_image_view *raw_zs =
      hk_feedback_zs_view(cmd, pRenderingInfo);
   if (raw_mask || raw_zs) {
      hk_cmd_render_merge_flush(cmd);
      if (!hk_feedback_decompress(cmd, pRenderingInfo, raw_mask, raw_zs))
         return;
   }
   if (!cmd->feedback_decompress && (raw_mask || raw_zs))
      cmd->sync_render_continuation = false;
   bool merge = false;
   /* The resuming render pass of a suspended one continues its render. */
   bool resume = false;
   if (cmd->merge_cs && cmd->merge_suspended) {
      resume =
         (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) && !cmd->in_meta;
      if (!resume)
         hk_cmd_render_merge_flush(cmd);
   }
   if (cmd->merge_cs && !resume) {
      merge = hk_render_merge_begin(cmd, pRenderingInfo);
      if (merge)
         hk_cmd_empty_barriers_flush(cmd, true);
      if (!merge) {
         hk_cmd_render_merge_flush(cmd);
      }
   }

   /* A continued render needs no dependency on itself. */
   cmd->sync_render_continuation =
      (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) != 0 && !resume &&
      !raw_mask && !raw_zs;
   cmd->sync_render_stages = cmd->sync_scope ?: HK_SYNC_GRAPHICS;
   cmd->state.gfx.render_gen++;
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));
   cmd->rendering = true;
   render->raw_color_mask = raw_mask;
   render->feedback = cmd->vk.dynamic_graphics_state.feedback_loops != 0;

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   for (uint32_t i = 0; i < MIN2(pRenderingInfo->colorAttachmentCount, 32);
        i++) {
      if (pRenderingInfo->pColorAttachments[i].imageView &&
          is_attachment_stored(&pRenderingInfo->pColorAttachments[i]))
         render->store_mask |= BITFIELD_BIT(i);
   }
   const VkRenderingAttachmentInfo *ds[2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (unsigned i = 0; i < 2; ++i) {
      if (ds[i] && ds[i]->imageView) {
         if (is_attachment_stored(ds[i]))
            render->ds_store_mask |= BITFIELD_BIT(i);
         if (vk_image_layout_is_read_only(
                ds[i]->imageLayout,
                i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
            render->ds_read_only |= BITFIELD_BIT(i);
      }
   }
   const VkRenderingAttachmentInfo *feedback_att[HK_MAX_RTS + 2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (uint32_t i = 0;
        i < MIN2(pRenderingInfo->colorAttachmentCount, HK_MAX_RTS); i++)
      feedback_att[2 + i] = &pRenderingInfo->pColorAttachments[i];
   for (unsigned i = 0; i < ARRAY_SIZE(feedback_att); ++i) {
      if (hk_attachment_may_be_sampled(feedback_att[i]))
         render->feedback = true;
   }
   for (unsigned i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      if (!(raw_mask & BITFIELD_BIT(i)))
         hk_feedback_invalidate_view(
            cmd, hk_image_view_from_handle(
                    pRenderingInfo->pColorAttachments[i].imageView));
   }
   /* Compressed ZLS stores re-encode the planes' tiles. */
   for (unsigned i = 0; i < 2 && !raw_zs; i++) {
      if (ds[i] && ds[i]->imageView)
         hk_feedback_invalidate_view(
            cmd, hk_image_view_from_handle(ds[i]->imageView));
   }
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
      hk_attachment_use_implicit_sync(cmd, &render->color_att[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);
   hk_attachment_use_implicit_sync(cmd, &render->depth_att);
   hk_attachment_use_implicit_sync(cmd, &render->stencil_att);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   u_foreach_bit(i, render->store_mask) {
      struct hk_image_view *view = render->color_att[i].iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);
      if (image->planes[view->planes[0].image_plane].layout.tiling ==
          AIL_TILING_LINEAR)
         render->linear_store = true;
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   if (!render->cr.zls_width) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = hk_format_to_pipe_format(render->color_att[i].vk_format);
   }

   /* For now, we force layered=true since it makes compatibility problems way
    * easier.
    */
   render->tilebuffer = (agx_has_raw_tiles(&hk_cmd_buffer_device(cmd)->dev)
                            ? agx_build_raw_tilebuffer_layout
                            : agx_build_tilebuffer_layout)(
      formats, render->color_att_count, render->tilebuffer.nr_samples, true);

   if (cmd->feedback_decompress) {
      /* A self-blit must retire each whole compression block before another
       * tile reads it. The metadata covers 16x16 texels for every colour bpp. */
      struct drm_asahi_cmd_render raw = {0};
      agx_tilebuffer_set_drm_cmd(&raw, &render->tilebuffer);
      assert(raw.utile_width_px % 16 == 0);
      assert(raw.utile_height_px % 16 == 0);
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area);
   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true, incomplete_render_area);

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false, incomplete_render_area);
   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true, incomplete_render_area);

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, false, NULL);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true, false,
                       NULL);

   const bool drop_stores =
      agx_has_launch_programs(&hk_cmd_buffer_device(cmd)->dev) &&
      !(render->flags & VK_RENDERING_SUSPENDING_BIT);
   /* The first render pass of a chain decides its loads. */
   if (!resume && !merge) {
      render->chain_load[0] = attach_z && attach_z->imageView &&
                              (attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ||
                               incomplete_render_area);
      render->chain_load[1] = attach_s && attach_s->imageView &&
                              (attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ||
                               incomplete_render_area);
   } else {
      memcpy(render->chain_load, cmd->merge_chain_load,
             sizeof(render->chain_load));
   }
   hk_pack_zls_control(&render->zls_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);
   hk_pack_zls_control(&render->zls_partial_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, true, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);
   if (raw_zs)
      hk_feedback_make_zs_raw(render);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;

   if (resume) {
      /* Same render pass instance: keep the open render's background and
       * load configuration; this pass's store operations apply at its end. */
      perf_debug(cmd, "Resuming the suspended render pass in its render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      render->feedback |= cmd->merge_feedback;
      render->resumed = true;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      cmd->merge_suspended = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   if (merge) {
      /* Continue the open render. Its background program, made for the
       * first pass, loads or clears every tile it processes. */
      perf_debug(cmd, "Merging render pass into the previous render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      /* The render's earlier passes may have read their attachments too. */
      render->feedback |= cmd->merge_feedback;
      render->merge_reach = cmd->merge_reach;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;
   cs->linear_store = render->linear_store;
   render->first_cs = cs;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
   /* Don't reorder compute across render passes.
    *
    * TODO: Check if this is necessary if the proper PipelineBarriers are
    * handled... there may be CTS bugs...
    */
   hk_cmd_buffer_end_compute(cmd);

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct libagx_decompress_images imgs = {
                  .compressed = view->planes[plane].emrt_texture,
                  .uncompressed = view->planes[plane].emrt_pbe,
               };

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                 layout, layer, level, base,
                                 hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear |= is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   if (clear_count > 0 && incomplete_render_area) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = resolved_clear;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                       const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs) {
      cs->cr.eot.main = (render->flags & VK_RENDERING_SUSPENDING_BIT)
                           ? render->cr.eot.partial
                           : render->cr.eot.main;
      /* zls_last already preserves suspended stores. Its first-stream
       * loads must retain CLEAR; the partial control would force a load. */
      cs->cr.zls_control =
         cs == render->first_cs ? render->zls_last : render->zls_partial_last;
   }

   perf_debug(cmd, "End rendering");
   const bool suspend = hk_render_suspend_keeps_open(cmd, cs);
   if (suspend || hk_render_merge_end(cmd, cs)) {
      cmd->merge_suspended = suspend;
      memcpy(cmd->merge_chain_load, render->chain_load,
             sizeof(cmd->merge_chain_load));
      /* Keep the render open for a following render pass to continue. */
      /* A continuing pass stores at least what the passes before it did, so
       * its mask covers the whole open render. */
      cmd->merge_store_mask = render->store_mask;
      cmd->merge_ds_store_mask = render->ds_store_mask;
      cmd->merge_feedback = render->feedback;
      cmd->merge_reach = hk_merge_graphics_stages() | render->merge_reach;
      cmd->merge_cs = cs;
      cmd->merge_first_cs = cs == render->first_cs;
      /* A render without draws yet: keep its state for the empty-render
       * optimization in case nothing continues it. */
      cmd->merge_empty = !cs->stats.cmds;
      if (cmd->merge_empty) {
         if (!cmd->merge_empty_render)
            cmd->merge_empty_render = malloc(sizeof(*render));
         if (cmd->merge_empty_render) {
            memcpy(cmd->merge_empty_render, render, sizeof(*render));
         } else {
            cmd->merge_cs = NULL;
            cmd->merge_empty = false;
            hk_cmd_buffer_end_graphics(cmd);
         }
      }
      /* Barriers before the next pass go ahead of the stream's first event. */
      cmd->merge_event =
         *util_dynarray_element(&cmd->sync_first_event, uint32_t, cs->sync_id);
      assert(cmd->merge_event != HK_SYNC_NONE &&
             "a stream with draws has a sync event");
   } else {
      hk_cmd_buffer_end_graphics(cmd);
   }

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   VkRenderingAttachmentFlagsInfoKHR vk_color_flags[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         need_resolve = true;

      vk_color_flags[i] = (VkRenderingAttachmentFlagsInfoKHR){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
         .flags = render->color_att[i].flags,
      };
      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .pNext = &vk_color_flags[i],
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentFlagsInfoKHR vk_depth_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->depth_att.flags,
   };
   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_depth_flags,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentFlagsInfoKHR vk_stencil_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->stencil_att.flags,
   };
   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_stencil_flags,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));
   cmd->rendering = false;

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}

/* The geometry heap's map in the read-only data. */
static struct poly_heap *
hk_heap_map(struct hk_device *dev)
{
   off_t off = dev->rodata.heap - dev->rodata.bo->va->addr;
   return agx_bo_map(dev->rodata.bo) + off;
}

static_assert(HK_HEAP_VA_MAX_B / HK_HEAP_MIN_B == 1u << (HK_HEAP_MAX_BOS - 1),
              "one backing BO per doubling");

/* Bind bo at offset_B in the geometry heap's VA range. */
static int
hk_heap_bind(struct hk_device *dev, struct agx_bo *bo, uint64_t offset_B)
{
   return agx_bo_bind(&dev->dev, bo, dev->heap_va->addr + offset_B, bo->size, 0,
                      DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
}

/* Back the geometry heap with one new BO of size_B bytes, replacing its
 * backing. The heap must be idle. */
static bool
hk_heap_back(struct hk_device *dev, uint64_t size_B)
{
   struct agx_bo *bo = agx_bo_create(&dev->dev, size_B, 0, 0, "Geometry heap");
   if (!bo)
      return false;

   if (dev->heap_bos) {
      agx_bo_bind(&dev->dev, NULL, dev->heap_va->addr, dev->heap_backed_B, 0,
                  DRM_ASAHI_BIND_UNBIND);
   }

   if (hk_heap_bind(dev, bo, 0)) {
      agx_bo_unreference(&dev->dev, bo);
      /* Keep the old backing mapped, if any. */
      uint64_t offset_B = 0;
      for (unsigned i = 0; i < dev->heap_bos; ++i) {
         hk_heap_bind(dev, dev->heap[i], offset_B);
         offset_B += dev->heap[i]->size;
      }
      return false;
   }

   for (unsigned i = 0; i < dev->heap_bos; ++i)
      agx_bo_unreference(&dev->dev, dev->heap[i]);
   dev->heap[0] = bo;
   dev->heap_bos = 1;
   dev->heap_backed_B = size_B;
   hk_heap_map(dev)->size = size_B - POLY_HEAP_SINK_B;
   return true;
}

/* Grow the geometry heap's backing to size_B bytes by binding a new BO after
 * it. Work still using the heap keeps its memory: the backing it has stays
 * bound, and the larger size, published once the new BO is bound, only lets
 * it allocate from bound memory. So growth need not wait for that work.
 */
static bool
hk_heap_extend(struct hk_device *dev, uint64_t size_B)
{
   assert(size_B > dev->heap_backed_B);
   if (dev->heap_bos == HK_HEAP_MAX_BOS)
      return false;

   struct agx_bo *bo = agx_bo_create(&dev->dev, size_B - dev->heap_backed_B, 0,
                                     0, "Geometry heap");
   if (!bo)
      return false;

   if (hk_heap_bind(dev, bo, dev->heap_backed_B)) {
      agx_bo_unreference(&dev->dev, bo);
      return false;
   }

   dev->heap[dev->heap_bos++] = bo;
   dev->heap_backed_B = size_B;
   hk_heap_map(dev)->size = size_B - POLY_HEAP_SINK_B;
   return true;
}

static void
hk_init_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   perf_debug(cmd, "Allocating heap");

   /* Unrolled and geometry-generated index buffers are allocated from this
    * heap. Apple9 draws address index buffers through the USC aperture.
    * Recorded draws address the heap by its VA range, so the range is fixed
    * and its backing grows instead, see hk_heap_prepare_submit. The USC
    * aperture is shared, so settle for a smaller range when it is crowded.
    */
   enum agx_va_flags flags = agx_has_usc_indices(&dev->dev) ? AGX_VA_USC : 0;
   for (uint64_t va_B = HK_HEAP_VA_MAX_B;
        !dev->heap_va && va_B >= HK_HEAP_GEOMETRY_MIN_B; va_B /= 2) {
      dev->heap_va = agx_va_alloc(&dev->dev, va_B, 65536, flags, 0);
   }
   if (!dev->heap_va)
      return;

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   *hk_heap_map(dev) = (struct poly_heap){
      .base = dev->heap_va->addr,
   };

   if (!hk_heap_back(dev, MIN2(cmd->heap_min_B, dev->heap_va->size_B))) {
      agx_va_free(&dev->dev, dev->heap_va, false);
      dev->heap_va = NULL;
   }
}

/* Whether every queue has finished its submissions, without waiting. */
static bool
hk_device_idle(struct hk_device *dev)
{
   vk_foreach_queue(iter, &dev->vk) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      uint64_t completed = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &completed, 1) ||
          completed < queue->drm.timeline_value)
         return false;
   }
   return true;
}

/* Return a spike's memory. high_water only grows, so the heap's need is kept
 * per period of HK_HEAP_DECAY_NS instead: when one has passed and the device is
 * already idle (never wait for it), start a new period from a cleared
 * high_water, and shrink the backing if it is more than twice what this
 * submission and the period needed (at least HK_HEAP_MIN_B).
 */
static void
hk_heap_decay(struct hk_device *dev, uint64_t min_B, uint64_t needed_B)
{
   dev->heap_period_need_B = MAX2(dev->heap_period_need_B, needed_B);

   uint64_t now = os_time_get_nano();
   if (now - dev->heap_period_start_ns < HK_HEAP_DECAY_NS ||
       !hk_device_idle(dev))
      return;

   uint64_t target_B = MAX3(
      min_B, HK_HEAP_MIN_B,
      util_next_power_of_two64(dev->heap_period_need_B + POLY_HEAP_SINK_B));
   if (dev->heap_backed_B > 2 * target_B) {
      perf_debug_dev(&dev->dev,
                     "Shrinking the geometry heap to %" PRIu64 " MiB",
                     target_B >> 20);
      if (hk_heap_back(dev, target_B))
         dev->heap_reported_B = 0;
   }

   /* The heap is idle, so nothing races these writes. */
   struct poly_heap *heap = hk_heap_map(dev);
   heap->high_water = 0;
   heap->bottom = 0;
   dev->heap_period_need_B = 0;
   dev->heap_period_start_ns = now;
}

/* Before a submission whose command buffers use the geometry heap, with the
 * heap lock held: back at least min_B bytes, and what earlier submissions
 * needed. Allocations that did not fit dropped their draws, and the heap grows
 * past them at once, to the next power of two.
 */
void
hk_heap_prepare_submit(struct hk_device *dev, uint64_t min_B)
{
   if (!dev->heap_bos)
      return;

   const struct poly_heap *heap = hk_heap_map(dev);
   uint64_t needed_B = poly_heap_needed_B(heap);
   if (needed_B > heap->size && needed_B > dev->heap_reported_B) {
      dev->heap_reported_B = needed_B;
      dev->heap_overflows++;
      mesa_logw_once("hk: geometry heap exhausted (needed %" PRIu64
                     " MiB, had %u MiB): geometry or tessellation output was "
                     "dropped",
                     needed_B >> 20, heap->size >> 20);
   }

   uint64_t target_B =
      MAX2(min_B, needed_B > heap->size
                     ? util_next_power_of_two64(needed_B + POLY_HEAP_SINK_B)
                     : 0);
   target_B = MIN2(target_B, dev->heap_va->size_B);
   if (target_B <= dev->heap_backed_B) {
      hk_heap_decay(dev, min_B, needed_B);
      return;
   }

   perf_debug_dev(&dev->dev, "Growing the geometry heap to %" PRIu64 " MiB",
                  target_B >> 20);
   hk_heap_extend(dev, target_B);
}

uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Geometry and tessellation need the heap's historical size; the other
    * users (unrolled fans, adjacency, robust index copies) start small, see
    * hk_heap_prepare_submit. */
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint64_t min_B =
      gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TESS_EVAL]
         ? HK_HEAP_GEOMETRY_MIN_B
         : HK_HEAP_MIN_B;
   cmd->heap_min_B = MAX2(cmd->heap_min_B, min_B);

   util_call_once_data(&dev->heap_init_once, hk_init_heap, cmd);
   if (!dev->heap_bos) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   /* Acquire the heap before this render's first allocation. The pre-graphics
    * stream waits for earlier work, including the previous heap consumer.
    * Resetting here also covers command-buffer resubmission without a separate
    * post-graphics Work just to release the allocation cursor.
    */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Resetting heap");
      uint64_t addr = dev->rodata.heap;

      cmd->heap_renders++;
      cmd->uses_heap = true;

      libagx_heap_reset(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, addr);
   }

   return dev->rodata.heap;
}

/* GPU-generated index/coordinate offsets use the single geometry heap, by its
 * fixed VA range: its backing may grow before a later submission. */
static struct agx_va *
hk_geometry_heap_va(struct hk_cmd_buffer *cmd)
{
   /* Recording can continue after OOM, but the command buffer cannot be
    * submitted. Keep address construction safe on those error paths. */
   if (!hk_heap(cmd))
      return NULL;
   return hk_cmd_buffer_device(cmd)->heap_va;
}

static struct agx_draw
hk_heap_indexed_indirect(struct hk_cmd_buffer *cmd, uint64_t ptr,
                         enum agx_index_size index_size, bool restart)
{
   struct agx_va *va = hk_geometry_heap_va(cmd);
   return agx_draw_indexed_indirect(ptr, va ? va->addr : 0, va ? va->size_B : 0,
                                    index_size, restart);
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params_for_vs(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                               const struct hk_shader *vs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_buffer_range_el = range_el;

      /* Apple9 software vertex shaders read the index size from here rather
       * than from a specialized prolog.
       */
      params.index_size_B = index_size_B;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   return hk_upload_vertex_params_for_vs(cmd, draw,
                                         hk_bound_sw_vs(&cmd->state.gfx));
}

static enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

static enum mesa_prim
hk_rast_prim(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (gs != NULL) {
      return gs->variants[HK_GS_VARIANT_RAST].info.gs.mode;
   } else {
      switch (dyn->ia.primitive_topology) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_LINES;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_TRIANGLES;
      default:
         return hk_gs_in_prim(cmd);
      }
   }
}

/* Apple9 draws address index buffers through the 32-bit USC aperture. */
static struct agx_ptr
hk_geom_index_alloc(struct hk_cmd_buffer *cmd, uint32_t size)
{
   if (agx_has_usc_indices(&hk_cmd_buffer_device(cmd)->dev))
      return hk_pool_usc_alloc(cmd, size, 4);
   else
      return hk_pool_alloc(cmd, size, 4);
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs ? fs->info.fs.interp.flat : 0;
   /* Tessellation precedes the geometry shader and has set up this draw's
    * parameters; the geometry shader recovers patch IDs from them. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL])
      params.tess_params = gfx->descriptors.root.draw.tess_params;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }
      params.pg_query_counter[i] = gfx->pg_query[i];

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      /* On failure the command buffer is in error and never submitted. */
      if (T.cpu)
         memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         struct agx_va *va = hk_geometry_heap_va(cmd);
         cmd->geom_index_buffer = va ? va->addr : 0;
         cmd->geom_index_count = va ? va->size_B : 0;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_geom_index_alloc(cmd, cmd->geom_index_count * 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      struct agx_ptr topology = hk_geom_index_alloc(cmd, gsi->max_indices * 4);
      if (topology.cpu)
         memcpy(topology.cpu, gsi->topology,
                MIN2(gsi->max_indices * 4, sizeof(gsi->topology)));
      cmd->geom_index_buffer = topology.gpu;
   }

   desc->root_dirty = true;
   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static void
hk_upload_tess_params_for_tcs(struct hk_cmd_buffer *cmd,
                              struct poly_tess_params *out,
                              struct agx_draw draw, struct hk_shader *tcs)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params args = {
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!args.points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      args.ccw = gfx->tess.info.ccw;
      args.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   /* heap is allocated by hk_heap */
   struct agx_va *va = hk_geometry_heap_va(cmd);
   args.patch_coord_buffer = va ? va->addr : 0;

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / args.input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * args.tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      args.tcs_buffer = blob.gpu + tcs_out_offs;
      args.patches_per_instance = in_patches;
      args.coord_allocs = blob.gpu + patch_coord_offs;
      args.nr_patches = unrolled_patches;
      args.out_draws = blob.gpu + draw_offs;
      args.counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      args.out_draws = hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = args.out_draws;
   memcpy(out, &args, sizeof(args));
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   hk_upload_tess_params_for_tcs(
      cmd, out, draw,
      hk_only_variant(cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL]));
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE,
      agx_nir_options_for_isa(agx_gather_device_key(&dev->dev).shader_isa),
      NULL);
   builder(&b, agx_device_target(&dev->dev), key->key);

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   hk_preprocess_nir_internal(dev->vk.physical, b.shader);

   struct hk_api_shader *s;
   if (hk_compile_shader(dev, &info, NULL, NULL, NULL, &s) != VK_SUCCESS)
      return NULL;

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size)
{
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s = hk_build_meta_shader_locked(dev, key, builder);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = gfx->index.restart,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, draw.index_size,
                                   false /* restart */);
}

/* Whether a vertex-shader pipeline exposes its provoking vertex: through a
 * flat fragment input, or a layer or viewport index written by the vertex
 * shader. Primitive ID is not taken from a vertex, and a
 * fragment shader's layer reads zero unless the vertex shader writes it. A
 * vertex-written primitive shading rate would observe it too, once supported. */
static bool
hk_provoking_observed(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   if (vs && hk_any_variant(vs)->b.info.writes_layer_viewport)
      return true;
   if (!fs || fs == dev->null_fs)
      return false;

   const struct hk_shader *shader = hk_only_variant(fs);
   const struct agx_varyings_fs *varyings = &shader->b.info.varyings.fs;
   for (unsigned i = 0; i < varyings->nr_bindings; ++i) {
      if (!varyings->bindings[i].smooth)
         return true;
   }

   return false;
}

/* Native Apple9 fans flat shade consistently only from their first vertex.
 * Draw fans with an observable last provoking vertex as an unrolled triangle
 * list. */
static struct agx_draw
hk_unroll_fan(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   perf_debug(cmd, "Unrolling a last-provoking triangle fan");

   uint64_t in_draw;
   if (draw.indexed) {
      draw = hk_draw_as_indexed_indirect(cmd, draw);
      in_draw = draw.b.ptr;
   } else if (agx_is_indirect(draw.b)) {
      in_draw = draw.b.ptr;
   } else {
      VkDrawIndirectCommand desc = {
         .vertexCount = draw.b.count[0],
         .instanceCount = draw.b.count[1],
         .firstVertex = draw.start,
         .firstInstance = draw.start_instance,
      };
      in_draw = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct libagx_unroll_geometry_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.indexed ? draw.index_buffer : 0,
      .in_draw = in_draw,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu,
      .restart_index = draw.restart ? gfx->index.restart : UINT32_MAX,
      .index_buffer_size_el = draw.indexed ? agx_draw_index_range_el(draw) : 0,
      .in_index_size_B =
         draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
      .flatshade_first = false,
   };

   libagx_unroll_geometry_struct(cmd, agx_1d(1024),
                                 AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                 poly_compact_prim(MESA_PRIM_TRIANGLE_FAN));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, AGX_INDEX_SIZE_U32,
                                   false /* restart */);
}

/* Run a software pre-rasterization stage, linked as s, as compute */
static void
hk_launch_prerast(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                  struct hk_linked_shader *linked, struct agx_grid grid)
{
   if (agx_is_shader_empty(&s->b))
      return;
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_ALL);
   if (!cs)
      return;

   uint64_t push = hk_launch_push_words(cmd, s, linked, cmd->state.gfx.root);
   if (!push)
      return;

   hk_reserve_scratch(cmd, cs, s);
   hk_launch_dispatch(cmd, cs, &linked->b.bo->shader_body,
                      &linked->b.program.profile, grid, cmd->state.gfx.root,
                      push, s);
}

/* Dispatch a software pre-rasterization stage, linked as linked */
static void
hk_dispatch_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                    struct hk_shader *s, struct hk_linked_shader *linked,
                    struct agx_grid grid, struct agx_workgroup wg)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (agx_has_launch_programs(&dev->dev)) {
      hk_launch_prerast(cmd, s, linked, grid);
      return;
   }

   if (agx_is_shader_empty(&s->b))
      return;

   uint32_t usc = hk_upload_usc_words(cmd, s, linked);
   hk_reserve_scratch(cmd, cs, s);
   hk_dispatch_with_usc(dev, cs, &s->b.info, usc, grid, wg);
}

/* Launch programs have no dynamic state flush ahead of the stages, so the
 * parameters the geometry stages read are uploaded here.
 */
static void
hk_launch_upload_gs_root(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
   desc->root.draw.geometry_params = hk_upload_geometry_params(cmd, draw);

   desc->root.draw.api_gs = !gs->is_passthrough;
   desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
   desc->root.draw.provoking =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2
                                                                           : 0;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

static struct agx_draw
hk_launch_gs_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct agx_grid grid_vs, grid_gs;
   bool launch = agx_has_launch_programs(&dev->dev);

   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   if (launch)
      hk_launch_upload_gs_root(cmd, draw);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   unsigned count_words = count->info.gs.count_words;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   if (false /* TODO */)
      perf_debug(cmd, "Transform feedbck");
   else if (count_words)
      perf_debug(cmd, "Geometry shader with counts");
   else
      perf_debug(cmd, "Geometry shader without counts");

   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .p = desc->root.draw.geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;

   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   bool xfb_or_queries =
      main->info.gs.xfb || (root->draw.pipeline_stats_flags & gs_queries);

   /* Launch programs count primitive queries in the pre-GS program too */
   if (launch) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
         xfb_or_queries |= gfx->pg_query[i] || gfx->xfb_query[i];
   }

   /* Without transform feedback or queries, a small direct draw is just the
    * VS and GS passes over its own allocations: record them into the first
    * two batch lanes (as for tessellation) so consecutive draws overlap.
    */
   const unsigned max_batch_prims = 2048;
   bool batch = launch && !agx_is_indirect(draw.b) && !xfb_or_queries &&
                grid_gs.count[0] * grid_gs.count[1] <= max_batch_prims;
   if (launch && !batch)
      hk_tess_batch_flush(cmd);

   /* Launch the vertex shader first */
   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[0] : NULL;
   hk_dispatch_prerast(cmd, cs, vs,
                       vs->info.stage == MESA_SHADER_VERTEX
                          ? gfx->linked[MESA_SHADER_VERTEX]
                          : vs->only_linked,
                       grid_vs, wg);
   cmd->tess_batch.capture = NULL;

   if (xfb_or_queries) {
      /* If we need counts, launch the count shader and prefix sum the results. */
      if (count_words) {
         perf_debug(dev, "Geometry shader count");
         hk_dispatch_prerast(cmd, cs, count, count->only_linked, grid_gs, wg);
      }

      if (count->info.gs.prefix_sum) {
         perf_debug(dev, "Geometry shader transform feedback prefix sum");
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      /* Transform feedback / query program */
      perf_debug(dev, "Transform feedback / geometry query");
      hk_dispatch_prerast(cmd, cs, pre_gs, pre_gs->only_linked, agx_1d(1),
                                 agx_workgroup(1, 1, 1));
   }

   /* Pre-rast geometry shader */
   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[1] : NULL;
   hk_dispatch_prerast(cmd, cs, main, main->only_linked, grid_gs, wg);
   cmd->tess_batch.capture = NULL;
   if (batch)
      cmd->tess_batch.draws++;

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else {
      if (agx_is_indirect(draw.b)) {
         return agx_draw_indirect(geometry_params +
                                  offsetof(struct poly_geometry_params, draw));
      } else {
         return (struct agx_draw){
            .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
         };
      }
   }
}

static_assert(HK_MAX_VIEWPORTS == AGX_PPP_MAX_VIEWPORTS,
              "Vulkan viewport limit matches the PPP viewport block");

static struct agx_ptr
hk_context_alloc_cb(void *data, size_t size, size_t alignment)
{
   return hk_pool_context_alloc(data, size, alignment);
}

static enum agx_primitive
hk_rast_primitive(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   default:
      return AGX_PRIMITIVE_TRIANGLES;
   }
}

/* Sample positions for a draw: the application's when enabled for this
 * sample count, the standard ones otherwise. */
static uint32_t
hk_draw_sample_positions(const struct vk_dynamic_graphics_state *dyn,
                         unsigned samples)
{
   return dyn->ms.sample_locations_enable && dyn->ms.sample_locations &&
                dyn->ms.sample_locations->per_pixel == samples
             ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
             : hk_default_sample_positions(samples);
}

/* Split before recording any of a draw's pre-rasterization work. Ending a
 * graphics stream also closes its pre/post compute streams and heap lifetime.
 * Reserve for every emulated multiview draw so these stay together.
 *
 * reads_buffers: the draw's tiling reads buffer memory itself (indices or
 * indirect arguments).
 */
static bool
hk_launch_prepare_draw(struct hk_cmd_buffer *cmd, bool reads_buffers)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      return false;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* A draw that selects a new occlusion segment in a stream without free
    * segments ends that stream (hk_occlusion_prepare_draw). Do it here,
    * before this function commits render state (depth bias representation,
    * sample positions, write and read classes) to the stream, so that state
    * lands on the stream the draw records into. Later splits here start
    * streams with free segments.
    */
   const struct hk_graphics_state *ogfx = &cmd->state.gfx;
   if (ogfx->occlusion.segment_dirty && ogfx->occlusion.active_count &&
       cs->occlusion_segments == HK_OCCLUSION_SEGMENTS) {
      perf_debug(cmd, "Splitting control stream for occlusion segments");
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
   }

   /* Sample positions are render state (the per-render multisample control),
    * so the first application draw of a render commits them and a draw with
    * other positions starts a new render. That lets them vary within a
    * subpass (variableSampleLocations, reported to Zink only). Meta draws
    * neither commit nor change them.
    */
   if (!cmd->in_meta) {
      unsigned samples = MAX2(
         render->tilebuffer.nr_samples ?: dyn->ms.rasterization_samples, 1);
      uint32_t positions = hk_draw_sample_positions(dyn, samples);
      if (cs->has_sample_locations && cs->ppp_multisamplectl != positions) {
         perf_debug(cmd, "Splitting Apple9 render: sample positions changed");
         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false);
         if (!cs)
            return false;
      }
      cs->has_sample_locations = true;
      cs->ppp_multisamplectl = positions;
   }

   unsigned views = util_bitcount(render->view_mask ?: 1);
   unsigned viewports =
      dyn->rs.rasterizer_discard_enable ? 1 : MAX2(dyn->vp.viewport_count, 1);
   bool bias = dyn->rs.depth_bias.enable;
   bool dbias_is_int =
      render->depth_att.vk_format == VK_FORMAT_D16_UNORM ||
      dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

   bool overflow =
      cs->scissor.size / AGX_SCISSOR_LENGTH + viewports * views > UINT16_MAX ||
      (bias &&
       cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH + views > UINT16_MAX);
   bool change_bias =
      bias && !u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
   if (unlikely(overflow || change_bias)) {
      perf_debug(cmd, "Splitting Apple9 render: %s",
                 overflow ? "scissor/depth bias overflow"
                          : "depth bias representation");
      /* The existing partial-render background, EOT and ZLS state stores and
       * reloads attachments. Only the final stream uses the API store ops.
       */
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }

   /* A stream that continued across a hoisted image barrier must not write
    * memory: tiles of its earlier passes may still read what it writes. */
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   bool writes = gfx->xfb_enabled;
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT + 1 && !writes; ++s) {
      if (s != MESA_SHADER_COMPUTE && gfx->shaders[s])
         writes = gfx->shaders[s]->writes_memory;
   }
   writes |= gfx->shaders[MESA_SHADER_TASK] &&
             gfx->shaders[MESA_SHADER_TASK]->writes_memory;
   writes |= gfx->shaders[MESA_SHADER_MESH] &&
             gfx->shaders[MESA_SHADER_MESH]->writes_memory;
   if (unlikely(writes && cs->merge_hoisted)) {
      perf_debug(
         cmd, "Splitting Apple9 render: memory writes after a hoisted barrier");
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }
   cs->merge_writes |= writes;

   /* What the draw's pre-rasterization work reads. hk-internal pre-raster
    * work (tessellation, geometry, transform feedback) keeps the tiling wait.
    * Emulated task draws retain their existing read classification. Meta draws
    * compile their pipelines like applications', so their shaders carry a read
    * class (vk_meta vertex shaders read push constants and its own vertex
    * data). */
   uint8_t reads = reads_buffers ? HK_READS_BUFFERS : 0;
   if (gfx->xfb_enabled || gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY])
      reads |= HK_READS_INTERNAL;
   for (unsigned s = 0; s < ARRAY_SIZE(gfx->shaders); ++s) {
      if (s == MESA_SHADER_FRAGMENT || s == MESA_SHADER_COMPUTE ||
          !gfx->shaders[s])
         continue;
      reads |= gfx->shaders[s]->pre_raster_reads;
      if (gfx->shaders[s]->writes_memory)
         reads |= HK_READS_WRITES;
   }
   cs->pre_raster_reads |= reads;
   cs->linear_store |= gfx->render.linear_store;

   return true;
}

/* Index fetch is per-command state. It does not feed the prepared raster,
 * shader bindings or launch records. Callers resolve null bindings and check
 * the aperture before using this for either the full or cached draw path. */
static void
hk_launch_set_index_fetch(struct agx_draw_pipeline *pipeline,
                          struct agx_draw draw, uint64_t buffer, uint64_t range,
                          uint64_t offset)
{
   pipeline->index_size = agx_index_size_to_B(draw.index_size);
   pipeline->index_buffer = buffer + offset;
   pipeline->index_extent =
      agx_is_indirect(draw.b) ? range - offset
                              : MIN2(range - offset, (uint64_t)draw.b.count[0] *
                                                        pipeline->index_size);
}

static void
hk_launch_emit_draw(struct hk_cs *cs, const struct agx_draw_pipeline *pipeline,
                    struct agx_draw draw)
{
   if (agx_is_indirect(draw.b))
      cs->current =
         agx_emit_indirect_draw(cs->current, pipeline, draw.b.ptr) - 4;
   else
      cs->current = agx_emit_direct_draw(
                       cs->current, pipeline, draw.b.count[0], draw.b.count[1],
                       draw.indexed ? draw.index_bias : draw.start) -
                    4;
}

/* Record the final draw packet after its resources and pipeline are prepared. */
static void
hk_launch_record_draw(struct hk_cs *cs,
                      const struct agx_draw_pipeline *pipeline,
                      struct agx_draw draw)
{
   hk_launch_emit_draw(cs, pipeline, draw);
}

static void
hk_launch_account_draw(struct hk_cs *cs,
                       const struct agx_draw_pipeline *pipeline,
                       struct agx_draw draw, bool emulated)
{
   if (cs->primitive_count < AGX_PRIMITIVE_COUNT_UNKNOWN) {
      /* Emulated output counts are not the API draw's primitive count. */
      agx_add_primitive_count(
         &cs->primitive_count,
         agx_is_indirect(draw.b) || pipeline->primitive_restart || emulated
            ? AGX_PRIMITIVE_COUNT_UNKNOWN
            : agx_count_primitives(pipeline->primitive, draw.b.count[0],
                                   draw.b.count[1]));
   }
   cs->stats.cmds++;
}

/* Whether a bound stage reads the draw index from the root table. Other
 * draws leave the root's draw_id unchanged: a multi-draw's identical draws
 * then reuse one root upload instead of uploading it per draw. */
static bool
hk_stages_read_draw_id(const struct hk_graphics_state *gfx)
{
   for (unsigned s = 0; s < ARRAY_SIZE(gfx->shaders); ++s) {
      if (s != MESA_SHADER_COMPUTE && gfx->shaders[s] &&
          gfx->shaders[s]->reads_draw_id)
         return true;
   }
   return false;
}

static void
hk_put_push(uint16_t *push, unsigned uniform, const void *data, size_t size)
{
   memcpy(push + uniform, data, size);
}

/* Push words mirroring Apple8's reserved uniforms; draw params by address */
uint64_t
hk_launch_push_words(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                     const struct hk_linked_shader *linked, uint64_t root)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_root_descriptor_table *table =
      s->info.stage == MESA_SHADER_COMPUTE ? &cmd->state.cs.descriptors.root
                                           : &gfx->descriptors.root;
   uint16_t push[AGX_LAUNCH_MAX_PUSH_WORDS * 2] = {0};
   unsigned root_unif = 0;

   if (s->info.stage == MESA_SHADER_VERTEX) {
      unsigned count =
         DIV_ROUND_UP(BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);

      for (unsigned i = 0; i < count; ++i) {
         hk_put_push(push, AGX_ABI_VUNI_VBO_BASE(i),
                     &table->draw.attrib_base[i], 8);
         hk_put_push(push, AGX_ABI_VUNI_VBO_CLAMP(count, i),
                     &table->draw.attrib_clamps[i], 4);
      }

      hk_put_push(push, AGX_ABI_VUNI_FIRST_VERTEX(count), &gfx->draw_params, 8);

      uint16_t draw_id = table->draw.draw_id;
      hk_put_push(push, AGX_ABI_VUNI_DRAW_ID(count), &draw_id, 2);

      if (linked && linked->sw_indexing) {
         hk_put_push(push, AGX_ABI_VUNI_VERTEX_PARAMS(count),
                     &table->draw.vertex_params, 8);
      }

      root_unif = AGX_ABI_VUNI_COUNT_VK(count);
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      assert(!agx_tilebuffer_spills(&gfx->render.tilebuffer));
      hk_put_push(push, AGX_ABI_FUNI_BLEND_R, table->draw.blend_constant, 16);
      root_unif = AGX_ABI_FUNI_ROOT;
   }

   /* The root and the set addresses, as the root table starts */
   hk_put_push(push, root_unif, &root, 8);
   hk_put_push(push, root_unif + 4, table->sets, 8 * s->info.set_count);

   unsigned size = 2 * (root_unif + 4 * (1 + s->info.set_count));
   assert(size <= sizeof(push));
   return hk_pool_upload(cmd, push, size, 16);
}

/* Link a draw's stage pair: varying layout and per-provoking-vertex tables */
static bool
hk_launch_link_stages(struct hk_cmd_buffer *cmd, const struct hk_shader *vs,
                      const struct hk_linked_shader *vs_linked,
                      const struct hk_shader *fs,
                      const struct hk_linked_shader *fs_linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_unlinked_uvs_layout *uvs = &vs->info.uvs;
   struct agx_draw_stage *vertex = &cmd->draw_stages[0];
   struct agx_draw_stage *fragment = &cmd->draw_stages[1];

   *vertex = vs_linked->draw_stages;
   *fragment = fs_linked->draw_stages;

   /* Varying groups as agx_assign_uvs orders them */
   uint64_t flat =
      (fs->info.fs.interp.flat & BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)) |
      VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT;
   uint64_t linear = fs->info.fs.interp.linear |
                     BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2);

   struct agx_varying_layout layout = {.count = uvs->user_size};
   u_foreach_bit64(loc, uvs->written) {
      layout.mask[loc] = BITFIELD_MASK(uvs->components[loc]);
      layout.group[loc] = agx_varying_group(flat, linear, loc);
   }

   vertex->varyings = fragment->varyings = layout;
   vertex->varying_components = fragment->varying_components = uvs->user_size;
   vertex->writes_point_size =
      uvs->group_offs[UVS_LAYER_VIEWPORT] > uvs->group_offs[UVS_PSIZ];
   vertex->writes_layer_viewport =
      uvs->group_offs[UVS_CLIP_DIST] > uvs->group_offs[UVS_LAYER_VIEWPORT];
   vertex->clip_distance_count = uvs->size - uvs->group_offs[UVS_CLIP_DIST];

   /* A table for each provoking vertex, for the flat shade model */
   unsigned bindings = fs_linked->b.cf.nr_bindings;
   size_t size =
      AGX_CF_BINDING_HEADER_LENGTH + bindings * AGX_CF_BINDING_LENGTH;
   bool primitive_id = false;

   for (unsigned pv = 0; pv < 3; ++pv) {
      struct agx_ptr t = hk_pool_usc_alloc(cmd, size, 16);
      if (!t.cpu)
         return false;

      agx_link_varyings_vs_fs(t.cpu, &gfx->linked_varyings, uvs->user_size,
                              (struct agx_varyings_fs *)&fs_linked->b.cf, pv, 0,
                              &primitive_id);

      fragment->coefficients[pv] = agx_usc_addr(&dev->dev, t.gpu);
   }

   fragment->coefficient_count = bindings;
   fragment->reads_primitive_id = primitive_id;
   fragment->reads_z = fs_linked->b.cf.reads_z;

   return agx_link_draw_pipeline(&cmd->draw_pipeline, vertex, fragment);
}

/* Reuse preparation, not emission: every draw still records its scope and
 * emits its ordinary VDM command. Root updates preserve the normal upload
 * order and rebuild only the tables/launches that bind the changed root.
 */
static bool
hk_launch_try_cached_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                          struct agx_draw draw)
{
   if (!hk_draw_record_reusable(cmd, draw))
      return false;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_draw_pipeline *pipeline = &cmd->draw_cache.pipeline;
   if (draw.indexed) {
      uint64_t offset =
         (uint64_t)draw.start * agx_index_size_to_B(draw.index_size);
      uint64_t address = draw.index_buffer + offset;
      if (!hk_va_in_usc(dev, address))
         return false;
      hk_launch_set_index_fetch(pipeline, draw, draw.index_buffer,
                                draw.index_buffer_range_B, offset);
      pipeline->primitive_restart =
         cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable;
      pipeline->restart_index = gfx->index.restart;
   } else {
      pipeline->index_size = 0;
      pipeline->primitive_restart = false;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   assert(cs && cs == cmd->draw_record_cs);

   if (hk_stages_read_draw_id(gfx) && desc->root.draw.draw_id != draw_id) {
      desc->root.draw.draw_id = draw_id;
      desc->root_dirty = true;
   }
   if (desc->root_dirty || desc->push_dirty) {
      if (desc->push_dirty)
         hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      uint64_t root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      if (!root)
         return true; /* The pool recorded the allocation failure. */
      if (root != gfx->root) {
         gfx->root = root;
         struct agx_draw_record *record =
            &cmd->draw_records[!cmd->draw_record_current];

         /* The push words hold the root address */
         const uint64_t tables[2] = {record->vertex_table,
                                     record->fragment_table};
         const uint64_t push[2] = {
            hk_launch_push_words(cmd, cmd->draw_cache.shader[0],
                                 cmd->draw_cache.linked[0], root),
            hk_launch_push_words(cmd, cmd->draw_cache.shader[1],
                                 cmd->draw_cache.linked[1], root),
         };
         if (!push[0] || !push[1])
            return true; /* The pool recorded the allocation failure. */

         if (!agx_update_draw_tables_alloc(&dev->dev, cmd, hk_usc_alloc_cb, cmd,
                                           hk_context_alloc_cb, pipeline,
                                           record, tables, push)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return true;
         }
         pipeline->ppp = record->ppp;
         pipeline->ppp_viewport_offset = record->ppp_viewport_offset;
         pipeline->ppp_viewport_words = record->ppp_viewport_words;
         pipeline->vertex_launch = record->launch[0] / 0x40;
      }
      desc->root_dirty = false;
   }

   hk_ensure_cs_has_space(cmd, cs, 256);
   hk_launch_emit_draw(cs, pipeline, draw);
   hk_launch_account_draw(cs, pipeline, draw, false);
   return true;
}

static void hk_pack_stencil_face(void *out, struct vk_stencil_test_face_state s,
                                 bool enabled);
static void hk_pack_depth_bias(struct hk_cmd_buffer *cmd, void *out);

static void
hk_launch_draw_impl(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    struct agx_draw draw)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* Geometry shaders can produce empty rasterization draws. */
   if (!agx_is_indirect(draw.b) && (!draw.b.count[0] || !draw.b.count[1]))
      return;
   if (hk_launch_try_cached_draw(cmd, draw_id, draw))
      return;
   cmd->draw_cache.valid = false;
   const char *failure = NULL;
   VkResult failure_result = VK_ERROR_FEATURE_NOT_PRESENT;
   struct hk_shader *shaders[2] = {NULL, NULL};
#define FAIL(result, why)                                                      \
   do {                                                                        \
      failure_result = result;                                                 \
      failure = why;                                                           \
      goto fail;                                                               \
   } while (0)
#define UNSUPPORTED(why) FAIL(VK_ERROR_FEATURE_NOT_PRESENT, why)
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   bool rasterizer_discard = dyn->rs.rasterizer_discard_enable;
   /* A geometry shader has already run as compute; its rasterization
    * variant is the hardware vertex program for its output primitives.
    */
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   /* A non-zero rasterization stream discards unless a multistream geometry
    * shader selects that stream.
    */
   if (dyn->rs.rasterization_stream != 0) {
      rasterizer_discard |=
         !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   /* Mesh shaders have written their records as compute. */
   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];

   /* Fast-linked programs; only the API vertex shader relinks per draw */
   shaders[0] = hk_bound_hw_vs(gfx);
   shaders[1] = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_linked_shader *linked[2] = {
      shaders[0] == hk_bound_sw_vs(gfx) ? gfx->linked[MESA_SHADER_VERTEX]
                                        : shaders[0]->only_linked,
      gfx->linked[MESA_SHADER_FRAGMENT],
   };

   /* Geometry and tessellation have already run as compute, and draw their
    * own output topology. Adjacency primitives without a geometry shader
    * draw a generated list without the adjacent vertices.
    */
   enum mesa_prim api_prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool adjacency = !mesh && !gs && !tes && mesa_prim_has_adjacency(api_prim);
   bool emulated = mesh || gs || tes || adjacency;
   /* Unsupported state must fail recording before any job can be submitted. */
   if (!linked[0] || !linked[1] ||
       (!rasterizer_discard &&
        (!dyn->vp.viewport_count ||
         dyn->vp.viewport_count > AGX_PPP_MAX_VIEWPORTS ||
         dyn->vp.scissor_count != dyn->vp.viewport_count)) ||
       (!emulated &&
        dyn->ia.primitive_topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
        dyn->ia.primitive_topology !=
           VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA))
      UNSUPPORTED("unsupported viewport, scissor or topology state");
   enum mesa_prim gs_prim = mesh        ? hk_only_variant(mesh)->info.mesh.prim
                            : gs        ? shaders[0]->info.gs.mode
                            : tes       ? gfx->tess.prim
                            : adjacency ? u_reduced_prim(api_prim)
                                        : MESA_PRIM_COUNT;
   bool points =
      emulated ? gs_prim == MESA_PRIM_POINTS
               : dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool lines =
      emulated
         ? (gs_prim == MESA_PRIM_LINES || gs_prim == MESA_PRIM_LINE_STRIP)
         : (dyn->ia.primitive_topology >= VK_PRIMITIVE_TOPOLOGY_LINE_LIST &&
            dyn->ia.primitive_topology <= VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
   if (cmd->draw_pipeline_ids[0] != linked[0]->draw_stages.program_id ||
       cmd->draw_pipeline_ids[1] != linked[1]->draw_stages.program_id) {
      if (!hk_launch_link_stages(cmd, shaders[0], linked[0], shaders[1],
                                 linked[1])) {
         cmd->draw_pipeline_ids[0] = cmd->draw_pipeline_ids[1] = 0;
         UNSUPPORTED("the vertex and fragment programs cannot be linked");
      }
      cmd->draw_pipeline_ids[0] = linked[0]->draw_stages.program_id;
      cmd->draw_pipeline_ids[1] = linked[1]->draw_stages.program_id;
      agx_build_ppp_static(&cmd->draw_pipeline, cmd->draw_ppp_static);
      cmd->draw_pipeline.ppp_static = cmd->draw_ppp_static;
   }
   const struct agx_draw_stage *fragment = &cmd->draw_stages[1];
   struct agx_draw_pipeline pipeline = cmd->draw_pipeline;
   /* Attachmentless render passes acquire their sample count from the first
    * drawing pipeline. Keep the native tile layout and submission state in
    * agreement with the fragment launch, including a pre-existing query CS.
    */
   if (!gfx->render.tilebuffer.nr_samples && dyn->ms.rasterization_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);
   }
   pipeline.samples = MAX2(gfx->render.tilebuffer.nr_samples, 1);

   /* The draw allocates the render's tile: its target formats, in order */
   cmd->draw_stages[1].render_targets = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(pipeline.color_formats); ++rt) {
      enum pipe_format format = gfx->render.tilebuffer.logical_format[rt];
      if (format != PIPE_FORMAT_NONE && !gfx->render.tilebuffer.spilled[rt])
         pipeline.color_formats[cmd->draw_stages[1].render_targets++] = format;
   }
   static const uint8_t primitives[] = {
      AGX_PRIMITIVE_POINTS,         AGX_PRIMITIVE_LINES,
      AGX_PRIMITIVE_LINE_STRIP,     AGX_PRIMITIVE_TRIANGLES,
      AGX_PRIMITIVE_TRIANGLE_STRIP, AGX_PRIMITIVE_TRIANGLE_FAN,
   };
   /* Mesa meta rectangles are already expanded to two ordinary triangles. */
   pipeline.primitive = emulated ? hk_rast_primitive(gs_prim)
                        : dyn->ia.primitive_topology ==
                                 VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA ||
                              gfx->fan_unrolled
                           ? AGX_PRIMITIVE_TRIANGLES
                           : primitives[dyn->ia.primitive_topology];
   /* Emulated stages emit their primitives for the same provoking mode.
    * Fans that are not unrolled have no observable provoking vertex, and
    * native fans flat shade consistently only from their first vertex. */
   const bool native_fan =
      !mesh && !gs && !tes &&
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
      !gfx->fan_unrolled;
   pipeline.flatshade_first =
      dyn->rs.provoking_vertex != VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ||
      native_fan;
   if (draw.indexed) {
      pipeline.index_size = agx_index_size_to_B(draw.index_size);
      uint64_t offset = agx_is_indirect(draw.b)
                           ? 0
                           : (uint64_t)draw.start * pipeline.index_size;
      uint64_t index_buffer = draw.index_buffer;
      uint64_t index_range = draw.index_buffer_range_B;
      /* Null bindings and reads past the binding fetch index zero. The
       * hardware bounds the fetch by the extent, and the source must lie in
       * the USC aperture, so use a small zeroed pool allocation.
       */
      if (offset >= index_range) {
         struct agx_ptr zero = hk_pool_usc_alloc(cmd, 16, 16);
         if (!zero.cpu)
            return;
         memset(zero.cpu, 0, 16);
         index_buffer = zero.gpu;
         index_range = 16;
         offset = 0;
      }
      pipeline.index_buffer = index_buffer + offset;
      if (!hk_va_in_usc(dev, pipeline.index_buffer))
         UNSUPPORTED("the index buffer is outside the shader window");
      hk_launch_set_index_fetch(&pipeline, draw, index_buffer, index_range,
                                offset);
      pipeline.primitive_restart =
         emulated ? draw.restart : dyn->ia.primitive_restart_enable;
      pipeline.restart_index =
         emulated ? BITFIELD_MASK(pipeline.index_size * 8) : gfx->index.restart;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      return;
   cs = hk_occlusion_prepare_draw(cmd, cs);
   if (!cs)
      return;
   cs->tib = gfx->render.tilebuffer;
   const struct vk_dynamic_graphics_state *ms_dyn =
      &cmd->vk.dynamic_graphics_state;
   gfx->sample_positions = hk_draw_sample_positions(ms_dyn, pipeline.samples);
   /* hk_launch_prepare_draw committed the render's positions; meta draws
    * only fill them in for a render that has none yet. */
   if (!cs->has_sample_locations)
      cs->ppp_multisamplectl = gfx->sample_positions;
   hk_reserve_scratch(cmd, cs, shaders[0]);
   hk_reserve_scratch(cmd, cs, shaders[1]);
   struct agx_draw_record *record =
      &cmd->draw_records[cmd->draw_record_current];
   agx_draw_record_reset(record);
   /* Rasterizer discard pipelines may omit viewport/scissor state entirely.
    * Supply harmless hardware state without consulting those ignored API
    * values or dropping the draw through the empty-scissor optimization.
    */
   const VkViewport discard_viewport = {.width = 1, .height = 1, .maxDepth = 1};
   const VkRect2D *area = &gfx->render.area;
   unsigned viewports = rasterizer_discard ? 1 : dyn->vp.viewport_count;
   record->viewport_count = viewports;
   /* Viewport scissors are consecutive; the shader's viewport index selects
    * both the transform and the scissor. The tile-granular region clip is
    * their union.
    */
   struct agx_scissor_packed scissors[AGX_PPP_MAX_VIEWPORTS];
   int bbox[4] = {INT_MAX, INT_MAX, INT_MIN, INT_MIN};
   for (unsigned v = 0; v < viewports; ++v) {
      const VkViewport *vp =
         rasterizer_discard ? &discard_viewport : &dyn->vp.viewports[v];
      float transform[6] = {
         vp->x + vp->width * 0.5f,
         vp->width * 0.5f,
         vp->y + vp->height * 0.5f,
         vp->height * 0.5f,
         vp->minDepth,
         vp->maxDepth - vp->minDepth,
      };
      if (v == 0) {
         for (unsigned axis = 0; axis < 3; ++axis) {
            record->viewport_translate[axis] = transform[axis * 2];
            record->viewport_scale[axis] = transform[axis * 2 + 1];
         }
      } else {
         memcpy(record->viewports[v - 1], transform, sizeof(transform));
      }
      const VkRect2D *sc = rasterizer_discard ? area : &dyn->vp.scissors[v];
      int minx = MAX2(sc->offset.x, area->offset.x);
      int miny = MAX2(sc->offset.y, area->offset.y);
      int maxx = MIN2((int64_t)sc->offset.x + sc->extent.width,
                      (int64_t)area->offset.x + area->extent.width);
      int maxy = MIN2((int64_t)sc->offset.y + sc->extent.height,
                      (int64_t)area->offset.y + area->extent.height);
      /* Clipping has a guard band, so primitives are also scissored to
       * their viewport, which may be inverted.
       */
      if (!rasterizer_discard) {
         minx = MAX2(minx, (int)floorf(MIN2(vp->x, vp->x + vp->width)));
         miny = MAX2(miny, (int)floorf(MIN2(vp->y, vp->y + vp->height)));
         maxx = MIN2(maxx, (int)ceilf(MAX2(vp->x, vp->x + vp->width)));
         maxy = MIN2(maxy, (int)ceilf(MAX2(vp->y, vp->y + vp->height)));
      }
      if (minx >= maxx || miny >= maxy) {
         /* Empty scissor: a zero-area rectangle. */
         maxx = minx = MAX2(minx, 0);
         maxy = miny = MAX2(miny, 0);
      } else {
         bbox[0] = MIN2(bbox[0], minx);
         bbox[1] = MIN2(bbox[1], miny);
         bbox[2] = MAX2(bbox[2], maxx);
         bbox[3] = MAX2(bbox[3], maxy);
      }
      /* Depth clamping bounds depth to the viewport's range, unless a range
       * is requested instead. Unclamped depth is bounded to [0, 1], which
       * is what depthClampZeroOne specifies when clamping is disabled.
       */
      float clamp_min = MIN2(vp->minDepth, vp->maxDepth);
      float clamp_max = MAX2(vp->minDepth, vp->maxDepth);
      if (dyn->vp.depth_clamp_mode ==
          VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT) {
         clamp_min = dyn->vp.depth_clamp_range.minDepthClamp;
         clamp_max = dyn->vp.depth_clamp_range.maxDepthClamp;
      }
      agx_pack(&scissors[v], SCISSOR, cfg) {
         cfg.min_x = minx;
         cfg.min_y = miny;
         cfg.max_x = maxx;
         cfg.max_y = maxy;
         cfg.min_z = dyn->rs.depth_clamp_enable ? clamp_min : 0;
         cfg.max_z = dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
      /* The scissor range bounds fixed-function depth only; shaders clamp
       * their own depth exports (hk_clamp_depth_export). */
      if (v == 0) {
         gfx->descriptors.root.draw.depth_clamp[0] =
            dyn->rs.depth_clamp_enable ? clamp_min : 0;
         gfx->descriptors.root.draw.depth_clamp[1] =
            dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
   }
   if (bbox[0] >= bbox[2]) {
      /* Every viewport scissor is empty, so nothing rasterizes. The hardware
       * vertex program still owns pre-rasterization side effects: geometry
       * shader stores and atomics run in its rasterization variant (poly
       * emits a culled dummy primitive for them), and so do vertex shader
       * stores and transform feedback. Mesh records were written as compute.
       * Only drop draws without such effects; otherwise run against a
       * zero-area region.
       */
      bool effects = gfx->xfb_enabled;
      for (unsigned s = 0; s < MESA_SHADER_FRAGMENT && !effects; ++s)
         effects = gfx->shaders[s] && gfx->shaders[s]->writes_memory;
      if (!effects)
         return;
      bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0;
   }
   record->scissor_min[0] = bbox[0];
   record->scissor_min[1] = bbox[1];
   record->scissor_max[0] = bbox[2];
   record->scissor_max[1] = bbox[3];
   unsigned scissor_count = cs->scissor.size / AGX_SCISSOR_LENGTH;
   if (scissor_count >= viewports &&
       !memcmp((uint8_t *)cs->scissor.data +
                  (scissor_count - viewports) * AGX_SCISSOR_LENGTH,
               scissors, viewports * sizeof(scissors[0]))) {
      record->scissor_index = scissor_count - viewports;
   } else {
      assert(scissor_count + viewports <= UINT16_MAX);
      record->scissor_index = scissor_count;
      util_dynarray_append_array(&cs->scissor, struct agx_scissor_packed,
                                 scissors, viewports);
   }
   bool bias = dyn->rs.depth_bias.enable && !points && !lines;
   if (bias) {
      bool d16 = gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
      /* The submission selects the matching depth bias mode */
      ASSERTED bool unorm = d16 || force_unorm;
      assert(cs->cr.dbias_is_int == (unorm ? U_TRISTATE_YES : U_TRISTATE_NO));
      struct agx_depth_bias_packed depth_bias;
      hk_pack_depth_bias(cmd, &depth_bias);
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      if (count && !memcmp((uint8_t *)cs->depth_bias.data +
                              (count - 1) * AGX_DEPTH_BIAS_LENGTH,
                           &depth_bias, sizeof(depth_bias))) {
         record->depth_bias_index = count - 1;
      } else {
         assert(count < UINT16_MAX);
         record->depth_bias_index = count;
         util_dynarray_append(&cs->depth_bias, depth_bias);
      }
   }
   bool stencil = dyn->ds.stencil.test_enable && gfx->render.stencil_att.iview;
   /* Without a depth attachment, the depth test is disabled. */
   bool depth_test = dyn->ds.depth.test_enable && gfx->render.depth_att.iview;
   record->depth_control = 0x200 | (1u << 16) | (bias ? 1u << 17 : 0) |
                           (stencil ? 3u << 18 : 0) |
                           (rasterizer_discard ? 1u << 21 : 0);
   record->occlusion_index = gfx->occlusion.index;
   record->visibility_mode = gfx->occlusion.mode;
   record->depth_control |= record->visibility_mode << 14;
   /* Early tests keep a hardware NEVER: the shader must not run at all. */
   bool force_never_in_shader =
      depth_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER &&
      shaders[1]->info.fs.writes_memory &&
      !shaders[1]->b.info.early_fragment_tests;
   gfx->descriptors.root.draw.force_never_in_shader = force_never_in_shader;
   /* The shader fails a forced NEVER test itself
    * (hk_clamp_depth_export). */
   VkCompareOp depth_compare = !depth_test ? VK_COMPARE_OP_ALWAYS
                               : force_never_in_shader
                                  ? VK_COMPARE_OP_LESS
                                  : dyn->ds.depth.compare_op;
   record->depth_face[0] = record->depth_face[1] =
      (agx_pack_line_width(dyn->rs.line.width) << 8) | (depth_compare << 24) |
      ((depth_test && dyn->ds.depth.write_enable) ? 0 : (1 << 21)) |
      (dyn->rs.polygon_mode << 18);
   for (unsigned face = 0; face < 2; ++face) {
      const struct vk_stencil_test_face_state *st =
         face ? &dyn->ds.stencil.back : &dyn->ds.stencil.front;
      record->depth_face[face] |= st->reference & 0xff;
      hk_pack_stencil_face(&record->stencil[face], *st, stencil);
   }
   struct agx_cull_packed cull;
   agx_pack(&cull, CULL, cfg) {
      cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
      cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
      cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
      cfg.depth_clamp = !cfg.depth_clip;
      /* Bresenham lines cover every sample of a covered pixel. */
      cfg.primitive_msaa =
         lines && dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
   }
   memcpy(&record->raster_control, &cull, 4);
   bool front_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
   /* The hardware-VS TES consumes the tessellator's unswapped indices, so
    * encode its domain winding here. */
   if (tes && !gs) {
      front_ccw ^= gfx->tess.info.ccw;
      front_ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }
   record->raster_control |= front_ccw ? 1u << 16 : 0;
   record->object_type = points  ? AGX_OBJECT_TYPE_POINT_SPRITE_UV01
                         : lines ? AGX_OBJECT_TYPE_LINE
                                 : AGX_OBJECT_TYPE_TRIANGLE;
   record->reads_tile = fragment->reads_tile;
   record->uses_discard = fragment->uses_discard;
   record->writes_depth = fragment->writes_depth;
   record->disable_tri_merging = fragment->disable_tri_merging;
   record->pixel_interlock = fragment->pixel_interlock;
   record->flatshade_first = pipeline.flatshade_first;
   gfx->descriptors.root.draw.api_sample_mask = dyn->ms.sample_mask;
   gfx->descriptors.root.draw.ppp_multisamplectl = gfx->sample_positions;
   gfx->descriptors.root.draw.samples_log2 = util_logbase2(pipeline.samples);
   gfx->descriptors.root.draw.clip_z_coeff =
      dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
   if (hk_stages_read_draw_id(gfx))
      gfx->descriptors.root.draw.draw_id = draw_id;
   gfx->descriptors.root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, &gfx->descriptors);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct agx_draw_stage *s = &cmd->draw_stages[stage];
      uint64_t sysvals = 0;
      hk_upload_texture_tables(cmd, &record->texture_table[stage],
                               &record->sampler_table[stage], &sysvals);

      record->push[stage] =
         hk_launch_push_words(cmd, shaders[stage], linked[stage], gfx->root);
      if (!record->push[stage])
         FAIL(VK_ERROR_OUT_OF_DEVICE_MEMORY, "out of memory");

      for (unsigned slot = 0; slot < s->resource_count; ++slot) {
         if (s->resource_binding[slot] != AGX_GRAPHICS_SYSVAL_BINDING)
            UNSUPPORTED("unknown uniform buffer binding");

         record->buffers[stage][slot] = sysvals;
      }
      uint64_t table = hk_pool_upload_last(
         cmd,
         stage ? HK_UPLOAD_LAST_FRAGMENT_TABLE : HK_UPLOAD_LAST_VERTEX_TABLE,
         record->buffers[stage], s->resource_count * sizeof(uint64_t), 64);
      if (stage)
         record->fragment_table = table;
      else
         record->vertex_table = table;
   }
   const struct agx_draw_record *previous =
      cmd->draw_record_cs == cs ? &cmd->draw_records[!cmd->draw_record_current]
                                : NULL;
   bool prepared = agx_prepare_draw_record_alloc(
      &dev->dev, cmd, hk_usc_alloc_cb, cmd, hk_context_alloc_cb, &pipeline,
      record, previous);
   if (!prepared)
      UNSUPPORTED("the draw state cannot be prepared");
   cmd->draw_record_current = !cmd->draw_record_current;
   cmd->draw_record_cs = cs;
   pipeline.ppp = record->ppp;
   pipeline.ppp_viewport_offset = record->ppp_viewport_offset;
   pipeline.ppp_viewport_words = record->ppp_viewport_words;
   pipeline.vertex_launch = record->launch[0] / 0x40;
   pipeline.pipeline_word = 0x01000000;
   hk_ensure_cs_has_space(cmd, cs, 256);
   hk_launch_record_draw(cs, &pipeline, draw);
   hk_launch_account_draw(cs, &pipeline, draw, gs || tes || mesh);
   /* Only ordinary single-view draws reuse prepared state. Other paths
    * still consume dynamic state in their software-stage lowering. */
   if (hk_launch_draw_cache_supported(cmd, draw)) {
      cmd->draw_cache.valid = true;
      cmd->draw_cache.render_gen = gfx->render_gen;
      cmd->draw_cache.first_vertex =
         draw.indexed ? draw.index_bias : draw.start;
      cmd->draw_cache.base_instance = draw.start_instance;
      cmd->draw_cache.pipeline = pipeline;
      for (unsigned stage = 0; stage < 2; ++stage) {
         cmd->draw_cache.shader[stage] = shaders[stage];
         cmd->draw_cache.linked[stage] = linked[stage];
      }
      /* Every dynamic, hk and shader dirty bit is consumed here, so the
       * next clean draw can reuse this state. G16 has no other dirty-bit
       * readers after this point (only shader resolution and the sysval
       * upload, both above); a new one must read its bits before this or
       * invalidate draw_cache itself. */
      vk_dynamic_graphics_state_clear_dirty(dyn);
      gfx->shaders_dirty = 0;
      gfx->dirty = 0;
      gfx->descriptors.root_dirty = false;
   }
   return;
fail:
   mesa_loge("hk: Apple9 draw failed: %s", failure);
#undef UNSUPPORTED
#undef FAIL
   vk_command_buffer_set_error(&cmd->vk, failure_result);
}

void
hk_launch_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
               struct agx_draw draw)
{
   hk_launch_draw_impl(cmd, draw_id, draw);
}

static struct agx_draw
hk_robust_index_copy(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1], draw.start,
                          draw.index_bias, draw.start_instance};
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }

   perf_debug(cmd, "copying robust index buffer");
   uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
   libagx_copy_robust_index(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                            out_draw, hk_heap(cmd), in_draw, draw.index_buffer,
                            draw.index_buffer_range_B >> draw.index_size,
                            agx_index_size_to_B(draw.index_size));

   return hk_heap_indexed_indirect(cmd, out_draw, draw.index_size,
                                   draw.restart);
}

static void
hk_merge_tess_info(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   struct hk_api_shader *tes_api = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_shader *tcs = hk_only_variant(tcs_api);
   struct hk_tess_info tes_info = hk_any_variant(tes_api)->info.tess.info;
   gfx->tess.info = hk_tess_info_merge(tes_info, tcs->info.tess.info);
   if (gfx->tess.info.points)
      gfx->tess.prim = MESA_PRIM_POINTS;
   else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES)
      gfx->tess.prim = MESA_PRIM_LINES;
   else
      gfx->tess.prim = MESA_PRIM_TRIANGLES;
}

/* Launch programs have no dynamic state flush ahead of the stages, so the
 * parameters the tessellation stages read are uploaded here.
 */
static struct agx_ptr
hk_launch_upload_tess_root(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
   struct agx_ptr tess_args =
      hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
   if (!tess_args.cpu)
      return tess_args;

   desc->root.draw.tess_params = tess_args.gpu;
   hk_upload_tess_params_for_tcs(cmd, tess_args.cpu, draw, tcs);
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   return tess_args;
}

static struct agx_draw
hk_launch_tess(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
               struct agx_draw draw, uint64_t c_prims, uint64_t c_inv)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct agx_grid grid_vs, grid_tcs, grid_tess;
   bool launch = agx_has_launch_programs(&dev->dev);

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   struct agx_ptr tess_args = {0};
   if (launch) {
      tess_args = hk_launch_upload_tess_root(cmd, draw);
      if (!tess_args.cpu)
         return draw;
   }

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->descriptors.root.draw.tess_params;
   struct hk_tess_info info = gfx->tess.info;

   perf_debug(cmd, "Tessellation");

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      perf_debug(cmd, "Indirect tessellation");

      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;

      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      if (hk_stat_enabled(tcs_stat)) {
         perf_debug(cmd, "Direct TCS statistic");
         libagx_increment_statistic(
            cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat, patches);
      }
   }

   unsigned nr_patches = tess_args.cpu && !agx_is_indirect(draw.b)
                            ? ((struct poly_tess_params *)tess_args.cpu)->nr_patches
                            : 0;

   /* With launch programs, a small direct draw without statistics or
    * transform feedback touches only its own allocations here, so its passes
    * can run alongside those of the neighbouring tessellated draws: record
    * them into the stage lanes (see hk_tess_batch_flush). The stage order
    * within each draw is kept by the barrier after every lane. A large draw
    * fills the GPU on its own and is kept inline.
    */
   const unsigned max_batch_patches = 2048;
   bool batch = launch && !agx_is_indirect(draw.b) &&
                (draw.b.count[0] / input_patch_size) * draw.b.count[1] <=
                   max_batch_patches &&
                !hk_stat_enabled(tcs_stat) && !hk_stat_enabled(c_prims) &&
                !hk_stat_enabled(c_inv) && !gfx->xfb_enabled;

   /* A full fold table starts a new batch. */
   if (launch &&
       (!batch || util_dynarray_num_elements(&cmd->tess_batch.fold,
                                             struct hk_tess_fold) >=
                     HK_TESS_FOLD_MAX_DRAWS))
      hk_tess_batch_flush(cmd);

   /* First launch the VS and TCS */
   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[0] : NULL;
   hk_dispatch_prerast(cmd, cs, vs, gfx->linked[MESA_SHADER_VERTEX], grid_vs,
                       agx_workgroup(64, 1, 1));

   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[1] : NULL;
   hk_dispatch_prerast(cmd, cs, tcs, tcs->only_linked, grid_tcs,
                       agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));

   if (batch) {
      /* The flush runs the count, prefix sum and emit passes of every folded
       * draw (hk_tess_fold_flush).
       */
      struct hk_tess_fold f = {
         .params = state,
         .nr_patches = nr_patches,
         .prim = info.mode,
      };
      util_dynarray_append(&cmd->tess_batch.fold, f);
      cmd->tess_batch.capture = NULL;
      cmd->tess_batch.draws++;
      return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws,
                                      AGX_INDEX_SIZE_U32, false);
   }

   cmd->tess_batch.capture = NULL;

   /* First generate counts, then prefix sum them, and then tessellate. */
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_COUNT, state);

   /* One workgroup walking every patch takes milliseconds for large draws,
    * so direct draws past a few blocks scan blocks in parallel instead.
    */
   const unsigned block_patches = 8192;
   if (nr_patches > 2 * block_patches) {
      uint64_t counts = ((struct poly_tess_params *)tess_args.cpu)->counts;
      unsigned nr_blocks = DIV_ROUND_UP(nr_patches, block_patches);
      uint64_t block_sums = hk_pool_alloc(cmd, nr_blocks * 4, 4).gpu;
      libagx_prefix_sum_tess_blocks(cmd, agx_1d(1024 * nr_blocks),
                                    AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                    block_sums, nr_patches, block_patches);
      libagx_prefix_sum_tess_offsets(
         cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, state, block_sums,
         nr_blocks, c_prims, c_inv, c_prims || c_inv);
      libagx_prefix_sum_tess_add(cmd, agx_1d(nr_patches - block_patches),
                                 AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                 block_sums, nr_patches, block_patches);
   } else {
      libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                             state, c_prims, c_inv, c_prims || c_inv);
   }

   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws, AGX_INDEX_SIZE_U32,
                                   false);
}

void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   uint64_t data = hk_shader_constant_data(shader);
   if (data || (desc->root_constant_stages & BITFIELD_BIT(stage))) {
      desc->root.constant_data[stage] = data;
      desc->root_constant_stages &= ~BITFIELD_BIT(stage);
      if (data)
         desc->root_constant_stages |= BITFIELD_BIT(stage);
      desc->root_dirty = true;
   }
   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT) {
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.shaders_dirty == 0)
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   desc->root_dirty = true;

   /* Geometry shading overrides the restart index, reemit on rebind */
   if (IS_SHADER_DIRTY(GEOMETRY)) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

      desc->root.draw.api_gs = gs && !gs->is_passthrough;
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   /* If we have a new VS/FS pair, UVS locations may have changed so need to
    * relink. We do this here because there's no dependence on the fast linked
    * shaders.
    */
   agx_assign_uvs(&gfx->linked_varyings, &hw_vs->info.uvs,
                  hk_only_variant(fs)->info.fs.interp.flat,
                  hk_only_variant(fs)->info.fs.interp.linear);

   /* Linked programs keep outputs in unlinked slots, see agx_link_program */
   if (agx_has_launch_programs(&hk_cmd_buffer_device(cmd)->dev)) {
      const struct agx_unlinked_uvs_layout *uvs = &hw_vs->info.uvs;
      unsigned slot = uvs->group_offs[UVS_VARYINGS];

      memset(gfx->linked_varyings.slots, 0,
             sizeof(gfx->linked_varyings.slots));
      u_foreach_bit64(loc, uvs->written) {
         gfx->linked_varyings.slots[loc] = slot;
         slot += uvs->components[loc];
      }
   }

   for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
      desc->root.draw.uvs_index[i] = gfx->linked_varyings.slots[i];
   }
}

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   struct agx_device_key dev_key = agx_gather_device_key(&dev->dev);
   bool launch_programs = agx_has_launch_programs(&dev->dev);

   nir_builder b = nir_builder_init_simple_shader(
      0, agx_nir_options_for_isa(dev_key.shader_isa), NULL);
   builder(&b, agx_device_target(&dev->dev), key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader, agx_device_target(&dev->dev));

   /* Apple9 parts read push words, and the linker adds the stop */
   struct agx_shader_key backend_key = {
      .dev = dev_key,
      .secondary = true,
      .no_stop = !stop || launch_programs,
      .graphics_abi = launch_programs,
   };

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);

   agx_compile_shader_nir(b.shader, &backend_key, part);

   ralloc_free(b.shader);

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return part;
}

struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part = hk_get_prolog_epilog_locked(
      dev, key, builder, preprocess_nir, stop, cf_base);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           agx_nir_vs_prolog, false, false, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   /* XXX: Fix this higher up the stack */
   linked->sw_indexing = !key->prolog.hw || key->prolog.adjacency;
   linked->b.uses_base_param |= linked->sw_indexing;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static bool
lower_fs_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT));
   return true;
}

static void
build_fs_prolog(nir_builder *b, const struct agx_isa_target *target,
                const void *key)
{
   agx_nir_fs_prolog(b, target, key);

   /* Lower load_stat_query_address_agx, needed for FS statistics */
   NIR_PASS(_, b->shader, hk_lower_uvs_index, target, MESA_SHADER_FRAGMENT, 0);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_root,
            nir_metadata_control_flow, NULL);
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    build_fs_prolog, false, false,
                                    key->prolog.cf_base);
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           agx_nir_fs_epilog, true, epilog_stop, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);

   /* The epilog may write depth the main shader does not, see late_zs */
   linked->draw_stages.writes_depth |= key->epilog.link.write_z;

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent)
      linked = ent->data;
   else if (shader->info.stage == MESA_SHADER_VERTEX)
      linked = hk_get_fast_linked_locked_vs(dev, shader, key);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      linked = hk_get_fast_linked_locked_fs(dev, shader, key);
   else
      UNREACHABLE("invalid stage");

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static void
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   mesa_shader_stage stage = shader->info.stage;

   if (cmd->state.gfx.linked[stage] != new) {
      cmd->state.gfx.linked[stage] = new;
      cmd->state.gfx.linked_dirty |= BITFIELD_BIT(stage);
   }
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static void
hk_pack_stencil_face(void *out, struct vk_stencil_test_face_state s,
                     bool enabled)
{
   if (enabled) {
      agx_pack(out, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_pack(out, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed;
   hk_pack_stencil_face(&packed, s, enabled);
   agx_ppp_push_packed(ppp, &packed, FRAGMENT_STENCIL);
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.viewport_count, 1);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < dyn->vp.scissor_count; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   struct agx_scissor_packed *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_pack(scissors + i, SCISSOR, cfg) {
         cfg.min_x = minx[i];
         cfg.min_y = miny[i];
         cfg.max_x = maxx[i];
         cfg.max_y = maxy[i];

         /* These settings in conjunction with the PPP control depth clip/clamp
          * settings implement depth clip/clamping. Properly setting them
          * together is required for conformant depth clip enable.
          *
          * TODO: Reverse-engineer the finer interactions here.
          */
         if (dyn->rs.depth_clamp_enable) {
            cfg.min_z = MIN2(vp->minDepth, vp->maxDepth);
            cfg.max_z = MAX2(vp->minDepth, vp->maxDepth);
         } else {
            cfg.min_z = 0.0;
            cfg.max_z = 1.0;
         }
      }
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   agx_ppp_fini(out, &ppp);
}

static enum agx_object_type
translate_object_type(enum mesa_prim topology)
{
   static_assert(MESA_PRIM_LINES < MESA_PRIM_LINE_STRIP);
   static_assert(MESA_PRIM_TRIANGLES >= MESA_PRIM_LINE_STRIP);

   if (topology == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (topology <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

static enum agx_primitive
translate_hw_primitive_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      UNREACHABLE("Invalid hardware primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   uint32_t index = cmd->state.gfx.index.restart;

   if (gs) {
      enum poly_gs_shape shape =
         gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape;
      index = BITFIELD_MASK(8 * poly_gs_index_size(shape));
   }

   /* VDM State updates are relatively expensive, so only emit them when the
    * restart index changes. This is simpler than accurate dirty tracking.
    */
   if (cs->restart_index != index) {
      uint8_t *out = cs->current;
      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = index;
      }

      cs->current = out;
      cs->restart_index = index;
   }
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool hw_vs_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(TESS_EVAL) ||
                      IS_SHADER_DIRTY(GEOMETRY);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = hw_vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull = IS_DIRTY(RS_CULL_MODE) ||
              IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
              IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
              IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
              IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
              (gfx->dirty & HK_DIRTY_PROVOKING) || IS_SHADER_DIRTY(TESS_CTRL) ||
              IS_SHADER_DIRTY(TESS_EVAL) || IS_DIRTY(TS_DOMAIN_ORIGIN),
      .cull_2 = varyings_dirty,

      .fragment_shader = fs_dirty || linked_fs_dirty || varyings_dirty ||
                         gfx->descriptors.root_dirty,

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = hw_vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   dirty.fragment_shader &= !linked_fs->b.no_op;

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          linked_fs->b.fragment_control) {

         cfg.tag_write_disable = hk_rast_discard(cmd);
      }
   }

   struct agx_fragment_face_packed fragment_face = {};
   struct agx_fragment_face_2_packed fragment_face_2 = {};

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;
      }

      agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);
      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = hw_vs->info.uvs.osel;

      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                &linked_fs->b.osel);
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;

         if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
             !gfx->shaders[MESA_SHADER_GEOMETRY]) {
            cfg.front_face_ccw ^= gfx->tess.info.ccw;
            cfg.front_face_ccw ^= dyn->ts.domain_origin ==
                                  VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
         }

         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = hk_rast_discard(cmd);

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, fs, linked_fs);
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = hw_vs->info.uvs.size;
      }
   }

   agx_ppp_fini(out, &ppp);
}

/*
 * Based somewhat on the calculation in the PowerVR driver, and mostly trial &
 * error to pass CTS. This is a mess.
 */
static float
hk_depth_bias_factor(VkFormat format, bool exact, bool force_unorm)
{
   if (format == VK_FORMAT_D16_UNORM) {
      return exact ? (1 << 16) : (1 << 15);
   } else if (force_unorm) {
      return exact ? (1ull << 24) : (1ull << 23);
   } else {
      return 1.0;
   }
}

static void
hk_pack_depth_bias(struct hk_cmd_buffer *cmd, void *out)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   bool exact = dyn->rs.depth_bias.exact;
   bool force_unorm =
      dyn->rs.depth_bias.representation ==
      VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

   agx_pack(out, DEPTH_BIAS, cfg) {
      cfg.slope_scale = dyn->rs.depth_bias.slope_factor;
      cfg.clamp = dyn->rs.depth_bias.clamp;
      cfg.depth_bias = dyn->rs.depth_bias.constant_factor;
      cfg.depth_bias /= hk_depth_bias_factor(
         cmd->state.gfx.render.depth_att.vk_format, exact, force_unorm);
   }
}

static bool
uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 1 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 2 ==
                 VK_BLEND_FACTOR_CONSTANT_ALPHA);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA);

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

void
agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                    uint64_t attribs_read, struct agx_velem_key *keys)
{
   u_foreach_bit(a, vi->attributes_valid) {
      struct vk_vertex_attribute_state attr = vi->attributes[a];

      assert(vi->bindings_valid & BITFIELD_BIT(attr.binding));
      struct vk_vertex_binding_state binding = vi->bindings[attr.binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(a));

      keys[slot] = (struct agx_velem_key){
         .format = hk_format_to_pipe_format(attr.format),
         .stride = binding.stride,
         .divisor = binding.divisor,
         .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
      };
   }
}

/* Root table fields and fast-linked programs shared by both ISAs */
static void
hk_flush_shader_parts(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                      struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || gt_dirty;
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   if (IS_DIRTY(CB_BLEND_CONSTANTS)) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(CB_BLEND_EQUATIONS) || IS_DIRTY(CB_BLEND_ENABLES)) {
      gfx->uses_blend_constant = uses_blend_constant(&dyn->cb);
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      desc->root.draw.api_sample_mask = dyn->ms.sample_mask;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(RS_RASTERIZATION_STREAM)) {
      desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
      desc->root_dirty = true;

      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      desc->root.draw.force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      desc->root_dirty = true;
   }

   /* The main shader must not run tests if the epilog will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   /* Draw-time Z/S feedback loops test late, on the depth the epilog writes */
   bool late_zs = fs && fs->info.fs.dynamic_late_zs &&
                  (dyn->feedback_loops &
                   (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));

   bool epilog_discards = dyn->ms.alpha_to_coverage_enable || late_zs ||
                          (fs && (fs->info.fs.epilog_key.write_z ||
                                  fs->info.fs.epilog_key.write_s));
   epilog_discards &= !nontrivial_force_early;

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(ATTACHMENT_FEEDBACK_LOOP_ENABLE)) {
      desc->root.draw.no_epilog_discard = !epilog_discards ? ~0 : 0;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty || true /* TODO */) {

      struct hk_fast_link_key_vs key = {
         .prolog.hw = (sw_vs == hw_vs),

         /* FIXME: handle pipeline robustness "properly" */
         .prolog.robustness.level =
            (dev->vk.enabled_features.robustBufferAccess2 ||
             dev->vk.enabled_features.pipelineRobustness)
               ? AGX_ROBUSTNESS_D3D
            : dev->vk.enabled_features.robustBufferAccess
               ? AGX_ROBUSTNESS_GL
               : AGX_ROBUSTNESS_DISABLED,

         .prolog.robustness.soft_fault = agx_has_soft_fault(&dev->dev),
         .prolog.static_vi = !sw_vs->info.vs.use_prolog,
      };

      enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

      /* The launch path unrolls adjacency without GS, see hk_draw_without_adj */
      if (mesa_prim_has_adjacency(prim) &&
          !(agx_has_launch_programs(&dev->dev) && key.prolog.hw)) {
         if (draw.restart) {
            prim = u_decomposed_prim(prim);
         }

         key.prolog.adjacency = prim;
      }

      if (key.prolog.adjacency || !key.prolog.hw) {
         key.prolog.sw_index_size_B =
            draw.indexed ? agx_index_size_to_B(draw.index_size) : 0;
      }

      static_assert(sizeof(key.prolog.component_mask) ==
                    sizeof(sw_vs->info.vs.attrib_components_read));
      BITSET_COPY(key.prolog.component_mask,
                  sw_vs->info.vs.attrib_components_read);

      if (sw_vs->info.vs.use_prolog) {
         agx_fill_velem_keys(dyn->vi, sw_vs->info.vs.attribs_read,
                             key.prolog.attribs);

         u_foreach_bit(a, dyn->vi->attributes_valid) {
            unsigned slot =
               util_bitcount64(sw_vs->info.vs.attribs_read & BITFIELD_MASK(a));

            key.prolog.attribs[slot].stride =
               dyn->vi_binding_strides[dyn->vi->attributes[a].binding];
         }
      }

      hk_update_fast_linked(cmd, sw_vs, &key);
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty ||
       (gfx->dirty & HK_DIRTY_VB)) {

      unsigned slot = 0;
      u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
         if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];
            struct hk_addr_range vb = gfx->vb[attr.binding];
            enum pipe_format fmt = hk_format_to_pipe_format(attr.format);
            enum pipe_format interchange_format = agx_vbo_internal_format(fmt);
            unsigned interchange_align =
               util_format_get_blocksize(interchange_format);

            desc->root.draw.attrib_clamps[slot] = agx_calculate_vbo_clamp(
               vb.addr, fmt, vb.range, dyn->vi_binding_strides[attr.binding],
               attr.offset, &desc->root.draw.attrib_base[slot]);

            desc->root.draw.attrib_strides[slot] =
               dyn->vi_binding_strides[attr.binding] / interchange_align;
         } else {
            desc->root.draw.attrib_base[slot] = AGX_ZERO_PAGE_ADDRESS;
            desc->root.draw.attrib_clamps[slot] = 0;
            desc->root.draw.attrib_strides[slot] = 0;
         }

         ++slot;
      }

      desc->root_dirty = true;
   }

   if (vgt_dirty || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS) || IS_DIRTY(CB_BLEND_ADVANCED) ||
       desc->root_dirty /* for pipeline stats */ || true) {

      unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
      unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
      bool has_sample_mask = api_sample_mask != tib_sample_mask;

      if (hw_vs->info.cull_distance_array_size) {
         perf_debug(cmd, "Emulating cull distance (size %u)",
                    hw_vs->info.cull_distance_array_size);
      }

      if (has_sample_mask) {
         perf_debug(cmd, "Emulating sample mask");
      }

      unsigned samples_shaded = 0;
      if (fs->info.fs.epilog_key.sample_shading)
         samples_shaded = dyn->ms.rasterization_samples;

      struct hk_fast_link_key_fs key = {
         .prolog.statistics = hk_stat_enabled(hk_pipeline_stat_addr(
            cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),

         .prolog.cull_distance_size = hw_vs->info.cull_distance_array_size,
         .prolog.api_sample_mask = has_sample_mask ? api_sample_mask : 0xff,
         .nr_samples_shaded = samples_shaded,
      };

      bool prolog_discards = has_sample_mask || key.prolog.cull_distance_size;

      bool needs_prolog = key.prolog.statistics || prolog_discards;

      if (needs_prolog) {
         /* With late main shader tests, the prolog runs tests if neither the
          * main shader nor epilog will.
          *
          * With (nontrivial) early main shader tests, the prolog does not
          * run tests, the tests will run at the start of the main shader.
          * This ensures tests are after API sample mask and cull distance
          * discards.
          */
         key.prolog.run_zs_tests = !nontrivial_force_early &&
                                   !fs->b.info.writes_sample_mask &&
                                   !epilog_discards && prolog_discards;

         if (key.prolog.cull_distance_size) {
            key.prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
         }
      }

      key.epilog = (struct agx_fs_epilog_key){
         .link = fs->info.fs.epilog_key,
         .nr_samples = MAX2(dyn->ms.rasterization_samples, 1),
         .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
         .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
         .blend.logicop_enable = dyn->cb.logic_op_enable,
         .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
      };

      for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
         int map = dyn->cal.color_map[rt];
         key.epilog.remap[rt] = map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
      }

      if (dyn->ms.alpha_to_one_enable || dyn->ms.alpha_to_coverage_enable ||
          dyn->cb.logic_op_enable) {

         perf_debug(
            cmd, "Epilog with%s%s%s",
            dyn->ms.alpha_to_one_enable ? " alpha-to-one" : "",
            dyn->ms.alpha_to_coverage_enable ? " alpha-to-coverage" : "",
            dyn->cb.logic_op_enable ? " logic-op" : "");
      }

      key.epilog.link.already_ran_zs |= nontrivial_force_early;
      key.epilog.link.write_z |= late_zs;

      /* The prolog kills samples for the API sample mask and cull distances */
      key.epilog.link.kills |= needs_prolog && prolog_discards;

      struct hk_rendering_state *render = &cmd->state.gfx.render;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         key.epilog.rt_formats[i] =
            hk_format_to_pipe_format(render->color_att[i].vk_format);

         const struct vk_color_blend_attachment_state *cb =
            &dyn->cb.attachments[i];

         bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
         unsigned write_mask = write_enable ? cb->write_mask : 0;

         /* nir_lower_blend always blends, so use a default blend state when
          * blending is disabled at an API level.
          */
         if (!dyn->cb.attachments[i].blend_enable) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
            };
         } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .advanced_blend = 1,
               .mode = agx_pack_blend_advanced(
                  vk_advanced_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_overlap_to_pipe(cb->blend_overlap),
                  cb->src_premultiplied, cb->dst_premultiplied,
                  cb->clamp_results),
            };

            assert(cb->clamp_results == false);
         } else {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  vk_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_factor_to_pipe(cb->src_color_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
                  vk_blend_op_to_pipe(cb->alpha_blend_op),
                  vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
            };
         }
      }

      hk_update_fast_linked(cmd, fs, &key);
   }

   /* If the vertex shader uses draw parameters, vertex uniforms are dirty every
    * draw. Fragment uniforms are unaffected.
    *
    * For a direct draw, we upload the draw parameters as-if indirect to
    * avoid keying to indirectness.
    */
   if (gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr;

         if (draw.indexed) {
            gfx->draw_params +=
               offsetof(VkDrawIndexedIndirectCommand, vertexOffset);
         } else {
            gfx->draw_params += offsetof(VkDrawIndirectCommand, firstVertex);
         }
      } else {
         uint32_t params[] = {
            draw.indexed ? draw.index_bias : draw.start,
            draw.start_instance,
         };

         gfx->draw_params = hk_pool_upload(cmd, params, sizeof(params), 4);
      }
   } else {
      gfx->draw_params = 0;
   }

   if (sw_vs->b.info.uses_draw_id) {
      /* TODO: rodata? */
      gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, 2, 4);
   } else {
      gfx->draw_id_ptr = 0;
   }
}

static void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

   if (!vk_dynamic_graphics_state_any_dirty(dyn) && !gfx->dirty &&
       !gfx->descriptors.root_dirty && !gfx->shaders_dirty &&
       !sw_vs->b.info.uses_draw_id && !sw_vs->b.info.uses_base_param &&
       !(gfx->linked[MESA_SHADER_VERTEX] &&
         gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param))
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;

   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || gt_dirty;

   hk_flush_shader_parts(cmd, draw_id, draw);

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || gt_dirty) {
      enum mesa_prim prim = hk_rast_prim(cmd);

      gfx->topology = translate_hw_primitive_topology(prim);
      gfx->object_type = translate_object_type(prim);
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX)) {
      unsigned provoking;
      if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
         provoking = 2;
      else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
         provoking = 1;
      else
         provoking = 0;

      if (provoking != gfx->provoking) {
         gfx->provoking = provoking;
         gfx->dirty |= HK_DIRTY_PROVOKING;

         gfx->descriptors.root.draw.provoking = provoking;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    *
    * Note that we ignore dyn->ms.rasterization_samples when we do have a sample
    * count from an attachment. In Vulkan, these have to match anyway, but DX12
    * drivers are robust against this scenarios and vkd3d-proton will go out of
    * spec here. No reason we can't be robust here too.
    */
   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);

      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint32_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if (!cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      gfx->descriptors.root.draw.ppp_multisamplectl = ctrl;
      gfx->descriptors.root_dirty = true;
   }

   /* Link varyings before uploading tessellation state, becuase the
    * gfx->generate_primitive_id boolean needs to be plumbed.
    */
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   if ((gfx->dirty & HK_DIRTY_PROVOKING) || vgt_dirty || linked_fs_dirty) {
      unsigned bindings = linked_fs->b.cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         struct agx_ptr t = hk_pool_usc_alloc(cmd, linkage_size, 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = agx_usc_addr(&dev->dev, t.gpu);
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || linked_vs->sw_indexing) {
      desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
      desc->root_dirty = true;
   }

   struct agx_ptr tess_args = {0};
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      tess_args = hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      gfx->descriptors.root.draw.tess_params = tess_args.gpu;
      gfx->descriptors.root_dirty = true;
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
      gfx->descriptors.root.draw.geometry_params =
         hk_upload_geometry_params(cmd, draw);

      gfx->descriptors.root_dirty = true;
   }

   /* Root must be uploaded after the above, which touch the root */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

      /* Tess parameters depend on the root address, so we defer the upload
       * until after uploading root. But the root depends on the tess address,
       * so we allocate tess parameters before uploading root.
       *
       * This whole mechanism is a mess ported over from the GL driver. I'm
       * planning to do a massive rework of indirect geom/tess so I'm trying not
       * to perfectionism it in the mean time.
       */
      if (tess_args.cpu) {
         hk_upload_tess_params(cmd, tess_args.cpu, draw);
      }
   }

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   if ((gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
       IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty || vgt_dirty ||
       gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, hw_vs);

      agx_push(out, VDM_STATE, cfg) {
         cfg.vertex_shader_word_0_present = true;
         cfg.vertex_shader_word_1_present = true;
         cfg.vertex_outputs_present = true;
         cfg.vertex_unknown_present = true;
      }

      agx_push_packed(out, hw_vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;

      agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);
      }

      agx_push_packed(out, hw_vs->info.uvs.vdm, VDM_STATE_VERTEX_OUTPUTS);

      agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
         cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
         cfg.unknown_4 = cfg.unknown_5 = hk_rast_discard(cmd);
         cfg.generate_primitive_id = gfx->generate_primitive_id;
      }

      /* Pad up to a multiple of 8 bytes */
      memset(out, 0, 4);
      out += 4;
   }

   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, AGX_DEPTH_BIAS_LENGTH);

      hk_pack_depth_bias(cmd, ptr);
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      hk_flush_vp_state(cmd, cs, &out);
   }

   hk_flush_ppp_state(cmd, cs, &out);
   cs->current = out;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;
}

static bool
hk_needs_index_robustness(struct hk_cmd_buffer *cmd, struct agx_draw *draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw->indexed)
      return false;

   /* Geometry or tessellation use robust software index buffer fetch anyway */
   if (gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   /* Soft fault does not cover the hardware index buffer fetch, use the zero
    * page instead. Launch programs fetch zeroes from a pool allocation, see
    * hk_launch_draw_impl.
    */
   if (draw->index_buffer_range_B == 0) {
      if (!agx_has_launch_programs(&dev->dev)) {
         draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
         draw->index_buffer_range_B = 4;
         draw->start = 0;
      }
      return false;
   }

   /* The launch program index fetch reads only from the USC aperture and is
    * bounded in whole words. Copy index buffers placed outside the aperture,
    * and, for robustness, those whose binding ends inside a word.
    */
   if (agx_has_launch_programs(&dev->dev)) {
      /* Check the whole binding, not just its base: a direct draw's
       * firstIndex (or an indirect draw's GPU-provided firstIndex) may
       * otherwise move the hardware pointer past the end of the USC aperture.
       * Subtraction avoids wrapping the end of a high-VA binding.
       */
      if (draw->index_buffer < dev->dev.shader_base ||
          draw->index_buffer - dev->dev.shader_base >= (1ull << 32) ||
          draw->index_buffer_range_B >
             (1ull << 32) - (draw->index_buffer - dev->dev.shader_base))
         return true;

      if (!((draw->index_buffer + draw->index_buffer_range_B) & 3))
         return false;
   }

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   if (agx_is_indirect(draw->b))
      return true;

   return agx_direct_draw_overreads_indices(*draw);
}

/* Direct draws through software vertex shading size the vertex output buffer
 * on the CPU (hk_upload_vertex_params). Its 32-bit size wraps for very large
 * draws, e.g. a vertex count of UINT32_MAX, and the vertex shader would then
 * write past a small allocation. Such a buffer cannot be allocated anyway, so
 * record the out-of-memory error and skip the draw.
 */
static bool
hk_sw_vs_outputs_fit(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (agx_is_indirect(draw.b) || (!gfx->shaders[MESA_SHADER_GEOMETRY] &&
                                   !gfx->shaders[MESA_SHADER_TESS_EVAL]))
      return true;

   uint64_t vertex_B =
      util_bitcount64(hk_bound_sw_vs(gfx)->b.info.outputs) * 16;
   uint64_t vertices = (uint64_t)draw.b.count[0] * draw.b.count[1];

   if (!vertex_B || vertices <= UINT32_MAX / vertex_B)
      return true;

   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return false;
}

static void
build_passthrough_gs(nir_builder *b, UNUSED const struct agx_isa_target *target,
                     const void *key)
{
   poly_nir_passthrough_gs(b, key);
}

static void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(TESS_EVAL) &&
       !IS_SHADER_DIRTY(GEOMETRY) && !IS_DIRTY(IA_PRIMITIVE_TOPOLOGY))
      return;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB or adjacency */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   bool needed = xfb_outputs;
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
      needed |= gfx->pg_query[i] != 0;

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!needed) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   size_t key_size = poly_passthrough_gs_key_size(xfb_outputs);
   struct poly_passthrough_gs_key *key = alloca(key_size);

   *key = (struct poly_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   for (uint32_t i = 0; i < NUM_TOTAL_VARYING_SLOTS; i++) {
      unsigned num_comps = 0;
      switch (i) {
         case VARYING_SLOT_TESS_LEVEL_OUTER:
         case VARYING_SLOT_TESS_LEVEL_INNER:
         case VARYING_SLOT_BOUNDING_BOX1:
            num_comps = 4;
            break;
         default:
            num_comps = nir_slot_num_components(i, MESA_SHADER_VERTEX);
            break;
      }
      key->output_components[i] = num_comps ?: 4;

      /* We don't care about VS varying types in AGX, just set everything to
       * uint32 to improve cache hits. */
      key->output_types[i] = nir_type_uint32;
   }

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   gs = hk_meta_shader(dev, build_passthrough_gs, key, key_size);

   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

/* Bind the shaders a draw runs and flush push descriptors */
static void
hk_flush_shader_state(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       gfx->shaders[MESA_SHADER_TESS_CTRL])
      hk_merge_tess_info(cmd);

   /* TODO: Try to reduce draw overhead of this */
   hk_handle_passthrough_gs(cmd, draw);

   hk_flush_shaders(cmd);

   if (gfx->descriptors.push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, &gfx->descriptors);
}

static struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (!cs)
      return NULL;

   /* Annoyingly,
    * VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT is
    * render pass state on Imaginapple but draw state in Vulkan. In practice,
    * Proton never changes it within a render pass, but we technically need to
    * handle the switch regardless. Do so early since `cs` will be invalidated
    * if we need to split the render pass to switch representation mid-frame.
    *
    * Note we only do this dance with depth bias is actually enabled to avoid
    * senseless control stream splits with DXVK.
    */
   if ((IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(RS_DEPTH_BIAS_ENABLE)) &&
       dyn->rs.depth_bias.enable) {

      bool dbias_is_int =
         (dyn->rs.depth_bias.representation ==
          VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT) ||
         (gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM);

      /* Attempt to set dbias_is_int per the draw requirement. If this fails,
       * flush the control stream and set it on the new control stream.
       */
      bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
      if (!succ) {
         perf_debug(cmd, "Splitting control stream due to depth bias");

         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);

         succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }
   }

   cs = hk_occlusion_prepare_draw(cmd, cs);
   if (!cs)
      return NULL;

   hk_ensure_cs_has_space(cmd, cs, 0x2000 /* TODO */);

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   hk_flush_shader_state(cmd, draw);

   if (draw.restart || gfx->shaders[MESA_SHADER_GEOMETRY])
      hk_flush_index(cmd, cs);

   hk_flush_dynamic_state(cmd, cs, draw_id, draw);
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

   /* TODO: check if necessary, blob does this. G16G bounds the fetch by the
    * exact extent; rounding would expose an out-of-range narrow index.
    */
   if (!agx_has_usc_indices(&hk_cmd_buffer_device(cmd)->dev)) {
      cmd->state.gfx.index.buffer.range =
         align(cmd->state.gfx.index.buffer.range, 4);
   }
}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

/* Queue the input assembly counts of direct draws recorded since the last
 * flush. A dispatch per draw serializes the GPU on statistics queries that
 * span many draws, so they are summed when recorded instead, and queued when
 * the counters change and when the query ends.
 */
void
hk_flush_ia_stats(struct hk_cmd_buffer *cmd)
{
   struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
   const uint32_t counts[5] = {s->vertices, s->prims, s->vertices, s->clipped,
                               s->clipped};
   for (unsigned i = 0; i < ARRAY_SIZE(counts); ++i) {
      if (counts[i] && s->addr[i] && hk_stat_enabled(s->addr[i])) {
         struct libagx_imm_write add = {.address = s->addr[i],
                                        .value = counts[i]};
         util_dynarray_append(&cmd->query_adds, add);
      }
   }
   memset(s, 0, sizeof(*s));
}

/* Update the input assembly, vertex shader and clipper statistics of a draw */
static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   uint64_t ia_vertices = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);
   uint64_t ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);
   uint64_t vs_invocations = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);
   uint64_t c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
   uint64_t c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   if (!hk_stat_enabled(ia_vertices) && !hk_stat_enabled(ia_prims) &&
       !hk_stat_enabled(vs_invocations) && !hk_stat_enabled(c_inv) &&
       !hk_stat_enabled(c_prims))
      return;

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
   uint32_t patch_size = dyn->ts.patch_control_points;

   if (!agx_is_indirect(draw.b) && !draw.restart) {
      struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
      const uint64_t addr[5] = {ia_vertices, ia_prims, vs_invocations, c_prims,
                                c_inv};
      if (memcmp(s->addr, addr, sizeof(addr))) {
         hk_flush_ia_stats(cmd);
         memcpy(s->addr, addr, sizeof(addr));
      }
      uint32_t prims = poly_decomposed_prims_for_vertices_with_tess(
                          prim, draw.b.count[0], patch_size) *
                       draw.b.count[1];
      s->vertices += draw.b.count[0] * draw.b.count[1];
      s->prims += prims;
      /* Clipper counters are written with the geometry or tessellation
       * output when those stages are used. */
      if (!geom && !tess)
         s->clipped += prims;
      return;
   }

   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      /* Direct draws pass their own indices (agx_draw_index_buffer) with a
       * range relative to the first index, so the descriptor's first index
       * (VkDrawIndexedIndirectCommand word 2) is 0. */
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used.
    */
   if (geom || tess) {
      c_prims = 0;
      c_inv = 0;
   }

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);

      libagx_increment_ia_restart(
         cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices, ia_prims,
         vs_invocations, c_prims, c_inv, draw_ptr, agx_draw_index_buffer(draw),
         agx_draw_index_range_el(draw), cmd->state.gfx.index.restart,
         index_size_B, prim, patch_size);
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

/* Per-draw preparation of launch programs, shared by every view */
static bool
hk_launch_begin_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id,
                     struct agx_draw *draw)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (!hk_feedback_prepare_draw(cmd))
      return false;
   /* Index fetch and indirect arguments are buffer reads of tiling. */
   if (!hk_launch_prepare_draw(cmd,
                               agx_is_indirect(draw->b) || draw->indexed))
      return false;
   if (cmd->feedback_raw.size) {
      for (unsigned stage = 0; stage <= MESA_SHADER_MESH; stage++) {
         if (cmd->state.gfx.shaders[stage] &&
             cmd->state.gfx.shaders[stage]->writes_memory) {
            hk_feedback_invalidate(cmd);
            break;
         }
      }
   }
   hk_ia_update(cmd, *draw);
   hk_flush_shader_state(cmd, *draw);
   if (!hk_sw_vs_outputs_fit(cmd, *draw))
      return false;
   /* Geometry and tessellation emulation decompose fans themselves. When
    * nothing observes the provoking vertex, the fan is drawn natively from
    * its first vertex, see hk_launch_draw_impl. */
   cmd->state.gfx.fan_unrolled =
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT &&
      !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
      !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
      hk_provoking_observed(cmd);
   if (cmd->state.gfx.fan_unrolled)
      *draw = hk_unroll_fan(cmd, *draw);
   /* Software stages read the draw ID from the root table they upload,
    * when one of the bound stages they run reads it. */
   if (hk_stages_read_draw_id(&cmd->state.gfx) &&
       cmd->state.gfx.descriptors.root.draw.draw_id != draw_id) {
      cmd->state.gfx.descriptors.root.draw.draw_id = draw_id;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/* Only geometry shaders observe adjacent vertices. Without one, draw the
 * primitives of the remaining vertices.
 */
static struct agx_draw
hk_draw_without_adj(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      prim = u_decomposed_prim(prim);
   }

   /* Launch programs draw a generated index list of the remaining vertices,
    * where the vertex prolog remaps the vertex ID otherwise.
    */
   if (agx_has_launch_programs(&dev->dev)) {
      uint64_t in_draw = draw.b.ptr;
      if (!agx_is_indirect(draw.b)) {
         uint32_t desc[5] = {draw.b.count[0], draw.b.count[1]};
         if (draw.indexed) {
            desc[2] = draw.start;
            desc[3] = draw.index_bias;
            desc[4] = draw.start_instance;
         } else {
            desc[2] = draw.start;
            desc[3] = draw.start_instance;
         }
         in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
      }

      uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
      uint64_t heap = hk_heap(cmd);
      libagx_unroll_adjacency(
         cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, heap, in_draw,
         draw.indexed ? draw.index_buffer : 0,
         draw.indexed ? draw.index_buffer_range_B >> draw.index_size : 0,
         draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim,
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);

      return hk_heap_indexed_indirect(cmd, out_draw, AGX_INDEX_SIZE_U32, false);
   }

   if (agx_is_indirect(draw.b)) {
      const size_t size = sizeof(VkDrawIndexedIndirectCommand);
      static_assert(sizeof(VkDrawIndexedIndirectCommand) >
                       sizeof(VkDrawIndirectCommand),
                    "allocation size is conservative");

      uint64_t out_draw = hk_pool_alloc(cmd, size, 4).gpu;
      struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

      libagx_draw_without_adj(
         cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, draw.b.ptr,
         desc->root.draw.vertex_params, draw.index_buffer,
         draw.indexed ? agx_draw_index_range_el(draw) : 0,
         draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim);

      return agx_draw_indirect(out_draw);
   } else {
      unsigned count = libagx_remap_adj_count(draw.b.count[0], prim);

      return (struct agx_draw){
         .b = agx_3d(count, draw.b.count[1], 1),
      };
   }
}

static void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw_)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   bool launch = agx_has_launch_programs(&dev->dev);

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;

   if (launch && !hk_launch_begin_draw(cmd, draw_id, &draw_))
      return;

   uint64_t stat_c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);

   uint64_t stat_c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      struct hk_cs *cs = NULL, *ccs = NULL;
      uint8_t *out = NULL;

      if (launch) {
         hk_flush_shader_parts(cmd, draw_id, draw);
      } else {
         cs = hk_flush_gfx_state(cmd, draw_id, draw);
         /* If we failed to allocate a control stream, we've already lost the
          * device. Just drop the draw so we don't crash.
          */
         if (!cs)
            return;
      }

      bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      bool needs_idx_robust = hk_needs_index_robustness(cmd, &draw);
      bool adj = mesa_prim_has_adjacency(
         vk_topology_to_mesa(dyn->ia.primitive_topology));
      adj &= !geom;
      needs_idx_robust &= !adj;

      if (!launch) {
         out = cs->current;
         assert(cs->current + 0x1000 < cs->end);

         if (tess && HK_PERF(dev, NOTESS))
            continue;

         cs->stats.calls++;

         if (geom || tess) {
            ccs = hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx,
                                               true, HK_SYNC_ALL);
            if (!ccs)
               return;
         }

         hk_ia_update(cmd, draw);
      }

      if (tess) {
         draw = hk_launch_tess(cmd, ccs, draw, geom ? 0 : stat_c_prims,
                               geom ? 0 : stat_c_inv);
      }

      if (geom) {
         draw = hk_launch_gs_prerast(cmd, ccs, draw);
      }

      if (adj) {
         assert(!geom && "geometry shaders handle adj directly");
         draw = hk_draw_without_adj(cmd, draw);
      }

      if (launch) {
         if (needs_idx_robust)
            draw = hk_robust_index_copy(cmd, draw);

         hk_launch_draw(cmd, draw_id, draw);
         continue;
      }

      enum agx_primitive topology = cmd->state.gfx.topology;
      if (needs_idx_robust) {
         assert(!geom && !tess && !adj);
         perf_debug(cmd, "lowering robust index buffer");

         cs->current = out;

         draw = hk_draw_as_indexed_indirect(cmd, draw);

         size_t size_B = libagx_draw_robust_index_vdm_size();
         uint64_t target = hk_cs_alloc_for_indirect(cs, size_B);

         libagx_draw_robust_index(cmd, agx_1d(32), AGX_BARRIER_ALL | AGX_PREGFX,
                                  target, hk_heap(cmd), draw.b.ptr,
                                  draw.index_buffer, draw.index_buffer_range_B,
                                  draw.restart, topology, draw.index_size);
      } else {
         cs->current = (void *)agx_vdm_draw((uint32_t *)out, dev->dev.chip,
                                            draw, topology);
      }

      if (cs->primitive_count < AGX_PRIMITIVE_COUNT_UNKNOWN) {
         agx_add_primitive_count(
            &cs->primitive_count,
            agx_is_indirect(draw.b) || draw.restart || geom || tess ||
                  needs_idx_robust
               ? AGX_PRIMITIVE_COUNT_UNKNOWN
               : agx_count_primitives(topology, draw.b.count[0],
                                      draw.b.count[1]));
      }
      cs->stats.cmds++;
   }
}

static void
hk_init_mesh_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   size_t size = 512 * 1024 * 1024;
   dev->mesh_heap = agx_bo_create(&dev->dev, size, 0, 0, "Mesh heap");

   off_t off = dev->rodata.mesh_heap - dev->rodata.bo->va->addr;
   struct poly_heap *map = agx_bo_map(dev->rodata.bo) + off;
   *map = (struct poly_heap){
      .base = dev->mesh_heap ? dev->mesh_heap->va->addr : 0,
      .size = dev->mesh_heap ? size : 0,
   };
}

/* Records of mesh shader draws sized on the GPU are allocated from a heap,
 * freed after each command buffer.
 */
static uint64_t
hk_mesh_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->mesh_heap_init_once, hk_init_mesh_heap, cmd);
   if (!dev->mesh_heap)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   if (!cmd->uses_mesh_heap) {
      hk_queue_write(cmd,
                     dev->rodata.mesh_heap + offsetof(struct poly_heap, bottom),
                     0, true /* after gfx */);
      cmd->uses_mesh_heap = true;
   }

   return dev->rodata.mesh_heap;
}

static uint64_t
hk_mesh_stat(struct hk_cmd_buffer *cmd, VkQueryPipelineStatisticFlagBits stat)
{
   uint64_t addr = hk_pipeline_stat_addr(cmd, stat);
   return hk_stat_enabled(addr) ? addr : 0;
}

/* Mesh shader draws run the task and mesh workgroups as compute kernels, and
 * then draw every primitive slot of the mesh workgroups' output records. The
 * workgroup grid is either given (direct) or read from grid_addr. For draws
 * from a count buffer, draws past the count at count_addr are empty.
 */
static void
hk_launch_draw_mesh(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    const uint32_t *direct, uint64_t grid_addr,
                    uint64_t count_addr)
{
   /* Static pipeline feedback may change after BeginRendering, and a failed
    * decompression poisons recording. */
   if (!hk_feedback_prepare_draw(cmd))
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   if (direct && (!direct[0] || !direct[1] || !direct[2]))
      return;

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];
   struct hk_api_shader *task = gfx->shaders[MESA_SHADER_TASK];
   if (!mesh)
      return;

   if (!hk_launch_prepare_draw(cmd, !direct || count_addr))
      return;
   struct hk_shader *main = hk_only_variant(mesh);
   struct hk_shader *task_main = task ? hk_only_variant(task) : NULL;
   const struct hk_mesh_info *info = &main->info.mesh;
   unsigned vertices =
      MAX2(info->max_primitives, 1) * mesa_vertices_per_prim(info->prim);

   struct hk_mesh_rast_key rast_key;
   memset(&rast_key, 0, sizeof(rast_key));
   rast_key.info = *info;
   rast_key.clip_distance_array_size = main->info.clip_distance_array_size;
   rast_key.cull_distance_array_size = main->info.cull_distance_array_size;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_api_shader *rast =
      hk_meta_shader(dev, hk_nir_mesh_rast, &rast_key, sizeof(rast_key));
   if (!rast) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   if (gfx->mesh_rast != rast) {
      gfx->mesh_rast = rast;
      gfx->shaders_dirty |= BITFIELD_BIT(MESA_SHADER_VERTEX);
   }

   hk_flush_shaders(cmd);
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   const struct agx_compute_profile *profile =
      &main->only_linked->b.program.profile;

   /* Draws without a task shader and with a known grid are set up here */
   bool cpu_setup = direct && !task;
   uint64_t direct_count =
      direct ? (uint64_t)direct[0] * direct[1] * direct[2] : 0;
   uint64_t records_B = direct_count * info->group_stride_B;
   if (cpu_setup && records_B > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   if (direct && !cpu_setup)
      grid_addr = hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
   /* Upload failed: the command buffer is in error. */
   if (!cpu_setup && !grid_addr)
      return;

   desc->root.draw.draw_id = draw_id;

   /* Compute before graphics runs ahead of the whole render, so every
    * draw has its own records.
    */
   hk_foreach_view(cmd) {
      struct agx_ptr params =
         hk_pool_alloc(cmd, sizeof(struct libagx_mesh_params), 16);
      if (!params.gpu)
         return;

      struct libagx_mesh_params *p = params.cpu;
      *p = (struct libagx_mesh_params){
         .mesh_primitives = gfx->mesh_pg_query,
         .mesh_invocations = hk_mesh_stat(
            cmd, VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT),
         .task_invocations = hk_mesh_stat(
            cmd, VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT),
      };
      if (cpu_setup) {
         p->outputs = hk_pool_alloc(cmd, records_B, 16).gpu;
         p->mesh_count = direct_count;
      }
      for (unsigned v = 0; v < ARRAY_SIZE(p->primitive_varying_B); ++v) {
         p->primitive_varying_B[v] = hk_mesh_slot_offset_B(
            info, info->primitive_slots, VARYING_SLOT_VAR0 + v);
      }

      desc->root.draw.mesh_params = params.gpu;
      desc->root_dirty = true;
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

      if (!cpu_setup) {
         libagx_mesh_setup(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                           params.gpu, grid_addr, count_addr, draw_id,
                           hk_mesh_heap(cmd), task != NULL,
                           task_main ? task_main->info.task.payload_B : 0,
                           info->group_stride_B, vertices);
      }

      if (task_main) {
         hk_launch_prerast(
            cmd, task_main, task_main->only_linked,
            agx_grid_indirect(params.gpu + offsetof(struct libagx_mesh_params,
                                                    task_dispatch)));
         libagx_mesh_prefix(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                            params.gpu, hk_mesh_heap(cmd), info->group_stride_B,
                            vertices);
      }

      struct agx_draw draw;
      if (cpu_setup) {
         hk_launch_prerast(cmd, main, main->only_linked,
                           agx_3d(direct[0] * profile->local_size[0],
                                  direct[1] * profile->local_size[1],
                                  direct[2] * profile->local_size[2]));
         draw = (struct agx_draw){.b = agx_3d(direct_count * vertices, 1, 1)};
      } else {
         hk_launch_prerast(
            cmd, main, main->only_linked,
            agx_grid_indirect(params.gpu + offsetof(struct libagx_mesh_params,
                                                    mesh_dispatch)));
         draw = agx_draw_indirect(params.gpu +
                                  offsetof(struct libagx_mesh_params, draw));
      }

      hk_flush_shader_parts(cmd, draw_id, draw);
      hk_launch_draw(cmd, draw_id, draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                       uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};

   hk_launch_draw_mesh(cmd, 0, groups, 0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, uint32_t drawCount,
                               uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   for (uint32_t i = 0; i < drawCount; ++i) {
      hk_launch_draw_mesh(cmd, i, NULL,
                          hk_buffer_address_ro(buffer, offset + i * stride), 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer,
                                    VkBuffer _buffer, VkDeviceSize offset,
                                    VkBuffer countBuffer,
                                    VkDeviceSize countBufferOffset,
                                    uint32_t maxDrawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   uint64_t count = hk_buffer_address_ro(count_buffer, countBufferOffset);
   for (uint32_t i = 0; i < maxDrawCount; ++i) {
      hk_launch_draw_mesh(cmd, i, NULL,
                          hk_buffer_address_ro(buffer, offset + i * stride),
                          count);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;

   if (HK_TEST_INDIRECTS) {
      uint32_t data[] = {
         vertexCount,
         instanceCount,
         firstVertex,
         firstInstance,
      };

      draw = agx_draw_indirect(hk_pool_upload(cmd, data, sizeof(data), 4));
   } else {
      draw = (struct agx_draw){
         .b = agx_3d(vertexCount, instanceCount, 1),
         .start = firstVertex,
         .start_instance = firstInstance,
      };
   }

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;

   if (HK_TEST_INDIRECTS && draw_id == 0) {
      uint32_t data[] = {
         indexCount, instanceCount, firstIndex, vertexOffset, firstInstance,
      };
      uint64_t addr = hk_pool_upload(cmd, data, sizeof(data), 4);

      draw = agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0);
   } else {
      draw =
         agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                          firstInstance, buf.addr, buf.range, 0, 0);
   }

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const uint32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw(cmd, draw_id, agx_draw_indirect(addr));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw(cmd, draw_id,
              agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");

   assert((stride % 4) == 0 && "aligned");

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, counter, counterBuffer);

   uint64_t indirect = hk_pool_alloc(cmd, sizeof(VkDrawIndirectCommand), 4).gpu;
   libagx_draw_xfb(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, indirect,
                   hk_buffer_address_ro(counter, counterBufferOffset),
                   counterOffset, vertexStride, instanceCount, firstInstance);
   hk_draw_indirect_inner(commandBuffer, indirect, 1, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   /* A null pCounterBuffers means no counter buffers. */
   for (unsigned i = 0; pCounterBuffers && i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   UNREACHABLE("stub");
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   UNREACHABLE("stub");
}
