/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "asahi/compiler/agx_nir.h"
#include "asahi/lib/agx_apple9.h"
#include "asahi/lib/agx_apple9_color.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "asahi/lib/agx_nir_apple9_tilebuffer.h"
#include "pipe/p_defines.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/u_prim.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "vk_util.h"

static struct agx_ptr
hk_apple9_usc_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_usc_alloc(data, size, alignment);
}

/* Bytes of the tile allocation of a layout, before its alignment */
static unsigned
hk_apple9_tile_bytes(const struct agx_tilebuffer_layout *tib)
{
   unsigned bytes = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(tib->logical_format); ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE && !tib->spilled[rt])
         bytes += 4 * agx_apple9_color_words(tib->logical_format[rt]);
   }
   return bytes;
}

/* Attachment textures for the background program, rt at texture 2 * rt */
static uint64_t
hk_apple9_render_sysvals(struct hk_cmd_buffer *cmd, bool layered)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   unsigned samples = MAX2(render->tilebuffer.nr_samples, 1);

   alignas(64) uint8_t sysvals[AGX_APPLE9_RUNTIME_SYSVAL_SIZE];
   hk_apple9_compute_sysvals(dev, sysvals);

   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view)
         continue;

      uint32_t info[5];
      memcpy(info, view->apple9_render[layered].dimensions, 16);
      info[4] = samples;
      memcpy(sysvals + AGX_APPLE9_TEXTURE_INFO_OFFSET +
                (2 * rt) * AGX_APPLE9_TEXTURE_INFO_STRIDE,
             info, sizeof(info));
   }

   return hk_pool_upload(cmd, sysvals, sizeof(sysvals), 64);
}

/* Target rt: texture 2 * rt, image rt, clear colour in push words 4 + 8 * rt */
struct hk_bg_eot
hk_apple9_launch_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                        struct agx_bg_eot_key *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   bool layered = key->tib.layered;
   VkResult error = VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct agx_ptr textures =
      hk_pool_alloc(cmd, 2 * HK_MAX_RTS * AGX_APPLE9_TEXTURE_TABLE_STRIDE, 64);
   if (!textures.cpu)
      goto fail;

   memset(textures.cpu, 0, 2 * HK_MAX_RTS * AGX_APPLE9_TEXTURE_TABLE_STRIDE);
   uint8_t push[2 * (4 + 8 * HK_MAX_RTS)] = {0};
   unsigned uniforms = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const struct hk_image_view *view =
         rt < render->color_att_count ? render->color_att[rt].iview : NULL;
      uint8_t *entry = textures.cpu;

      if (key->op[rt] == AGX_BG_LOAD) {
         if (!view || !view->apple9_render[layered].texture_valid)
            goto unsupported;

         memcpy(entry + (2 * rt) * AGX_APPLE9_TEXTURE_TABLE_STRIDE,
                view->apple9_render[layered].texture,
                AGX_APPLE9_TEXTURE_TABLE_STRIDE);
      } else if (key->op[rt] == AGX_BG_CLEAR) {
         memcpy(push + 2 * (4 + 8 * rt),
                &info->pColorAttachments[rt].clearValue, 16);
         uniforms = MAX2(uniforms, 4 + 8 * rt + 8);
      } else if (key->op[rt] == AGX_EOT_STORE) {
         void *store = entry + rt * AGX_APPLE9_TEXTURE_TABLE_STRIDE;

         /* Feedback-raw attachments store uncompressed, unlike the prepacked */
         if (view && (render->raw_color_mask & BITFIELD_BIT(rt))) {
            if (!hk_apple9_image_view_eot(view, layered, true, store))
               goto unsupported;
         } else if (view && view->apple9_render[layered].store_valid) {
            memcpy(store, view->apple9_render[layered].store,
                   AGX_APPLE9_TEXTURE_TABLE_STRIDE);
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

   const struct agx_apple9_compute_profile *p = &shader->info.apple9_compute;
   struct agx_apple9_render_stage stage = {
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

   uint64_t resources[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   for (unsigned i = 0; i < p->resource_binding_count; ++i) {
      if (p->resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_UBO ||
          p->resource_binding[i] != AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
         goto unsupported;

      resources[i] = hk_apple9_render_sysvals(cmd, layered);
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
   if (!agx_apple9_prepare_graphics_stage(
          &dev->dev, cmd, hk_apple9_usc_alloc, &stage, true, table,
          textures.gpu, dev->apple9_fetch_sampler->va->addr, push_addr,
          agx_apple9_tile_sample_bytes(hk_apple9_tile_bytes(&key->tib),
                                       samples),
          samples, &launch))
      goto fail;

   return (struct hk_bg_eot){.usc = launch};

unsupported:
   error = VK_ERROR_FEATURE_NOT_PRESENT;
fail:
   vk_command_buffer_set_error(&cmd->vk, error);
   return (struct hk_bg_eot){0};
}

static_assert(HK_MAX_VIEWPORTS == AGX_APPLE9_MAX_VIEWPORTS,
              "Vulkan viewport limit matches the PPP viewport block");

static struct agx_ptr
hk_apple9_context_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_context_alloc(data, size, alignment);
}

static enum agx_primitive
hk_apple9_rast_primitive(enum mesa_prim prim)
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

/* Split before recording any of a draw's pre-rasterization work. Ending a
 * graphics stream also closes its pre/post compute streams and heap lifetime.
 * Reserve for every emulated multiview draw so these stay together.
 */
/* Sample positions for a draw: the application's when enabled for this
 * sample count, the standard ones otherwise. */
static uint32_t
hk_apple9_draw_sample_positions(const struct vk_dynamic_graphics_state *dyn,
                                unsigned samples)
{
   return dyn->ms.sample_locations_enable && dyn->ms.sample_locations &&
                dyn->ms.sample_locations->per_pixel == samples
             ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
             : hk_default_sample_positions(samples);
}

bool
hk_apple9_prepare_draw(struct hk_cmd_buffer *cmd, bool reads_buffers)
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
      uint32_t positions = hk_apple9_draw_sample_positions(dyn, samples);
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
hk_apple9_set_index_fetch(struct agx_apple9_render_pipeline *pipeline,
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
hk_apple9_emit_ordinary_draw(struct hk_cs *cs,
                             const struct agx_apple9_render_pipeline *pipeline,
                             struct agx_draw draw)
{
   if (agx_is_indirect(draw.b))
      cs->current =
         agx_apple9_emit_indirect_draw(cs->current, pipeline, draw.b.ptr) - 4;
   else
      cs->current = agx_apple9_emit_direct_draw(
                       cs->current, pipeline, draw.b.count[0], draw.b.count[1],
                       draw.indexed ? draw.index_bias : draw.start) -
                    4;
}

/* Record the final draw packet after its resources and pipeline are prepared. */
static void
hk_apple9_record_draw(struct hk_cs *cs,
                      const struct agx_apple9_render_pipeline *pipeline,
                      struct agx_draw draw)
{
   hk_apple9_emit_ordinary_draw(cs, pipeline, draw);
}

static void
hk_apple9_account_draw(struct hk_cs *cs,
                       const struct agx_apple9_render_pipeline *pipeline,
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
bool
hk_apple9_reads_draw_id(const struct hk_graphics_state *gfx)
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
hk_apple9_push_words(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                     const struct hk_linked_shader *linked, uint64_t root)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_root_descriptor_table *table =
      s->info.stage == MESA_SHADER_COMPUTE ? &cmd->state.cs.descriptors.root
                                           : &gfx->descriptors.root;
   uint16_t push[AGX_APPLE9_MAX_PUSH_WORDS * 2] = {0};
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
hk_apple9_link_stages(struct hk_cmd_buffer *cmd, const struct hk_shader *vs,
                      const struct hk_linked_shader *vs_linked,
                      const struct hk_shader *fs,
                      const struct hk_linked_shader *fs_linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_unlinked_uvs_layout *uvs = &vs->info.uvs;
   struct agx_apple9_render_stage *vertex = &cmd->apple9_stage[0];
   struct agx_apple9_render_stage *fragment = &cmd->apple9_stage[1];

   *vertex = vs_linked->apple9_stage;
   *fragment = fs_linked->apple9_stage;

   /* Varying groups as agx_assign_uvs orders them */
   uint64_t flat =
      (fs->info.fs.interp.flat & BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)) |
      VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT;
   uint64_t linear = fs->info.fs.interp.linear |
                     BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2);

   struct agx_apple9_varying_layout layout = {.count = uvs->user_size};
   u_foreach_bit64(loc, uvs->written) {
      layout.mask[loc] = BITFIELD_MASK(uvs->components[loc]);
      layout.group[loc] = agx_apple9_varying_group(flat, linear, loc);
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

      if (bindings) {
         agx_link_varyings_vs_fs(t.cpu, &gfx->linked_varyings, uvs->user_size,
                                 (struct agx_varyings_fs *)&fs_linked->b.cf, pv,
                                 0, &primitive_id);
      } else {
         agx_pack(t.cpu, CF_BINDING_HEADER, cfg) {
            cfg.number_of_32_bit_slots = 1;
         }
      }

      fragment->coefficients[pv] = agx_usc_addr(&dev->dev, t.gpu);
   }

   fragment->coefficient_count = bindings;
   fragment->reads_primitive_id = primitive_id;
   fragment->apple9_reads_z = fs_linked->b.cf.reads_z;

   return agx_apple9_link_render_pipeline(&cmd->apple9_linked, vertex,
                                          fragment);
}

/* Reuse preparation, not emission: every draw still records its scope and
 * emits its ordinary VDM command. Root updates preserve the normal upload
 * order and rebuild only the tables/launches that bind the changed root.
 */
static bool
hk_apple9_try_cached_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                          struct agx_draw draw)
{
   if (!hk_apple9_draw_state_reusable(cmd, draw))
      return false;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_apple9_render_pipeline *pipeline =
      &cmd->apple9_draw_state.pipeline;
   if (draw.indexed) {
      uint64_t offset =
         (uint64_t)draw.start * agx_index_size_to_B(draw.index_size);
      uint64_t address = draw.index_buffer + offset;
      if (address < dev->dev.shader_base ||
          address >= dev->dev.shader_base + (1ull << 32))
         return false;
      hk_apple9_set_index_fetch(pipeline, draw, draw.index_buffer,
                                draw.index_buffer_range_B, offset);
      pipeline->primitive_restart =
         cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable;
      pipeline->restart_index = gfx->index.restart;
   } else {
      pipeline->index_size = 0;
      pipeline->primitive_restart = false;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   assert(cs && cs == cmd->apple9_prev_cs);

   if (hk_apple9_reads_draw_id(gfx) && desc->root.draw.draw_id != draw_id) {
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
         struct agx_apple9_uniform_draw *record =
            &cmd->apple9_draw[!cmd->apple9_draw_current];

         /* The push words hold the root address */
         const uint64_t tables[2] = {record->vertex_table,
                                     record->fragment_table};
         const uint64_t push[2] = {
            hk_apple9_push_words(cmd, cmd->apple9_draw_state.shader[0],
                                 cmd->apple9_draw_state.linked[0], root),
            hk_apple9_push_words(cmd, cmd->apple9_draw_state.shader[1],
                                 cmd->apple9_draw_state.linked[1], root),
         };
         if (!push[0] || !push[1])
            return true; /* The pool recorded the allocation failure. */

         if (!agx_apple9_update_draw_tables_alloc(
                &dev->dev, cmd, hk_apple9_usc_alloc, cmd,
                hk_apple9_context_alloc, pipeline, record, tables, push)) {
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
   hk_apple9_emit_ordinary_draw(cs, pipeline, draw);
   hk_apple9_account_draw(cs, pipeline, draw, false);
   return true;
}

static void
hk_apple9_draw_impl(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    struct agx_draw draw)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* Geometry shaders can produce empty rasterization draws. */
   if (!agx_is_indirect(draw.b) && (!draw.b.count[0] || !draw.b.count[1]))
      return;
   if (hk_apple9_try_cached_draw(cmd, draw_id, draw))
      return;
   cmd->apple9_draw_state.valid = false;
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
         dyn->vp.viewport_count > AGX_APPLE9_MAX_VIEWPORTS ||
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
   if (cmd->apple9_linked_id[0] != linked[0]->apple9_stage.program_id ||
       cmd->apple9_linked_id[1] != linked[1]->apple9_stage.program_id) {
      if (!hk_apple9_link_stages(cmd, shaders[0], linked[0], shaders[1],
                                 linked[1])) {
         cmd->apple9_linked_id[0] = cmd->apple9_linked_id[1] = 0;
         UNSUPPORTED("the vertex and fragment programs cannot be linked");
      }
      cmd->apple9_linked_id[0] = linked[0]->apple9_stage.program_id;
      cmd->apple9_linked_id[1] = linked[1]->apple9_stage.program_id;
      agx_apple9_build_ppp_static(&cmd->apple9_linked, cmd->apple9_ppp_static);
      cmd->apple9_linked.ppp_static = cmd->apple9_ppp_static;
   }
   const struct agx_apple9_render_stage *fragment = &cmd->apple9_stage[1];
   struct agx_apple9_render_pipeline pipeline = cmd->apple9_linked;
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
   cmd->apple9_stage[1].render_targets = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(pipeline.color_formats); ++rt) {
      enum pipe_format format = gfx->render.tilebuffer.logical_format[rt];
      if (format != PIPE_FORMAT_NONE && !gfx->render.tilebuffer.spilled[rt])
         pipeline.color_formats[cmd->apple9_stage[1].render_targets++] = format;
   }
   static const uint8_t primitives[] = {
      AGX_PRIMITIVE_POINTS,         AGX_PRIMITIVE_LINES,
      AGX_PRIMITIVE_LINE_STRIP,     AGX_PRIMITIVE_TRIANGLES,
      AGX_PRIMITIVE_TRIANGLE_STRIP, AGX_PRIMITIVE_TRIANGLE_FAN,
   };
   /* Mesa meta rectangles are already expanded to two ordinary triangles. */
   pipeline.primitive = emulated ? hk_apple9_rast_primitive(gs_prim)
                        : dyn->ia.primitive_topology ==
                                 VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA ||
                              gfx->apple9_fan_unrolled
                           ? AGX_PRIMITIVE_TRIANGLES
                           : primitives[dyn->ia.primitive_topology];
   /* Emulated stages emit their primitives for the same provoking mode.
    * Fans that are not unrolled have no observable provoking vertex, and
    * native fans flat shade consistently only from their first vertex. */
   const bool native_fan =
      !mesh && !gs && !tes &&
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
      !gfx->apple9_fan_unrolled;
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
      if (pipeline.index_buffer < dev->dev.shader_base ||
          pipeline.index_buffer >= dev->dev.shader_base + (1ull << 32))
         UNSUPPORTED("the index buffer is outside the shader window");
      hk_apple9_set_index_fetch(&pipeline, draw, index_buffer, index_range,
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
   gfx->apple9_sample_positions =
      hk_apple9_draw_sample_positions(ms_dyn, pipeline.samples);
   /* hk_apple9_prepare_draw committed the render's positions; meta draws
    * only fill them in for a render that has none yet. */
   if (!cs->has_sample_locations)
      cs->ppp_multisamplectl = gfx->apple9_sample_positions;
   hk_reserve_scratch(cmd, cs, shaders[0]);
   hk_reserve_scratch(cmd, cs, shaders[1]);
   struct agx_apple9_uniform_draw *record =
      &cmd->apple9_draw[cmd->apple9_draw_current];
   agx_apple9_uniform_draw_reset(record);
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
   struct agx_scissor_packed scissors[AGX_APPLE9_MAX_VIEWPORTS];
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
       * their own depth exports (hk_apple9_clamp_depth_export). */
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
      bool exact = dyn->rs.depth_bias.exact;
      bool unorm = d16 || force_unorm;
      /* UNORM representations use a fixed depth unit: 2^-16 for D16 and
       * 2^-24 when forced, or twice that unless exact. Floating depth uses
       * the exponent-dependent unit 2^(e - 23) directly. The submission
       * selects the matching mode.
       */
      float scale = d16           ? 1.0f / (exact ? (1 << 16) : (1 << 15))
                    : force_unorm ? 1.0f / (exact ? (1 << 24) : (1 << 23))
                                  : 1.0f;
      assert(cs->cr.dbias_is_int == (unorm ? U_TRISTATE_YES : U_TRISTATE_NO));
      struct agx_depth_bias_packed depth_bias;
      agx_pack(&depth_bias, DEPTH_BIAS, cfg) {
         cfg.depth_bias = dyn->rs.depth_bias.constant_factor * scale;
         cfg.slope_scale = dyn->rs.depth_bias.slope_factor;
         cfg.clamp = dyn->rs.depth_bias.clamp;
      }
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
    * (hk_apple9_clamp_depth_export). */
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
      agx_pack(&record->stencil[face], FRAGMENT_STENCIL, cfg) {
         cfg.compare = stencil ? st->op.compare : AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = stencil ? st->write_mask : 0xff;
         cfg.read_mask = stencil ? st->compare_mask : 0xff;
         cfg.depth_pass = stencil ? st->op.pass : AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = stencil ? st->op.depth_fail : AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = stencil ? st->op.fail : AGX_STENCIL_OP_KEEP;
      }
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
   gfx->descriptors.root.draw.ppp_multisamplectl = gfx->apple9_sample_positions;
   gfx->descriptors.root.draw.samples_log2 = util_logbase2(pipeline.samples);
   gfx->descriptors.root.draw.clip_z_coeff =
      dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
   gfx->descriptors.root.draw.rasterized_prim =
      (points  ? 2
       : lines ? 1
               : 0) |
      (!emulated &&
             dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
             dyn->rs.provoking_vertex ==
                VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
          ? 4
          : 0);
   if (hk_apple9_reads_draw_id(gfx))
      gfx->descriptors.root.draw.draw_id = draw_id;
   gfx->descriptors.root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, &gfx->descriptors);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct agx_apple9_render_stage *s = &cmd->apple9_stage[stage];
      uint64_t sysvals = 0;
      hk_apple9_upload_textures(cmd, &record->texture_table[stage],
                                &record->sampler_table[stage], &sysvals);

      record->push[stage] =
         hk_apple9_push_words(cmd, shaders[stage], linked[stage], gfx->root);
      if (!record->push[stage])
         FAIL(VK_ERROR_OUT_OF_DEVICE_MEMORY, "out of memory");

      for (unsigned slot = 0; slot < s->resource_count; ++slot) {
         if (s->resource_binding[slot] != AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
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
   const struct agx_apple9_uniform_draw *previous =
      cmd->apple9_prev_cs == cs ? &cmd->apple9_draw[!cmd->apple9_draw_current]
                                : NULL;
   bool prepared = agx_apple9_prepare_draw_alloc(
      &dev->dev, cmd, hk_apple9_usc_alloc, cmd, hk_apple9_context_alloc,
      &pipeline, record, previous);
   if (!prepared)
      UNSUPPORTED("the draw state cannot be prepared");
   cmd->apple9_draw_current = !cmd->apple9_draw_current;
   cmd->apple9_prev_cs = cs;
   pipeline.ppp = record->ppp;
   pipeline.ppp_viewport_offset = record->ppp_viewport_offset;
   pipeline.ppp_viewport_words = record->ppp_viewport_words;
   pipeline.vertex_launch = record->launch[0] / 0x40;
   pipeline.pipeline_word = 0x01000000;
   hk_ensure_cs_has_space(cmd, cs, 256);
   hk_apple9_record_draw(cs, &pipeline, draw);
   hk_apple9_account_draw(cs, &pipeline, draw, gs || tes || mesh);
   /* Only ordinary single-view draws reuse prepared state. Other paths
    * still consume dynamic state in their software-stage lowering. */
   if (hk_apple9_draw_cache_supported(cmd, draw)) {
      cmd->apple9_draw_state.valid = true;
      cmd->apple9_draw_state.render_gen = gfx->render_gen;
      cmd->apple9_draw_state.first_vertex =
         draw.indexed ? draw.index_bias : draw.start;
      cmd->apple9_draw_state.base_instance = draw.start_instance;
      cmd->apple9_draw_state.pipeline = pipeline;
      for (unsigned stage = 0; stage < 2; ++stage) {
         cmd->apple9_draw_state.shader[stage] = shaders[stage];
         cmd->apple9_draw_state.linked[stage] = linked[stage];
      }
      /* Every dynamic, hk and shader dirty bit is consumed here, so the
       * next clean draw can reuse this state. G16 has no other dirty-bit
       * readers after this point (only shader resolution and the sysval
       * upload, both above); a new one must read its bits before this or
       * invalidate apple9_draw_state itself. */
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
hk_apple9_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
               struct agx_draw draw)
{
   hk_apple9_draw_impl(cmd, draw_id, draw);
}
