/*
 * Copyright 2026 Niklas Sheth
 * Copyright 2026 Youwes09
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws with launch programs: the linked stage pair, the per-draw tables,
 * launches and PPP record, and the VDM draw commands.
 */

#ifndef AGX_DRAW_TABLE_H
#define AGX_DRAW_TABLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_launch_abi.h"
#include "asahi/lib/agx_launch_program.h"
#include "asahi/lib/agx_ppp_record.h"
#include "util/format/u_formats.h"
#include "agx_bo.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx_bo;
struct agx_device;
struct agx_pool;
struct agx_ptr;
struct agx_shader_body;

struct agx_draw_stage {
   /* Stable compiled-object identity; zero for externally supplied stages. */
   uint64_t program_id;
   /* Owned by the compiled shader; each using batch retains a BO reference. */
   struct agx_bo *bo;
   /* Optional body identity; whole-BO callers leave this NULL. */
   struct agx_shader_body *body;
   const uint8_t *binary;
   size_t binary_size;
   uint32_t preamble_offset, preamble_size;
   uint32_t ubo_mask;
   uint32_t scratch_size;
   uint8_t atomic_frame_size;
   uint16_t publication_count;
   bool publication_count_valid;
   uint32_t resource_ssbo_mask, resource_write_mask;
   uint8_t resource_count;
   uint8_t resource_binding[AGX_MAX_GRAPHICS_BUFFERS];
   uint32_t texture_mask, sampler_mask, image_mask;
   bool uses_texel_fetch;
   bool uses_discard;
   bool writes_depth;
   bool reads_tile;
   bool disable_tri_merging;
   /* Pixel interlock: order fragments per pixel (fragment control "Pixel
    * order") instead of per covered sample. */
   bool pixel_interlock;

   /* Scalar interface counts used by the bounded stage linker. */
   uint8_t position_components;
   uint8_t varying_components;
   struct agx_varying_layout varyings;
   struct agx_interp_mask linear_mask, flat_mask;
   bool reads_z;
   bool reads_primitive_id;
   bool writes_point_size;
   bool writes_layer_viewport;
   uint8_t clip_distance_count;
   bool reads_point_coord;
   uint8_t render_targets;
   /* Optional per-shader launch templates; NULL builds every launch. */
   struct agx_stage_launch_cache *launch_cache;
   /* Fragment-owned interpolation tables for each provoking vertex. */
   uint32_t coefficients[3];
   uint16_t coefficient_count;
};

struct agx_draw_pipeline {
   /* Vertex fetch is lowered into the API vertex program. The stages belong
    * to their compiled shaders, which outlive every draw using them. */
   const struct agx_draw_stage *vertex;
   const struct agx_draw_stage *fragment;

   /* Color surfaces in draw-buffer order; absent attachments have address 0. */
   uint64_t color_targets[8];
   enum pipe_format color_formats[8];
   uint8_t samples;

   /* The first AGX_PPP_STATIC_SIZE bytes of the PPP record that depend
    * only on the linked vertex and fragment stages
    * (agx_build_ppp_static), built once per link; NULL builds them in
    * every draw. */
   const uint8_t *ppp_static;

   /* Batch-owned PPP record, relative to the render-context aperture. */
   uint32_t ppp;
   /* Viewport block within the PPP record and its length in words. Zero
    * selects the single-viewport block at 0xc0.
    */
   uint16_t ppp_viewport_offset;
   uint16_t ppp_viewport_words;

   /* VDM linkage words produced by the bounded pipeline linker. */
   uint32_t pipeline_word;
   uint32_t vertex_launch;
   /* Scalar vertex outputs, position included */
   uint8_t output_count;

   uint64_t index_buffer;
   uint32_t index_extent;
   uint8_t index_size;
   uint8_t primitive;
   bool primitive_restart;
   bool flatshade_first;
   uint32_t restart_index;
};

/* CPU snapshot used while encoding a draw and for adjacent state reuse.
 * Encoded allocations belong to the batch's growing pools. */
struct agx_draw_record {
   uint64_t code[2];
   uint32_t launch[2], ppp, coefficients;
   uint64_t program_id[2];
   uint16_t tile_bytes, cf_count;
   uint8_t samples;
   /* CPU copy for bounded record reuse; never read back write-combined BOs.
    * Records with several viewports append the viewport block after the
    * fixed 256-byte state.
    */
   uint8_t ppp_record[AGX_PPP_RECORD_MAX];
   uint16_t ppp_size, ppp_viewport_offset, ppp_viewport_words;
   uint64_t buffers[2][AGX_MAX_GRAPHICS_BUFFERS];
   uint8_t buffer_count[2];
   uint64_t vertex_table;
   uint64_t fragment_table;
   /* Graphics stage indices: vertex = 0, fragment = 1. */
   uint64_t texture_table[2], sampler_table[2];
   /* Push words of each stage (AGX_LAUNCH_PUSH_ROOT) */
   uint64_t push[2];
   uint32_t depth_control, depth_face[2], stencil[2];
   uint32_t raster_control;
   uint8_t object_type;
   float viewport_translate[3], viewport_scale[3];
   /* Viewports after the first, as {tx, sx, ty, sy, tz, sz}; zero or one
    * viewport_count selects the single viewport above. Scissors for the
    * additional viewports follow scissor_index consecutively.
    */
   uint8_t viewport_count;
   float viewports[AGX_PPP_MAX_VIEWPORTS - 1][6];
   uint32_t scissor_index;
   uint16_t depth_bias_index;
   uint16_t occlusion_index;
   uint8_t visibility_mode;
   uint16_t scissor_min[2], scissor_max[2];
   bool reads_tile;
   bool uses_discard;
   bool writes_depth;
   bool disable_tri_merging;
   bool pixel_interlock;
   bool flatshade_first;
   uint8_t provoking_vertex;
};

/* Clear a record for reuse without touching its large arrays:
 * agx_prepare_draw_record_alloc rebuilds ppp_record, and buffers and
 * viewports are read only up to the counts the caller fills in. */
static inline void
agx_draw_record_reset(struct agx_draw_record *draw)
{
   memset(draw, 0, offsetof(struct agx_draw_record, ppp_record));
   memset(&draw->ppp_size, 0,
          offsetof(struct agx_draw_record, buffers) -
             offsetof(struct agx_draw_record, ppp_size));
   memset(&draw->buffer_count, 0,
          offsetof(struct agx_draw_record, viewports) -
             offsetof(struct agx_draw_record, buffer_count));
   memset(&draw->scissor_index, 0,
          sizeof(*draw) - offsetof(struct agx_draw_record, scissor_index));
}

/* Whether a draw can take the previous draw's launch for stage (0 vertex, 1
 * fragment) or its coefficient table. Both need nonzero, equal program ids:
 * a stage without an id is never reused. draw->program_id, code, tables,
 * samples, tile_bytes and provoking_vertex must already be filled in. */
static inline bool
agx_draw_reuses_launch(const struct agx_draw_record *previous,
                       const struct agx_draw_record *draw, unsigned stage)
{
   return previous && draw->program_id[stage] &&
          previous->program_id[stage] == draw->program_id[stage] &&
          previous->code[stage] == draw->code[stage] &&
          previous->texture_table[stage] == draw->texture_table[stage] &&
          previous->sampler_table[stage] == draw->sampler_table[stage] &&
          previous->push[stage] == draw->push[stage] &&
          (stage ? previous->fragment_table == draw->fragment_table
                 : previous->vertex_table == draw->vertex_table) &&
          (!stage || (previous->samples == draw->samples &&
                      previous->tile_bytes == draw->tile_bytes));
}

static inline bool
agx_draw_reuses_coefficients(const struct agx_draw_record *previous,
                             const struct agx_draw_record *draw)
{
   return previous && draw->program_id[0] && draw->program_id[1] &&
          previous->program_id[0] == draw->program_id[0] &&
          previous->program_id[1] == draw->program_id[1] &&
          previous->provoking_vertex == draw->provoking_vertex;
}

#define AGX_CF_TABLE_SIZE   0x2a0
#define AGX_CF_TABLE_STRIDE 0x2c0
#define AGX_CF_TABLES_SIZE  (3 * AGX_CF_TABLE_STRIDE)
unsigned agx_coefficients_size(const struct agx_draw_stage *fragment);
void agx_init_coefficients(struct agx_device *dev,
                           struct agx_draw_stage *fragment,
                           struct agx_ptr allocation);

/* The caller owns both immutable arenas. Context allocations must lie wholly
 * within the render-context aperture; USC allocations use the shader aperture.
 * A false return leaves the draw unpublished, including on allocation failure.
 */
bool agx_prepare_draw_record_alloc(
   struct agx_device *dev, void *usc_allocator,
   struct agx_ptr (*usc_alloc)(void *, size_t, size_t), void *context_allocator,
   struct agx_ptr (*context_alloc)(void *, size_t, size_t),
   const struct agx_draw_pipeline *pipeline, struct agx_draw_record *draw,
   const struct agx_draw_record *previous);

bool agx_prepare_draw_record(struct agx_device *dev, struct agx_pool *usc_pool,
                             struct agx_pool *context_pool,
                             const struct agx_draw_pipeline *pipeline,
                             struct agx_draw_record *draw,
                             const struct agx_draw_record *previous);

/* Update the tables and push words (unless push is NULL) of a prepared draw
 * with unchanged shader and raster state. Failure leaves the draw unchanged.
 */
bool agx_update_draw_tables_alloc(
   struct agx_device *dev, void *usc_allocator,
   struct agx_ptr (*usc_alloc)(void *, size_t, size_t), void *context_allocator,
   struct agx_ptr (*context_alloc)(void *, size_t, size_t),
   const struct agx_draw_pipeline *pipeline, struct agx_draw_record *draw,
   const uint64_t tables[2], const uint64_t push[2]);

bool agx_link_draw_pipeline(struct agx_draw_pipeline *pipeline,
                            const struct agx_draw_stage *vertex,
                            const struct agx_draw_stage *fragment);

size_t agx_direct_draw_size(const struct agx_draw_pipeline *pipeline);

uint8_t *agx_emit_direct_draw(uint8_t *out,
                              const struct agx_draw_pipeline *pipeline,
                              unsigned vertex_count, unsigned instance_count,
                              unsigned vertex_start);

uint8_t *agx_emit_indirect_draw(uint8_t *out,
                                const struct agx_draw_pipeline *pipeline,
                                uint64_t indirect);

static inline struct agx_shader_body *
agx_draw_stage_body(const struct agx_draw_stage *stage)
{
   return stage->body ? stage->body : &stage->bo->shader_body;
}

#ifdef __cplusplus
}
#endif

#endif
