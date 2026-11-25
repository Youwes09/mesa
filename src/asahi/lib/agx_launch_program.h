/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_LAUNCH_PROGRAM_H
#define AGX_LAUNCH_PROGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "util/simple_mtx.h"

#ifdef __cplusplus
extern "C" {
#endif

enum agx_launch_stage {
   AGX_LAUNCH_VERTEX,
   AGX_LAUNCH_FRAGMENT,
   AGX_LAUNCH_COMPUTE,
};

enum {
   AGX_GRAPHICS_LAUNCH_SIZE = 256,
   AGX_COMPUTE_LAUNCH_SIZE = 1024,
};

struct agx_launch_parameters {
   uint64_t shader_base;
   uint64_t resource_table;
   /* Byte offset of the stage entry within the USC heap. */
   uint32_t entry_offset;
   /* Compiler high-water mark in 32-bit publication slots. */
   uint16_t publication_count;
   uint16_t frame_extent_a;
   uint16_t frame_extent_b;
   uint8_t tile_bytes;
   uint8_t samples;
   uint32_t threadgroup_memory_bytes;
   /* Visible compute buffers, after the group-count pointer. */
   unsigned resource_count;
   /* Optional tail transfer to a compiler-owned setup body ending in STOP.
    * Its BO has the same ownership/lifetime as the main shader. */
   uint64_t preamble_address;
   uint64_t launch_address;
   /* Compute: the root words (agx_apple9_compute_root_words of them), which
    * the launch materializes as literals instead of loading them from
    * resource_table. Required for agx_launch_program_build; templates leave
    * patchable slots instead (agx_launch_template_set_roots). */
   const uint32_t *root_words;
};

/* Current T8132 execution evidence covers these allocation sizes. This does
 * not imply that the compiler implements shared load/store/barrier lowering. */
bool agx_launch_threadgroup_memory_supported(uint32_t bytes);

/* Offset of the encoded entry field, for diagnostics and frame inspection. */
size_t agx_launch_call_offset(enum agx_launch_stage stage,
                              unsigned resource_count);

/* Build from stage parameters only; no executable input is read.
 * On invalid input leave the destination untouched; on success zero all unused
 * bytes through capacity. The fixed allocation sizes above include padding. */
bool agx_launch_program_build(uint8_t *out, size_t capacity,
                              enum agx_launch_stage stage,
                              const struct agx_launch_parameters *params);

/* A launch program built once, at pipeline creation, for repeated emission.
 * Only the resource table address, the entry offset and the program's own
 * address (which positions the preamble branch) vary between launches of a
 * pipeline; agx_launch_template_emit patches exactly those fields.
 * A plain value: it holds no pointers.
 */
struct agx_launch_template {
   uint8_t program[AGX_COMPUTE_LAUNCH_SIZE];
   uint64_t shader_base;
   uint64_t preamble_address;
   /* Encoded length of the program; emission zero-fills to capacity. */
   uint16_t length;
   uint16_t capacity;
   uint16_t target_at;
   /* Offset of the preamble branch, or 0 without a preamble. */
   uint16_t branch_at;
   /* Compute: offset and count of the root-word literal slots. */
   uint16_t roots_at;
   uint16_t root_words;
   enum agx_launch_stage stage;
   bool valid;
};

/* Build a template from stage parameters. resource_table, entry_offset and
 * launch_address are per-launch and ignored. Returns false (and leaves the
 * template invalid) when agx_launch_program_build would reject the stage
 * parameters. */
bool agx_launch_template_init(struct agx_launch_template *tmpl, size_t capacity,
                              enum agx_launch_stage stage,
                              const struct agx_launch_parameters *params);

/* Emit the program for one launch: the same bytes agx_launch_program_build
 * produces with the template's parameters and these per-launch fields. */
bool agx_launch_template_emit(uint8_t *out,
                              const struct agx_launch_template *tmpl,
                              uint64_t resource_table, uint32_t entry_offset,
                              uint64_t launch_address);

/* Write a compute launch's root words into the literal slots of a program
 * emitted from tmpl. The words are frozen when the launch is built, where a
 * load would read them at launch: the resource record is CPU-written once per
 * dispatch, no GPU pass writes it, and the roots are addresses, so
 * update-after-bind descriptor contents are unaffected. */
bool agx_launch_template_set_roots(uint8_t *out,
                                   const struct agx_launch_template *tmpl,
                                   const uint32_t *root_words);

struct agx_draw_stage;
struct agx_device;
struct agx_ptr;

/* Graphics launch programs of one compiled stage, built on first use for each
 * fragment tile allocation and sample count (vertex stages use 0/0) and then
 * patched per launch. Entries are immutable once published; owned by the
 * compiled shader alongside its BO. */
#define AGX_STAGE_LAUNCH_TEMPLATES 4
struct agx_stage_launch_cache {
   simple_mtx_t lock;
   /* Published entries; written with release ordering under the lock. */
   unsigned count;
   struct {
      uint8_t tile_bytes, samples;
      struct agx_launch_template tmpl;
   } entry[AGX_STAGE_LAUNCH_TEMPLATES];
};

void agx_stage_launch_cache_init(struct agx_stage_launch_cache *c);
void agx_stage_launch_cache_finish(struct agx_stage_launch_cache *c);

/* Per-sample tile allocation for a fragment-stage launch or render. As in
 * agx_build_tilebuffer_layout, multisampled renders keep a nonempty
 * allocation even without colour targets: a large G16 2x depth-only render
 * with zero-byte fragment launches can leave its fragment stage unretired.
 */
static inline unsigned
agx_launch_tile_bytes(unsigned color_bytes, unsigned samples)
{
   return color_bytes || samples <= 1 ? color_bytes : 8;
}

/* Prepare one native stage independently, including background/EOT helpers.
 * The caller retains the shader BO and immutable allocation until completion.
 * push is the address of the stage's push words (AGX_APPLE9_PUSH_ROOT).
 */
bool agx_prepare_stage_launch(struct agx_device *dev, void *allocator,
                              struct agx_ptr (*alloc)(void *, size_t, size_t),
                              const struct agx_draw_stage *shader,
                              bool fragment, uint64_t table, uint64_t textures,
                              uint64_t samplers, uint64_t push,
                              unsigned tile_bytes, unsigned samples,
                              uint32_t *launch);

#ifdef __cplusplus
}
#endif

#endif
