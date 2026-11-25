/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_CDM_H
#define AGX_CDM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "asahi/compiler/agx_apple9_profile.h"
#include "asahi/lib/agx_launch_program.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx_bo;
struct agx_device;
struct agx_pool;
struct agx_ptr;
struct agx_shader_body;

#define AGX_APPLE9_COMPUTE_RESOURCE_STRIDE                0x20u
#define AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE       0x100u
#define AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET         0xc0u
#define AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE                0x2cu
#define AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE       0x28u
#define AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE 0x1cu

size_t agx_apple9_compute_launch_size(
   const struct agx_apple9_compute_profile *profile);

/* Map a package argument record to the caller's logical buffer namespace and
 * binding.  Compiler-generated profiles may mix UBO inputs with SSBO inputs
 * and outputs without changing native package argument order. */
enum agx_apple9_compute_resource_kind agx_apple9_compute_resource_kind(
   const struct agx_apple9_compute_profile *profile, unsigned argument);

enum agx_apple9_compute_geometry_mode {
   AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
   AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT,
};

struct agx_apple9_compute_geometry {
   enum agx_apple9_compute_geometry_mode mode;
   union {
      uint32_t threads[3];
      uint64_t group_counts;
   };
   uint32_t local[3];
};

/* Populate the group-count pointer and inline counts in a resource record. */
bool agx_apple9_build_compute_geometry_fields(
   void *record, size_t record_size, uint64_t record_address,
   const struct agx_apple9_compute_geometry *geometry);

/* Build immutable resource and launch records in the batch's USC pool. */
bool agx_apple9_prepare_compute_dispatch_alloc(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   struct agx_shader_body *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address);

/* As agx_apple9_prepare_compute_dispatch_alloc, emitting the launch program
 * from tmpl (agx_apple9_compute_launch_template of the same profile and
 * preamble) when it is non-NULL and valid. Graphics ABI programs read their
 * push words from push. HK_APPLE9_LAUNCH_CHECK=1 checks against a full build.
 */
bool agx_apple9_prepare_compute_dispatch_alloc_tmpl(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   struct agx_shader_body *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers, uint64_t push,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, const struct agx_launch_template *tmpl,
   uint64_t *launch_address);

/* Build a compute pipeline's launch template, once at pipeline creation. */
bool agx_apple9_compute_launch_template(
   struct agx_launch_template *tmpl,
   const struct agx_apple9_compute_profile *profile, uint64_t usc_exec_base,
   uint64_t preamble_address);

bool agx_apple9_prepare_compute_dispatch(
   struct agx_device *dev, struct agx_pool *usc_pool, struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers, uint64_t push,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address);

/* Record-size constants include the optional trailing CDM barrier. Without
 * it, the emitted launch is one uint32_t shorter; the caller supplies the next
 * dependency barrier or a final stream barrier.
 */
/* Why the emitters below would reject a dispatch, or NULL if they accept it.
 * global is unused for indirect dispatches. */
const char *agx_apple9_dispatch_reject_reason(
   uint64_t launch, bool indirect, uint64_t indirect_address,
   const uint32_t global[3], const uint32_t local[3],
   const struct agx_apple9_compute_profile *profile);

bool agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile,
   bool barrier);

bool agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   bool indirect_local, const struct agx_apple9_compute_profile *profile,
   bool barrier);

/* Replace the trailing barrier of a dispatch record emitted with one by the
 * compute-only barrier (0x160 in the barrier bits): it orders the following
 * dispatches and makes this one's writes visible to them, but doesn't drain.
 * Consumers outside compute, and indirect grid fetches, still need the full
 * barrier before them.
 */
void agx_apple9_set_compute_barrier(void *record, unsigned record_size);

#ifdef __cplusplus
}
#endif

#endif
