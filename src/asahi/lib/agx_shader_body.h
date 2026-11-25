/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/list.h"

/* The launch target carries 2 * entry_offset + 0x2a in 24 bits. Entries
 * reside in this compact USC range and branch to independently allocated
 * shader bodies anywhere in the 4-GiB USC heap. An entry belongs to its
 * shader BO and cannot be recycled while any batch references that BO. */
#define AGX_ENTRY_ARENA_SIZE 0x00800000u
/* The arena is backed on demand in chunks of this size, see
 * agx_shader_entry_back. */
#define AGX_ENTRY_CHUNK_SIZE 0x00100000u
#define AGX_ENTRY_CHUNKS      (AGX_ENTRY_ARENA_SIZE / AGX_ENTRY_CHUNK_SIZE)
#define AGX_ENTRY_HEADER_SIZE 0x00000340u
#define AGX_ENTRY_BLOCK_SIZE  0x000000c0u
#define AGX_ENTRY_CODE_OFFSET 0x00000080u
#define AGX_ENTRY_ALIGNMENT   0x00000040u
#define AGX_ENTRY_MAX_OFFSET  ((0xffffffu - 0x2au) / 2u)

struct agx_bo;
struct agx_device;

/* Immutable shader code at a stable address, entered through the shader entry
 * arena. Entry identity and recording ownership belong to the body, which may
 * occupy only part of its BO.
 *
 * A whole-BO body is embedded in agx_bo and has no destroy callback: its
 * references delegate to that BO. Separately allocated bodies start with one
 * reference and release their storage through destroy after the last owner.
 */
struct agx_shader_body {
   struct agx_bo *bo;
   uint64_t addr;
   int32_t refcnt;
   void (*destroy)(struct agx_device *dev, struct agx_shader_body *body);

   /* Protected by agx_device::entry_lock. Entries may be evicted only
    * when no recording pins them; a later use allocates another entry. */
   uint32_t entry_offset;
   uint32_t entry_pins;
   bool entry_fixed;
   struct list_head entry_link;
};

#ifdef __cplusplus
extern "C" {
#endif

void agx_shader_body_init_bo(struct agx_bo *bo);
void agx_shader_body_reference(struct agx_shader_body *body);
void agx_shader_body_unreference(struct agx_device *dev,
                                 struct agx_shader_body *body);
void agx_shader_body_finish(struct agx_device *dev,
                            struct agx_shader_body *body);

/* Write the header of the shader entry arena */
void agx_init_shader_entries(struct agx_device *dev);

/* The body's offset in the shader entry arena, allocating an entry if it has
 * none, or 0 if the arena is full. allocator is passed to the device's pin
 * hook, which ties the entry to the recording's lifetime.
 */
uint32_t agx_shader_entry(struct agx_device *dev, void *allocator,
                          struct agx_shader_body *body, bool compute);

#ifdef __cplusplus
}
#endif
