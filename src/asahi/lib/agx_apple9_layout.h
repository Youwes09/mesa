/* SPDX-License-Identifier: MIT */
#ifndef AGX_APPLE9_LAYOUT_H
#define AGX_APPLE9_LAYOUT_H

#include <stdint.h>

#define AGX_APPLE9_TEXTURE_TABLE_STRIDE 32

/* Texture indices a shader can name; indices 128-255 hold the cube aliases */
#define AGX_APPLE9_MAX_TEXTURE_INDICES 128

/* Cube aliases: 2D array views of cube faces, for non-seamless sampling. Index
 * i aliases at CUBE_ALIAS_INDEX(i), a handle at +CUBE_ALIAS_HANDLE_OFFSET.
 */
#define AGX_APPLE9_CUBE_ALIAS_INDEX(i) (AGX_APPLE9_MAX_TEXTURE_INDICES + (i))
#define AGX_APPLE9_CUBE_ALIAS_HANDLE_OFFSET AGX_APPLE9_TEXTURE_TABLE_STRIDE

/* The launch target carries 2 * entry_offset + 0x2a in 24 bits. Entries
 * reside in this compact USC range and branch to independently allocated
 * shader bodies anywhere in the 4-GiB USC heap. An entry belongs to its
 * shader BO and cannot be recycled while any batch references that BO. */
#define AGX_APPLE9_ENTRY_ARENA_SIZE 0x00800000u
/* The arena is backed on demand in chunks of this size, see
 * agx_apple9_entry_back. */
#define AGX_APPLE9_ENTRY_CHUNK_SIZE 0x00100000u
#define AGX_APPLE9_ENTRY_CHUNKS                                                \
   (AGX_APPLE9_ENTRY_ARENA_SIZE / AGX_APPLE9_ENTRY_CHUNK_SIZE)
#define AGX_APPLE9_ENTRY_HEADER_SIZE 0x00000340u
#define AGX_APPLE9_ENTRY_BLOCK_SIZE  0x000000c0u
#define AGX_APPLE9_ENTRY_CODE_OFFSET 0x00000080u
#define AGX_APPLE9_ENTRY_ALIGNMENT   0x00000040u
#define AGX_APPLE9_ENTRY_MAX_OFFSET  ((0xffffffu - 0x2au) / 2u)

/* Driver sysvals share the buffer table: API UBOs 0-31, vertex buffers 32-63 */
#define AGX_APPLE9_GRAPHICS_SYSVAL_BINDING 64
#define AGX_APPLE9_SAMPLER_BIAS_OFFSET     16
#define AGX_APPLE9_SAMPLER_BIAS_COUNT      32
#define AGX_APPLE9_TEXTURE_INFO_OFFSET                                         \
   (16 + AGX_APPLE9_SAMPLER_BIAS_COUNT * sizeof(float))
#define AGX_APPLE9_TEXTURE_INFO_STRIDE 32
#define AGX_APPLE9_POINT_SIZE_OFFSET                                           \
   (AGX_APPLE9_TEXTURE_INFO_OFFSET + 32 * AGX_APPLE9_TEXTURE_INFO_STRIDE)
#define AGX_APPLE9_FIRST_FAN_OFFSET       (AGX_APPLE9_POINT_SIZE_OFFSET + 4)
#define AGX_APPLE9_POLYGON_STIPPLE_OFFSET (AGX_APPLE9_POINT_SIZE_OFFSET + 16)
/* Line stipple word, then the viewport 0 scale and offset as vec3s */
#define AGX_APPLE9_LINE_STIPPLE_OFFSET                                         \
   (AGX_APPLE9_POLYGON_STIPPLE_OFFSET + 32 * 4)
#define AGX_APPLE9_VIEWPORT_SCALE_OFFSET  (AGX_APPLE9_LINE_STIPPLE_OFFSET + 16)
#define AGX_APPLE9_VIEWPORT_OFFSET_OFFSET (AGX_APPLE9_LINE_STIPPLE_OFFSET + 32)
#define AGX_APPLE9_GRAPHICS_SYSVAL_SIZE   (AGX_APPLE9_LINE_STIPPLE_OFFSET + 48)

/* Optional per-dispatch sampler state for APIs with mutable samplers */
#define AGX_APPLE9_RUNTIME_SAMPLER_OFFSET AGX_APPLE9_GRAPHICS_SYSVAL_SIZE
#define AGX_APPLE9_RUNTIME_SAMPLER_STRIDE 64

/* hk descriptors live in the USC aperture, handles are offsets from its base */
#define AGX_APPLE9_USC_BASE                   (1ull << 40)
#define AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY   2
#define AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY   3
#define AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO    0
#define AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER 0
/* Native index of the texel fetch sampler, for register-indexed samplers */
#define AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET                                  \
   (AGX_APPLE9_RUNTIME_SAMPLER_OFFSET + 32 * AGX_APPLE9_RUNTIME_SAMPLER_STRIDE)
/* Per-draw color attachment write enables, one bit per attachment */
#define AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET                                   \
   (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 4)
/* Packed sample positions, in the PPP multisample control format */
#define AGX_APPLE9_SAMPLE_POSITIONS_OFFSET                                     \
   (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 8)
/* Per-viewport line stipple transform, vec4 (scale.xy, offset.xy) each */
#define AGX_APPLE9_VIEWPORT_TABLE_OFFSET                                       \
   (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 16)
#define AGX_APPLE9_VIEWPORT_TABLE_COUNT 16
/* Software polygon point state (struct agx_apple9_polygon_point_params) */
#define AGX_APPLE9_POLYGON_POINT_OFFSET                                        \
   (AGX_APPLE9_VIEWPORT_TABLE_OFFSET + AGX_APPLE9_VIEWPORT_TABLE_COUNT * 16)
#define AGX_APPLE9_RUNTIME_SYSVAL_SIZE (AGX_APPLE9_POLYGON_POINT_OFFSET + 32)

enum agx_apple9_polygon_point_flags {
   AGX_APPLE9_PP_CULL_FRONT = 1 << 0,
   AGX_APPLE9_PP_CULL_BACK = 1 << 1,
   AGX_APPLE9_PP_FRONT_CCW = 1 << 2,
   AGX_APPLE9_PP_DEPTH_CLIP = 1 << 3,
   AGX_APPLE9_PP_DEPTH_BIAS = 1 << 4,
   /* The depth bias unit depends on the polygon's depth (float formats) */
   AGX_APPLE9_PP_DEPTH_BIAS_FLOAT = 1 << 5,
};

struct agx_apple9_polygon_point_params {
   uint32_t flags;
   float bias_constant, bias_slope, bias_clamp;
   /* Minimum resolvable depth difference of unorm formats */
   float bias_unit;
   /* Viewport 0 maxDepth - minDepth */
   float depth_range;
   uint32_t pad[2];
};

#endif
