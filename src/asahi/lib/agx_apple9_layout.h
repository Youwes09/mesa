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
#define AGX_APPLE9_GRAPHICS_SYSVAL_SIZE                                        \
   (AGX_APPLE9_POLYGON_STIPPLE_OFFSET + 32 * 4)

/* Optional per-dispatch sampler state for APIs with mutable samplers */
#define AGX_APPLE9_RUNTIME_SAMPLER_OFFSET AGX_APPLE9_GRAPHICS_SYSVAL_SIZE
#define AGX_APPLE9_RUNTIME_SAMPLER_STRIDE 64

/* The 32-bit words of a runtime sampler record */
enum agx_apple9_sampler_field {
   AGX_APPLE9_SAMP_WRAP_S,
   AGX_APPLE9_SAMP_WRAP_T,
   AGX_APPLE9_SAMP_WRAP_R,
   AGX_APPLE9_SAMP_MIN,
   AGX_APPLE9_SAMP_MAG,
   AGX_APPLE9_SAMP_MIP,
   AGX_APPLE9_SAMP_COMPARE,
   AGX_APPLE9_SAMP_MIN_LOD,
   AGX_APPLE9_SAMP_MAX_LOD,
   AGX_APPLE9_SAMP_BIAS,
   AGX_APPLE9_SAMP_BORDER,
   AGX_APPLE9_SAMP_PIXEL_COORDS,
   AGX_APPLE9_SAMP_MAX_ANISOTROPY,
   /* Nonzero if cube maps are sampled without filtering across faces */
   AGX_APPLE9_SAMP_SEAMFUL_CUBE,
};

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
#define AGX_APPLE9_RUNTIME_SYSVAL_SIZE                                         \
   (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 16)

#endif
