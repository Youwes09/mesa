/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * Texture, PBE and sampler descriptors in the 32-byte records that shaders
 * address within the USC aperture (agx_has_usc_descriptors).
 */

#ifndef AGX_USC_DESCRIPTOR_H
#define AGX_USC_DESCRIPTOR_H

#include <stdbool.h>
#include <stdint.h>
#include "util/format/u_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Native 3D texture depth is an 11-bit minus-one field. Array textures have
 * a wider layer count; their range does not apply to 3D sampling.
 */
#define AGX_USC_MAX_3D_DIMENSION 2048

/* Direct resource table slot size; sampler payloads occupy only eight bytes. */
#define AGX_USC_SAMPLER_STRIDE 32

/* Linear sampled textures encode (row_bytes / 16) - 1 in 16 bits. */
#define AGX_USC_MAX_LINEAR_STRIDE (1u << 20)

bool agx_usc_texture_format_supported(enum pipe_format format);

/* Translate common image and sampler state to USC descriptor records */
struct agx_texture_packed;
struct agx_sampler_packed;
struct AGX_SAMPLER;
struct AGX_TEXTURE;
bool agx_pack_usc_texture(void *out, const struct AGX_TEXTURE *state);
void agx_set_null_usc_texture(void *out);
bool agx_pack_usc_sampler(void *out, const struct AGX_SAMPLER *state);
bool agx_translate_usc_texture(void *out,
                               const struct agx_texture_packed *state);
struct agx_pbe_packed;
struct ail_layout;
/* layer_base is the selected first layer's level-zero address. The native
 * descriptor retains the allocation's mip tree and selects level separately.
 */
bool agx_translate_usc_pbe(void *out, const struct agx_pbe_packed *state,
                           enum pipe_format format,
                           const struct ail_layout *layout, uint64_t layer_base,
                           unsigned level, unsigned layer_count, bool layered,
                           unsigned dimension);

void agx_pack_basic_usc_sampler(void *out, bool min_linear, bool mag_linear,
                                unsigned mip_filter, float min_lod,
                                float max_lod, unsigned wrap_s, unsigned wrap_t,
                                unsigned max_anisotropy);
void agx_pack_usc_fetch_sampler(void *out);

/* RGB32 buffer views are three scalar texels in the physical descriptor */
static inline bool
agx_texture_is_rgb32(enum pipe_format format)
{
   return format == PIPE_FORMAT_R32G32B32_FLOAT ||
          format == PIPE_FORMAT_R32G32B32_UINT ||
          format == PIPE_FORMAT_R32G32B32_SINT;
}

#ifdef __cplusplus
}
#endif

#endif
