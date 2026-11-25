/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
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
#define AGX_APPLE9_MAX_3D_TEXTURE_DIMENSION 2048

/* Direct resource table slot size; sampler payloads occupy only eight bytes. */
#define AGX_APPLE9_SAMPLER_TABLE_STRIDE 32

/* Linear sampled textures encode (row_bytes / 16) - 1 in 16 bits. */
#define AGX_APPLE9_MAX_LINEAR_STRIDE (1u << 20)

bool agx_apple9_texture_format_supported(enum pipe_format format);

/* Translate common image/sampler state to native Apple9 table records. */
struct agx_texture_packed;
struct agx_sampler_packed;
struct AGX_SAMPLER;
struct AGX_TEXTURE;
bool agx_apple9_pack_texture(void *out, const struct AGX_TEXTURE *state);
void agx_apple9_set_null_texture(void *out);
bool agx_apple9_pack_sampler_state(void *out, const struct AGX_SAMPLER *state);
bool agx_apple9_translate_texture(void *out,
                                  const struct agx_texture_packed *state);
struct agx_pbe_packed;
struct ail_layout;
/* layer_base is the selected first layer's level-zero address. The native
 * descriptor retains the allocation's mip tree and selects level separately.
 */
bool agx_apple9_translate_pbe(void *out, const struct agx_pbe_packed *state,
                              enum pipe_format format,
                              const struct ail_layout *layout,
                              uint64_t layer_base, unsigned level,
                              unsigned layer_count, bool layered,
                              unsigned dimension);

void agx_apple9_pack_sampler(void *out, bool min_linear, bool mag_linear,
                             unsigned mip_filter, float min_lod, float max_lod,
                             unsigned wrap_s, unsigned wrap_t,
                             unsigned max_anisotropy);
void agx_apple9_pack_fetch_sampler(void *out);

#ifdef __cplusplus
}
#endif

#endif
