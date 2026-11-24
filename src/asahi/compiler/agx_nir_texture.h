/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

/* Texture backend flags */
#define AGX_TEXTURE_FLAG_NO_CLAMP (1 << 0)

/* Indicates that the sampler should be overriden to clamp to 0 instead of 1 */
#define AGX_TEXTURE_FLAG_CLAMP_TO_0 (1 << 1)

/* A 2D access the driver rewrote as layer 0 of a one-layer array, so the
 * Apple9 texture lowering still treats it as 2D, for unnormalized samplers. */
#define AGX_TEXTURE_FLAG_ARRAYED_2D (1 << 2)

/* A size, level or sample query of a sampled view, answered from the view's
 * texture metadata (Apple9 texture_info) rather than its texture descriptor.
 * agx_nir_lower_texture_queries leaves it to the Apple9 backend. */
#define AGX_TEXTURE_FLAG_METADATA_QUERY (1 << 3)

/* An Apple9 sample with a minimum LOD in its parameters */
#define AGX_TEXTURE_FLAG_MIN_LOD (1 << 4)

/* Texel buffers lowered to (at most) 16384x16384 2D textures */
#define AGX_TEXTURE_BUFFER_WIDTH      16384
#define AGX_TEXTURE_BUFFER_MAX_HEIGHT 16384
#define AGX_TEXTURE_BUFFER_MAX_SIZE                                            \
   (AGX_TEXTURE_BUFFER_WIDTH * AGX_TEXTURE_BUFFER_MAX_HEIGHT)

/* Apple9 backend texture flags, disjoint from the AGX_TEXTURE_FLAG_* bits */

/* A seamful cube sample, of the faces as layers of the cube's 2D array alias */
#define AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY (1 << 8)
/* Texel fetch whose level, layer and sample are already bounded */
#define AGX_APPLE9_TEXTURE_FETCH_BOUNDED (1 << 10)
/* The final form has no offset encoding, so lower the offset in NIR */
#define AGX_APPLE9_TEXTURE_OFFSET_FALLBACK (1 << 12)
/* Explicit level from the offset lowering, already including the LOD bias */
#define AGX_APPLE9_TEXTURE_BIASED_LEVEL (1 << 13)

struct nir_builder;
struct nir_def;
struct nir_shader;
struct nir_tex_instr;

/* Runtime sampler record field (enum agx_apple9_sampler_field) of tex */
struct nir_def *agx_apple9_sampler_field(struct nir_builder *b,
                                         const struct nir_tex_instr *tex,
                                         unsigned field);

/* Unnormalized coordinates and offsets the instructions cannot encode */
bool agx_nir_lower_apple9_sampler_state_runtime(struct nir_shader *nir);
