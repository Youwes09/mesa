/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "asahi/genxml/agx_pack.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "util/format/u_formats.h"

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

/* Storage image and texel buffer descriptors of texture_exports targets:
 * drivers fill them, and agx_nir_lower_storage.c reads them through texture
 * handles.
 */

/* The texture and PBE words every storage descriptor begins with */
struct agx_apple9_storage_words {
   struct agx_texture_packed tex;
   struct agx_pbe_packed pbe;
};
static_assert(sizeof(struct agx_apple9_storage_words) == 48, "no holes");

/* Conversion of a texel component, in bits 21:18 of its layout word */
enum agx_apple9_texel_kind {
   AGX_APPLE9_TEXEL_ZERO,
   AGX_APPLE9_TEXEL_ONE,
   AGX_APPLE9_TEXEL_UINT,
   AGX_APPLE9_TEXEL_SINT,
   AGX_APPLE9_TEXEL_UNORM,
   AGX_APPLE9_TEXEL_SNORM,
   AGX_APPLE9_TEXEL_FLOAT16,
   AGX_APPLE9_TEXEL_FLOAT32,
   AGX_APPLE9_TEXEL_FLOAT11,
   AGX_APPLE9_TEXEL_RGB9E5,
   AGX_APPLE9_TEXEL_SRGB,
};

/* Storage images: base is the residency table, elements level | layer << 16 */
struct agx_apple9_texel_metadata {
   uint64_t base;
   uint32_t elements;
   uint32_t size_B;
   uint32_t component[4];
   /* Sparse buffer page residency, indexed by address / AIL_PAGESIZE, or 0 */
   uint64_t residency;
};

/* Uniform texel buffer, with a native texture entry as in sampled images */
struct agx_apple9_texel_descriptor {
   struct agx_texture_packed tex;
   struct agx_apple9_texel_metadata metadata;
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
};
static_assert(sizeof(struct agx_apple9_texel_descriptor) == 96,
              "Apple9 texel descriptor has no holes");
static_assert(offsetof(struct agx_apple9_texel_descriptor, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");

enum agx_apple9_storage_native_flags {
   AGX_APPLE9_STORAGE_READ_VALID = 1 << 0,
   AGX_APPLE9_STORAGE_WRITE_VALID = 1 << 1,
   AGX_APPLE9_STORAGE_LAYER_WRITE_VALID = 1 << 2,
};

/* Texel layout for formatless access, and a read-only native texture entry */
struct agx_apple9_storage_descriptor {
   struct agx_apple9_storage_words common;
   struct agx_apple9_texel_metadata metadata;
   /* enum agx_apple9_storage_native_flags */
   uint32_t native_ok;
   /* enum agx_apple9_store_encode of the view format */
   uint32_t store_encode;
   union {
      /* Native texture entry, indexed by its USC address (Honeykrisp) */
      uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
      /* Or its index from the texture root (Gallium) */
      uint32_t texture_index;
   };
};
static_assert(sizeof(struct agx_apple9_storage_descriptor) == 128,
              "Apple9 storage descriptor retains the 128-byte ABI");
static_assert(offsetof(struct agx_apple9_storage_descriptor, texture) ==
                 (AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY + 1) *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native storage entry is one table slot past the image one");

/* native_ok: flags in bits 0-2, PBE index from the texture root in 3-29 */
#define AGX_APPLE9_STORAGE_PBE_SHIFT 3
#define AGX_APPLE9_STORAGE_PBE_MASK  ((1u << 27) - 1)

void agx_apple9_texel_layout(enum pipe_format format,
                             struct agx_apple9_texel_metadata *metadata);

/* Formatless stores: constant shifts for canonical formats, else the layout */
enum agx_apple9_store_encode {
   AGX_APPLE9_STORE_ENCODE_GENERIC = 0,
   /* 32-bit float, uint or sint channels, stored as they are */
   AGX_APPLE9_STORE_ENCODE_RAW32,
   /* 8-bit unsigned normalized channels */
   AGX_APPLE9_STORE_ENCODE_UNORM8,
   /* 16-bit float channels */
   AGX_APPLE9_STORE_ENCODE_FLOAT16,
   /* R11G11B10 unsigned float */
   AGX_APPLE9_STORE_ENCODE_FLOAT11,
};

enum agx_apple9_store_encode agx_apple9_store_encode(enum pipe_format format);

/* Whether the PBE stores this format from 32-bit data */
bool agx_apple9_native_store_format(enum pipe_format format);

/* Sparse image records: residency table address, then level | layer << 16 */
#define AGX_APPLE9_DESCRIPTOR_RESIDENCY                                        \
   (AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO + AGX_APPLE9_TEXTURE_INFO_STRIDE)

/* Raster-order tile pages from page; the mip tail has no tiles_per_row */
struct agx_apple9_residency_level {
   uint32_t page;
   uint16_t tiles_per_row;
   uint8_t shift_x, shift_y;
};

struct agx_apple9_residency {
   uint32_t layer_pages;
   uint32_t pages;
   uint32_t pad[2];
   struct agx_apple9_residency_level level[AIL_MAX_MIP_LEVELS];
   /* One byte per page of each layer in turn, nonzero if bound */
   uint8_t resident[];
};
static_assert(sizeof(struct agx_apple9_residency) == 144,
              "Apple9 residency header is packed");

/* table_indices: descriptors hold texture_index rather than a texture entry */
bool agx_nir_lower_storage(struct nir_shader *nir, bool table_indices,
                           bool reinterpret);
bool agx_nir_lower_sparse_residency(struct nir_shader *nir);
bool agx_nir_lower_texel_buffers(struct nir_shader *nir);

/* Run after agx_preprocess_nir, once storage images are in device memory */
bool agx_nir_lower_image_barriers(struct nir_shader *nir);

/* Whether a shader can use native PBE stores: no image read can follow one */
bool agx_native_store_shader(const struct nir_shader *nir);

/* Storage texel encoding, by store encoding or layout; exposed for tests */
struct nir_def *agx_encode_texel_fast(struct nir_builder *b,
                                      struct nir_def *value,
                                      struct nir_def *layout, bool floating,
                                      struct nir_def *encode);
