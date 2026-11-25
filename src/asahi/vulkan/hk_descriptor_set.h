/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_descriptor_memory.h"
#include "hk_private.h"

#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_usc_descriptor.h"
#include "asahi/lib/agx_apple9_color.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "hk_device.h"
#include "vk_descriptor_update_template.h"
#include "vk_object.h"

#include "util/list.h"
#include "util/vma.h"

struct hk_descriptor_set_layout;

struct hk_sampled_image_descriptor {
   struct agx_texture_packed tex;
   struct agx_sampler_packed sampler;
   uint16_t sampler_index;

   /* Negative if there is no border colour, else the clamp=0 sampler index used
    * for custom border colour emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   /* We want LOD bias and min LOD together as fp16 in this order and 32-bit
    * aligned. This lets us keep everything vectorized on the shader side, since
    * this matches the AGX2 lod_bias + min_lod descriptor.
    */
   uint16_t lod_bias_fp16;
   uint16_t min_lod_fp16;

   /* This is only used for txf, it can be separate from the other stuff */
   uint16_t min_lod_uint16;

   /* Added to explicit LODs, see hk_lower_d3d_txl_rounding */
   uint16_t txl_nudge_fp16;
   union {
      /* TODO: This should probably be a heap! */
      uint32_t border[4];

      /* Apple9 has no custom border colors. Instead, the view records the
       * residency table of its image and its base subresource.
       */
      struct {
         uint32_t residency_lo, residency_hi;
         uint16_t base_level, base_layer;
         uint32_t pad;
      } apple9;
   };
   uint8_t pad2[4];
};
static_assert(sizeof(struct hk_sampled_image_descriptor) == 64,
              "hk_sampled_image_descriptor has no holes");

struct hk_storage_image_descriptor {
   struct agx_texture_packed tex;
   struct agx_pbe_packed pbe;
};
static_assert(sizeof(struct hk_storage_image_descriptor) == 48,
              "hk_storage_image_descriptor has no holes");
static_assert(sizeof(struct hk_storage_image_descriptor) ==
                 sizeof(struct agx_apple9_storage_words),
              "Apple9 storage descriptors begin with it");

/* G16 descriptor records, 128 bytes each so translation layers can copy
 * them with their 128-byte paths: an image part and a sampler part. A
 * combined image sampler is the image part followed by the sampler part.
 * Shaders address each part through its own handle; the native entries keep
 * the table slots AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY and
 * AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY past their part's base, as the texel
 * and storage descriptors do.
 */
struct hk_apple9_image_part {
   /* Same layout as the per-draw texture info record. */
   uint32_t texture_info[AGX_APPLE9_TEXTURE_INFO_STRIDE / 4];
   /* Sparse residency table and the view's base level and layer. */
   uint32_t residency_lo, residency_hi;
   uint16_t base_level, base_layer;
   uint16_t min_lod_fp16, min_lod_uint16;
   uint8_t pad0[16];
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   /* 2D array alias of cube views (agx_apple9_layout.h), zero otherwise */
   uint8_t cube_alias[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_image_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_image_part, residency_lo) ==
                 AGX_APPLE9_DESCRIPTOR_RESIDENCY,
              "residency table where shaders read it");
static_assert(offsetof(struct hk_apple9_image_part, texture_info) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO,
              "texture info where shaders read it");
static_assert(offsetof(struct hk_apple9_image_part, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");
static_assert(offsetof(struct hk_apple9_image_part, cube_alias) ==
                 offsetof(struct hk_apple9_image_part, texture) +
                    AGX_APPLE9_CUBE_ALIAS_HANDLE_OFFSET,
              "cube alias where the compiler reads it");

struct hk_apple9_sampler_part {
   /* Same layout as the per-draw runtime sampler record. */
   uint32_t runtime_sampler[AGX_APPLE9_RUNTIME_SAMPLER_STRIDE / 4];
   uint16_t lod_bias_fp16, txl_nudge_fp16;
   uint8_t pad0[28];
   uint8_t sampler[AGX_APPLE9_SAMPLER_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_sampler_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_sampler_part, runtime_sampler) ==
                 AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER,
              "runtime sampler where shaders read it");
static_assert(offsetof(struct hk_apple9_sampler_part, sampler) ==
                 AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY *
                    AGX_APPLE9_SAMPLER_TABLE_STRIDE,
              "native sampler entry is where register indices expect it");

/* A combined image sampler is an image part followed by a sampler part; a
 * multi-plane descriptor repeats the whole record per plane. */
#define HK_APPLE9_COMBINED_DESCRIPTOR_SIZE                                     \
   (sizeof(struct hk_apple9_image_part) + sizeof(struct hk_apple9_sampler_part))
#define HK_MAX_SAMPLED_DESCRIPTOR_SIZE (3 * HK_APPLE9_COMBINED_DESCRIPTOR_SIZE)

void hk_apple9_texture_info(uint32_t *data,
                            const struct agx_texture_packed *state);
bool hk_apple9_runtime_sampler(uint32_t *data,
                               const struct agx_sampler_packed *state,
                               bool clamp_to_edge);
/* A custom border sampler takes its packed colour from border. */
bool hk_apple9_native_sampler(void *out, const struct agx_sampler_packed *state,
                              bool clamp_to_edge, const uint32_t *border);

/* This has to match nir_address_format_64bit_bounded_global */
struct hk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

struct hk_device;
struct hk_descriptor_set_layout;
void hk_descriptor_pack_embedded_samplers(
   struct hk_device *dev, const struct hk_descriptor_set_layout *layout,
   void *map);

struct hk_descriptor_pool {
   struct vk_object_base base;

   struct list_head sets;

   struct hk_descriptor_memory mem;

   /* VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT: never read by the GPU. */
   bool host_only;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

struct hk_descriptor_set {
   struct vk_object_base base;

   /* Link in hk_descriptor_pool::sets */
   struct list_head link;

   struct hk_descriptor_set_layout *layout;
   void *mapped_ptr;
   uint64_t addr;
   uint32_t size;
   struct hk_descriptor_allocation alloc;

   struct hk_buffer_address dynamic_buffers[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

static inline uint64_t
hk_descriptor_set_addr(const struct hk_descriptor_set *set)
{
   return set->addr;
}

struct hk_push_descriptor_set {
   uint8_t data[HK_PUSH_DESCRIPTOR_SET_SIZE];
   /* Byte extent of the current layout, including padding between bindings. */
   uint32_t size;
};

void hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                                   struct hk_descriptor_set_layout *layout,
                                   uint32_t write_count,
                                   const VkWriteDescriptorSet *writes);

void hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data);
