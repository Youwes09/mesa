/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "agx_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum render targets per framebuffer. This is NOT architectural, but it
 * is the ~universal API limit so there's no point in allowing more.
 */
#define AGX_MAX_RENDER_TARGETS (8)

/* Forward declarations to keep the header lean */
struct nir_shader;
struct nir_def;
struct agx_isa_target;
struct nir_builder;

struct agx_tilebuffer_layout {
   /* Logical format of each render target. Use agx_tilebuffer_physical_format
    * to get the physical format.
    */
   enum pipe_format logical_format[AGX_MAX_RENDER_TARGETS];

   /* Which render targets are spilled. */
   bool spilled[AGX_MAX_RENDER_TARGETS];

   /* Offset into the sample of each render target. If a render target is
    * spilled, its offset is UNDEFINED. Use agx_tilebuffer_offset_B to access.
    */
   uint8_t _offset_B[AGX_MAX_RENDER_TARGETS];

   /* Total bytes per sample, rounded up as needed. Spilled render targets do
    * not count against this.
    */
   uint8_t sample_size_B;

   /* Number of samples per pixel */
   uint8_t nr_samples;

   /* If layered rendering is used */
   bool layered;

   /* Selected tile size in pixels. Either 32*32, 32*16, or 16*16. */
   uint16_t tile_size;

   /* Render targets are whole tile-format words, accessed raw */
   bool raw_tiles;

   /* USC word corresponding to this configuration of the tilebuffer */
   struct agx_usc_shared_packed usc;
};

static inline void
agx_tilebuffer_set_drm_cmd(struct drm_asahi_cmd_render *c,
                           const struct agx_tilebuffer_layout *layout)
{
   c->utile_width_px = layout->tile_size >= 32 * 16 ? 32 : 16;
   c->utile_height_px = layout->tile_size <= 32 * 16 ? 16 : 32;
}

/*
 * _offset_B is undefined for non-spilled render targets. This safe accessor
 * asserts that render targets are not spilled rather than returning garbage.
 */
static inline uint8_t
agx_tilebuffer_offset_B(struct agx_tilebuffer_layout *layout, unsigned rt)
{
   assert(rt < AGX_MAX_RENDER_TARGETS);
   assert(!layout->spilled[rt] && "precondition");

   return layout->_offset_B[rt];
}

static inline bool
agx_tilebuffer_spills(struct agx_tilebuffer_layout *layout)
{
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (layout->spilled[rt])
         return true;
   }

   return false;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered);

struct agx_tilebuffer_layout
agx_build_raw_tilebuffer_layout(const enum pipe_format *formats,
                                uint8_t nr_cbufs, uint8_t nr_samples,
                                bool layered);

void agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                                unsigned nr_samples);

bool agx_nir_lower_tilebuffer(struct nir_shader *shader,
                              struct agx_tilebuffer_layout *tib,
                              uint8_t *colormasks, unsigned *bindless_base,
                              struct nir_def *write_samples, bool *translucent);

bool agx_nir_lower_to_per_sample(struct nir_shader *shader);

bool agx_nir_lower_monolithic_msaa(struct nir_shader *shader,
                                   const struct agx_isa_target *target,
                                   uint8_t nr_samples);

bool agx_nir_lower_sample_intrinsics(struct nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);

uint32_t agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib);

enum pipe_format
agx_tilebuffer_physical_format(struct agx_tilebuffer_layout *tib, unsigned rt);

bool agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib,
                                  unsigned rt);

void agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib);

#ifdef __cplusplus
} /* extern C */
#endif

/* Tile representations of render targets in raw tile layouts */

static inline bool
agx_tile_format_is_half(enum pipe_format format)
{
   return format == PIPE_FORMAT_R16_FLOAT ||
          format == PIPE_FORMAT_R16G16_FLOAT ||
          format == PIPE_FORMAT_R16G16B16A16_FLOAT;
}

/* The PBE converts float tiles to normalized memory, FP32 for unorm16 */
static inline enum pipe_format
agx_raw_tile_format(enum pipe_format format)
{
   switch (format) {
   /* Tiles are RGBA, the PBE descriptor permutes the channels on export */
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return PIPE_FORMAT_R8G8B8X8_UNORM;
   case PIPE_FORMAT_B8G8R8A8_SRGB:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return PIPE_FORMAT_R8G8B8X8_SRGB;
   case PIPE_FORMAT_B8G8R8A8_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   case PIPE_FORMAT_B8G8R8A8_UINT:
      return PIPE_FORMAT_R8G8B8A8_UINT;
   case PIPE_FORMAT_B8G8R8A8_SINT:
      return PIPE_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UINT:
      return PIPE_FORMAT_R16G16B16A16_UINT;
   /* Padded 10- and 12-bit formats export like 16-bit normalized ones */
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_X6R10_UNORM:
   case PIPE_FORMAT_X4R12_UNORM:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_X6R10X6G10_UNORM:
   case PIPE_FORMAT_X4R12X4G12_UNORM:
      return PIPE_FORMAT_R32G32_FLOAT;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_X6R10X6G10X6B10X6A10_UNORM:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   default:
      return format;
   }
}

static inline unsigned
agx_raw_tile_words(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE)
      return 1;
   return DIV_ROUND_UP(util_format_get_blocksize(agx_raw_tile_format(format)),
                       4);
}

/* Raw tile accesses preserve format bits, including exact integer outputs */
static inline enum pipe_format
agx_raw_tile_access_format(unsigned words)
{
   return words == 4   ? PIPE_FORMAT_R32G32B32A32_UINT
          : words == 3 ? PIPE_FORMAT_R32G32B32_UINT
          : words == 2 ? PIPE_FORMAT_R32G32_UINT
                       : PIPE_FORMAT_R32_UINT;
}

static inline bool
agx_tile_format_is_normalized(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
       (desc->nr_channels != 1 && desc->nr_channels != 2 &&
        desc->nr_channels != 4) ||
       !desc->channel[0].normalized ||
       (desc->channel[0].size != 8 && desc->channel[0].size != 16))
      return false;

   for (unsigned c = 1; c < desc->nr_channels; ++c) {
      if (desc->channel[c].size != desc->channel[0].size ||
          (desc->channel[c].type != UTIL_FORMAT_TYPE_VOID &&
           (!desc->channel[c].normalized ||
            desc->channel[c].type != desc->channel[0].type)))
         return false;
   }

   return true;
}

static inline bool
agx_tile_format_is_wide(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   return desc->layout == UTIL_FORMAT_LAYOUT_PLAIN &&
          (desc->nr_channels == 1 || desc->nr_channels == 2 ||
           desc->nr_channels == 4) &&
          ((desc->channel[0].pure_integer &&
            (desc->channel[0].size == 8 || desc->channel[0].size == 16 ||
             desc->channel[0].size == 32)) ||
           (desc->channel[0].type == UTIL_FORMAT_TYPE_FLOAT &&
            desc->channel[0].size == 32));
}

static inline bool
agx_raw_tile_is_packed(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UINT:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}
