/* SPDX-License-Identifier: MIT */
#ifndef AGX_APPLE9_COLOR_H
#define AGX_APPLE9_COLOR_H

#include <stdbool.h>
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/macros.h"

/* Tile representations of Apple9 render targets */

static inline unsigned
agx_apple9_color_components(enum pipe_format format)
{
   return format == PIPE_FORMAT_NONE
             ? 4
             : util_format_description(format)->nr_channels;
}

static inline bool
agx_apple9_color_is_half(enum pipe_format format)
{
   return format == PIPE_FORMAT_R16_FLOAT ||
          format == PIPE_FORMAT_R16G16_FLOAT ||
          format == PIPE_FORMAT_R16G16B16A16_FLOAT;
}

/* The PBE converts float tiles to normalized memory, FP32 for unorm16 */
static inline enum pipe_format
agx_apple9_color_tile_format(enum pipe_format format)
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
agx_apple9_color_words(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE)
      return 1;
   return DIV_ROUND_UP(
      util_format_get_blocksize(agx_apple9_color_tile_format(format)), 4);
}

/* Raw tile accesses preserve format bits, including exact integer outputs */
static inline enum pipe_format
agx_apple9_color_raw_format(unsigned words)
{
   return words == 4   ? PIPE_FORMAT_R32G32B32A32_UINT
          : words == 3 ? PIPE_FORMAT_R32G32B32_UINT
          : words == 2 ? PIPE_FORMAT_R32G32_UINT
                       : PIPE_FORMAT_R32_UINT;
}

static inline bool
agx_apple9_color_is_normalized(enum pipe_format format)
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
agx_apple9_color_is_wide(enum pipe_format format)
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
agx_apple9_color_is_packed(enum pipe_format format)
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

/* Tile formats the PBE exports without expansion, it applies signedness */
static inline unsigned
agx_apple9_block_export_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (format == PIPE_FORMAT_R10G10B10A2_UNORM ||
       format == PIPE_FORMAT_B10G10R10A2_UNORM ||
       format == PIPE_FORMAT_R11G11B10_FLOAT)
      return 7;
   if (!desc->channel[0].pure_integer && desc->channel[0].size == 8)
      return 7;
   switch (desc->channel[0].size) {
   case 8:
      return 5;
   case 16:
      return 1;
   case 32:
      return 3;
   default:
      return ~0u;
   }
}

#endif
