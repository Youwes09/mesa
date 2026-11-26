/*
 * Copyright 2026 Niklas Sheth
 * Copyright 2026 Youwes09
 * SPDX-License-Identifier: MIT
 */

#include "agx_usc_descriptor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/layout/layout.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"
#include "agx_helpers.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"

bool
agx_translate_usc_pbe(void *out, const struct agx_pbe_packed *state,
                      enum pipe_format format, const struct ail_layout *layout,
                      uint64_t layer_base, unsigned level, unsigned layer_count,
                      bool layered, unsigned dimension)
{
   if (dimension > 15 || !layout->levels || layout->levels > 16 ||
       level >= layout->levels || !layout->width_px || !layout->height_px ||
       layout->width_px > 16384 || layout->height_px > 16384 || !layer_count ||
       layer_count > 65536)
      return false;

   bool linear = layout->tiling == AIL_TILING_LINEAR;
   if (linear && (level || layer_count > 1 || layout->compressed))
      return false;
   unsigned descriptor_level = linear ? level : 0;
   uint64_t base = layer_base + layout->level_offsets_B[descriptor_level] -
                   layout->level_offsets_B[0];
   if ((base & 15) || (base >> 44))
      return false;
   uint32_t stride = linear ? ail_get_linear_stride_B(layout, level) : 0;
   if (linear &&
       (!stride || (stride & 15) || stride > AGX_USC_MAX_LINEAR_STRIDE))
      return false;

   agx_unpack(stderr, state, PBE, pbe);
   if (layout->compressed &&
       (!pbe.compressed || !pbe.extended || !pbe.acceleration_buffer ||
        (pbe.acceleration_buffer & 15) || (pbe.acceleration_buffer >> 48)))
      return false;

   /* Packed tile words already have memory channel order. */
   bool packed = agx_raw_tile_is_packed(agx_raw_tile_format(format));

   agx_pack(out, PBE_G16, cfg) {
      cfg.dimension = dimension;
      cfg.layout = pbe.layout;
      cfg.channels = pbe.channels;
      cfg.type = pbe.type;
      cfg.swizzle_r = packed ? AGX_CHANNEL_R : pbe.swizzle_r;
      cfg.swizzle_g = packed ? AGX_CHANNEL_G : pbe.swizzle_g;
      cfg.swizzle_b = packed ? AGX_CHANNEL_B : pbe.swizzle_b;
      cfg.swizzle_a = packed ? AGX_CHANNEL_A : pbe.swizzle_a;
      cfg.width = u_minify(layout->width_px, descriptor_level);
      cfg.height = u_minify(layout->height_px, descriptor_level);
      cfg.samples = pbe.samples;
      cfg.compressed = layout->compressed;
      cfg.buffer = base;

      if (layout->compressed) {
         cfg.extended = true;
         cfg.acceleration_buffer = pbe.acceleration_buffer;
      }

      if (!linear && layout->levels > 1) {
         cfg.unk_mipmapped = true;
         cfg.extended = true;
         cfg.level = level;
         cfg.levels = layout->levels;
      }

      if (linear) {
         cfg.stride = stride / 16;
      } else if (layered) {
         cfg.layers = layer_count;
         cfg.page_aligned_layers = layout->page_aligned_layers;
      }
   }

   return true;
}

bool
agx_usc_texture_format_supported(enum pipe_format format)
{
   if (agx_tile_format_is_wide(format) || agx_raw_tile_is_packed(format) ||
       agx_tile_format_is_normalized(format))
      return true;
   switch (format) {
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8X8_SNORM:
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_Z32_FLOAT:
   /* The transfer helper stores these as depth32f and separate stencil;
    * sampling the logical combined format reads its depth component. */
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_Z16_UNORM:
   /* BC block formats sample natively, as in Honeykrisp. Without them Gallium
    * decompresses S3TC/RGTC to RGBA8 on upload and cannot offer BPTC. */
   case PIPE_FORMAT_DXT1_RGB:
   case PIPE_FORMAT_DXT1_RGBA:
   case PIPE_FORMAT_DXT1_SRGB:
   case PIPE_FORMAT_DXT1_SRGBA:
   case PIPE_FORMAT_DXT3_RGBA:
   case PIPE_FORMAT_DXT3_SRGBA:
   case PIPE_FORMAT_DXT5_RGBA:
   case PIPE_FORMAT_DXT5_SRGBA:
   case PIPE_FORMAT_RGTC1_UNORM:
   case PIPE_FORMAT_RGTC1_SNORM:
   case PIPE_FORMAT_RGTC2_UNORM:
   case PIPE_FORMAT_RGTC2_SNORM:
   case PIPE_FORMAT_BPTC_RGB_FLOAT:
   case PIPE_FORMAT_BPTC_RGB_UFLOAT:
   case PIPE_FORMAT_BPTC_RGBA_UNORM:
   case PIPE_FORMAT_BPTC_SRGBA:
      return true;
   default:
      return false;
   }
}

static unsigned
usc_sampler_address(unsigned wrap)
{
   /* Native descriptor address codes, captured for each axis. Wrapping in
    * hardware preserves filtering across seams. */
   switch (wrap) {
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      return 0;
   case PIPE_TEX_WRAP_REPEAT:
      return 1;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      return 2;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      return 3;
   case PIPE_TEX_WRAP_CLAMP:
      return 4;
   /* Mirrored clamp-to-edge descriptor code, from change-one-parameter
    * descriptor captures (agx-re docs/descriptors). */
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      return 5;
   default:
      UNREACHABLE("unsupported sampler address mode");
   }
}

static struct AGX_SAMPLER_G16
usc_sampler(bool min_linear, bool mag_linear, unsigned mip_filter,
            float min_lod, float max_lod, unsigned wrap_s, unsigned wrap_t,
            unsigned max_anisotropy)
{
   assert(mip_filter <= 2);
   max_lod = CLAMP(max_lod, 0.0f, 14.0f);
   min_lod = CLAMP(min_lod, 0.0f, max_lod);

   /* Round the LODs to the nearest 1/64. Fractional maximum LODs blend into
    * the next level, as a biased trilinear sample clamped to 2.1 shows. */
   return (struct AGX_SAMPLER_G16){
      AGX_SAMPLER_G16_header,
      .minimum_lod = roundf(min_lod * 64) / 64,
      .maximum_lod = roundf(max_lod * 64) / 64,
      .maximum_anisotropy =
         util_next_power_of_two(CLAMP(max_anisotropy, 1, 16)),
      .magnify = mag_linear ? AGX_FILTER_LINEAR : AGX_FILTER_NEAREST,
      .minify = min_linear ? AGX_FILTER_LINEAR : AGX_FILTER_NEAREST,
      .mip_filter = mip_filter,
      .wrap_s = usc_sampler_address(wrap_s),
      .wrap_t = usc_sampler_address(wrap_t),
      .compare_func = AGX_COMPARE_FUNC_NEVER,
      .unk_42 = true,
   };
}

void
agx_pack_basic_usc_sampler(void *out, bool min_linear, bool mag_linear,
                           unsigned mip_filter, float min_lod, float max_lod,
                           unsigned wrap_s, unsigned wrap_t,
                           unsigned max_anisotropy)
{
   struct AGX_SAMPLER_G16 sampler =
      usc_sampler(min_linear, mag_linear, mip_filter, min_lod, max_lod, wrap_s,
                  wrap_t, max_anisotropy);
   AGX_SAMPLER_G16_pack(out, &sampler);
}

bool
agx_translate_usc_texture(void *out, const struct agx_texture_packed *state)
{
   agx_unpack(stderr, state, TEXTURE, tex);
   return agx_pack_usc_texture(out, &tex);
}

void
agx_set_null_usc_texture(void *out)
{
   /* A one-texel, zero-swizzled view of the zero page, as agx_set_null_texture */
   agx_pack(out, TEXTURE_G16, cfg) {
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.channels = AGX_CHANNELS_R8;
      cfg.type = AGX_TEXTURE_TYPE_UNORM /* don't care */;
      cfg.swizzle_r = AGX_CHANNEL_0;
      cfg.swizzle_g = AGX_CHANNEL_0;
      cfg.swizzle_b = AGX_CHANNEL_0;
      cfg.swizzle_a = AGX_CHANNEL_0;
      cfg.address = AGX_ZERO_PAGE_ADDRESS;
   }
}

bool
agx_pack_usc_texture(void *out, const struct AGX_TEXTURE *state)
{
   struct AGX_TEXTURE tex = *state;
   if ((tex.address & 15) || (tex.address >> 48))
      return false;

   if (tex.dimension == AGX_TEXTURE_DIMENSION_3D &&
       tex.depth > AGX_USC_MAX_3D_DIMENSION)
      return false;

   bool single_layer_linear =
      tex.layout == AGX_LAYOUT_LINEAR && tex.extended &&
      (tex.dimension == AGX_TEXTURE_DIMENSION_1D ||
       tex.dimension == AGX_TEXTURE_DIMENSION_2D ||
       tex.dimension == AGX_TEXTURE_DIMENSION_1D_ARRAY ||
       tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY) &&
      tex.depth_linear == 1;
   if ((tex.extended && !tex.compressed && !single_layer_linear) ||
       (tex.compressed &&
        (!tex.extended || tex.layout == AGX_LAYOUT_LINEAR ||
         !tex.acceleration_buffer || (tex.acceleration_buffer & 15) ||
         (tex.acceleration_buffer >> 48))) ||
       tex.mode != AGX_IMAGE_MODE_NORMAL ||
       (tex.layout != AGX_LAYOUT_LINEAR && tex.layout != AGX_LAYOUT_TWIDDLED &&
        tex.layout != AGX_LAYOUT_GPU))
      return false;

   bool array = tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
                tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ||
                tex.dimension == AGX_TEXTURE_DIMENSION_1D_ARRAY;
   bool ms = tex.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
             tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;

   /* Cube arrays keep their native dimension, with the layer count field
    * holding the number of cubes. */
   array |= tex.dimension == AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
   /* A single linear layer needs no extended array stride. Preserve the
    * dimension expected by the shader, including array coordinates. */
   if (single_layer_linear) {
      tex.extended = false;
      /* The legacy extended flag overlaps the unpacked stride's top bit. */
      tex.stride &= BITFIELD_MASK(21);
      tex.depth = 1;
   }
   agx_pack(out, TEXTURE_G16, cfg) {
      cfg.dimension = tex.dimension;
      cfg.layout = tex.layout;
      cfg.channels = tex.channels;
      cfg.type = tex.type;
      cfg.swizzle_r = tex.swizzle_r;
      cfg.swizzle_g = tex.swizzle_g;
      cfg.swizzle_b = tex.swizzle_b;
      cfg.swizzle_a = tex.swizzle_a;
      cfg.width = tex.width;
      cfg.height = tex.height;
      cfg.samples = ms ? tex.samples : AGX_SAMPLE_COUNT_2;
      cfg.unk_mipmapped = tex.unk_mipmapped;
      cfg.compressed = tex.compressed;
      cfg.address = tex.address;
      cfg.srgb = tex.srgb;
      cfg.srgb_2_channel = tex.srgb_2_channel;

      if (tex.layout == AGX_LAYOUT_LINEAR) {
         cfg.stride = tex.stride;
      } else {
         if (array || tex.dimension == AGX_TEXTURE_DIMENSION_3D)
            cfg.depth = tex.depth;

         cfg.page_aligned_layers = tex.page_aligned_layers;
      }

      cfg.extended = tex.unk_mipmapped || tex.compressed;
      if (tex.compressed)
         cfg.acceleration_buffer = tex.acceleration_buffer;

      cfg.first_level = tex.first_level;
      cfg.last_level = tex.last_level;
   }

   return true;
}

bool
agx_pack_usc_sampler(void *out, const struct AGX_SAMPLER *state)
{
   const struct AGX_SAMPLER sampler = *state;
   if (sampler.magnify > AGX_FILTER_LINEAR ||
       sampler.minify > AGX_FILTER_LINEAR ||
       sampler.mip_filter > AGX_MIP_FILTER_LINEAR ||
       sampler.wrap_s > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.wrap_t > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.wrap_r > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.border_colour == AGX_BORDER_COLOUR_CUSTOM ||
       sampler.pixel_coordinates)
      return false;

   const unsigned wrap[] = {
      PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_REPEAT,
      PIPE_TEX_WRAP_MIRROR_REPEAT, PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      PIPE_TEX_WRAP_CLAMP,         PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE,
   };
   struct AGX_SAMPLER_G16 g16 = usc_sampler(
      sampler.minify == AGX_FILTER_LINEAR, sampler.magnify == AGX_FILTER_LINEAR,
      sampler.mip_filter, sampler.minimum_lod, sampler.maximum_lod,
      wrap[sampler.wrap_s], wrap[sampler.wrap_t], sampler.maximum_anisotropy);
   g16.wrap_r = sampler.wrap_r;
   g16.border_colour = sampler.border_colour;
   g16.seamful_cube_maps = sampler.seamful_cube_maps;
   if (sampler.compare_enable)
      g16.compare_func = sampler.compare_func;

   memset(out, 0, AGX_USC_SAMPLER_STRIDE);
   AGX_SAMPLER_G16_pack(out, &g16);
   return true;
}

/* Texel fetches return zero outside the image, as image robustness
 * requires: clamp every axis to the transparent black border.
 */
void
agx_pack_usc_fetch_sampler(void *out)
{
   struct AGX_SAMPLER_G16 sampler =
      usc_sampler(false, false, 1, 0, 14, PIPE_TEX_WRAP_CLAMP_TO_BORDER,
                  PIPE_TEX_WRAP_CLAMP_TO_BORDER, 1);
   sampler.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
   AGX_SAMPLER_G16_pack(out, &sampler);
}
