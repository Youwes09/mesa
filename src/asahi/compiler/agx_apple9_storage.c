/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_storage.h"
#include "util/format/u_format.h"

/* Per component: offset [7:0], shift [12:8], width - 1 [17:13], kind [21:18] */
void
agx_apple9_texel_layout(enum pipe_format format,
                        struct agx_apple9_texel_metadata *m)
{
   const struct util_format_description *desc = util_format_description(format);
   m->size_B = util_format_get_blocksize(format);
   for (unsigned c = 0; c < 4; ++c) {
      unsigned swizzle = desc->swizzle[c];
      unsigned kind, offset = 0, shift = 0, bits = 1;
      if (swizzle >= 4) {
         kind = swizzle == PIPE_SWIZZLE_1 ? AGX_APPLE9_TEXEL_ONE
                                          : AGX_APPLE9_TEXEL_ZERO;
      } else {
         const struct util_format_channel_description *chan =
            &desc->channel[swizzle];
         offset = (chan->shift / 32) * 4;
         shift = chan->shift % 32;
         bits = chan->size;
         if (format == PIPE_FORMAT_R9G9B9E5_FLOAT)
            kind = AGX_APPLE9_TEXEL_RGB9E5;
         else if (format == PIPE_FORMAT_R11G11B10_FLOAT)
            kind = AGX_APPLE9_TEXEL_FLOAT11;
         else if (chan->type == UTIL_FORMAT_TYPE_FLOAT)
            kind =
               bits == 16 ? AGX_APPLE9_TEXEL_FLOAT16 : AGX_APPLE9_TEXEL_FLOAT32;
         else if (chan->type == UTIL_FORMAT_TYPE_SIGNED)
            kind = chan->normalized ? AGX_APPLE9_TEXEL_SNORM
                                    : AGX_APPLE9_TEXEL_SINT;
         else
            kind = chan->normalized ? AGX_APPLE9_TEXEL_UNORM
                                    : AGX_APPLE9_TEXEL_UINT;
         if (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB && c < 3)
            kind = AGX_APPLE9_TEXEL_SRGB;
      }
      m->component[c] =
         offset | (shift << 8) | ((bits - 1) << 13) | (kind << 18);
   }
}

enum agx_apple9_store_encode
agx_apple9_store_encode(enum pipe_format format)
{
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return AGX_APPLE9_STORE_ENCODE_FLOAT11;
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB || !desc->nr_channels)
      return AGX_APPLE9_STORE_ENCODE_GENERIC;
   const struct util_format_channel_description *first = &desc->channel[0];
   unsigned w = first->size;
   for (unsigned c = 0; c < desc->nr_channels; ++c) {
      const struct util_format_channel_description *chan = &desc->channel[c];
      if (desc->swizzle[c] != c || chan->size != w || chan->shift != c * w ||
          chan->type != first->type || chan->normalized != first->normalized ||
          chan->pure_integer != first->pure_integer)
         return AGX_APPLE9_STORE_ENCODE_GENERIC;
   }
   if (w == 32 && !first->normalized &&
       (first->type == UTIL_FORMAT_TYPE_FLOAT ||
        first->type == UTIL_FORMAT_TYPE_UNSIGNED ||
        first->type == UTIL_FORMAT_TYPE_SIGNED))
      return AGX_APPLE9_STORE_ENCODE_RAW32;
   if (w == 8 && first->normalized && first->type == UTIL_FORMAT_TYPE_UNSIGNED)
      return AGX_APPLE9_STORE_ENCODE_UNORM8;
   if (w == 16 && first->type == UTIL_FORMAT_TYPE_FLOAT)
      return AGX_APPLE9_STORE_ENCODE_FLOAT16;
   return AGX_APPLE9_STORE_ENCODE_GENERIC;
}

bool
agx_apple9_native_store_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R11G11B10_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
      return true;
   default:
      return false;
   }
}
