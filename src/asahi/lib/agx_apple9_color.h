/* SPDX-License-Identifier: MIT */
#ifndef AGX_APPLE9_COLOR_H
#define AGX_APPLE9_COLOR_H

#include <stdbool.h>
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/macros.h"

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
