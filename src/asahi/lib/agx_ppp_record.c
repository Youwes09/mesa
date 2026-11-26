/*
 * Copyright 2026 Niklas Sheth
 * Copyright 2026 Youwes09
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_record.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/u_math.h"
#include "agx_draw_table.h"

/* The default coefficient table occupies the end of the graphics resource
 * page, after per-draw argument records. Draw-specific tables live in their
 * own slots on the following pages so mixed interpolation can use one
 * descriptor per scalar without overwriting the next draw's roots. */
#define DEFAULT_CF_TABLE 0x203d00u

static unsigned
cf_binding_count(unsigned components)
{
   return 1 + DIV_ROUND_UP(components, 4);
}

static void
build_direct_bind_group(uint8_t *page, unsigned varying_components)
{
   static const struct {
      uint16_t offset;
      uint32_t value;
   } common[] = {
      {0x00, 0x00800000}, {0x04, 0x00010100}, {0x08, 0x0000c9c0},
      {0x10, 0x01000000}, {0x14, 0x00066420}, {0x1c, 0x0c0a0000},
      {0x20, 0x00010000}, {0x2c, 0x00000006}, {0x30, 0x010000b4},
      {0x34, 0x00040200}, {0x38, 0x07200f00}, {0x3c, 0x0e000000},
      {0x40, 0x07200f00}, {0x44, 0x0e000000}, {0x4c, 0x02000048},
      {0x50, 0x00000200}, {0x54, 0x07e00000}, {0x58, 0x07e00000},
      {0x5c, 0x0000000f}, {0x60, 0x00410000}, {0x68, 0x00000080},
      {0x6c, 0x00200000}, {0x70, 0x00000480},
   };

   memset(page, 0, 0x80);
   for (unsigned i = 0; i < ARRAY_SIZE(common); ++i)
      put_u32(page + common[i].offset, common[i].value);

   /* T8140 direct-render deltas reused by G16. */
   put_u32(page + 0x04, 0);
   put_u32(page + 0x08, 0);
   put_u32(page + 0x14, 0x00004e19);
   put_u32(page + 0x20, 0);
   put_u32(page + 0x2c, 4);
   put_u32(page + 0x5c, 0x0001ffff);

   /* Small G16 packing deltas established independently on T8132. */
   page[0x05] = 0x01;
   page[0x06] = 0x02;
   page[0x08] = 0x80;
   page[0x09] = 0x04;
   page[0x15] = 0x8c;
   page[0x22] = 0x01;
   assert(varying_components <= AGX_MAX_EXPORTED_COMPONENTS);
   page[0x2c] = 4 + varying_components;
   /* G16's fragment-state layout moves the coefficient pointer ahead of
    * the pipeline words. */
   put_u32(page + 0x04, 0x100 | (cf_binding_count(varying_components) << 16));
   put_u32(page + 0x08, DEFAULT_CF_TABLE);
   put_u32(page + 0x18, varying_components / 8);
}

/* The PPP record bytes that depend only on the linked stages: header,
 * varying group counts and the direct bind group with its stage-derived bits.
 * Draw-dependent words (coefficients, coefficient count, fragment launch,
 * tile allocation, raster, depth, stencil and viewports) are applied per draw
 * on top. */
void
agx_build_ppp_static(const struct agx_draw_pipeline *pipeline,
                     uint8_t out[AGX_PPP_STATIC_SIZE])
{
   uint8_t *ppp = out;
   memset(ppp, 0, AGX_PPP_STATIC_SIZE);
   put_u32(ppp, 0x10040000);
   unsigned counts[3] = {0};
   const struct agx_varying_layout *varyings = &pipeline->vertex->varyings;
   for (unsigned i = 0; i < ARRAY_SIZE(varyings->mask); ++i) {
      assert(varyings->group[i] < ARRAY_SIZE(counts));
      counts[varyings->group[i]] += util_bitcount(varyings->mask[i]);
   }
   put_u32(ppp + 4, counts[0] | (counts[1] << 8) | (counts[2] << 16));
   uint8_t *group = ppp + 0x40;
   build_direct_bind_group(group, pipeline->vertex->varying_components);
   if (pipeline->fragment->reads_z)
      put_u32(group + 0x20, get_u32(group + 0x20) | (1u << 21));
   if (pipeline->fragment->reads_primitive_id)
      put_u32(group + 0x68, get_u32(group + 0x68) | (1u << 12));
   if (pipeline->vertex->writes_point_size) {
      put_u32(group + 0x20, get_u32(group + 0x20) | (1u << 18));
   }
   if (pipeline->vertex->writes_layer_viewport)
      put_u32(group + 0x20, get_u32(group + 0x20) | (3u << 19));
   put_u32(group + 0x20,
           get_u32(group + 0x20) |
              BITFIELD_MASK(pipeline->vertex->clip_distance_count));
   put_u32(group + 0x2c, 4 + pipeline->vertex->varying_components +
                            pipeline->vertex->writes_point_size +
                            pipeline->vertex->clip_distance_count +
                            pipeline->vertex->writes_layer_viewport);
   /* Match the textured setup's native state, including with two user
    * varyings. This field's full resource-count formula is unresolved;
    * the varying-only estimate is insufficient to describe the captures. */
   put_u32(group + 0x18, MAX2(get_u32(group + 0x18), 1));
}
