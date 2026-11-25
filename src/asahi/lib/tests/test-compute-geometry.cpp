/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_cdm.h"
#include "asahi/lib/agx_launch_program.h"
#include "asahi/lib/agx_apple9_layout.h"

#include <cstring>
#include <gtest/gtest.h>

static uint32_t
get_u32(const uint8_t *bytes, size_t offset)
{
   uint32_t value;
   memcpy(&value, bytes + offset, sizeof(value));
   return value;
}

static uint64_t
get_u64(const uint8_t *bytes, size_t offset)
{
   uint64_t value;
   memcpy(&value, bytes + offset, sizeof(value));
   return value;
}

TEST(ComputeGeometry, DirectPublishesGroupCounts)
{
   uint8_t record[0x100] = {};
   const uint64_t address = 0x1000200000ull;
   const agx_cdm_geometry geometry = {
      .mode = AGX_CDM_GEOMETRY_DIRECT,
      .threads = {12, 10, 6},
      .local = {4, 5, 3},
   };
   ASSERT_TRUE(agx_cdm_build_geometry_fields(record, sizeof(record),
                                                        address, &geometry));
   EXPECT_EQ(get_u64(record, 0x00), address + 0xc0);
   for (unsigned d = 0; d < 3; ++d) {
      EXPECT_EQ(get_u32(record, 0xc0 + d * 4),
                geometry.threads[d] / geometry.local[d]);
   }
}

TEST(ComputeGeometry, IndirectPublishesCallerRecord)
{
   uint8_t record[0x100];
   memset(record, 0xa5, sizeof(record));
   const uint64_t address = 0x1000200000ull;
   const agx_cdm_geometry geometry = {
      .mode = AGX_CDM_GEOMETRY_INDIRECT,
      .group_counts = 0x2fff123400ull,
      .local = {4, 4, 1},
   };
   ASSERT_TRUE(agx_cdm_build_geometry_fields(record, sizeof(record),
                                                        address, &geometry));
   /* Every visible pointer remains intact even at the expanded capacity. */
   for (unsigned i = 0x08; i < 0xc0; ++i)
      EXPECT_EQ(record[i], 0xa5);
   for (unsigned i = 0xcc; i < sizeof(record); ++i)
      EXPECT_EQ(record[i], 0xa5);
   EXPECT_EQ(get_u64(record, 0x00), geometry.group_counts);
   for (unsigned d = 0; d < 3; ++d) {
      EXPECT_EQ(get_u32(record, 0xc0 + d * 4), 0u);
   }
}

TEST(ComputeGeometry, RejectsMalformedIndirectGeometry)
{
   uint8_t record[0x100] = {};
   agx_cdm_geometry geometry = {
      .mode = AGX_CDM_GEOMETRY_INDIRECT,
      .group_counts = 0x2fff123402ull,
      .local = {4, 4, 1},
   };
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   geometry.group_counts &= ~3ull;
   geometry.local[2] = 65;
   geometry.local[1] = 17;
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, 0xcb, 0x1000200000ull, &geometry));
}

TEST(ComputeGeometry, RejectsMalformedDirectGeometry)
{
   uint8_t record[0x100] = {};
   agx_cdm_geometry geometry = {
      .mode = AGX_CDM_GEOMETRY_DIRECT,
      .threads = {12, 10, 0},
      .local = {4, 5, 3},
   };
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));

   geometry.threads[2] = 6;
   geometry.local[0] = 1024;
   geometry.local[1] = 2;
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));

   geometry.mode = (agx_cdm_geometry_mode)99;
   EXPECT_FALSE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
}

TEST(ComputeGeometry, RoundsPartialGroupsBeyondApiAxisLimit)
{
   uint8_t record[0x100] = {};
   agx_cdm_geometry geometry = {
      .mode = AGX_CDM_GEOMETRY_DIRECT,
      .threads = {65535 * 7, 11, 1},
      .local = {7, 5, 3},
   };
   ASSERT_TRUE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_EQ(get_u32(record, 0xc0), 65535u);
   EXPECT_EQ(get_u32(record, 0xc4), 3u);
   EXPECT_EQ(get_u32(record, 0xc8), 1u);
   /* Internal dispatches use 32-bit counts, beyond the API's 65535 limit. */
   geometry.threads[0]++;
   ASSERT_TRUE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_EQ(get_u32(record, 0xc0), 65536u);
   geometry.threads[0] = UINT32_MAX;
   ASSERT_TRUE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_EQ(get_u32(record, 0xc0), 613566757u);
   geometry.local[0] = 1;
   ASSERT_TRUE(agx_cdm_build_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_EQ(get_u32(record, 0xc0), UINT32_MAX);
}

/* As agx_build_tilebuffer_layout, multisampled zero-target renders keep a
 * nonempty per-sample tile allocation. */
TEST(TileAllocation, MultisampledZeroTargetTileAllocation)
{
   EXPECT_EQ(agx_launch_tile_bytes(0, 1), 0u);
   EXPECT_EQ(agx_launch_tile_bytes(0, 2), 8u);
   EXPECT_EQ(agx_launch_tile_bytes(0, 4), 8u);
   EXPECT_EQ(agx_launch_tile_bytes(4, 2), 4u);
   EXPECT_EQ(agx_launch_tile_bytes(24, 4), 24u);
}
