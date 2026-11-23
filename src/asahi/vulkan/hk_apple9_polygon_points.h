/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "compiler/nir/nir.h"
#include "poly/nir/poly_nir.h"

struct nir_builder;

/* Points a clipped triangle can emit: its 3 vertices plus one per clip
 * plane (w, near, far and up to 8 clip distances). */
#define HK_PP_MAX_VERTICES (3 + 3 + 8)

struct hk_pp_info {
   unsigned clip, cull;
   /* Slots to emit besides position and clip/cull distances. */
   uint64_t outputs;
   const unsigned *output_components;
   /* Flat inputs of the fragment shader. */
   uint64_t flat;
   /* The polygon slot (back-facing, primitive ID), or 0. */
   unsigned face_slot;
};

/* Load components of a slot of the triangle's logical vertex 0, 1 or 2. */
typedef nir_def *(*hk_pp_load_fn)(struct nir_builder *b, void *data,
                                  unsigned vertex, unsigned location,
                                  unsigned offset, unsigned component,
                                  unsigned comps, nir_alu_type type);

/* Emit the triangle's points with emit_vertex, or with only set, store the
 * outputs of its point number only, placed behind the eye if it has fewer. */
void hk_apple9_pp_emit_triangle(struct nir_builder *b,
                                const struct hk_pp_info *info,
                                hk_pp_load_fn load, void *data,
                                nir_def *provoking, nir_def *prim,
                                nir_def *only);

/* The passthrough geometry shader drawing triangles as polygon points */
void hk_apple9_polygon_points_gs(struct nir_builder *b,
                                 const struct poly_passthrough_gs_key *key,
                                 uint64_t flat, unsigned face_slot);

/* Read the facing and primitive ID of polygon points from the polygon slot */
bool hk_nir_lower_polygon_point_fs(nir_shader *fs, unsigned slot);

/* The polygon slot: the polygon's back-facing and primitive ID. Producers
 * cannot write the last generic slot (the output limit stops before it), and
 * it shares the slot of the line stipple word, as a draw draws lines or
 * points.
 */
#define HK_POLYGON_POINT_SLOT VARYING_SLOT_VAR31
