/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_COMPILE_APPLE9_H
#define AGX_COMPILE_APPLE9_H

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_apple9_color.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "util/blend.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "agx_apple9_profile.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Whether tex's texel offset has a native immediate encoding: a constant
 * offset in [-8, 7] on a 2D (non-array) or 3D sample, gather or comparison.
 * If so, *bits receives x | y << 4 | z << 8 in signed 4-bit fields. */
struct nir_tex_instr;
bool agx_apple9_texture_native_offset(const struct nir_tex_instr *tex,
                                      uint16_t *bits);

/* The sample mode word (instruction bytes 6-7) for an ordinary sample form,
 * and whether that form has a measured offset variant (returned in
 * *offset_mode). Shared by the packer and the backend's offset fallback. */
uint16_t agx_apple9_texture_mode(bool explicit_lod, bool gradient, bool bias,
                                 unsigned dimension, bool shadow, bool min_lod);
bool agx_apple9_texture_offset_mode(uint16_t mode, uint16_t *offset_mode);

/* Draw-time compiles: allow one helper thread for independent RA candidates. */
void agx_apple9_set_parallel_ra(bool enable);

bool agx_nir_hoist_apple9_uniform_reads(nir_shader *shader);

/* Schedule texture operations without increasing estimated scalar-word
 * pressure. */
bool agx_nir_hoist_apple9_samples(nir_shader *shader);
/* While set on this thread (compiling a spill probe), register allocation
 * keeps its first candidate. */
void agx_apple9_ra_set_probe(bool probe);
/* Free the first allocations kept from the last probe. */
void agx_apple9_ra_release_probe_results(void);
bool agx_apple9_ra_is_probe(void);
bool agx_nir_hoist_apple9_derivative_sources(nir_shader *shader);

/* After generic texture lowering, backend1 holds a packed Apple9 LOD/bias
 * word rather than the original floating-point LOD or integer fetch level. */
bool agx_nir_lower_apple9_texture_lod(nir_shader *shader);

/* agx_compile_apple9_tiny's reason when a compute program has no observable
 * effect; the caller may run nothing for it instead of failing. */
extern const char agx_apple9_compute_without_effects[];

/*
 * Compile compute NIR through Apple9's semantic instruction selection and
 * register allocator. Arithmetic, memory operations, and structured control
 * flow share the same lowering used by graphics shaders. Unsupported NIR or
 * resource requirements return a diagnostic; no shader replay is substituted.
 *
 * On failure, *reason points at a static diagnostic string when reason is
 * non-NULL. On success, out owns a malloc-backed main/preamble binary and may
 * be released with free(out->binary), like agx_compile_shader_nir output.
 */
bool agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                             struct agx_apple9_compute_profile *profile,
                             const char **reason);

/*
 * Compile graphics NIR through the common semantic VIR and register allocator.
 * Vertex attributes and fragment varyings are packed by semantic location and
 * component; fragment variants use the producer's layout. Smooth inputs use
 * coefficient-aware perspective multiplication. Flat and noperspective inputs
 * have separate coefficient modes. Centroid and sample inputs evaluate at
 * an allocated position selected from coverage or the requested sample index.
 * The common sample loop lowers per-sample shading and output masks.
 *
 * Buffer resources use a shader-loaded address table. Texture and sampler
 * bindings are compacted independently. Cube projection and integer fetches
 * select native operations; queries and explicit gradients use ordinary NIR
 * lowering.
 * Structured branches and loops use the common execution-mask model. Fragment
 * outputs include format-aware color stores, blending, and depth export.
 *
 * The caller supplies hardware clip coordinates and compatible stage and
 * render-target state. The single-target entry point below retains RGBA8
 * defaults; the MRT entry point takes explicit attachment formats and state.
 */
bool agx_compile_apple9_fragment(nir_shader *nir, struct agx_shader_part *out,
                                 const char **reason);



/* Standard independent RGB/alpha blending, plus the RGBA write mask. */
struct agx_apple9_blend {
   uint8_t rgb_src, rgb_dst, alpha_src, alpha_dst;
   uint8_t rgb_func, alpha_func, colormask, logicop_func;
   /* NONE retains the RGBA8 layout used by standalone compiler callers. */
   enum pipe_format format;
   uint8_t samples; /* Zero retains single-sample compiler callers. */
   uint8_t disabled_samples;
   uint8_t alpha_to_coverage       : 1;
   uint8_t alpha_to_one            : 1;
   uint8_t multisample_disabled    : 1;
   uint8_t color_attachment_unused : 1;
   /* Vulkan sample-mask writes also apply to single-sample rendering. */
   uint8_t sample_mask_single_sample : 1;
   uint8_t logicop_enable            : 1;
   /* Writes are also gated by the attachment's bit in the per-draw color
    * write enable sysval. */
   uint8_t dynamic_enable : 1;
   /* Sample positions are read from the per-draw sysval rather than the
    * standard pattern, for programmable sample locations. */
   uint8_t dynamic_sample_positions : 1;
   uint8_t advanced_mode, advanced_overlap;
   uint8_t src_premultiplied, dst_premultiplied;
   /* Optional per-draw depth range for shader-triggered early tests. */
   bool clamp_early_depth;
   uint8_t early_depth_clamp_ubo;
   /* GL's by-location dual-source workaround, only set on RT0 when SRC1
    * blending is active. This is part of the fragment variant key. */
   bool dual_source_by_location;
   uint32_t early_depth_clamp_offset_B;
};

static inline bool
agx_apple9_blend_uses_src1(const struct agx_apple9_blend *blend)
{
   if (blend->logicop_enable || blend->advanced_mode)
      return false;

   const uint8_t factors[] = {
      blend->rgb_src,
      blend->rgb_dst,
      blend->alpha_src,
      blend->alpha_dst,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      enum pipe_blendfactor factor =
         util_blendfactor_without_invert((enum pipe_blendfactor)factors[i]);
      if (factor == PIPE_BLENDFACTOR_SRC1_COLOR ||
          factor == PIPE_BLENDFACTOR_SRC1_ALPHA)
         return true;
   }
   return false;
}

enum agx_apple9_sampler_flags {
   AGX_APPLE9_CUSTOM_BORDER = 1 << 0,
   AGX_APPLE9_CLAMP_SHADOW_REFERENCE = 1 << 1,
};

struct agx_apple9_sampler_key {
   uint32_t flags;
   float border[4];
   uint8_t wrap[3];
   uint8_t min_filter, mag_filter, mip_filter, compare_func, seamless_cube_map;
   float min_lod, max_lod, lod_bias;
   unsigned max_anisotropy;
};

void agx_nir_legalize_apple9_texture_sources(nir_shader *nir);
struct nir_builder;
struct nir_tex_instr;
struct nir_def *agx_apple9_texture_info(struct nir_builder *b,
                                        const struct nir_tex_instr *tex,
                                        unsigned field);
bool agx_nir_lower_apple9_runtime_sampler_state(nir_shader *nir,
                                                const char **reason);
/* A NULL key reads the runtime sampler metadata; otherwise specialize to key. */


bool agx_nir_lower_apple9_texture_offsets(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   const char **reason);

struct agx_apple9_texture_mapping {
   uint8_t samplers[32];
   uint32_t white_samplers;
   uint32_t seamful_cubes;
   /* Samplers used by texel-offset samples. Offsets the sample instruction
    * cannot encode are lowered with runtime sampler state, so the driver
    * must fill these runtime sampler records of the graphics sysval table. */
   uint32_t runtime_samplers;
};

/* Mask of sampler indices sampled with texel offsets; all samplers when an
 * offset sample has no sampler index yet. */
uint32_t agx_nir_apple9_offset_samplers(const nir_shader *nir);


/* Old-compiler texture flags */
#define AGX_APPLE9_TEXTURE_LOD_QUERY     (1 << 9)
#define AGX_APPLE9_TEXTURE_HANDLE_INDEX (1 << 11)
#define AGX_APPLE9_TEXTURE_CUBE_SEAMLESS (1 << 14)

bool agx_nir_lower_apple9_sampler_state(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   struct agx_apple9_texture_mapping *mapping, const char **reason);

/* Per-target blend array contains MAX2(nr_targets, 1) entries, up to eight.
 * Entry zero also carries multisample state for depth-only rendering. */
bool agx_compile_apple9_fragment_mrt(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, unsigned nr_targets,
   struct agx_shader_part *out, const char **reason);

bool agx_compile_apple9_fragment_blend(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, struct agx_shader_part *out,
   const char **reason);

/* Specialize fragment coefficient reads to the producing vertex layout. */
bool agx_compile_apple9_fragment_inputs(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   struct agx_shader_part *out, const char **reason);

/* With AGX_APPLE9_ASM, print the shader's final hardware-state info as one
 * APPLE9_INFO line, so identity checks see state the program bytes don't
 * carry. Call it once the driver has made its last change to the info. */
void agx_apple9_dump_info(const struct agx_shader_info *info);

bool agx_apple9_vertex_format_supported(enum pipe_format format);

/* Vertex pulling uses ordinary typed buffer loads in the API main. Buffer
 * addresses remain draw state; format, stride and attribute offset specialize
 * the fetch. Multiple attributes sharing a binding use one resource argument. */
/* Private bindings for the software transform-feedback vertex job. Public
 * Apple9 vertex shaders do not expose SSBOs; these use the normal resource ABI.
 */
#define AGX_APPLE9_XFB_PARAMS      26
#define AGX_APPLE9_XFB_BUFFER_BASE 27
#define AGX_APPLE9_XFB_INDICES     31

struct agx_apple9_xfb_params {
   uint32_t first_vertex, index_bias, base_instance, draw_id;
   uint32_t first_capture, vertices_per_instance, instance_bias, reserved;
};

/* Instance-rate attribute whose every instance reads the first instance's
 * element.
 */
#define AGX_APPLE9_VERTEX_DIVISOR_ZERO UINT32_MAX

struct agx_apple9_vertex_layout {
   uint64_t outputs_flat, outputs_linear;
   uint32_t stride[16];
   uint32_t divisor[16]; /* Zero selects the per-vertex stream. */
   /* UBO containing first vertex/base vertex and base instance, respectively.
    * It may point into an indirect descriptor written by the GPU. */
   uint16_t draw_params_ubo;
   /* Optional table of {uint64_t address, uint64_t size} per buffer binding.
    * Nonzero selects bounded global fetches, with full-width byte offsets. */
   uint16_t buffer_descriptor_ubo;
   /* The buffer table instead holds {uint64_t address, uint64_t size,
    * uint32_t stride, uint32_t pad[3]} per binding, and stride[] is ignored. */
   bool dynamic_stride;
   /* Vertex input robustness is disabled: descriptor-table fetches skip the
    * bounds check (an out-of-bounds attribute is undefined). Zero, the
    * default, keeps robust fetches. Requires dynamic_stride, and the driver
    * must describe a null binding as the zero page with size and stride 0. */
   bool unbounded;
   bool clip_halfz; /* Convert GL [-w,w] depth to hardware [0,w]. */
   uint8_t clip_distance_enable;
   bool cull_distance_mask; /* Cull signs are combined by software geometry. */
   bool capture_xfb;
   bool rasterize_points;
   uint8_t xfb_mode;
   uint8_t xfb_index_size;
   bool xfb_flatshade_first;
   bool
      ignore_point_size; /* Valid only when rasterizing non-point primitives. */
   enum pipe_format format[16];
   uint32_t offset[16]; /* Attribute byte offset within its vertex buffer. */
   uint8_t
      buffer[16]; /* Gallium vertex-buffer binding, independent of location. */
};

bool agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason);

/* The same vertex fetch lowering is used by hardware VS and Poly compute VS. */
bool agx_nir_lower_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   bool software);

/* Compile the bounded procedural vertex stage described above. */
bool agx_compile_apple9_vertex(nir_shader *nir, struct agx_shader_part *out,
                               const char **reason);

#ifdef __cplusplus
}
#endif

#endif
