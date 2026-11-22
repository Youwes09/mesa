/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;
struct nir_instr;
struct nir_builder;
struct nir_def;
struct agx_apple9_compute_profile;

bool agx_nir_lower_address(struct nir_shader *shader);
bool agx_nir_lower_sink_address(struct nir_shader *shader);
bool agx_nir_lower_algebraic_late(struct nir_shader *shader);
bool agx_nir_cleanup_amul(struct nir_shader *shader);
bool agx_nir_fuse_lea(struct nir_shader *shader);
bool agx_nir_lower_lea(struct nir_shader *shader);
bool agx_nir_fuse_selects(struct nir_shader *shader);
bool agx_nir_fuse_algebraic_late(struct nir_shader *shader);
bool agx_nir_fence_images(struct nir_shader *shader);
bool agx_nir_lower_layer(struct nir_shader *s);
bool agx_nir_lower_clip_distance(struct nir_shader *s);
bool agx_nir_lower_subgroups(struct nir_shader *s);
bool agx_nir_lower_fminmax(struct nir_shader *s);
bool agx_nir_lower_accurate_frcp(struct nir_shader *s);

bool agx_nir_lower_texture_early(struct nir_shader *s, bool support_lod_bias,
                                 bool software_buffer_bounds);
bool agx_nir_lower_texture(struct nir_shader *s);
bool agx_nir_lower_images(struct nir_shader *s);
bool agx_nir_lower_texture_queries(struct nir_shader *s);
bool agx_nir_lower_multisampled_image_store(struct nir_shader *s);
bool agx_nir_needs_texture_crawl(struct nir_instr *instr);

/* Apple9 lowerings for FP32 subnormal flushing, conversions and sine */
bool agx_nir_lower_apple9_math(struct nir_shader *shader);
bool agx_nir_opt_apple9_bounds(struct nir_shader *shader);
bool agx_nir_lower_apple9_compute_derivatives(struct nir_shader *shader);
bool agx_nir_lower_apple9_denorms(struct nir_shader *shader);
bool agx_nir_lower_apple9_denorm_preserve(struct nir_shader *shader);
bool agx_nir_lower_apple9_subgroup_denorms(struct nir_shader *shader);
bool agx_nir_legalize_apple9_memory(struct nir_shader *nir);
bool agx_nir_legalize_apple9_alu(struct nir_shader *nir);
unsigned agx_nir_lower_apple9_abi(struct nir_shader *nir, bool graphics,
                                  struct agx_apple9_compute_profile *profile);
bool agx_nir_preload_apple9_push(struct nir_shader *nir, unsigned first,
                                 unsigned count);
bool agx_nir_lower_apple9_coverage(struct nir_shader *nir, bool linked);
unsigned
agx_apple9_shared_root(const struct agx_apple9_compute_profile *profile);
struct nir_def *agx_nir_sincos_apple9(struct nir_builder *b, struct nir_def *x,
                                      bool cosine);

#ifdef __cplusplus
} /* extern C */
#endif
