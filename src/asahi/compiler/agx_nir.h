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
struct agx_compute_profile;
struct agx_isa_target;

bool agx_nir_lower_address(struct nir_shader *shader,
                           const struct agx_isa_target *target);
bool agx_nir_lower_sink_address(struct nir_shader *shader);
bool agx_nir_lower_algebraic_late(struct nir_shader *shader, bool words);
bool agx_nir_cleanup_amul(struct nir_shader *shader);
bool agx_nir_fuse_lea(struct nir_shader *shader);
bool agx_nir_lower_lea(struct nir_shader *shader);
bool agx_nir_fuse_selects(struct nir_shader *shader);
bool agx_nir_fuse_algebraic_late(struct nir_shader *shader, bool words);
bool agx_nir_fence_images(struct nir_shader *shader);
bool agx_nir_lower_layer(struct nir_shader *s);
bool agx_nir_lower_clip_distance(struct nir_shader *s);
bool agx_nir_lower_subgroups(struct nir_shader *s,
                             const struct agx_isa_target *target);
bool agx_nir_lower_fminmax(struct nir_shader *s);
bool agx_nir_lower_accurate_frcp(struct nir_shader *s);

bool agx_nir_lower_texture_early(struct nir_shader *s,
                                 const struct agx_isa_target *target,
                                 bool support_lod_bias,
                                 bool software_buffer_bounds);
bool agx_nir_lower_texture(struct nir_shader *s,
                           const struct agx_isa_target *target);
bool agx_nir_lower_images(struct nir_shader *s);
bool agx_nir_lower_texture_queries(struct nir_shader *s);
bool agx_nir_lower_multisampled_image_store(struct nir_shader *s);
bool agx_nir_needs_texture_crawl(struct nir_instr *instr);

bool agx_nir_lower_compute_derivatives(struct nir_shader *shader);

/* FP32 subnormal flushing and preservation, for flushes_denorms */
bool agx_nir_flush_denorms(struct nir_shader *shader);
bool agx_nir_flush_subgroup_denorms(struct nir_shader *shader);
bool agx_nir_lower_denorm_preserve(struct nir_shader *shader);

/* Word registers and memory: conversions, sine, narrow arithmetic and access */
bool agx_nir_lower_word_math(struct nir_shader *shader,
                             const struct agx_isa_target *target);
bool agx_nir_lower_word_alu(struct nir_shader *nir);
bool agx_nir_lower_word_memory(struct nir_shader *nir);
struct nir_def *agx_nir_sincos_fp32(struct nir_builder *b, struct nir_def *x,
                                    bool cosine);
bool agx_nir_lower_handle_offsets(struct nir_shader *nir);
struct nir_def *agx_nir_descriptor_address(struct nir_builder *b,
                                           struct nir_def *handle,
                                           unsigned offset);
unsigned agx_nir_lower_root_uniforms(struct nir_shader *nir, bool graphics,
                                     struct agx_compute_profile *profile);
bool agx_nir_preload_push(struct nir_shader *nir, unsigned first,
                          unsigned count);
bool agx_nir_lower_indirect_io(struct nir_shader *nir);

/* User varying components, limited to AGX_MAX_EXPORTED_COMPONENTS */
unsigned agx_user_varying_components(const struct nir_shader *nir);
bool agx_varyings_fit(const struct nir_shader *nir);
bool agx_nir_lower_killed_samples(struct nir_shader *nir, bool load_killed,
                                  bool export_killed);
unsigned agx_shared_root(const struct agx_compute_profile *profile);

#ifdef __cplusplus
} /* extern C */
#endif
