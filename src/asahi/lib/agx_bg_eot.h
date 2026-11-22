/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_apple9_profile.h"
#include "asahi/compiler/agx_compile.h"
#include "util/simple_mtx.h"
#include "agx_tilebuffer.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "pool.h"

struct agx_precompiled_shader {
   struct agx_shader b;
   struct agx_bo *bo;
   uint64_t ptr;
   bool apple9;
   struct agx_apple9_compute_profile apple9_profile;
};

struct agx_bg_eot_cache {
   struct agx_device *dev;
   struct agx_pool pool;
   simple_mtx_t lock;

   /* Map from agx_bg_eot_key to agx_bg_eot_shader */
   struct hash_table *ht;

   struct agx_precompiled_shader *precomp[LIBAGX_NUM_PROGRAMS];
};

/*
 * Get a precompiled shader, uploading if necessary. This is thread-safe.
 */
struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program);

/*
 * Get the address of the cached helper program. This is thread-safe.
 */
uint64_t agx_helper_program(struct agx_bg_eot_cache *cache);

enum agx_bg_eot_op {
   AGX_BG_EOT_NONE,
   AGX_BG_CLEAR,
   AGX_BG_LOAD,
   AGX_EOT_STORE,
};

struct agx_bg_eot_key {
   struct agx_tilebuffer_layout tib;
   enum agx_bg_eot_op op[8];
   unsigned reserved_preamble;
};

struct agx_bg_eot_shader {
   struct agx_bg_eot_key key;
   struct agx_shader_info info;
   struct agx_bo *bo;
   uint64_t ptr;
};

struct agx_bg_eot_shader *agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                                                struct agx_bg_eot_key *key);

/* The tile program, lowered for agx_compile_shader_nir with the key it fills */
struct nir_shader *agx_nir_bg_eot_shader(const struct agx_bg_eot_key *key,
                                         enum agx_shader_isa isa,
                                         struct agx_shader_key *compiler_key);

void agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev);
void agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache);
