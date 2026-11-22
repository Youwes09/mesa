/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "agx_compile.h"
#include "agx_device.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_serialize.h"
#include "pool.h"

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def, nir_bindless_image_agx(
                     b, nir_imul_imm(b, intr->src[0].ssa, 24), .desc_set = 0));

   return true;
}

static void
agx_lower_bg_eot_shader(nir_shader *shader, struct agx_tilebuffer_layout *tib)
{
   agx_nir_lower_texture(shader);
   agx_preprocess_nir(shader);
   if (tib) {
      unsigned bindless_base = 0;
      NIR_PASS(_, shader, agx_nir_lower_tilebuffer, tib, NULL, &bindless_base,
               NULL, NULL);
      NIR_PASS(_, shader, agx_nir_lower_monolithic_msaa, tib->nr_samples);
      NIR_PASS(_, shader, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, shader, agx_nir_lower_texture);
      NIR_PASS(_, shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
               nir_metadata_control_flow, NULL);
   }
}

static struct agx_bg_eot_shader *
agx_compile_bg_eot_shader(struct agx_bg_eot_cache *cache, nir_shader *shader,
                          struct agx_shader_key *key)
{
   struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
   struct agx_shader_part bin;
   agx_compile_shader_nir(shader, key, &bin);

   res->info = bin.info;
   res->ptr = agx_pool_upload_aligned_with_bo(
      &cache->pool, bin.binary, bin.info.binary_size, 128, &res->bo);
   free(bin.binary);
   ralloc_free(shader);

   return res;
}

static nir_def *
build_background_op(nir_builder *b, enum agx_bg_eot_op op, unsigned rt,
                    unsigned nr, bool msaa, bool layered)
{
   if (op == AGX_BG_LOAD) {
      nir_def *coord = nir_load_pixel_coord(b);

      /* MSAA/array lowerings assume 32-bit coordinates, could be lifted but
       * that's going to need more work.
       */
      if (layered || msaa)
         coord = nir_u2u32(b, coord);

      if (layered) {
         coord = nir_vec3(b, nir_channel(b, coord, 0), nir_channel(b, coord, 1),
                          nir_load_layer_id(b));
      }

      b->shader->info.fs.uses_sample_shading |= msaa;

      nir_def *tex = nir_txf(
         b, coord, .ms_index = msaa ? nir_load_sample_id(b) : NULL,
         .texture_index = rt * 2,
         .dim = msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
         .is_array = layered,

         /* The type doesn't matter as long as it matches the store */
         .dest_type = nir_type_uint32,

         /* Layer is necessarily already in-bounds so we do not want the
          * compiler to clamp it, which would require reading the descriptor
          */
         .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);

      return nir_trim_vector(b, tex, nr);
   } else {
      assert(op == AGX_BG_CLEAR);

      return nir_load_preamble(b, nr, 32, 4 + (rt * 8));
   }
}

static nir_shader *
agx_build_background_shader(const struct agx_bg_eot_key *key,
                            enum agx_shader_isa isa)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, agx_nir_options_for_isa(isa), "agx_background");
   b.shader->info.fs.untyped_color_outputs = true;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      unsigned nr = util_format_get_nr_components(key->tib.logical_format[rt]);
      bool msaa = key->tib.nr_samples > 1;
      bool layered = key->tib.layered;
      assert(nr > 0);

      nir_store_output(
         &b, build_background_op(&b, key->op[rt], rt, nr, msaa, layered),
         nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(nr),
         .src_type = nir_type_uint32,
         .io_semantics.location = FRAG_RESULT_DATA0 + rt,
         .io_semantics.num_slots = 1);

      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   struct agx_tilebuffer_layout tib = key->tib;
   agx_lower_bg_eot_shader(b.shader, &tib);
   return b.shader;
}

/* Apple9 block image stores each render target's tile at the tile origin */
static nir_shader *
agx_build_end_of_tile_shader_apple9(const struct agx_bg_eot_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options_apple9, "agx_eot");

   enum glsl_sampler_dim dim =
      (key->tib.nr_samples > 1) ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;

   assert(key->tib.apple9 && "Apple9 tile layout");
   unsigned height = key->tib.tile_size == 32 * 16 ? 16 : 32;
   nir_def *tile = nir_load_tile_id_agx(&b);
   nir_def *xy = nir_vec2(&b, nir_imul_imm(&b, nir_channel(&b, tile, 0), 32),
                          nir_imul_imm(&b, nir_channel(&b, tile, 1), height));

   nir_def *coords =
      key->tib.layered
         ? nir_vec4(&b, nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
                    nir_load_layer_id(&b), nir_imm_int(&b, 0))
         : nir_pad_vec4(&b, xy);

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE || key->tib.spilled[rt])
         continue;

      assert(key->op[rt] == AGX_EOT_STORE);
      enum pipe_format format = key->tib.logical_format[rt];

      nir_image_store_block_agx(
         &b, nir_imm_int(&b, rt),
         nir_imm_int(&b, agx_tilebuffer_offset_B(&key->tib, rt)), coords,
         .format = agx_apple9_color_tile_format(format), .image_dim = dim,
         .image_array = key->tib.layered);
   }

   b.shader->info.num_images = AGX_MAX_RENDER_TARGETS;
   agx_lower_bg_eot_shader(b.shader, NULL);
   return b.shader;
}

static nir_shader *
agx_build_end_of_tile_shader(const struct agx_bg_eot_key *key,
                             enum agx_shader_isa isa)
{
   if (isa == AGX_SHADER_ISA_APPLE9)
      return agx_build_end_of_tile_shader_apple9(key);

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "agx_eot");

   enum glsl_sampler_dim dim =
      (key->tib.nr_samples > 1) ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      /* The end-of-tile shader is unsuitable to handle spilled render targets.
       * Skip them. If blits are needed with spilled render targets, other parts
       * of the driver need to implement them.
       */
      if (key->tib.spilled[rt])
         continue;

      assert(key->op[rt] == AGX_EOT_STORE);
      unsigned offset_B = agx_tilebuffer_offset_B(&key->tib, rt);

      nir_def *layer = nir_undef(&b, 1, 16);
      if (key->tib.layered)
         layer = nir_u2u16(&b, nir_load_layer_id(&b));

      nir_image_store_block_agx(
         &b, nir_imm_intN_t(&b, rt, 16), nir_imm_int(&b, offset_B), layer,
         .format = agx_tilebuffer_physical_format(&key->tib, rt),
         .image_dim = dim, .image_array = key->tib.layered);
   }

   agx_lower_bg_eot_shader(b.shader, NULL);
   return b.shader;
}

/* See agx_bg_eot.h */
nir_shader *
agx_nir_bg_eot_shader(const struct agx_bg_eot_key *key, enum agx_shader_isa isa,
                      struct agx_shader_key *compiler_key)
{
   bool eot = false;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
      eot |= key->op[rt] == AGX_EOT_STORE;

   *compiler_key = (struct agx_shader_key){
      .dev.shader_isa = isa,
      .reserved_preamble = key->reserved_preamble,
      .fs.ignore_tib_dependencies = !eot,
   };

   return eot ? agx_build_end_of_tile_shader(key, isa)
              : agx_build_background_shader(key, isa);
}

struct agx_bg_eot_shader *
agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                      struct agx_bg_eot_key *key)
{
   struct hash_entry *ent = _mesa_hash_table_search(cache->ht, key);
   if (ent)
      return ent->data;

   struct agx_shader_key compiler_key;
   nir_shader *nir = agx_nir_bg_eot_shader(
      key, agx_gather_device_key(cache->dev).shader_isa, &compiler_key);

   struct agx_bg_eot_shader *ret =
      agx_compile_bg_eot_shader(cache, nir, &compiler_key);

   ret->key = *key;
   _mesa_hash_table_insert(cache->ht, &ret->key, ret);
   return ret;
}

DERIVE_HASH_TABLE(agx_bg_eot_key);

void
agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev)
{
   agx_pool_init(&cache->pool, dev, "Internal programs",
                 AGX_BO_EXEC | AGX_BO_LOW_VA, true);
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->ht = agx_bg_eot_key_table_create(NULL);
   cache->dev = dev;
}

void
agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache)
{
   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      if (cache->precomp[i] && cache->precomp[i]->apple9)
         agx_bo_unreference(cache->dev, cache->precomp[i]->bo);
   }
   agx_pool_cleanup(&cache->pool);
   _mesa_hash_table_destroy(cache->ht, NULL);
   simple_mtx_destroy(&cache->lock);
   cache->ht = NULL;
   cache->dev = NULL;
}

/* Apple9 programs get their own BO. False if the USC aperture is full */
static bool
agx_upload_apple9_precompiled(struct agx_bg_eot_cache *cache,
                              struct agx_precompiled_shader *p,
                              unsigned program)
{
   const uint32_t *bin = cache->dev->libagx_programs[program];
   assert(bin != NULL && "the program never runs on Apple9");

   const struct agx_precompiled_apple9_kernel_info *info = (const void *)bin;
   const uint32_t *local = info->profile.local_size;

   p->apple9 = true;
   p->apple9_profile = info->profile;
   p->b.workgroup = agx_workgroup(local[0], local[1], local[2]);
   p->bo = agx_bo_create(cache->dev, info->binary_size, 0,
                         AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                         "Apple9 internal shader");
   if (!p->bo)
      return false;

   memcpy(agx_bo_map(p->bo), info + 1, info->binary_size);
   p->ptr = p->bo->va->addr;
   return true;
}

static struct agx_precompiled_shader *
agx_get_precompiled_locked(struct agx_bg_eot_cache *cache, unsigned program)
{
   simple_mtx_assert_locked(&cache->lock);

   /* It is possible that, while waiting for the lock, another thread uploaded
    * the shader. Check for that so we don't double-upload.
    */
   if (cache->precomp[program])
      return cache->precomp[program];

   /* Otherwise, we need to upload. */
   struct agx_precompiled_shader *p =
      rzalloc(cache->ht, struct agx_precompiled_shader);

   if (agx_is_apple9(cache->dev)) {
      if (!agx_upload_apple9_precompiled(cache, p, program)) {
         ralloc_free(p);
         return NULL;
      }
      p_atomic_set(&cache->precomp[program], p);
      return p;
   }

   const uint32_t *bin = cache->dev->libagx_programs[program];
   const struct agx_precompiled_kernel_info *info = (void *)bin;
   const void *binary = (const uint8_t *)bin + sizeof(*info);

   assert(info->main_offset == 0 || program != LIBAGX_HELPER);

   p->b.workgroup =
      agx_workgroup(info->workgroup_size[0], info->workgroup_size[1],
                    info->workgroup_size[2]);

   p->ptr = agx_pool_upload_aligned_with_bo(&cache->pool, binary,
                                            info->binary_size, 128, &p->bo);

   /* Bake launch */
   agx_pack(&p->b.launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = 8 /* same encoding as 0 */;
      cfg.sampler_state_register_count = 1;
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   /* Bake USC */
   struct agx_usc_builder b =
      agx_usc_builder(p->b.usc.data, sizeof(p->b.usc.data));

   agx_usc_immediates(&b, &info->rodata, p->ptr);

   if (info->uses_txf)
      agx_usc_push_packed(&b, SAMPLER, cache->dev->txf_sampler);

   agx_usc_shared(&b, info->local_size, info->imageblock_stride, 0);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(cache->dev, p->ptr + info->main_offset);
      cfg.unk_2 = 3;
   }

   agx_usc_pack(&b, REGISTERS, cfg) {
      cfg.register_count = info->nr_gprs;
      cfg.spill_size = 0;
   }

   if (info->nr_preamble_gprs) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(cache->dev, p->ptr + info->preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   p->b.usc.size = b.head - p->b.usc.data;

   /* We must only write to the cache once we are done compiling, since other
    * threads may be reading the cache concurrently. Do this last.
    */
   p_atomic_set(&cache->precomp[program], p);
   return p;
}

struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program)
{
   /* Shaders are immutable once written, so if we atomically read a non-NULL
    * shader, then we have a valid cached shader and are done.
    */
   struct agx_precompiled_shader *ret = p_atomic_read(cache->precomp + program);

   if (ret != NULL)
      return ret;

   /* Otherwise, take the lock and upload. */
   simple_mtx_lock(&cache->lock);
   ret = agx_get_precompiled_locked(cache, program);
   simple_mtx_unlock(&cache->lock);

   return ret;
}

uint64_t
agx_helper_program(struct agx_bg_eot_cache *cache)
{
   struct agx_precompiled_shader *pc =
      agx_get_precompiled(cache, LIBAGX_HELPER);

   return pc->ptr | 1;
}
