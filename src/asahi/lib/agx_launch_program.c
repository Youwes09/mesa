/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_launch_program.h"
#include "asahi/compiler/agx3_emit.h"
#include "asahi/lib/agx_launch_abi.h"
#include "asahi/lib/agx_resource_layout.h"

#include <stdio.h>
#include <string.h>

#include "util/macros.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_draw_table.h"
#include "agx_shader_body.h"
#include "pool.h"

/* First of the registers that receive the root words. */
#define LAUNCH_PENDING_BASE 18

static void
write16(uint8_t *p, uint16_t value)
{
   p[0] = value;
   p[1] = value >> 8;
}

bool
agx_launch_threadgroup_memory_supported(uint32_t bytes)
{
   return bytes == 0 || (bytes >= 128 && bytes <= AGX_LAUNCH_MAX_SHARED_BYTES &&
                         !(bytes & (bytes - 1)));
}

size_t
agx_launch_call_offset(enum agx_launch_stage stage, unsigned resource_count)
{
   if (stage == AGX_LAUNCH_VERTEX || stage == AGX_LAUNCH_FRAGMENT)
      return 68;
   if (stage != AGX_LAUNCH_COMPUTE || !resource_count ||
       resource_count > AGX_COMPUTE_DIRECT_MAX_RESOURCES)
      return 0;
   /* The table address literals, one literal per root word and the
    * allocation, then the entry field inside the target. */
   return 26 + 8 * agx_compute_root_words(resource_count);
}

static void
encode_allocation(uint8_t out[8], enum agx_launch_stage stage,
                  const struct agx_launch_parameters *params)
{
   memset(out, 0, 8);
   out[0] = 0x77;
   out[2] = 0x2a;
   out[3] = 0x41;
   if (stage == AGX_LAUNCH_COMPUTE && params->threadgroup_memory_bytes) {
      write16(out + 4, (params->threadgroup_memory_bytes << 2) | 0x80);
      out[6] = params->threadgroup_memory_bytes >> 14;
   } else if (stage == AGX_LAUNCH_FRAGMENT) {
      unsigned units = (params->tile_bytes + 7) / 8;
      if (params->samples == 4) {
         out[2] = 0xaa;
         out[3] = 2 * units;
         out[4] = 0x43;
      } else {
         out[3] = (params->samples == 1 ? 0x41 : 0x40) + 2 * units;
         out[4] = 0x4b;
      }
   }
}

static bool
encode_target(uint8_t out[10], enum agx_launch_stage stage,
              uint64_t shader_base, uint32_t entry_offset)
{
   if (entry_offset & 1)
      return false;
   if (entry_offset > AGX_ENTRY_MAX_OFFSET)
      return false;
   /* The header and fixed control fields were constructed from bit-forcing
    * and bit-clearing tests with authored shaders, including multi-frame
    * returned-atomic checks. Reserved and modifier bits are zero. The two
    * control bits below are required: clearing byte 7 bit 1 times out, and
    * clearing byte 8 bit 7 can suppress a later dispatch. Their individual
    * scheduling semantics are not yet established. */
   uint32_t entry = 2 * entry_offset + 0x2a;
   memset(out, 0, 10);
   write16(out, 0x177);
   out[2] = entry;
   out[3] = entry >> 8;
   out[4] = entry >> 16;
   out[7] = 2;
   out[8] = 0x80;
   return true;
}

static void
encode_frame(uint8_t out[10], const struct agx_launch_parameters *params)
{
   /* A publication unit holds two 32-bit slots. The independent A/B extents
    * support scratch and returned atomics. No further shared setup is needed. */
   memset(out, 0, 10);
   out[0] = 0xf7;
   out[2] = 0x2a;
   write16(out + 3, ((params->publication_count + 1) / 2) << 7);
   write16(out + 5, params->frame_extent_a);
   write16(out + 7, params->frame_extent_b);
}

static bool
launch_build(uint8_t *out, size_t capacity, enum agx_launch_stage stage,
             const struct agx_launch_parameters *params,
             struct agx_launch_template *tmpl)
{
   if (!out || !params || (unsigned)stage > AGX_LAUNCH_COMPUTE)
      return false;
   bool compute = stage == AGX_LAUNCH_COMPUTE;
   size_t size = compute ? AGX_COMPUTE_LAUNCH_SIZE : AGX_GRAPHICS_LAUNCH_SIZE;
   if (capacity < size || params->publication_count > 1022 ||
       (params->threadgroup_memory_bytes &&
        (!compute || !agx_launch_threadgroup_memory_supported(
                        params->threadgroup_memory_bytes))) ||
       (compute && !agx_launch_call_offset(stage, params->resource_count)) ||
       (stage == AGX_LAUNCH_FRAGMENT &&
        (params->tile_bytes > 128 ||
         (params->samples != 1 && params->samples != 2 &&
          params->samples != 4))))
      return false;

   /* Build privately so even an encoder failure cannot publish partial code. */
   uint8_t program[AGX_COMPUTE_LAUNCH_SIZE] = {0};
   const unsigned root = 2, pending = LAUNCH_PENDING_BASE;
   /* Table loads use a full address in a GPR pair. They have no compact
    * USC-relative pointer field or 512-MiB placement restriction. */
   agx3_emit_literal32(program, root, params->resource_table);
   agx3_emit_literal32(program + 8, root + 1, params->resource_table >> 32);
   unsigned at = 16;
   unsigned words = compute ? agx_compute_root_words(params->resource_count)
                            : AGX_LAUNCH_GRAPHICS_ROOT_WORDS;
   unsigned roots = words / 2;
   if (compute) {
      /* The root words are known when the launch is emitted: materialize
       * them instead of loading them from the per-dispatch resource record,
       * which would put a dependent memory round trip in front of every
       * dispatch. Templates leave the slots for set_roots. */
      static const uint32_t zero[2 * (AGX_COMPUTE_VISIBLE_ARGUMENT_BASE +
                                      AGX_COMPUTE_DIRECT_MAX_RESOURCES)];
      const uint32_t *values = params->root_words;
      if (!values && !tmpl)
         return false;
      if (tmpl) {
         values = zero;
         tmpl->roots_at = at;
         tmpl->root_words = words;
      }
      for (unsigned i = 0; i < words; ++i) {
         if (!agx3_emit_literal32(program + at, pending + i, values[i]))
            return false;
         at += 8;
      }
   } else {
      for (int i = (roots - 1) & ~1; i >= 0; i -= 2) {
         if (!agx3_emit_pointer_load(program + at, pending + 2 * i, root, i,
                                     i + 2 <= roots ? 2 : 1))
            return false;
         at += 14;
      }
   }
   encode_allocation(program + at, stage, params);
   at += 8;
   if (tmpl)
      tmpl->target_at = at;
   if (!encode_target(program + at, stage, params->shader_base,
                      params->entry_offset))
      return false;
   at += 10;
   encode_frame(program + at, params);
   at += 10;
   /* Only loaded roots are pending on slot 2. */
   for (unsigned i = 0; i < words; ++i) {
      if (!agx3_emit_argument_word(program + at, i, pending + i,
                                   i == 0 && !compute))
         return false;
      at += 4;
   }
   if (params->preamble_address) {
      /* Keep compiled setup in the persistent shader BO. Recopying a long
       * body into every launch record defeats instruction-cache reuse. */
      if (((params->preamble_address | params->launch_address) & 1) ||
          params->launch_address < params->shader_base ||
          params->preamble_address < params->shader_base ||
          params->launch_address - params->shader_base > UINT32_MAX - at ||
          params->preamble_address - params->shader_base > UINT32_MAX)
         return false;
      int64_t displacement =
         (int64_t)(params->preamble_address - params->shader_base) -
         (int64_t)(params->launch_address - params->shader_base + at);
      if (!agx3_emit_branch(program + at, true, displacement))
         return false;
      if (tmpl)
         tmpl->branch_at = at;
      at += 10;
   }
   /* The preamble's STOP, or this STOP when no setup remains, completes the
    * argument context and starts the selected main shader. */
   write16(program + at, 0x0e);
   at += 4;
   if (at > size)
      return false;
   if (tmpl)
      tmpl->length = at;
   memcpy(out, program, at);
   memset(out + at, 0, capacity - at);
   return true;
}

bool
agx_launch_program_build(uint8_t *out, size_t capacity,
                         enum agx_launch_stage stage,
                         const struct agx_launch_parameters *params)
{
   return launch_build(out, capacity, stage, params, NULL);
}

bool
agx_launch_template_init(struct agx_launch_template *tmpl, size_t capacity,
                         enum agx_launch_stage stage,
                         const struct agx_launch_parameters *params)
{
   memset(tmpl, 0, sizeof(*tmpl));
   if (!params || capacity > sizeof(tmpl->program))
      return false;
   /* Placeholders for the per-launch fields: a valid even entry, and a
    * program address that keeps the preamble branch encodable. */
   struct agx_launch_parameters p = *params;
   p.resource_table = 0;
   p.entry_offset = 0;
   p.launch_address = p.preamble_address ? p.shader_base : 0;
   if (!launch_build(tmpl->program, capacity, stage, &p, tmpl))
      return false;
   tmpl->shader_base = p.shader_base;
   tmpl->preamble_address = p.preamble_address;
   tmpl->capacity = capacity;
   tmpl->stage = stage;
   tmpl->valid = true;
   return true;
}

bool
agx_launch_template_set_roots(uint8_t *out,
                              const struct agx_launch_template *tmpl,
                              const uint32_t *root_words)
{
   if (!tmpl->valid || !tmpl->root_words || !root_words)
      return false;
   for (unsigned i = 0; i < tmpl->root_words; ++i) {
      if (!agx3_emit_literal32(out + tmpl->roots_at + 8 * i,
                               LAUNCH_PENDING_BASE + i, root_words[i]))
         return false;
   }
   return true;
}

bool
agx_launch_template_emit(uint8_t *out, const struct agx_launch_template *tmpl,
                         uint64_t resource_table, uint32_t entry_offset,
                         uint64_t launch_address)
{
   if (!tmpl->valid)
      return false;
   uint8_t head[16], target[10], branch[10];
   const unsigned root = 2;
   agx3_emit_literal32(head, root, resource_table);
   agx3_emit_literal32(head + 8, root + 1, resource_table >> 32);
   if (!encode_target(target, tmpl->stage, tmpl->shader_base, entry_offset))
      return false;
   if (tmpl->branch_at) {
      /* As launch_build: the displacement is relative to the branch. */
      uint64_t base = tmpl->shader_base;
      if ((launch_address & 1) || launch_address < base ||
          launch_address - base > UINT32_MAX - tmpl->branch_at)
         return false;
      int64_t displacement = (int64_t)(tmpl->preamble_address - base) -
                             (int64_t)(launch_address - base + tmpl->branch_at);
      if (!agx3_emit_branch(branch, true, displacement))
         return false;
   }
   memcpy(out, tmpl->program, tmpl->length);
   memcpy(out, head, sizeof(head));
   memcpy(out + tmpl->target_at, target, sizeof(target));
   if (tmpl->branch_at)
      memcpy(out + tmpl->branch_at, branch, sizeof(branch));
   memset(out + tmpl->length, 0, tmpl->capacity - tmpl->length);
   return true;
}

void
agx_stage_launch_cache_init(struct agx_stage_launch_cache *c)
{
   memset(c, 0, sizeof(*c));
   simple_mtx_init(&c->lock, mtx_plain);
}

void
agx_stage_launch_cache_finish(struct agx_stage_launch_cache *c)
{
   simple_mtx_destroy(&c->lock);
}

static void
stage_launch_params(struct agx_launch_parameters *params,
                    const struct agx_device *dev,
                    const struct agx_draw_stage *shader, bool fragment,
                    unsigned tile_bytes, unsigned samples)
{
   *params = (struct agx_launch_parameters){
      .preamble_address =
         shader->preamble_size
            ? agx_draw_stage_body(shader)->addr + shader->preamble_offset
            : 0,
      .shader_base = dev->shader_base,
      .publication_count = shader->publication_count,
      .frame_extent_a = MAX2(shader->scratch_size, shader->atomic_frame_size),
      .frame_extent_b = shader->scratch_size,
      .tile_bytes = fragment ? tile_bytes : 0,
      .samples = fragment ? samples : 0,
   };
}

/* Linear search of the first count published templates. */
static const struct agx_launch_template *
stage_launch_find(struct agx_stage_launch_cache *c, unsigned count,
                  unsigned tile_bytes, unsigned samples)
{
   for (unsigned i = 0; i < count; ++i) {
      if (c->entry[i].tile_bytes == tile_bytes &&
          c->entry[i].samples == samples)
         return &c->entry[i].tmpl;
   }
   return NULL;
}

/* The stage's template for this tile allocation and sample count, building
 * it on first use; NULL if the cache is full or the parameters are rejected.
 * Published entries are immutable: lookups read the count with acquire
 * ordering and take no lock; only building locks. */
static const struct agx_launch_template *
stage_launch_template(const struct agx_device *dev,
                      const struct agx_draw_stage *shader, bool fragment,
                      unsigned tile_bytes, unsigned samples)
{
   struct agx_stage_launch_cache *c = shader->launch_cache;
   if (!fragment)
      tile_bytes = samples = 0;
   const struct agx_launch_template *found = stage_launch_find(
      c, __atomic_load_n(&c->count, __ATOMIC_ACQUIRE), tile_bytes, samples);
   if (found)
      return found;
   simple_mtx_lock(&c->lock);
   unsigned i = c->count;
   found = stage_launch_find(c, i, tile_bytes, samples);
   if (!found && i < AGX_STAGE_LAUNCH_TEMPLATES) {
      struct agx_launch_parameters params;
      stage_launch_params(&params, dev, shader, fragment, tile_bytes, samples);
      if (agx_launch_template_init(
             &c->entry[i].tmpl, AGX_GRAPHICS_LAUNCH_SIZE,
             fragment ? AGX_LAUNCH_FRAGMENT : AGX_LAUNCH_VERTEX, &params)) {
         c->entry[i].tile_bytes = tile_bytes;
         c->entry[i].samples = samples;
         found = &c->entry[i].tmpl;
         __atomic_store_n(&c->count, i + 1, __ATOMIC_RELEASE);
      }
   }
   simple_mtx_unlock(&c->lock);
   return found;
}

bool
agx_prepare_stage_launch(struct agx_device *dev, void *allocator,
                         struct agx_ptr (*alloc)(void *, size_t, size_t),
                         const struct agx_draw_stage *shader, bool fragment,
                         uint64_t table, uint64_t textures, uint64_t samplers,
                         uint64_t push, unsigned tile_bytes, unsigned samples,
                         uint32_t *launch)
{
   if (!shader->bo || !shader->publication_count_valid)
      return false;
   unsigned entry =
      agx_shader_entry(dev, allocator, agx_draw_stage_body(shader), false);
   if (!entry)
      return false;
   if (shader->preamble_offset > shader->binary_size ||
       shader->preamble_size > shader->binary_size - shader->preamble_offset) {
      fprintf(
         stderr,
         "draw: stage %u invalid preamble offset %u size %u in binary %zu\n",
         fragment, shader->preamble_offset, shader->preamble_size,
         (size_t)shader->binary_size);
      return false;
   }
   /* The launch loads GRAPHICS_ROOT_WORDS 32-bit words. Align the following
    * launch program to 64 bytes; the old 256-byte root reservation copied
    * another 192 bytes of zeros per stage without any consumer. */
   enum { roots_size = ALIGN_POT(AGX_LAUNCH_GRAPHICS_ROOT_WORDS * 4, 64) };
   const unsigned launch_size = AGX_GRAPHICS_LAUNCH_SIZE;
   struct agx_launch_parameters params;
   stage_launch_params(&params, dev, shader, fragment, tile_bytes, samples);
   params.entry_offset = entry;

   /* With a template, only the roots address, entry and program address are
    * per launch: emit the program next to fresh roots, without encoding it
    * or hashing it into the upload cache. */
   const struct agx_launch_template *tmpl =
      shader->launch_cache
         ? stage_launch_template(dev, shader, fragment, tile_bytes, samples)
         : NULL;
   if (tmpl) {
      struct agx_ptr state = alloc(allocator, roots_size + launch_size, 64);
      if (!state.cpu)
         return false;
      uint8_t *roots = state.cpu;
      memset(roots, 0, roots_size);
      put_u64(roots, textures);
      put_u64(roots + 8, samplers);
      put_u64(roots + 0x10, table);
      put_u64(roots + 8 * AGX_LAUNCH_PUSH_ROOT, push);
      if (!agx_launch_template_emit(roots + roots_size, tmpl, state.gpu, entry,
                                    state.gpu + roots_size))
         return false;
      *launch = agx_usc_addr(dev, state.gpu + roots_size);
      return true;
   }

   uint8_t canonical[roots_size + AGX_GRAPHICS_LAUNCH_SIZE] = {0};
   uint8_t *roots = canonical;
   put_u64(roots, textures);
   put_u64(roots + 8, samplers);
   put_u64(roots + 0x10, table);
   put_u64(roots + 8 * AGX_LAUNCH_PUSH_ROOT, push);
   struct agx_ptr state = alloc(allocator, sizeof(canonical), 64);
   if (!state.cpu)
      return false;
   roots = state.cpu;
   memcpy(roots, canonical, roots_size);
   params.launch_address = state.gpu + roots_size;
   params.resource_table = state.gpu;
   if (!agx_launch_program_build(
          roots + roots_size, launch_size,
          fragment ? AGX_LAUNCH_FRAGMENT : AGX_LAUNCH_VERTEX, &params)) {
      fprintf(
         stderr,
         "draw: stage %u launch rejected: entry=0x%x "
         "base=0x%llx roots=0x%llx launch=0x%llx preamble=0x%llx "
         "publications=%u scratch=%u tile=%u samples=%u\n",
         fragment, params.entry_offset, (unsigned long long)params.shader_base,
         (unsigned long long)params.resource_table,
         (unsigned long long)params.launch_address,
         (unsigned long long)params.preamble_address, params.publication_count,
         params.frame_extent_a, params.tile_bytes, params.samples);
      return false;
   }
   *launch = agx_usc_addr(dev, state.gpu + roots_size);
   return true;
}
