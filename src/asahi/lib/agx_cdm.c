/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_cdm.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/lib/pool.h"
#include "util/u_math.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "agx_shader_body.h"

struct cdm_abi_desc {
   uint8_t resource_count;
   uint16_t resource_record_size;
   bool supports_indirect_dispatch;
};

static const struct cdm_abi_desc *
cdm_abi(const struct agx_compute_profile *profile)
{
   static const struct cdm_abi_desc ssbo8_superset = {
      .resource_count = AGX_COMPUTE_MAX_RESOURCES,
      .resource_record_size = AGX_CDM_SUPERSET_RESOURCE_STRIDE,
      .supports_indirect_dispatch = true,
   };

   if (!profile)
      return NULL;

   switch (profile->abi) {
   case AGX_COMPUTE_ABI_DIRECT_BUFFERS:
   case AGX_COMPUTE_ABI_DESCRIPTOR_TABLES:
      return &ssbo8_superset;
   default:
      return NULL;
   }
}

static bool
cdm_profile_valid(const struct agx_compute_profile *profile,
                  const struct cdm_abi_desc *abi)
{
   /* Programs with the graphics argument layout may have no buffer table */
   bool tables = profile && profile->abi == AGX_COMPUTE_ABI_DESCRIPTOR_TABLES;
   if (!profile || !abi || (profile->resource_binding_count == 0 && !tables) ||
       profile->resource_binding_count > abi->resource_count ||
       (profile->abi == AGX_COMPUTE_ABI_DIRECT_BUFFERS &&
        profile->resource_binding_count > AGX_COMPUTE_DIRECT_MAX_RESOURCES) ||
       !agx_launch_threadgroup_memory_supported(
          profile->required_threadgroup_memory_bytes) ||
       (profile->atomic_frame_size != 0 && profile->atomic_frame_size != 4) ||
       profile->scratch_size > AGX_LAUNCH_MAX_SCRATCH_BYTES ||
       profile->publication_count > 1022 || (profile->scratch_size & 15))
      return false;

   const unsigned active = profile->resource_binding_count;
   const uint32_t resource_mask = BITFIELD_MASK(active);
   const uint32_t read_mask = profile->resource_read_mask;
   const uint32_t write_mask = profile->resource_write_mask;
   if ((!tables && !write_mask && !profile->writes_global) ||
       ((read_mask | write_mask) & ~resource_mask))
      return false;

   uint64_t local_threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (profile->variable_local_size) {
         if (profile->local_size[d] != 0)
            return false;
      } else {
         if (!profile->local_size[d] ||
             profile->local_size[d] > 1024 / local_threads)
            return false;
         local_threads *= profile->local_size[d];
      }
   }

   for (unsigned i = 0; i < active; ++i) {
      if (profile->resource_kind[i] > AGX_COMPUTE_RESOURCE_SHARED ||
          (profile->resource_kind[i] == AGX_COMPUTE_RESOURCE_SHARED &&
           (!profile->required_threadgroup_memory_bytes ||
            ((read_mask | write_mask) & BITFIELD_BIT(i)))) ||
          ((write_mask & BITFIELD_BIT(i)) &&
           profile->resource_kind[i] != AGX_COMPUTE_RESOURCE_SSBO))
         return false;
   }

   return true;
}

/* Launch parameters of a compute profile; the caller sets the per-launch
 * entry, resource table and program address. */
static bool
cdm_launch_params(struct agx_launch_parameters *out, uint64_t usc_exec_base,
                  const struct agx_compute_profile *profile,
                  uint64_t preamble_address)
{
   const struct cdm_abi_desc *abi = cdm_abi(profile);
   if (!abi || !cdm_profile_valid(profile, abi))
      return false;

   struct agx_launch_parameters params = {
      .publication_count = profile->publication_count,
      .preamble_address = preamble_address,
   };
   params.shader_base = usc_exec_base;
   params.resource_count = profile->abi == AGX_COMPUTE_ABI_DESCRIPTOR_TABLES
                              ? AGX_LAUNCH_GRAPHICS_ROOT_WORDS / 2 - 1
                              : profile->resource_binding_count;
   params.threadgroup_memory_bytes = profile->required_threadgroup_memory_bytes;
   if (profile->scratch_size) {
      params.frame_extent_a = profile->scratch_size;
      params.frame_extent_b = profile->scratch_size;
   }
   params.frame_extent_a =
      MAX2(params.frame_extent_a, profile->atomic_frame_size);
   *out = params;
   return true;
}

static bool
cdm_build_launch(uint8_t *out, uint64_t usc_exec_base, uint64_t launch_address,
                 uint64_t resource_address, const uint32_t *root_words,
                 uint32_t main_offset,
                 const struct agx_compute_profile *profile,
                 uint64_t preamble_address)
{
   struct agx_launch_parameters params;
   if (main_offset < AGX_ENTRY_HEADER_SIZE + AGX_ENTRY_CODE_OFFSET ||
       !cdm_launch_params(&params, usc_exec_base, profile, preamble_address))
      return false;
   params.entry_offset = main_offset;
   params.resource_table = resource_address;
   params.launch_address = launch_address;
   params.root_words = root_words;
   return agx_launch_program_build(out, agx_cdm_launch_size(profile),
                                   AGX_LAUNCH_COMPUTE, &params);
}

bool
agx_cdm_launch_template(struct agx_launch_template *tmpl,
                        const struct agx_compute_profile *profile,
                        uint64_t usc_exec_base, uint64_t preamble_address)
{
   struct agx_launch_parameters params;
   memset(tmpl, 0, sizeof(*tmpl));
   return cdm_launch_params(&params, usc_exec_base, profile,
                            preamble_address) &&
          agx_launch_template_init(tmpl, agx_cdm_launch_size(profile),
                                   AGX_LAUNCH_COMPUTE, &params);
}

size_t
agx_cdm_launch_size(const struct agx_compute_profile *profile)
{
   const struct cdm_abi_desc *abi = cdm_abi(profile);
   return abi ? AGX_COMPUTE_LAUNCH_SIZE : 0;
}

static size_t
cdm_resource_record_size(const struct cdm_abi_desc *abi)
{
   return abi && abi->resource_record_size ? abi->resource_record_size
                                           : AGX_CDM_RESOURCE_STRIDE;
}

enum agx_compute_resource_kind
agx_cdm_resource_kind(const struct agx_compute_profile *profile,
                      unsigned argument)
{
   const struct cdm_abi_desc *abi = cdm_abi(profile);
   if (!abi || argument >= profile->resource_binding_count)
      return AGX_COMPUTE_RESOURCE_SSBO;

   return profile->resource_kind[argument];
}

static bool
cdm_build_geometry_fields(void *record_, size_t record_size,
                          uint64_t record_address,
                          const struct agx_cdm_geometry *geometry)
{
   const unsigned offset = AGX_CDM_GEOMETRY_GROUPS_OFFSET;
   if (!record_ || !geometry || record_size < offset + 12 ||
       record_address > UINT64_MAX - offset)
      return false;

   uint64_t local_threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!geometry->local[d] || geometry->local[d] > 1024 / local_threads)
         return false;
      local_threads *= geometry->local[d];
   }

   uint32_t groups[3] = {0};
   uint64_t pointer;
   switch (geometry->mode) {
   case AGX_CDM_GEOMETRY_DIRECT:
      for (unsigned d = 0; d < 3; ++d) {
         if (!geometry->threads[d])
            return false;
         /* CDM permits a partial final workgroup. Avoid overflowing the
          * numerator when rounding up on the CPU. */
         groups[d] = geometry->threads[d] / geometry->local[d] +
                     (geometry->threads[d] % geometry->local[d] != 0);
      }
      pointer = record_address + offset;
      break;
   case AGX_CDM_GEOMETRY_INDIRECT:
      pointer = geometry->group_counts;
      if (!pointer || (pointer & 3))
         return false;
      break;
   default:
      return false;
   }

   uint8_t *record = record_;
   put_u64(record, pointer);
   for (unsigned d = 0; d < 3; ++d)
      put_u32(record + offset + d * sizeof(uint32_t), groups[d]);
   return true;
}

bool
agx_cdm_prepare_dispatch_alloc_tmpl(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   struct agx_shader_body *body, const struct agx_compute_profile *profile,
   const uint64_t *resources, unsigned resource_count, uint64_t textures,
   uint64_t samplers, uint64_t push, const struct agx_cdm_geometry *geometry,
   uint64_t preamble_address, const struct agx_launch_template *tmpl,
   uint64_t *launch_address)
{
   const struct cdm_abi_desc *abi = cdm_abi(profile);
   if (!abi || (resource_count && !resources) || !geometry || !launch_address ||
       !cdm_profile_valid(profile, abi) ||
       resource_count != profile->resource_binding_count ||
       !!preamble_address != !!profile->preamble_size)
      return false;

   uint32_t entry = agx_shader_entry(dev, allocator, body, true);
   if (!entry)
      return false;

   size_t record_size = cdm_resource_record_size(abi);
   /* The launch materializes its root words from the record's head. */
   struct agx_launch_parameters lp;
   if (!cdm_launch_params(&lp, dev->shader_base, profile, preamble_address) ||
       4 * agx_compute_root_words(lp.resource_count) > record_size)
      return false;
   struct agx_ptr record = alloc(allocator, record_size, 64);
   if (!record.cpu)
      return false;
   memset(record.cpu, 0, record_size);
   if (!cdm_build_geometry_fields(record.cpu, record_size, record.gpu,
                                      geometry))
      return false;
   if (profile->abi == AGX_COMPUTE_ABI_DESCRIPTOR_TABLES) {
      struct agx_ptr table = {0};
      if (resource_count) {
         table = alloc(allocator, resource_count * 8, 64);
         if (!table.cpu)
            return false;
         memcpy(table.cpu, resources, resource_count * 8);
      }
      static_assert(AGX_LAUNCH_PUSH_ROOT == 5, "last root");
      uint64_t roots[AGX_LAUNCH_GRAPHICS_ROOT_WORDS / 2] = {
         textures,
         samplers,
         table.gpu,
         get_u64(record.cpu),
         AGX_COMPUTE_SHARED_ROOT,
         push,
      };
      memcpy(record.cpu, roots, sizeof(roots));
   } else {
      for (unsigned i = 0; i < resource_count; i++)
         put_u64(
            (uint8_t *)record.cpu + (AGX_COMPUTE_VISIBLE_ARGUMENT_BASE + i) * 8,
            resources[i]);
   }

   struct agx_ptr launch = alloc(allocator, agx_cdm_launch_size(profile), 64);
   if (!launch.cpu)
      return false;
   if (tmpl && tmpl->valid) {
      /* The template was built from this profile and preamble; only the
       * record, entry and program address differ per dispatch. */
      assert(tmpl->preamble_address == preamble_address &&
             tmpl->shader_base == dev->shader_base);
      if (entry < AGX_ENTRY_HEADER_SIZE + AGX_ENTRY_CODE_OFFSET ||
          !agx_launch_template_emit(launch.cpu, tmpl, record.gpu, entry,
                                    launch.gpu) ||
          !agx_launch_template_set_roots(launch.cpu, tmpl, record.cpu))
         return false;
   } else if (!cdm_build_launch(launch.cpu, dev->shader_base, launch.gpu,
                                record.gpu, record.cpu, entry, profile,
                                preamble_address)) {
      return false;
   }
   *launch_address = launch.gpu;
   return true;
}

static struct agx_ptr
pool_alloc(void *pool, size_t size, size_t alignment)
{
   return agx_pool_alloc_aligned(pool, size, alignment);
}

bool
agx_cdm_prepare_dispatch(struct agx_device *dev, struct agx_pool *usc_pool,
                         struct agx_bo *body,
                         const struct agx_compute_profile *profile,
                         const uint64_t *resources, unsigned resource_count,
                         uint64_t textures, uint64_t samplers, uint64_t push,
                         const struct agx_cdm_geometry *geometry,
                         uint64_t preamble_address, uint64_t *launch_address)
{
   return agx_cdm_prepare_dispatch_alloc_tmpl(
      dev, usc_pool, pool_alloc, &body->shader_body, profile, resources,
      resource_count, textures, samplers, push, geometry, preamble_address,
      NULL, launch_address);
}

void
agx_cdm_set_compute_barrier(void *record, unsigned record_size)
{
   agx_pack((uint8_t *)record + record_size - 4, CDM_BARRIER, cfg) {
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_8 = true;
   }
}

static void
cdm_launch(void *out, uint64_t launch, enum agx_cdm_mode mode)
{
   agx_pack(out, CDM_LAUNCH_G16, cfg) {
      cfg.unk_19 = true;
      cfg.unk_23 = true;
      cfg.mode = mode;
      cfg.unk_38 = true;
      cfg.unk_56 = true;
      cfg.pipeline_lo = (uint32_t)(launch >> 6);
      cfg.pipeline_hi = launch >> 40;
      cfg.unk_126 = true;
   }
}

/* The full post-dispatch barrier of the Apple8 path. GPU cache ordering is
 * independent of asynchronous CPU submission. */
static void
cdm_barrier(void *out)
{
   agx_pack(out, CDM_BARRIER, cfg) {
      cfg.unk_0 = true;
      cfg.unk_1 = true;
      cfg.unk_2 = true;
      cfg.usc_cache_inval = true;
      cfg.unk_4 = true;
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_7 = true;
      cfg.unk_8 = true;
      cfg.unk_9 = true;
      cfg.unk_10 = true;
      cfg.unk_11 = true;
      cfg.unk_12 = true;
      cfg.unk_13 = true;
      cfg.unk_14 = true;
      cfg.unk_15 = true;
      cfg.unk_16 = true;
      cfg.unk_17 = true;
      cfg.unk_18 = true;
      cfg.unk_19 = true;
   }
}

const char *
agx_cdm_dispatch_reject_reason(uint64_t launch, bool indirect,
                               uint64_t indirect_address,
                               const uint32_t global[3],
                               const uint32_t local[3],
                               const struct agx_compute_profile *profile)
{
   const struct cdm_abi_desc *abi = cdm_abi(profile);
   if (!abi)
      return "no compute ABI for the profile";
   if (launch & 0x3f)
      return "launch address not 64-byte aligned";
   if (!cdm_profile_valid(profile, abi))
      return "invalid compute profile";
   if (indirect) {
      if (!abi->supports_indirect_dispatch)
         return "ABI has no indirect dispatch";
      if (!indirect_address)
         return "null indirect address";
      if (indirect_address & 3)
         return "indirect address not 4-byte aligned";
   } else if (!global) {
      return "no grid";
   }
   if (!local)
      return "no local size";
   uint64_t threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!indirect && !global[d])
         return "zero grid axis";
      if (!local[d])
         return "zero local axis";
      if (!profile->variable_local_size && local[d] != profile->local_size[d])
         return "local size differs from the compiled one";
      if (local[d] > 1024 / threads)
         return "more than 1024 threads per workgroup";
      threads *= local[d];
   }
   return NULL;
}

bool
agx_cdm_emit_direct_dispatch(void *out, uint64_t launch,
                             const uint32_t global[3], const uint32_t local[3],
                             const struct agx_compute_profile *profile,
                             bool barrier)
{
   if (!out ||
       agx_cdm_dispatch_reject_reason(launch, false, 0, global, local, profile))
      return false;

   uint8_t *record = out;
   cdm_launch(record, launch, AGX_CDM_MODE_DIRECT);
   record += AGX_CDM_LAUNCH_G16_LENGTH;

   agx_pack(record, CDM_GLOBAL_SIZE, cfg) {
      cfg.x = global[0];
      cfg.y = global[1];
      cfg.z = global[2];
   }
   record += AGX_CDM_GLOBAL_SIZE_LENGTH;

   agx_pack(record, CDM_LOCAL_SIZE, cfg) {
      cfg.x = local[0];
      cfg.y = local[1];
      cfg.z = local[2];
   }
   record += AGX_CDM_LOCAL_SIZE_LENGTH;

   if (barrier)
      cdm_barrier(record);
   return true;
}

bool
agx_cdm_emit_indirect_dispatch(void *out, uint64_t launch, uint64_t indirect,
                               const uint32_t local[3], bool indirect_local,
                               const struct agx_compute_profile *profile,
                               bool barrier)
{
   if (!out || agx_cdm_dispatch_reject_reason(launch, true, indirect, NULL,
                                              local, profile))
      return false;

   uint8_t *record = out;
   cdm_launch(record, launch,
              indirect_local ? AGX_CDM_MODE_INDIRECT_LOCAL
                             : AGX_CDM_MODE_INDIRECT_GLOBAL);
   record += AGX_CDM_LAUNCH_G16_LENGTH;

   agx_pack(record, INDIRECT_G16, cfg) {
      cfg.address_hi = indirect >> 32;
      cfg.address_lo = indirect;
   }
   record += AGX_INDIRECT_G16_LENGTH;

   /* Indirect local reads the local size with the thread counts, indirect
    * global reads workgroup counts and takes the local size from here. */
   if (!indirect_local) {
      agx_pack(record, CDM_LOCAL_SIZE, cfg) {
         cfg.x = local[0];
         cfg.y = local[1];
         cfg.z = local[2];
      }
      record += AGX_CDM_LOCAL_SIZE_LENGTH;
   }

   if (barrier)
      cdm_barrier(record);
   return true;
}
