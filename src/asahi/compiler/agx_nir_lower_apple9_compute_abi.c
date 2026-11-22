/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_apple9_profile.h"
#include "agx_compile.h"
#include "agx_nir.h"

/*
 * Apple9 compute shaders get their arguments as root words: 64-bit pointers
 * that the launch program writes to the start of the uniform file. There are
 * no other preloaded uniforms.
 *
 * With the direct ABI, the first root points to the three workgroup counts
 * and root 1 + i is the address of resource i, in the order of the compute
 * profile. With more than 18 resources, six roots point to texture, sampler
 * and buffer tables, the workgroup counts and threadgroup memory instead, and
 * buffer addresses are read from the buffer table.
 *
 * The profile maps resources to API bindings: UBO and SSBO bindings, and a
 * threadgroup memory resource whose root is a tag selecting threadgroup
 * memory. Kernel inputs, which Apple8 preloads as uniforms, are passed in
 * memory as UBO 0, aligned to 8 bytes.
 */

#define DESCRIPTOR_TABLE_BUFFERS_ROOT 2

struct resource {
   enum agx_apple9_compute_resource_kind kind;
   uint8_t binding;
   bool read, write;
};

struct abi {
   struct agx_apple9_compute_profile *profile;
   struct resource resources[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   unsigned count;
};

/* Kernel inputs are 16-bit uniforms counted from 0, which are UBO 0 bytes */
static bool
lower_kernel_input(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   assert(nir_intrinsic_preamble_class(intr) == nir_preamble_class_general);
   b->cursor = nir_before_instr(&intr->instr);

   unsigned offset = nir_intrinsic_base(intr) * 2;
   unsigned size = intr->def.num_components * intr->def.bit_size / 8;

   /* Drivers align the input buffer to 8 bytes */
   nir_def_replace(&intr->def,
                   nir_load_ubo(b, intr->def.num_components, intr->def.bit_size,
                                nir_imm_int(b, 0), nir_imm_int(b, offset),
                                .align_mul = 8, .align_offset = offset % 8,
                                .range_base = offset, .range = size));
   return true;
}

static bool
is_buffer_access(nir_intrinsic_instr *intr, bool *ubo, bool *write)
{
   *ubo = intr->intrinsic == nir_intrinsic_load_ubo;
   *write = intr->intrinsic == nir_intrinsic_store_ssbo ||
            intr->intrinsic == nir_intrinsic_ssbo_atomic ||
            intr->intrinsic == nir_intrinsic_ssbo_atomic_swap;

   return *ubo || *write || intr->intrinsic == nir_intrinsic_load_ssbo;
}

static unsigned
binding_of(nir_intrinsic_instr *intr)
{
   nir_src *src = intr->intrinsic == nir_intrinsic_store_ssbo ? &intr->src[1]
                                                              : &intr->src[0];

   /* Drivers lower dynamic bindings to global memory */
   assert(nir_src_is_const(*src) && nir_src_as_uint(*src) <= UINT8_MAX);
   return nir_src_as_uint(*src);
}

static struct resource *
find_resource(struct abi *abi, enum agx_apple9_compute_resource_kind kind,
              unsigned binding)
{
   for (unsigned i = 0; i < abi->count; ++i) {
      if (abi->resources[i].kind == kind &&
          abi->resources[i].binding == binding)
         return &abi->resources[i];
   }

   return NULL;
}

static bool
gather_resource(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct abi *abi = data;
   bool ubo, write;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      abi->profile->writes_global = true;
      return false;
   default:
      break;
   }

   if (!is_buffer_access(intr, &ubo, &write))
      return false;

   enum agx_apple9_compute_resource_kind kind =
      ubo ? AGX_APPLE9_COMPUTE_RESOURCE_UBO : AGX_APPLE9_COMPUTE_RESOURCE_SSBO;
   unsigned binding = binding_of(intr);
   struct resource *res = find_resource(abi, kind, binding);

   if (!res) {
      assert(abi->count < AGX_APPLE9_COMPUTE_MAX_RESOURCES);
      res = &abi->resources[abi->count++];
      *res = (struct resource){.kind = kind, .binding = binding};
   }

   res->read |= intr->intrinsic != nir_intrinsic_store_ssbo;
   res->write |= write;
   return false;
}

/* Read-only resources first, then by kind and binding, as the Apple9
 * compiler orders them.
 */
static int
compare_resources(const void *a_, const void *b_)
{
   const struct resource *a = a_, *b = b_;

   if (a->write != b->write)
      return (int)a->write - (int)b->write;
   else if (a->kind != b->kind)
      return (int)a->kind - (int)b->kind;
   else
      return (int)b->binding - (int)a->binding;
}

static nir_def *
load_root(nir_builder *b, unsigned root)
{
   return nir_load_preamble(b, 1, 64, .base = root * 4);
}

static bool
tables(const struct agx_apple9_compute_profile *profile)
{
   return profile->abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES;
}

static nir_def *
resource_address(nir_builder *b, struct abi *abi,
                 enum agx_apple9_compute_resource_kind kind, unsigned binding)
{
   struct resource *res = find_resource(abi, kind, binding);
   unsigned i = res - abi->resources;

   if (tables(abi->profile)) {
      nir_def *table = load_root(b, DESCRIPTOR_TABLE_BUFFERS_ROOT);
      return nir_load_global_constant(
         b, 1, 64, nir_iadd_imm(b, table, i * 8), .align_mul = 8,
         .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE);
   } else {
      return load_root(b, AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + i);
   }
}

static bool
lower_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct abi *abi = data;
   bool ubo, write;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic == nir_intrinsic_load_num_workgroups) {
      unsigned root = tables(abi->profile) ? AGX_APPLE9_COMPUTE_GROUPS_ROOT : 0;

      nir_def *counts = nir_load_global_constant(
         b, 3, 32, load_root(b, root), .align_mul = 4,
         .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE);

      nir_def_replace(
         &intr->def,
         nir_u2uN(b, nir_trim_vector(b, counts, intr->def.num_components),
                  intr->def.bit_size));
      return true;
   }

   if (!is_buffer_access(intr, &ubo, &write))
      return false;

   enum agx_apple9_compute_resource_kind kind =
      ubo ? AGX_APPLE9_COMPUTE_RESOURCE_UBO : AGX_APPLE9_COMPUTE_RESOURCE_SSBO;
   nir_def *base = resource_address(b, abi, kind, binding_of(intr));
   nir_def *offset = nir_get_io_offset_src(intr)->ssa;

   /* Atomics are naturally aligned */
   bool atomic = intr->intrinsic == nir_intrinsic_ssbo_atomic ||
                 intr->intrinsic == nir_intrinsic_ssbo_atomic_swap;
   unsigned align_mul = atomic ? 4 : nir_intrinsic_align_mul(intr);
   unsigned align_offset = atomic ? 0 : nir_intrinsic_align_offset(intr);

   /* Memory instructions scale a 32-bit index by the element size. The byte
    * offset of an aligned access is a multiple of it, so the index is exact.
    */
   unsigned bit_size = intr->intrinsic == nir_intrinsic_store_ssbo
                          ? nir_src_bit_size(intr->src[0])
                          : intr->def.bit_size;
   unsigned shift = util_logbase2(
      MIN3(bit_size / 8, 4, nir_combined_align(align_mul, align_offset)));

   nir_def *addr = nir_ulea_agx(b, base, nir_ushr_imm(b, offset, shift),
                                nir_imm_int(b, shift));
   enum gl_access_qualifier access =
      nir_intrinsic_has_access(intr) ? nir_intrinsic_access(intr) : 0;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
      nir_def_replace(&intr->def,
                      nir_load_global_constant(
                         b, intr->def.num_components, intr->def.bit_size, addr,
                         .align_mul = align_mul, .align_offset = align_offset,
                         .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE));
      break;

   case nir_intrinsic_load_ssbo:
      nir_def_replace(
         &intr->def,
         nir_load_global(b, intr->def.num_components, intr->def.bit_size, addr,
                         .align_mul = align_mul, .align_offset = align_offset,
                         .access = access));
      break;

   case nir_intrinsic_store_ssbo:
      nir_store_global(b, intr->src[0].ssa, addr,
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .align_mul = align_mul, .align_offset = align_offset,
                       .access = access);
      nir_instr_remove(&intr->instr);
      break;

   case nir_intrinsic_ssbo_atomic:
      nir_def_replace(
         &intr->def,
         nir_global_atomic(b, intr->def.bit_size, addr, intr->src[2].ssa,
                           .atomic_op = nir_intrinsic_atomic_op(intr)));
      break;

   case nir_intrinsic_ssbo_atomic_swap:
      nir_def_replace(
         &intr->def,
         nir_global_atomic_swap(b, intr->def.bit_size, addr, intr->src[2].ssa,
                                intr->src[3].ssa,
                                .atomic_op = nir_intrinsic_atomic_op(intr)));
      break;

   default:
      UNREACHABLE("buffer access");
   }

   return true;
}

/*
 * Lowers the compute ABI and fills the profile the driver launches the shader
 * with. Returns the first uniform word after the roots, where preamble
 * results may start.
 */
unsigned
agx_nir_lower_apple9_compute_abi(nir_shader *nir,
                                 struct agx_apple9_compute_profile *profile)
{
   struct abi abi = {.profile = profile};
   *profile = (struct agx_apple9_compute_profile){
      .abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
   };

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_kernel_input,
            nir_metadata_control_flow, NULL);

   nir_shader_intrinsics_pass(nir, gather_resource, nir_metadata_all, &abi);
   qsort(abi.resources, abi.count, sizeof(abi.resources[0]), compare_resources);

   if (nir->info.shared_size) {
      assert(abi.count < AGX_APPLE9_COMPUTE_MAX_RESOURCES);
      assert(nir->info.shared_size <= AGX_APPLE9_MAX_SHARED_BYTES);

      abi.resources[abi.count++] = (struct resource){
         .kind = AGX_APPLE9_COMPUTE_RESOURCE_SHARED,
      };

      profile->required_threadgroup_memory_bytes =
         util_next_power_of_two(MAX2(nir->info.shared_size, 128));
   }

   if (abi.count > AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES)
      profile->abi = AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES;

   profile->resource_binding_count = abi.count;
   for (unsigned i = 0; i < abi.count; ++i) {
      profile->resource_binding[i] = abi.resources[i].binding;
      profile->resource_kind[i] = abi.resources[i].kind;

      if (abi.resources[i].read)
         profile->resource_read_mask |= BITFIELD_BIT(i);

      if (abi.resources[i].write)
         profile->resource_write_mask |= BITFIELD_BIT(i);
   }

   profile->variable_local_size = nir->info.workgroup_size_variable;
   for (unsigned i = 0; i < 3; ++i)
      profile->local_size[i] = nir->info.workgroup_size[i];

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_access,
            nir_metadata_control_flow, &abi);

   return tables(profile) ? AGX_APPLE9_GRAPHICS_ROOT_WORDS
                          : agx_apple9_compute_preamble_base(abi.count);
}

/* The root naming threadgroup memory */
unsigned
agx_apple9_shared_root(const struct agx_apple9_compute_profile *profile)
{
   if (tables(profile))
      return AGX_APPLE9_COMPUTE_SHARED_ARGUMENT;

   for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
      if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED)
         return AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + i;
   }

   UNREACHABLE("no threadgroup memory");
}
