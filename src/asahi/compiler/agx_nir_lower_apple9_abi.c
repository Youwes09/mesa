/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_apple9_profile.h"
#include "agx_compile.h"
#include "agx_nir.h"

/* Arguments are 64-bit roots at the start of the uniform file, one per resource
 * for direct compute, else six table roots. Push words follow the roots.
 */

#define DESCRIPTOR_TABLE_BUFFERS_ROOT 2

struct resource {
   enum agx_apple9_compute_resource_kind kind;
   uint8_t binding;
   bool read, write;
};

struct abi {
   struct agx_apple9_compute_profile *profile;
   struct hash_table *ranges;
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

/* Push words follow the roots in the uniform file */
static bool
lower_push_word(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   assert(nir_intrinsic_preamble_class(intr) == nir_preamble_class_general);
   nir_intrinsic_set_base(
      intr, nir_intrinsic_base(intr) + 2 * AGX_APPLE9_GRAPHICS_ROOT_WORDS);
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

/* Read-only resources first, then by kind and binding */
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

   /* The index is scaled by the element size, which divides the offset */
   unsigned bit_size = intr->intrinsic == nir_intrinsic_store_ssbo
                          ? nir_src_bit_size(intr->src[0])
                          : intr->def.bit_size;
   unsigned shift = util_logbase2(
      MIN3(bit_size / 8, 4, nir_combined_align(align_mul, align_offset)));

   /* Fold a constant into the byte offset if it neither wraps nor misaligns */
   uint32_t k = 0;
   if (!atomic) {
      nir_scalar s = nir_scalar_resolved(offset, 0);

      if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
         for (unsigned i = 0; i < 2; ++i) {
            nir_scalar c = nir_scalar_chase_alu_src(s, i);
            nir_scalar x = nir_scalar_chase_alu_src(s, 1 - i);

            if (!nir_scalar_is_const(c))
               continue;

            uint32_t v = nir_scalar_as_uint(c);
            if (v <= INT16_MAX && (v & BITFIELD_MASK(shift)) == 0 &&
                !nir_addition_might_overflow(b->shader, abi->ranges, x, v)) {
               k = v;
               offset = nir_mov_scalar(b, x);
            }

            break;
         }
      }
   }

   nir_def *addr = nir_ulea_agx(b, base, nir_ushr_imm(b, offset, shift),
                                nir_imm_int(b, shift));
   if (k)
      addr = nir_iadd_imm(b, addr, k);
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

/* Returns the uniform word after the roots, where push words may start */
unsigned
agx_nir_lower_apple9_abi(nir_shader *nir, bool graphics,
                         struct agx_apple9_compute_profile *profile)
{
   struct abi abi = {.profile = profile};
   /* Precompiled profiles are copied byte for byte, padding included */
   memset(profile, 0, sizeof(*profile));
   profile->abi = graphics ? AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES
                           : AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            graphics ? lower_push_word : lower_kernel_input,
            nir_metadata_control_flow, NULL);

   nir_shader_intrinsics_pass(nir, gather_resource, nir_metadata_all, &abi);

   /* Textures, samplers and PBE images are in tables pointed to by roots */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex ||
                (instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(instr)->intrinsic ==
                    nir_intrinsic_image_store_apple9_agx))
               profile->abi = AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES;
         }
      }
   }
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

   if (mesa_shader_stage_is_compute(nir->info.stage)) {
      profile->variable_local_size = nir->info.workgroup_size_variable;
      for (unsigned i = 0; i < 3; ++i)
         profile->local_size[i] = nir->info.workgroup_size[i];
   }

   abi.ranges = _mesa_pointer_hash_table_create(NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_access,
            nir_metadata_control_flow, &abi);
   _mesa_hash_table_destroy(abi.ranges, NULL);

   return tables(profile) ? AGX_APPLE9_GRAPHICS_ROOT_WORDS
                          : agx_apple9_compute_preamble_base(abi.count);
}

struct preload {
   /* 32-bit push words read by the shader */
   BITSET_DECLARE(words, AGX_APPLE9_MAX_PUSH_WORDS);
   unsigned first, count;
   nir_def *values[AGX_APPLE9_MAX_PUSH_WORDS];
};

static bool
is_push_word(nir_intrinsic_instr *intr, struct preload *p)
{
   if (intr->intrinsic != nir_intrinsic_load_preamble ||
       nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
      return false;

   unsigned base = nir_intrinsic_base(intr);
   return base >= p->first && base < p->first + p->count;
}

static bool
gather_push_words(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preload *p = data;
   if (!is_push_word(intr, p))
      return false;

   unsigned start = nir_intrinsic_base(intr) - p->first;
   unsigned halves = intr->def.num_components * intr->def.bit_size / 16;
   assert(start + halves <= p->count && "push words are within the range");

   for (unsigned h = start; h < start + halves; ++h)
      BITSET_SET(p->words, h / 2);

   return false;
}

/* In the preamble, push words are the values it loaded */
static bool
rewrite_push_words(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preload *p = data;
   if (!is_push_word(intr, p))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned start = nir_intrinsic_base(intr) - p->first;
   unsigned halves = intr->def.num_components * intr->def.bit_size / 16;
   nir_def *words[AGX_APPLE9_MAX_PUSH_WORDS / 4];
   unsigned first = start / 2, n = DIV_ROUND_UP(start + halves, 2) - first;

   assert(n <= ARRAY_SIZE(words));
   for (unsigned i = 0; i < n; ++i)
      words[i] = p->values[first + i];

   nir_def *v = nir_extract_bits(b, words, n, (start & 1) * 16,
                                 intr->def.num_components, intr->def.bit_size);
   nir_def_replace(&intr->def, v);
   return true;
}

/* Load the push words read at the start of the formed preamble */
bool
agx_nir_preload_apple9_push(nir_shader *nir, unsigned first, unsigned count)
{
   assert(first % 2 == 0 && count % 2 == 0);
   assert(count / 2 <= AGX_APPLE9_MAX_PUSH_WORDS);

   struct preload *p = rzalloc(NULL, struct preload);
   p->first = first;
   p->count = count;

   nir_shader_intrinsics_pass(nir, gather_push_words, nir_metadata_all, p);

   if (BITSET_IS_EMPTY(p->words)) {
      ralloc_free(p);
      return false;
   }

   nir_function_impl *main = nir_shader_get_entrypoint(nir);
   if (!main->preamble) {
      nir_function *func = nir_function_create(nir, "@preamble");
      func->is_preamble = true;
      nir_function_impl_create(func);
      main->preamble = func;
   }

   nir_function_impl *preamble = main->preamble->impl;
   nir_builder b = nir_builder_at(nir_before_impl(preamble));
   nir_def *push =
      nir_load_preamble(&b, 1, 64, .base = 4 * AGX_APPLE9_PUSH_ROOT);

   /* Load runs of words with vector loads */
   unsigned w;
   BITSET_FOREACH_SET(w, p->words, AGX_APPLE9_MAX_PUSH_WORDS) {
      if (p->values[w])
         continue;

      unsigned n = 1;
      while (n < 4 && BITSET_TEST(p->words, w + n))
         ++n;

      nir_def *v = nir_load_global_constant(
         &b, n, 32, nir_iadd_imm(&b, push, w * 4), .align_mul = 4,
         .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE);

      for (unsigned i = 0; i < n; ++i) {
         p->values[w + i] = nir_channel(&b, v, i);
         nir_store_preamble(&b, p->values[w + i], .base = first + 2 * (w + i));
      }
   }

   nir_function_intrinsics_pass(preamble, rewrite_push_words,
                                nir_metadata_control_flow, p);
   ralloc_free(p);
   return true;
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
