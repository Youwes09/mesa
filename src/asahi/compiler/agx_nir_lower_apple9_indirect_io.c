/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

/* Apple9 instructions name varyings by immediate. Dynamic input indices become
 * selects, dynamic outputs a temporary array stored out at the end.
 */

static bool
is_compact(nir_io_semantics sem)
{
   return sem.location == VARYING_SLOT_CLIP_DIST0 ||
          sem.location == VARYING_SLOT_CLIP_DIST1;
}

static bool
lower_input(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_interpolated_input &&
       intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (nir_src_is_const(*offset))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *index = offset->ssa;
   nir_def *result = NULL;

   /* Compact arrays index scalars, other arrays vec4 slots */
   bool compact = is_compact(sem);
   unsigned count = compact ? sem.num_slots * 4 : sem.num_slots;
   unsigned first = compact ? nir_intrinsic_component(intr) : 0;

   for (unsigned i = first; i < count; ++i) {
      unsigned slot = compact ? i / 4 : i;
      nir_def *slot_offset = nir_imm_int(b, slot);

      nir_intrinsic_instr *load =
         nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
      nir_builder_instr_insert(b, &load->instr);
      nir_src_rewrite(nir_get_io_offset_src(load), slot_offset);
      if (compact)
         nir_intrinsic_set_component(load, i % 4);

      nir_def *value = &load->def;
      result =
         result ? nir_bcsel(b, nir_ieq_imm(b, index, i - first), value, result)
                : value;
   }

   nir_def_replace(&intr->def, result);
   return true;
}

struct output_array {
   nir_variable *var;
   unsigned location, num_slots, bit_size;
   uint8_t written[64];
};

static struct output_array *
find_array(struct output_array *arrays, unsigned nr, nir_io_semantics sem)
{
   for (unsigned i = 0; i < nr; ++i) {
      if (sem.location >= arrays[i].location &&
          sem.location < arrays[i].location + arrays[i].num_slots)
         return &arrays[i];
   }

   return NULL;
}

static bool
lower_outputs(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct output_array arrays[VARYING_SLOT_MAX];
   unsigned nr = 0;

   /* Find the arrays indexed by a value */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_src_is_const(*nir_get_io_offset_src(intr)))
            continue;

         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         assert(!is_compact(sem) && "clip distances are stored whole");

         if (find_array(arrays, nr, sem))
            continue;

         assert(nr < ARRAY_SIZE(arrays));
         unsigned bit_size = nir_src_bit_size(intr->src[0]);
         const struct glsl_type *elem = glsl_vector_type(
            bit_size == 16 ? GLSL_TYPE_UINT16 : GLSL_TYPE_UINT, 4);

         arrays[nr++] = (struct output_array){
            .var = nir_local_variable_create(
               impl, glsl_array_type(elem, sem.num_slots, 0),
               "indexed outputs"),
            .location = sem.location,
            .num_slots = sem.num_slots,
            .bit_size = bit_size,
         };
      }
   }

   if (!nr)
      return false;

   /* Store every element of those arrays to the temporary instead */
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;

         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         struct output_array *array = find_array(arrays, nr, sem);
         if (!array)
            continue;

         b.cursor = nir_before_instr(instr);
         nir_src *offset = nir_get_io_offset_src(intr);
         nir_def *index =
            nir_iadd_imm(&b, offset->ssa, sem.location - array->location);
         unsigned component = nir_intrinsic_component(intr);
         unsigned mask = nir_intrinsic_write_mask(intr) << component;

         nir_def *src = intr->src[0].ssa;
         assert(src->bit_size == array->bit_size);
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c) {
            channels[c] =
               (c >= component && c < component + src->num_components)
                  ? nir_channel(&b, src, c - component)
                  : nir_undef(&b, 1, src->bit_size);
         }
         nir_def *value = nir_vec(&b, channels, 4);

         nir_deref_instr *deref = nir_build_deref_array(
            &b, nir_build_deref_var(&b, array->var), index);
         nir_store_deref(&b, deref, value, mask);

         /* Any element may be written at a dynamic index */
         for (unsigned s = 0; s < array->num_slots; ++s) {
            if (nir_src_is_const(*offset) &&
                s != nir_src_as_uint(*offset) + sem.location - array->location)
               continue;

            array->written[s] |= mask;
         }

         nir_instr_remove(instr);
      }
   }

   /* Then write the arrays out at the end */
   b.cursor = nir_after_impl(impl);

   for (unsigned i = 0; i < nr; ++i) {
      struct output_array *array = &arrays[i];

      for (unsigned s = 0; s < array->num_slots; ++s) {
         if (!array->written[s])
            continue;

         nir_def *value =
            nir_load_deref(&b, nir_build_deref_array_imm(
                                  &b, nir_build_deref_var(&b, array->var), s));

         nir_store_output(
            &b, value, nir_imm_int(&b, 0), .write_mask = array->written[s],
            .src_type = nir_type_uint | array->bit_size,
            .io_semantics = {.location = array->location + s, .num_slots = 1});
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   return true;
}

bool
agx_nir_lower_apple9_indirect_io(nir_shader *nir)
{
   bool progress = false;

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_input,
               nir_metadata_control_flow, NULL);
   } else if (nir->info.stage == MESA_SHADER_VERTEX ||
              nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(progress, nir, lower_outputs);
   }

   return progress;
}
