/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_linker.h"
#include "compiler/nir/nir_builder.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "util/u_dynarray.h"
#include "hk_cmd_buffer.h"
#include "hk_shader.h"

/* Line stipple: segment vertices carry a flat word encoding the counter as
 * dir * p + c, and the fragment shader demotes fragments by its pattern bit.
 */

/* The snapped window position of a clip-space position in a viewport */
nir_def *
hk_nir_line_stipple_window(nir_builder *b, nir_def *pos, nir_def *viewport)
{
   /* Vertices behind the eye clip the segment; keep the divide finite */
   nir_def *w = nir_fmax(b, nir_channel(b, pos, 3), nir_imm_float(b, 0x1p-16));
   nir_def *ndc = nir_fdiv(b, nir_trim_vector(b, pos, 2), w);

   nir_def *index = nir_umin_imm(b, viewport, HK_MAX_VIEWPORTS - 1);
   nir_def *table = hk_nir_load_root(
      b, 1, 64, hk_root_descriptor_offset(draw.stipple_viewports));
   nir_def *transform = nir_load_global_constant(
      b, 4, 32, nir_iadd(b, table, nir_u2u64(b, nir_imul_imm(b, index, 16))),
      .align_mul = 16);

   return agx_nir_line_stipple_snap(
      b, nir_ffma(b, ndc, nir_channels(b, transform, 0x3),
                  nir_channels(b, transform, 0xc)));
}

/* Major-axis length of a segment, the counter's advance along it */
nir_def *
hk_nir_line_stipple_length(nir_builder *b, nir_def *w0, nir_def *w1)
{
   nir_def *d = nir_fabs(b, nir_fsub(b, w1, w0));
   return nir_fmax(b, nir_channel(b, d, 0), nir_channel(b, d, 1));
}

/* The flat word of the segment from w0 to w1, with counter base at w0 */
nir_def *
hk_nir_line_stipple_word(nir_builder *b, nir_def *w0, nir_def *w1,
                         nir_def *base)
{
   nir_def *d = nir_fabs(b, nir_fsub(b, w1, w0));
   nir_def *x_major = nir_fge(b, nir_channel(b, d, 0), nir_channel(b, d, 1));
   nir_def *s0 =
      nir_bcsel(b, x_major, nir_channel(b, w0, 0), nir_channel(b, w0, 1));
   nir_def *s1 =
      nir_bcsel(b, x_major, nir_channel(b, w1, 0), nir_channel(b, w1, 1));
   nir_def *negative = nir_flt(b, s1, s0);

   /* Only the counter modulo the pattern's period matters */
   nir_def *stipple =
      hk_nir_load_root(b, 1, 32, hk_root_descriptor_offset(draw.line_stipple));
   nir_def *period =
      nir_u2f32(b, nir_imul_imm(b, nir_ushr_imm(b, stipple, 16), 16));
   base = nir_fsub(
      b, base, nir_fmul(b, nir_ffloor(b, nir_fdiv(b, base, period)), period));

   nir_def *c =
      nir_bcsel(b, negative, nir_fadd(b, base, s0), nir_fsub(b, base, s0));
   nir_def *fixed = nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b, c, 256.0)));

   return nir_ior(
      b, nir_ishl_imm(b, fixed, 2),
      nir_ior(b, nir_bcsel(b, x_major, nir_imm_int(b, 2), nir_imm_int(b, 0)),
              nir_b2i32(b, negative)));
}

void
hk_nir_store_line_stipple(nir_builder *b, nir_def *word)
{
   nir_store_output(
      b, word, nir_imm_int(b, 0), .write_mask = 0x1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = AGX_LINE_STIPPLE_SLOT, .num_slots = 1});
   b->shader->info.outputs_written |= BITFIELD64_BIT(AGX_LINE_STIPPLE_SLOT);
}

/* The viewport of a vertex of the passthrough geometry shader's input */
static nir_def *
passthrough_viewport(nir_builder *b, nir_def *vertex, bool viewport)
{
   if (!viewport)
      return nir_imm_int(b, 0);

   return nir_load_per_vertex_input(
      b, 1, 32, vertex, nir_imm_int(b, 0), .dest_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VIEWPORT, .num_slots = 1});
}

/* Counter at each strip segment's start, in 1/256 pixel: entry i is the pair
 * (sum of lengths before segment i, counter), exact below 2^24 pixel strips.
 */
#define HK_STIPPLE_PREFIX_THREADS 1024

/* The words of a value of each subgroup, for a workgroup scan */
#define HK_STIPPLE_PREFIX_SHARED_B (4 * 32 + 8)

/* Inclusive workgroup scan of v with op (iadd or umax), and the total */
static nir_def *
stipple_wg_scan(nir_builder *b, nir_def *v, nir_op op, nir_def **total)
{
   nir_def *incl = nir_inclusive_scan(b, v, .reduction_op = op);
   nir_def *sg = nir_load_subgroup_id(b);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *last = nir_ieq(b, lane, nir_iadd_imm(b, nir_load_subgroup_size(b), -1));

   nir_push_if(b, last);
   nir_store_shared(b, incl, nir_imul_imm(b, sg, 4), .base = 0, .align_mul = 4);
   nir_pop_if(b, NULL);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   nir_def *groups = nir_load_num_subgroups(b);
   nir_def *before = nir_imm_int(b, 0), *all = nir_imm_int(b, 0);
   for (unsigned k = 0; k < 32; ++k) {
      nir_def *word = nir_load_shared(b, 1, 32, nir_imm_int(b, 4 * k),
                                      .base = 0, .align_mul = 4);
      word = nir_bcsel(b, nir_ult_imm(b, groups, k + 1), nir_imm_int(b, 0),
                       word);
      all = nir_build_alu2(b, op, all, word);
      before = nir_bcsel(b, nir_ult(b, nir_imm_int(b, k), sg),
                         nir_build_alu2(b, op, before, word), before);
   }

   /* The shared words are reused by the next scan */
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   *total = all;
   return nir_build_alu2(b, op, before, incl);
}

/* The clip position of a vertex shader output vertex */
static nir_def *
stipple_vertex_output(nir_builder *b, nir_def *vp, nir_def *vertex,
                      unsigned location, unsigned comps)
{
   nir_def *addr = poly_vertex_output_address(
      b, vp, poly_vertex_outputs(b, vp), vertex, nir_imm_int(b, location));
   return nir_load_global_constant(b, comps, 32, addr, .align_mul = 4);
}

void
hk_nir_line_stipple_prefix(nir_builder *b, const void *key_)
{
   const struct hk_line_stipple_prefix_key *key = key_;
   nir_shader *s = b->shader;
   s->info.name = "line stipple prefix";
   s->info.workgroup_size[0] = HK_STIPPLE_PREFIX_THREADS;
   s->info.workgroup_size[1] = s->info.workgroup_size[2] = 1;
   s->info.shared_size = HK_STIPPLE_PREFIX_SHARED_B;

   nir_def *tid = nir_load_local_invocation_index(b);
   nir_def *vp = nir_load_vertex_param_buffer_poly(b);
   nir_def *p = nir_load_geometry_param_buffer_poly(b);
   nir_def *n = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, p, offsetof(struct poly_geometry_params, input_primitives)),
      .align_mul = 4);
   nir_def *per = nir_load_global_constant(
      b, 1, 32, nir_iadd_imm(b, p, offsetof(struct poly_geometry_params, grid)),
      .align_mul = 4);
   nir_def *verts = nir_load_global_constant(
      b, 1, 32, nir_iadd_imm(b, vp, offsetof(struct poly_vertex_params, grid)),
      .align_mul = 4);
   nir_def *last =
      nir_ieq_imm(b,
                  hk_nir_load_root(b, 1, 16,
                                   hk_root_descriptor_offset(draw.provoking)),
                  2);

   /* The counters: given, or allocated from the heap by the first thread */
   nir_def *slot = hk_nir_load_root(
      b, 1, 64, hk_root_descriptor_offset(draw.stipple_counters));
   nir_push_if(b, nir_ieq_imm(b, tid, 0));
   {
      nir_def *given = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, slot, offsetof(struct hk_line_stipple_counters, counters)),
         .align_mul = 8);
      nir_push_if(b, nir_ieq_imm(b, given, 0));
      nir_def *allocated;
      {
         nir_def *heap = nir_load_global(
            b, 1, 64,
            nir_iadd_imm(b, slot, offsetof(struct hk_line_stipple_counters, heap)),
            .align_mul = 8);
         nir_def *units = nir_udiv_imm(
            b, nir_iadd_imm(b, nir_imul_imm(b, n, 8), POLY_HEAP_UNIT_B - 1),
            POLY_HEAP_UNIT_B);
         nir_def *first = nir_global_atomic(
            b, 32, nir_iadd_imm(b, heap, offsetof(struct poly_heap, bottom)),
            units, .atomic_op = nir_atomic_op_iadd);
         nir_def *size = nir_load_global(
            b, 1, 32, nir_iadd_imm(b, heap, offsetof(struct poly_heap, size)),
            .align_mul = 4);
         nir_def *end =
            nir_imul_imm(b, nir_u2u64(b, nir_iadd(b, first, units)),
                         POLY_HEAP_UNIT_B);
         nir_def *base = nir_load_global(
            b, 1, 64, nir_iadd_imm(b, heap, offsetof(struct poly_heap, base)),
            .align_mul = 8);
         allocated = nir_bcsel(
            b, nir_ult(b, nir_u2u64(b, size), end), nir_imm_int64(b, 0),
            nir_iadd(b, base,
                     nir_imul_imm(b, nir_u2u64(b, first), POLY_HEAP_UNIT_B)));
         nir_store_global(
            b, allocated,
            nir_iadd_imm(b, slot,
                         offsetof(struct hk_line_stipple_counters, counters)),
            .align_mul = 8);
      }
      nir_pop_if(b, NULL);
      nir_def *counters = nir_if_phi(b, allocated, given);
      nir_store_shared(b, counters, nir_imm_int(b, 4 * 32), .base = 0,
                       .align_mul = 8);
   }
   nir_pop_if(b, NULL);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   nir_def *counters = nir_load_shared(b, 1, 64, nir_imm_int(b, 4 * 32),
                                       .base = 0, .align_mul = 8);

   /* An exhausted heap leaves no counters, and every segment starts at 0 */
   nir_push_if(b, nir_ine_imm(b, counters, 0));
   {
      nir_variable *start =
         nir_local_variable_create(b->impl, glsl_uint_type(), "start");
      nir_variable *sum =
         nir_local_variable_create(b->impl, glsl_uint_type(), "sum");
      nir_variable *strip =
         nir_local_variable_create(b->impl, glsl_uint_type(), "strip");
      nir_store_var(b, start, nir_imm_int(b, 0), 1);
      nir_store_var(b, sum, nir_imm_int(b, 0), 1);
      nir_store_var(b, strip, nir_imm_int(b, 0), 1);

      nir_push_loop(b);
      {
         nir_def *first = nir_load_var(b, start);
         nir_break_if(b, nir_uge(b, first, n));

         nir_def *i = nir_iadd(b, first, tid);
         nir_def *valid = nir_ult(b, i, n);
         nir_def *instance = nir_udiv(b, i, per);
         nir_def *j = nir_isub(b, i, nir_imul(b, instance, per));

         /* Segment j is vertices (j, j + 1), or (2j, 2j + 1) once unrolled */
         nir_def *local = key->restart ? nir_imul_imm(b, j, 2) : j;
         nir_def *v0 = nir_iadd(b, nir_imul(b, instance, verts), local);
         nir_def *v1 = nir_iadd_imm(b, v0, 1);

         nir_def *length = nir_imm_int(b, 0);
         nir_def *starts = nir_ieq_imm(b, j, 0);
         nir_push_if(b, valid);
         nir_def *fixed, *new_strip;
         {
            nir_def *vp_index = nir_imm_int(b, 0);
            if (key->viewport) {
               vp_index = stipple_vertex_output(
                  b, vp, nir_bcsel(b, last, v1, v0), VARYING_SLOT_VIEWPORT, 1);
            }

            nir_def *w0 = hk_nir_line_stipple_window(
               b, stipple_vertex_output(b, vp, v0, VARYING_SLOT_POS, 4),
               vp_index);
            nir_def *w1 = hk_nir_line_stipple_window(
               b, stipple_vertex_output(b, vp, v1, VARYING_SLOT_POS, 4),
               vp_index);
            fixed = nir_f2u32(
               b, nir_fround_even(
                     b, nir_fmul_imm(b, hk_nir_line_stipple_length(b, w0, w1),
                                     256.0)));

            /* A restart starts a new strip */
            new_strip = starts;
            if (key->restart) {
               nir_def *size = poly_index_size(b, vp);
               nir_def *unbroken = nir_imm_false(b);
               nir_push_if(b, nir_iand(b, nir_ine_imm(b, size, 0),
                                       nir_inot(b, starts)));
               nir_def *a = poly_load_index_buffer(b, vp, local, size);
               nir_def *c = poly_load_index_buffer(
                  b, vp, nir_iadd_imm(b, local, -1), size);
               nir_def *differs = nir_ine(b, a, c);
               nir_pop_if(b, NULL);
               nir_def *broken = nir_if_phi(b, differs, unbroken);
               new_strip = nir_ior(b, starts, broken);
            }
         }
         nir_pop_if(b, NULL);
         length = nir_if_phi(b, fixed, length);
         starts = nir_if_phi(b, new_strip, starts);

         /* The sum before each segment, and its strip's first segment */
         nir_def *total_length, *latest;
         nir_def *before = nir_isub(
            b, stipple_wg_scan(b, length, nir_op_iadd, &total_length), length);
         before = nir_iadd(b, before, nir_load_var(b, sum));
         nir_def *strip_first = stipple_wg_scan(
            b, nir_bcsel(b, starts, nir_iadd_imm(b, i, 1), nir_imm_int(b, 0)),
            nir_op_umax, &latest);
         strip_first = nir_umax(b, strip_first, nir_load_var(b, strip));

         nir_push_if(b, valid);
         {
            nir_store_global(
               b, nir_vec2(b, before, nir_iadd_imm(b, strip_first, -1)),
               nir_iadd(b, counters, nir_u2u64(b, nir_imul_imm(b, i, 8))),
               .align_mul = 8);
         }
         nir_pop_if(b, NULL);

         nir_store_var(b, sum, nir_iadd(b, nir_load_var(b, sum), total_length),
                       1);
         nir_store_var(b, strip, nir_umax(b, nir_load_var(b, strip), latest), 1);
         nir_store_var(b, start, nir_iadd_imm(b, first, HK_STIPPLE_PREFIX_THREADS),
                       1);
      }
      nir_pop_loop(b, NULL);

      /* Every sum is written: subtract the sum before each strip's start */
      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_global);

      nir_store_var(b, start, tid, 1);
      nir_push_loop(b);
      {
         nir_def *i = nir_load_var(b, start);
         nir_break_if(b, nir_uge(b, i, n));

         nir_def *entry =
            nir_iadd(b, counters, nir_u2u64(b, nir_imul_imm(b, i, 8)));
         nir_def *pair = nir_load_global(b, 2, 32, entry, .align_mul = 8);
         nir_def *first_sum = nir_load_global(
            b, 1, 32,
            nir_iadd(b, counters,
                     nir_u2u64(b, nir_imul_imm(b, nir_channel(b, pair, 1), 8))),
            .align_mul = 8);
         nir_store_global(b, nir_isub(b, nir_channel(b, pair, 0), first_sum),
                          nir_iadd_imm(b, entry, 4), .align_mul = 4);

         nir_store_var(b, start, nir_iadd_imm(b, i, HK_STIPPLE_PREFIX_THREADS),
                       1);
      }
      nir_pop_loop(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_progress(true, b->impl, nir_metadata_none);
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
}

/* Line passthrough GS: segments restart the counter, strips continue it */
void
hk_nir_line_stipple_passthrough(nir_shader *gs, bool strip, bool viewport)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *pos[2], *window[2];
   nir_def *provoking =
      nir_bcsel(&b, nir_ine_imm(&b, nir_load_provoking_last(&b), 0),
                nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_def *vp = passthrough_viewport(&b, provoking, viewport);
   for (unsigned i = 0; i < 2; ++i) {
      pos[i] = nir_load_per_vertex_input(
         &b, 4, 32, nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      window[i] = hk_nir_line_stipple_window(&b, pos[i], vp);
   }

   nir_def *base = nir_imm_float(&b, 0.0);
   if (strip) {
      nir_def *slot = hk_nir_load_root(
         &b, 1, 64, hk_root_descriptor_offset(draw.stipple_counters));
      nir_def *counters = nir_load_global(
         &b, 1, 64,
         nir_iadd_imm(&b, slot,
                      offsetof(struct hk_line_stipple_counters, counters)),
         .align_mul = 8);

      nir_push_if(&b, nir_ine_imm(&b, counters, 0));
      nir_def *counter;
      {
         nir_def *per = nir_load_global_constant(
            &b, 1, 32,
            nir_iadd_imm(&b, nir_load_geometry_param_buffer_poly(&b),
                         offsetof(struct poly_geometry_params, grid)),
            .align_mul = 4);
         nir_def *i = nir_iadd(&b, nir_imul(&b, nir_load_instance_id(&b), per),
                               nir_load_primitive_id(&b));
         nir_def *fixed = nir_load_global(
            &b, 1, 32,
            nir_iadd(&b, counters,
                     nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, i, 8), 4))),
            .align_mul = 4);

         /* Only the counter modulo the pattern's period matters */
         nir_def *stipple = hk_nir_load_root(
            &b, 1, 32, hk_root_descriptor_offset(draw.line_stipple));
         nir_def *period =
            nir_imul_imm(&b, nir_ushr_imm(&b, stipple, 16), 16 * 256);
         counter = nir_fmul_imm(&b, nir_u2f32(&b, nir_umod(&b, fixed, period)),
                                1.0 / 256.0);
      }
      nir_pop_if(&b, NULL);
      base = nir_if_phi(&b, counter, base);
   }

   nir_def *word = hk_nir_line_stipple_word(&b, window[0], window[1], base);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic !=
                nir_intrinsic_emit_vertex)
            continue;

         b.cursor = nir_before_instr(instr);
         hk_nir_store_line_stipple(&b, word);
      }
   }

   nir_progress(true, impl, nir_metadata_none);
}

/* GS line strips: each vertex after a strip's first emits its own segment */
struct gs_slot {
   nir_io_semantics sem;
   nir_alu_type type;
   uint8_t mask;
};

struct gs_outputs {
   struct gs_slot slot[VARYING_SLOT_MAX];
   unsigned first, count;
   nir_variable *cur, *prev;
};

/* Outputs are kept as words: private memory is accessed a word at a time */
static nir_def *
gs_load_slot(nir_builder *b, nir_variable *var, unsigned slot)
{
   nir_def *c[4];
   for (unsigned i = 0; i < 4; ++i)
      c[i] = nir_load_array_var_imm(b, var, slot * 4 + i);
   return nir_vec(b, c, 4);
}

static nir_def *
gs_tap(nir_builder *b, struct gs_outputs *o, unsigned location)
{
   return gs_load_slot(b, o->cur, location - o->first);
}

static void
gs_emit_outputs(nir_builder *b, struct gs_outputs *o, bool prev, nir_def *word)
{
   for (unsigned i = 0; i < o->count; ++i) {
      struct gs_slot *s = &o->slot[o->first + i];
      if (!s->mask)
         continue;

      nir_def *value = gs_load_slot(b, prev ? o->prev : o->cur, i);
      nir_store_output(b, value, nir_imm_int(b, 0), .write_mask = s->mask,
                       .src_type = s->type, .io_semantics = s->sem);
   }

   hk_nir_store_line_stipple(b, word);
}

void
hk_nir_lower_gs_line_stipple(nir_shader *gs)
{
   assert(gs->info.gs.output_primitive == MESA_PRIM_LINE_STRIP);
   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* Keep 32-bit output slots in variables until their vertex is emitted */
   struct gs_outputs *o = calloc(1, sizeof(*o));
   unsigned first = VARYING_SLOT_MAX, end = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;

         assert(nir_src_bit_size(intr->src[0]) == 32);
         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         unsigned mask = nir_intrinsic_write_mask(intr)
                         << nir_intrinsic_component(intr);
         for (unsigned i = 0; i < sem.num_slots; ++i) {
            struct gs_slot *s = &o->slot[sem.location + i];
            s->sem = sem;
            s->sem.location = sem.location + i;
            s->sem.num_slots = 1;
            s->type = nir_intrinsic_src_type(intr);
            s->mask |= mask;
         }
         first = MIN2(first, sem.location);
         end = MAX2(end, sem.location + sem.num_slots);
      }
   }

   if (first < end) {
      o->first = first;
      o->count = end - first;
      const struct glsl_type *type =
         glsl_array_type(glsl_uint_type(), o->count * 4, 0);
      o->cur = nir_local_variable_create(impl, type, "stipple_cur");
      o->prev = nir_local_variable_create(impl, type, "stipple_prev");
   }

   nir_variable *have_prev =
      nir_local_variable_create(impl, glsl_bool_type(), "stipple_have_prev");
   nir_variable *prev_window =
      nir_local_variable_create(impl, glsl_vec_type(2), "stipple_prev_window");
   nir_variable *base =
      nir_local_variable_create(impl, glsl_float_type(), "stipple_base");
   nir_store_var(&b, have_prev, nir_imm_false(&b), 1);
   nir_store_var(&b, base, nir_imm_float(&b, 0.0), 1);

   bool pos = o->count && o->slot[VARYING_SLOT_POS].mask;
   bool viewport = o->count && o->slot[VARYING_SLOT_VIEWPORT].mask;

   /* Gather the shader's intrinsics before the rewrite adds its own */
   struct util_dynarray intrinsics = UTIL_DYNARRAY_INIT;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic)
            util_dynarray_append(&intrinsics, nir_instr_as_intrinsic(instr));
      }
   }

   {
      util_dynarray_foreach(&intrinsics, nir_intrinsic_instr *, it) {
         nir_intrinsic_instr *intr = *it;
         nir_instr *instr = &intr->instr;
         b.cursor = nir_before_instr(instr);

         if (intr->intrinsic == nir_intrinsic_store_output) {
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            nir_def *index = nir_imul_imm(
               &b, nir_iadd_imm(&b, intr->src[1].ssa, sem.location - o->first),
               4);
            unsigned c0 = nir_intrinsic_component(intr);
            u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
               nir_store_array_var(
                  &b, o->cur, nir_iadd_imm(&b, index, c0 + c),
                  nir_bitcast_vector(&b, nir_channel(&b, intr->src[0].ssa, c),
                                     32),
                  0x1);
            }
            nir_instr_remove(instr);
         } else if (intr->intrinsic == nir_intrinsic_end_primitive) {
            nir_store_var(&b, have_prev, nir_imm_false(&b), 1);
            nir_store_var(&b, base, nir_imm_float(&b, 0.0), 1);
            nir_instr_remove(instr);
         } else if (intr->intrinsic == nir_intrinsic_emit_vertex) {
            unsigned stream = nir_intrinsic_stream_id(intr);
            nir_def *clip = pos ? gs_tap(&b, o, VARYING_SLOT_POS)
                                : nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0);
            nir_def *vp =
               viewport
                  ? nir_channel(&b, gs_tap(&b, o, VARYING_SLOT_VIEWPORT), 0)
                  : nir_imm_int(&b, 0);
            nir_def *window = hk_nir_line_stipple_window(&b, clip, vp);

            nir_push_if(&b, nir_load_var(&b, have_prev));
            {
               nir_def *w0 = nir_load_var(&b, prev_window);
               nir_def *start = nir_load_var(&b, base);
               nir_def *word = hk_nir_line_stipple_word(&b, w0, window, start);

               gs_emit_outputs(&b, o, true, word);
               nir_emit_vertex(&b, .stream_id = stream);
               gs_emit_outputs(&b, o, false, word);
               nir_emit_vertex(&b, .stream_id = stream);
               nir_end_primitive(&b, .stream_id = stream);

               nir_store_var(
                  &b, base,
                  nir_fadd(&b, start,
                           hk_nir_line_stipple_length(&b, w0, window)),
                  1);
            }
            nir_pop_if(&b, NULL);

            for (unsigned i = 0; i < o->count * 4; ++i) {
               nir_store_array_var_imm(
                  &b, o->prev, i, nir_load_array_var_imm(&b, o->cur, i), 0x1);
            }
            nir_store_var(&b, prev_window, window, 0x3);
            nir_store_var(&b, have_prev, nir_imm_true(&b), 1);
            nir_instr_remove(instr);
         }
      }
   }

   util_dynarray_fini(&intrinsics);
   free(o);

   /* Every vertex but a strip's first emits a segment of two */
   gs->info.gs.vertices_out = MAX2(gs->info.gs.vertices_out, 1) * 2;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
}

/* Demote fragments whose stipple bit is clear */
bool
hk_nir_lower_line_stipple_fs(nir_shader *fs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *word = nir_load_input(
      &b, 1, 32, nir_imm_int(&b, 0), .dest_type = nir_type_uint32,
      .io_semantics = {.location = AGX_LINE_STIPPLE_SLOT, .num_slots = 1});
   fs->info.inputs_read |= BITFIELD64_BIT(AGX_LINE_STIPPLE_SLOT);

   nir_def *pixel = nir_fadd_imm(
      &b, nir_u2f32(&b, nir_u2u32(&b, nir_load_pixel_coord(&b))), 0.5);
   nir_def *x_major = nir_ine_imm(&b, nir_iand_imm(&b, word, 2), 0);
   nir_def *negative = nir_ine_imm(&b, nir_iand_imm(&b, word, 1), 0);
   nir_def *c =
      nir_fmul_imm(&b, nir_i2f32(&b, nir_ishr_imm(&b, word, 2)), 1.0 / 256.0);
   nir_def *p = nir_bcsel(&b, x_major, nir_channel(&b, pixel, 0),
                          nir_channel(&b, pixel, 1));
   nir_def *along =
      nir_bcsel(&b, negative, nir_fsub(&b, c, p), nir_fadd(&b, c, p));
   agx_nir_line_stipple_test(
      &b, nir_f2u32(&b, nir_fmax(&b, along, nir_imm_float(&b, 0.0))));

   NIR_PASS(_, fs, nir_recompute_io_bases, nir_var_shader_in);
   return nir_progress(true, impl, nir_metadata_none);
}
