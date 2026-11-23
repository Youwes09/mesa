/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_linker.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "hk_cmd_buffer.h"
#include "hk_shader.h"

/*
 * Line stipple on Apple9. The stage feeding the rasterizer gives every vertex
 * of a line segment one flat word describing the segment, and the fragment
 * shader demotes fragments whose stipple bit is clear
 * (agx_nir_line_stipple_test).
 *
 * A fragment's stipple counter is the counter at the segment's first vertex
 * plus its distance from that vertex along the segment's major axis, in
 * window space. Along the major axis this is dir * p + c, where p is the
 * fragment's coordinate, dir the sign of the segment's direction and
 * c = base - dir * s0 for the first vertex's coordinate s0. The word holds c
 * in 1/256 pixel fixed point shifted left by two, whether the segment is
 * x-major in bit 1, and a negative direction in bit 0. Vertices are snapped
 * to 1/256 pixel, so c is exact, and the base is reduced modulo the
 * pattern's period, which keeps it small.
 *
 * Each segment has its own vertices: the passthrough geometry shader and the
 * mesh rasterizing shader emit them per segment, and geometry shaders
 * emitting line strips are rewritten to emit each segment separately.
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

/*
 * The passthrough geometry shader of a line draw. The counter restarts at
 * every independent segment. Along strips it continues: each segment adds the
 * lengths of the strip's earlier segments, read back from the vertex stage's
 * outputs. Like the gl driver, strips longer than
 * AGX_LINE_STIPPLE_MAX_STRIP_SEGMENTS restart at each later segment, which
 * bounds the work of every invocation.
 */
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
      nir_variable *i = nir_local_variable_create(impl, glsl_uint_type(), NULL);
      nir_variable *sum =
         nir_local_variable_create(impl, glsl_float_type(), NULL);
      nir_store_var(&b, i, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum, nir_imm_float(&b, 0.0), 1);
      nir_def *prim = nir_load_primitive_id(&b);
      nir_def *limit = nir_bcsel(
         &b, nir_ult_imm(&b, prim, AGX_LINE_STIPPLE_MAX_STRIP_SEGMENTS), prim,
         nir_imm_int(&b, 0));

      nir_push_loop(&b);
      {
         nir_def *j = nir_load_var(&b, i);
         nir_break_if(&b, nir_uge(&b, j, limit));
         nir_def *a = hk_nir_line_stipple_window(
            &b, nir_load_strip_position_poly(&b, j), vp);
         nir_def *c = hk_nir_line_stipple_window(
            &b, nir_load_strip_position_poly(&b, nir_iadd_imm(&b, j, 1)), vp);
         nir_store_var(&b, sum,
                       nir_fadd(&b, nir_load_var(&b, sum),
                                hk_nir_line_stipple_length(&b, a, c)),
                       1);
         nir_store_var(&b, i, nir_iadd_imm(&b, j, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      base = nir_load_var(&b, sum);
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
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
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
