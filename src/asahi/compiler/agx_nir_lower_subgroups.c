/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/list.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

/* XXX: cribbed from nak, move to common */
static nir_def *
nir_udiv_round_up(nir_builder *b, nir_def *n, nir_def *d)
{
   return nir_udiv(b, nir_iadd(b, n, nir_iadd_imm(b, d, -1)), d);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_vote_any: {
      nir_def *ballot = nir_quad_ballot_agx(b, 16, intr->src[0].ssa);
      nir_def_rewrite_uses(&intr->def, nir_ine_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_quad_vote_all: {
      nir_def *ballot =
         nir_quad_ballot_agx(b, 16, nir_inot(b, intr->src[0].ssa));
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_elect: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, active_id, 0));
      return true;
   }

   case nir_intrinsic_first_invocation: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def *is_first = nir_ieq_imm(b, active_id, 0);
      nir_def *first_bit = nir_ballot(b, 1, 32, is_first);
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, first_bit));
      return true;
   }

   case nir_intrinsic_last_invocation: {
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, active_mask));
      return true;
   }

   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      /* The common lowering does:
       *
       *    vote_all(x == read_first(x))
       *
       * This is not optimal for AGX, since we have ufind_msb but not ctz, so
       * it's cheaper to read the last invocation than the first. So we do:
       *
       *    vote_all(x == read_last(x))
       *
       * implemented with lowered instructions as
       *
       *    ballot(x != broadcast(x, ffs(ballot(true)))) == 0
       */
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *active_bit = nir_ufind_msb(b, active_mask);
      nir_def *other = nir_read_invocation(b, intr->src[0].ssa, active_bit);
      nir_def *is_ne;

      if (intr->intrinsic == nir_intrinsic_vote_feq) {
         is_ne = nir_fneu(b, other, intr->src[0].ssa);
      } else {
         is_ne = nir_ine(b, other, intr->src[0].ssa);
      }

      nir_def *ballot = nir_ballot(b, 1, 32, is_ne);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_load_num_subgroups: {
      nir_def *workgroup_size = nir_load_workgroup_size(b);
      workgroup_size = nir_imul(b,
                                nir_imul(b, nir_channel(b, workgroup_size, 0),
                                         nir_channel(b, workgroup_size, 1)),
                                nir_channel(b, workgroup_size, 2));
      nir_def *subgroup_size = nir_imm_int(b, 32);
      nir_def *num_subgroups =
         nir_udiv_round_up(b, workgroup_size, subgroup_size);
      nir_def_rewrite_uses(&intr->def, num_subgroups);
      return true;
   }

   case nir_intrinsic_shuffle: {
      nir_def *data = intr->src[0].ssa;
      nir_def *target = intr->src[1].ssa;

      /* The hardware shuffle instruction chooses a single index within the
       * target quad to shuffle each source quad with. Consequently, the low
       * 2-bits of shuffle indices should not be quad divergent.  To implement
       * arbitrary shuffle, pull each low 2-bits index in the quad separately.
       */
      nir_def *quad_start = nir_iand_imm(b, target, 0x1c);
      nir_def *result = NULL;

      for (unsigned i = 0; i < 4; ++i) {
         nir_def *target_i = nir_iadd_imm(b, quad_start, i);
         nir_def *shuf = nir_read_invocation(b, data, target_i);

         if (result)
            result = nir_bcsel(b, nir_ieq(b, target, target_i), shuf, result);
         else
            result = shuf;
      }

      nir_def_rewrite_uses(&intr->def, result);
      return true;
   }

   case nir_intrinsic_inclusive_scan: {
      /* If we got here, we support the corresponding exclusive scan in
       * hardware, so just handle the last element.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      nir_def *data = intr->src[0].ssa;

      b->cursor = nir_after_instr(&intr->instr);
      intr->intrinsic = nir_intrinsic_exclusive_scan;
      nir_def *accum = nir_build_alu2(b, red_op, data, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, accum);
      return true;
   }

   case nir_intrinsic_ballot: {
      /* Optimize popcount(ballot(true)) to load_active_subgroup_count_agx() */
      if (!nir_src_is_const(intr->src[0]) || !nir_src_as_bool(intr->src[0]) ||
          !list_is_singular(&intr->def.uses))
         return false;

      nir_src *use = list_first_entry(&intr->def.uses, nir_src, use_link);
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op != nir_op_bit_count)
         return false;

      nir_def_rewrite_uses(&alu->def,
                           nir_load_active_subgroup_count_agx(b, 32));
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_subgroup_filter(const nir_intrinsic_instr *intr, UNUSED const void *data)
{
   /* Use default behaviour for everything but scans */
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_reduce)
      return true;

   if (intr->def.num_components > 1 || intr->def.bit_size == 1)
      return true;

   /* Hardware supports quad ops but no other support clustered reductions. */
   if (nir_intrinsic_has_cluster_size(intr)) {
      unsigned cluster = nir_intrinsic_cluster_size(intr);
      if (cluster && cluster != 4 && cluster < 32)
         return true;
   }

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imul:
   case nir_op_fmin:
   case nir_op_fmax:
      /* no hardware scan for this opcode, always lower it */
      return true;

   case nir_op_iadd:
   case nir_op_iand:
   case nir_op_ixor:
   case nir_op_ior:
      /* these have dedicated 64-bit lowering paths that use the 32-bit hardware
       * instructions so are likely better than the full lowering.
       */
      return false;

   default:
      /* otherwise, lower 64-bit, since the hw ops are at most 32-bit. */
      return intr->def.bit_size == 64;
   }
}

/* Minimal subgroups have an exclusive iadd scan and umax reduction, 32-bit */
static bool
lower_subgroup_filter_minimal(const nir_intrinsic_instr *intr,
                              UNUSED const void *data)
{
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_reduce)
      return true;

   if (intr->def.num_components > 1 || intr->def.bit_size != 32)
      return true;

   if (intr->intrinsic == nir_intrinsic_reduce) {
      return nir_intrinsic_reduction_op(intr) != nir_op_umax ||
             nir_intrinsic_cluster_size(intr) != 0;
   }

   return nir_intrinsic_reduction_op(intr) != nir_op_iadd;
}

/* Without quad ballots and active-lane registers, use a full ballot */
static bool
lower_minimal(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl;

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_ballot_agx: {
      nir_def *ballot = nir_ballot(b, 1, 32, intr->src[0].ssa);
      nir_def *quad = nir_iand_imm(b, nir_load_subgroup_invocation(b), ~3);
      repl = nir_iand_imm(b, nir_ushr(b, ballot, quad), 0xf);
      break;
   }

   case nir_intrinsic_load_active_subgroup_count_agx:
      repl = nir_bit_count(b, nir_ballot(b, 1, 32, nir_imm_true(b)));
      break;

   case nir_intrinsic_load_active_subgroup_invocation_agx: {
      /* Count the active lanes below this one */
      nir_def *active = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *below = nir_iadd_imm(
         b, nir_ishl(b, nir_imm_int(b, 1), nir_load_subgroup_invocation(b)),
         -1);
      repl = nir_bit_count(b, nir_iand(b, active, below));
      break;
   }

   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, repl, intr->def.bit_size));
   return true;
}

bool
agx_nir_lower_subgroups(nir_shader *s)
{
   bool minimal = agx_nir_target(s)->minimal_subgroups;

   /* First, do as much common lowering as we can */
   nir_lower_subgroups_options opts = {
      .filter = minimal ? lower_subgroup_filter_minimal : lower_subgroup_filter,
      .lower_read_first_invocation = true,
      .lower_inverse_ballot = true,
      .lower_to_scalar = true,
      .lower_relative_shuffle = true,
      .lower_rotate_to_shuffle = true,
      .lower_subgroup_masks = true,
      .lower_reduce = true,
      .lower_vote = true,
      .lower_quad = minimal,
      .lower_quad_vote = minimal,
      .ballot_components = 1,
      .ballot_bit_size = 32,
      .subgroup_size = 32,
   };

   bool progress = nir_lower_subgroups(s, &opts);

   /* Then do AGX-only lowerings on top */
   progress |=
      nir_shader_intrinsics_pass(s, lower, nir_metadata_control_flow, NULL);

   if (minimal) {
      progress |= nir_shader_intrinsics_pass(s, lower_minimal,
                                             nir_metadata_control_flow, NULL);
   }

   return progress;
}

/* Compute linear groups are consecutive sets of four local invocations.
 * Use the same subgroup lane transport as quad operations, with explicit
 * subtraction so no fragment-only derivative instruction is required.
 */
static bool
lower_compute_derivative(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   bool y, coarse;
   switch (intr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
      y = false;
      break;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      y = true;
      break;
   default:
      return false;
   }
   coarse = intr->intrinsic == nir_intrinsic_ddx_coarse ||
            intr->intrinsic == nir_intrinsic_ddy_coarse;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *lane = nir_load_subgroup_invocation(b);
   unsigned step = y ? 2 : 1;
   nir_def *lo = nir_iand_imm(b, lane, ~(coarse ? 3u : step));
   nir_def *hi = nir_ior_imm(b, lo, step);
   nir_def *value = intr->src[0].ssa;
   nir_def *a = nir_shuffle(b, value, lo);
   nir_def *z = nir_shuffle(b, value, hi);
   nir_def_replace(&intr->def, nir_fsub(b, z, a));
   return true;
}

bool
agx_nir_lower_compute_derivatives(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.derivative_group != DERIVATIVE_GROUP_LINEAR)
      return false;

   return nir_shader_intrinsics_pass(shader, lower_compute_derivative,
                                     nir_metadata_control_flow, NULL);
}
