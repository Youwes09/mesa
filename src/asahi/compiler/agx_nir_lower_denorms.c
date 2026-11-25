/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_apple9_layout.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/set.h"
#include "agx_apple9_machine.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

/* Flush FP32 subnormals the hardware does not flush itself, where observable */
static nir_def *
flush_fp32(nir_builder *b, nir_def *x)
{
   /* Integer classification must distinguish subnormals from signed zero,
    * normals, infinities and NaNs without first consuming x as a float. */
   nir_def *tiny = nir_ult_imm(b, nir_iand_imm(b, x, 0x7fffffff), 0x00800000);
   return nir_bcsel(b, tiny, nir_iand_imm(b, x, 0x80000000), x);
}

static bool
fp32_result_is_canonical(nir_scalar s, unsigned depth,
                         const BITSET_WORD *canonical_uniforms)
{
   if (s.def->bit_size != 32 || depth == 8)
      return false;

   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      uint32_t magnitude = nir_scalar_as_uint(s) & 0x7fffffff;
      return magnitude == 0 || magnitude >= 0x00800000;
   }

   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_foreach_phi_src(src, nir_def_as_phi(s.def)) {
         if (!fp32_result_is_canonical(nir_get_scalar(src->src.ssa, s.comp),
                                       depth + 1, canonical_uniforms))
            return false;
      }
      return true;
   }

   if (canonical_uniforms &&
       nir_def_instr_type(s.def) == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_preamble_class(intr) == nir_preamble_class_general) {
         unsigned word = nir_intrinsic_base(intr) + s.comp;
         return word < AGX_APPLE9_UNIFORM_COUNT &&
                BITSET_TEST(canonical_uniforms, word);
      }
   }

   if (!nir_scalar_is_alu(s))
      return false;

   nir_alu_instr *alu = nir_def_as_alu(s.def);
   if (alu->op == nir_op_bcsel)
      return fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 1), depth + 1,
                                      canonical_uniforms) &&
             fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 2), depth + 1,
                                      canonical_uniforms);

   /* This query runs before rewriting any instruction. Under explicit FTZ,
    * floating ALU results are canonical either natively or after this pass.
    * The initial consumer check excludes sign/min/max operations. Only after
    * selecting all checks may pruning defer one to a bit-preserving user.
    */
   return nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) ==
          nir_type_float;
}

static bool
fp32_flush_is_unobservable(nir_def *def, unsigned depth,
                           const struct set *flush)
{
   if (depth == 8)
      return false;

   nir_foreach_use(src, def) {
      if (nir_src_is_if(src))
         return false;
      nir_instr *instr = nir_src_use_instr(src);
      if (flush && instr->type == nir_instr_type_phi) {
         if (!fp32_flush_is_unobservable(&nir_instr_as_phi(instr)->def,
                                         depth + 1, flush))
            return false;
         continue;
      }
      if (instr->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      /* All forms of these native FP32 operations consume subnormals as
       * signed zero. Min/max and sign operations do not: their selected or
       * modified operand bits can escape through a store or integer use. */
      case nir_op_fadd:
      case nir_op_fsub:
      case nir_op_fmul:
      case nir_op_ffma:
      case nir_op_ffma_weak:
      case nir_op_flt:
      case nir_op_fge:
      case nir_op_feq:
      case nir_op_fneu:
         break;
      case nir_op_bcsel:
         if (!flush || src == &alu->src[0].src ||
             !fp32_flush_is_unobservable(&alu->def, depth + 1, flush))
            return false;
         break;
      case nir_op_fneg:
      case nir_op_fabs:
      case nir_op_fmin:
      case nir_op_fmax:
      case nir_op_fsat:
      case nir_op_fcanonicalize:
         /* These commute with signed-zero canonicalization: sign changes
          * preserve magnitude, and native min/max comparisons use DAZ.
          * Delay an upstream check only to an independently retained check
          * or to a proven DAZ use of the entire chain. Do not use this rule
          * while constructing the initial canonical-producer proof. */
         if (!flush ||
             (!_mesa_set_search(flush, alu) &&
              !fp32_flush_is_unobservable(&alu->def, depth + 1, flush)))
            return false;
         break;
      default:
         if (!nir_op_is_vec_or_mov(alu->op) ||
             !fp32_flush_is_unobservable(&alu->def, depth + 1, flush))
            return false;
         break;
      }
   }
   return true;
}

static bool
fp32_minmax_has_normal_bound(nir_scalar s, bool lower,
                             const uint32_t *uniform_constants)
{
   s = nir_scalar_chase_movs(s);
   uint32_t value;
   if (nir_scalar_is_const(s)) {
      value = nir_scalar_as_uint(s);
   } else if (nir_def_instr_type(s.def) == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic != nir_intrinsic_load_preamble ||
          nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
         return false;
      unsigned word = nir_intrinsic_base(intr) + s.comp;
      if (word >= AGX_APPLE9_UNIFORM_COUNT)
         return false;
      value = uniform_constants[word];
   } else {
      return false;
   }

   /* max(x, positive-normal) and min(x, negative-normal) cannot select a
    * subnormal. Infinities are valid bounds; NaNs are not. Unknown uniform
    * constants are represented by zero, which never establishes a bound. */
   uint32_t magnitude = value & 0x7fffffff;
   return (value >> 31) == !lower && magnitude >= 0x00800000 &&
          magnitude <= 0x7f800000;
}

static bool
needs_fp32_flush(nir_alu_instr *alu, const BITSET_WORD *canonical_uniforms,
                 const uint32_t *uniform_constants)
{
   if (alu->def.bit_size != 32 ||
       nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) !=
          nir_type_float)
      return false;

   switch (alu->op) {
   /* T8132 exact-bit probes separate input DAZ from output FTZ for these
    * native operations, including compact, immediate and source-modifier
    * forms. FMA retains an underflowing product until the final rounding. */
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_ffma_weak:
      return false;
   /* Normal inputs producing subnormal exp2 results also flush on T8132. */
   case nir_op_fexp2:
      return false;
   /* These results are integral, or have magnitudes bounded away from the
    * FP32 subnormal interval by their input format. */
   case nir_op_b2f32:
   case nir_op_i2f32:
   case nir_op_u2f32:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_fsqrt:
   case nir_op_frsq:
   case nir_op_flog2:
      return false;
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fsat:
   case nir_op_fcanonicalize:
   case nir_op_fmin:
   case nir_op_fmax: {
      /* These operations cannot create a subnormal from canonical operands.
       * Prove every component, including swizzles, before eliding the check. */
      bool canonical = true;
      for (unsigned c = 0; c < alu->def.num_components; ++c) {
         nir_scalar result = nir_get_scalar(&alu->def, c);
         if ((alu->op == nir_op_fmin || alu->op == nir_op_fmax) &&
             (fp32_minmax_has_normal_bound(nir_scalar_chase_alu_src(result, 0),
                                           alu->op == nir_op_fmax,
                                           uniform_constants) ||
              fp32_minmax_has_normal_bound(nir_scalar_chase_alu_src(result, 1),
                                           alu->op == nir_op_fmax,
                                           uniform_constants)))
            continue;
         for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
            canonical &= fp32_result_is_canonical(
               nir_scalar_chase_alu_src(result, i), 0, canonical_uniforms);
      }
      if (canonical)
         return false;
      break;
   }
   case nir_op_f2f32:
      if (alu->src[0].src.ssa->bit_size == 16)
         return false;
      break;
   default:
      break;
   }

   return !fp32_flush_is_unobservable(&alu->def, 0, NULL);
}

static bool
lower_denorm(nir_builder *b, nir_alu_instr *alu, void *data)
{
   bool flush = _mesa_set_search(data, alu) != NULL;
   if (alu->op == nir_op_fcanonicalize && alu->def.bit_size == 32) {
      b->cursor = nir_before_instr(&alu->instr);
      nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
      nir_def_replace(&alu->def, flush ? flush_fp32(b, x) : x);
      return true;
   }
   if (!flush)
      return false;

   /* Run after algebraic optimization, which can replace arithmetic with
    * min/max or sign operations. Moves, selects, loads and stores are bit
    * preserving and deliberately do not enter this float-result path. */
   b->cursor = nir_after_instr(&alu->instr);
   nir_def *result = flush_fp32(b, &alu->def);
   nir_def_rewrite_uses_after(&alu->def, result);
   return true;
}

static bool
lower_denorm_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                       UNUSED void *data)
{
   bool affected;
   switch (intr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      affected = true;
      break;
   default:
      affected = nir_intrinsic_has_atomic_op(intr) &&
                 (nir_intrinsic_atomic_op(intr) == nir_atomic_op_fadd ||
                  nir_intrinsic_atomic_op(intr) == nir_atomic_op_fmin ||
                  nir_intrinsic_atomic_op(intr) == nir_atomic_op_fmax);
      break;
   }
   if (!affected || intr->def.bit_size != 32)
      return false;

   /* Floating atomic arithmetic must canonicalize its returned value too.
    * Integer atomics and bit-preserving exchange/load remain untouched. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *result = flush_fp32(b, &intr->def);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

static bool
lower_denorm_reduction(nir_builder *b, nir_intrinsic_instr *intr,
                       UNUSED void *data)
{
   if ((intr->intrinsic != nir_intrinsic_reduce &&
        intr->intrinsic != nir_intrinsic_inclusive_scan &&
        intr->intrinsic != nir_intrinsic_exclusive_scan) ||
       intr->def.bit_size != 32 ||
       (nir_intrinsic_reduction_op(intr) != nir_op_fmin &&
        nir_intrinsic_reduction_op(intr) != nir_op_fmax))
      return false;

   /* A scan's first active lane, or a one-element cluster, may bypass every
    * ALU operation. Keep the original reduction's result semantics before
    * generic subgroup lowering turns it into bit-preserving shuffles/selects. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *result = flush_fp32(b, &intr->def);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

bool
agx_nir_flush_subgroup_denorms(nir_shader *shader)
{
   return nir_is_denorm_flush_to_zero(
             shader->info.float_controls_execution_mode, 32) &&
          nir_shader_intrinsics_pass(shader, lower_denorm_reduction,
                                     nir_metadata_control_flow, NULL);
}

bool
agx_nir_flush_denorms(nir_shader *shader)
{
   /* Decide on the original SSA graph, before explicit integer flushes hide
    * the provenance of floating results. Keep this independent of fast-math
    * permissions: removing an unobservable flush preserves even exact math. */
   struct set *flush = _mesa_pointer_set_create(NULL);
   if (nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode,
                                   32)) {
      BITSET_DECLARE(canonical_uniforms, AGX_APPLE9_UNIFORM_COUNT) = {0};
      uint32_t uniform_constants[AGX_APPLE9_UNIFORM_COUNT] = {0};
      nir_function *preamble = nir_shader_get_entrypoint(shader)->preamble;
      if (preamble) {
         /* Extraction replaces SSA edges by uniform transfers. Retain the
          * canonical-result proof across those transfers, including promoted
          * constants. A store is a raw-bit use, so its floating producer may
          * not defer a required flush to a consumer in the main shader. */
         nir_foreach_block(block, preamble->impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_preamble ||
                   nir_intrinsic_preamble_class(intr) !=
                      nir_preamble_class_general ||
                   intr->src[0].ssa->bit_size != 32)
                  continue;
               unsigned base = nir_intrinsic_base(intr);
               for (unsigned c = 0; c < intr->src[0].ssa->num_components; ++c) {
                  nir_scalar value =
                     nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, c));
                  if (base + c < AGX_APPLE9_UNIFORM_COUNT &&
                      nir_scalar_is_const(value))
                     uniform_constants[base + c] = nir_scalar_as_uint(value);
                  if (base + c < AGX_APPLE9_UNIFORM_COUNT &&
                      fp32_result_is_canonical(
                         nir_get_scalar(intr->src[0].ssa, c), 0, NULL))
                     BITSET_SET(canonical_uniforms, base + c);
               }
            }
         }
      }
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   needs_fp32_flush(nir_instr_as_alu(instr), canonical_uniforms,
                                    uniform_constants))
                  _mesa_set_add(flush, nir_instr_as_alu(instr));
            }
         }
      }
      /* Prune only after the original graph's checks have been selected.
       * A downstream check can absorb an upstream check through operations
       * commuting with canonicalization. Remove entries immediately so no
       * later proof can rely on a check already discarded. */
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               struct set_entry *entry = _mesa_set_search(flush, alu);
               if (entry && fp32_flush_is_unobservable(&alu->def, 0, flush))
                  _mesa_set_remove(flush, entry);
            }
         }
      }
   }
   bool progress = nir_shader_alu_pass(shader, lower_denorm,
                                       nir_metadata_control_flow, flush);
   _mesa_set_destroy(flush, NULL);
   if (nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode,
                                   32))
      progress |= nir_shader_intrinsics_pass(shader, lower_denorm_intrinsic,
                                             nir_metadata_control_flow, NULL);
   return progress;
}
