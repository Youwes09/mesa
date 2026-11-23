/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"

/* Apple9 coverage writes can revive killed samples and do not mask tile stores
 * or depth/stencil exports, so track killed samples and apply them to all.
 */
static nir_def *
mask_live(nir_builder *b, nir_def *mask, nir_variable *killed)
{
   nir_def *dead = nir_u2uN(b, nir_load_var(b, killed), mask->bit_size);
   return nir_iand(b, mask, nir_inot(b, dead));
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_variable *killed = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_sample_mask_agx: {
      nir_def *target = intr->src[0].ssa;
      nir_def *live = mask_live(b, intr->src[1].ssa, killed);
      nir_src_rewrite(&intr->src[1], live);

      nir_def *dead =
         nir_iand(b, nir_u2u16(b, target), nir_inot(b, nir_u2u16(b, live)));
      nir_store_var(b, killed, nir_ior(b, nir_load_var(b, killed), dead), 1);
      return true;
   }

   case nir_intrinsic_store_local_pixel_agx:
      nir_src_rewrite(&intr->src[1], mask_live(b, intr->src[1].ssa, killed));
      return true;

   case nir_intrinsic_store_zs_agx:
      nir_src_rewrite(&intr->src[0], mask_live(b, intr->src[0].ssa, killed));
      return true;

   case nir_intrinsic_load_sample_mask: {
      nir_def *in = nir_load_sample_mask_in(b);
      nir_def_replace(
         &intr->def, nir_u2uN(b, mask_live(b, in, killed), intr->def.bit_size));
      return true;
   }

   case nir_intrinsic_load_helper_invocation:
      nir_def_replace(&intr->def,
                      nir_ieq_imm(b, nir_load_sample_mask_in(b), 0));
      return true;

   default:
      return false;
   }
}

bool
agx_nir_lower_apple9_coverage(nir_shader *nir, bool load_killed,
                              bool export_killed)
{
   assert(nir->info.stage == MESA_SHADER_FRAGMENT);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   nir_variable *killed =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "killed samples");

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(
      &b, killed,
      load_killed
         ? nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FOUT_KILLED_SAMPLES)
         : nir_imm_intN_t(&b, 0, 16),
      1);

   nir_shader_intrinsics_pass(nir, lower, nir_metadata_none, killed);

   if (export_killed) {
      b.cursor = nir_after_impl(impl);
      nir_export_agx(&b, nir_load_var(&b, killed),
                     .base = AGX_ABI_FOUT_KILLED_SAMPLES);
   }

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   return true;
}
