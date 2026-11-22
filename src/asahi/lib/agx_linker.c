/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_linker.h"
#include <stddef.h>
#include <stdint.h>
#include "asahi/compiler/agx_apple9_encoding.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "agx_scratch.h"

/*
 * When sample shading is used with a non-monolithic fragment shader, we
 * fast-link a program with the following structure:
 *
 *    Fragment prolog;
 *
 *    for (u16 sample_bit = 1; sample_bit < (1 << # of samples); ++sample_bit) {
 *       API fragment shader;
 *       Fragment epilog;
 *    }
 *
 * This means the prolog runs per-pixel but the fragment shader and epilog run
 * per-sample. To do this, we need to generate the loop on the fly. The
 * following binary sequences form the relevant loop.
 */

static_assert(AGX_ABI_FIN_SAMPLE_MASK == 2, "r1l known");

/* clang-format off */
static const uint8_t sample_loop_header[] = {
   /* mov_imm r0l, 0x0, 0b0 */
   0x62, 0x00, 0x00, 0x00,

   /* mov_imm r1l, 0x0, 0b0 */
   0x62, 0x04, 0x01, 0x00,
};

#define STOP                                                                   \
   /* stop */                                                                  \
   0x88, 0x00,                                                                 \
                                                                               \
   /* trap */                                                                  \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,                             \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,

static const uint8_t stop[] = {STOP};

static const uint8_t sample_loop_footer[] = {
   /* iadd r1l, 0, r1l, lsl 1 */
   0x0e, 0x04, 0x00, 0x20, 0x84, 0x00, 0x00, 0x00,

   /* while_icmp r0l, ult, r1h, 0, 1 */
   0x52, 0x2c, 0x42, 0x00, 0x00, 0x00,

   /* jmp_exec_any */
   0x00, 0xc0, 0x00, 0x00, 0x00, 0x00,

   /* pop_exec r0l, 1 */
   0x52, 0x0e, 0x00, 0x00, 0x00, 0x00,

   STOP
};

/* Offset in sample_loop_footer to the jmp_exec_any's target */
#define SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS (16)

/* Offset of the jmp_exec_any, for calculating the PC offsets */
#define SAMPLE_LOOP_FOOTER_JMP_OFFS (14)

/* Offset in sample_loop_footer to the while_icmp's sample count immediate. Bit
 * position in the byte given by the shift.
 */
#define SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS (11)
#define SAMPLE_LOOP_FOOTER_COUNT_SHIFT (4)
/* clang-format on */

/* Merge the CF binding lists from the prolog to handle cull distance */
static void
link_cf(struct agx_linked_shader *linked, struct agx_shader_part *main,
        struct agx_shader_part *prolog)
{
   memcpy(&linked->cf, &main->info.varyings.fs, sizeof(struct agx_varyings_fs));

   struct agx_varyings_fs *prolog_vary =
      prolog ? &prolog->info.varyings.fs : NULL;

   if (prolog_vary && prolog_vary->nr_bindings) {
      assert(!prolog_vary->reads_z);
      linked->cf.nr_cf = MAX2(linked->cf.nr_cf, prolog_vary->nr_cf);

      assert(linked->cf.nr_bindings + prolog_vary->nr_bindings <=
                ARRAY_SIZE(linked->cf.bindings) &&
             "bounded by # of coeff registers");

      memcpy(linked->cf.bindings + linked->cf.nr_bindings,
             prolog_vary->bindings,
             sizeof(struct agx_cf_binding) * prolog_vary->nr_bindings);

      linked->cf.nr_bindings += prolog_vary->nr_bindings;
   }
}

static void
append(struct util_dynarray *d, const void *data, size_t size)
{
   memcpy(util_dynarray_grow_bytes(d, 1, size), data, size);
}

static void
append_tile(struct util_dynarray *d, bool begin, unsigned control)
{
   uint8_t bytes[6];
   agx_apple9_encode_tile(bytes, begin, control);
   append(d, bytes, sizeof(bytes));
}

static void
append_halt(struct util_dynarray *d)
{
   uint8_t bytes[4];
   agx_apple9_encode_halt(bytes);
   append(d, bytes, sizeof(bytes));
}

/* Sample loop: r1 is the sample bit, r2 a temporary, r0 the killed samples */
static_assert(AGX_ABI_FIN_SAMPLE_MASK == 2, "r1 known");
static_assert(AGX_ABI_FOUT_KILLED_SAMPLES == 0, "r0 known");

/* clang-format off */
static const uint8_t apple9_sample_loop_header[] = {
   /* mov r2 = #0xffff */
   0x2c, 0xff, 0x02, 0x00, 0x1e, 0x0c, 0x07, 0x00,

   /* and r2 = r1, r2 */
   0x2b, 0x83, 0x17, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

   /* pcmp.cond=ult r2, #end: bits 1-6 of the end in byte 3 */
   0x0a, 0x05, 0x22, 0x80, 0x05, 0x02,

   /* push_exec.inv p0 */
   0x0f, 0x05, 0x54, 0x21,

   /* pcmp.bank=1.cond=ieq r0, r0 */
   0x2a, 0x01, 0x23, 0x01, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00,

   /* break.bank=1.scopes=3 */
   0x8f, 0x05, 0x54, 0x03, 0x00, 0x01,

   /* pop_exec */
   0x0f, 0x06, 0x04, 0x01, 0x00, 0x00,
};

#define APPLE9_SAMPLE_LOOP_END_PATCH_OFFS (18 + 3)

/* ishl r1 = r1, #0x1 */
static const uint8_t apple9_sample_loop_shift[] = {
   0x27, 0x00, 0x54, 0x02, 0x02, 0x04, 0x04, 0x00, 0xf0, 0x11, 0x01, 0x00,
};

/* pop_loop */
static const uint8_t apple9_pop_loop[] = {
   0x0f, 0x06, 0x04, 0x02, 0x00, 0x00,
};
/* clang-format on */

/* See agx_linker.h */
void
agx_apple9_link(struct agx_apple9_program *p, bool fragment,
                struct agx_shader_part *main, struct agx_shader_part *prolog,
                struct agx_shader_part *epilog, unsigned nr_samples_shaded,
                const uint8_t *uvs_slots)
{
   struct agx_shader_part *parts[] = {prolog, main, epilog};
   struct util_dynarray d;
   util_dynarray_init(&d, NULL);

   memset(p, 0, sizeof(*p));
   p->profile = main->info.apple9_compute;
   assert(p->profile.abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES &&
          "parts are compiled for the graphics ABI");

   unsigned domain = 0;
   bool reads = false, writes = false, zs_release = false;

   for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
      if (!parts[i])
         continue;

      const struct agx_shader_info *info = &parts[i]->info;
      const struct agx_apple9_compute_profile *pp = &info->apple9_compute;

      assert((parts[i] == main || !pp->resource_binding_count) &&
             "only the main shader uses the buffer table");

      p->nr_gprs = MAX2(p->nr_gprs, info->nr_gprs);
      p->profile.publication_count =
         MAX2(p->profile.publication_count, pp->publication_count);
      p->profile.scratch_size = MAX2(p->profile.scratch_size, pp->scratch_size);
      p->profile.atomic_frame_size =
         MAX2(p->profile.atomic_frame_size, pp->atomic_frame_size);

      /* Image writes and tile coordinates need a wider domain */
      if (info->apple9_tile_domain) {
         assert(!domain || domain == info->apple9_tile_domain ||
                domain == 0x600 || info->apple9_tile_domain == 0x600);
         if (domain != 0x700 && domain != 0xf00)
            domain = info->apple9_tile_domain;
      }

      reads |= info->apple9_tile_reads;
      writes |= info->apple9_tile_writes;
      zs_release |= info->apple9_zs_release;
   }

   /* No sample is killed yet */
   if (fragment) {
      uint8_t bytes[2];
      agx_apple9_encode_mov_imm7(bytes, AGX_ABI_FOUT_KILLED_SAMPLES / 2, 0);
      append(&d, bytes, sizeof(bytes));
   }

   if (fragment && domain) {
      append_tile(&d, true, domain);

      if (reads || writes)
         append_tile(&d, true, 0x800 | (writes ? 0xc : 0x8));
   }

   /* The prolog runs per pixel, the main shader and epilog per sample */
   if (prolog)
      append(&d, prolog->binary, prolog->info.main_size);

   size_t loop_start = 0;
   if (nr_samples_shaded > 1) {
      uint8_t bytes[2];
      agx_apple9_encode_mov_imm7(bytes, AGX_ABI_FIN_SAMPLE_MASK / 2, 1);
      append(&d, bytes, sizeof(bytes));

      loop_start = d.size;
      uint8_t header[sizeof(apple9_sample_loop_header)];
      memcpy(header, apple9_sample_loop_header, sizeof(header));
      header[APPLE9_SAMPLE_LOOP_END_PATCH_OFFS] |= 1u << nr_samples_shaded;
      append(&d, header, sizeof(header));
   } else if (nr_samples_shaded) {
      uint8_t bytes[2];
      agx_apple9_encode_mov_imm7(bytes, AGX_ABI_FIN_SAMPLE_MASK / 2, 1);
      append(&d, bytes, sizeof(bytes));
   }

   for (unsigned i = 1; i < ARRAY_SIZE(parts); ++i) {
      if (parts[i])
         append(&d, parts[i]->binary, parts[i]->info.main_size);
   }

   if (nr_samples_shaded > 1) {
      append(&d, apple9_sample_loop_shift, sizeof(apple9_sample_loop_shift));

      uint8_t jmp[10];
      ASSERTED bool ok =
         agx_apple9_encode_branch(jmp, true, (int64_t)loop_start - d.size);
      assert(ok);
      append(&d, jmp, sizeof(jmp));
      append(&d, apple9_pop_loop, sizeof(apple9_pop_loop));
   }

   /* Commit the vertex outputs to their varying slots */
   for (unsigned i = 0; i < AGX_APPLE9_MAX_UVS_SLOTS; ++i) {
      unsigned word = main->info.apple9_uvs_exports[i];

      if (word) {
         uint8_t bytes[8];
         ASSERTED bool ok = agx_apple9_encode_vary_store(
            bytes, word - 1, uvs_slots ? uvs_slots[i] : i);
         assert(ok && "encodable slot");
         append(&d, bytes, sizeof(bytes));
      }
   }

   if (fragment) {
      if (zs_release)
         append_tile(&d, false, 1);

      if (reads || writes)
         append_tile(&d, false, domain == 0xf00 ? 0x300 : 0x20c);
   }

   append_halt(&d);
   p->profile.preamble_offset = d.size;

   /* The main shader's preamble is last and stops */
   struct agx_shader_part *preambles[] = {prolog, epilog, main};
   bool any = false;

   for (unsigned i = 0; i < ARRAY_SIZE(preambles); ++i) {
      if (!preambles[i])
         continue;

      const struct agx_apple9_compute_profile *pp =
         &preambles[i]->info.apple9_compute;

      if (pp->preamble_size) {
         append(&d, (uint8_t *)preambles[i]->binary + pp->preamble_offset,
                pp->preamble_size);
         any = true;
      }
   }

   if (any && !main->info.apple9_compute.preamble_size)
      append_halt(&d);

   p->profile.preamble_size = any ? d.size - p->profile.preamble_offset : 0;
   p->preamble_offset = p->profile.preamble_offset;
   p->preamble_size = p->profile.preamble_size;
   if (!any)
      p->profile.preamble_offset = p->preamble_offset = 0;

   p->size = d.size;
   p->binary = d.data;
}

void
agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
              bool fragment, struct agx_shader_part *main,
              struct agx_shader_part *prolog, struct agx_shader_part *epilog,
              unsigned nr_samples_shaded, const uint8_t *apple9_uvs_slots)
{
   if (main->info.apple9_compute.abi) {
      struct agx_apple9_program *p = &linked->apple9;
      agx_apple9_link(p, fragment, main, prolog, epilog, nr_samples_shaded,
                      apple9_uvs_slots);

      linked->bo = agx_bo_create(dev, p->size, 0, AGX_BO_EXEC | AGX_BO_LOW_VA,
                                 "Linked executable");
      memcpy(agx_bo_map(linked->bo), p->binary, p->size);
      free(p->binary);
      p->binary = NULL;

      struct agx_shader_part *parts[] = {prolog, main, epilog};
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         if (parts[i]) {
            linked->uses_base_param |= parts[i]->info.uses_base_param;
            linked->uses_txf |= parts[i]->info.uses_txf;
         }
      }

      if (fragment)
         link_cf(linked, main, prolog);

      return;
   }

   size_t size = 0;
   unsigned nr_gprs = 0, scratch_size = 0;
   bool reads_tib = false, writes_sample_mask = false,
        disable_tri_merging = false, tag_write_disable = true;

   if (nr_samples_shaded) {
      size += sizeof(sample_loop_header);

      if (nr_samples_shaded > 1)
         size += sizeof(sample_loop_footer);
      else
         size += sizeof(stop);
   }

   struct agx_shader_part *parts[] = {prolog, main, epilog};

   for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size += part->info.main_size;

      nr_gprs = MAX2(nr_gprs, part->info.nr_gprs);
      scratch_size = MAX2(scratch_size, part->info.scratch_size);
      reads_tib |= part->info.reads_tib;
      writes_sample_mask |= part->info.writes_sample_mask;
      disable_tri_merging |= part->info.disable_tri_merging;
      linked->uses_base_param |= part->info.uses_base_param;
      linked->uses_txf |= part->info.uses_txf;
      tag_write_disable &= part->info.tag_write_disable;
   }

   assert(size > 0 && "must stop");

   linked->bo = agx_bo_create(dev, size, 0, AGX_BO_EXEC | AGX_BO_LOW_VA,
                              "Linked executable");
   uint8_t *linked_map = agx_bo_map(linked->bo);

   size_t offset = 0;

   /* FS prolog happens per-pixel, outside the sample loop */
   if (prolog) {
      size_t sz = prolog->info.main_size;
      memcpy(linked_map + offset, prolog->binary, sz);
      offset += sz;
   }

   if (nr_samples_shaded) {
      memcpy(linked_map + offset, sample_loop_header,
             sizeof(sample_loop_header));
      offset += sizeof(sample_loop_header);
   }

   size_t sample_loop_begin = offset;

   /* Main shader and epilog happen in the sample loop, so start from i=1 */
   for (unsigned i = 1; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size_t sz = part->info.main_size;
      memcpy(linked_map + offset, part->binary + part->info.main_offset, sz);
      offset += sz;
   }

   if (nr_samples_shaded > 1) {
      assert(sample_loop_footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] == 0);

      /* Make a stack copy of the footer so we can efficiently patch it */
      uint8_t footer[sizeof(sample_loop_footer)];
      memcpy(footer, sample_loop_footer, sizeof(footer));

      /* Patch in sample end */
      uint8_t end = (1u << nr_samples_shaded) - 1;
      footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] =
         end << SAMPLE_LOOP_FOOTER_COUNT_SHIFT;

      /* Patch in the branch target */
      int32_t loop_size = offset - sample_loop_begin;
      int32_t branch_offs = -(SAMPLE_LOOP_FOOTER_JMP_OFFS + loop_size);
      int32_t *target = (int32_t *)(footer + SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS);
      *target = branch_offs;

      /* Copy in the patched footer */
      memcpy(linked_map + offset, footer, sizeof(footer));
      offset += sizeof(footer);
   } else if (nr_samples_shaded) {
      /* Just end after the first sample, no need to loop for a single sample */
      memcpy(linked_map + offset, stop, sizeof(stop));
      offset += sizeof(stop);
   }

   assert(offset == size);

   agx_pack(&linked->shader, USC_SHADER, cfg) {
      cfg.code = agx_usc_addr(dev, linked->bo->va->addr);
      cfg.unk_2 = fragment ? 2 : 3;

      if (fragment)
         cfg.loads_varyings = linked->cf.nr_bindings > 0;
   }

   agx_pack(&linked->regs, USC_REGISTERS, cfg) {
      cfg.register_count = nr_gprs;
      cfg.unk_1 = fragment;
      cfg.spill_size = scratch_size ? agx_scratch_get_bucket(scratch_size) : 0;
      cfg.unk_4 = 1;
   }

   if (fragment) {
      agx_pack(&linked->fragment_props, USC_FRAGMENT_PROPERTIES, cfg) {
         cfg.early_z_testing = !writes_sample_mask;
         cfg.unk_2 = true;
         cfg.unk_3 = 0xf;
         cfg.unk_4 = 0x2;
         cfg.unk_5 = 0x0;
      }

      /* If we skip the fragment shader, we can optimize the pass type */
      if (tag_write_disable) {
         reads_tib = false;
         writes_sample_mask = false;
         linked->no_op = true;
      }

      agx_pack(&linked->fragment_control, FRAGMENT_CONTROL, cfg) {
         cfg.tag_write_disable = tag_write_disable;
         cfg.disable_tri_merging = disable_tri_merging;

         if (reads_tib && writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;
         else if (reads_tib)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT;
         else if (writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
         else
            cfg.pass_type = AGX_PASS_TYPE_OPAQUE;
      }

      link_cf(linked, main, prolog);

      agx_pack(&linked->osel, OUTPUT_SELECT, cfg) {
         cfg.varyings = linked->cf.nr_bindings > 0;
         cfg.frag_coord_z = linked->cf.reads_z;
      }
   }
}
