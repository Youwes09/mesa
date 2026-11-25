/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX3_EMIT_H
#define AGX3_EMIT_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "agx3_pack.h"

/* Instructions of the drivers' launch code and the linker. The helpers
 * leave out untouched when the values do not fit.
 */

/* 32-bit register r0-r63 */
static inline struct agx3_operand
agx3_reg32(unsigned reg)
{
   return (struct agx3_operand){
      .kind = AGX3_KIND_REG,
      .value = reg * 2,
      .size32 = true,
   };
}

static inline bool
agx3_emit(uint8_t *out, agx3_pack_fn pack, struct agx3_fields f)
{
   uint8_t bytes[AGX3_MAX_LENGTH];
   unsigned length = pack(bytes, &f);

   if (length)
      memcpy(out, bytes, length);

   return length != 0;
}

/* Materializes an arbitrary 32-bit value into r0..r63. A seventh destination
 * bit hung the GPU.
 */
static inline bool
agx3_emit_literal32(uint8_t out[8], unsigned dst, uint32_t value)
{
   return agx3_emit(out, agx3_pack_mov_imm32,
                    (struct agx3_fields){
                       .dest = agx3_encode_dst(agx3_reg32(dst)),
                       .value = value,
                    });
}

/* Entry-table loads: base is a full address in a GPR pair, offset is in
 * eight-byte units, and results remain pending on dependency slot 2.
 * The established widths are one or two consecutive 64-bit pointers.
 */
static inline bool
agx3_emit_pointer_load(uint8_t out[14], unsigned dst, unsigned base,
                       unsigned offset, unsigned count)
{
   if ((count != 1 && count != 2) || dst + 2 * count > 64 || base >= 63 ||
       offset > 255)
      return false;

   struct agx3_operand pair = agx3_reg32(base);
   pair.size32 = false;
   pair.size64 = true;

   return agx3_emit(
      out, agx3_pack_device_load,
      (struct agx3_fields){
         .dest = agx3_encode_reg(agx3_reg32(dst)),
         .src = {agx3_encode_membase(pair),
                 agx3_encode_memindex(
                    (struct agx3_operand){.kind = AGX3_KIND_NONE})},
         .format = count == 2 ? AGX3_MEMFORMAT_I32X4 : AGX3_MEMFORMAT_I32X2,
         .next = 1,
         .offset = offset * 8,
         .shift = 3,
         .slot = AGX3_SLOT_SLOT2,
      });
}

/* Signed, instruction-start-relative branch used by shader control flow and
 * persistent shader entries. The low bit is reserved for instruction alignment.
 */
static inline bool
agx3_emit_branch(uint8_t out[10], bool any, int64_t displacement)
{
   if (!out || (displacement & 1))
      return false;

   return agx3_emit(out, any ? agx3_pack_jmp_exec_any : agx3_pack_jmp_exec_none,
                    (struct agx3_fields){.offset = displacement});
}

/* Transfer one pending word to the main's argument window. A first transfer
 * waits for the table-load group on slot 2; subsequent transfers reuse it.
 * Argument word 2*i is the low half of pointer i, word 2*i+1 its high half.
 */
static inline bool
agx3_emit_argument_word(uint8_t out[4], unsigned dst, unsigned src, bool wait)
{
   return agx3_emit(out, agx3_pack_mov_launch,
                    (struct agx3_fields){
                       .dest = agx3_encode_dst(agx3_reg32(dst)),
                       .src = {agx3_encode_reg(agx3_reg32(src))},
                       .wait = wait,
                    });
}

/* Commit export register `word` to varying slot `slot`, without waiting */
static inline bool
agx3_emit_vary_store(uint8_t out[8], unsigned word, unsigned slot)
{
   if (slot >= 138)
      return false;

   struct agx3_operand value = {
      .kind = AGX3_KIND_EXPORT,
      .value = word * 2,
      .size32 = true,
   };

   return agx3_emit(out, agx3_pack_st_vary,
                    (struct agx3_fields){
                       .src = {agx3_encode_export(value)},
                       .slot = slot,
                    });
}

/* Change the varying slot of an encoded st_vary, keeping its wait. The slot
 * is at bits 37-44.
 */
static inline bool
agx3_relocate_vary_store(uint8_t bytes[8], unsigned slot)
{
   if (slot >= 138)
      return false;

   bytes[4] = (bytes[4] & 0x1f) | ((slot & 7) << 5);
   bytes[5] = (bytes[5] & 0xe0) | (slot >> 3);
   return true;
}

/* Open (begin_tile) or close (end_tile) tile and raster order domains */
static inline void
agx3_emit_tile(uint8_t out[6], bool begin, unsigned control)
{
   agx3_emit(out, begin ? agx3_pack_begin_tile : agx3_pack_end_tile,
             (struct agx3_fields){.control = control & 0xffff});
}

/* mov r0..r15 = 0..0x7f */
static inline bool
agx3_emit_mov_imm7(uint8_t out[2], unsigned dst, unsigned value)
{
   return agx3_emit(out, agx3_pack_mov_imm,
                    (struct agx3_fields){
                       .dest = agx3_encode_dst(agx3_reg32(dst)),
                       .value = value,
                    });
}

static inline void
agx3_emit_halt(uint8_t out[4])
{
   agx3_emit(out, agx3_pack_stop, (struct agx3_fields){0});
}

#endif
