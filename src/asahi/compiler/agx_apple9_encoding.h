/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_ENCODING_H
#define AGX_APPLE9_ENCODING_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Native mode 2 materializes an arbitrary 32-bit value into
 * r0..r63. Shared by ordinary compiler lowering and stage ABI setup.
 * Destination bits 0..3 occupy byte 0's high nibble; bits 4..5 occupy
 * byte 2 bits 6..7. Byte 2 bit 5 is a separate native modifier, not a
 * destination extension; this established scalar form leaves it clear. */
static inline bool
agx_apple9_encode_literal32(uint8_t out[8], unsigned dst, uint32_t value)
{
   /* Six destination bits. Setting bit 60 as a seventh hung the GPU, and
    * native Metal never materializes a long immediate above r63. */
   if (dst >= 64)
      return false;
   const uint8_t bytes[8] = {
      (uint8_t)(((dst & 0xf) << 4) | 0x0c), (uint8_t)(0x80 | (value & 0x7f)),
      (uint8_t)(((dst >> 4) << 6) | 0x02),  (uint8_t)((value >> 24) & 0xfe),
      (uint8_t)((value >> 6) & 0x1e),       (uint8_t)((value >> 9) & 0x0c),
      (uint8_t)((value >> 13) & 0xff),      (uint8_t)((value >> 21) & 0x0f),
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* T8132 entry-table loads: base is a full address in a GPR pair, offset is
 * in eight-byte units, and results remain pending on dependency slot 2.
 * The established widths are one or two consecutive 64-bit pointers. */
static inline bool
agx_apple9_encode_pointer_load(uint8_t out[14], unsigned dst, unsigned base,
                               unsigned offset, unsigned count)
{
   if ((count != 1 && count != 2) || dst >= 64 || dst + 2 * count > 64 ||
       base >= 63 || offset > 255)
      return false;
   const uint8_t bytes[14] = {
      0x67,
      0,
      0x54,
      (uint8_t)(dst << 1),
      (uint8_t)base,
      0,
      0,
      0,
      (uint8_t)(count == 2 ? 0x57 : 0x59),
      0,
      (uint8_t)offset,
      0x40,
      0x26,
      0,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Signed, instruction-start-relative branch used by shader control flow and
 * persistent shader entries. The low bit is reserved for instruction alignment.
 */
static inline bool
agx_apple9_encode_branch(uint8_t out[10], bool any, int64_t displacement)
{
   if (!out || (displacement & 1) || displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47))
      return false;
   uint8_t bytes[10] = {0x0f, (uint8_t)(any ? 0x00 : 0x01), 0x54};
   for (unsigned byte = 0; byte < 6; ++byte)
      bytes[3 + byte] = (uint64_t)displacement >> (8 * byte);
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Transfer one pending word to the main's argument window. A first transfer
 * waits for the table-load group on slot 2; subsequent transfers reuse it.
 * Argument word 2*i is the low half of pointer i, word 2*i+1 its high half. */
static inline bool
agx_apple9_encode_argument_word(uint8_t out[4], unsigned dst, unsigned src,
                                bool wait)
{
   if (dst >= 64 || src >= 64)
      return false;
   const uint8_t bytes[4] = {
      (uint8_t)(((dst & 15) << 4) | 0x0b),
      (uint8_t)((src << 1) & 0x7f),
      (uint8_t)(9 | ((dst >> 4) << 6)),
      (uint8_t)(4 | (wait ? 0x40 : 0)),
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Commit export register `word` to varying slot `slot`, without waiting */
static inline bool
agx_apple9_encode_vary_store(uint8_t out[8], unsigned word, unsigned slot)
{
   if (word >= 128 || slot >= 138)
      return false;
   const uint8_t bytes[8] = {
      0x57,
      0x06,
      0x54,
      (uint8_t)(word << 1),
      (uint8_t)((slot & 7) << 5),
      (uint8_t)(0x40 | (slot >> 3)),
      0,
      0,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Change the varying slot of an encoded VARY_STORE, keeping its wait */
static inline bool
agx_apple9_relocate_vary_store(uint8_t bytes[8], unsigned slot)
{
   if (slot >= 138)
      return false;

   bytes[4] = (bytes[4] & 0x1f) | ((slot & 7) << 5);
   bytes[5] = (bytes[5] & 0xe0) | (slot >> 3);
   return true;
}

/* Open (TILE_ACCESS) or close (TILE_FENCE) tile and raster order domains */
static inline void
agx_apple9_encode_tile(uint8_t out[6], bool begin, unsigned control)
{
   const uint8_t bytes[6] = {
      (uint8_t)(begin ? 0x87 : 0x07), 2, 0x54, (uint8_t)(control & 0xff),
      (uint8_t)(control >> 8),        0,
   };
   memcpy(out, bytes, sizeof(bytes));
}

/* mov r0..r15 = 0..0x7f */
static inline bool
agx_apple9_encode_mov_imm7(uint8_t out[2], unsigned dst, unsigned value)
{
   if (dst >= 16 || value > 0x7f)
      return false;
   out[0] = (uint8_t)((dst << 4) | 0x0c);
   out[1] = (uint8_t)value;
   return true;
}

static inline void
agx_apple9_encode_halt(uint8_t out[4])
{
   const uint8_t bytes[4] = {0x0e, 0, 0, 0};
   memcpy(out, bytes, sizeof(bytes));
}

#endif
