/*
 * Copyright 2026 Youwes09
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX3_MACHINE_H
#define AGX3_MACHINE_H

#include <stdbool.h>
#include <stdint.h>
#include "agx3_forms.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Two independently addressable 16-bit halves per 32-bit GPR. */
#define AGX3_PUBLICATION_COUNT   128
#define AGX3_GPR_COUNT           96
#define AGX3_UNIFORM_COUNT       256
#define AGX3_HALF_REGISTER_COUNT (2 * AGX3_GPR_COUNT)

/* Compact scratch encodings have different address ranges. */
#define AGX3_FILL_WORD_COUNT 1024
#define AGX3_SAVE_WORD_COUNT 2048

enum agx3_evidence {
   AGX3_EVIDENCE_UNRESOLVED = 0,
   AGX3_EVIDENCE_BYTE_DIFF,
   AGX3_EVIDENCE_HARDWARE,
};

enum agx3_register_file {
   AGX3_FILE_NONE = 0,
   AGX3_FILE_GPR = 1 << 0,
   AGX3_FILE_UNIFORM = 1 << 1,
   AGX3_FILE_IMMEDIATE = 1 << 2,
   AGX3_FILE_IMPLICIT = 1 << 3,
};

enum agx3_operand_role {
   AGX3_OPERAND_DEST,
   AGX3_OPERAND_SRC0,
   AGX3_OPERAND_SRC1,
   AGX3_OPERAND_SRC2,
   AGX3_OPERAND_SRC3,
   AGX3_OPERAND_SRC4,
   AGX3_OPERAND_SRC5,
   AGX3_OPERAND_SRC6,
   AGX3_OPERAND_SRC7,
   AGX3_OPERAND_SRC8,
   AGX3_OPERAND_SRC9,
   AGX3_OPERAND_INDEX,
   AGX3_OPERAND_STORE_DATA,
   AGX3_OPERAND_ATOMIC_DATA,
};

enum agx3_width {
   AGX3_WIDTH_16 = 1 << 0,
   AGX3_WIDTH_32 = 1 << 1,
   AGX3_WIDTH_64 = 1 << 2,
};

enum agx3_operand_flag {
   /* The bit packing is sufficiently understood for physical allocation. */
   AGX3_OPERAND_ALLOCATABLE = 1 << 0,

   /* This is a hard encoding restriction, not merely a size preference. */
   AGX3_OPERAND_HARD_LOW = 1 << 1,

   /* Prefer this range for code size, but another encoding may exist. */
   AGX3_OPERAND_COMPACT_PREFERRED = 1 << 2,

   /* Register bits are distributed among source/liveness/cache fields. */
   AGX3_OPERAND_SCATTERED = 1 << 3,

   /* The consumer obtains this value from the immediately preceding op. */
   AGX3_OPERAND_IMPLICIT = 1 << 4,

   /* A separate encoding can select the uniform file in this operand slot. */
   AGX3_OPERAND_UNIFORM_ALTERNATIVE = 1 << 5,

   /* The instruction destroys this input even when its value has later uses. */
   AGX3_OPERAND_CLOBBER = 1 << 6,

   /* Scalar index addresses export storage rather than the ordinary GPR bank. */
   AGX3_OPERAND_PUBLICATION = 1 << 7,
};

struct agx3_operand_constraint {
   enum agx3_operand_role role;
   uint8_t files;
   uint8_t widths;

   /* Inclusive index range in the selected file; 0xff when not applicable. */
   uint8_t min_index;
   uint8_t max_index;
   /* Independent uniform limit for operands admitting more than one file. */
   uint8_t uniform_max_index;

   /* Alignment in 16-bit allocation units. */
   uint8_t alignment_halves;
   uint8_t flags;
   enum agx3_evidence evidence;
};

/*
 * Instructions encode pending-result dependencies in one of four
 * physical layouts. VIR names at most one logical slot for every consumer;
 * the selected machine encoding owns the translation into either a binary
 * slot index or a one-hot physical bit.
 */
enum agx3_dependency_layout {
   AGX3_DEPENDENCY_NONE = 0,
   AGX3_DEPENDENCY_INDEX_45_47,
   AGX3_DEPENDENCY_INDEX_61_63,
   AGX3_DEPENDENCY_MASK_12_17,
   AGX3_DEPENDENCY_MASK_45_47_61_63,
   AGX3_DEPENDENCY_MASK_61_63_77_79,
   AGX3_DEPENDENCY_MASK_61_63_93_95,
};

/*
 * Pending producers grouped by what a completion barrier must order against.
 * Memory results are device, private and threadgroup loads and returning
 * atomics. Publications are scalar exports, consumed only through
 * their readers' dependency fields. Everything else (fixed producers such as
 * system values and mesh control, texture reads, scratch traffic, tile
 * loads) is "other".
 */
enum agx3_completion_class {
   AGX3_COMPLETION_MEMORY = 1 << 0,
   AGX3_COMPLETION_PUBLICATION = 1 << 1,
   AGX3_COMPLETION_OTHER = 1 << 2,
};

#define AGX3_MAX_ENCODING_OPERANDS 11

struct agx3_encoding_info {
   const char *name;
   uint8_t length;
   uint8_t operand_count;
   bool allocator_safe;
   enum agx3_evidence evidence;
   enum agx3_dependency_layout dependency_layout;

   /* A consumer that waits through its dependency field may keep the
    * completed operands live: each register source has its own release
    * flag. Without it, a folded wait must also be the operand's last use. */
   bool dependency_retains_sources;

   /* Implicit producer slot, 1..6, or zero for no fixed assignment. This is
    * scheduler metadata, not an encoded output-tag field. Existing producers
    * with a programmable output tag continue to allocate their slot normally. */
   uint8_t fixed_producer_slot;

   /* Other slots that also complete this fixed producer. They are reserved
    * while it is pending, but are alternative selectors, not extra waits.
    * Ordinary producers still complete only on their own assigned slot. */
   uint8_t fixed_producer_alternates;

   /* The instruction consumes completion of all previously issued, still
    * pending producers, independently of its GPR operands. This is a
    * scheduling requirement, not a claim that the encoding has an input
    * dependency field. The scheduler derives any wait from its live state. */
   bool requires_prior_completion;

   /* Pending producer classes the instruction does not order against, even
    * though it requires prior completion of the others. See
    * agx3_completion_class. */
   uint8_t prior_completion_exempt;

   /* Several native instructions each write part of the destination, so it
    * cannot share a register with a source. */
   bool split_result;

   /* Completion class of an asynchronous result, or zero if synchronous */
   uint8_t completion;

   /* The sources are read until the result completes */
   bool reads_sources;

   /* The instruction waits for export group 0 itself */
   bool waits_exports;

   struct agx3_operand_constraint operands[AGX3_MAX_ENCODING_OPERANDS];
};

enum agx3_occupancy_model {
   AGX3_OCCUPANCY_UNMEASURED,
};

struct agx3_machine {
   uint16_t gpr_count;
   uint16_t half_register_count;

   bool hardware_register_interlocks;
   bool software_waits;
   bool spilling_supported;
   enum agx3_occupancy_model occupancy_model;
};

extern const struct agx3_machine agx3_machine;

const struct agx3_encoding_info *
agx3_encoding_info(enum agx3_encoding encoding);

struct agx_instr;

/* Encoding form of an allocated instruction, or AGX3_ENC_PSEUDO */
enum agx3_encoding agx3_encoding_for(const struct agx_instr *I);

#ifdef __cplusplus
}
#endif

#endif
