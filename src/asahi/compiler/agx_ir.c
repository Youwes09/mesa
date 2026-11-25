/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "asahi/isa/disasm.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

bool
agx_allows_16bit_immediate(agx_instr *I)
{
   return (I->op == AGX_OPCODE_DEVICE_LOAD) ||
          (I->op == AGX_OPCODE_DEVICE_STORE) ||
          (I->op == AGX_OPCODE_STACK_LOAD) ||
          (I->op == AGX_OPCODE_STACK_STORE) ||
          (I->op == AGX_OPCODE_UNIFORM_STORE) || (I->op == AGX_OPCODE_ATOMIC) ||
          (I->op == AGX_OPCODE_PHI) || (I->op == AGX_OPCODE_TEX_STATE_STORE) ||
          (I->op == AGX_OPCODE_SAMPLER_STATE_STORE);
}

unsigned
agx_negate_src_index(agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_IMAD:
      return 2;
   case AGX_OPCODE_IADD:
      return 1;
   default:
      UNREACHABLE("not allowed");
   }
}

/* 128 GPRs, 256 uniform words and 4 KiB of spill slots */
static const struct agx_isa_target agx_apple8_target = {
   .isa = AGX_SHADER_ISA_APPLE8,
   .gpr_halves = 256,
   .uniform_halves = 512,
   .export_halves = 0,
   .memory_halves = 2048,
   .mov_imm_halves = 256,
   .spill_reserved_halves = 8,
   .spill_copy_reg = 2,
   .spill_swap_regs = {4, 6},
   .scoreboard_slots = 2,
   .occupancy_table = true,
   .pack = agx2_pack_binary,
   .disassemble = agx2_disassemble,
};

/* 96 GPRs, 256 uniform words, 128 export words and a 32 KiB private frame */
static const struct agx_isa_target agx_apple9_target = {
   .isa = AGX_SHADER_ISA_APPLE9,
   .gpr_halves = 192,
   .uniform_halves = 512,
   .export_halves = 256,
   .memory_halves = 16384,
   .mov_imm_halves = 128,
   .spill_reserved_halves = 10,
   .spill_copy_reg = 2,
   .spill_swap_regs = {6, 8},
   .spill_index_reg = 4,
   .word_registers = true,
   .word_memory = true,
   .coherent_memory = true,
   .tile_domains = true,
   .varying_exports = true,
   .texture_exports = true,
   .trailing_preamble = true,
   .mask_stack = true,
   .flushes_denorms = true,
   .minimal_subgroups = true,
   .root_uniforms = true,
   .uniform_dests = true,
   .dependency_waits = true,
   .operand_limits = true,
   .avoid_export_targets = true,
   .raw_loads = true,
   .scoreboard_slots = 6,
   .shader_sized_frame = true,
   .occupancy_table = false,
   .pack = agx3_pack_binary,
};

const struct agx_isa_target *
agx_get_isa_target(enum agx_shader_isa isa)
{
   switch (isa) {
   case AGX_SHADER_ISA_DEFAULT:
   case AGX_SHADER_ISA_APPLE8:
      return &agx_apple8_target;
   case AGX_SHADER_ISA_APPLE9:
      return &agx_apple9_target;
   }

   UNREACHABLE("Invalid shader ISA");
}
