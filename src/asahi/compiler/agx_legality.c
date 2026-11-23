/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compiler.h"

/* Apple9 operand legality, for the validator and passes rewriting operands.
 * Register limits are agx_set_reg_constraints'. Apple8 is always legal here.
 */

/* A GPR value or register */
static bool
is_reg(agx_index idx)
{
   return (idx.type == AGX_INDEX_NORMAL || idx.type == AGX_INDEX_REGISTER) &&
          idx.reg_class == RA_GPR;
}

static bool
is_export(agx_index idx)
{
   return (idx.type == AGX_INDEX_NORMAL || idx.type == AGX_INDEX_REGISTER) &&
          idx.reg_class == RA_EXPORT && idx.size == AGX_SIZE_32;
}

static bool
reg32(agx_index idx)
{
   return is_reg(idx) && idx.size == AGX_SIZE_32 && agx_channels(idx) == 1;
}

static bool
reg16(agx_index idx)
{
   return is_reg(idx) && idx.size == AGX_SIZE_16 && agx_channels(idx) == 1;
}

/* A 32-bit uniform word */
static bool
uni32(agx_index idx)
{
   return idx.type == AGX_INDEX_UNIFORM && idx.size == AGX_SIZE_32 &&
          !(idx.value & 1) && idx.value < 512;
}

static bool
is_imm(agx_index idx)
{
   return idx.type == AGX_INDEX_IMMEDIATE;
}

static bool
imm8(agx_index idx)
{
   return is_imm(idx) && idx.value <= UINT8_MAX && !idx.abs;
}

static bool
plain(agx_index idx)
{
   return !idx.abs && !idx.neg;
}

/* An integer ALU source: a register, a uniform word or an 8-bit constant */
static bool
int_src(agx_index idx, bool allow_imm)
{
   return plain(idx) && (reg32(idx) || uni32(idx) || (allow_imm && imm8(idx)));
}

static bool
convert_mode_ok(const agx_instr *I)
{
   if (!is_imm(I->src[0]))
      return false;

   switch (I->src[0].value) {
   case AGX_CONVERT_U32_TO_F:
   case AGX_CONVERT_S32_TO_F:
      return I->round == AGX_ROUND_RTE;
   case AGX_CONVERT_F_TO_U32:
   case AGX_CONVERT_F_TO_S32:
      return I->round == AGX_ROUND_RTZ;
   default:
      return false;
   }
}

/* Device memory accesses scale the index by at most 16 bytes */
static unsigned
memory_shift(const agx_instr *I)
{
   unsigned element = I->format == AGX_FORMAT_I8    ? 0
                      : I->format == AGX_FORMAT_I16 ? 1
                                                    : 2;
   return element + I->shift;
}

static bool
memory_format_ok(const agx_instr *I)
{
   if (I->format != AGX_FORMAT_I8 && I->format != AGX_FORMAT_I16 &&
       I->format != AGX_FORMAT_I32)
      return false;

   if (!I->mask || !util_is_power_of_two_nonzero(I->mask + 1))
      return false;

   return (I->mask == 1 || I->format == AGX_FORMAT_I32) &&
          memory_shift(I) <= 4 && !I->coherent;
}

/* A 64-bit base: a uniform pair, as a binding, or a register pair */
static bool
memory_base_ok(agx_index base, unsigned max_binding)
{
   if (base.size != AGX_SIZE_64 || !plain(base))
      return false;

   if (base.type == AGX_INDEX_UNIFORM)
      return !(base.value & 3) && (base.value >> 2) < max_binding;

   return is_reg(base) && agx_channels(base) == 1;
}

static bool
vec32(agx_index idx, unsigned channels)
{
   return is_reg(idx) && idx.size == AGX_SIZE_32 &&
          agx_channels(idx) == channels && plain(idx);
}

static bool
float_src_ok(agx_index idx, bool imm_wide, bool allow_half)
{
   if (is_imm(idx))
      return agx_apple9_float_immediate_ok(idx, imm_wide);

   return reg32(idx) || uni32(idx) || (allow_half && reg16(idx));
}

/* min/max as a select of its own compared operands, GTN/LTN for floats */
static bool
is_minmax(const agx_instr *I)
{
   bool same =
      agx_is_equiv(I->src[0], I->src[2]) && agx_is_equiv(I->src[1], I->src[3]);
   bool swapped =
      agx_is_equiv(I->src[0], I->src[3]) && agx_is_equiv(I->src[1], I->src[2]);

   if (!same && !swapped)
      return false;

   if (I->op == AGX_OPCODE_ICMPSEL) {
      if (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ)
         return false;

      for (unsigned s = 0; s < 4; ++s) {
         if (!plain(I->src[s]) || !(reg32(I->src[s]) || uni32(I->src[s])))
            return false;
      }

      return true;
   } else {
      if (I->fcond != AGX_FCOND_GTN && I->fcond != AGX_FCOND_LTN)
         return false;

      agx_index a = I->src[0], b = I->src[1];
      if (!agx_is_equiv_with_modifiers(a, I->src[same ? 2 : 3]) ||
          !agx_is_equiv_with_modifiers(b, I->src[same ? 3 : 2]) || !plain(a) ||
          b.abs)
         return false;

      /* At most one constant, which cannot be negated */
      if (is_imm(a) && is_imm(b))
         return false;

      agx_index k = is_imm(a) ? a : b;
      agx_index x = is_imm(a) ? b : a;
      if (is_imm(k) && (k.neg || !agx_apple9_float_immediate_ok(k, true)))
         return false;

      return (reg32(x) || uni32(x)) && (is_imm(k) || reg32(k) || uni32(k));
   }
}

static bool
select_ok(const agx_instr *I)
{
   bool floating = I->op == AGX_OPCODE_FCMPSEL;

   if (I->dest[0].size != AGX_SIZE_32)
      return false;

   if (is_minmax(I))
      return true;

   bool swap = false;
   if (floating) {
      switch (I->fcond) {
      case AGX_FCOND_EQ:
      case AGX_FCOND_LT:
      case AGX_FCOND_GT:
      case AGX_FCOND_GE:
         break;
      case AGX_FCOND_LE:
         swap = true;
         break;
      default:
         return false;
      }
   } else {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
      case AGX_ICOND_ULT:
      case AGX_ICOND_UGT:
      case AGX_ICOND_SLT:
      case AGX_ICOND_SGT:
         break;
      default:
         return false;
      }
   }

   agx_index a = I->src[swap ? 1 : 0], b = I->src[swap ? 0 : 1];

   /* Operand A cannot be a constant or negated */
   if (is_imm(a) || a.neg || !(reg32(a) || uni32(a)))
      return false;

   if (is_imm(b)) {
      if (b.neg || b.abs ||
          (floating ? !agx_apple9_float_immediate_ok(b, true)
                    : b.value > UINT8_MAX))
         return false;
   } else if (!(reg32(b) || uni32(b))) {
      return false;
   }

   for (unsigned s = 2; s < 4; ++s) {
      agx_index x = I->src[s];
      if (is_imm(x) ? (x.value > UINT8_MAX || !plain(x))
                    : !(reg32(x) || uni32(x)))
         return false;
   }

   /* Only float selects have modifiers */
   if (!floating) {
      for (unsigned s = 0; s < 4; ++s) {
         if (!plain(I->src[s]))
            return false;
      }
   }

   return true;
}

static bool
special_ok(const agx_instr *I)
{
   agx_index src = I->src[0];
   bool rounding = I->op == AGX_OPCODE_FLOOR || I->op == AGX_OPCODE_CEIL ||
                   I->op == AGX_OPCODE_TRUNC || I->op == AGX_OPCODE_ROUNDEVEN;

   /* A uniform source only works with these */
   bool uniform_ok = rounding || I->op == AGX_OPCODE_RSQRT ||
                     I->op == AGX_OPCODE_EXP2 || I->op == AGX_OPCODE_LOG2;

   return reg32(I->dest[0]) && (reg32(src) || (uniform_ok && uni32(src))) &&
          !(I->saturate && rounding);
}

static bool
fadd_fmul_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1];

   /* FP32/FP16 conversions are fmov with different sizes */
   if (I->op == AGX_OPCODE_FADD && agx_is_equiv(b, agx_negzero()) && plain(b) &&
       I->dest[0].size != a.size) {
      return plain(a) && !I->saturate &&
             ((reg16(I->dest[0]) && reg32(a)) ||
              (reg32(I->dest[0]) && reg16(a)));
   }

   /* Products into export registers */
   if (I->op == AGX_OPCODE_FMUL && is_export(I->dest[0])) {
      return reg32(a) && reg32(b) && plain(a) && !b.abs && !I->saturate;
   }

   if (!reg32(I->dest[0]))
      return false;

   /* One constant and a register. Modified float moves become bitwise ops */
   if (is_imm(a) || is_imm(b)) {
      agx_index k = is_imm(a) ? a : b;
      agx_index x = is_imm(a) ? b : a;
      bool fmov = I->op == AGX_OPCODE_FADD && agx_is_equiv(k, agx_negzero());

      return !is_imm(x) && reg32(x) && (!x.abs || fmov) &&
             agx_apple9_float_immediate_ok(k, true);
   }

   return float_src_ok(a, false, true) && float_src_ok(b, false, true);
}

static bool
ffma_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1], c = I->src[2];

   /* A constant multiplier goes in operand B */
   if (is_imm(a)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   return reg32(I->dest[0]) && !is_imm(a) && (reg32(a) || uni32(a)) &&
          float_src_ok(b, false, true) && !(is_imm(b) && b.neg) &&
          float_src_ok(c, true, true) && !(is_imm(c) && c.neg);
}

static bool
iadd_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1];

   if (!reg32(I->dest[0]) || I->saturate || I->shift || (a.neg && b.neg))
      return false;

   for (unsigned s = 0; s < 2; ++s) {
      agx_index x = I->src[s];

      /* abs zero-extends a 16-bit register; nothing else has modifiers */
      if (reg16(x))
         continue;

      if (x.abs || !(reg32(x) || uni32(x) || is_imm(x)) ||
          (is_imm(x) && x.value > UINT8_MAX))
         return false;
   }

   return true;
}

static bool
imad_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1], c = I->src[2];

   if (I->shift)
      return false;

   /* 32x32 to 64-bit multiply */
   if (I->dest[0].size == AGX_SIZE_64) {
      return is_reg(I->dest[0]) && reg32(a) && reg32(b) && a.abs == b.abs &&
             !a.neg && !b.neg && agx_is_equiv(c, agx_zero());
   }

   if (!reg32(I->dest[0]))
      return false;

   /* Operand A cannot be a constant, so one moves over to B */
   if (is_imm(a)) {
      agx_index t = a;
      a = b;
      b = t;
   }

   if (is_imm(a))
      return false;

   agx_index srcs[3] = {a, b, c};
   for (unsigned s = 0; s < 3; ++s) {
      agx_index x = srcs[s];

      if (x.neg)
         return false;

      /* Only operand A sign-extends a 16-bit register */
      if (reg16(x)) {
         if (s > 0 && !x.abs)
            return false;
      } else if (x.abs || !(reg32(x) || uni32(x) || imm8(x))) {
         return false;
      }
   }

   return true;
}

static bool
shift_ok(const agx_instr *I)
{
   bool arithmetic = I->op == AGX_OPCODE_ASR;
   agx_index src = arithmetic ? I->src[0] : I->src[1];
   agx_index amount = arithmetic ? I->src[1] : I->src[2];

   if (!reg32(I->dest[0]) || !int_src(src, false))
      return false;

   if (!arithmetic) {
      /* ishl is bfi into zero, ushr is bfeil from zero, maybe with bfi_mask */
      if (!agx_is_equiv(I->src[0], agx_zero()))
         return false;

      if (I->bfi_mask &&
          (I->op == AGX_OPCODE_BFI || !is_imm(amount) || I->bfi_mask >= 64))
         return false;
   }

   return is_imm(amount) ? amount.value < 32 : int_src(amount, false);
}

static bool
logic_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1];

   switch (I->truth_table) {
   case 0x8: /* and */
   case 0xe: /* or */
   case 0x6: /* xor */
      break;
   case 0xa: /* a */
      b = a;
      break;
   case 0xc: /* b */
      a = b;
      break;
   default:
      return false;
   }

   if (is_export(I->dest[0]))
      return reg32(a) && reg32(b) && plain(a) && plain(b);

   return reg32(I->dest[0]) && !(is_imm(a) && is_imm(b)) && int_src(a, true) &&
          int_src(b, true);
}

static bool
device_memory_ok(const agx_instr *I)
{
   bool load = I->op == AGX_OPCODE_DEVICE_LOAD;
   agx_index data = load ? I->dest[0] : I->src[0];
   agx_index base = load ? I->src[0] : I->src[1];
   agx_index index = load ? I->src[1] : I->src[2];

   if (!memory_format_ok(I) || !memory_base_ok(base, 256) ||
       !vec32(data, util_bitcount(I->mask)))
      return false;

   if (is_imm(index)) {
      /* A constant index is a byte offset from a register pair */
      return load && is_reg(base) &&
             ((int64_t)index.value << memory_shift(I)) + I->byte_offset <=
                INT16_MAX;
   }

   return reg32(index) && plain(index);
}

static bool
atomic_ok(const agx_instr *I)
{
   bool cmpxchg = I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG;

   return (agx_is_null(I->dest[0]) || reg32(I->dest[0])) &&
          vec32(I->src[0], cmpxchg ? 2 : 1) && memory_base_ok(I->src[1], 128) &&
          reg32(I->src[2]) && plain(I->src[2]);
}

/* Threadgroup memory: an immediate root and a 16-bit word index */
static bool
local_ok(const agx_instr *I)
{
   bool load = I->op == AGX_OPCODE_LOCAL_LOAD;
   bool atomic = I->op == AGX_OPCODE_LOCAL_ATOMIC;
   agx_index base = load ? I->src[0] : I->src[1];
   agx_index index = load ? I->src[1] : I->src[2];

   if (!is_imm(base) || base.value >= 64 || !(reg16(index) || reg32(index)) ||
       !plain(index))
      return false;

   if (atomic) {
      bool cmpxchg = I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG;
      return reg32(I->dest[0]) && vec32(I->src[0], cmpxchg ? 2 : 1);
   }

   return I->format == AGX_FORMAT_I32 && I->mask == 1 &&
          reg32(load ? I->dest[0] : I->src[0]);
}

/* A constant byte offset, or a register holding a word index */
static bool
stack_ok(const agx_instr *I)
{
   bool load = I->op == AGX_OPCODE_STACK_LOAD;
   agx_index offset = I->src[load ? 0 : 1];
   unsigned words = load ? 1024 : 2048;

   if (I->format != AGX_FORMAT_I32 || I->mask != 1 ||
       !reg32(load ? I->dest[0] : I->src[0]))
      return false;

   if (is_imm(offset))
      return !(offset.value & 3) && offset.value / 4 < words;
   else
      return reg32(offset) && plain(offset);
}

static bool
tile_ok(const agx_instr *I)
{
   bool load = I->op == AGX_OPCODE_LD_TILE;
   agx_index mask = I->src[load ? 0 : 1];

   if (I->format != AGX_FORMAT_I32 || I->mask != 1 || I->pixel_offset >= 32 ||
       !reg32(load ? I->dest[0] : I->src[0]))
      return false;

   /* With explicit coordinates the sample mask is a constant */
   if (I->explicit_coords) {
      return load && is_imm(mask) && mask.value >= 1 && mask.value <= 15 &&
             reg32(I->src[1]);
   }

   /* A register mask is a 16-bit half, or the low half of a word */
   return is_imm(mask) ? mask.value == 1 : (reg16(mask) || reg32(mask));
}

static bool
pred_compare_ok(const agx_instr *I)
{
   agx_index a = I->src[0], b = I->src[1];

   if (I->pred >= 6 || !plain(a) || !plain(b) || !(reg32(a) || uni32(a)))
      return false;

   if (I->op == AGX_OPCODE_ICMP_PRED) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
      case AGX_ICOND_ULT:
      case AGX_ICOND_UGT:
      case AGX_ICOND_SLT:
      case AGX_ICOND_SGT:
         break;
      default:
         return false;
      }

      return is_imm(b) ? b.value <= UINT8_MAX : (reg32(b) || uni32(b));
   } else {
      switch (I->fcond) {
      case AGX_FCOND_EQ:
      case AGX_FCOND_LT:
      case AGX_FCOND_GT:
      case AGX_FCOND_GE:
      case AGX_FCOND_LE:
         break;
      default:
         return false;
      }

      return is_imm(b) ? agx_apple9_float_immediate_ok(b, true)
                       : (reg32(b) || uni32(b));
   }
}

static bool apple9_instr_is_legal(const agx_instr *I);

/* Texture table index: a constant or a 32-bit register */
static bool
table_index_ok(agx_index idx, unsigned limit)
{
   return (is_imm(idx) && idx.value < limit) || (reg32(idx) && plain(idx));
}

/* Constant offsets, and the texture in a register if the sampler is */
static bool
texture_ok(const agx_instr *I)
{
   agx_index params = I->src[0];
   unsigned comps = I->gather != AGX_GATHER_NONE ? 4 : util_bitcount(I->mask);

   return agx_apple9_texture_encodable(I) && is_export(params) &&
          agx_channels(params) == agx_apple9_texture_params(I) &&
          agx_is_null(I->src[1]) && agx_is_null(I->src[2]) &&
          table_index_ok(I->src[3], 256) && table_index_ok(I->src[4], 32) &&
          (is_imm(I->src[4]) || !is_imm(I->src[3])) &&
          (I->offset ? is_imm(I->src[5]) : agx_is_null(I->src[5])) &&
          !I->sparse && I->dest[0].size == AGX_SIZE_32 && is_reg(I->dest[0]) &&
          agx_channels(I->dest[0]) == comps;
}

/* Compares to 0/1 are lowered to selects */
static bool
cmp_ok(const agx_instr *I)
{
   agx_index srcs[4] = {I->src[0], I->src[1], agx_immediate(1),
                        agx_immediate(0)};
   agx_instr sel = *I;

   sel.op = I->op == AGX_OPCODE_ICMP ? AGX_OPCODE_ICMPSEL : AGX_OPCODE_FCMPSEL;
   sel.src = srcs;
   sel.nr_srcs = 4;
   return apple9_instr_is_legal(&sel);
}

static bool
apple9_instr_is_legal(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_ICMP:
   case AGX_OPCODE_FCMP:
      return cmp_ok(I);

   case AGX_OPCODE_MOV:
      /* Constants use mov_imm, export registers are written from registers */
      if (is_export(I->dest[0]))
         return reg32(I->src[0]) && plain(I->src[0]);

      return !is_imm(I->src[0]);

   case AGX_OPCODE_COLLECT:
      /* A collect into export registers becomes export moves of registers */
      if (I->dest[0].reg_class == RA_EXPORT) {
         for (unsigned s = 0; s < I->nr_srcs; ++s) {
            if (!agx_is_null(I->src[s]) && I->src[s].type != AGX_INDEX_UNDEF &&
                !(reg32(I->src[s]) && plain(I->src[s])))
               return false;
         }

         return true;
      }

      FALLTHROUGH;

   case AGX_OPCODE_PHI:
      /* Parallel copies write wide constants as an add of two 8-bit ones */
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (is_imm(I->src[s]) && I->src[s].value > UINT8_MAX)
            return false;
      }

      return true;

   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
      return fadd_fmul_ok(I);

   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL:
   case AGX_OPCODE_HFMA:
      /* FP16 in the low halves of registers, only B of an add negated */
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         agx_index x = I->src[s];
         if (!reg32(x) || x.abs ||
             (x.neg && !(I->op == AGX_OPCODE_HADD && s == 1)))
            return false;
      }

      return reg32(I->dest[0]) && !I->saturate;

   case AGX_OPCODE_F32_TO_F16:
   case AGX_OPCODE_F16_TO_F32:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]) &&
             !I->saturate;

   case AGX_OPCODE_FMUL_PERSPECTIVE:
      return reg32(I->dest[0]) && reg32(I->src[0]) && reg32(I->src[1]) &&
             plain(I->src[0]) && plain(I->src[1]) && !I->saturate &&
             I->component < AGX_APPLE9_NUM_COEFFICIENTS;

   case AGX_OPCODE_FFMA:
      return ffma_ok(I);

   case AGX_OPCODE_IADD:
      return iadd_ok(I);

   case AGX_OPCODE_IMAD:
      return imad_ok(I);

   case AGX_OPCODE_ASR:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
      return shift_ok(I);

   case AGX_OPCODE_CONVERT:
      return convert_mode_ok(I) && reg32(I->dest[0]) &&
             int_src(I->src[1], false);

   case AGX_OPCODE_BITREV:
   case AGX_OPCODE_POPCOUNT:
   case AGX_OPCODE_FFS:
      return reg32(I->dest[0]) && int_src(I->src[0], false);

   case AGX_OPCODE_RCP:
   case AGX_OPCODE_RSQRT:
   case AGX_OPCODE_SRSQRT:
   case AGX_OPCODE_SIN_FACTOR:
   case AGX_OPCODE_EXP2:
   case AGX_OPCODE_LOG2:
   case AGX_OPCODE_FLOOR:
   case AGX_OPCODE_CEIL:
   case AGX_OPCODE_TRUNC:
   case AGX_OPCODE_ROUNDEVEN:
      return special_ok(I);

   case AGX_OPCODE_DFDX:
   case AGX_OPCODE_DFDY:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]);

   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_FCMPSEL:
      return select_ok(I);

   case AGX_OPCODE_HMIN:
   case AGX_OPCODE_HMAX:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]) &&
             reg32(I->src[1]) && plain(I->src[1]);

   case AGX_OPCODE_HCMPSEL:
      /* Registers only, without modifiers */
      for (unsigned s = 0; s < 4; ++s) {
         if (!reg32(I->src[s]) || !plain(I->src[s]))
            return false;
      }

      return reg32(I->dest[0]) &&
             (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
              I->fcond == AGX_FCOND_GT || I->fcond == AGX_FCOND_GE ||
              I->fcond == AGX_FCOND_LE);

   case AGX_OPCODE_HCMP:
      /* Lowered to an FP16 select of registers */
      return reg32(I->src[0]) && plain(I->src[0]) && reg32(I->src[1]) &&
             plain(I->src[1]);

   case AGX_OPCODE_BITOP:
      return logic_ok(I);

   case AGX_OPCODE_MOV_IMM:
      return reg32(I->dest[0]) && I->imm <= UINT32_MAX;

   case AGX_OPCODE_GET_SR:
   case AGX_OPCODE_GET_SR_COVERAGE:
   case AGX_OPCODE_GET_SR_BARRIER:
      return reg32(I->dest[0]) && agx_apple9_sr_supported(I->sr);

   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_DEVICE_STORE:
      return device_memory_ok(I);

   case AGX_OPCODE_ATOMIC:
      return atomic_ok(I);

   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
   case AGX_OPCODE_LOCAL_ATOMIC:
      return local_ok(I);

   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
      return stack_ok(I);

   case AGX_OPCODE_UNIFORM_STORE:
      return reg32(I->src[0]) && plain(I->src[0]) && is_imm(I->src[1]) &&
             !(I->src[1].value & 1) && I->src[1].value < 512 && I->mask == 1;

   case AGX_OPCODE_SIMD_PREFIX:
      return I->simd_op == AGX_SIMD_OP_IADD && reg32(I->dest[0]) &&
             reg32(I->src[0]) && plain(I->src[0]);

   case AGX_OPCODE_SIMD_REDUCE:
      return I->simd_op == AGX_SIMD_OP_UMAX && plain(I->src[0]) &&
             ((reg32(I->dest[0]) && reg32(I->src[0])) ||
              (reg16(I->dest[0]) && reg16(I->src[0])));

   case AGX_OPCODE_SHUFFLE:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]) &&
             (reg16(I->src[1]) || reg32(I->src[1])) && plain(I->src[1]);

   case AGX_OPCODE_BALLOT:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]);

   case AGX_OPCODE_ICMP_PRED:
   case AGX_OPCODE_FCMP_PRED:
      return pred_compare_ok(I);

   case AGX_OPCODE_HCMP_PRED:
      /* FP16 compares take registers only */
      return pred_compare_ok(I) && reg32(I->src[0]) && reg32(I->src[1]);

   case AGX_OPCODE_IF_PRED:
      return I->pred < 8;

   case AGX_OPCODE_WHILE_PRED:
      return I->pred < 6;

   case AGX_OPCODE_BREAK_PRED:
      return I->pred >= 1 && I->pred < 6 && I->nest >= 2 &&
             I->nest <= UINT8_MAX;

   case AGX_OPCODE_POP_EXEC:
      return I->nest == 1;

   case AGX_OPCODE_PACK_HALF:
      return reg32(I->dest[0]) && reg32(I->src[0]) && plain(I->src[0]) &&
             reg32(I->src[1]) && plain(I->src[1]);

   case AGX_OPCODE_UNPACK_NORM:
      return is_reg(I->dest[0]) && I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 2 && reg32(I->src[0]) &&
             plain(I->src[0]);

   case AGX_OPCODE_PACK_UNORM_2X16:
   case AGX_OPCODE_PACK_UNORM_4X8:
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (!reg32(I->src[s]) || !plain(I->src[s]))
            return false;
      }

      return reg32(I->dest[0]);

   case AGX_OPCODE_CUBE:
      return is_reg(I->dest[0]) && I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == (I->imm == 0 ? 2 : 1) &&
             reg32(I->src[0]) && plain(I->src[0]) && reg32(I->src[1]) &&
             plain(I->src[1]) && reg32(I->src[2]) && plain(I->src[2]) &&
             I->imm <= 2;

   case AGX_OPCODE_PACK_FORMAT:
      for (unsigned s = 0; s < 3; ++s) {
         if (!reg32(I->src[s]) || !plain(I->src[s]))
            return false;
      }

      return reg32(I->dest[0]) &&
             (I->format == AGX_FORMAT_RGB10A2
                 ? reg32(I->src[3]) && plain(I->src[3])
                 : I->format == AGX_FORMAT_RG11B10F && agx_is_null(I->src[3]));

   case AGX_OPCODE_WRITE_COVERAGE:
      return is_export(I->src[0]) && agx_channels(I->src[0]) == 1;

   case AGX_OPCODE_PUBLICATION_PAIR:
      return is_export(I->dest[0]) && agx_channels(I->dest[0]) == 2 &&
             reg32(I->src[0]) && plain(I->src[0]) && reg32(I->src[1]) &&
             plain(I->src[1]);

   case AGX_OPCODE_ZS_STORE:
      return is_export(I->src[0]) && agx_channels(I->src[0]) == 2 &&
             I->zs >= 1 && I->zs <= 3;

   case AGX_OPCODE_ST_VARY:
      return is_imm(I->src[0]) &&
             I->src[0].value < AGX_APPLE9_MAX_UVS_SLOTS &&
             is_export(I->src[1]) && agx_channels(I->src[1]) == 1;

   case AGX_OPCODE_ITER:
      return is_imm(I->src[0]) &&
             I->src[0].value < AGX_APPLE9_NUM_COEFFICIENTS &&
             I->channels == 1 && reg32(I->dest[0]) &&
             (I->interpolation == AGX_INTERPOLATION_CENTER || reg32(I->src[1]));

   case AGX_OPCODE_LDCF:
      return is_imm(I->src[0]) && I->src[0].value < 512 && I->channels == 3 &&
             vec32(I->dest[0], 3);

   case AGX_OPCODE_INTERPOLATION_POSITION:
      return reg32(I->dest[0]) && reg32(I->src[0]) &&
             I->interpolation != AGX_INTERPOLATION_CENTER;

   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ST_TILE:
      return tile_ok(I);

   case AGX_OPCODE_BEGIN_TILE:
      return I->imm == 0x700 || I->imm == 0xf00 || I->imm == 0x600 ||
             (I->imm & ~0xfcull) == 0x800 || I->imm == 1;

   case AGX_OPCODE_END_TILE:
      return I->imm == 0x300 || (I->imm & ~0xfcull) == 0 ||
             (I->imm & ~0xfcull) == 0x200 || I->imm == 1;

   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
      return texture_ok(I);

   case AGX_OPCODE_IMAGE_WRITE:
      /* 2D only, the image index in the low register of an aligned pair */
      return is_export(I->src[0]) && agx_channels(I->src[0]) == 8 &&
             agx_is_null(I->src[1]) && agx_is_null(I->src[2]) &&
             agx_is_null(I->src[3]) && is_reg(I->src[4]) &&
             I->src[4].size == AGX_SIZE_32 && agx_channels(I->src[4]) == 2 &&
             I->dim == AGX_DIM_2D && !I->coherent;

   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      /* Implicit tile coordinates, a 4-bit format and a table index */
      return is_export(I->src[0]) && agx_channels(I->src[0]) == 4 &&
             I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value < 16 &&
             agx_is_null(I->src[2]) && agx_is_null(I->src[3]) &&
             I->format <= 15 && !I->explicit_coords &&
             (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_2D_ARRAY ||
              I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY);

   case AGX_OPCODE_IMAGE_LOAD:
      /* No known Apple9 encoding */
      return false;

   default:
      /* Pseudo-instructions and operations without operand restrictions */
      return true;
   }
}

bool
agx_instr_is_legal(const agx_context *ctx, const agx_instr *I)
{
   if (ctx->target.isa != AGX_SHADER_ISA_APPLE9)
      return true;

   return apple9_instr_is_legal(I);
}

bool
agx_src_is_legal(const agx_context *ctx, agx_instr *I, unsigned s,
                 agx_index src)
{
   if (ctx->target.isa != AGX_SHADER_ISA_APPLE9)
      return true;

   agx_index old = I->src[s];
   I->src[s] = src;
   bool legal = apple9_instr_is_legal(I);
   I->src[s] = old;
   return legal;
}

/* Hard register limits of Apple9 operations, in 16-bit units */
static bool
apple9_constraints(const agx_instr *I, struct agx_reg_constraint *c)
{
   unsigned D = I->nr_dests;
   const uint16_t r63 = 2 * 63;
   bool any = true;

   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
   case AGX_OPCODE_GET_SR:
   case AGX_OPCODE_GET_SR_COVERAGE:
   case AGX_OPCODE_GET_SR_BARRIER:
   case AGX_OPCODE_SIMD_PREFIX:
      c[0].max = r63;
      for (unsigned s = 0; s < I->nr_srcs; ++s)
         c[D + s].max = r63;
      break;

   case AGX_OPCODE_SIMD_REDUCE:
      /* Halves below 128 */
      c[0].max = c[D].max = 127;
      break;

   case AGX_OPCODE_SHUFFLE:
      c[0].max = c[D].max = c[D + 1].max = r63;
      c[D + 1].align = 2;
      break;

   case AGX_OPCODE_BALLOT:
      c[0].max = c[D].max = r63;
      c[D].clobber = true;
      break;

   case AGX_OPCODE_FMUL_PERSPECTIVE:
      c[0].max = c[D].max = c[D + 1].max = r63;
      break;

   case AGX_OPCODE_LOCAL_LOAD:
      c[0].max = c[D + 1].max = r63;
      c[D + 1].align = 2;
      break;

   case AGX_OPCODE_LOCAL_STORE:
      c[D].max = c[D + 2].max = r63;
      c[D + 2].align = 2;
      c[D].clobber = true;
      break;

   case AGX_OPCODE_LOCAL_ATOMIC:
      c[0].max = c[D + 2].max = r63;
      c[D].max = r63 + 2 - agx_index_size_16(I->src[0]);
      c[D + 2].align = 2;
      c[D].clobber = c[D + 2].clobber = true;
      break;

   case AGX_OPCODE_STACK_LOAD:
      /* Indexed private memory reaches r0-r63 only */
      if (I->src[0].type == AGX_INDEX_IMMEDIATE)
         return false;

      c[0].max = c[D].max = r63;
      break;

   case AGX_OPCODE_STACK_STORE:
      if (I->src[1].type == AGX_INDEX_IMMEDIATE)
         return false;

      c[D].max = c[D + 1].max = r63;
      c[D].clobber = true;
      break;

   case AGX_OPCODE_ATOMIC:
      /* Every register operand */
      for (unsigned s = 0; s < 3; ++s)
         c[D + s].clobber = true;
      break;

   case AGX_OPCODE_INTERPOLATION_POSITION:
      c[D].clobber = true;
      break;

   case AGX_OPCODE_PACK_HALF:
      /* The first conversion writes the destination before the second reads */
      c[0].early_clobber = true;
      break;

   case AGX_OPCODE_UNPACK_NORM:
      /* An aligned register pair */
      c[0].align = 4;
      break;

   case AGX_OPCODE_PACK_UNORM_4X8:
      /* The halves are written by separate instructions */
      c[0].early_clobber = true;
      break;

   case AGX_OPCODE_PUBLICATION_PAIR:
      /* An even pair of export registers below 64 */
      c[0].max = 2 * 62;
      c[0].align = 4;
      break;

   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      /* Parameters in four export registers at 0, 4, 8 or 12 */
      c[D].max = 2 * 12;
      c[D].align = 8;
      break;

   case AGX_OPCODE_IMAGE_WRITE:
      /* Eight exports at 0, 8, 16 or 24, the image in an aligned pair */
      c[D].max = 2 * 24;
      c[D].align = 16;
      c[D + 4].align = 4;
      break;

   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD: {
      /* Parameters in exports 0-31, results in r0-r63 with constant indices */
      unsigned words = agx_apple9_texture_params(I);
      unsigned last = words == 2 ? 30 : words == 4 ? 28 : 32 - words;
      c[D].max = 2 * last;
      c[D].align = words == 2 ? 4 : 8;

      if (I->src[3].type == AGX_INDEX_IMMEDIATE)
         c[0].max = 2 * (64 - agx_channels(I->dest[0]));
      break;
   }

   case AGX_OPCODE_FFMA:
      /* Operand B can only be the low half of a register */
      if (I->src[1].size == AGX_SIZE_16)
         c[D + 1].align = 2;
      else
         any = false;
      break;

   default:
      any = false;
      break;
   }

   return any;
}

/* Exports are never moved, so narrow definitions to their readers' limits */
static void
constrain_export_definitions(agx_context *ctx)
{
   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   uint8_t *dest_index = calloc(ctx->alloc, sizeof(*dest_index));

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         if (I->dest[d].reg_class == RA_EXPORT) {
            defs[I->dest[d].value] = I;
            dest_index[I->dest[d].value] = d;
         }
      }
   }

   agx_foreach_instr_global(ctx, I) {
      if (!I->reg_constraints)
         continue;

      agx_foreach_ssa_src(I, s) {
         if (I->src[s].reg_class != RA_EXPORT)
            continue;

         struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
         agx_instr *def = defs[I->src[s].value];
         unsigned d = dest_index[I->src[s].value];
         assert(def != NULL && "export values are defined before use");

         if (!def->reg_constraints) {
            unsigned n = def->nr_dests + def->nr_srcs;
            def->reg_constraints =
               ralloc_array(def, struct agx_reg_constraint, n);

            for (unsigned i = 0; i < n; ++i)
               def->reg_constraints[i] =
                  (struct agx_reg_constraint){0, UINT16_MAX, 1};
         }

         struct agx_reg_constraint *dc = &def->reg_constraints[d];
         dc->min = MAX2(dc->min, c.min);
         dc->max = MIN2(dc->max, c.max);
         dc->align = MAX2(dc->align, c.align);
      }
   }

   free(defs);
   free(dest_index);
}

void
agx_set_reg_constraints(agx_context *ctx)
{
   if (ctx->target.isa != AGX_SHADER_ISA_APPLE9)
      return;

   agx_foreach_instr_global(ctx, I) {
      unsigned n = I->nr_dests + I->nr_srcs;
      struct agx_reg_constraint c[AGX_MAX_NORMAL_SOURCES + 4];

      if (I->reg_constraints || n > ARRAY_SIZE(c))
         continue;

      for (unsigned i = 0; i < n; ++i)
         c[i] = (struct agx_reg_constraint){0, UINT16_MAX, 1};

      if (apple9_constraints(I, c)) {
         I->reg_constraints = ralloc_array(I, struct agx_reg_constraint, n);
         memcpy(I->reg_constraints, c, sizeof(c[0]) * n);
      }
   }

   constrain_export_definitions(ctx);
}
