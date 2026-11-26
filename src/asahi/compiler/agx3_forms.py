"""
Copyright 2026 Youwes09
SPDX-License-Identifier: MIT
"""

# Encoding forms of the AGX3 instructions, as the compiler uses them: the
# instructions each is packed as, the constraints on its operands and its
# scheduling rules. agx3_forms.h.py and agx3_forms.c.py generate the
# compiler's table from them, in this order.
#
# A form's encoding lists the instructions of AGX3.xml it is packed as:
# alternatives separated by ";", instructions in sequence, "|" between
# instructions that can fill a position, "*n" repeating one. Forms that are
# not packed are not in the table: they document sequences the packer does
# not use.
#
# The table is conservative: a form is allocator_safe when every register
# field it needs is understood well enough to pack, not merely when the
# compiler has emitted it. A register field whose high bits also carry
# cache, release or file state is not.

machine = {
   "gpr_count": 96,
   "half_register_count": 192,
   "hardware_register_interlocks": True,
   "software_waits": False,
   "spilling_supported": True,
   "publication_count": 128,
   "uniform_count": 256,
   "fill_word_count": 1024,
   "save_word_count": 2048,
}

forms = []

class Operand(object):
   def __init__(self, role, max, files, widths, min, uniform_max, align,
                flags, evidence):
      self.role = role
      self.max = max
      self.files = files
      self.widths = widths
      self.min = min
      self.uniform_max = uniform_max
      self.align = align
      self.flags = flags
      self.evidence = evidence

class Form(object):
   def __init__(self, name, length, encoding, allocator_safe, evidence,
                dependency, retains, fixed_slot, alternates, prior, exempt,
                split, packed, completion, reads_sources, waits_exports):
      self.name = name
      self.length = length
      self.encoding = encoding
      self.allocator_safe = allocator_safe
      self.evidence = evidence
      self.dependency = dependency
      self.retains = retains
      self.fixed_slot = fixed_slot
      self.alternates = alternates
      self.prior = prior
      self.exempt = exempt
      self.split = split
      self.packed = packed
      self.completion = completion
      self.reads_sources = reads_sources
      self.waits_exports = waits_exports
      self.operands = []

def form(name, length, encoding, allocator_safe = True, evidence = "hardware",
         dependency = "none", retains = False, fixed_slot = 0, alternates = [],
         prior = False, exempt = [], split = False, packed = True,
         completion = None, reads_sources = False, waits_exports = False):
   forms.append(Form(name, length, encoding, allocator_safe, evidence,
                     dependency, retains, fixed_slot, alternates, prior, exempt,
                     split, packed, completion, reads_sources, waits_exports))

# Adds an operand to the last form
def operand(role, max, files = ["gpr"], widths = [32], min = 0,
            uniform_max = 0, align = 2, flags = ["allocatable"],
            evidence = "hardware"):
   forms[-1].operands.append(Operand(role, max, files, widths, min,
                                     uniform_max, align, flags, evidence))

# Roles of the operands of the instructions whose operands are not, in order,
# the dest and the sources src0, src1... of their forms. None marks an operand
# with no entry in the table.
roles = {
   "device_load": ["dest", "src1", "index"],
   "device_store": ["store_data", "src1", "index"],
   "atomic": ["atomic_data", None, "index"],
   "atomic_indirect": ["atomic_data", "src1", "index"],
   "atomic_result": ["src0"],
   "local_atomic": ["dest", "atomic_data", "index"],
   "texture": ["dest", None, None, None],
   "texture_reg": ["dest", None, None, None],
   "texture_reg2": ["dest", None, None, None],
}

form("mov_imm_compact", 2, "mov_imm")
operand("dest", 15, flags = ["allocatable", "hard_low", "compact_preferred"])

form("mov_imm32", 8, "mov_imm32")
operand("dest", 63, flags = ["allocatable", "scattered"])

form("get_sr", 4, "get_sr")
operand("dest", 63, flags = ["allocatable", "compact_preferred"])

form("get_sr_zext16", 8, "get_sr16 zext16")
operand("dest", 63, flags = ["allocatable", "compact_preferred"])

form("bit_unary", 8, "popcount|ffs|bitrev|popcount_uniform",
     dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("uint_to_float", 8, "u32_to_f", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("sint_to_float", 8, "s32_to_f", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("float_to_sint", 10, "f_to_s32", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("float_to_uint", 10, "f_to_u32", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("float_to_half", 6, "fadd_f16", dependency = "index_45_47")
operand("dest", 95)
operand("src0", 95)

form("float_to_half_zext", 10, "fadd_f16 zext16", dependency = "index_45_47",
     packed = False)
operand("dest", 63)
operand("src0", 95)

form("pack_half_2x16", 12, "fadd_f16 fadd_f16", split = True)
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)

form("pack_unorm_2x16", 10, "pack_unorm_2x16")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)

form("pack_unorm_4x8", 20, "pack_unorm_4x8*2", split = True)
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)

form("pack_format", 14, "pack_rgb10a2", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)

form("pack_format_rgb", 14, "pack_rg11b10f", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("half_to_float", 6, "fadd", dependency = "index_45_47")
operand("dest", 95)
operand("src0", 95)

form("unpack_norm", 8, "unpack_norm", dependency = "mask_12_17")
operand("dest", 94, widths = [64], align = 4)
operand("src0", 95)

form("half2", 6, "hadd|hmul|fadd_f16|fmul_f16", dependency = "index_45_47")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)

form("half3", 8, "hfma", dependency = "index_61_63")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("half_minmax", 6, "hmax|hmin", dependency = "index_45_47")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)

form("half_special", 10,
     "rsqrt_f16|exp2_f16|log2_f16|rint_f16|floor_f16|ceil_f16|trunc_f16|srsqrt_f16_f32|rcp_f16|rcp_f16_f32",
     dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95)

form("half_compare_select", 10, "fcmpsel", dependency = "index_61_63")
operand("dest", 95, flags = ["allocatable", "compact_preferred"])
operand("src0", 95, flags = ["allocatable", "scattered"])
operand("src1", 95, flags = ["allocatable", "scattered"])
operand("src2", 95, flags = ["allocatable", "scattered"])
operand("src3", 95, flags = ["allocatable", "scattered"])

form("half_predicate_short", 6, "fcmp_pred", dependency = "index_45_47")
operand("src0", 95)
operand("src1", 95)

form("half_predicate_extended", 10, "fcmp_pred_ext", dependency = "index_61_63")
operand("src0", 95)
operand("src1", 95)

form("float2_immediate_compact", 6, "fadd|fmul", dependency = "index_45_47",
     retains = True)
operand("dest", 95, flags = ["allocatable", "compact_preferred"])
operand("src0", 95, flags = ["allocatable", "compact_preferred"])

form("float2_immediate_extended", 8, "fadd_mod|fmul_mod",
     dependency = "mask_45_47_61_63", retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, flags = ["allocatable", "scattered"])

form("float2_compact", 6, "fadd|fmul", dependency = "index_45_47",
     retains = True)
operand("dest", 95, widths = [16, 32], align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], widths = [16, 32],
        uniform_max = 255, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src1", 95, files = ["gpr", "uniform"], widths = [16, 32],
        uniform_max = 255, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered", "uniform_alternative"])

form("float2_base", 4, "fadd_base|fmul_base")
operand("dest", 63)
operand("src0", 63)
operand("src1", 63)

form("float2_project", 8, "fmul_cf")
operand("dest", 63, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src0", 63, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src1", 63, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])

form("float2_modifier_extended", 8, "fadd_mod|fmul_mod",
     dependency = "mask_45_47_61_63", retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("float2_abs_extended", 10, "fadd_abs", dependency = "mask_45_47_61_63",
     retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("float2_mul_abs_extended", 12, "fmul_abs", dependency = "mask_45_47_61_63",
     retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("float3_extended", 8, "ffma", dependency = "index_61_63", retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src2", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("float3_compact", 6, "ffma_compact")
operand("dest", 63)
operand("src0", 63)
operand("src1", 63, files = ["gpr", "uniform", "immediate"], uniform_max = 63)
operand("src2", 63)

form("float3_saturate_extended", 10, "ffma_sat",
     dependency = "mask_61_63_77_79", retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src2", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("float3_modifier_extended", 12, "ffma_mod",
     dependency = "mask_61_63_77_79", retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src2", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("cube", 12, "cube")
operand("dest", 95)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("float_special", 10,
     "rsqrt|srsqrt|sin_factor|exp2|log2|rint|floor|ceil|trunc|rcp",
     dependency = "mask_12_17", retains = True)
operand("dest", 95)
operand("src0", 95)

form("derivative", 10, "dfdx|dfdy", dependency = "mask_12_17")
operand("dest", 95)
operand("src0", 95)

form("int_add_extended", 10, "iadd|isub", dependency = "mask_12_17",
     retains = True)
operand("dest", 95, widths = [16, 32], align = 1)
operand("src0", 95, files = ["gpr", "uniform", "immediate"], widths = [16, 32],
        uniform_max = 255, align = 1)
operand("src1", 95, files = ["gpr", "uniform", "immediate"], widths = [16, 32],
        uniform_max = 255, align = 1,
        flags = ["allocatable", "uniform_alternative"])

form("int_mad_extended", 12, "imad", dependency = "mask_12_17", retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255)
operand("src2", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255)

form("int_mul_wide", 12, "umul_wide|imul_wide", dependency = "mask_12_17",
     retains = True)
operand("dest", 94, widths = [64], align = 4)
operand("src0", 95)
operand("src1", 95)

# Min/max may wait for a pending load while keeping the source live for a later
# read
form("minmax_compact", 6, "fmax|fmin|umax|umin|imax|imin",
     dependency = "index_45_47", retains = True)
operand("dest", 95, widths = [16, 32], align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], widths = [16, 32],
        uniform_max = 255, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], widths = [16, 32],
        uniform_max = 255, align = 1,
        flags = ["allocatable", "compact_preferred", "scattered"])

form("select_gpr_wide", 10, "fcmpsel|icmpsel", dependency = "index_61_63",
     retains = True)
operand("dest", 95, flags = ["allocatable", "compact_preferred"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src2", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src3", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("select_modifier_extended", 14, "fcmpsel_mod",
     dependency = "mask_61_63_93_95")
operand("dest", 95, flags = ["allocatable", "compact_preferred"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src2", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src3", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("logic_extended", 10, "bitop", dependency = "mask_45_47_61_63",
     retains = True)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255,
        flags = ["allocatable", "scattered"])
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255,
        flags = ["allocatable", "scattered"])

form("logic_export", 10, "bitop_export", dependency = "mask_45_47_61_63",
     retains = True)
operand("dest", 127, flags = ["allocatable", "scattered", "publication"])
operand("src0", 95, flags = ["allocatable", "scattered"])
operand("src1", 95, flags = ["allocatable", "scattered"])

form("logic_uniform", 10, "bitop", dependency = "mask_45_47_61_63",
     packed = False)
operand("dest", 95, flags = ["allocatable", "scattered"])
operand("src0", 95, flags = ["allocatable", "scattered"])

form("store_uniform", 10, "iadd", dependency = "mask_12_17", retains = True)
operand("dest", 255, files = ["uniform"], flags = [])
operand("src0", 95)

form("store_uniform_add", 10, "iadd|isub", dependency = "mask_12_17")
operand("dest", 255, files = ["uniform"], flags = [])
operand("src0", 95)
operand("src1", 95)

form("store_uniform_mad", 12, "imad", dependency = "mask_12_17")
operand("dest", 255, files = ["uniform"], flags = [])
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("store_uniform_mul_wide", 12, "umul_wide|imul_wide",
     dependency = "mask_12_17")
operand("dest", 254, files = ["uniform"], widths = [64], align = 4, flags = [])
operand("src0", 95)
operand("src1", 95)

form("shift_extended", 10, "asr", dependency = "mask_12_17", retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("shift_arith_register", 10, "asr_reg", dependency = "mask_12_17",
     retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("shift_logical_immediate", 12, "shl|shr", dependency = "mask_12_17",
     retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("shift_logical_register", 12, "shl_reg|shr_reg", dependency = "mask_12_17",
     retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
operand("src1", 95, files = ["gpr", "uniform"], uniform_max = 255)

form("predicate_compare_short", 6, "fcmp_pred|icmp_pred",
     dependency = "index_45_47", retains = True)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255)

form("predicate_compare_extended", 10, "fcmp_pred_ext|icmp_pred_ext",
     dependency = "index_61_63", retains = True)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
operand("src1", 95, files = ["gpr", "uniform", "immediate"], uniform_max = 255)

form("predicate_compare_loop", 10, "fcmp_pred_ext|icmp_pred_ext",
     dependency = "index_61_63", packed = False)
operand("src0", 95)
operand("src1", 95)

form("exec_mask_push", 4, "if_pred")

form("exec_mask_else", 4, "else")

form("exec_mask_pop", 6, "pop_exec")

form("loop_mask_push", 4, "begin_loop")

form("loop_mask_update", 4, "while_pred")

form("loop_mask_pop", 6, "end_loop")

form("jmp_exec_any", 10, "jmp_exec_any")

form("jmp_exec_none", 10, "jmp_exec_none")

form("break_mask_unwind", 6, "break_pred")

form("get_draw_id", 4, "get_sr", fixed_slot = 1, completion = "other")
operand("dest", 63, flags = ["allocatable", "hard_low"])

form("get_coverage", 4, "get_sr", fixed_slot = 1, completion = "other")
operand("dest", 63, flags = ["allocatable", "hard_low"])

form("float2_export", 8, "fmul_export", dependency = "mask_45_47_61_63",
     retains = True)
operand("dest", 127, flags = ["allocatable", "publication"])
operand("src0", 95)
operand("src1", 95)

form("block_store_params", 30, "bitop_export*3", packed = False)
operand("dest", 15, align = 8)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("block_store_extended_params", 40, "bitop_export*4", packed = False)
operand("dest", 15, align = 8)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)

form("image_write_params", 80, "bitop_export*8", packed = False)
operand("dest", 24, align = 16)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)
operand("src4", 95)
operand("src5", 95)
operand("src6", 95)
operand("src7", 95)

form("image_write", 22, "image_write join_export", waits_exports = True)
operand("src0", 24, align = 16)
operand("src1", 25)
operand("src2", 26)
operand("src3", 27)
operand("src4", 28)
operand("src5", 29)
operand("src6", 30)
operand("src7", 31)
operand("src8", 94, align = 4)
operand("src9", 95)

form("block_image_store", 18, "block_image_store join_export",
     waits_exports = True)
operand("src0", 15, align = 8)
operand("src1", 15)
operand("src2", 15)

form("block_image_store_extended", 18, "block_image_store join_export",
     waits_exports = True)
operand("src0", 15, align = 8)
operand("src1", 15)
operand("src2", 15)
operand("src3", 15)

form("texture_coords", 16, "fmul_export*2", packed = False)
operand("dest", 31, align = 4)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("texture_lod_params", 30, "bitop_export*3", packed = False)
operand("dest", 31, align = 8)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)

form("texture_volume_params", 40, "bitop_export*4", packed = False)
operand("dest", 31, align = 8)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)

form("texture_lod", 14, "texture", completion = "other",
     reads_sources = True, waits_exports = True)
operand("dest", 63)
operand("src0", 31, align = 8)
operand("src1", 31)
operand("src2", 31)
operand("src3", 31)

form("texture_grad_params", 60, "bitop_export*6", packed = False)
operand("dest", 31, align = 8)
operand("src0", 95)
operand("src1", 95)
operand("src2", 95)
operand("src3", 95)
operand("src4", 95)
operand("src5", 95)

form("texture_grad_array_params", 80, "bitop_export*8", evidence = "byte_diff",
     packed = False)
operand("dest", 31, align = 8, evidence = "byte_diff")
operand("src0", 95, evidence = "byte_diff")
operand("src1", 95, evidence = "byte_diff")
operand("src2", 95, evidence = "byte_diff")
operand("src3", 95, evidence = "byte_diff")
operand("src4", 95, evidence = "byte_diff")
operand("src5", 95, evidence = "byte_diff")
operand("src6", 95, evidence = "byte_diff")
operand("src7", 95, evidence = "byte_diff")

form("texture_grad_volume_params", 90, "bitop_export*9", evidence = "byte_diff",
     packed = False)
operand("dest", 31, align = 8, evidence = "byte_diff")
operand("src0", 95, evidence = "byte_diff")
operand("src1", 95, evidence = "byte_diff")
operand("src2", 95, evidence = "byte_diff")
operand("src3", 95, evidence = "byte_diff")
operand("src4", 95, evidence = "byte_diff")
operand("src5", 95, evidence = "byte_diff")
operand("src6", 95, evidence = "byte_diff")
operand("src7", 95, evidence = "byte_diff")
operand("src8", 95, evidence = "byte_diff")

form("texture_grad", 14, "texture", completion = "other",
     reads_sources = True, waits_exports = True)
operand("dest", 63)
operand("src0", 31, align = 8)

form("texture_sample", 14, "texture", completion = "other",
     reads_sources = True, waits_exports = True)
operand("dest", 63)
operand("src0", 31, align = 4)
operand("src1", 31)

form("texture_sample_indexed", 20, "texture_reg|texture_reg2",
     completion = "other", reads_sources = True, waits_exports = True)
operand("dest", 95)
operand("src0", 31, align = 4)
operand("src1", 31)

form("texture_lod_indexed", 20, "texture_reg|texture_reg2",
     completion = "other", reads_sources = True, waits_exports = True)
operand("dest", 95)
operand("src0", 31, align = 8)
operand("src1", 31)
operand("src2", 31)
operand("src3", 31)

form("texture_grad_indexed", 20, "texture_reg|texture_reg2",
     completion = "other", reads_sources = True, waits_exports = True)
operand("dest", 95)
operand("src0", 31, align = 8)

form("interpolation_position", 8, "interpolation_position",
     evidence = "byte_diff", dependency = "mask_12_17")
operand("dest", 95, evidence = "byte_diff")
operand("src0", 95, align = 4, flags = ["allocatable", "clobber"])

form("iter_coord", 10, "iter_coord", evidence = "byte_diff")
operand("dest", 95, evidence = "byte_diff")
operand("src0", 95, align = 1, evidence = "byte_diff")

form("iter", 10, "iter")
operand("dest", 95)

form("iter_flat", 6, "ldcf", completion = "other")
operand("dest", 93)

form("vary_store", 8, "st_vary", dependency = "mask_12_17")
operand("src0", 127, flags = ["allocatable", "publication"])

# Fragments with discard or depth output wait for the coverage publication
# here, rather than before the tile access
form("coverage", 6, "write_coverage", dependency = "mask_12_17", prior = True)
operand("src0", 63)

form("depth_store", 6, "zs_store", prior = True)
operand("src0", 62, align = 4, flags = ["allocatable", "publication"])
operand("src1", 63, flags = ["allocatable", "publication"])

form("publication_pair", 20, "bitop_export*2")
operand("dest", 62, align = 4, flags = ["allocatable", "publication"])
operand("src0", 95)
operand("src1", 95)

form("publication_split", 10, "bitop_export")
operand("dest", 62, align = 4, flags = ["allocatable", "publication"])
operand("src0", 95)

# The tile access opens with the coverage publication still pending: the
# coverage instruction waits for it
form("tile_access", 6, "begin_tile", prior = True, exempt = ["publication"])

form("tile_load", 12, "ld_tile", completion = "other")
operand("dest", 95)

form("tile_store", 12, "st_tile")
operand("src0", 95)

form("tile_load_mask", 12, "ld_tile_mask", evidence = "byte_diff",
     completion = "other")
operand("dest", 95, evidence = "byte_diff")
operand("src0", 95, evidence = "byte_diff")

# Waits for the coverage read supplying a dynamic mask
form("tile_store_mask", 12, "st_tile_mask", evidence = "byte_diff",
     dependency = "mask_12_17")
operand("src0", 95, evidence = "byte_diff")
operand("src1", 95, evidence = "byte_diff")

form("tile_fence", 6, "end_tile", prior = True)

form("device_load", 14, "device_load", dependency = "mask_12_17",
     retains = True, completion = "memory")
operand("dest", 95, widths = [16, 32, 64], align = 1)
operand("index", 95)

form("device_load_indirect", 14, "device_load", dependency = "mask_12_17",
     retains = True, completion = "memory")
operand("dest", 95, widths = [16, 32, 64], align = 1)
operand("index", 95)
operand("src1", 94, evidence = "byte_diff")
operand("src2", 95, evidence = "byte_diff")

form("device_load_pointer", 14, "device_load", evidence = "byte_diff",
     dependency = "mask_12_17", retains = True, completion = "memory")
operand("dest", 95, widths = [16, 32, 64], align = 1)
operand("src1", 94, evidence = "byte_diff")
operand("src2", 95, evidence = "byte_diff")

form("launch_move", 4, "mov_launch", packed = False)
operand("dest", 63)
operand("src0", 63)

form("uniform_move", 4, "mov_uniform", evidence = "byte_diff")
operand("dest", 63, flags = ["allocatable", "scattered"],
        evidence = "byte_diff")

form("device_store", 14, "device_store", dependency = "mask_12_17")
operand("index", 95)
operand("store_data", 95, widths = [16, 32, 64], align = 1)

form("device_atomic", 14, "atomic", dependency = "mask_12_17")
operand("index", 95, flags = ["allocatable", "scattered", "clobber"])
operand("atomic_data", 95, flags = ["allocatable", "scattered", "clobber"])

form("device_atomic_indirect", 14, "atomic_indirect", dependency = "mask_12_17")
operand("index", 95, flags = ["allocatable", "scattered", "clobber"])
operand("atomic_data", 95, flags = ["allocatable", "scattered", "clobber"])
operand("src1", 94, flags = ["allocatable", "clobber"])
operand("src2", 95, flags = ["allocatable", "clobber"])

form("device_atomic_return", 22, "atomic atomic_result",
     dependency = "mask_12_17", completion = "memory")
operand("dest", 95)
operand("index", 95, flags = ["allocatable", "scattered", "clobber"])
operand("atomic_data", 95, flags = ["allocatable", "scattered", "clobber"])

form("device_atomic_indirect_return", 22, "atomic_indirect atomic_result",
     dependency = "mask_12_17", completion = "memory")
operand("dest", 95)
operand("index", 95, flags = ["allocatable", "scattered", "clobber"])
operand("atomic_data", 95, flags = ["allocatable", "scattered", "clobber"])
operand("src1", 94, flags = ["allocatable", "clobber"])
operand("src2", 95, flags = ["allocatable", "clobber"])

form("subgroup_scan_iadd", 8, "simd_prefix_iadd")
operand("dest", 63)
operand("src0", 63)

form("subgroup_reduce_umax", 8, "simd_reduce_umax|simd_reduce_umax32")
operand("dest", 63)
operand("src0", 63)

form("subgroup_broadcast", 10, "shuffle")
operand("dest", 63)
operand("src0", 63)
operand("src1", 63)

form("subgroup_ballot", 10, "ballot", evidence = "byte_diff")
operand("dest", 63, evidence = "byte_diff")
operand("src0", 63, flags = ["allocatable", "clobber"], evidence = "byte_diff")

form("workgroup_barrier", 6, "threadgroup_barrier", prior = True,
     exempt = ["memory"])

# Fences and barriers are issued with loads and returning atomics outstanding,
# which their results' consumers wait for. The scheduler still drains before
# the raster order fence, which has no such evidence.
form("device_fence", 6, "memory_barrier|raster_order_fence",
     evidence = "byte_diff", prior = True, exempt = ["memory"])

form("halt", 4, "stop", prior = True)

form("shared_load", 14, "local_load", fixed_slot = 6, completion = "memory")
operand("dest", 63)
operand("src0", 63)

form("shared_atomic", 12, "local_atomic", fixed_slot = 1, completion = "memory")
operand("dest", 63)
operand("atomic_data", 63, flags = ["allocatable", "clobber"])
operand("index", 63, flags = ["allocatable", "clobber"])

form("shared_store", 14, "local_store")
operand("src0", 63, flags = ["allocatable", "clobber"])
operand("src1", 63)

form("private_load", 12, "stack_load_indexed", fixed_slot = 6,
     completion = "other")
operand("dest", 63)
operand("src0", 63)

form("private_store", 10, "stack_store_indexed")
operand("src0", 63, flags = ["allocatable", "clobber"])
operand("src1", 63)

form("device_atomic_result", 8, "atomic_result", packed = False)
operand("src0", 95)

form("spill_store", 10, "stack_store", dependency = "mask_45_47_61_63",
     completion = "other", reads_sources = True)
operand("src0", 95, flags = ["allocatable", "scattered"])

form("spill_load", 8, "stack_load", dependency = "mask_45_47_61_63",
     completion = "other")
operand("dest", 95, flags = ["allocatable", "scattered"])

form("device_store_indirect", 14, "device_store", dependency = "mask_12_17")
operand("index", 95)
operand("store_data", 95, widths = [16, 32, 64], align = 1)
operand("src1", 94)
operand("src2", 95)

form("tile_load_coords", 12, "ld_tile_coords", completion = "other")
operand("dest", 95)
operand("src0", 95)

# Waits for the slots in its mask; the empty mask is a nop
form("wait", 2, "wait")

form("float_special_uniform", 10,
     "rsqrt|srsqrt|sin_factor|exp2|log2|rint|floor|ceil|trunc|rcp_uniform",
     dependency = "mask_12_17", retains = True)
operand("dest", 95)
operand("src0", 95, files = ["gpr", "uniform"], uniform_max = 255)
