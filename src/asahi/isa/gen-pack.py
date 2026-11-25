# Copyright 2026 The Asahi Linux Contributors
# SPDX-License-Identifier: MIT

# Generates instruction packers from an ISA description: for each instruction
# a function packing its fields over its template, and for each operand
# encoding a function packing an operand into the value of a field.

import sys
from mako.template import Template
from mako import exceptions
import isa as isa_xml

isa = isa_xml.load(sys.argv[1])
prefix = isa.name

# Operand properties and modifiers, as the disassembler names them
FLAGS = ['size32', 'size64', 'cache', 'lu', 'neg', 'abs', 'sx']

# Operand kinds: the immediates of the operand encodings
KINDS = ['none'] + sorted({c.kind.name for e in isa.encodings.values()
                           for c in e.cases if c.kind is not None})

def deposit(bits, value, out):
    """Statements depositing the bits of value into out, checking that value
    has no bits the field cannot hold."""
    lines = []
    n = 0
    for r in bits.ranges:
        if isinstance(r, isa_xml.ConstantBit):
            lines.append(f'ok &= (({value} >> {n}) & 1) == {r.value};')
        else:
            shifted = f'{value} >> {n}' if n else value
            lines.append(f'{out}({r.start}, {r.size}, {shifted});')
        n += r.size
    if n < 64:
        lines.append(f'ok &= ({value} >> {n}) == 0;')
    return lines

def case_condition(case):
    kind = case.kind.name if case.kind is not None else 'none'
    conds = [f'op.kind == {prefix.upper()}_KIND_{kind.upper()}']
    mods = {m.name for m in case.modifiers}
    for flag in FLAGS:
        if flag in mods:
            continue
        if flag in case.properties:
            conds.append(f'op.{flag}')
        else:
            conds.append(f'!op.{flag}')
    if case.kind is not None and case.kind.shift:
        conds.append(f'(op.value & {(1 << case.kind.shift) - 1}) == 0')
    return ' && '.join(conds)

def case_body(case):
    lines = []
    if case.kind is not None:
        x = f'(op.value >> {case.kind.shift})' if case.kind.shift else 'op.value'
        lines += deposit(case.kind.bits, x, 'SET')
    for mod in case.modifiers:
        assert mod.name in FLAGS, f'operand modifier {mod.name}'
        lines.append(f'SET({mod.bits.ranges[0].start}, 1, op.{mod.name});')
        assert len(mod.bits.ranges) == 1 and mod.bits.size() == 1
    return lines

def fields(ins):
    """Named fields of an instruction other than its operands"""
    out = [x for x in ins.immediates + ins.modifiers if x.name != 'length']
    names = [x.name for x in out]
    assert len(names) == len(set(names)), ins.name
    assert not {'dest', 'src'} & set(names), ins.name
    return out

# Fields of all instructions, as one struct
all_fields = sorted({f.name for ins in isa.instrs.values() for f in fields(ins)})
max_sources = max(len(ins.sources) for ins in isa.instrs.values())

def unused(ins):
    """Fields an instruction does not have, which must be zero"""
    used = {f.name for f in fields(ins)}
    out = [f'v->{f}' for f in all_fields if f not in used]
    if not ins.dests:
        out.append('v->dest')
    out += [f'v->src[{i}]' for i in range(len(ins.sources), max_sources)]
    return out

def chunk_bytes(value, length):
    return ', '.join(f'0x{b:02x}' for b in value.to_bytes(length, 'little'))

template = """/*
 * Generated from ${isa.name.upper()}.xml by gen-pack.py. Do not edit.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define ${P}_MAX_LENGTH ${isa.length}

enum ${p}_kind {
% for kind in KINDS:
   ${P}_KIND_${kind.upper()},
% endfor
};

/* An operand as the disassembler prints it. Register, uniform and export
 * numbers count 16-bit halves.
 */
struct ${p}_operand {
   enum ${p}_kind kind;
   uint32_t value;
% for flag in FLAGS:
   bool ${flag};
% endfor
};

% for kind, enum in isa.enums.items():
/* ${enum.name} */
enum {
% for n, v in enum.values.items():
   ${P}_${kind.upper()}_${v.name.upper()} = ${n},
% endfor
};

% endfor
static inline void
${p}_set(uint8_t *bytes, unsigned start, unsigned width, uint64_t value)
{
   for (unsigned i = 0; i < width; ++i) {
      if ((value >> i) & 1)
         bytes[(start + i) / 8] |= 1 << ((start + i) % 8);
   }
}

/* Operand encodings: the value of an operand field, or UINT64_MAX */
% for name, enc in isa.encodings.items():
static inline uint64_t
${p}_encode_${name}(struct ${p}_operand op)
{
% for case in enc.cases:
   if (${case_condition(case)}) {
      uint64_t v = ${hex(case.exact.value)};
      bool ok = true;
#define SET(start, width, x) v |= ((uint64_t)(x) & ((1ull << (width)) - 1)) << (start)
% for line in case_body(case):
      ${line}
% endfor
#undef SET
      if (ok && (v & ${hex(case.exact.mask)}) == ${hex(case.exact.value)})
         return v;
   }

% endfor
   return UINT64_MAX;
}

% endfor
/* Fields of an instruction: its operand fields, as encoded by the operand
 * encodings, and its other fields. Fields an instruction does not have must
 * be zero.
 */
struct ${p}_fields {
   uint64_t dest;
   uint64_t src[${max_sources}];
% for f in all_fields:
   int64_t ${f};
% endfor
};

/* Instruction packers: the length of the instruction, or 0 for fields it
 * cannot hold
 */
typedef unsigned (*${p}_pack_fn)(uint8_t *out, const struct ${p}_fields *v);

% for name, ins in isa.instrs.items():
static inline unsigned
${p}_pack_${name}(uint8_t *out, const struct ${p}_fields *v)
{
   static const uint8_t template[] = {${chunk_bytes(ins.exact.value, ins.length[0])}};
   bool ok = true;
   memcpy(out, template, sizeof(template));
#define SET(start, width, x) ${p}_set(out, start, width, x)
% for op in ins.dests:
% for line in deposit(op.bits, 'v->dest', 'SET'):
   ${line}
% endfor
% endfor
% for i, op in enumerate(ins.sources):
% for line in deposit(op.bits, f'v->src[{i}]', 'SET'):
   ${line}
% endfor
% endfor
% for f in fields(ins):
% if f.kind == 'signed':
   ok &= v->${f.name} >= -(1ll << ${f.bits.size() - 1}) && v->${f.name} < (1ll << ${f.bits.size() - 1});
% for line in deposit(f.bits, f'((uint64_t)v->{f.name} & {hex((1 << f.bits.size()) - 1)})', 'SET'):
   ${line}
% endfor
% else:
% for line in deposit(f.bits, f'(uint64_t)v->{f.name}', 'SET'):
   ${line}
% endfor
% endif
% endfor
#undef SET
   ok &= (${' | '.join(unused(ins))}) == 0;
   return ok ? sizeof(template) : 0;
}

% endfor
"""

try:
    print(Template(template).render(isa=isa, p=prefix, P=prefix.upper(),
                                    FLAGS=FLAGS, KINDS=KINDS, deposit=deposit,
                                    case_condition=case_condition,
                                    case_body=case_body, fields=fields,
                                    all_fields=all_fields,
                                    max_sources=max_sources, unused=unused,
                                    chunk_bytes=chunk_bytes))
except:
    print(exceptions.text_error_template().render())
