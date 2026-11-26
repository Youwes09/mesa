# Copyright 2026 Youwes09
# SPDX-License-Identifier: MIT

# Checks the AGX3 forms of agx3_forms.py: that each is consistent with its
# instructions in AGX3.xml, that the compiler's table generated from them
# matches, and, with FORMS, that the packer's instructions decode to
# instructions of their forms.
#
# usage: test-agx3-forms.py AGX3.xml agx3-forms-dump [FORMS]

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import isa as isa_xml
import agx3_forms

errors = []

def error(msg):
    errors.append(msg)

xml_path, dump = sys.argv[1:3]
vectors = sys.argv[3] if len(sys.argv) > 3 else None
root = ET.parse(xml_path).getroot()
isa = isa_xml.ISA(root)

FORM_KEYS = ['length', 'allocator_safe', 'evidence', 'dependency', 'retains',
             'fixed_slot', 'alternates', 'prior', 'exempt', 'split']
OPERAND_KEYS = ['files', 'widths', 'min', 'max', 'uniform_max', 'align',
                'flags', 'evidence']

def attrs(line):
    return dict(kv.split('=', 1) for kv in line.split()[2:])

# A value of the table in the dump's spelling
def dump_value(v):
    if isinstance(v, bool):
        return str(int(v))
    elif isinstance(v, list):
        return ','.join(str(x) for x in v) or 'none'
    else:
        return str(v)

# The C table
table = {}
order = []
machine = None
for line in subprocess.run([dump], capture_output=True, text=True,
                           check=True).stdout.splitlines():
    kind = line.split()[0]
    if kind == 'machine':
        machine = dict(kv.split('=', 1) for kv in line.split()[1:])
    elif kind == 'form':
        name = line.split()[1]
        table[name] = (attrs(line), [])
        order.append(name)
    elif kind == 'operand':
        table[order[-1]][1].append((line.split()[1], attrs(line)))

# Register files
for key, value in machine.items():
    if dump_value(agx3_forms.machine.get(key)) != value:
        error(f'machine {key}: agx3_forms.py {agx3_forms.machine.get(key)}, '
              f'table {value}')

# Forms. Those not packed describe encodings the packer does not use
forms = {f.name: f for f in agx3_forms.forms}
unpacked = {n for n, f in forms.items() if not f.packed}
for name in sorted(set(table) ^ (set(forms) - unpacked)):
    error(f'form {name} only in the {"table" if name in table else "agx3_forms.py"}')

def sequences(spec):
    """Alternatives of a form's encoding: lists of positions, each a list of
    instruction names, expanding NAME*N repetitions."""
    out = []
    for alternative in spec.split(';'):
        positions = []
        for position in alternative.split():
            names, _, count = position.partition('*')
            positions += [names.split('|')] * int(count or 1)
        out.append(positions)
    return out

def lengths(positions):
    total = {0}
    for position in positions:
        total = {t + isa.instrs[n].length[0] for t in total for n in position
                 if n in isa.instrs}
    return total

# Value bits of an operand: an instruction bit, or None if not encoded
def value_bits(op):
    out = []
    for r in op.bits.ranges:
        if isinstance(r, isa_xml.ConstantBit):
            out.append(None if r.value == 0 else 'one')
        else:
            out += list(range(r.start, r.end + 1))
    return out

# Largest 32-bit register (or uniform or export word) an operand reaches
def reach(op, names):
    best = None
    vbits = value_bits(op)
    for case in isa.encodings[op.kind].cases:
        imm = case.kind
        if imm is None or imm.name not in names:
            continue
        units = 0
        n = 0
        for r in imm.bits.ranges:
            for b in range(r.start, r.end + 1):
                if b < len(vbits) and vbits[b] is not None:
                    units |= 1 << n
                n += 1
        words = (units << imm.shift) >> 1
        best = words if best is None else max(best, words)
    return best

def find_operand(ins, role):
    if ins.name in agx3_forms.roles:
        roles = agx3_forms.roles[ins.name]
        return (ins.dests + ins.sources)[roles.index(role)] if role in roles else None
    if role == 'dest':
        return ins.dests[0] if ins.dests else None
    if role.startswith('src'):
        i = int(role[3:])
        return ins.sources[i] if i < len(ins.sources) else None
    return None

# The dependency field of each layout: a slot index or a slot mask
DEPENDENCY = {
    'index_45_47': ('wait', '45:47'),
    'index_61_63': ('wait', '61:63'),
    'mask_12_17': ('waitmask', '12:17'),
    'mask_45_47_61_63': ('waitmask', '45:47 61:63'),
    'mask_61_63_77_79': ('waitmask', '61:63 77:79'),
    'mask_61_63_93_95': ('waitmask', '61:63 93:95'),
}

for name in order:
    if name not in forms:
        continue
    form = forms[name]
    info, operands = table[name]

    for key in FORM_KEYS:
        v = dump_value(getattr(form, key))
        if v != info[key]:
            error(f'{name} {key}: agx3_forms.py {v}, table {info[key]}')

    py_roles = [o.role for o in form.operands]
    if py_roles != [r for r, _ in operands]:
        error(f'{name} operands: agx3_forms.py {py_roles}, '
              f'table {[r for r, _ in operands]}')
    else:
        for o, (role, t) in zip(form.operands, operands):
            for key in OPERAND_KEYS:
                v = dump_value(getattr(o, key))
                if v != t[key]:
                    error(f'{name} {role} {key}: agx3_forms.py {v}, table {t[key]}')

    # The instructions of the form
    alternatives = sequences(form.encoding)
    for positions in alternatives:
        for position in positions:
            for ins in position:
                if ins not in isa.instrs:
                    error(f'{name}: no instruction {ins}')

    if int(info['length']) not in set().union(*[lengths(p) for p in alternatives]):
        error(f'{name}: no encoding of {info["length"]} bytes')

    # The dependency field is in the first instruction
    if info['dependency'] != 'none':
        kind, bits = DEPENDENCY[info['dependency']]
        for positions in alternatives:
            for ins in positions[0]:
                mods = [m for m in isa.instrs[ins].modifiers
                        if m.kind == kind and str(m.bits) == bits]
                if len(mods) != 1:
                    error(f'{name}: {ins} has no {info["dependency"]} field')

    # Register ranges of single-instruction forms
    if len(alternatives) != 1 or len(alternatives[0]) != 1:
        continue

    for role, t in operands:
        files = t['files'].split(',')
        for ins in alternatives[0][0]:
            # Table operands the instruction does not encode, like pair halves
            op = find_operand(isa.instrs[ins], role)
            if op is None:
                continue

            flags = t['flags'].split(',')
            if 'gpr' in files:
                kind = 'export' if 'publication' in flags or op.kind == 'export' else 'reg'
                r = reach(op, [kind])
                if r is None:
                    if files == ['gpr']:
                        error(f'{name}: {ins} {role} names no register')
                elif r < int(t['max']):
                    error(f'{name}: {ins} {role} reaches {kind} {r}, table {t["max"]}')
            if 'uniform' in files:
                limit = t['max'] if files == ['uniform'] else t['uniform_max']
                r = reach(op, ['uniform'])
                if r is None:
                    error(f'{name}: {ins} {role} names no uniform')
                elif r < int(limit):
                    error(f'{name}: {ins} {role} reaches uniform {r}, table {limit}')

# Instructions of a form
def instructions(form):
    return {ins for positions in sequences(form.encoding)
            for position in positions for ins in position}

# Every instruction is packed as some form
for ins in sorted(set(isa.instrs) - set().union(*map(instructions, forms.values()))):
    error(f'{ins}: no form')

# The packer's instructions
def decode(raw, avail):
    for ins in isa.instrs.values():
        n = ins.length[0]
        if n <= avail and (raw & ((1 << (8 * n)) - 1) & ins.exact.mask) == ins.exact.value:
            return ins
    return None

if vectors:
    for line in open(vectors):
        form, code = line.split()
        data = bytes.fromhex(code)
        ins = decode(int.from_bytes(data, 'little'), len(data))
        if ins is None or ins.length[0] != len(data):
            error(f'{form} {code}: not one instruction')
        elif ins.name not in instructions(forms[form]):
            error(f'{form} {code}: decodes to {ins.name}, of another form')

for e in errors:
    print(e)
print('Fail.' if errors else 'Pass.')
sys.exit(1 if errors else 0)
