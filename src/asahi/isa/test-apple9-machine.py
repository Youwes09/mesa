# Copyright 2026 The Asahi Linux Contributors
# SPDX-License-Identifier: MIT

# Checks the AGX3.xml forms against the compiler's machine table and, with
# FORMS, that the packer's instructions decode to instructions of their forms.
#
# usage: test-apple9-machine.py AGX3.xml apple9-machine-dump [FORMS]

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa as isa_xml

errors = []

def error(msg):
    errors.append(msg)

xml_path, dump = sys.argv[1:3]
vectors = sys.argv[3] if len(sys.argv) > 3 else None
root = ET.parse(xml_path).getroot()
isa = isa_xml.ISA(root)

# Defaults of the XML form attributes, in the dump's spelling
FORM_DEFAULTS = {
    'allocator_safe': '1', 'evidence': 'hardware', 'dependency': 'none',
    'retains': '0', 'fixed_slot': '0', 'alternates': 'none', 'prior': '0',
    'exempt': 'none', 'split': '0',
}
FORM_KEYS = ['length'] + list(FORM_DEFAULTS)

OPERAND_DEFAULTS = {
    'files': 'gpr', 'widths': '32', 'min': '0', 'uniform_max': '0',
    'align': '2', 'flags': 'allocatable', 'evidence': 'hardware',
}
OPERAND_KEYS = ['files', 'widths', 'min', 'max', 'uniform_max', 'align',
                'flags', 'evidence']

BOOLEAN = {'true': '1', 'false': '0'}

def attrs(line):
    return dict(kv.split('=', 1) for kv in line.split()[2:])

def xml_value(el, key, defaults):
    v = el.get(key, defaults.get(key))
    if v is None:
        error(f'{el.get("name") or el.get("role")}: no {key}')
    return BOOLEAN.get(v, v)

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
el = root.find('machine')
if el is None:
    error('no machine element')
else:
    for key, value in machine.items():
        if BOOLEAN.get(el.get(key), el.get(key)) != value:
            error(f'machine {key}: XML {el.get(key)}, table {value}')

# Forms
forms = {f.get('name'): f for f in root.iter('form')}
for name in sorted(set(table) ^ set(forms)):
    error(f'form {name} only in the {"table" if name in table else "XML"}')

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
    for op in ins.dests + ins.sources:
        if op.role == role:
            return op
    if role == 'dest':
        return ins.dests[0] if ins.dests else None
    if role.startswith('src'):
        # Operands without a role are the table's sources in order
        srcs = [op for op in ins.sources if op.role is None]
        i = int(role[3:])
        return srcs[i] if i < len(srcs) else None
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
        v = xml_value(form, key, FORM_DEFAULTS)
        if v != info[key]:
            error(f'{name} {key}: XML {v}, table {info[key]}')

    xml_operands = [(o.get('role'), o) for o in form.findall('operand')]
    if [r for r, _ in xml_operands] != [r for r, _ in operands]:
        error(f'{name} operands: XML {[r for r, _ in xml_operands]}, '
              f'table {[r for r, _ in operands]}')
    else:
        for (role, o), (_, t) in zip(xml_operands, operands):
            for key in OPERAND_KEYS:
                v = xml_value(o, key, OPERAND_DEFAULTS)
                if v != t[key]:
                    error(f'{name} {role} {key}: XML {v}, table {t[key]}')

    # The instructions of the form
    spec = form.get('encoding')
    if spec is None:
        error(f'{name}: no encoding')
        continue

    alternatives = sequences(spec)
    for positions in alternatives:
        for position in positions:
            for ins in position:
                if ins not in isa.instrs:
                    error(f'{name}: no instruction {ins}')
                elif name not in isa.instrs[ins].attrib.get('form', '').split():
                    error(f'{name}: {ins} does not list the form')

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

# Every form an instruction lists exists
for ins in isa.instrs.values():
    for f in ins.attrib.get('form', '').split():
        if f not in forms:
            error(f'{ins.name}: no form {f}')
    if 'form' not in ins.attrib:
        error(f'{ins.name}: no form')

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
        elif form not in ins.attrib.get('form', '').split():
            error(f'{form} {code}: decodes to {ins.name}, of another form')

for e in errors:
    print(e)
print('Fail.' if errors else 'Pass.')
sys.exit(1 if errors else 0)
