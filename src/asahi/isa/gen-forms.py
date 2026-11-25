# Copyright 2026 The Asahi Linux Contributors
# SPDX-License-Identifier: MIT

# Generates the compiler's table of encoding forms from the forms of an ISA
# description: their operand constraints and scheduling rules. The forms are
# enumerated in agx3_machine.h, in the order of the description.
#
# usage: gen-forms.py ISA.xml

import sys
import xml.etree.ElementTree as ET
from mako.template import Template
from mako import exceptions

root = ET.parse(sys.argv[1]).getroot()
forms = [f for f in root.iter('form') if f.get('packed') != 'false']
machine = root.find('machine')

FORM_DEFAULTS = {
    'allocator_safe': 'true', 'evidence': 'hardware', 'dependency': 'none',
    'retains': 'false', 'fixed_slot': '0', 'alternates': 'none',
    'prior': 'false', 'exempt': 'none', 'split': 'false',
}

OPERAND_DEFAULTS = {
    'files': 'gpr', 'widths': '32', 'min': '0', 'uniform_max': '0',
    'align': '2', 'flags': 'allocatable', 'evidence': 'hardware',
}

def attr(el, key, defaults):
    return el.get(key, defaults.get(key))

def flag_set(prefix, value):
    if value == 'none':
        return '0'
    return ' | '.join(f'{prefix}{x.upper()}' for x in value.split(','))

def slots(value):
    if value == 'none':
        return '0'
    return hex(sum(1 << int(x) for x in value.split(',')))

def boolean(value):
    assert value in ['true', 'false']
    return value

def operand(o):
    a = lambda key: attr(o, key, OPERAND_DEFAULTS)
    files = a('files').split(',')
    implicit = files == ['implicit']
    return {
        'role': f'AGX3_OPERAND_{o.get("role").upper()}',
        'files': flag_set('AGX3_FILE_', a('files')),
        'widths': flag_set('AGX3_WIDTH_', a('widths')),
        'min_index': '0xff' if implicit else a('min'),
        'max_index': '0xff' if implicit else o.get('max'),
        'uniform_max_index': a('uniform_max'),
        'alignment_halves': a('align'),
        'flags': flag_set('AGX3_OPERAND_', a('flags')),
        'evidence': f'AGX3_EVIDENCE_{a("evidence").upper()}',
    }

def form(f):
    a = lambda key: attr(f, key, FORM_DEFAULTS)
    return {
        'name': f'"{f.get("name")}"',
        'length': f.get('length'),
        'operand_count': str(len(f.findall('operand'))),
        'allocator_safe': boolean(a('allocator_safe')),
        'evidence': f'AGX3_EVIDENCE_{a("evidence").upper()}',
        'dependency_layout': f'AGX3_DEPENDENCY_{a("dependency").upper()}',
        'dependency_retains_sources': boolean(a('retains')),
        'fixed_producer_slot': a('fixed_slot'),
        'fixed_producer_alternates': slots(a('alternates')),
        'requires_prior_completion': boolean(a('prior')),
        'prior_completion_exempt': flag_set('AGX3_COMPLETION_', a('exempt')),
        'split_result': boolean(a('split')),
    }

source = """/*
 * Generated from ${xml} by gen-forms.py. Do not edit.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include "agx3_machine.h"

% for key in ['gpr_count', 'half_register_count', 'publication_count', 'uniform_count', 'fill_word_count', 'save_word_count']:
static_assert(AGX3_${key.upper()} == ${machine.get(key)}, "${key}");
% endfor

const struct agx3_machine agx3_machine = {
   .gpr_count = AGX3_GPR_COUNT,
   .half_register_count = AGX3_HALF_REGISTER_COUNT,
   .hardware_register_interlocks = ${machine.get('hardware_register_interlocks')},
   .software_waits = ${machine.get('software_waits')},
   .spilling_supported = ${machine.get('spilling_supported')},
   .occupancy_model = AGX3_OCCUPANCY_UNMEASURED,
};

% for i, f in enumerate(forms):
static_assert(AGX3_ENC_${f.get('name').upper()} == ${i}, "form order");
% endfor
static_assert(AGX3_ENC_COUNT == ${len(forms)}, "form count");

static const struct agx3_encoding_info encodings[] = {
% for f in forms:
   {
% for k, v in form(f).items():
      .${k} = ${v},
% endfor
% if f.findall('operand'):
      .operands = {
% for o in f.findall('operand'):
         {
% for k, v in operand(o).items():
            .${k} = ${v},
% endfor
         },
% endfor
      },
% endif
   },
% endfor
};

const struct agx3_encoding_info *
agx3_encoding_info(enum agx3_encoding encoding)
{
   assert(encoding < AGX3_ENC_COUNT);
   return &encodings[encoding];
}
"""

try:
    print(Template(source).render(
        xml=sys.argv[1].split('/')[-1], forms=forms, machine=machine,
        form=form, operand=operand))
except:
    print(exceptions.text_error_template().render())
