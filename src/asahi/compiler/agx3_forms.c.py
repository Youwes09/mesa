template = """/*
 * Generated from agx3_forms.py by agx3_forms.c.py. Do not edit.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include "agx3_machine.h"

% for key in ['gpr_count', 'half_register_count', 'publication_count', 'uniform_count', 'fill_word_count', 'save_word_count']:
static_assert(AGX3_${key.upper()} == ${machine[key]}, "${key}");
% endfor

const struct agx3_machine agx3_machine = {
   .gpr_count = AGX3_GPR_COUNT,
   .half_register_count = AGX3_HALF_REGISTER_COUNT,
   .hardware_register_interlocks = ${boolean(machine['hardware_register_interlocks'])},
   .software_waits = ${boolean(machine['software_waits'])},
   .spilling_supported = ${boolean(machine['spilling_supported'])},
   .occupancy_model = AGX3_OCCUPANCY_UNMEASURED,
};

static const struct agx3_encoding_info encodings[] = {
% for f in forms:
   {
      .name = "${f.name}",
      .length = ${f.length},
      .operand_count = ${len(f.operands)},
      .allocator_safe = ${boolean(f.allocator_safe)},
      .evidence = AGX3_EVIDENCE_${f.evidence.upper()},
      .dependency_layout = AGX3_DEPENDENCY_${f.dependency.upper()},
      .dependency_retains_sources = ${boolean(f.retains)},
      .fixed_producer_slot = ${f.fixed_slot},
      .fixed_producer_alternates = ${slots(f.alternates)},
      .requires_prior_completion = ${boolean(f.prior)},
      .prior_completion_exempt = ${flag_set('AGX3_COMPLETION_', f.exempt)},
      .split_result = ${boolean(f.split)},
% if f.operands:
      .operands = {
% for o in f.operands:
<% implicit = o.files == ['implicit'] %>\\
         {
            .role = AGX3_OPERAND_${o.role.upper()},
            .files = ${flag_set('AGX3_FILE_', o.files)},
            .widths = ${flag_set('AGX3_WIDTH_', o.widths)},
            .min_index = ${'0xff' if implicit else o.min},
            .max_index = ${'0xff' if implicit else o.max},
            .uniform_max_index = ${o.uniform_max},
            .alignment_halves = ${o.align},
            .flags = ${flag_set('AGX3_OPERAND_', o.flags)},
            .evidence = AGX3_EVIDENCE_${o.evidence.upper()},
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

from mako.template import Template
from agx3_forms import forms, machine

def boolean(x):
   return 'true' if x else 'false'

def flag_set(prefix, values):
   return ' | '.join(f'{prefix}{str(x).upper()}' for x in values) or '0'

def slots(values):
   return hex(sum(1 << x for x in values)) if values else '0'

print(Template(template).render(forms=[f for f in forms if f.packed],
                                machine=machine, boolean=boolean,
                                flag_set=flag_set, slots=slots))
