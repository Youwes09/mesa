template = """/*
 * Generated from agx3_forms.py by agx3_forms.h.py. Do not edit.
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX3_FORMS_H
#define AGX3_FORMS_H

/*
 * These are encoding forms, not NIR operations. An operation may have both a
 * compact low-register form and a longer form with a different source layout.
 * The allocator must choose a form before the packer commits physical
 * registers.
 */
enum agx3_encoding {
% for f in forms:
   AGX3_ENC_${f.name.upper()},
% endfor
   AGX3_ENC_COUNT,

   /* IR pseudo-instructions have no encoding */
   AGX3_ENC_PSEUDO = 0xff,
};

#endif
"""

from mako.template import Template
from agx3_forms import forms

print(Template(template).render(forms=[f for f in forms if f.packed]))
