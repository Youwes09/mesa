# Copyright 2025 Valve Corporation
# Copyright 2020-2021 Collabora
# SPDX-License-Identifier: MIT

agx2_disasm_c = custom_target(
  input : ['gen-disasm.py', 'AGX2.xml'],
  output : 'agx2_disasm.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('isa.py'),
)

libagx2_disasm = static_library(
  'agx2_disasm',
  agx2_disasm_c,
  include_directories : [inc_include, inc_src],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'agx2_disasm',
    executable(
      'agx2_disasm_test',
      files('test/test-disassembler.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : [libagx2_disasm],
    ),
    suite : ['asahi'],
    args : files('test/cases.txt'),
  )

 test(
    'agx2_python_disasm',
    prog_python,
    args : files('test-disasm.py', 'test/cases.txt'),
    suite : ['asahi']
  )
endif

# Apple9 (G16, G17P)
agx3_pack_h = custom_target(
  input : ['gen-pack.py', 'AGX3.xml'],
  output : 'agx3_pack.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('isa.py'),
)

idep_agx3_pack_h = declare_dependency(
  sources : [agx3_pack_h],
  include_directories : include_directories('.'),
)

agx3_disasm_c = custom_target(
  input : ['gen-disasm.py', 'AGX3.xml'],
  output : 'agx3_disasm.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('isa.py'),
)

libagx3_disasm = static_library(
  'agx3_disasm',
  agx3_disasm_c,
  include_directories : [inc_include, inc_src],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'agx3_disasm',
    executable(
      'agx3_disasm_test',
      files('test/test-disassembler.c'),
      c_args : [c_msvc_compat_args, no_override_init_args,
                '-DDISASSEMBLE_INSTR=agx3_disassemble_instr'],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : [libagx3_disasm],
    ),
    suite : ['asahi'],
    args : files('test/agx3-cases.txt'),
  )

  test(
    'agx3_python_disasm',
    prog_python,
    args : [files('test-disasm.py', 'test/agx3-cases.txt'), '--isa',
            files('AGX3.xml')],
    suite : ['asahi']
  )
endif
