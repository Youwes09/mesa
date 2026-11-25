/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Prints the compiler's table of forms for test-agx3-forms.py to check */

#include <stdio.h>
#include "asahi/compiler/agx3_machine.h"
#include "util/macros.h"

static const char *evidence[] = {
   [AGX3_EVIDENCE_UNRESOLVED] = "unresolved",
   [AGX3_EVIDENCE_BYTE_DIFF] = "byte_diff",
   [AGX3_EVIDENCE_HARDWARE] = "hardware",
};

static const char *dependency[] = {
   [AGX3_DEPENDENCY_NONE] = "none",
   [AGX3_DEPENDENCY_INDEX_45_47] = "index_45_47",
   [AGX3_DEPENDENCY_INDEX_61_63] = "index_61_63",
   [AGX3_DEPENDENCY_MASK_12_17] = "mask_12_17",
   [AGX3_DEPENDENCY_MASK_45_47_61_63] = "mask_45_47_61_63",
   [AGX3_DEPENDENCY_MASK_61_63_77_79] = "mask_61_63_77_79",
   [AGX3_DEPENDENCY_MASK_61_63_93_95] = "mask_61_63_93_95",
};

static const char *roles[] = {
   [AGX3_OPERAND_DEST] = "dest",
   [AGX3_OPERAND_SRC0] = "src0",
   [AGX3_OPERAND_SRC1] = "src1",
   [AGX3_OPERAND_SRC2] = "src2",
   [AGX3_OPERAND_SRC3] = "src3",
   [AGX3_OPERAND_SRC4] = "src4",
   [AGX3_OPERAND_SRC5] = "src5",
   [AGX3_OPERAND_SRC6] = "src6",
   [AGX3_OPERAND_SRC7] = "src7",
   [AGX3_OPERAND_SRC8] = "src8",
   [AGX3_OPERAND_SRC9] = "src9",
   [AGX3_OPERAND_INDEX] = "index",
   [AGX3_OPERAND_STORE_DATA] = "store_data",
   [AGX3_OPERAND_ATOMIC_DATA] = "atomic_data",
};

static const char *files[] = {"gpr", "uniform", "immediate", "implicit"};

static const char *widths[] = {"16", "32", "64"};

static const char *operand_flags[] = {
   "allocatable", "hard_low", "compact_preferred", "scattered",
   "implicit",    "uniform_alternative", "clobber", "publication",
};

static const char *completion[] = {"memory", "publication", "other"};

/* Space-separated names of the set bits, or "none" */
static void
print_set(const char *key, unsigned mask, const char **names, unsigned count)
{
   printf(" %s=", key);

   if (!mask) {
      printf("none");
      return;
   }

   const char *sep = "";
   for (unsigned i = 0; i < count; ++i) {
      if (mask & (1u << i)) {
         printf("%s%s", sep, names[i]);
         sep = ",";
      }
   }

   if (mask >> count)
      printf("%sunknown%#x", sep, mask >> count);
}

/* Slots 1-6 as a set */
static void
print_slots(const char *key, unsigned mask)
{
   static const char *slots[] = {"0", "1", "2", "3", "4", "5", "6", "7"};
   print_set(key, mask, slots, ARRAY_SIZE(slots));
}

int
main(void)
{
   const struct agx3_machine *m = &agx3_machine;
   printf("machine gpr_count=%u half_register_count=%u "
          "hardware_register_interlocks=%u software_waits=%u "
          "spilling_supported=%u publication_count=%u uniform_count=%u "
          "fill_word_count=%u save_word_count=%u\n",
          m->gpr_count, m->half_register_count, m->hardware_register_interlocks,
          m->software_waits, m->spilling_supported, AGX3_PUBLICATION_COUNT,
          AGX3_UNIFORM_COUNT, AGX3_FILL_WORD_COUNT, AGX3_SAVE_WORD_COUNT);

   for (unsigned e = 0; e < AGX3_ENC_COUNT; ++e) {
      const struct agx3_encoding_info *info = agx3_encoding_info(e);

      printf("form %s length=%u allocator_safe=%u evidence=%s dependency=%s "
             "retains=%u",
             info->name, info->length, info->allocator_safe,
             evidence[info->evidence], dependency[info->dependency_layout],
             info->dependency_retains_sources);
      printf(" fixed_slot=%u", info->fixed_producer_slot);
      print_slots("alternates", info->fixed_producer_alternates);
      printf(" prior=%u", info->requires_prior_completion);
      print_set("exempt", info->prior_completion_exempt, completion,
                ARRAY_SIZE(completion));
      printf(" split=%u\n", info->split_result);

      for (unsigned i = 0; i < info->operand_count; ++i) {
         const struct agx3_operand_constraint *op = &info->operands[i];

         printf("operand %s", roles[op->role]);
         print_set("files", op->files, files, ARRAY_SIZE(files));
         print_set("widths", op->widths, widths, ARRAY_SIZE(widths));
         printf(" min=%u max=%u uniform_max=%u align=%u", op->min_index,
                op->max_index, op->uniform_max_index, op->alignment_halves);
         print_set("flags", op->flags, operand_flags,
                   ARRAY_SIZE(operand_flags));
         printf(" evidence=%s\n", evidence[op->evidence]);
      }
   }

   return 0;
}
