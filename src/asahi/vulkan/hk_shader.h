/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_apple9_profile.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_draw_table.h"
#include "asahi/lib/agx_launch_program.h"
#include "poly/nir/poly_nir.h"
#include "util/macros.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "hk_descriptor_memory.h"

#include "asahi/lib/agx_cdm.h"
#include "asahi/lib/agx_draw_table.h"
#include "asahi/lib/agx_launch_program.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_private.h"

#include "shader_enums.h"
#include "vk_pipeline_cache.h"

#include "nir_defines.h"

#include "vk_shader.h"

struct hk_physical_device;
struct hk_pipeline_compilation_ctx;
struct vk_descriptor_set_layout;
struct vk_graphics_pipeline_state;
struct vk_pipeline_cache;
struct vk_pipeline_layout;
struct vk_pipeline_robustness_state;
struct vk_shader_module;

struct hk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct hk_tess_info) == 4, "packed");

static struct hk_tess_info
hk_tess_info_merge(struct hk_tess_info a, struct hk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct hk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

/* Texture operations whose descriptors may change after binding, or be
 * partially bound, must read them at execution time on G16. */
#define HK_TEX_DESCRIPTOR_HANDLE (1u << 30)

/* On a nir_texop_lod_bias query: load the sampler's explicit-LOD nudge
 * instead of its LOD bias, see hk_lower_d3d_txl_rounding. */
#define HK_TEX_TXL_NUDGE (1u << 29)

/* Mesh shaders run as compute kernels. Each workgroup writes a record of
 * group_stride_B bytes: its vertex and primitive counts, padded to 16 bytes,
 * then max_vertices vertices and max_primitives primitives, each the varying
 * slots in the respective mask in slot order. Slots have the words of a vec4,
 * except for scalar builtins and the primitive indices, which have their
 * primitive's vertex count. The primitive slots include the primitive indices
 * and cull flag.
 */
struct hk_mesh_info {
   uint64_t vertex_slots;
   uint64_t primitive_slots;
   uint32_t group_stride_B;
   uint16_t max_vertices, max_primitives;
   uint8_t prim;
   uint8_t _pad[7];
};

/* Fragment shaders read user per-primitive varyings from the output records,
 * at the byte offset of the primitive, passed in this flat varying.
 */
#define HK_MESH_PRIMITIVE_SLOT VARYING_SLOT_VAR31


static inline bool
hk_mesh_varying_in_memory(unsigned slot)
{
   return slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_VAR31;
}

static inline unsigned
hk_mesh_slot_words(const struct hk_mesh_info *info, unsigned slot)
{
   switch (slot) {
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_PRIMITIVE_SHADING_RATE:
   case VARYING_SLOT_CULL_PRIMITIVE:
      return 1;
   case VARYING_SLOT_PRIMITIVE_INDICES:
      return mesa_vertices_per_prim(info->prim);
   default:
      return 4;
   }
}

/* Offset of a slot within a vertex or primitive, or their size with
 * slot = 64.
 */
static inline unsigned
hk_mesh_slot_offset_B(const struct hk_mesh_info *info, uint64_t mask,
                      unsigned slot)
{
   unsigned offset_B = 0;
   u_foreach_bit64(s, mask & BITFIELD64_MASK(slot))
      offset_B += hk_mesh_slot_words(info, s) * 4;
   return offset_B;
}

static inline unsigned
hk_mesh_primitives_offset_B(const struct hk_mesh_info *info)
{
   return 16 + info->max_vertices *
                  hk_mesh_slot_offset_B(info, info->vertex_slots, 64);
}

void hk_mesh_gather_info(const nir_shader *nir, struct hk_mesh_info *info);
void hk_nir_lower_mesh_main(nir_shader *nir, const struct hk_mesh_info *info);
void hk_nir_lower_task_main(nir_shader *nir);
uint32_t hk_nir_task_mesh_bound(nir_shader *nir);
/* Draw-time mesh record rasterizer, built with hk_meta_shader */
struct hk_mesh_rast_key {
   struct hk_mesh_info info;
   uint8_t clip_distance_array_size, cull_distance_array_size;
};

void hk_nir_mesh_rast(struct nir_builder *b, const void *key);

struct hk_shader_info {
   union {
      struct {
         uint32_t attribs_read;
         BITSET_DECLARE(attrib_components_read, AGX_MAX_ATTRIBS * 4);
         bool use_prolog;
         uint8_t _pad[7];
      } vs;

      struct {
         struct agx_interp_info interp;
         struct agx_fs_epilog_link_info epilog_key;

         bool reads_sample_mask;
         bool post_depth_coverage;
         bool uses_sample_shading;
         bool early_fragment_tests;
         bool writes_memory;

         /* Apple9: triangle merging is off for off-centre interpolation */
         bool pixel_interlock;
         bool disable_tri_merging;

         /* Apple9: export depth for the epilog in Z/S feedback loop draws */
         bool dynamic_late_zs;
      } fs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct hk_tess_info info;
      } tess;

      struct poly_gs_info gs;

      struct hk_mesh_info mesh;

      struct {
         uint32_t payload_B;
         /* Per task workgroup; UINT32_MAX means not statically bounded. */
         uint32_t max_mesh_groups;
      } task;

      /* Used to initialize the union for other stages */
      uint8_t _pad[32];
   };

   struct agx_unlinked_uvs_layout uvs;

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   mesa_shader_stage stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t set_count;

   /* XXX: is there a less goofy way to do this? I really don't want dynamic
    * allocation here.
    */
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[64];
};

/*
 * Hash table keys for fast-linked shader variants. These contain the entire
 * prolog/epilog key so we only do 1 hash table lookup instead of 2 in the
 * general case where the linked shader is already ready.
 */
struct hk_fast_link_key_vs {
   struct agx_vs_prolog_key prolog;
};

struct hk_fast_link_key_fs {
   unsigned nr_samples_shaded;
   struct agx_fs_prolog_key prolog;
   struct agx_fs_epilog_key epilog;
};

struct hk_shader {
#ifndef NDEBUG
   /* Final-NIR validation uses the owning API shader's upload mask. */
   const struct hk_api_shader *debug_owner;
#endif
   struct agx_shader_part b;

   struct hk_shader_info info;
   struct agx_fragment_face_2_packed frag_face;
   struct agx_counts_packed counts;

   const void *code_ptr;
   uint32_t code_size;

   const void *data_ptr;
   uint32_t data_size;

   /* Preamble and constant data on Apple8, constant data on Apple9 */
   struct agx_bo *bo;

   /* Apple9 compute launch template, patched per dispatch, or NULL */
   struct agx_launch_template *stage_launch_cache;

   /* Cache of fast linked variants */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
   } linked;

   /* If there's only a single possibly linked variant, direct pointer. TODO:
    * Union with the cache to save some space?
    */
   struct hk_linked_shader *only_linked;

   /* Address to the uploaded preamble section. Preambles are uploaded
    * separately from fast-linked main shaders.
    */
   uint64_t preamble_addr;

   /* Address of the start of the shader data section */
   uint64_t data_addr;
};

enum hk_vs_variant {
   /* Hardware vertex shader, when next stage is fragment */
   HK_VS_VARIANT_HW,

   /* Hardware compute shader, when next is geometry/tessellation */
   HK_VS_VARIANT_SW,

   HK_VS_VARIANTS,
};

enum hk_gs_variant {
   /* Hardware vertex shader used for rasterization */
   HK_GS_VARIANT_RAST,

   /* Main compute shader */
   HK_GS_VARIANT_MAIN,

   /* Count compute shader */
   HK_GS_VARIANT_COUNT,

   /* Pre-GS compute shader */
   HK_GS_VARIANT_PRE,

   HK_GS_VARIANTS,
};

/* clang-format off */
static const char *hk_gs_variant_name[] = {
   [HK_GS_VARIANT_RAST] = "Rasterization",
   [HK_GS_VARIANT_MAIN] = "Main",
   [HK_GS_VARIANT_COUNT] = "Count",
   [HK_GS_VARIANT_PRE] = "Pre-GS",
};
/* clang-format on */

static inline unsigned
hk_num_variants(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      return HK_VS_VARIANTS;

   case MESA_SHADER_GEOMETRY:
      return HK_GS_VARIANTS;

   default:
      return 1;
   }
}

/*
 * An hk_api shader maps 1:1 to a VkShader object. An hk_api_shader may contain
 * multiple hardware hk_shader's, built at shader compile time. This complexity
 * is required to efficiently implement the legacy geometry pipeline.
 */
enum hk_pre_raster_reads {
   /* Sampled, storage or input-attachment images: may alias attachments. */
   HK_READS_IMAGES = BITFIELD_BIT(0),
   /* Buffers of any kind, vertex attributes, descriptors and indirect
    * arguments: only memory-writing renders can produce them. */
   HK_READS_BUFFERS = BITFIELD_BIT(1),
   /* Not classified: keep the tiling wait. */
   HK_READS_UNKNOWN = BITFIELD_BIT(2),
   /* hk-internal pre-rasterization work (meta, transform feedback,
    * tessellation, geometry): keep the tiling wait. */
   HK_READS_INTERNAL = BITFIELD_BIT(3),
   /* A pre-rasterization stage stores to memory or runs atomics
    * (vertexPipelineStoresAndAtomics): keep the tiling wait. */
   HK_READS_WRITES = BITFIELD_BIT(5),
};

uint8_t hk_nir_pre_raster_reads(const nir_shader *nir);
bool hk_nir_reads_draw_id(const nir_shader *nir);
uint64_t hk_nir_push_constant_mask(const nir_shader *nir);
#ifndef NDEBUG
uint64_t hk_nir_final_push_constant_mask(const nir_shader *nir);
#endif

struct hk_api_shader {
   struct vk_shader vk;

   /* Is this an internal passthrough geometry shader? */
   bool is_passthrough;

   /* The program stores to memory or runs atomics (any variant). */
   bool writes_memory;

   /* Memory a pre-rasterization program reads (enum hk_pre_raster_reads):
    * images or buffers. Decides whether its render's tiling must wait for
    * earlier renders' writes (hk_queue). */
   uint8_t pre_raster_reads;

   /* The program reads the draw index (gl_DrawID / DrawIndex), which hk
    * passes in the root descriptor table. Draws whose bound stages do not
    * read it leave the root unchanged, so consecutive draws of a multi-draw
    * share one root upload. */
   bool reads_draw_id;

   /* API push-constant dwords read by this stage. Collected before Apple9
    * root lowering; an indirect or unrecognized root access keeps all words.
    * This is upload metadata and does not change the compiled program. */
   uint64_t push_constant_mask;

   /* Immutable after compilation/cache reload; avoids walking variants at
    * every shader bind, including shaders with no constant data. */
   uint64_t constant_data_addr;

   struct hk_shader variants[];
};

bool hk_lower_bounded_global_loads(nir_shader *nir, bool soft_fault,
                                   bool per_component);

#define hk_foreach_variant(api_shader, var)                                    \
   for (struct hk_shader *var = api_shader->variants;                          \
        var < api_shader->variants + hk_num_variants(api_shader->vk.stage);    \
        ++var)

static const char *
hk_variant_name(struct hk_api_shader *obj, struct hk_shader *variant)
{
   unsigned i = variant - obj->variants;
   assert(i < hk_num_variants(obj->vk.stage));

   if (hk_num_variants(obj->vk.stage) == 1) {
      return NULL;
   } else if (obj->vk.stage == MESA_SHADER_GEOMETRY) {
      assert(i < ARRAY_SIZE(hk_gs_variant_name));
      return hk_gs_variant_name[i];
   } else {
      assert(i < 2);
      return i == HK_VS_VARIANT_SW ? "Software" : "Hardware";
   }
}

static struct hk_shader *
hk_only_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   assert(hk_num_variants(obj->vk.stage) == 1);
   return &obj->variants[0];
}

/* Large constants are extracted before variants are cloned or generated.
 * Generated variants may not use them, but all original variants retain the
 * same data layout. Any uploaded copy can serve the API stage's root slot
 * without a draw-path rebind.
 */
static inline void
hk_api_shader_init_constant_data(struct hk_api_shader *shader)
{
   const struct hk_shader *data = NULL;
   for (unsigned v = 0; v < hk_num_variants(shader->vk.stage); ++v) {
      const struct hk_shader *s = &shader->variants[v];
      if (!s->data_size)
         continue;
      assert(!data || (data->data_size == s->data_size &&
                       !memcmp(data->data_ptr, s->data_ptr, s->data_size)));
      if (!data)
         data = s;
   }
   shader->constant_data_addr = data ? data->data_addr : 0;
}

static inline uint64_t
hk_shader_constant_data(const struct hk_api_shader *shader)
{
   return shader ? shader->constant_data_addr : 0;
}

static struct hk_shader *
hk_any_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   return &obj->variants[0];
}

static struct hk_shader *
hk_main_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_MAIN];
}

static struct hk_shader *
hk_count_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_COUNT];
}

static struct hk_shader *
hk_pre_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_PRE];
}

#define HK_MAX_LINKED_USC_SIZE                                                 \
   (AGX_USC_PRESHADER_LENGTH + AGX_USC_FRAGMENT_PROPERTIES_LENGTH +            \
    AGX_USC_REGISTERS_LENGTH + AGX_USC_SHADER_LENGTH + AGX_USC_SHARED_LENGTH + \
    AGX_USC_SAMPLER_LENGTH + (AGX_USC_UNIFORM_LENGTH * 9))

struct hk_linked_shader {
   struct agx_linked_shader b;

   /* The program as launched draws see it, see hk_cmd_draw.c */
   struct agx_draw_stage draw_stages;
   struct agx_stage_launch_cache stage_launch_cache;

   /* True if the VS prolog uses software indexing, either for geom/tess or
    * adjacency primitives.
    */
   bool sw_indexing;

   /* Distinct from hk_shader::counts due to addition of cf_binding_count, which
    * is delayed since it depends on cull distance.
    */
   struct agx_fragment_shader_word_0_packed fs_counts;

   /* Baked USC words to bind this linked shader */
   struct {
      uint8_t data[HK_MAX_LINKED_USC_SIZE];
      size_t size;
   } usc;
};

struct hk_linked_shader *hk_fast_link(struct hk_device *dev, bool fragment,
                                      struct hk_shader *main,
                                      struct agx_shader_part *prolog,
                                      struct agx_shader_part *epilog,
                                      unsigned nr_samples_shaded);

extern const struct vk_device_shader_ops hk_device_shader_ops;

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev);

static inline nir_address_format
hk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage,
                        unsigned nr_vbos);

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts);

VkResult hk_compile_shader(struct hk_device *dev,
                           struct vk_shader_compile_info *info,
                           const struct vk_graphics_pipeline_state *state,
                           const struct vk_features *features,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_api_shader **shader_out);

void hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev,
                                nir_shader *nir);

void hk_api_shader_destroy(struct vk_device *vk_dev,
                           struct vk_shader *vk_shader,
                           const VkAllocationCallbacks *pAllocator);

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs);

struct hk_api_shader *hk_meta_shader(struct hk_device *dev,
                                     hk_internal_builder_t builder, void *data,
                                     size_t data_size);

struct agx_shader_part *hk_get_prolog_epilog(struct hk_device *dev, void *data,
                                             size_t data_size,
                                             hk_internal_builder_t builder,
                                             bool preprocess_nir, bool stop,
                                             unsigned cf_base);

struct nir_def *hk_nir_load_root(struct nir_builder *b, unsigned num_components,
                                 unsigned bit_size, unsigned offset);
