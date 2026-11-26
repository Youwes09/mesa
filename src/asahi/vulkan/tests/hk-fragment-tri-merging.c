/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Triangle merging shades adjacent strip and fan triangles on one plane, which
 * is inexact away from the pixel centre. Compile real fragment shaders (BO
 * allocation replaced with host memory) and check that off-centre
 * interpolation disables merging, that pixel-centre interpolation keeps it,
 * and that the choice survives serialization.
 */
#include "../hk_shader.c"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)
#include "hk-test-usc.h"

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   static unsigned allocations;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = dev->shader_base + ++allocations * 0x100000;
   return bo;
}

void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   ++bo->refcnt;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo || --bo->refcnt)
      return;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

static nir_shader *
fragment_source(bool centroid)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, agx_nir_options_for_isa(AGX_SHADER_ISA_APPLE9),
      "tri merging fragment");
   /* An application shader: hk does not gate internal ones. */
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;
   nir_def *bary = centroid ? nir_load_barycentric_centroid(
                                 &b, 32, .interp_mode = INTERP_MODE_SMOOTH)
                            : nir_load_barycentric_pixel(
                                 &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *value = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_store_output(
      &b, value, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static bool
roundtrip_disables(struct hk_device *dev, struct hk_api_shader *fs)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(hk_api_shader_serialize(&dev->vk, &fs->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *result = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &result) ==
         VK_SUCCESS);
   blob_finish(&blob);
   struct hk_api_shader *copy = container_of(result, struct hk_api_shader, vk);
   bool disable = copy->variants[0].info.fs.disable_tri_merging;
   copy->vk.ops->destroy(&dev->vk, &copy->vk, NULL);
   return disable;
}

/* Returns whether the compiled fragment shader disables triangle merging. */
static bool
disables_merging(struct hk_device *dev, bool centroid)
{
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_FRAGMENT,
      .nir = fragment_source(centroid),
      .robustness = &vk_robustness_disabled,
   };
   struct hk_api_shader *fs = NULL;
   CHECK(hk_compile_shader(dev, &info, NULL, NULL, NULL, &fs) == VK_SUCCESS);
   bool disable = fs->variants[0].info.fs.disable_tri_merging;
   CHECK(roundtrip_disables(dev, fs) == disable);
   fs->vk.ops->destroy(&dev->vk, &fs->vk, NULL);
   return disable;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.params.gpu_generation = 16;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);

   /* Off-centre interpolation keeps triangles apart */
   CHECK(disables_merging(dev, true));
   /* Pixel-centre interpolation merges */
   CHECK(!disables_merging(dev, false));

   simple_mtx_destroy(&pdev->debug_compile_lock);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   puts("off-centre interpolation disables triangle merging");
   return 0;
}
