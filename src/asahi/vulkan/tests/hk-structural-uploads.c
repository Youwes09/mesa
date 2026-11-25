/* SPDX-License-Identifier: MIT
 * Exercise actual render/view ownership with host-backed command BOs. */
#include "../hk_cmd_buffer.h"
#include "../hk_device.h"
#include "../hk_image.h"
#include "../hk_image_view.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_usc_descriptor.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct agx_bo *bos[128];
static unsigned allocations, live_bos, view_cases;
static bool fail_alloc;
VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int,
                            const char *, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   CHECK(fail_alloc);
   return error;
}
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
void __wrap_agx_bo_cache_evict_all(struct agx_device *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   if (fail_alloc)
      return NULL;
   CHECK(allocations + 1 < ARRAY_SIZE(bos));
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = (++allocations) * (uint64_t)HK_CMD_BO_SIZE;
   bos[allocations] = bo;
   ++live_bos;
   return bo;
}
void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   CHECK(bo->refcnt > 0);
   if (--bo->refcnt)
      return;
   bos[bo->va->addr / HK_CMD_BO_SIZE] = NULL;
   free(bo->_map);
   free(bo->va);
   free(bo);
   --live_bos;
}
void
__wrap_agx_bo_cache_evict_all(struct agx_device *dev)
{
}
static void *
mapped(uint64_t gpu)
{
   for (unsigned i = 1; i <= allocations; ++i)
      if (bos[i] && gpu >= bos[i]->va->addr &&
          gpu < bos[i]->va->addr + bos[i]->size)
         return (uint8_t *)bos[i]->_map + gpu - bos[i]->va->addr;
   CHECK(false);
   return NULL;
}

static void
check_view(struct hk_device *dev, VkFormat format, unsigned samples,
           unsigned level, unsigned layers, bool linear, bool compressed)
{
   const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {127, 65, 1},
      .mipLevels = samples > 1 || linear ? 1 : 4,
      .arrayLayers = layers + 1,
      .samples = samples,
      .tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   struct hk_image image = {0};
   vk_image_init(&dev->vk, &image.vk, &ci);
   image.plane_count = 1;
   image.planes[0].addr = UINT64_C(0x2000000000);
   struct ail_layout *layout = &image.planes[0].layout;
   *layout = (struct ail_layout){
      .width_px = ci.extent.width,
      .height_px = ci.extent.height,
      .depth_px = ci.arrayLayers,
      .levels = ci.mipLevels,
      .sample_count_sa = samples,
      .format = vk_format_to_pipe_format(format),
      .tiling = linear ? AIL_TILING_LINEAR : AIL_TILING_GPU,
      .renderable = true,
      .compressed = compressed,
   };
   ail_make_miptree(layout);
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = hk_image_to_handle(&image),
      .viewType =
         layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 1, layers},
   };
   VkImageView handle;
   CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &handle) ==
         VK_SUCCESS);
   struct hk_image_view *view = hk_image_view_from_handle(handle);
   for (unsigned layered = 0; layered < 2; ++layered) {
      const struct hk_render_descriptors *owned =
         &view->render_descriptors[layered];
      agx_unpack(stderr,
                 layered ? &view->planes[0].layered_background
                         : &view->planes[0].background,
                 TEXTURE, legacy);
      CHECK(owned->dimensions[0] == u_minify(legacy.width, legacy.first_level));
      CHECK(owned->dimensions[1] ==
            u_minify(legacy.height, legacy.first_level));
      CHECK(owned->dimensions[2] == legacy.depth);
      CHECK(owned->dimensions[3] == 1);
      CHECK(owned->array ==
            (legacy.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
             legacy.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED));
      legacy.srgb = legacy.srgb_2_channel = false;
      struct agx_texture_packed packed;
      agx_pack(&packed, TEXTURE, cfg) {
         cfg = legacy;
      }
      uint8_t expected[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      bool valid = agx_translate_usc_texture(expected, &packed);
      CHECK(owned->texture_valid == valid);
      if (valid)
         CHECK(!memcmp(owned->texture, expected, sizeof(expected)));
      valid = hk_image_view_eot(view, layered, false, expected);
      CHECK(owned->store_valid == valid);
      if (valid)
         CHECK(!memcmp(owned->store, expected, sizeof(expected)));
   }
   hk_DestroyImageView(hk_device_to_handle(dev), handle, NULL);
   vk_image_finish(&image.vk);
   ++view_cases;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers =
      vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->dev.chip = AGX_CHIP_G16G;
   hk_device_init_usc_heap(dev);
   const VkFormat formats[] = {
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB,
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_R32G32B32A32_UINT};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned layers = 1; layers <= 3; layers += 2) {
         for (unsigned samples = 1; samples <= 4; samples *= 2) {
            unsigned levels = samples == 1 ? 4 : 1;
            for (unsigned l = 0; l < levels; ++l)
               check_view(dev, formats[f], samples, l, layers, false, false);
         }
         check_view(dev, formats[f], 1, 0, layers, true, false);
      }
   }
   for (unsigned samples = 1; samples <= 4; samples *= 2)
      check_view(dev, VK_FORMAT_R8G8B8A8_UNORM, samples, 0, 3, false, true);

   hk_device_finish_usc_heap(dev);
   CHECK(live_bos == 0);
   free(pdev);
   free(dev);
   printf("PASS %u render view cases\n", view_cases);
   return 0;
}
