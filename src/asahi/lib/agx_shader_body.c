/* SPDX-License-Identifier: MIT */
#include "agx_shader_body.h"
#include "asahi/compiler/agx3_emit.h"
#include "util/u_atomic.h"
#include "util/u_call_once.h"
#include "util/vma.h"
#include "agx_apple9_layout.h"
#include "agx_device.h"

static inline void
put_u16(void *ptr, uint16_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline void
put_u32(void *ptr, uint32_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

void
agx_shader_body_init_bo(struct agx_bo *bo)
{
   bo->shader_body = (struct agx_shader_body){
      .bo = bo,
      .addr = bo->va->addr,
   };
}

void
agx_shader_body_reference(struct agx_shader_body *body)
{
   if (body->destroy) {
      assert(p_atomic_read(&body->refcnt) > 0);
      p_atomic_inc(&body->refcnt);
   } else {
      agx_bo_reference(body->bo);
   }
}

void
agx_shader_body_finish(struct agx_device *dev, struct agx_shader_body *body)
{
   assert(!body->entry_pins);
   if (!body->entry_offset && !body->entry_fixed)
      return;

   simple_mtx_lock(&dev->entry_lock);
   if (body->entry_offset) {
      util_vma_heap_free(&dev->entry_heap,
                         body->entry_offset - AGX_ENTRY_CODE_OFFSET,
                         AGX_ENTRY_BLOCK_SIZE);
      body->entry_offset = 0;
      list_del(&body->entry_link);
   }
   body->entry_fixed = false;
   simple_mtx_unlock(&dev->entry_lock);
}

void
agx_shader_body_unreference(struct agx_device *dev,
                            struct agx_shader_body *body)
{
   if (!body)
      return;
   if (!body->destroy) {
      agx_bo_unreference(dev, body->bo);
      return;
   }

   assert(p_atomic_read(&body->refcnt) > 0);
   if (p_atomic_dec_zero(&body->refcnt)) {
      agx_shader_body_finish(dev, body);
      /* Storage release may take the BO lock. Never nest it inside the entry
       * lock, which whole-BO destruction takes in the opposite order. */
      body->destroy(dev, body);
   }
}

/* Constants of compute entries, built once */
static uint8_t compute_entry_constants[0x40];
static void build_constant_program(uint8_t *out, unsigned slots);
static util_once_flag compute_entry_once = UTIL_ONCE_FLAG_INIT;

static void
init_compute_entry(void)
{
   build_constant_program(compute_entry_constants, 30);
}

static void
fill_helper_table(uint8_t *image, unsigned base, unsigned slots)
{
   assert(slots > 0 && slots <= 10);
   for (unsigned index = 0; index < slots; ++index) {
      uint8_t *record = image + base + (index * 0x10);
      memset(record, 0, 0x10);
      record[0] = 0x0f;
      record[2] = 0x54;
      record[3] = (slots - index) * 0x10;
      record[10] = record[12] = record[14] = 0x06;
   }

   const uint8_t terminal[0x10] = {
      0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
      0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
   };
   memcpy(image + base + slots * 0x10, terminal, sizeof(terminal));
   memcpy(image + base + (slots + 1) * 0x10, terminal, sizeof(terminal));
}

static void
build_constant_program(uint8_t *out, unsigned slots)
{
   assert(slots <= 30);
   memset(out, 0, 0x40);
   put_u32(out, 0x0e);
   for (unsigned index = 0; index < slots; ++index)
      put_u16(out + 4 + index * 2, 0x0006);
}

static bool
build_body_entry(uint8_t *out, uint64_t entry, uint64_t body)
{
   int64_t displacement = (int64_t)body - (int64_t)entry;
   /* Both allocations live within the same 4-GiB USC heap. Wider virtual
    * placement is outside the current entry ABI's validated domain. */
   if ((entry | body) & 1 || displacement < -(int64_t)UINT32_MAX ||
       displacement > UINT32_MAX)
      return false;
   if (!agx3_emit_branch(out, true, displacement))
      return false;
   put_u32(out + 10, 0x0e); /* No active lanes: terminate */
   return true;
}

void
agx_init_shader_entries(struct agx_device *dev)
{
   uint8_t header[AGX_ENTRY_HEADER_SIZE];
   memset(header, 0, AGX_ENTRY_HEADER_SIZE);
   put_u32(header, AGX_ENTRY_HEADER_SIZE);
   for (unsigned offset = 0x40; offset < AGX_ENTRY_HEADER_SIZE; offset += 2)
      put_u16(header + offset, 0x0006);
   fill_helper_table(header, 0x100, 10);
   fill_helper_table(header, 0x200, 10);
   agx_shader_entry_write(dev, 0, header, AGX_ENTRY_HEADER_SIZE);
}

/* The launch target addresses entries in a few megabytes, room for about
 * 43k shader bodies. Draw-time variants can exceed that over a session, so
 * entries are only fixed while a recording that may encode them is alive:
 * the device's pin hook ties the body to the recording's lifetime. When the
 * arena is full, free the least recently pinned unpinned entries. No live
 * recording, and hence no pending batch, encodes them; their bodies get a new
 * entry on their next use. Launch records cached by content cannot alias a
 * recycled entry, since they also encode the body's preamble address.
 */
static void
evict_entries(struct agx_device *dev)
{
   const unsigned target = AGX_ENTRY_ARENA_SIZE / AGX_ENTRY_BLOCK_SIZE / 4;
   unsigned freed = 0;
   list_for_each_entry_safe(struct agx_shader_body, body, &dev->entry_lru,
                            entry_link) {
      if (freed == target)
         break;
      if (body->entry_pins || body->entry_fixed)
         continue;
      util_vma_heap_free(&dev->entry_heap,
                         body->entry_offset - AGX_ENTRY_CODE_OFFSET,
                         AGX_ENTRY_BLOCK_SIZE);
      body->entry_offset = 0;
      list_del(&body->entry_link);
      ++freed;
   }
}

uint32_t
agx_shader_entry(struct agx_device *dev, void *allocator,
                 struct agx_shader_body *body, bool compute)
{
   simple_mtx_lock(&dev->entry_lock);
   if (!allocator || !dev->pin_entry || !dev->pin_entry(allocator, body))
      body->entry_fixed = true;
   uint32_t entry = body->entry_offset;
   if (entry) {
      list_move_to(&body->entry_link, &dev->entry_lru);
   } else {
      uint32_t block = util_vma_heap_alloc(
         &dev->entry_heap, AGX_ENTRY_BLOCK_SIZE, AGX_ENTRY_ALIGNMENT);
      if (!block) {
         evict_entries(dev);
         block = util_vma_heap_alloc(&dev->entry_heap, AGX_ENTRY_BLOCK_SIZE,
                                     AGX_ENTRY_ALIGNMENT);
      }
      if (block && !agx_shader_entry_back(dev, block + AGX_ENTRY_BLOCK_SIZE)) {
         util_vma_heap_free(&dev->entry_heap, block, AGX_ENTRY_BLOCK_SIZE);
         block = 0;
      }
      if (block) {
         entry = block + AGX_ENTRY_CODE_OFFSET;
         uint8_t bytes[AGX_ENTRY_BLOCK_SIZE];
         memset(bytes, 0, AGX_ENTRY_BLOCK_SIZE);
         put_u32(bytes, AGX_ENTRY_BLOCK_SIZE);
         if (compute) {
            util_call_once(&compute_entry_once, init_compute_entry);
            memcpy(bytes + 0x40, compute_entry_constants, 0x40);
         }
         if (build_body_entry(bytes + AGX_ENTRY_CODE_OFFSET,
                              dev->shader_base + entry, body->addr)) {
            agx_shader_entry_write(dev, block, bytes, AGX_ENTRY_BLOCK_SIZE);
            body->entry_offset = entry;
            list_addtail(&body->entry_link, &dev->entry_lru);
         } else {
            util_vma_heap_free(&dev->entry_heap, block, AGX_ENTRY_BLOCK_SIZE);
            entry = 0;
         }
      }
   }
   simple_mtx_unlock(&dev->entry_lock);
   return entry;
}
