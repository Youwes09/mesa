/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_bo.h"
#include <inttypes.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#include "util/hash_table.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "agx_device.h"
#include "decode.h"
#include "layout.h"

/* Helper to calculate the bucket index of a BO */
static unsigned
agx_bucket_index(size_t size)
{
   /* Round down to POT to compute a bucket index */
   unsigned bucket_index = util_logbase2(size);

   /* Clamp to supported buckets. Huge allocations use the largest bucket */
   bucket_index = CLAMP(bucket_index, MIN_BO_CACHE_BUCKET, MAX_BO_CACHE_BUCKET);

   /* Reindex from 0 */
   return (bucket_index - MIN_BO_CACHE_BUCKET);
}

static struct list_head *
agx_bucket(struct agx_device *dev, size_t size)
{
   return &dev->bo_cache.buckets[agx_bucket_index(size)];
}

static void
agx_bo_cache_remove_locked(struct agx_device *dev, struct agx_bo *bo)
{
   simple_mtx_assert_locked(&dev->bo_cache.lock);
   list_del(&bo->bucket_link);
   list_del(&bo->lru_link);
   dev->bo_cache.size -= bo->size;
}

/* Tries to fetch a BO of sufficient size with the appropriate flags from the
 * BO cache. If it succeeds, it returns that BO and removes the BO from the
 * cache. If it fails, it returns NULL signaling the caller to allocate a new
 * BO. */

struct agx_bo *
agx_bo_cache_fetch(struct agx_device *dev, size_t size, size_t align,
                   uint32_t flags, const bool dontwait)
{
   simple_mtx_lock(&dev->bo_cache.lock);
   struct list_head *bucket = agx_bucket(dev, size);
   struct agx_bo *bo = NULL;

   /* Iterate the bucket looking for something suitable, most recently freed
    * first. Reusing the oldest entry would cycle a steady workload through
    * every cached BO, so that none ever ages out of the cache. */
   list_for_each_entry_rev(struct agx_bo, entry, bucket, bucket_link) {
      if (entry->size < size || entry->flags != flags)
         continue;

      /* Do not return more than 2x oversized BOs. */
      if (entry->size > 2 * size)
         continue;

      if (align > entry->align)
         continue;

      /* This one works, use it */
      agx_bo_cache_remove_locked(dev, entry);
      bo = entry;
      break;
   }
   simple_mtx_unlock(&dev->bo_cache.lock);

   return bo;
}

static void
agx_bo_cache_evict_stale_bos(struct agx_device *dev, unsigned tv_sec)
{
   struct timespec time;

   clock_gettime(CLOCK_MONOTONIC, &time);
   list_for_each_entry_safe(struct agx_bo, entry, &dev->bo_cache.lru,
                            lru_link) {
      /* We want all entries that have been used more than 1 sec ago to be
       * dropped, others can be kept.  Note the <= 2 check and not <= 1. It's
       * here to account for the fact that we're only testing ->tv_sec, not
       * ->tv_nsec.  That means we might keep entries that are between 1 and 2
       * seconds old, but we don't really care, as long as unused BOs are
       * dropped at some point.
       */
      if (time.tv_sec - entry->last_used <= 2)
         break;

      agx_bo_cache_remove_locked(dev, entry);
      agx_bo_free(dev, entry);
   }
}

static void
agx_bo_cache_put_locked(struct agx_device *dev, struct agx_bo *bo)
{
   struct list_head *bucket = agx_bucket(dev, bo->size);
   struct timespec time;

   /* Add us to the bucket */
   list_addtail(&bo->bucket_link, bucket);

   /* Add us to the LRU list and update the last_used field. */
   list_addtail(&bo->lru_link, &dev->bo_cache.lru);
   clock_gettime(CLOCK_MONOTONIC, &time);
   bo->last_used = time.tv_sec;

   /* Update statistics */
   dev->bo_cache.size += bo->size;

   if (0) {
      printf("BO cache: %zu KiB (+%zu KiB from %s, hit/miss %" PRIu64
             "/%" PRIu64 ")\n",
             DIV_ROUND_UP(dev->bo_cache.size, 1024),
             DIV_ROUND_UP(bo->size, 1024), bo->label,
             p_atomic_read(&dev->bo_cache.hits),
             p_atomic_read(&dev->bo_cache.misses));
   }

   /* Update label for debug */
   bo->label = "Unused (BO cache)";

   /* Let's do some cleanup in the BO cache while we hold the lock. */
   agx_bo_cache_evict_stale_bos(dev, time.tv_sec);
}

/* Tries to add a BO to the cache. Returns if it was successful */
static bool
agx_bo_cache_put(struct agx_device *dev, struct agx_bo *bo)
{
   if ((bo->flags & AGX_BO_SHARED) ||
       (bo->va && (bo->va->flags & (AGX_VA_REPLAY | AGX_VA_FIXED)))) {
      return false;
   } else {
      simple_mtx_lock(&dev->bo_cache.lock);
      agx_bo_cache_put_locked(dev, bo);
      simple_mtx_unlock(&dev->bo_cache.lock);

      return true;
   }
}

/* Close a BO that is neither referenced nor cached. Closing unmaps it, so its
 * VA is free for reuse once the close returns.
 */
void
agx_bo_free(struct agx_device *dev, struct agx_bo *bo)
{
   /* The kernel can hand the handle out again as soon as the close starts.
    * Clear the table entry first and use only this copy afterwards.
    */
   struct agx_bo dead = *bo;
   memset(bo, 0, sizeof(*bo));
   __sync_synchronize();

   if (dead._map)
      munmap(dead._map, dead.size);
   if (dead.prime_fd != -1)
      close(dead.prime_fd);

   struct drm_gem_close args = {.handle = dead.handle};
   if (drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &args)) {
      mesa_loge("DRM_IOCTL_GEM_CLOSE failed: %m");
      /* The old mapping may still exist, so keep its address reserved. */
      free(dead.va);
      return;
   }

   agx_va_free(dev, dead.va, false);
}

void
agx_bo_cache_evict_all(struct agx_device *dev)
{
   simple_mtx_lock(&dev->bo_cache.lock);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i) {
      struct list_head *bucket = &dev->bo_cache.buckets[i];

      list_for_each_entry_safe(struct agx_bo, entry, bucket, bucket_link) {
         agx_bo_cache_remove_locked(dev, entry);
         agx_bo_free(dev, entry);
      }
   }
   simple_mtx_unlock(&dev->bo_cache.lock);
}

void
agx_bo_cache_init(struct agx_device *dev)
{
   simple_mtx_init(&dev->bo_cache.lock, mtx_plain);
   list_inithead(&dev->bo_cache.lru);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i)
      list_inithead(&dev->bo_cache.buckets[i]);
}

void
agx_bo_cache_finish(struct agx_device *dev)
{
   agx_bo_cache_evict_all(dev);
   simple_mtx_destroy(&dev->bo_cache.lock);
}

void
agx_bo_reference(struct agx_bo *bo)
{
   if (bo) {
      ASSERTED int count = p_atomic_inc_return(&bo->refcnt);
      assert(count != 1);
   }
}

struct label_stat {
   const char *label;
   uint32_t count;
   size_t alloc_B;
   size_t mapped_B;
};

static void
account_bo(struct label_stat *stat, struct agx_bo *bo)
{
   stat->count++;
   stat->alloc_B += bo->size;

   if (bo->_map != NULL)
      stat->mapped_B += bo->size;
}

static void
print_size(FILE *fp, size_t size_B)
{
   if (size_B >= (1024 * 1024 * 1024)) {
      fprintf(fp, "%.1f GiB", (double)size_B / (double)(1024 * 1024 * 1024));
   } else if (size_B >= (1024 * 1024)) {
      fprintf(fp, "%.1f MiB", (double)size_B / (double)(1024 * 1024));
   } else if (size_B >= 1024) {
      fprintf(fp, "%zu KiB", DIV_ROUND_UP(size_B, 1024));
   } else {
      fprintf(fp, "%zu B", size_B);
   }
}

static void
print_stat(FILE *fp, struct label_stat *stat)
{
   const char *BOLD = "\e[1m";
   const char *RESET = "\e[0m";

   fprintf(fp, "%s%s%s: ", BOLD, stat->label, RESET);
   print_size(fp, stat->alloc_B);

   if (stat->mapped_B) {
      fprintf(fp, ", mapped ");
      print_size(fp, stat->mapped_B);
   }

   fprintf(fp, ", %u BOs\n", stat->count);
}

static int
compare_size(const void *a_, const void *b_)
{
   const struct label_stat *const *label_a = a_;
   const struct label_stat *const *label_b = b_;

   size_t a = (*label_a)->alloc_B;
   size_t b = (*label_b)->alloc_B;

   return (a > b) ? 1 : (a < b) ? -1 : 0;
}

static void
agx_bo_dump_all(struct agx_device *dev)
{
   struct label_stat accum = {.label = "Total"};
   struct hash_table *totals = _mesa_string_hash_table_create(NULL);
   bool verbose = dev->debug & AGX_DBG_BODUMPVERBOSE;

   if (verbose)
      fprintf(stderr, "---\n");

   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->size)
         continue;

      if (verbose) {
         fprintf(stderr, "%u: %s %zu KiB\n", handle, bo->label,
                 bo->size / 1024);
      }

      account_bo(&accum, bo);

      assert(bo->label != NULL && "Everything must be labeled");

      struct hash_entry *ent = _mesa_hash_table_search(totals, bo->label);
      struct label_stat *ls;
      if (ent != NULL) {
         ls = ent->data;
      } else {
         ls = rzalloc(totals, struct label_stat);
         ls->label = bo->label;
         _mesa_hash_table_insert(totals, bo->label, ls);
      }

      account_bo(ls, bo);
   }

   if (verbose) {
      fprintf(stderr, "\n");
   }

   unsigned nr_labels = _mesa_hash_table_num_entries(totals);
   struct label_stat **stats =
      rzalloc_array(totals, struct label_stat *, nr_labels);
   unsigned i = 0;

   hash_table_foreach(totals, ent) {
      assert(i < nr_labels);
      stats[i++] = ent->data;
   }

   assert(i == nr_labels);

   /* Sort labels in ascending order of size */
   qsort(stats, nr_labels, sizeof(struct label_stat *), compare_size);

   for (i = 0; i < nr_labels; ++i) {
      print_stat(stderr, stats[i]);
   }

   print_stat(stderr, &accum);

   if (verbose)
      fprintf(stderr, "---\n\n");
   else
      fprintf(stderr, "\n");

   ralloc_free(totals);
}

static void
agx_bo_dump_all_periodic(struct agx_device *dev)
{
   if (likely(!(dev->debug & (AGX_DBG_BODUMP | AGX_DBG_BODUMPVERBOSE))))
      return;

   static time_t agx_last_dumped_time = 0;

   time_t now = time(NULL);
   if (now == agx_last_dumped_time)
      return;

   agx_bo_dump_all(dev);
   agx_last_dumped_time = now;
}

void
agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;

   assert(p_atomic_read(&bo->refcnt) > 0);

   /* Drop references other than the last without the lock. The last one is
    * dropped under bo_map_lock, which agx_bo_import also holds, so an import
    * never finds a shared BO that is being freed.
    * Checking the count again after taking the lock is not enough: an import
    * and a release could both run in between, and two threads would then free
    * the same BO (as iris does, see iris_bo_unreference).
    */
   int c = p_atomic_read(&bo->refcnt), old;
   while (c != 1 && (old = p_atomic_cmpxchg(&bo->refcnt, c, c - 1)) != c)
      c = old;
   if (c != 1)
      return;

   pthread_mutex_lock(&dev->bo_map_lock);

   if (p_atomic_dec_zero(&bo->refcnt)) {
      assert(!p_atomic_read_relaxed(&bo->writer));

      agx_shader_body_finish(dev, &bo->apple9_body);

      if (dev->debug & AGX_DBG_TRACE)
         agxdecode_track_free(dev->agxdecode, bo);

      if (!agx_bo_cache_put(dev, bo))
         agx_bo_free(dev, bo);
   }

   agx_bo_dump_all_periodic(dev);

   pthread_mutex_unlock(&dev->bo_map_lock);
}

static void agx_bo_finish_create(struct agx_device *dev, struct agx_bo *bo,
                                 const char *label);

struct agx_bo *
agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
              enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo;
   assert(size > 0);

   /* BOs are allocated in pages */
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   /* See if we have a BO already in the cache */
   bo = agx_bo_cache_fetch(dev, size, align, flags, true);

   /* Update stats based on the first attempt to fetch */
   if (bo != NULL)
      p_atomic_inc(&dev->bo_cache.hits);
   else
      p_atomic_inc(&dev->bo_cache.misses);

   /* Otherwise, allocate a fresh BO. If allocation fails, we can try waiting
    * for something in the cache. But if there's no nothing suitable, we should
    * flush the cache to make space for the new allocation.
    */
   if (!bo)
      bo = dev->ops.bo_alloc(dev, size, align, flags, 0, 0);
   if (!bo)
      bo = agx_bo_cache_fetch(dev, size, align, flags, false);
   if (!bo) {
      agx_bo_cache_evict_all(dev);
      bo = dev->ops.bo_alloc(dev, size, align, flags, 0, 0);
   }

   if (!bo) {
      fprintf(stderr, "BO creation failed\n");
      return NULL;
   }

   agx_bo_finish_create(dev, bo, label);
   return bo;
}

struct agx_bo *
agx_bo_create_va(struct agx_device *dev, size_t size, unsigned align,
                 enum agx_bo_flags flags, enum agx_va_flags va_flags,
                 uint64_t fixed_va, const char *label)
{
   if (!va_flags)
      return agx_bo_create(dev, size, align, flags, label);

   assert(size > 0);
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   /* A cached BO keeps the VA it was created with, so these never come from
    * the cache (and never go back to it, see agx_bo_cache_put).
    */
   struct agx_bo *bo =
      dev->ops.bo_alloc(dev, size, align, flags, va_flags, fixed_va);
   if (!bo) {
      agx_bo_cache_evict_all(dev);
      bo = dev->ops.bo_alloc(dev, size, align, flags, va_flags, fixed_va);
   }
   if (!bo)
      return NULL;

   agx_bo_finish_create(dev, bo, label);
   return bo;
}

static void
agx_bo_finish_create(struct agx_device *dev, struct agx_bo *bo,
                     const char *label)
{
   bo->label = label;
   agx_shader_body_init_bo(bo);
   p_atomic_set(&bo->refcnt, 1);

   if (dev->debug & AGX_DBG_TRACE) {
      agx_bo_map(bo);
      agxdecode_track_alloc(dev->agxdecode, bo);
   }

   if (unlikely(dev->debug & (AGX_DBG_BODUMP | AGX_DBG_BODUMPVERBOSE))) {
      pthread_mutex_lock(&dev->bo_map_lock);
      agx_bo_dump_all_periodic(dev);
      pthread_mutex_unlock(&dev->bo_map_lock);
   }
}
