/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * The PPP record of draws with launch programs: its stage-derived part, built
 * once per linked pipeline.
 */

#ifndef AGX_PPP_RECORD_H
#define AGX_PPP_RECORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct agx_draw_pipeline;

#define AGX_PPP_MAX_VIEWPORTS 16
/* Fixed state, then a viewport block of 8n + 2 words at 0x100. */
#define AGX_PPP_RECORD_MAX 0x340

#define AGX_PPP_STATIC_SIZE 0xc0
void agx_build_ppp_static(const struct agx_draw_pipeline *pipeline,
                          uint8_t out[AGX_PPP_STATIC_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
