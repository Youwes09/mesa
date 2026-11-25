/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_PPP_RECORD_H
#define AGX_PPP_RECORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct agx_apple9_render_pipeline;

#define AGX_APPLE9_MAX_VIEWPORTS 16
/* Fixed state, then a viewport block of 8n + 2 words at 0x100. */
#define AGX_APPLE9_PPP_RECORD_MAX 0x340

#define AGX_APPLE9_PPP_STATIC_SIZE 0xc0
void
agx_apple9_build_ppp_static(const struct agx_apple9_render_pipeline *pipeline,
                            uint8_t out[AGX_APPLE9_PPP_STATIC_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
