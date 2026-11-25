/* SPDX-License-Identifier: MIT */
#ifndef VK_TEST_DEVICE_H
#define VK_TEST_DEVICE_H

#include <stdbool.h>
#include <string.h>
#include <vulkan/vulkan.h>

/* Test the G16 and G17P hardware paths without depending on a marketing
 * name. Keep software implementations from satisfying a hardware test. */
static inline bool
test_native_device(const VkPhysicalDeviceProperties *props)
{
   return props->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
          (strstr(props->deviceName, "(G16G ") ||
           strstr(props->deviceName, "(G16C ") ||
           strstr(props->deviceName, "(G17P "));
}

#endif
