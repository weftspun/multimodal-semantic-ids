// SPDX-License-Identifier: MIT
// Headless Vulkan compute runner API (see vkc.cpp).
#pragma once
#include <cstddef>
#include <cstdint>

#include "volk.h"

struct VkcBuf {
	VkBuffer buf;
	VkDeviceMemory mem;
	void *map;
	size_t bytes;
};

struct VkcPipeline {
	VkShaderModule sm;
	VkDescriptorSetLayout dsl;
	VkPipelineLayout pl;
	VkPipeline pipe;
	VkDescriptorPool dp;
	VkDescriptorSet ds;
	uint32_t n_bind;
};

bool vkc_init();
VkcBuf vkc_buffer(size_t bytes);
bool vkc_pipeline(const char *spv_path, const char *entry, uint32_t n_bind, VkcPipeline *out);
bool vkc_run(VkcPipeline *p, VkcBuf *bufs, uint32_t n, uint32_t gx, uint32_t gy, uint32_t gz);
