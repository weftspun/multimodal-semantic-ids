// SPDX-License-Identifier: MIT
// Headless Vulkan compute runner for the Slang training kernels: create N
// host-visible storage buffers, load a SPIR-V module, dispatch one entry point,
// read buffers back through the persistent map.  Generalises vk_lbs_host.cpp
// (arbitrary buffer count, entry name, group counts) so each training kernel
// (forward, backward, Adam) reuses one device + dispatch path.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "volk.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vkc.h"

static VkInstance g_inst;
static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qf;
static VkCommandPool g_cmdPool;

#define VKCHECK(x)                                          \
	do {                                                    \
		VkResult _r = (x);                                  \
		if (_r != VK_SUCCESS) {                             \
			fprintf(stderr, "vkc: %s = %d\n", #x, (int)_r); \
			return false;                                   \
		}                                                   \
	} while (0)

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
			return i;
		}
	}
	return UINT32_MAX;
}

bool vkc_init() {
	if (volkInitialize() != VK_SUCCESS) {
		fprintf(stderr, "vkc: no Vulkan loader\n");
		return false;
	}
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_1;
	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	VKCHECK(vkCreateInstance(&ici, nullptr, &g_inst));
	volkLoadInstance(g_inst);

	uint32_t nd = 0;
	vkEnumeratePhysicalDevices(g_inst, &nd, nullptr);
	std::vector<VkPhysicalDevice> pds(nd);
	vkEnumeratePhysicalDevices(g_inst, &nd, pds.data());
	g_pd = VK_NULL_HANDLE;
	for (VkPhysicalDevice cand : pds) {
		uint32_t nq = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, nullptr);
		std::vector<VkQueueFamilyProperties> qs(nq);
		vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, qs.data());
		for (uint32_t i = 0; i < nq; i++) {
			if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				g_pd = cand;
				g_qf = i;
				break;
			}
		}
		if (g_pd) {
			break;
		}
	}
	if (!g_pd) {
		fprintf(stderr, "vkc: no compute queue\n");
		return false;
	}
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(g_pd, &props);
	fprintf(stderr, "vkc: device %s\n", props.deviceName);

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = g_qf;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	VKCHECK(vkCreateDevice(g_pd, &dci, nullptr, &g_dev));
	volkLoadDevice(g_dev);
	vkGetDeviceQueue(g_dev, g_qf, 0, &g_queue);

	VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cpci.queueFamilyIndex = g_qf;
	cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VKCHECK(vkCreateCommandPool(g_dev, &cpci, nullptr, &g_cmdPool));
	return true;
}

VkcBuf vkc_buffer(size_t bytes) {
	VkcBuf b{};
	b.bytes = bytes;
	VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size = bytes;
	bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	vkCreateBuffer(g_dev, &bi, nullptr, &b.buf);
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
	VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize = mr.size;
	ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(g_dev, &ai, nullptr, &b.mem);
	vkBindBufferMemory(g_dev, b.buf, b.mem, 0);
	vkMapMemory(g_dev, b.mem, 0, bytes, 0, &b.map);
	memset(b.map, 0, bytes);
	return b;
}

bool vkc_pipeline(const char *spv_path, const char *entry, uint32_t n_bind, VkcPipeline *out) {
	FILE *f = fopen(spv_path, "rb");
	if (!f) {
		fprintf(stderr, "vkc: cannot open %s\n", spv_path);
		return false;
	}
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<uint32_t> code(sz / 4);
	if (fread(code.data(), 1, sz, f) != (size_t)sz) {
		fclose(f);
		return false;
	}
	fclose(f);
	VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smi.codeSize = sz;
	smi.pCode = code.data();
	VKCHECK(vkCreateShaderModule(g_dev, &smi, nullptr, &out->sm));

	std::vector<VkDescriptorSetLayoutBinding> lb(n_bind);
	for (uint32_t i = 0; i < n_bind; i++) {
		lb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	}
	VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	dli.bindingCount = n_bind;
	dli.pBindings = lb.data();
	VKCHECK(vkCreateDescriptorSetLayout(g_dev, &dli, nullptr, &out->dsl));
	VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &out->dsl;
	VKCHECK(vkCreatePipelineLayout(g_dev, &pli, nullptr, &out->pl));
	VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.module = out->sm;
	cpi.stage.pName = entry;
	cpi.layout = out->pl;
	VKCHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &out->pipe));

	VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_bind};
	VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpi.maxSets = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &ps;
	VKCHECK(vkCreateDescriptorPool(g_dev, &dpi, nullptr, &out->dp));
	VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dai.descriptorPool = out->dp;
	dai.descriptorSetCount = 1;
	dai.pSetLayouts = &out->dsl;
	VKCHECK(vkAllocateDescriptorSets(g_dev, &dai, &out->ds));
	out->n_bind = n_bind;
	return true;
}

bool vkc_run(VkcPipeline *p, VkcBuf *bufs, uint32_t n, uint32_t gx, uint32_t gy, uint32_t gz) {
	std::vector<VkDescriptorBufferInfo> dbi(n);
	std::vector<VkWriteDescriptorSet> wr(n);
	for (uint32_t i = 0; i < n; i++) {
		dbi[i] = {bufs[i].buf, 0, VK_WHOLE_SIZE};
		wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wr[i].dstSet = p->ds;
		wr[i].dstBinding = i;
		wr[i].descriptorCount = 1;
		wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		wr[i].pBufferInfo = &dbi[i];
	}
	vkUpdateDescriptorSets(g_dev, n, wr.data(), 0, nullptr);

	VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cbi.commandPool = g_cmdPool;
	cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = 1;
	VkCommandBuffer cmd;
	VKCHECK(vkAllocateCommandBuffers(g_dev, &cbi, &cmd));
	VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bbi);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pl, 0, 1, &p->ds, 0, nullptr);
	vkCmdDispatch(cmd, gx, gy, gz);
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(g_queue);
	vkFreeCommandBuffers(g_dev, g_cmdPool, 1, &cmd);
	return true;
}
