#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>

namespace ggml_vk {

// Manages descriptor sets with bindless support (VK_EXT_descriptor_indexing).
// Uses a large descriptor pool with UPDATE_AFTER_BIND for dynamic buffer binding.
class DescriptorManager {
public:
    struct BufferBinding {
        vk::Buffer buffer;
        uint64_t offset;
        uint64_t range;
        uint32_t bindless_index;  // index into the bindless array
    };

    DescriptorManager(vk::Device device,
                      uint32_t max_bindless_buffers = 262144,
                      uint32_t max_per_dispatch_sets = 65536);
    ~DescriptorManager();

    DescriptorManager(const DescriptorManager&) = delete;
    DescriptorManager& operator=(const DescriptorManager&) = delete;

    // -- Bindless buffer array (global, set=0) --

    // Register a buffer in the bindless array. Returns the bindless index.
    [[nodiscard]] uint32_t register_buffer(vk::Buffer buffer,
                                           uint64_t offset,
                                           uint64_t range);
    // Update an existing bindless slot.
    void update_buffer(uint32_t index,
                       vk::Buffer buffer,
                       uint64_t offset,
                       uint64_t range);

    // Unregister a buffer and return its bindless index to the free list.
    void unregister_buffer(uint32_t index);

    // Reset the bindless array, reclaiming all indices. Call only when no buffers are in use.
    void reset_bindless();

    // Returns the bindless descriptor set (set=0, bound once per command buffer).
    [[nodiscard]] vk::DescriptorSet bindless_set() const { return _bindless_set; }
    [[nodiscard]] vk::DescriptorSetLayout bindless_layout() const { return _bindless_layout; }

    // -- Per-dispatch descriptor sets (set=1) --

    // Allocate a descriptor set for a pipeline layout.
    // Sets are reused across graph computes; reset_pools() only resets the index.
    [[nodiscard]] vk::DescriptorSet allocate_set(vk::DescriptorSetLayout layout);

    // Write buffer descriptors to a set. Returns the set.
    void write_buffers(vk::DescriptorSet set,
                       uint32_t first_binding,
                       vk::ArrayProxy<const vk::DescriptorBufferInfo> buffer_infos);

    // Reset per-frame index (call at start of each graph execution).
    void reset_pools();

private:
    vk::Device _device;
    mutable std::mutex _mutex;

    // Bindless
    vk::DescriptorPool _bindless_pool{};
    vk::DescriptorSetLayout _bindless_layout{};
    vk::DescriptorSet _bindless_set{};
    uint32_t _max_bindless_buffers{};
    uint32_t _bindless_count{0};
    std::vector<uint32_t> _bindless_free_list;

    // Per-dispatch: sets are allocated once and reused forever.
    std::vector<vk::DescriptorPool> _per_dispatch_pools;
    std::vector<vk::DescriptorSet> _per_dispatch_sets;
    uint32_t _per_dispatch_next_idx{0};
    static constexpr uint32_t kSetsPerPool = 256;

    void ensure_per_dispatch_pool();
};

} // namespace ggml_vk
