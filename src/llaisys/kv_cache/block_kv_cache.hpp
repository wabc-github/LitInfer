#pragma once
#include "llaisys/tensor.h"
#include <vector>
#include <cstddef>
#include <algorithm>

namespace llaisys {
namespace models {

// 每个物理 KV block（页）容纳的 token 数。
constexpr int KV_BLOCK_TOKENS = 16;

// 单层的物理 KV block 池：所有序列共享的分页池。
// 物理池一次性分配为 4D 张量 [num_blocks, BLOCK_TOKENS, nkvh, dh]，
// 但具体哪些块被占用由 free_list / used 动态管理（按需分页）。
struct KVBlockPool {
    llaisysTensor_t k_pool{nullptr};  // [num_blocks, BLOCK_TOKENS, nkvh, dh]
    llaisysTensor_t v_pool{nullptr};  // [num_blocks, BLOCK_TOKENS, nkvh, dh]
    int num_blocks{0};
    std::vector<int> free_list;       // 空闲物理块 id（末尾弹出，O(1)）

    // 初始化：创建物理池张量，并把全部块放入 free list。
    void init(int num_blocks_, size_t nkvh, size_t dh,
              llaisysDataType_t dtype, llaisysDeviceType_t device, int devid) {
        destroy();
        num_blocks = num_blocks_;
        size_t shape[4] = {
            static_cast<size_t>(num_blocks),
            static_cast<size_t>(KV_BLOCK_TOKENS),
            nkvh, dh
        };
        k_pool = tensorCreate(shape, 4, dtype, device, devid);
        v_pool = tensorCreate(shape, 4, dtype, device, devid);
        free_list.clear();
        free_list.reserve(num_blocks);
        // 从后往前压，使 alloc 时先拿到块 0（顺序上更直观，与单序列映射一致）。
        for (int i = num_blocks - 1; i >= 0; --i) free_list.push_back(i);
    }

    void destroy() {
        if (k_pool) { tensorDestroy(k_pool); k_pool = nullptr; }
        if (v_pool) { tensorDestroy(v_pool); v_pool = nullptr; }
        free_list.clear();
        num_blocks = 0;
    }

    // 分配一个空闲物理块；无空闲返回 -1。
    int alloc() {
        if (free_list.empty()) return -1;
        int id = free_list.back();
        free_list.pop_back();
        return id;
    }

    // 归还物理块。
    void free_block(int id) {
        if (id < 0 || id >= num_blocks) return;
        free_list.push_back(id);
    }

    // 当前已占用物理块数。
    int used_blocks() const { return num_blocks - static_cast<int>(free_list.size()); }
};

// 单序列的分页元信息：block table（logical block -> physical block）。
struct SequenceKVMeta {
    size_t seq_len{0};
    std::vector<int> block_table;  // 下标是 logical block，值是 physical block

    // 为接下来 n_new 个新 token 预留物理块（懒分配：跨过块边界才申请新页）。
    // 返回 false 表示池已满。
    bool reserve_for_tokens(size_t n_new, KVBlockPool &pool) {
        size_t needed_blocks =
            (seq_len + n_new + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS;
        while (block_table.size() < needed_blocks) {
            int b = pool.alloc();
            if (b < 0) return false;
            block_table.push_back(b);
        }
        return true;
    }

    // 把第 t 个 token 位置映射到 (物理块 id, 块内偏移)。
    void map_position(size_t t, int &phys, int &offset) const {
        size_t lb = t / KV_BLOCK_TOKENS;
        offset = static_cast<int>(t % KV_BLOCK_TOKENS);
        phys = block_table[lb];
    }

    // 释放本序列占用的全部物理块（归还到池）。
    void release(KVBlockPool &pool) {
        for (int b : block_table) pool.free_block(b);
        block_table.clear();
        seq_len = 0;
    }
};

} // namespace models
} // namespace llaisys
