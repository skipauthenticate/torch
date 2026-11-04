#include "runtime/expert_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

#include "core/cuda_check.h"
#include "core/error.h"

namespace torch::runtime {
namespace {

constexpr const char* kKinds[3] = {"gate", "up", "down"};

// The largest expert slice of each kind over all layers, plus the page a read may start inside.
void slot_sizes(const model::Config& config, const gguf::File& shard1, uint64_t out[3]) {
    for (int k = 0; k < 3; ++k) {
        uint64_t most = 0;
        for (int il = 0; il < config.n_layer; ++il) {
            const std::string name = "blk." + std::to_string(il) + ".ffn_" + kKinds[k] + "_exps.weight";
            const gguf::TensorInfo* t = shard1.tensor(name);
            if (t == nullptr)
                throw Error("The model file has no tensor " + name + "; use the supported model file.");
            most = std::max(most, t->slice_bytes());
        }
        out[k] = page_ceil(most) + kPage;
    }
}

}  // namespace

uint64_t ExpertStore::staging_bytes(const model::Config& config, const gguf::File& shard1, int slots) {
    uint64_t sizes[3];
    slot_sizes(config, shard1, sizes);
    const uint64_t tables = uint64_t{3} * static_cast<uint64_t>(config.n_layer * slots) * sizeof(void*);
    return static_cast<uint64_t>(slots) * (sizes[0] + sizes[1] + sizes[2]) + page_ceil(tables);
}

ExpertStore::ExpertStore(const model::Config& config, const std::vector<model::Layer>& layers, const DirectFile& shard1,
                         int resident, int slots, ReadPool& pool)
    : layers_(layers), file_(shard1), pool_(pool), resident_(resident), slots_(slots) {
    uint64_t sizes[3] = {};
    for (int k = 0; k < 3; ++k) {
        uint64_t most = 0;
        for (const model::Layer& L : layers) {
            const model::ExpertTensor& t = k == 0 ? L.gate_exps : k == 1 ? L.up_exps : L.down_exps;
            most = std::max(most, t.slice_bytes);
        }
        sizes[k] = page_ceil(most) + kPage;
        slot_bytes_[k] = sizes[k];
    }
    void* staging = nullptr;
    TORCH_CUDA_CHECK(
        cudaHostAlloc(&staging, static_cast<size_t>(slots) * (sizes[0] + sizes[1] + sizes[2]), cudaHostAllocMapped));
    staging_ = static_cast<uint8_t*>(staging);
    void* tables = nullptr;
    TORCH_CUDA_CHECK(cudaHostAlloc(&tables, size_t{3} * static_cast<size_t>(config.n_layer * slots) * sizeof(void*),
                                   cudaHostAllocMapped));
    tables_ = static_cast<const uint8_t**>(tables);
}

ExpertStore::~ExpertStore() {
    if (staging_)
        cudaFreeHost(staging_);
    if (tables_)
        cudaFreeHost(const_cast<uint8_t**>(tables_));
}

ExpertStore::Tables ExpertStore::prepare(int layer, std::span<const int32_t> ids) {
    if (static_cast<int>(ids.size()) > slots_)
        throw Error("A layer routed to more experts than the store has slots for; this is a bug in the engine.");
    const model::Layer& L = layers_[static_cast<size_t>(layer)];
    const model::ExpertTensor* tensors[3] = {&L.gate_exps, &L.up_exps, &L.down_exps};
    const uint8_t** table = tables_ + size_t{3} * static_cast<size_t>(layer * slots_);

    struct Miss {
        int slot;
        int32_t expert;
    };
    Miss misses[16];
    int n_miss = 0;
    for (int s = 0; s < static_cast<int>(ids.size()); ++s) {
        const int32_t e = ids[static_cast<size_t>(s)];
        if (e < resident_) {
            for (int k = 0; k < 3; ++k)
                table[k * slots_ + s] = tensors[k]->resident + static_cast<uint64_t>(e) * tensors[k]->slice_bytes;
            ++hits_;
        } else {
            misses[n_miss++] = {s, e};
            ++misses_;
        }
    }
    // One job per matrix of each missing expert, so the pool keeps up to 3 x misses reads in flight.
    pool_.run(3 * n_miss, [&](int job) {
        const Miss& m = misses[job / 3];
        const int k = job % 3;
        const model::ExpertTensor& t = *tensors[k];
        uint8_t* slot = staging_ + static_cast<uint64_t>(m.slot) * (slot_bytes_[0] + slot_bytes_[1] + slot_bytes_[2]);
        for (int before = 0; before < k; ++before)
            slot += slot_bytes_[before];
        table[k * slots_ + m.slot] =
            file_.read(t.file_offset + static_cast<uint64_t>(m.expert) * t.slice_bytes, t.slice_bytes, slot);
    });
    return {table, table + slots_, table + 2 * slots_};
}

}  // namespace torch::runtime
