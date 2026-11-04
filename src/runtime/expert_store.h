// Where the bytes of each routed expert are for the layer being computed (architecture decisions 6 and 8).
//
// A resident expert is read in place from the weight arena. Any other expert is read from shard 1 into a staging slot
// of mapped pinned memory, at the same offset within a page as in the file (so the format's 32-byte alignment holds).
// Both kinds go into one pointer table that the expert kernels read, so they run the same code on the same bytes
// whatever the residency, and the output never depends on which experts are resident (architecture goal 4).
//
// Milestone M2 reads misses synchronously: after a layer's router, the caller waits for the routing, prepare() reads
// the misses, and the expert kernels follow. M5 makes the reads asynchronous and predicted.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gguf/gguf.h"
#include "model/config.h"
#include "model/weights.h"
#include "runtime/direct_io.h"

namespace torch::runtime {

class ExpertStore {
public:
    // Per-slot device-readable tables of expert matrix addresses, for gemv_q8_experts.
    struct Tables {
        const uint8_t* const* gate;
        const uint8_t* const* up;
        const uint8_t* const* down;
    };

    // Bytes of the staging slots and tables for `slots` experts per layer.
    static uint64_t staging_bytes(const model::Config& config, const gguf::File& shard1, int slots);

    ExpertStore(const model::Config& config, const std::vector<model::Layer>& layers, const DirectFile& shard1,
                int resident, int slots, ReadPool& pool);
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;
    ~ExpertStore();

    // Points the tables of `layer` at the experts `ids` (at most `slots`), reading the non-resident ones into staging
    // before it returns. Staging slots are shared by all layers, so the GPU must have finished the previous layer's
    // experts. Throws Error when a read fails.
    Tables prepare(int layer, std::span<const int32_t> ids);

    int resident() const noexcept { return resident_; }
    uint64_t hits() const noexcept { return hits_; }
    uint64_t misses() const noexcept { return misses_; }

private:
    const std::vector<model::Layer>& layers_;
    const DirectFile& file_;
    ReadPool& pool_;
    int resident_;
    int slots_;
    uint64_t slot_bytes_[3] = {};       // staging bytes of one slot of gate, up, down
    uint8_t* staging_ = nullptr;        // [slots][gate | up | down], page-aligned parts
    const uint8_t** tables_ = nullptr;  // [layers][gate, up, down][slots], mapped
    uint64_t hits_ = 0, misses_ = 0;
};

}  // namespace torch::runtime
