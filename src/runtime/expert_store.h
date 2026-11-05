// Where the bytes of every routed expert are (architecture decisions 6 and 8).
//
// Resident experts live in mapped pinned memory, each matrix in a slot of its own layer's exact size, filled at start
// by O_DIRECT reads; a device map gives every (layer, expert) the addresses of its gate, up and down matrices, or null
// when the expert is not resident. The gate kernel (expert_gate.h) turns a layer's routing into that layer's pointer
// table on the GPU. For an absent expert it posts a request to the store's mailbox, and the store's I/O thread reads
// the three matrices into staging slots, at the same offset within a page as in the file (so the format's 32-byte
// alignment holds), and publishes them. The expert kernels read every expert through the table, so they run the same
// code on the same bytes whatever the residency (architecture goal 4).
//
// Miss reads are hedged (hedged_reads.h): one still pending after 50 ms is issued again into a spare staging buffer,
// because the development Orin's SSD occasionally leaves a read uncompleted for 30 s.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "gguf/gguf.h"
#include "model/config.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/expert_gate.h"
#include "runtime/hedged_reads.h"

namespace torch::runtime {

class ExpertStore {
public:
    // Bytes of memory one expert of each layer takes (its three slots), and the staging buffers, mailbox, map and zero
    // buffer the store needs besides. The memory budget and the placement are planned from these.
    static std::vector<uint64_t> layer_expert_bytes(const model::Config& config, const gguf::File& shard1);
    static uint64_t fixed_bytes(const model::Config& config, const gguf::File& shard1);

    // Reads the experts in resident[layer] (expert ids) into memory and starts the I/O thread. Throws Error when a read
    // or an allocation fails.
    ExpertStore(const model::Config& config, const std::vector<model::Layer>& layers, const DirectFile& shard1,
                const std::vector<std::vector<int32_t>>& resident);
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;
    ~ExpertStore();

    // Makes the experts in more[layer] resident as well, in memory of their own. Call before the first token. Throws
    // Error when a read or an allocation fails, or when an expert is already resident.
    void add(const std::vector<std::vector<int32_t>>& more);

    // What the gate kernel and the expert kernels read; all device-accessible.
    const uint8_t* const* map() const noexcept { return map_; }
    const uint8_t** table(int layer) const noexcept { return tables_ + int64_t{3} * layer * slots_; }
    int slots() const noexcept { return slots_; }
    Mailbox* mailbox() const noexcept { return mailbox_; }
    uint32_t* seq() const noexcept { return seq_; }
    unsigned long long* counters() const noexcept { return counters_; }
    unsigned long long* route_counts() const noexcept { return route_counts_; }
    const uint8_t* zeros() const noexcept { return zeros_; }

    // Routed experts found resident and read from the SSD so far, and how often each (layer, expert) was routed
    // (both wait for the GPU).
    std::pair<uint64_t, uint64_t> hits_and_misses() const;
    std::vector<uint64_t> read_route_counts() const;
    // Milliseconds the I/O thread has spent serving misses, and reads it had to issue twice.
    double read_ms() const noexcept { return static_cast<double>(read_ns_.load()) / 1e6; }
    uint64_t hedged_reads() const noexcept { return io_->hedged(); }
    // Throws Error when a miss could not be served (a failed read or the GPU's 2 s timeout); call after a token.
    void check() const;

private:
    void load(const std::vector<std::vector<int32_t>>& resident);  // reads them and updates the device map
    void serve();

    const std::vector<model::Layer>& layers_;
    const DirectFile& file_;
    int slots_;
    int n_expert_;
    ReadPool pool_;                               // loads the resident experts
    std::vector<uint8_t*> arenas_;                // resident experts, mapped pinned: one allocation per load
    std::vector<const uint8_t*> host_map_;        // the device map's contents
    uint8_t* staging_ = nullptr;                  // the staging buffers, mapped pinned
    std::unique_ptr<HedgedReads> io_;             // the miss reads, into the staging buffers
    const uint8_t** map_ = nullptr;               // device: [layer][expert][kind]
    const uint8_t** tables_ = nullptr;            // device: [layer][kind][slot]
    Mailbox* mailbox_ = nullptr;                  // mapped pinned
    uint32_t* seq_ = nullptr;                     // device
    unsigned long long* counters_ = nullptr;      // device: hits, misses
    unsigned long long* route_counts_ = nullptr;  // device: [layer][expert]
    size_t n_route_counts_ = 0;
    uint8_t* zeros_ = nullptr;  // device: a zero matrix for timed-out requests
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::atomic<uint64_t> read_ns_{0};
    std::thread thread_;
};

}  // namespace torch::runtime
