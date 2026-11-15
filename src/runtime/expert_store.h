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
//
// Swaps (core/swaps.h) change which experts are resident while memory stays constant: between windows, a miss of the
// last window, still in its staging buffers, is copied into the slots of a resident expert of its layer, which stops
// being resident, and the map entries follow. Same-layer slots have the same size and keep the file's offset within a
// page. The copies run on the store's own stream, overlapping whatever the caller does before its next window
// (drafting); before_window() waits for them and frees the staging buffers, which stay reserved until then (or until a
// request would not fit), so a window's misses can still be copied after it. The gate kernel stamps each routed
// (layer, expert) with its window in last_use(), the recency swaps go by.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
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

    // An expert in device memory the caller lends (the KV cache's unused tail, torch run --lend-kv): its gate, up and
    // down slots one after another from `at` (page-aligned), layer_expert_bytes() of its layer in all.
    struct Tenant {
        int layer, expert;
        uint8_t* at;
    };
    // Makes the tenants resident, read through pinned bounce buffers and copied to the device. Call between windows.
    // Throws Error when a read fails or an expert is already resident.
    void add_tenants(const std::vector<Tenant>& tenants);
    // Makes every expert with a matrix in [begin, end) not resident (the map entries become null, as a swap's do) and
    // returns them as (layer, expert); the memory is the caller's again. Call between windows, after before_window().
    std::vector<std::pair<int, int>> evict(const uint8_t* begin, const uint8_t* end);

    // What the gate kernel and the expert kernels read; all device-accessible.
    const uint8_t* const* map() const noexcept { return map_; }
    // The tables the gate kernel writes for a layer's window (expert_gate.h).
    GateTables gate_tables(int layer) const noexcept;
    Mailbox* mailbox() const noexcept { return mailbox_; }
    uint32_t* seq() const noexcept { return seq_; }
    unsigned long long* counters() const noexcept { return counters_; }
    unsigned long long* route_counts() const noexcept { return route_counts_; }
    const uint8_t* zeros() const noexcept { return zeros_; }
    uint32_t* last_use() const noexcept { return last_use_; }  // [layer][expert], mapped pinned: read by the host too

    // Routed experts found resident and read from the SSD so far, and how often each (layer, expert) was routed
    // (both wait for the GPU).
    std::pair<uint64_t, uint64_t> hits_and_misses() const;
    std::vector<uint64_t> read_route_counts() const;
    // Milliseconds the I/O thread has spent serving misses, and reads it had to issue twice.
    double read_ms() const noexcept { return static_cast<double>(read_ns_.load()) / 1e6; }
    uint64_t hedged_reads() const noexcept { return io_->hedged(); }
    // Throws Error when a miss could not be served (a failed read or the GPU's 2 s timeout); call after a token.
    void check() const;

    // An expert whose gate, up and down matrices are in the staging buffers.
    struct Staged {
        int layer, expert;
        const uint8_t* data[3];
    };
    bool resident(int layer, int expert) const noexcept;
    // A resident expert's gate, up and down matrices (all null when it is not resident).
    std::array<const uint8_t*, 3> resident_matrices(int layer, int expert) const noexcept;
    // For a prefill chunk's routed experts (milestone M6): reads up to kGateSlots experts of a layer into the staging
    // buffers, hedged as a miss is, and returns their gate, up and down bytes (3 per expert, in order). The buffers
    // stay the caller's until the next call or before_window(); the GPU must be done with the previous call's. Call
    // between windows, as the rest. Throws Error when a read fails.
    std::vector<const uint8_t*> stage_for_prefill(int layer, std::span<const int32_t> experts);
    // The window stamps the gate kernel writes, n_layer * n_expert of them (wait for the GPU before reading).
    std::span<const uint32_t> stamps() const noexcept { return {last_use_, n_route_counts_}; }
    void clear_stamps();  // a new sequence: its positions start again at 0
    // The misses served since the last before_window(), in order. Call between windows (the GPU idle), as the rest.
    std::vector<Staged> staged() const;
    // Queues the copy of staged expert `in` into the slots of resident expert `out` of the same layer and changes the
    // map: `in` becomes resident and `out` not. With on_cpu the CPU copies now instead: inside a window, where GPU work
    // queued on another stream can wait behind the window that waits for the I/O thread. Throws Error when `in` is
    // resident or `out` is not.
    void swap_in(const Staged& in, int out, bool on_cpu = false);
    // Marks the end of the swaps queued since the last before_window(), which waits for their copies.
    void end_swaps();
    // Before a window: waits for the swaps' copies and map entries, then frees the staging buffers.
    void before_window();
    // When a miss does not fit next to the window's held misses, the I/O thread hands those to `f` (which may swap
    // them in with swap_in on the CPU; their layers are done for this window) before it frees their buffers.
    // Without it they are dropped. Set before the first window.
    void on_full(std::function<void(std::span<const Staged>)> f) { on_full_ = std::move(f); }

private:
    void load(const std::vector<std::vector<int32_t>>& resident);  // reads them and updates the device map
    void serve();

    const std::vector<model::Layer>& layers_;
    const DirectFile& file_;
    int k_;  // routes per token
    int n_expert_;
    ReadPool pool_;                               // loads the resident experts
    std::vector<uint8_t*> arenas_;                // resident experts, mapped pinned: one allocation per load
    std::vector<const uint8_t*> host_map_;        // the device map's contents
    uint8_t* staging_ = nullptr;                  // the staging buffers, mapped pinned
    std::unique_ptr<HedgedReads> io_;             // the miss reads, into the staging buffers
    const uint8_t** map_ = nullptr;               // mapped pinned: [layer][expert][kind]
    const uint8_t** tables_ = nullptr;            // device: [layer][3][kGateSlots]
    const uint8_t** down_routes_ = nullptr;       // device: [layer][3][kGateColumns * k]: down, gate, up routes
    uint8_t* columns_ = nullptr;                  // device: [layer][kGateSlots]
    int8_t* ranks_ = nullptr;                     // device: [layer][kGateSlots][kGateColumns]
    int32_t* counts_ = nullptr;                   // device: [layer]
    Mailbox* mailbox_ = nullptr;                  // mapped pinned
    uint32_t* seq_ = nullptr;                     // device
    unsigned long long* counters_ = nullptr;      // device: hits, misses
    unsigned long long* route_counts_ = nullptr;  // device: [layer][expert]
    size_t n_route_counts_ = 0;
    uint8_t* zeros_ = nullptr;       // device: a zero matrix for timed-out requests
    mutable std::mutex io_mutex_;    // io_, staged_ and held_: the I/O thread and swaps between windows
    std::vector<Staged> staged_;     // misses served since the last before_window()
    int held_ = 0;                   // experts whose staging buffers are reserved
    cudaStream_t copies_ = nullptr;  // the swaps' copies
    cudaEvent_t copied_ = nullptr;   // recorded after them
    bool pending_ = false;           // swaps queued since the last before_window()
    uint32_t* last_use_ = nullptr;   // mapped pinned: [layer][expert] window stamps
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> swap_failed_{false};
    std::function<void(std::span<const Staged>)> on_full_;
    std::atomic<uint64_t> read_ns_{0};
    std::thread thread_;
};

}  // namespace torch::runtime
