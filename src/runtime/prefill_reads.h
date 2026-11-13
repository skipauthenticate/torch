// Non-resident experts read ahead for prefill chunks (milestone M6).
//
// A wide chunk routes every expert of a layer, so the experts a layer misses are known before its router runs: the
// chunk's reads are queued at its start, layer after layer, and a background thread reads them while the GPU computes
// the layers before. Two pools of `slots` experts each (gate, up and down slots of the largest size, in mapped pinned
// memory, filled by O_DIRECT reads that are hedged as misses are) take requests in turn: request t lands in pool t % 2
// once the GPU is done with request t - 2 (release()). Each request is up to `slots` experts of one layer.
#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gguf/gguf.h"
#include "model/config.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/hedged_reads.h"

namespace torch::runtime {

class PrefillReads {
public:
    // Bytes of the two pools of `slots` experts each.
    static uint64_t bytes(const model::Config& config, const gguf::File& shard1, int slots);

    PrefillReads(const std::vector<model::Layer>& layers, const DirectFile& file, int slots);
    PrefillReads(const PrefillReads&) = delete;
    PrefillReads& operator=(const PrefillReads&) = delete;
    ~PrefillReads();  // waits for the reads in flight

    int slots() const noexcept { return slots_; }
    // Queues `experts` (at most slots()) of `layer`; returns the request's ticket (0, 1, 2, ... since the last
    // reset()).
    int request(int layer, std::vector<int32_t> experts);
    // Waits for a ticket's reads: each expert's gate, up and down bytes, 3 pointers per expert in order. Throws Error
    // when a read failed.
    std::vector<const uint8_t*> wait(int ticket);
    // The GPU is done with the ticket's bytes once the work queued on `stream` so far has run (an event marks it).
    void release(int ticket, cudaStream_t stream);
    // Drops the requests not yet waited for (after waiting for those in flight) and starts the tickets at 0 again.
    void reset();

private:
    struct Request {
        int layer;
        std::vector<int32_t> experts;
        std::vector<const uint8_t*> data;
        bool done = false, failed = false;
        cudaEvent_t released = nullptr;  // recorded after the GPU's last read of the bytes (null: nothing to wait for)
        bool has_released = false;
    };
    void work();

    const std::vector<model::Layer>& layers_;
    const int slots_;
    uint8_t* memory_ = nullptr;  // both pools, mapped pinned
    std::unique_ptr<HedgedReads> io_[2];
    std::vector<cudaEvent_t> events_;  // one per ticket in flight, created on demand
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Request> queue_;   // ticket t at queue_[t]
    int next_ = 0, reading_ = 0;  // the next ticket to hand out, the next to read (queue_[t] is ticket t)
    bool stop_ = false, busy_ = false;
    std::thread thread_;
};

}  // namespace torch::runtime
