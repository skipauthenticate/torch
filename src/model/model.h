// The model on the GPU: weights, the state of one sequence, and the one-token forward pass (milestone M2).
//
// decode() runs a token as one replayed CUDA graph (or kernel by kernel) and waits once, at the end. After each layer's
// router a gate kernel places the routed experts on the GPU; an expert that is not resident is read by the expert
// store's I/O thread while the gate waits (architecture decision 6), so the host never synchronizes inside a token.
// The layer math is docs/architecture.md's "The model"; tests/model/parity.py checks the result against a second
// implementation of the model.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "core/budget.h"
#include "core/placement.h"
#include "model/config.h"

namespace torch::model {

struct Options {
    std::filesystem::path shard1;  // the weights
    std::filesystem::path shard2;  // the PLE n-gram table
    // Tokens the KV cache holds. Milestone M2 computes attention densely, which is the model's exact attention only
    // up to indexer_top_k tokens, so the context is capped there.
    int context = 2048;
    std::optional<int> resident;  // experts 0 to resident - 1 of every layer resident, exactly, with no top-up (tests)
    // Route counts (core/placement.h) ranking which experts to keep resident; without them, and without `resident`,
    // as many of every layer's first experts as the budget allows.
    std::filesystem::path placement;
    bool logits = true;  // copy every token's 248,320 logits to the host (per-step output, parity tests)
};

struct ExpertStats {
    int resident;     // experts resident, over all layers
    int experts;      // experts in the model, over all layers
    uint64_t hits;    // routed experts found resident
    uint64_t misses;  // routed experts read from the SSD
    uint64_t hedged;  // miss reads issued a second time because the first was late (expert_store.h)
};

// Where decode time went, summed over every decode() call (host wall-clock, milliseconds).
struct DecodeTimes {
    double total = 0;
    double ple = 0;           // reading and dequantizing the PLE rows
    double ple_max = 0;       // the slowest token's PLE rows
    uint64_t ple_slow = 0;    // tokens whose PLE rows took over 5 ms
    uint64_t ple_hedged = 0;  // PLE row reads issued a second time (hedged_reads.h)
    double gpu = 0;           // waiting for the GPU after the last launch: the work not yet done, including misses
    double expert_reads = 0;  // the I/O thread reading non-resident experts (overlaps `gpu`)
};

class Model {
public:
    // Plans the memory budget, then loads the weights. Throws Error when a file is missing or not the supported
    // model, or when memory is short.
    explicit Model(const Options& options);
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    const Config& config() const noexcept;
    const Budget& budget() const noexcept;
    ExpertStats expert_stats() const;  // waits for the GPU
    DecodeTimes decode_times() const noexcept;
    RouteCounts route_counts() const;  // how often each (layer, expert) was routed so far (waits for the GPU)
    int position() const noexcept;     // tokens in the sequence so far

    // Appends `token` to the sequence and runs it through the model. The decode loop does not throw
    // (CONTRIBUTING.md): when the token is out of range, the context is full, or the GPU or a read fails, it returns
    // false and error() says why. After a GPU or read failure the sequence state is undefined; reset() before
    // decoding again. A token runs as one replayed CUDA graph, recorded at the first token; TORCH_EAGER=1 launches
    // kernel by kernel instead (the result is the same, bit for bit).
    bool decode(int32_t token) noexcept;
    // The last token's next-token logits (n_vocab floats, valid until the next decode), or empty when Options::logits
    // is false.
    std::span<const float> logits() const noexcept;
    // The most likely next token after the last decode, computed on the GPU; ties go to the lowest id.
    int32_t argmax() const noexcept;
    const std::string& error() const noexcept;

    // Starts a new, empty sequence.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch::model
