// The model on the GPU: weights, the state of one sequence, and the one-token forward pass (milestone M2).
//
// decode() runs the layers eagerly, kernel by kernel, on one CUDA stream. After each layer's router it waits for the
// routing so the expert store can read the experts that are not resident; that one host round trip per layer is the
// place M3's per-layer graph segments start and end (architecture decision 6). The layer math is docs/architecture.md's
// "The model"; tests/model/parity.py checks the result against a second implementation of the model.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "core/budget.h"
#include "model/config.h"

namespace torch::model {

struct Options {
    std::filesystem::path shard1;  // the weights
    std::filesystem::path shard2;  // the PLE n-gram table
    // Tokens the KV cache holds. Milestone M2 computes attention densely, which is the model's exact attention only
    // up to indexer_top_k tokens, so the context is capped there.
    int context = 2048;
    std::optional<int> resident;  // experts per layer to keep resident; by default as many as the budget allows
};

struct ExpertStats {
    int resident;     // per layer
    uint64_t hits;    // routed experts found resident
    uint64_t misses;  // routed experts read from the SSD
};

// Where decode time went, summed over every decode() call (host wall-clock, milliseconds).
struct DecodeTimes {
    double total = 0;
    double ple = 0;           // reading and dequantizing the PLE rows
    double routing = 0;       // waiting for each layer's routing (the GPU work up to its router)
    double expert_reads = 0;  // reading non-resident experts into staging
    double finish = 0;        // waiting for the rest of the GPU work and the logits
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
    ExpertStats expert_stats() const noexcept;
    DecodeTimes decode_times() const noexcept;
    int position() const noexcept;  // tokens in the sequence so far

    // Appends `token` to the sequence, runs it through the model and returns its next-token logits (n_vocab floats,
    // valid until the next call). The decode loop does not throw (CONTRIBUTING.md): when the token is out of range,
    // the context is full, or the GPU or a read fails, it returns an empty span and error() says why. After a GPU or
    // read failure the sequence state is undefined; reset() before decoding again.
    std::span<const float> decode(int32_t token) noexcept;
    const std::string& error() const noexcept;

    // Starts a new, empty sequence.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch::model
