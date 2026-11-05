// The model's weights in engine memory: every tensor the forward pass reads, checked for name, shape and format.
//
// Everything lives in one arena of mapped pinned memory (cudaHostAlloc with cudaHostAllocMapped): the GPU reads it
// as fast as device memory on the Orin (102%, measured with bench/memory_bandwidth.cu), and O_DIRECT reads land in it
// with no bounce copy (architecture decision 8). Tensors keep the file's bytes exactly, except the one F16 vector
// (the PLE convolution taps), which is converted to F32 at load because the kernel multiplies it with F32 activations.
//
// The arena holds the non-expert tensors. The routed experts are placed by runtime::ExpertStore. Indexer tensors are
// not loaded: below indexer_top_k tokens sparse attention selects every cell, and decode is limited to that context
// until M6.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "formats/types.h"
#include "gguf/gguf.h"
#include "model/config.h"
#include "runtime/direct_io.h"

namespace torch::model {

// `rows` rows of `cols` values in `type`, back to back, as the matrix-vector kernels read them.
struct Matrix {
    Type type = Type::F32;
    int rows = 0, cols = 0;
    const uint8_t* data = nullptr;  // device-readable (and host-readable: the arena is mapped)

    uint64_t row_bytes() const noexcept {
        return torch::row_bytes(*type_info(static_cast<uint32_t>(type)), static_cast<uint64_t>(cols));
    }
};

// One expert tensor (`ffn_{gate,up,down}_exps`): per expert a rows x cols matrix, experts back to back.
struct ExpertTensor {
    Type type = Type::F32;
    int rows = 0, cols = 0;
    uint64_t slice_bytes = 0;  // one expert
    uint64_t file_offset = 0;  // of expert 0 in shard 1
};

struct Layer {
    bool recurrent = false;
    bool ple = false;

    // Hyper-connection mixers before the mixer (attn) and before the experts (ffn). Norm weights are hc x n_embd.
    const float* hc_attn_norm = nullptr;
    Matrix hc_attn_down, hc_attn_up, hc_attn_inject;
    const float* hc_ffn_norm = nullptr;
    Matrix hc_ffn_down, hc_ffn_up, hc_ffn_inject;

    // Gated DeltaNet (recurrent layers).
    Matrix qkv, z, alpha, beta, ssm_out;
    const float* conv = nullptr;  // [channels][4] taps
    const float* dt_bias = nullptr;
    const float* a = nullptr;
    const float* ssm_norm = nullptr;

    // Gated attention (the other layers). wq interleaves [q | gate] per head.
    Matrix wq, wk, wv, wo;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;

    // Experts: router, the shared expert with its sigmoid gate, the routed experts.
    Matrix router, shared_gate_inp, shared_gate, shared_up, shared_down;
    ExpertTensor gate_exps, up_exps, down_exps;

    // Per-layer embedding block (the PLE layer only). Norm weights are hc x n_embd.
    Matrix ple_key, ple_value;
    const float* ple_norm_key = nullptr;
    const float* ple_norm_query = nullptr;
    const float* ple_norm_conv = nullptr;
    const float* ple_conv = nullptr;  // [hc * n_embd][4] taps, F32
};

class WeightLoader;  // weights.cpp

class Weights {
public:
    // Bytes the arena needs. The memory budget is planned from this before anything is allocated.
    static uint64_t base_bytes(const Config& config, const gguf::File& shard1);

    // Allocates the arena and reads the weights into it. Throws Error when a tensor is missing, has an unexpected
    // shape or format, or cannot be read.
    Weights(const Config& config, const gguf::File& shard1, const runtime::DirectFile& file, runtime::ReadPool& pool);
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    ~Weights();

    std::vector<Layer> layers;
    Matrix token_embd;  // rows are read by the CPU
    const float* head_hc_norm = nullptr;
    Matrix head_hc_down, head_hc_up, output;

    uint64_t arena_bytes() const noexcept { return arena_bytes_; }

private:
    Weights() = default;  // the dry run of base_bytes()
    void bind(WeightLoader& loader, const Config& config);

    uint8_t* arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
};

}  // namespace torch::model
