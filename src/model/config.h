// The model's hyperparameters, read from the GGUF metadata of shard 1 and checked against what the kernels support.
//
// The engine supports one model, so instead of handling every value a qwen4exp file could hold, Config::read refuses
// a file whose shape differs from Qwen3.8-Flash-Next's in any way a kernel depends on. The values are listed in
// docs/architecture.md ("The model").
#pragma once

#include <cstdint>
#include <vector>

#include "gguf/gguf.h"

namespace torch::model {

struct Config {
    int n_layer = 0;
    int n_embd = 0;  // width of one hyper-connection stream
    int n_vocab = 0;
    int context_length = 0;
    float eps = 0;  // RMS norm epsilon

    // Gated full attention, every full_attention_interval-th layer.
    int full_attention_interval = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_rot = 0;  // RoPE dimensions: NeoX pairs (i, i + n_rot / 2)
    float rope_base = 0;
    int indexer_top_k = 0;  // below this many tokens, sparse attention selects every cell

    // Gated DeltaNet, the other layers.
    int ssm_k_heads = 0, ssm_v_heads = 0, ssm_head_dim = 0, ssm_conv = 0;

    // Mixture of experts.
    int n_expert = 0, n_expert_used = 0, n_ff_expert = 0, n_ff_shared = 0;

    // Hyper-connections: hc streams of n_embd, mixed through rank-hc_rank projections.
    int hc = 0, hc_rank = 0;

    // Per-layer embeddings from the hashed n-gram table.
    int ple_layer = 0, ple_ngram = 0, ple_heads_per_ngram = 0, ple_dim = 0, ple_conv = 0;
    uint32_t ple_eos = 0;  // fills the n-gram window before the first token and after an end of sequence
    std::vector<uint64_t> ple_multipliers, ple_head_offsets, ple_head_vocab;

    uint32_t eos_token = 0;

    bool recurrent(int layer) const noexcept { return (layer + 1) % full_attention_interval != 0; }
    int ple_heads() const noexcept { return (ple_ngram - 1) * ple_heads_per_ngram; }
    int conv_channels() const noexcept { return (2 * ssm_k_heads + ssm_v_heads) * ssm_head_dim; }

    // Reads and checks the metadata and the token embedding's shape. Throws Error naming the first key that is
    // missing or has a value the engine does not support.
    static Config read(const gguf::File& shard1);
};

}  // namespace torch::model
