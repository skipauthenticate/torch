#include "model/config.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "core/error.h"

namespace torch::model {
namespace {

constexpr std::string_view kArch = "qwen4exp";

std::string key(std::string_view name) {
    return std::string(kArch) + "." + std::string(name);
}

template <class T>
T required(const gguf::File& f, const std::string& name) {
    const std::optional<T> v = f.get<T>(name);
    if (!v)
        throw Error("The model file has no usable " + name + "; use the Qwen3.8-Flash-Next GGUF this engine supports.");
    return *v;
}

int count(const gguf::File& f, std::string_view name) {
    const uint32_t v = required<uint32_t>(f, key(name));
    if (v == 0 || v > (1u << 24))
        throw Error("The model file's " + key(name) + " is " + std::to_string(v) +
                    ", which this engine does not support.");
    return static_cast<int>(v);
}

// The kernels are written for these values of Qwen3.8-Flash-Next.
void expect(bool ok, std::string_view what) {
    if (!ok)
        throw Error("The model file's " + std::string(what) +
                    " differs from Qwen3.8-Flash-Next's; this engine runs only that model.");
}

}  // namespace

Config Config::read(const gguf::File& f) {
    expect(f.get<std::string>("general.architecture") == std::string(kArch), "general.architecture");
    Config c;
    c.n_layer = count(f, "block_count");
    c.n_embd = count(f, "embedding_length");
    c.context_length = count(f, "context_length");
    c.eps = required<float>(f, key("attention.layer_norm_rms_epsilon"));

    c.full_attention_interval = count(f, "full_attention_interval");
    c.n_head = count(f, "attention.head_count");
    c.n_head_kv = count(f, "attention.head_count_kv");
    c.head_dim = count(f, "attention.key_length");
    c.n_rot = count(f, "rope.dimension_count");
    c.rope_base = required<float>(f, key("rope.freq_base"));
    c.indexer_top_k = count(f, "attention.indexer.top_k");

    c.ssm_k_heads = count(f, "ssm.group_count");
    c.ssm_v_heads = count(f, "ssm.time_step_rank");
    c.ssm_head_dim = count(f, "ssm.state_size");
    c.ssm_conv = count(f, "ssm.conv_kernel");

    c.n_expert = count(f, "expert_count");
    c.n_expert_used = count(f, "expert_used_count");
    c.n_ff_expert = count(f, "expert_feed_forward_length");
    c.n_ff_shared = count(f, "expert_shared_feed_forward_length");

    c.hc = count(f, "hyper_connection.count");
    c.hc_rank = count(f, "hyper_connection.low_rank");

    const auto ple_layers = required<std::vector<int32_t>>(f, key("ple.layers"));
    expect(ple_layers.size() == 1, "ple.layers");
    c.ple_layer = ple_layers[0];
    c.ple_ngram = count(f, "ple.ngram_size");
    c.ple_heads_per_ngram = count(f, "ple.heads_per_ngram");
    c.ple_dim = count(f, "embedding_length_per_layer_input");
    c.ple_conv = count(f, "ple.conv_kernel");
    c.ple_eos = required<uint32_t>(f, key("ple.eos_token_id"));
    c.ple_multipliers = required<std::vector<uint64_t>>(f, key("ple.layer_multipliers"));
    c.ple_head_offsets = required<std::vector<uint64_t>>(f, key("ple.head_offsets"));
    c.ple_head_vocab = required<std::vector<uint64_t>>(f, key("ple.head_vocab_sizes"));
    c.eos_token = required<uint32_t>(f, "tokenizer.ggml.eos_token_id");

    const gguf::TensorInfo* embd = f.tensor("token_embd.weight");
    expect(embd != nullptr && embd->n_dims == 2 && embd->shape[0] == c.n_embd, "token_embd.weight");
    c.n_vocab = static_cast<int>(embd->shape[1]);
    c.n_vocab_used = c.n_vocab;
    if (const auto types = f.get<std::vector<int32_t>>("tokenizer.ggml.token_type"))
        for (size_t v = std::min(types->size(), static_cast<size_t>(c.n_vocab)); v > 0 && (*types)[v - 1] == 5; --v)
            c.n_vocab_used = static_cast<int>(v - 1);

    // The shapes the kernels are written for (src/kernels/*/ headers).
    expect(c.head_dim == 256 && c.n_head % c.n_head_kv == 0 && c.n_rot % 2 == 0 && c.n_rot <= c.head_dim,
           "attention shape");
    expect(c.ssm_head_dim == 128 && c.ssm_conv == 4 && c.ssm_v_heads % c.ssm_k_heads == 0, "DeltaNet shape");
    expect(c.n_expert <= 1024 && c.n_expert_used <= 16 && c.n_embd % 256 == 0 && c.n_ff_expert % 32 == 0 &&
               c.n_ff_shared % 32 == 0,
           "expert shape");
    expect(c.ple_ngram == 3 && c.ple_conv == 4 && c.ple_heads() * c.ple_dim == c.n_embd, "PLE shape");
    expect(c.ple_layer >= 0 && c.ple_layer < c.n_layer && c.recurrent(c.ple_layer), "ple.layers");
    expect(static_cast<int>(c.ple_multipliers.size()) >= c.ple_ngram &&
               static_cast<int>(c.ple_head_offsets.size()) >= c.ple_heads() &&
               static_cast<int>(c.ple_head_vocab.size()) >= c.ple_heads(),
           "PLE hash metadata");
    // M-RoPE sections: for text every position axis is equal, so they must cover exactly the n_rot / 2 pairs.
    const auto sections = required<std::vector<int32_t>>(f, key("rope.dimension_sections"));
    expect(std::accumulate(sections.begin(), sections.end(), 0) * 2 == c.n_rot, "rope.dimension_sections");
    return c;
}

}  // namespace torch::model
