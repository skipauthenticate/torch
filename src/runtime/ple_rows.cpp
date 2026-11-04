#include "runtime/ple_rows.h"

#include <cstdlib>

#include "core/error.h"
#include "formats/dequant.h"

namespace torch::runtime {

PleRows::PleRows(const model::Config& config, const gguf::File& shard2, const DirectFile& file, ReadPool& pool)
    : config_(config), file_(file), pool_(pool) {
    const gguf::TensorInfo* t = shard2.tensor("per_layer_token_embd.weight");
    if (t == nullptr || t->n_dims != 2 || t->shape[0] != config.ple_dim ||
        t->type != static_cast<uint32_t>(Type::IQ4_NL))
        throw Error("Shard 2 has no IQ4_NL per_layer_token_embd.weight of width " + std::to_string(config.ple_dim) +
                    "; pass the n-gram table shard of the supported model.");
    for (int h = 0; h < config.ple_heads(); ++h) {
        if (config.ple_head_offsets[h] + config.ple_head_vocab[h] > static_cast<uint64_t>(t->shape[1]))
            throw Error("The PLE head ranges in the metadata exceed the n-gram table; use matching shards.");
    }
    table_offset_ = shard2.data_offset() + t->offset;
    row_bytes_ = t->row_bytes();
    for (int h = 0; h < config.ple_heads(); ++h) {
        auto* buffer = static_cast<uint8_t*>(std::aligned_alloc(kPage, 2 * kPage));
        if (buffer == nullptr)
            throw Error("Out of memory for the PLE read buffers.");
        buffers_.push_back(buffer);
    }
}

PleRows::~PleRows() {
    for (uint8_t* b : buffers_)
        std::free(b);
}

std::vector<uint64_t> PleRows::row_ids(const model::Config& c, std::span<const int32_t> history, size_t i) {
    const int n_gram = c.ple_ngram;
    std::vector<uint64_t> context(static_cast<size_t>(n_gram));
    context[0] = static_cast<uint64_t>(history[i]);
    bool cut = false;
    for (int back = 1; back < n_gram; ++back) {
        const int64_t t = cut || i < static_cast<size_t>(back) ? -1 : history[i - static_cast<size_t>(back)];
        cut = cut || t < 0 || t == static_cast<int64_t>(c.ple_eos);
        context[static_cast<size_t>(back)] = cut ? c.ple_eos : static_cast<uint64_t>(t);
    }
    std::vector<uint64_t> rows(static_cast<size_t>(c.ple_heads()));
    for (int n = 2; n <= n_gram; ++n) {
        uint64_t mixed = context[0] * c.ple_multipliers[0];  // wraps modulo 2^64, as the reference's uint64_t
        for (int j = 1; j < n; ++j)
            mixed ^= context[static_cast<size_t>(j)] * c.ple_multipliers[static_cast<size_t>(j)];
        for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
            const size_t h = static_cast<size_t>((n - 2) * c.ple_heads_per_ngram + g);
            rows[h] = mixed % c.ple_head_vocab[h] + c.ple_head_offsets[h];
        }
    }
    return rows;
}

void PleRows::read(std::span<const int32_t> history, size_t i, float* out) {
    const std::vector<uint64_t> rows = row_ids(config_, history, i);
    pool_.run(config_.ple_heads(), [&](int h) {
        const uint8_t* row = file_.read(table_offset_ + rows[static_cast<size_t>(h)] * row_bytes_, row_bytes_,
                                        buffers_[static_cast<size_t>(h)]);
        dequant_row(Type::IQ4_NL, row, config_.ple_dim,
                    out + static_cast<size_t>(h) * static_cast<size_t>(config_.ple_dim));
    });
}

}  // namespace torch::runtime
