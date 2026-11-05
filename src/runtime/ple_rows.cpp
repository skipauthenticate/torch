#include "runtime/ple_rows.h"

#include <cstdlib>

#include "core/error.h"
#include "formats/dequant.h"

namespace torch::runtime {

namespace {

constexpr int kSpareBuffers = 4;  // for reads that lost to their hedge and are still in flight

}  // namespace

PleRows::PleRows(const model::Config& config, const gguf::File& shard2, const DirectFile& file) : config_(config) {
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
    const int count = config.ple_heads() + kSpareBuffers;
    buffers_ = static_cast<uint8_t*>(std::aligned_alloc(kPage, static_cast<size_t>(count) * 2 * kPage));
    if (buffers_ == nullptr)
        throw Error("Out of memory for the PLE read buffers.");
    reads_ = std::make_unique<HedgedReads>(file, config.ple_heads());
    reads_->add_pool(buffers_, 2 * kPage, count);
}

PleRows::~PleRows() {
    reads_.reset();  // waits for reads still in flight before their buffers are freed
    std::free(buffers_);
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
    const size_t heads = rows.size();
    std::vector<HedgedReads::Request> requests(heads);
    for (size_t h = 0; h < heads; ++h)
        requests[h] = {0, table_offset_ + rows[h] * row_bytes_, row_bytes_};
    std::vector<const uint8_t*> data(heads);
    reads_->read_all(requests, data);
    for (size_t h = 0; h < heads; ++h) {
        if (data[h] == nullptr) {
            reads_->release();
            throw Error("A read of the PLE rows from shard 2 failed; check the SSD and the file.");
        }
        dequant_row(Type::IQ4_NL, data[h], config_.ple_dim, out + h * static_cast<size_t>(config_.ple_dim));
    }
    reads_->release();
}

}  // namespace torch::runtime
