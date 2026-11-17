#include "runtime/ple_rows.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>

#include "core/error.h"
#include "formats/dequant.h"

namespace torch::runtime {

namespace {

constexpr int kSpareBuffers = 4;  // for reads that lost to their hedge and are still in flight

}  // namespace

PleRows::PleRows(const model::Config& config, const gguf::File& model, const DirectFile& file, int columns)
    : config_(config), columns_(columns) {
    const gguf::TensorInfo* t = model.tensor("per_layer_token_embd.weight");
    if (t == nullptr || t->n_dims != 2 || t->shape[0] != config.ple_dim ||
        t->type != static_cast<uint32_t>(Type::IQ4_NL))
        throw Error("The model has no IQ4_NL per_layer_token_embd.weight of width " + std::to_string(config.ple_dim) +
                    "; use a supported model file with its n-gram table shard.");
    for (int h = 0; h < config.ple_heads(); ++h) {
        if (config.ple_head_offsets[h] + config.ple_head_vocab[h] > static_cast<uint64_t>(t->shape[1]))
            throw Error("The PLE head ranges in the metadata exceed the n-gram table; use matching shards.");
    }
    table_offset_ = model.data_offset() + t->offset;
    row_bytes_ = t->row_bytes();
    const int count = columns * config.ple_heads() + kSpareBuffers;
    buffers_ = static_cast<uint8_t*>(std::aligned_alloc(kPage, static_cast<size_t>(count) * 2 * kPage));
    if (buffers_ == nullptr)
        throw Error("Out of memory for the PLE read buffers.");
    // One thread per read of a column, up to 32: a window's reads are small and latency-bound.
    reads_ = std::make_unique<HedgedReads>(file, std::min(32, columns * config.ple_heads()));
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

void PleRows::read_window(std::span<const int32_t> history, size_t first, int count, float* out) {
    if (count < 1 || count > columns_)
        throw Error("A PLE read of " + std::to_string(count) + " tokens exceeds its " + std::to_string(columns_) +
                    " columns; this is a bug in the engine.");
    const size_t heads = static_cast<size_t>(config_.ple_heads()), n = heads * static_cast<size_t>(count);
    std::vector<HedgedReads::Request> requests(n);
    for (int t = 0; t < count; ++t) {
        const std::vector<uint64_t> rows = row_ids(config_, history, first + static_cast<size_t>(t));
        for (size_t h = 0; h < heads; ++h)
            requests[static_cast<size_t>(t) * heads + h] = {0, table_offset_ + rows[h] * row_bytes_, row_bytes_};
    }
    std::vector<const uint8_t*> data(n);
    reads_->read_all(requests, data);
    for (size_t r = 0; r < n; ++r) {
        if (data[r] == nullptr) {
            reads_->release();
            throw Error("A read of the PLE rows from the n-gram table failed; check the SSD and the file.");
        }
        dequant_row(Type::IQ4_NL, data[r], config_.ple_dim, out + r * static_cast<size_t>(config_.ple_dim));
    }
    reads_->release();
}

void PleRows::read_tokens(std::span<const int32_t> history, size_t first, int count, float* out) {
    const size_t heads = static_cast<size_t>(config_.ple_heads()), D = static_cast<size_t>(config_.ple_dim);
    const size_t n = heads * static_cast<size_t>(count);
    std::vector<uint64_t> unique;    // distinct row offsets, in order of first use
    std::vector<uint32_t> which(n);  // each (token, head)'s index in unique
    std::unordered_map<uint64_t, uint32_t> index;
    index.reserve(n);
    for (int t = 0; t < count; ++t) {
        const std::vector<uint64_t> rows = row_ids(config_, history, first + static_cast<size_t>(t));
        for (size_t h = 0; h < heads; ++h) {
            const uint64_t offset = table_offset_ + rows[h] * row_bytes_;
            const auto [it, fresh] = index.emplace(offset, static_cast<uint32_t>(unique.size()));
            if (fresh)
                unique.push_back(offset);
            which[static_cast<size_t>(t) * heads + h] = it->second;
        }
    }
    std::vector<float> values(unique.size() * D);
    const size_t batch = static_cast<size_t>(columns_) * heads;  // the reads one read_all may hold
    for (size_t b = 0; b < unique.size(); b += batch) {
        const size_t m = std::min(batch, unique.size() - b);
        std::vector<HedgedReads::Request> requests(m);
        for (size_t i = 0; i < m; ++i)
            requests[i] = {0, unique[b + i], row_bytes_};
        std::vector<const uint8_t*> data(m);
        reads_->read_all(requests, data);
        for (size_t i = 0; i < m; ++i) {
            if (data[i] == nullptr) {
                reads_->release();
                throw Error("A read of the PLE rows from the n-gram table failed; check the SSD and the file.");
            }
            dequant_row(Type::IQ4_NL, data[i], config_.ple_dim, values.data() + (b + i) * D);
        }
        reads_->release();
    }
    for (size_t r = 0; r < n; ++r)
        std::copy_n(values.data() + static_cast<size_t>(which[r]) * D, D, out + r * D);
}

}  // namespace torch::runtime
