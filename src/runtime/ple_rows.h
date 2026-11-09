// The per-layer embedding (PLE) input of a token: one row from each of 16 hashed n-gram tables, read from shard 2.
//
// The rows of head h hash the token and its predecessors (bigrams for heads 0-7, trigrams for 8-15) into a
// prime-sized slice of per_layer_token_embd, 320,001,536 IQ4_NL rows of 160 values (26.8 GiB) that stay on the SSD.
// A row is 90 bytes, so each lookup is one O_DIRECT read of the page (or two) holding it; the 16 reads run in
// parallel, hedged like the expert reads (hedged_reads.h: one 4 KiB read here once stalled for 30 s), and the rows are
// dequantized to F32 on the CPU. Architecture decision 9 adds a row cache and read-ahead in M5.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "gguf/gguf.h"
#include "model/config.h"
#include "runtime/direct_io.h"
#include "runtime/hedged_reads.h"

namespace torch::runtime {

class PleRows {
public:
    // Reads go through a pool of buffers for `columns` tokens at once (read_window).
    PleRows(const model::Config& config, const gguf::File& shard2, const DirectFile& file, int columns = 1);
    PleRows(const PleRows&) = delete;
    PleRows& operator=(const PleRows&) = delete;
    ~PleRows();

    // Writes the rows for token history[i] to out, ple_heads x ple_dim floats, head after head. Throws Error when a
    // read fails.
    void read(std::span<const int32_t> history, size_t i, float* out) { read_window(history, i, 1, out); }

    // Writes the rows of history[first] .. history[first + count - 1] to out, one token after another (ple_heads x
    // ple_dim floats each), issuing all count x ple_heads reads at once. count is at most the constructor's columns.
    void read_window(std::span<const int32_t> history, size_t first, int count, float* out);

    // The table row each head reads for history[i]. Predecessors before the start of the history read as the PLE
    // end-of-sequence token, and an end of sequence among the predecessors cuts the window there (the token's own
    // end of sequence does not).
    static std::vector<uint64_t> row_ids(const model::Config& config, std::span<const int32_t> history, size_t i);

    uint64_t hedged_reads() const noexcept { return reads_->hedged(); }

private:
    const model::Config& config_;
    uint64_t table_offset_ = 0;  // of row 0 in shard 2
    uint64_t row_bytes_ = 0;
    int columns_ = 1;
    uint8_t* buffers_ = nullptr;  // page-aligned two-page buffers: one per (column, head) and kSpareBuffers
    std::unique_ptr<HedgedReads> reads_;
};

}  // namespace torch::runtime
