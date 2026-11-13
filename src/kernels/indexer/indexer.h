// The sparse attention indexer of the 12 trunk attention layers (docs/architecture.md decision 10), for the columns of
// a verify window. Each layer scores the pooled keys of the visible 4-token blocks with its own 4 query heads of 128
// values, keeps the 512 best blocks, and the attention then reads only those blocks' cells plus the 0-3 open tail
// cells. Up to 2,051 tokens (512 complete blocks and a tail of 3) every block is selected, so the attention is exactly
// dense; from 2,052 the lowest-scoring blocks are dropped.
//
// Column t of a window at position p = *pos is the token at position p + t. With n = p + t + 1 visible tokens it sees
// nb = n / 4 complete blocks (block b = positions 4b .. 4b + 3), so a column that closes a block scores it, and the
// tail is the n % 4 cells after them. Everything the indexer keeps is indexed by position and written before it is
// read: the 16-row ring of raw keys (row = position % 16) and the pooled key of block b (row b). A rejected column's
// rows are rewritten by the next window before any column reads them, so a window's commit has nothing to do for the
// indexer.
//
// Numerics (F32 unless stated; the CPU reference in tests/parity/indexer_test.cpp does the same operations):
//  - raw key: the projection's output rounded to F16, no norm, no rotary embedding (stored in the ring);
//  - pooled key of block b: mean = ((k0 + k1) + k2 + k3) * 0.25f of the four F16 raw keys, then the RMS norm over its
//    128 values (weight w, scale rsqrtf(sum(x^2) / 128 + eps), out = (x * scale) * w; the weights already hold the +1),
//    then the 64-dimension NeoX rotary embedding at position 4b, stored as F16;
//  - query head h: the norm and the rotary embedding at position p + t on the projection's 128 values, kept in F32;
//  - score = ((0 + relu(d0) * s) + relu(d1) * s) + relu(d2) * s) + relu(d3) * s, s = 1 / sqrt(128) as F32, each d_h the
//    dot of query head h with the F16 pooled key accumulated by fmaf over the 128 dimensions in order, each product and
//    sum rounded separately (no fusion); then + 0.0f, which turns -0.0 into +0.0, so every score is a float >= 0 whose
//    bit pattern orders like its value;
//  - selection: the 512 highest scores, ties to the lower block index; with at most 512 visible blocks, all of them.
//
// Parity test: tests/parity/indexer_test.cpp. Bytes moved per window and layer: the pooled keys once per column
// (256 B per block; 0.2 GB per column at 262,144 tokens over 12 layers, from the L2 when they fit).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

inline constexpr int kIndexHeads = 4;
inline constexpr int kIndexDim = 128;
inline constexpr int kIndexPool = 4;                                                  // tokens per block
inline constexpr int kIndexTopBlocks = 512;                                           // blocks kept
inline constexpr int kIndexRing = 16;                                                 // raw key rows kept per layer
inline constexpr int kIndexMaxCells = kIndexTopBlocks * kIndexPool + kIndexPool - 1;  // 2,051

// What the attention reads for each column of a window: the selected blocks in ascending order and the cell count.
// Cell i of column t is block blocks[t][i / 4] * 4 + i % 4 for i < 4 * nsel, then the tail cells in position order.
struct IndexCells {
    int32_t* blocks;  // [kMaxColumns][kIndexTopBlocks]
    int32_t* info;    // [kMaxColumns][4]: nsel, first tail position, cell count, unused
};

// Floats of score scratch for a context of max_ctx tokens and `columns` columns.
int64_t index_score_floats(int max_ctx, int columns);

// Per column: the projection's 4 query heads q_raw[t * 512 ..] are normed (weight q_norm) and rotated at position
// *pos + t into q_out[t * 512 ..], and the raw key k_raw[t * 128 ..] is rounded to F16 into ring row (*pos + t) % 16.
// n_rot = 64 dimensions rotate (NeoX pairs i, i + n_rot / 2), base is the rotary base, eps the norm's.
void index_prepare(const float* q_raw, const float* k_raw, const float* q_norm, float* q_out, uint16_t* ring,
                   const int32_t* pos, int n_rot, float base, float eps, int columns, cudaStream_t stream);

// Pools the blocks the window completes (those whose last position lies in *pos .. *pos + columns - 1) from the ring
// into pooled[b * 128 ..] (F16), with weight k_norm.
void index_pool(const uint16_t* ring, const float* k_norm, uint16_t* pooled, const int32_t* pos, int n_rot, float base,
                float eps, int columns, cudaStream_t stream);

// scores[t * max_blocks + b] for every column t with more than 512 visible blocks and each of its visible blocks
// b < (*pos + t + 1) / 4 (other entries are not written). max_blocks = max_ctx / 4. Any number of columns: above 8 they
// go in groups of 8, group g reading its first position from pos[8 g], so pos must then hold every column's position
// (pos[t] = *pos + t), as the window's and a chunk's position arrays do.
void index_score(const float* q, const uint16_t* pooled, float* scores, const int32_t* pos, int max_blocks, int columns,
                 cudaStream_t stream);

// A prefill chunk of `columns` tokens at positions *pos .. *pos + columns - 1 (pos holds them all, pos[t] = *pos + t):
// index_prepare with the raw keys going to raw[t * 128 ..] (F16) instead of the ring, then index_pool_chunk, which
// pools every block the chunk completes (members before the chunk from the ring) and leaves the chunk's last 16 raw
// keys in the ring. Each pooled key is index_pool's bit for bit, so a chunk equals feeding the same tokens in windows.
// Scoring, selection and attention then run on batches of the chunk's columns (pos + first column of the batch).
void index_prepare_chunk(const float* q_raw, const float* k_raw, const float* q_norm, float* q_out, uint16_t* raw,
                         const int32_t* pos, int n_rot, float base, float eps, int columns, cudaStream_t stream);
void index_pool_chunk(uint16_t* ring, const uint16_t* raw, const float* k_norm, uint16_t* pooled, const int32_t* pos,
                      int n_rot, float base, float eps, int columns, cudaStream_t stream);

// Writes every column's cell list: all visible blocks when there are at most 512, else the top 512 by score.
void index_select(const float* scores, const IndexCells& cells, const int32_t* pos, int max_blocks, int columns,
                  cudaStream_t stream);

}  // namespace torch::kernels
