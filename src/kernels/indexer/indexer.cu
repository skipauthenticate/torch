#include <algorithm>

#include "kernels/common.cuh"
#include "kernels/indexer/indexer.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = kIndexDim;  // one thread per dimension in the prepare and pool kernels
constexpr int kSelectThreads = 1024;

__device__ __forceinline__ float f16_load(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// The rotary embedding of dimension i < n_rot / 2 of a head held in shared memory (head[i], head[i + n_rot / 2]),
// at `position`: the expressions of rope_kernel (kernels/attention).
__device__ __forceinline__ void rotate(float* head, int i, int n_rot, int position, float base) {
    const int half_rot = n_rot / 2;
    const float theta_scale = powf(base, -2.0f / static_cast<float>(n_rot));
    const float theta = static_cast<float>(position) * powf(theta_scale, static_cast<float>(i));
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = head[i], x1 = head[i + half_rot];
    head[i] = x0 * c - x1 * s;
    head[i + half_rot] = x0 * s + x1 * c;
}

// RMS norm with weight over the kIndexDim values in `head`, thread i owning value i (norm_kernel's expressions).
__device__ __forceinline__ float norm_value(float x, const float* w, float eps) {
    float sum = 0.0f;
    sum += x * x;
    sum = block_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(kIndexDim) + eps);
    float v = x * scale;
    return v * w[threadIdx.x];
}

// Block (head, column): heads 0-3 are the query heads, head 4 the raw key (to the ring, or to raw[t] for a chunk).
__global__ void __launch_bounds__(kThreads)
    index_prepare_kernel(const float* q_raw, const float* k_raw, const float* q_norm, float* q_out, uint16_t* ring,
                         uint16_t* raw, const int32_t* pos, int n_rot, float base, float eps) {
    __shared__ float head[kIndexDim];
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int position = *pos + t;
    if (h == kIndexHeads) {
        const uint16_t key = __half_as_ushort(__float2half_rn(k_raw[int64_t{t} * kIndexDim + i]));
        if (raw)
            raw[int64_t{t} * kIndexDim + i] = key;
        else
            ring[(position % kIndexRing) * kIndexDim + i] = key;
        return;
    }
    const int64_t at = (int64_t{t} * kIndexHeads + h) * kIndexDim + i;
    head[i] = norm_value(q_raw[at], q_norm, eps);
    __syncthreads();
    if (i < n_rot / 2)
        rotate(head, i, n_rot, position, base);
    __syncthreads();
    q_out[at] = head[i];
}

// Block j pools block b = *pos / 4 + j when its last position 4b + 3 is inside the window.
__global__ void __launch_bounds__(kThreads)
    index_pool_kernel(const uint16_t* ring, const float* k_norm, uint16_t* pooled, const int32_t* pos, int n_rot,
                      float base, float eps, int columns) {
    __shared__ float head[kIndexDim];
    const int p = *pos, b = p / kIndexPool + blockIdx.x, i = threadIdx.x;
    if (kIndexPool * b + kIndexPool - 1 < p || kIndexPool * b + kIndexPool - 1 > p + columns - 1)
        return;  // the whole block
    float s = f16_load(ring[((kIndexPool * b) % kIndexRing) * kIndexDim + i]);
    for (int m = 1; m < kIndexPool; ++m)
        s += f16_load(ring[((kIndexPool * b + m) % kIndexRing) * kIndexDim + i]);
    head[i] = norm_value(s * 0.25f, k_norm, eps);
    __syncthreads();
    if (i < n_rot / 2)
        rotate(head, i, n_rot, kIndexPool * b, base);
    __syncthreads();
    pooled[int64_t{b} * kIndexDim + i] = __half_as_ushort(__float2half_rn(head[i]));
}

// A chunk's pooling: block j pools block b = *pos / 4 + j when its last position 4b + 3 is inside the chunk, members
// before the chunk from the ring and the others from raw[position - *pos]; the expressions of index_pool_kernel.
__global__ void __launch_bounds__(kThreads)
    index_pool_chunk_kernel(const uint16_t* ring, const uint16_t* raw, const float* k_norm, uint16_t* pooled,
                            const int32_t* pos, int n_rot, float base, float eps, int columns) {
    __shared__ float head[kIndexDim];
    const int p = *pos, b = p / kIndexPool + blockIdx.x, i = threadIdx.x;
    if (kIndexPool * b + kIndexPool - 1 < p || kIndexPool * b + kIndexPool - 1 > p + columns - 1)
        return;  // the whole block
    const auto member = [&](int m) {
        const int q = kIndexPool * b + m;
        return f16_load(q < p ? ring[(q % kIndexRing) * kIndexDim + i] : raw[int64_t{q - p} * kIndexDim + i]);
    };
    float s = member(0);
    for (int m = 1; m < kIndexPool; ++m)
        s += member(m);
    head[i] = norm_value(s * 0.25f, k_norm, eps);
    __syncthreads();
    if (i < n_rot / 2)
        rotate(head, i, n_rot, kIndexPool * b, base);
    __syncthreads();
    pooled[int64_t{b} * kIndexDim + i] = __half_as_ushort(__float2half_rn(head[i]));
}

// Block j: the chunk's raw key of column columns - min(columns, 16) + j into its ring row.
__global__ void __launch_bounds__(kThreads)
    index_ring_store_kernel(const uint16_t* raw, uint16_t* ring, const int32_t* pos, int columns) {
    const int t = columns - min(columns, kIndexRing) + static_cast<int>(blockIdx.x);
    ring[((*pos + t) % kIndexRing) * kIndexDim + threadIdx.x] = raw[int64_t{t} * kIndexDim + threadIdx.x];
}

// Windows of up to kScoreSmall columns: one thread per (block, column), the four heads' scores of a pooled key against
// the column's query (the key read once per column: the fewest bytes in flight per score, best at 1 and 2 columns).
__global__ void __launch_bounds__(128) index_score_columns_kernel(const float* q, const uint16_t* pooled, float* scores,
                                                                  const int32_t* pos, int max_blocks) {
    __shared__ float query[kIndexHeads * kIndexDim];
    const int t = blockIdx.y;
    const int visible = (*pos + t + 1) / kIndexPool;
    if (visible <= kIndexTopBlocks)
        return;  // everything is selected, nothing is scored
    for (int i = threadIdx.x; i < kIndexHeads * kIndexDim; i += blockDim.x)
        query[i] = q[t * kIndexHeads * kIndexDim + i];
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= visible)
        return;
    const uint4* key = reinterpret_cast<const uint4*>(pooled + int64_t{b} * kIndexDim);
    float k[kIndexDim];
#pragma unroll
    for (int v = 0; v < kIndexDim / 8; ++v) {
        const uint4 raw = key[v];
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 pair =
                __half22float2(__halves2half2(__ushort_as_half(words[w] & 0xffff), __ushort_as_half(words[w] >> 16)));
            k[8 * v + 2 * w] = pair.x, k[8 * v + 2 * w + 1] = pair.y;
        }
    }
    const float weight = 1.0f / sqrtf(static_cast<float>(kIndexDim));
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < kIndexHeads; ++h) {
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < kIndexDim; ++d)
            dot = __fmaf_rn(query[h * kIndexDim + d], k[d], dot);
        score = __fadd_rn(score, __fmul_rn(fmaxf(dot, 0.0f), weight));
    }
    scores[int64_t{t} * max_blocks + b] = __fadd_rn(score, 0.0f);
}

constexpr int kScoreSmall = 2;

// Wider windows: scores in tiles of kScoreTile blocks, as many blocks as fit on the SMs at once looping over the tiles
// (each key read once for all columns). Lane t * 4 + h of
// every warp keeps query head h of column t in registers; warp w scores blocks 4w .. 4w + 3 of a tile, four dot
// products at once, against the tile's keys staged as F32 in shared memory (the next tile's keys load while this one is
// scored). Each dot is the sequential fmaf over the 128 dimensions and the four head lanes of a column add in head
// order through shuffles, so every score is the one-thread-per-(block, column) expression bit for bit. Lanes of columns
// past `columns` idle.
constexpr int kScoreWarps = 4;
constexpr int kScoreTile = 4 * kScoreWarps;
constexpr int kKeyStride = kIndexDim + 4;                                       // floats per staged key row
constexpr int kTileLoads = kScoreTile * kIndexDim / 8 / (kScoreWarps * kWarp);  // 16-byte loads per thread per tile
static_assert(kIndexHeads * kMaxColumns == kWarp, "a warp's lanes are the (column, head) pairs");
static_assert(kTileLoads * kScoreWarps * kWarp * 8 == kScoreTile * kIndexDim, "the threads load a tile evenly");

__global__ void __launch_bounds__(kScoreWarps* kWarp)
    index_score_kernel(const float* q, const uint16_t* pooled, float* scores, const int32_t* pos, int max_blocks,
                       int columns) {
    __shared__ __align__(16) float keys[2][kScoreTile * kKeyStride];
    const int p = *pos;
    const int last_visible = (p + columns) / kIndexPool;  // the last column's visible blocks
    if (last_visible <= kIndexTopBlocks)
        return;  // everything is selected, nothing is scored
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    const int t = lane / kIndexHeads, h = lane % kIndexHeads;
    const bool active = t < columns;
    const int visible = (p + t + 1) / kIndexPool;
    float query[kIndexDim];
#pragma unroll
    for (int d = 0; d < kIndexDim; d += 4) {
        const float4 v = active ? *reinterpret_cast<const float4*>(q + (t * kIndexHeads + h) * kIndexDim + d)
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        query[d] = v.x, query[d + 1] = v.y, query[d + 2] = v.z, query[d + 3] = v.w;
    }
    const float weight = 1.0f / sqrtf(static_cast<float>(kIndexDim));
    // Piece i of a tile is 8 halves: block i / 16, dimensions 8 (i % 16) ..; thread x loads pieces x, x + 256, ...
    uint4 raw[kTileLoads];
    const auto load = [&](int first) {
#pragma unroll
        for (int k = 0; k < kTileLoads; ++k) {
            const int i = threadIdx.x + k * kScoreWarps * kWarp, b = min(first + i / 16, last_visible - 1);
            raw[k] = reinterpret_cast<const uint4*>(pooled + int64_t{b} * kIndexDim)[i % 16];
        }
    };
    const auto stage = [&](float* to) {
#pragma unroll
        for (int k = 0; k < kTileLoads; ++k) {
            const int i = threadIdx.x + k * kScoreWarps * kWarp;
            float* row = to + (i / 16) * kKeyStride + (i % 16) * 8;
            const uint32_t words[4] = {raw[k].x, raw[k].y, raw[k].z, raw[k].w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const float2 pair = __half22float2(
                    __halves2half2(__ushort_as_half(words[w] & 0xffff), __ushort_as_half(words[w] >> 16)));
                row[2 * w] = pair.x, row[2 * w + 1] = pair.y;
            }
        }
    };
    int first = blockIdx.x * kScoreTile, buffer = 0;
    if (first < last_visible) {
        load(first);
        stage(keys[0]);
    }
    for (; first < last_visible; first += gridDim.x * kScoreTile, buffer ^= 1) {
        const int next = first + gridDim.x * kScoreTile;
        if (next < last_visible)
            load(next);   // in flight while this tile is scored
        __syncthreads();  // this tile's keys are staged
        const float* tile = keys[buffer];
        float dot[4] = {};
#pragma unroll
        for (int d = 0; d < kIndexDim; d += 4) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float4 key = *reinterpret_cast<const float4*>(&tile[(4 * warp + k) * kKeyStride + d]);
                dot[k] = __fmaf_rn(query[d], key.x, dot[k]);
                dot[k] = __fmaf_rn(query[d + 1], key.y, dot[k]);
                dot[k] = __fmaf_rn(query[d + 2], key.z, dot[k]);
                dot[k] = __fmaf_rn(query[d + 3], key.w, dot[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float r = __fmul_rn(fmaxf(dot[k], 0.0f), weight);
            const int base = lane - h;  // head 0's lane of this column
            const float r0 = __shfl_sync(0xffffffffu, r, base), r1 = __shfl_sync(0xffffffffu, r, base + 1);
            const float r2 = __shfl_sync(0xffffffffu, r, base + 2), r3 = __shfl_sync(0xffffffffu, r, base + 3);
            const int block = first + 4 * warp + k;
            if (active && h == 0 && visible > kIndexTopBlocks && block < visible) {
                float score = 0.0f;
                score = __fadd_rn(score, r0), score = __fadd_rn(score, r1);
                score = __fadd_rn(score, r2), score = __fadd_rn(score, r3);
                scores[int64_t{t} * max_blocks + block] = __fadd_rn(score, 0.0f);
            }
        }
        if (next < last_visible)
            stage(keys[buffer ^ 1]);  // the other buffer: its last readers passed this tile's barrier
    }
}

static_assert(kSelectThreads == kWarp * kWarp, "block_scan scans the warp totals in one warp");

// Exclusive prefix sum of one int per thread over the block (integer sums, so any order gives the same result), and the
// block's total in *total when given; `buffer` holds kWarp ints.
__device__ int block_scan(int* buffer, int value, int* total = nullptr) {
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    int inclusive = value;
    for (int offset = 1; offset < kWarp; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset)
            inclusive += other;
    }
    if (lane == kWarp - 1)
        buffer[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int total = buffer[lane];
        for (int offset = 1; offset < kWarp; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, total, offset);
            if (lane >= offset)
                total += other;
        }
        buffer[lane] = total;
    }
    __syncthreads();
    const int before = inclusive - value + (warp > 0 ? buffer[warp - 1] : 0);
    if (total)
        *total = buffer[kWarp - 1];
    __syncthreads();  // the next call reuses the buffer
    return before;
}

// Block t: column t's cell list. The radix passes count a warp's equal bytes with one shared atomic (scores share their
// top bytes, so per-key atomics would queue on a few bins); the compaction then gives each thread a run of consecutive
// blocks per chunk, so selected blocks come out in ascending order after one scan per chunk.
__global__ void __launch_bounds__(kSelectThreads)
    index_select_kernel(const float* scores, IndexCells cells, const int32_t* pos, int max_blocks) {
    __shared__ unsigned bins[256];
    __shared__ int scan[kWarp];
    __shared__ unsigned prefix, remaining;
    const int t = blockIdx.x, tid = threadIdx.x;
    const int n = *pos + t + 1, visible = n / kIndexPool;
    int32_t* const out = cells.blocks + t * kIndexTopBlocks;
    if (tid == 0) {
        const int nsel = visible < kIndexTopBlocks ? visible : kIndexTopBlocks;
        cells.info[4 * t] = nsel;
        cells.info[4 * t + 1] = kIndexPool * visible;
        cells.info[4 * t + 2] = kIndexPool * nsel + (n - kIndexPool * visible);
    }
    if (visible <= kIndexTopBlocks) {
        for (int i = tid; i < visible; i += kSelectThreads)
            out[i] = i;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + int64_t{t} * max_blocks);
    // Radix select of the 512th highest key, a byte at a time from the top.
    if (tid == 0)
        prefix = 0, remaining = kIndexTopBlocks;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kSelectThreads)
            bins[i] = 0;
        __syncthreads();
        const unsigned high = shift == 24 ? 0u : prefix;                  // the bytes already fixed
        for (int base = 0; base < visible; base += 4 * kSelectThreads) {  // every lane of a warp runs every round
            unsigned k[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {  // the four loads first, so they overlap
                const int b = base + tid + j * kSelectThreads;
                k[j] = b < visible ? key[b] : 0u;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool counted =
                    base + tid + j * kSelectThreads < visible && (shift == 24 || (k[j] >> (shift + 8)) == high);
                const unsigned bin = counted ? (k[j] >> shift) & 255u : 256u;
                const unsigned peers = __match_any_sync(0xffffffffu, bin);
                if (counted && __ffs(peers) - 1 == tid % kWarp)
                    atomicAdd(&bins[bin], static_cast<unsigned>(__popc(peers)));
            }
        }
        __syncthreads();
        // Thread i counts bin 255 - i; the bin holding the wanted key is the one where the keys above it fall short of
        // `want` and the keys up to it reach it.
        const int want = static_cast<int>(remaining), count = tid < 256 ? static_cast<int>(bins[255 - tid]) : 0;
        const int above = block_scan(scan, count);
        if (tid < 256 && above < want && above + count >= want) {
            remaining = static_cast<unsigned>(want - above);  // how many of this byte's keys are still wanted
            prefix = (shift == 24 ? 0u : prefix << 8) | static_cast<unsigned>(255 - tid);
        }
        __syncthreads();
    }
    const unsigned threshold = prefix;  // the 512th highest key; the first `remaining` keys equal to it are kept
    const int quota = static_cast<int>(remaining);
    int written = 0, ties = 0;  // over the chunks so far, the same in every thread
    // consecutive blocks per thread in a chunk: as few as cover the blocks in one chunk, at most 16
    const int run = min(16, (visible + kSelectThreads - 1) / kSelectThreads);
    for (int base = 0; base < visible; base += run * kSelectThreads) {
        const int first = base + tid * run, last = min(first + run, visible);
        int above = 0, tie = 0;
        for (int b = first; b < last; ++b)
            above += key[b] > threshold, tie += key[b] == threshold;
        // one scan of both counts: keys above the threshold in the low 16 bits, ties in the high ones
        int total = 0;
        const int before = block_scan(scan, above | tie << 16, &total);
        const int ties_left = max(quota - ties, 0);  // ties still kept, first come
        int above_before = before & 0xffff, ties_before = before >> 16;
        for (int b = first; b < last; ++b) {
            const unsigned k = key[b];
            if (k > threshold)
                out[written + above_before++ + min(ties_before, ties_left)] = b;
            else if (k == threshold && ties_before++ < ties_left)
                out[written + above_before + ties_before - 1] = b;
        }
        written += (total & 0xffff) + min(total >> 16, ties_left);
        ties += total >> 16;
    }
}

}  // namespace

int64_t index_score_floats(int max_ctx, int columns) {
    return int64_t{max_ctx / kIndexPool} * columns;
}

void index_prepare(const float* q_raw, const float* k_raw, const float* q_norm, float* q_out, uint16_t* ring,
                   const int32_t* pos, int n_rot, float base, float eps, int columns, cudaStream_t stream) {
    index_prepare_kernel<<<dim3(kIndexHeads + 1, columns), kThreads, 0, stream>>>(q_raw, k_raw, q_norm, q_out, ring,
                                                                                  nullptr, pos, n_rot, base, eps);
}

void index_prepare_chunk(const float* q_raw, const float* k_raw, const float* q_norm, float* q_out, uint16_t* raw,
                         const int32_t* pos, int n_rot, float base, float eps, int columns, cudaStream_t stream) {
    index_prepare_kernel<<<dim3(kIndexHeads + 1, columns), kThreads, 0, stream>>>(q_raw, k_raw, q_norm, q_out, nullptr,
                                                                                  raw, pos, n_rot, base, eps);
}

void index_pool_chunk(uint16_t* ring, const uint16_t* raw, const float* k_norm, uint16_t* pooled, const int32_t* pos,
                      int n_rot, float base, float eps, int columns, cudaStream_t stream) {
    index_pool_chunk_kernel<<<(columns + kIndexPool - 1) / kIndexPool + 1, kThreads, 0, stream>>>(
        ring, raw, k_norm, pooled, pos, n_rot, base, eps, columns);
    // after the pooling: the rows it overwrites may be the ones the block straddling the chunk start read
    index_ring_store_kernel<<<std::min(columns, kIndexRing), kThreads, 0, stream>>>(raw, ring, pos, columns);
}

void index_pool(const uint16_t* ring, const float* k_norm, uint16_t* pooled, const int32_t* pos, int n_rot, float base,
                float eps, int columns, cudaStream_t stream) {
    index_pool_kernel<<<(columns + kIndexPool - 1) / kIndexPool + 1, kThreads, 0, stream>>>(ring, k_norm, pooled, pos,
                                                                                            n_rot, base, eps, columns);
}

void index_score(const float* q, const uint16_t* pooled, float* scores, const int32_t* pos, int max_blocks, int columns,
                 cudaStream_t stream) {
    static const int resident = [] {  // blocks that run at once: several tiles' key loads overlap
        int device = 0, sms = 0, per_sm = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, index_score_kernel, kScoreWarps * kWarp, 0);
        return sms * std::max(per_sm, 1);
    }();
    // a chunk's columns in groups of kMaxColumns (a warp's lanes are 8 columns' 4 heads); both kernels give the same
    // bits
    for (int g = 0; g < columns; g += kMaxColumns) {
        const int n = std::min(kMaxColumns, columns - g);
        const float* const qg = q + int64_t{g} * kIndexHeads * kIndexDim;
        float* const sg = scores + int64_t{g} * max_blocks;
        if (n <= kScoreSmall) {
            index_score_columns_kernel<<<dim3((max_blocks + 127) / 128, n), 128, 0, stream>>>(qg, pooled, sg, pos + g,
                                                                                              max_blocks);
        } else {
            const int tiles = (max_blocks + kScoreTile - 1) / kScoreTile;
            index_score_kernel<<<std::min(tiles, resident), kScoreWarps * kWarp, 0, stream>>>(qg, pooled, sg, pos + g,
                                                                                              max_blocks, n);
        }
    }
}

void index_select(const float* scores, const IndexCells& cells, const int32_t* pos, int max_blocks, int columns,
                  cudaStream_t stream) {
    index_select_kernel<<<columns, kSelectThreads, 0, stream>>>(scores, cells, pos, max_blocks);
}

}  // namespace torch::kernels
