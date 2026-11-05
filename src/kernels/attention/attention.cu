#include "core/cuda_check.h"
#include "core/error.h"
#include "kernels/attention/attention.h"
#include "kernels/common.cuh"

namespace torch::kernels {
namespace {

constexpr int kHeadDim = 256;  // one thread per dimension in attention_decode
constexpr int kThreads = 256;
constexpr int kGroup = 12;   // query heads per KV head
constexpr int kChunk = 128;  // positions per pass-1 block
constexpr int kChunkThreads = 256;
constexpr int kQuadRows = kChunk * 4 / kChunkThreads;  // rows per quad of threads

static_assert(kQuadRows == 2, "pass 1 gives each quad of threads two rows of a chunk");

int chunk_count(int max_ctx) {
    return (max_ctx + kChunk - 1) / kChunk;
}

// Scratch layout: the chunk outputs [heads][chunks][256], then the (maximum, sum) pairs [heads][chunks][2].
__host__ __device__ constexpr int64_t heads_scratch_offset(int heads, int chunks) {
    return int64_t{heads} * chunks * kHeadDim;
}

// Block h, thread i < n_rot / 2.
__global__ void rope_kernel(float* x, int64_t head_stride, int n_rot, const int32_t* pos, float base) {
    const int i = threadIdx.x;
    float* xh = x + blockIdx.x * head_stride;
    const float theta_scale = powf(base, -2.0f / static_cast<float>(n_rot));
    const float theta = static_cast<float>(*pos) * powf(theta_scale, static_cast<float>(i));
    const float c = cosf(theta), s = sinf(theta);
    const int half_rot = n_rot / 2;
    const float x0 = xh[i], x1 = xh[i + half_rot];
    xh[i] = x0 * c - x1 * s;
    xh[i + half_rot] = x0 * s + x1 * c;
}

__global__ void store_kv_kernel(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int n,
                                const int32_t* pos) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n)
        return;
    const int64_t at = int64_t{*pos} * n + i;
    k_cache[at] = __half_as_ushort(__float2half_rn(k[i]));
    v_cache[at] = __half_as_ushort(__float2half_rn(v[i]));
}

// One block per head, thread i owns element i of a 256-element head: q heads (blocks 0..heads-1), then k heads. The
// steps and expressions are those of norm_kernel (RMS norm with weight, same block_sum), rope_kernel and
// store_kv_kernel, so the results are bitwise those of the five launches.
__global__ void __launch_bounds__(kThreads)
    qk_norm_rope_store_kernel(float* q, int64_t q_stride, const float* q_weight, float* k, const float* k_weight,
                              const float* v, uint16_t* k_cache, uint16_t* v_cache, int heads, int n_rot,
                              const int32_t* pos, float base, float eps) {
    __shared__ float head[kHeadDim];
    const int b = blockIdx.x, i = threadIdx.x;
    const bool is_q = b < heads;
    float* x = is_q ? q + b * q_stride : k + (b - heads) * kHeadDim;
    const float* w = is_q ? q_weight : k_weight;
    float sum = 0.0f;
    for (int e = i; e < kHeadDim; e += kThreads)
        sum += x[e] * x[e];
    sum = block_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(kHeadDim) + eps);
    float value = x[i] * scale;
    value = value * w[i];
    head[i] = value;
    __syncthreads();
    const int half_rot = n_rot / 2;
    if (i < half_rot) {
        const float theta_scale = powf(base, -2.0f / static_cast<float>(n_rot));
        const float theta = static_cast<float>(*pos) * powf(theta_scale, static_cast<float>(i));
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = head[i], x1 = head[i + half_rot];
        head[i] = x0 * c - x1 * s;
        head[i + half_rot] = x0 * s + x1 * c;
    }
    __syncthreads();
    x[i] = head[i];
    if (!is_q) {
        const int64_t at = int64_t{*pos} * (gridDim.x - heads) * kHeadDim + (b - heads) * kHeadDim + i;
        k_cache[at] = __half_as_ushort(__float2half_rn(head[i]));
        v_cache[at] = __half_as_ushort(__float2half_rn(v[(b - heads) * kHeadDim + i]));
    }
}

// Pass 1. Block (KV head, chunk) reads its chunk of K and V once and serves the kGroup query heads of that KV head.
// It writes, per (query head, chunk), the chunk maximum, the sum of exp(score - maximum) and the unnormalized
// weighted sum of V into scratch. Chunks past *pos write nothing; attention_combine never reads them.
__global__ void __launch_bounds__(kChunkThreads, 2)
    attention_chunk_kernel(const float* q, int64_t q_stride, const uint16_t* k_cache, const uint16_t* v_cache,
                           float* scratch, const int32_t* pos, int kv_heads, float scale, int chunks) {
    const int kv = blockIdx.x, chunk = blockIdx.y;
    const int first = chunk * kChunk, rows = *pos + 1;
    if (first >= rows)
        return;
    const int n = min(kChunk, rows - first);
    // The query, later the sums of row groups 2 and 3.
    __shared__ __align__(16) float shared[2 * kGroup * kHeadDim];
    __shared__ __align__(16) float weight[kGroup][kChunk];  // scores, then exp(score - chunk maximum)
    const int d = threadIdx.x, lane = d % kWarp, warp = d / kWarp;
    const int64_t row_len = int64_t{kv_heads} * kHeadDim;
    const int64_t kv_offset = int64_t{kv} * kHeadDim;

    // Four threads (a quad) share kQuadRows rows, so each query read serves both rows. Thread j of a quad reads the
    // 16-byte pieces j, j + 4, ... of its key rows; every read is issued before the query is staged so their latency
    // overlaps.
    const int quad = d / 4, piece = d % 4;
    uint4 raw[kQuadRows][kHeadDim / 32];
#pragma unroll
    for (int r = 0; r < kQuadRows; ++r) {
        const int row = quad + r * (kChunkThreads / 4);
        const uint4* key_row =
            reinterpret_cast<const uint4*>(k_cache + (first + (row < n ? row : 0)) * row_len + kv_offset);
#pragma unroll
        for (int i = 0; i < kHeadDim / 32; ++i)
            raw[r][i] = key_row[i * 4 + piece];
    }

#pragma unroll

    for (int i = d; i < kGroup * kHeadDim; i += kChunkThreads)
        shared[i] = q[(kv * kGroup + i / kHeadDim) * q_stride + i % kHeadDim];
    __syncthreads();

    // The quad combines in a fixed order, ((j0 + j1) + (j2 + j3)).
    float dot[kQuadRows][kGroup] = {};
#pragma unroll
    for (int i = 0; i < kHeadDim / 32; ++i) {
        float key[kQuadRows][8];
#pragma unroll
        for (int r = 0; r < kQuadRows; ++r) {
            const uint32_t words[4] = {raw[r][i].x, raw[r][i].y, raw[r][i].z, raw[r][i].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 pair = __half22float2(
                    __halves2half2(__ushort_as_half(words[e] & 0xffff), __ushort_as_half(words[e] >> 16)));
                key[r][2 * e] = pair.x, key[r][2 * e + 1] = pair.y;
            }
        }
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            const float4* qv = reinterpret_cast<const float4*>(&shared[g * kHeadDim + (i * 4 + piece) * 8]);
            const float4 q0 = qv[0], q1 = qv[1];
#pragma unroll
            for (int r = 0; r < kQuadRows; ++r) {
                dot[r][g] += q0.x * key[r][0] + q0.y * key[r][1] + q0.z * key[r][2] + q0.w * key[r][3];
                dot[r][g] += q1.x * key[r][4] + q1.y * key[r][5] + q1.z * key[r][6] + q1.w * key[r][7];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kQuadRows; ++r) {
        const int row = quad + r * (kChunkThreads / 4);
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            dot[r][g] += __shfl_xor_sync(0xffffffffu, dot[r][g], 1);
            dot[r][g] += __shfl_xor_sync(0xffffffffu, dot[r][g], 2);
            if (row < n && piece == 0)
                weight[g][row] = dot[r][g] * scale;
        }
    }
    __syncthreads();

    // Softmax statistics: warp g takes head g; lane l covers positions l, l + 32, ...
    for (int g = warp; g < kGroup; g += kChunkThreads / kWarp) {
        float peak = -INFINITY;
        for (int p = lane; p < n; p += kWarp)
            peak = fmaxf(peak, weight[g][p]);
        peak = warp_max(peak);
        float sum = 0.0f;
        for (int p = lane; p < n; p += kWarp) {
            weight[g][p] = expf(weight[g][p] - peak);
            sum += weight[g][p];
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            const int slot = (kv * kGroup + g) * chunks + chunk;
            float* const stats = scratch + heads_scratch_offset(kv_heads * kGroup, chunks);
            stats[2 * slot] = peak;
            stats[2 * slot + 1] = sum;
        }
    }
    __syncthreads();

    // Weighted sum of V. Row group k = d / 64 takes rows 32k..32k+31 and column c = d % 64 takes dimensions 4c..4c+3
    // (an 8-byte read); eight rows at a time are read before any is used, and share two 16-byte weight reads.
    constexpr int kRowsPerGroup = kChunk / 4;
    const int group = d / 64, col = d % 64;
    const int row_begin = group * kRowsPerGroup, row_end = min(n, row_begin + kRowsPerGroup);
    float acc[kGroup][4] = {};
    const uint16_t* value = v_cache + first * row_len + kv_offset + col * 4;
    auto load_value = [&](int r) {
        const uint2 raw = *reinterpret_cast<const uint2*>(value + r * row_len);
        const float2 lo =
            __half22float2(__halves2half2(__ushort_as_half(raw.x & 0xffff), __ushort_as_half(raw.x >> 16)));
        const float2 hi =
            __half22float2(__halves2half2(__ushort_as_half(raw.y & 0xffff), __ushort_as_half(raw.y >> 16)));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    };
    int r = row_begin;
    for (; r + 8 <= row_end; r += 8) {
        float4 v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            v[i] = load_value(r + i);
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            const float4 w0 = *reinterpret_cast<const float4*>(&weight[g][r]);
            const float4 w1 = *reinterpret_cast<const float4*>(&weight[g][r + 4]);
            const float ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[g][0] += ws[i] * v[i].x, acc[g][1] += ws[i] * v[i].y;
                acc[g][2] += ws[i] * v[i].z, acc[g][3] += ws[i] * v[i].w;
            }
        }
    }
    for (; r < row_end; ++r) {
        const float4 v = load_value(r);
        for (int g = 0; g < kGroup; ++g) {
            acc[g][0] += weight[g][r] * v.x, acc[g][1] += weight[g][r] * v.y;
            acc[g][2] += weight[g][r] * v.z, acc[g][3] += weight[g][r] * v.w;
        }
    }

    // Add the four row groups in a fixed order, ((0 + 2) + (1 + 3)), through shared memory.
    auto slot = [&](int buffer, int g) {
        return reinterpret_cast<float4*>(&shared[(buffer * kGroup + g) * kHeadDim]) + col;
    };
    auto store = [&](int buffer) {
        for (int g = 0; g < kGroup; ++g)
            *slot(buffer, g) = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    };
    auto add = [&](int buffer) {
        for (int g = 0; g < kGroup; ++g) {
            const float4 o = *slot(buffer, g);
            acc[g][0] += o.x, acc[g][1] += o.y, acc[g][2] += o.z, acc[g][3] += o.w;
        }
    };
    if (group >= 2)
        store(group - 2);
    __syncthreads();
    if (group < 2)
        add(group);
    __syncthreads();
    if (group == 1)
        store(0);
    __syncthreads();
    if (group == 0) {
        add(0);
        for (int g = 0; g < kGroup; ++g)
            *reinterpret_cast<float4*>(&scratch[(int64_t{kv * kGroup + g} * chunks + chunk) * kHeadDim + col * 4]) =
                make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    }
}

// Pass 2. Block h merges the chunks of query head h in ascending order and applies the gate.
__global__ void __launch_bounds__(kThreads)
    attention_combine_kernel(const float* gate, int64_t q_stride, const float* scratch, float* out, const int32_t* pos,
                             int heads, int chunks) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int used = *pos / kChunk + 1;
    const float* stats = scratch + heads_scratch_offset(heads, chunks);
    float peak = -INFINITY;
    for (int c = 0; c < used; ++c)
        peak = fmaxf(peak, stats[2 * (h * chunks + c)]);
    float sum = 0.0f, acc = 0.0f;
    for (int c = 0; c < used; ++c) {
        const float scale = expf(stats[2 * (h * chunks + c)] - peak);
        sum += stats[2 * (h * chunks + c) + 1] * scale;
        acc += scratch[(int64_t{h} * chunks + c) * kHeadDim + d] * scale;
    }
    out[h * kHeadDim + d] = acc / sum * sigmoid(gate[h * q_stride + d]);
}

}  // namespace

void rope_neox(float* x, int64_t head_stride, int heads, int n_rot, const int32_t* pos, float base,
               cudaStream_t stream) {
    rope_kernel<<<heads, n_rot / 2, 0, stream>>>(x, head_stride, n_rot, pos, base);
}

void store_kv(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int n, const int32_t* pos,
              cudaStream_t stream) {
    store_kv_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(k, v, k_cache, v_cache, n, pos);
}

void qk_norm_rope_store(float* q, int64_t q_stride, const float* q_weight, float* k, const float* k_weight,
                        const float* v, uint16_t* k_cache, uint16_t* v_cache, int heads, int kv_heads, int n_rot,
                        const int32_t* pos, float base, float eps, cudaStream_t stream) {
    qk_norm_rope_store_kernel<<<heads + kv_heads, kThreads, 0, stream>>>(q, q_stride, q_weight, k, k_weight, v, k_cache,
                                                                         v_cache, heads, n_rot, pos, base, eps);
}

int64_t attention_scratch_floats(int heads, int max_ctx) {
    return int64_t{heads} * chunk_count(max_ctx) * (kHeadDim + 2);
}

void attention_decode(const float* q, const float* gate, int64_t q_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, float* out, float* scratch, const int32_t* pos, int heads, int kv_heads,
                      float scale, int max_ctx, cudaStream_t stream) {
    if (heads != kv_heads * kGroup)
        throw Error("attention_decode needs 12 query heads per KV head.");
    const int chunks = chunk_count(max_ctx);
    attention_chunk_kernel<<<dim3(kv_heads, chunks), kChunkThreads, 0, stream>>>(q, q_stride, k_cache, v_cache, scratch,
                                                                                 pos, kv_heads, scale, chunks);
    attention_combine_kernel<<<heads, kThreads, 0, stream>>>(gate, q_stride, scratch, out, pos, heads, chunks);
}

}  // namespace torch::kernels
