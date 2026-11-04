#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/gemv/dot_q8.cuh"
#include "kernels/gemv/gemv.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / kWarp;  // one warp per row

// One warp per 32 values.
__global__ void quantize_q8_kernel(const float* x, int64_t blocks, BlockQ8* out) {
    const int64_t block = blockIdx.x * int64_t{kRowsPerBlock} + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (block >= blocks)
        return;
    const float v = x[block * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : static_cast<int>(roundf(v / d));
    out[block].q[lane] = static_cast<int8_t>(q);
    const int sum = __reduce_add_sync(0xffffffffu, q);
    if (lane == 0) {
        out[block].d = __half_as_ushort(__float2half_rn(d));
        out[block].sum = static_cast<int16_t>(sum);
    }
}

// Row blockIdx.x * 8 + warp of slot blockIdx.y. Dense matrices pass `w`; experts pass the pointer tables.
template <Type T, bool kSwiglu>
__global__ void __launch_bounds__(kThreads)
    gemv_q8_kernel(const uint8_t* w, const uint8_t* const* table, const uint8_t* const* up_table, int rows, int units,
                   int64_t row_bytes, const BlockQ8* x, int64_t x_stride, float* y) {
    const int slot = blockIdx.y;
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (row >= rows)
        return;
    const uint8_t* gate = (table != nullptr ? table[slot] : w) + row * row_bytes;
    const uint8_t* up = kSwiglu ? up_table[slot] + row * row_bytes : nullptr;
    const BlockQ8* xs = x + slot * x_stride;
    float acc = 0.0f, acc_up = 0.0f;
    for (int u = lane; u < units; u += kWarp) {
        acc += dot_unit<T>(gate, u, xs[u]);
        if constexpr (kSwiglu)
            acc_up += dot_unit<T>(up, u, xs[u]);
    }
    acc = warp_sum(acc);
    if constexpr (kSwiglu)
        acc_up = warp_sum(acc_up);
    if (lane == 0)
        y[int64_t{slot} * rows + row] = kSwiglu ? silu(acc) * acc_up : acc;
}

__device__ __forceinline__ float bf16_low(uint32_t v) {
    return __uint_as_float(v << 16);
}
__device__ __forceinline__ float bf16_high(uint32_t v) {
    return __uint_as_float(v & 0xffff0000u);
}

// Lane l sums 8-element chunks l, l + 32, ... element by element, then the warp adds the lane sums.
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_kernel(const uint8_t* w, int rows, int chunks, int64_t row_bytes, const float* x, float* y) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (row >= rows)
        return;
    const auto* r = reinterpret_cast<const uint4*>(w + row * row_bytes);
    const auto* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += kWarp) {
        const uint4 v = r[c];
        const float4 a = x4[2 * c], b = x4[2 * c + 1];
        acc += bf16_low(v.x) * a.x;
        acc += bf16_high(v.x) * a.y;
        acc += bf16_low(v.y) * a.z;
        acc += bf16_high(v.y) * a.w;
        acc += bf16_low(v.z) * b.x;
        acc += bf16_high(v.z) * b.y;
        acc += bf16_low(v.w) * b.z;
        acc += bf16_high(v.w) * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0)
        y[row] = acc;
}

// Calls f.template operator()<T>() for the formats gemv_q8 takes; false for any other.
template <class F>
bool with_q8_type(Type type, F&& f) {
    switch (type) {
        case Type::Q8_0:
            f.template operator()<Type::Q8_0>();
            return true;
        case Type::Q2_0:
            f.template operator()<Type::Q2_0>();
            return true;
        case Type::Q4_K:
            f.template operator()<Type::Q4_K>();
            return true;
        case Type::Q5_K:
            f.template operator()<Type::Q5_K>();
            return true;
        case Type::Q6_K:
            f.template operator()<Type::Q6_K>();
            return true;
        case Type::IQ2_S:
            f.template operator()<Type::IQ2_S>();
            return true;
        case Type::IQ3_XXS:
            f.template operator()<Type::IQ3_XXS>();
            return true;
        case Type::IQ3_S:
            f.template operator()<Type::IQ3_S>();
            return true;
        case Type::IQ4_NL:
            f.template operator()<Type::IQ4_NL>();
            return true;
        case Type::IQ4_XS:
            f.template operator()<Type::IQ4_XS>();
            return true;
        default:
            return false;
    }
}

int64_t q8_row_bytes(Type type, int cols) {
    return static_cast<int64_t>(row_bytes(*type_info(static_cast<uint32_t>(type)), static_cast<uint64_t>(cols)));
}

}  // namespace

void quantize_q8(const float* x, int64_t n, BlockQ8* out, cudaStream_t stream) {
    const int64_t blocks = n / 32;
    quantize_q8_kernel<<<blocks_for(blocks, kRowsPerBlock), kThreads, 0, stream>>>(x, blocks, out);
}

void upload_iq_tables() {
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq2s_grid, kIq2sGrid, sizeof kIq2sGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3xxs_grid, kIq3xxsGrid, sizeof kIq3xxsGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3s_grid, kIq3sGrid, sizeof kIq3sGrid));
}

bool gemv_q8_supports(Type type) {
    return with_q8_type(type, []<Type>() {});
}

void gemv_q8(Type type, const uint8_t* w, int rows, int cols, const BlockQ8* x, float* y, cudaStream_t stream) {
    with_q8_type(type, [&]<Type T>() {
        gemv_q8_kernel<T, false><<<blocks_for(rows, kRowsPerBlock), kThreads, 0, stream>>>(
            w, nullptr, nullptr, rows, cols / 32, q8_row_bytes(type, cols), x, 0, y);
    });
}

void gemv_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, cudaStream_t stream) {
    gemv_bf16_kernel<<<blocks_for(rows, kRowsPerBlock), kThreads, 0, stream>>>(w, rows, cols / 8, int64_t{cols} * 2, x,
                                                                               y);
}

void gemv_q8_experts(Type type, const uint8_t* const* table, const uint8_t* const* up_table, int slots, int rows,
                     int cols, const BlockQ8* x, int64_t x_stride, float* y, cudaStream_t stream) {
    const dim3 grid(blocks_for(rows, kRowsPerBlock), slots);
    const int64_t rb = q8_row_bytes(type, cols);
    with_q8_type(type, [&]<Type T>() {
        if (up_table != nullptr)
            gemv_q8_kernel<T, true>
                <<<grid, kThreads, 0, stream>>>(nullptr, table, up_table, rows, cols / 32, rb, x, x_stride, y);
        else
            gemv_q8_kernel<T, false>
                <<<grid, kThreads, 0, stream>>>(nullptr, table, nullptr, rows, cols / 32, rb, x, x_stride, y);
    });
}

}  // namespace torch::kernels
