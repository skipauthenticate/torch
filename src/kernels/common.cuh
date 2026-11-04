// Device helpers every kernel folder shares: warp and block reductions, the activation functions, unaligned loads.
//
// Reductions use a fixed butterfly order, so a result never depends on timing. The activation functions are written
// in their plain forms, with IEEE expf, logf and division (no fast math).
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace torch::kernels {

inline constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

// Sum over a block whose size is a multiple of 32 (at most 1,024); every thread gets the result.
__device__ inline float block_sum(float v) {
    __shared__ float partial[32];
    __syncthreads();  // a previous call may still be reading partial
    v = warp_sum(v);
    if (threadIdx.x % kWarp == 0)
        partial[threadIdx.x / kWarp] = v;
    __syncthreads();
    v = threadIdx.x % kWarp < blockDim.x / kWarp ? partial[threadIdx.x % kWarp] : 0.0f;  // every warp gets the total
    return warp_sum(v);
}

__device__ inline float block_max(float v) {
    __shared__ float partial[32];
    __syncthreads();
    v = warp_max(v);
    if (threadIdx.x % kWarp == 0)
        partial[threadIdx.x / kWarp] = v;
    __syncthreads();
    v = threadIdx.x % kWarp < blockDim.x / kWarp ? partial[threadIdx.x % kWarp] : -INFINITY;
    return warp_max(v);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float f16_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Little-endian loads from 2-byte-aligned addresses: most block layouts are only 2-byte aligned.
__device__ __forceinline__ uint32_t load_u16(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

__device__ __forceinline__ uint32_t load_u32_a2(const uint8_t* p) {
    return load_u16(p) | load_u16(p + 2) << 16;
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {  // 4-byte aligned
    return *reinterpret_cast<const uint32_t*>(p);
}

inline int blocks_for(int64_t n, int threads) {
    return static_cast<int>((n + threads - 1) / threads);
}

}  // namespace torch::kernels
