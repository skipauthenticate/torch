#include "kernels/common.cuh"
#include "kernels/sampling/argmax.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;

// True when (v, i) beats (best_v, best_i): a larger value, or the same value at a lower index. NaN never wins.
__device__ __forceinline__ bool better(float v, int i, float best_v, int best_i) {
    return v > best_v || (v == best_v && i < best_i);
}

__device__ void block_best(float& v, int& i) {
    __shared__ float values[kWarp];
    __shared__ int indices[kWarp];
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, i, offset);
        if (better(ov, oi, v, i))
            v = ov, i = oi;
    }
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (lane == 0)
        values[warp] = v, indices[warp] = i;
    __syncthreads();
    if (warp == 0) {
        v = lane < kThreads / kWarp ? values[lane] : -INFINITY;
        i = lane < kThreads / kWarp ? indices[lane] : INT32_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, offset);
            const int oi = __shfl_xor_sync(0xffffffffu, i, offset);
            if (better(ov, oi, v, i))
                v = ov, i = oi;
        }
    }
}

__global__ void __launch_bounds__(kThreads) stripes(const float* x, int n, ArgmaxResult* scratch) {
    float v = -INFINITY;
    int i = INT32_MAX;
    for (int j = blockIdx.x * kThreads + threadIdx.x; j < n; j += gridDim.x * kThreads) {
        if (better(x[j], j, v, i))
            v = x[j], i = j;
    }
    block_best(v, i);
    if (threadIdx.x == 0)
        scratch[blockIdx.x] = {i, v};
}

__global__ void __launch_bounds__(kThreads) final_pass(const ArgmaxResult* scratch, ArgmaxResult* out) {
    float v = -INFINITY;
    int i = INT32_MAX;
    if (threadIdx.x < kArgmaxBlocks)
        v = scratch[threadIdx.x].value, i = scratch[threadIdx.x].index;
    block_best(v, i);
    if (threadIdx.x == 0)
        *out = {i, v};
}

}  // namespace

void argmax(const float* x, int n, ArgmaxResult* scratch, ArgmaxResult* out, cudaStream_t stream) {
    stripes<<<kArgmaxBlocks, kThreads, 0, stream>>>(x, n, scratch);
    final_pass<<<1, kThreads, 0, stream>>>(scratch, out);
}

}  // namespace torch::kernels
