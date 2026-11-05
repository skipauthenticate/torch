// The one definition of how 32 activations become a BlockQ8 (gemv.h), shared by quantize_q8 and every kernel that
// writes Q8 blocks as it produces its output, so they all give bitwise the same blocks.
#pragma once

#include <cuda_fp16.h>

#include "kernels/common.cuh"
#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// Called by all 32 lanes of a warp; lane l holds value l of the block.
__device__ __forceinline__ void quantize_q8_block(float v, BlockQ8* out) {
    const int lane = threadIdx.x % kWarp;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : static_cast<int>(roundf(v / d));
    out->q[lane] = static_cast<int8_t>(q);
    const int sum = __reduce_add_sync(0xffffffffu, q);
    if (lane == 0) {
        out->d = __half_as_ushort(__float2half_rn(d));
        out->sum = static_cast<int16_t>(sum);
    }
}

}  // namespace torch::kernels
