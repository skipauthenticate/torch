// The most likely next token, on the GPU, so a token does not need its 248,320 logits copied to the host.
//
// Two launches: 64 blocks each find the largest value of a stripe of the logits, then one block reduces their
// candidates. Ties go to the lowest index, as everywhere in the engine (docs/architecture.md, "Numerics"). The result
// is the index and its logit, 8 bytes the host can read.
//
// Parity test: tests/parity/argmax_test.cpp. Bytes moved: the logits once (~1 MB).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

struct ArgmaxResult {
    int32_t index;
    float value;
};

inline constexpr int kArgmaxBlocks = 64;

// out = the largest x[i] for i < n and its lowest index. scratch holds kArgmaxBlocks results.
void argmax(const float* x, int n, ArgmaxResult* scratch, ArgmaxResult* out, cudaStream_t stream);

}  // namespace torch::kernels
