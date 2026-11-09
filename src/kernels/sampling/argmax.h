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

// out = the largest x[i] for i < n and its lowest index. scratch holds kArgmaxBlocks results. With `columns`, a
// window's logits [T][n] in one launch: out[t] for column t, scratch [T][kArgmaxBlocks].
void argmax(const float* x, int n, ArgmaxResult* scratch, ArgmaxResult* out, cudaStream_t stream, int columns = 1);

// prob = 1 / (sum over i < n of expf(x[i] - best->value)): the softmax probability of the largest of n logits, as
// argmax found it. One block; a fixed-order sum, so the result never depends on timing.
void argmax_prob(const float* x, int n, const ArgmaxResult* best, float* prob, cudaStream_t stream);

}  // namespace torch::kernels
