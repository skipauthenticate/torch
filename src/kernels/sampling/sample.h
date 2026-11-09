// Sampling the next token of every column of a verify window on the GPU, and the exact accept step for point-mass
// drafts. The full 248,320-logit distribution of each column is built on the device; nothing but the chosen tokens
// leaves it.
//
// The distribution of a column, in this order:
//  1. Penalties on the raw logit x of every token with a history count n > 0: x = x / repetition_penalty if x > 0,
//     else x * repetition_penalty; then x = x - frequency_penalty * n - presence_penalty. The history count of column i
//     is counts[v] (prompt and generated tokens before the window) plus the number of j < i with drafts[j] == v: the
//     inputs of the window's columns 1..i, which column i's distribution is conditional on.
//  2. Temperature: x / temperature. temperature == 0 means greedy: the token is the lowest index of the largest x, no
//     random numbers are drawn, and the steps below do not apply.
//  3. Softmax over the whole vocabulary: p = exp(x - max) / sum, F32, fixed reduction order.
//  4. top_k > 0 keeps the top_k tokens in the order (p descending, index ascending); top_p < 1 keeps the shortest
//  prefix
//     of that order whose probability reaches top_p (the token that crosses the threshold is kept); both are applied to
//     the probabilities of step 3 and a token must satisfy both. Ties at the boundary go to the lower index.
//  5. min_p > 0 drops the kept tokens with p < min_p * (largest p).
//  6. The kept probabilities are renormalised: q = p / (sum of the kept p).
// Selections are exact (a radix select on the probability bits, no sort) and every sum has a fixed order or is an
// integer, so a column's distribution does not depend on timing.
//
// Random numbers: Philox-4x32-10 keyed by the 64-bit seed with the counter (position, purpose, 0, 0), where position
// is the position of the token being sampled. A token is drawn by inverse transform on the kept tokens in increasing
// index order with u = the Philox word 0 scaled to [0, 1). The same request therefore draws the same tokens however
// its windows are cut.
//
// Speculative decoding with point-mass drafts (the draft is a token, a point mass proposal) is exact when the
// accept test reuses that draw. A draft x for column i is accepted with probability q_i(x) and otherwise replaced by a
// token drawn from q_i with x removed and renormalised. Drawing y ~ q_i with the column's own random number and
// accepting exactly when y == x does both: P(y = x) = q_i(x), and given y != x, y has the distribution q_i(v) /
// (1 - q_i(x)). So accept_drafts keeps the sampled token of the first column that disagrees with its draft, and the
// tokens a window produces are the very tokens the same seed produces without speculation, for any drafts.
//
// Parity test: tests/parity/sample_test.cpp (unit cases, and chi-square tests of the token distributions and of the
// acceptance rate against a double-precision reference over thousands of seeds).
// Bytes moved: the logits are read once and the probabilities (columns * vocab floats of scratch) written and read
// about a dozen times, mostly from L2.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

struct SamplingParams {
    float temperature = 1.0f;
    int top_k = 0;       // 0: off
    float top_p = 1.0f;  // 1: off
    float min_p = 0.0f;  // 0: off
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    uint64_t seed = 0;
};

// vocab must be a multiple of 4 (the model's 248,320 is) and at most 262,144. Floats of scratch sample_columns needs.
int64_t sampling_scratch_floats(int vocab, int columns);

// sampled[i] = the token drawn from column i's distribution (logits is [columns][vocab]). counts is [vocab] int32 of
// the history before the window, or nullptr when no penalty is set; drafts[0 .. columns - 2] are the inputs of columns
// 1 .. columns - 1 (nullptr for one column). *position is the position of the token drawn from column 0 (column i
// draws at *position + i), in device memory so a captured graph can replay it. One block per column.
void sample_columns(const float* logits, int vocab, int columns, const int32_t* counts, const int32_t* drafts,
                    const SamplingParams& params, const int32_t* position, float* scratch, int32_t* sampled,
                    cudaStream_t stream);

// The accept step for a window of `columns` columns whose inputs are [t0, drafts[0], ... drafts[columns - 2]]: the
// longest prefix of drafts equal to the sampled tokens is accepted, so with k accepted drafts *accepted = k + 1 columns
// are valid (the number to commit in conv_commit, ple_commit and the replay of gdn_window) and tokens[0 .. k] = sampled
// [0 .. k]: the k accepted drafts followed by the first token the target disagrees with, or the bonus token of the last
// column when every draft passed. tokens[k + 1 ..] is set to -1. Output in device memory.
void accept_drafts(const int32_t* sampled, const int32_t* drafts, int columns, int32_t* accepted, int32_t* tokens,
                   cudaStream_t stream);

}  // namespace torch::kernels
