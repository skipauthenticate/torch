#include "kernels/common.cuh"
#include "kernels/ple/ple.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kRing = 9;

// One block per stream.
__global__ void __launch_bounds__(kThreads)
    ple_gate_kernel(const float* key_n, const float* query, float* gate, int n) {
    const int64_t base = int64_t{blockIdx.x} * n;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads)
        dot += key_n[base + i] * query[base + i];
    dot = block_sum(dot);
    if (threadIdx.x == 0) {
        const float v = dot * (1.0f / sqrtf(static_cast<float>(n)));
        const float mag = sqrtf(fmaxf(fabsf(v), 1e-6f));
        gate[blockIdx.x] = sigmoid(v > 0.0f ? mag : (v < 0.0f ? -mag : 0.0f));
    }
}

__global__ void ple_gated_value_kernel(const float* value, const float* gate, float* gated, int n, int hc) {
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    for (int s = 0; s < hc; ++s)
        gated[int64_t{s} * n + e] = value[e] * gate[s];
}

__global__ void ple_conv_kernel(float* res, const float* gated, const float* x, float* hist, const float* w,
                                int channels, const int32_t* pos_ptr) {
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels)
        return;
    const int pos = *pos_ptr;
    const auto back = [&](int b) { return pos - b < 0 ? 0.0f : hist[int64_t{(pos - b) % kRing} * channels + c]; };
    const float* wc = w + 4 * c;
    const float xc = x[c];
    const float conv = wc[0] * back(9) + wc[1] * back(6) + wc[2] * back(3) + wc[3] * xc;
    hist[int64_t{pos % kRing} * channels + c] = xc;
    res[c] += gated[c] + silu(conv);
}

// ple_conv for a window: grid (blocks, columns). Column t sits at position pos + t; a tap at position pos + t - b reads
// the column buffer x when t - b >= 0 and the committed ring otherwise, and nothing is written to the ring.
__global__ void ple_conv_window_kernel(float* res, const float* gated, const float* x, const float* hist,
                                       const float* w, int channels, const int32_t* pos_ptr) {
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels)
        return;
    const int t = blockIdx.y, pos = *pos_ptr;
    const int64_t at = t * int64_t{channels} + c;
    const auto back = [&](int b) {
        if (pos + t - b < 0)
            return 0.0f;
        return t - b >= 0 ? x[(t - b) * int64_t{channels} + c] : hist[int64_t{(pos + t - b) % kRing} * channels + c];
    };
    const float* wc = w + 4 * c;
    const float xc = x[at];
    const float conv = wc[0] * back(9) + wc[1] * back(6) + wc[2] * back(3) + wc[3] * xc;
    res[at] += gated[at] + silu(conv);
}

// Rows of positions pos .. pos + accepted - 1 from the first `accepted` columns of x.
__global__ void ple_commit_kernel(float* hist, const float* x, int channels, const int32_t* pos_ptr, int accepted,
                                  const int32_t* accepted_ptr) {
    if (accepted_ptr != nullptr)
        accepted = *accepted_ptr;
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels)
        return;
    const int pos = *pos_ptr;
    for (int t = 0; t < accepted; ++t)
        hist[int64_t{(pos + t) % kRing} * channels + c] = x[t * int64_t{channels} + c];
}

}  // namespace

void ple_gate(const float* key_n, const float* query, float* gate, int n, int hc, cudaStream_t stream) {
    ple_gate_kernel<<<hc, kThreads, 0, stream>>>(key_n, query, gate, n);
}

void ple_gated_value(const float* value, const float* gate, float* gated, int n, int hc, cudaStream_t stream) {
    ple_gated_value_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(value, gate, gated, n, hc);
}

void ple_conv(float* res, const float* gated, const float* x, float* hist, const float* w, int channels,
              const int32_t* pos, cudaStream_t stream) {
    ple_conv_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(res, gated, x, hist, w, channels, pos);
}

void ple_conv_window(float* res, const float* gated, const float* x, const float* hist, const float* w, int channels,
                     const int32_t* pos, int columns, cudaStream_t stream) {
    ple_conv_window_kernel<<<dim3(blocks_for(channels, kThreads), columns), kThreads, 0, stream>>>(res, gated, x, hist,
                                                                                                   w, channels, pos);
}

void ple_commit(float* hist, const float* x, int channels, const int32_t* pos, int accepted, cudaStream_t stream) {
    ple_commit_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(hist, x, channels, pos, accepted,
                                                                               nullptr);
}

void ple_commit(float* hist, const float* x, int channels, const int32_t* pos, const int32_t* accepted,
                cudaStream_t stream) {
    ple_commit_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(hist, x, channels, pos, 0, accepted);
}

}  // namespace torch::kernels
