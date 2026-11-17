// Recurrent DeltaNet over a prefill chunk: gdn_window with N columns (the window kernel, state in registers) for the
// model's 36 DeltaNet layers, each with its own state, timed with CUDA events over one CUDA graph. Inputs are random
// in conv_silu_l2_window's output layout; seconds per chunk of N tokens and per 4,096 tokens.
//
// usage: torch_bench_deltanet MODEL-00001-of-0000N.gguf [--columns N, default 4096]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "gguf/gguf.h"
#include "kernels/deltanet/deltanet.h"
#include "model/config.h"

namespace {

using namespace torch;

float* random_device(size_t n, float lo, float hi, uint64_t seed) {
    std::vector<float> h(n);
    for (float& x : h) {
        seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
        x = lo + (hi - lo) * static_cast<float>(seed >> 40) / 16777216.0f;
    }
    float* d = nullptr;
    TORCH_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    TORCH_CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    int n = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--columns" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else
            path = a;
    }
    if (path.empty() || n < 1) {
        std::fprintf(stderr, "usage: torch_bench_deltanet MODEL-00001-of-0000N.gguf [--columns N]\n");
        return 2;
    }
    const model::Config cfg = model::Config::read(gguf::File::open_model(path, /*header_only=*/true));
    int layers = 0;
    for (int il = 0; il < cfg.n_layer; ++il)
        layers += cfg.recurrent(il);
    const int KH = cfg.ssm_k_heads, HV = cfg.ssm_v_heads, DS = cfg.ssm_head_dim, CC = cfg.conv_channels();
    const size_t N = static_cast<size_t>(n), state = static_cast<size_t>(HV) * DS * DS;
    std::vector<float*> states(static_cast<size_t>(layers));
    for (int l = 0; l < layers; ++l)
        states[static_cast<size_t>(l)] = random_device(2 * state, -0.1f, 0.1f, 7 + static_cast<uint64_t>(l));
    float* x = random_device(N * CC, -0.1f, 0.1f, 1);  // q | k | v per column, as conv_silu_l2_window writes them
    float* alpha = random_device(N * HV, -1.0f, 1.0f, 2);
    float* beta = random_device(N * HV, -1.0f, 1.0f, 3);
    float* dt = random_device(HV, -1.0f, 1.0f, 4);
    float* a = random_device(HV, -1.0f, -0.1f, 5);
    float *gates = nullptr, *out = nullptr;
    TORCH_CUDA_CHECK(cudaMalloc(&gates, N * HV * 2 * sizeof(float)));
    TORCH_CUDA_CHECK(cudaMalloc(&out, N * HV * DS * sizeof(float)));
    const int QK = KH * DS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(DS));
    cudaStream_t stream;
    TORCH_CUDA_CHECK(cudaStreamCreate(&stream));
    const auto launch = [&] {
        for (int l = 0; l < layers; ++l) {
            float* s = states[static_cast<size_t>(l)];
            kernels::gdn_window(s, s + state, x, x + QK, x + 2 * QK, CC, alpha, beta, dt, a, gates, out, KH, HV, scale,
                                n, stream);
        }
    };
    launch();  // warm-up, and lazy module loading outside the capture
    TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    TORCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t start, stop;
    TORCH_CUDA_CHECK(cudaEventCreate(&start));
    TORCH_CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int kReplays = 3;
    TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    TORCH_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int r = 0; r < kReplays; ++r)
        TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    TORCH_CUDA_CHECK(cudaEventRecord(stop, stream));
    TORCH_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double per_chunk = ms / kReplays / 1000.0;
    std::printf(
        "gdn_window, %d columns, %d layers x %d heads x %d: %.3f s per chunk (%.0f us per layer, %.2f us per "
        "column and layer), %.3f s per 4,096 tokens\n",
        n, layers, HV, DS, per_chunk, per_chunk * 1e6 / layers, per_chunk * 1e6 / layers / n, per_chunk * 4096.0 / n);
    return 0;
}
