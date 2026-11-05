// What a kernel costs in a replayed CUDA graph beyond its own work: the time per node of a chain of N tiny dependent
// kernels (one block, so the GPU is nearly idle), and of N kernels that each fill the GPU with a short, cheap pass;
// plus the host's cudaGraphLaunch call and a launch-and-wait round trip. Decode runs ~500-2,000 kernels a
// token, so this bounds how many launches a token can afford.
//
// usage: torch_bench_graph_gaps
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "core/cuda_check.h"

namespace {

__global__ void tiny(int* p) {
    if (threadIdx.x == 0)
        *p += 1;
}

// Every SM gets work: 16 blocks x 256 threads each touch one float of a 4 MiB buffer.
__global__ void wide(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] * 0.5f + 1.0f;
}

// Microseconds per node of a replayed graph of n copies of a launch, best of 20 replays.
template <class Launch>
double per_node_us(int n, Launch launch, cudaStream_t s) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    TORCH_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i)
        launch(s);
    TORCH_CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    TORCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t a, b;
    TORCH_CUDA_CHECK(cudaEventCreate(&a));
    TORCH_CUDA_CHECK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < 21; ++r) {
        TORCH_CUDA_CHECK(cudaEventRecord(a, s));
        TORCH_CUDA_CHECK(cudaGraphLaunch(exec, s));
        TORCH_CUDA_CHECK(cudaEventRecord(b, s));
        TORCH_CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (r > 0)
            best = std::min(best, ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return 1000.0 * best / n;
}

}  // namespace

int main() {
    try {
        cudaStream_t s;
        TORCH_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        int* counter = nullptr;
        float* buffer = nullptr;
        const int n_floats = 16 * 256;
        TORCH_CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
        TORCH_CUDA_CHECK(cudaMalloc(&buffer, n_floats * sizeof(float)));
        std::printf("Replayed CUDA graph, time per node (best of 20 replays)\n");
        for (const int n : {100, 500, 1000}) {
            const double t = per_node_us(n, [&](cudaStream_t st) { tiny<<<1, 32, 0, st>>>(counter); }, s);
            const double w = per_node_us(n, [&](cudaStream_t st) { wide<<<16, 256, 0, st>>>(buffer, n_floats); }, s);
            std::printf("  %4d nodes: one-block kernels %.2f us per node; 16-block kernels %.2f us per node\n", n, t,
                        w);
        }

        // The host side of a token: the cudaGraphLaunch call, and launch plus wait for a one-node graph.
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        TORCH_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        tiny<<<1, 32, 0, s>>>(counter);
        TORCH_CUDA_CHECK(cudaStreamEndCapture(s, &graph));
        TORCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        std::vector<double> call_us, trip_us;
        for (int r = 0; r < 201; ++r) {
            const auto start = std::chrono::steady_clock::now();
            TORCH_CUDA_CHECK(cudaGraphLaunch(exec, s));
            const auto launched = std::chrono::steady_clock::now();
            TORCH_CUDA_CHECK(cudaStreamSynchronize(s));
            const auto done = std::chrono::steady_clock::now();
            if (r == 0)
                continue;  // the first launch uploads the graph
            call_us.push_back(std::chrono::duration<double, std::micro>(launched - start).count());
            trip_us.push_back(std::chrono::duration<double, std::micro>(done - start).count());
        }
        std::sort(call_us.begin(), call_us.end());
        std::sort(trip_us.begin(), trip_us.end());
        std::printf("  cudaGraphLaunch call: median %.1f us, p99 %.1f us\n", call_us[call_us.size() / 2],
                    call_us[call_us.size() * 99 / 100]);
        std::printf("  launch and wait for a one-node graph: median %.1f us, p99 %.1f us\n",
                    trip_us[trip_us.size() / 2], trip_us[trip_us.size() * 99 / 100]);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaFree(buffer);
        cudaFree(counter);
        cudaStreamDestroy(s);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_bench_graph_gaps: %s\n", e.what());
        return 1;
    }
    return 0;
}
