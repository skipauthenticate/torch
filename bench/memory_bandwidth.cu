// How fast the GPU streams each kind of memory the engine can allocate: device memory (cudaMalloc), mapped pinned
// host memory (cudaHostAlloc with cudaHostAllocMapped) and ordinary memory registered as mapped (cudaHostRegister);
// the CPU can write the last two from a file read.
// Architecture decision 8 keeps resident experts in mapped memory if it reads within 5% of device memory.
//
// usage: torch_bench_memory [GiB per buffer, default 1]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "core/cuda_check.h"

namespace {

// Each thread reads 16-byte words with a grid stride and folds them into one word, so nothing is optimized away.
__global__ void stream_read(const uint4* data, size_t words, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * size_t{blockDim.x} + threadIdx.x; i < words; i += size_t{gridDim.x} * blockDim.x) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u)
        *sink = acc;
}

// Best of ten timed passes over the buffer, in GB/s (10^9 bytes per second).
double read_bandwidth(const void* device_view, size_t bytes, unsigned* sink) {
    cudaEvent_t start, stop;
    TORCH_CUDA_CHECK(cudaEventCreate(&start));
    TORCH_CUDA_CHECK(cudaEventCreate(&stop));
    const auto* data = static_cast<const uint4*>(device_view);
    const size_t words = bytes / sizeof(uint4);
    float best = 1e30f;
    for (int pass = 0; pass < 11; ++pass) {
        TORCH_CUDA_CHECK(cudaEventRecord(start));
        stream_read<<<16 * 8, 512>>>(data, words, sink);  // 8 blocks of 512 threads on each of the 16 SMs
        TORCH_CUDA_CHECK(cudaEventRecord(stop));
        TORCH_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (pass > 0)  // the first pass warms up
            best = std::min(best, ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(bytes) / (best * 1e-3) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    const double gib = argc > 1 ? std::atof(argv[1]) : 1.0;
    const size_t bytes = static_cast<size_t>(gib * (1u << 30)) / 4096 * 4096;
    if (bytes == 0) {
        std::fprintf(stderr, "usage: torch_bench_memory [GiB per buffer, default 1]\n");
        return 2;
    }
    try {
        unsigned* sink = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));

        void* device = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&device, bytes));
        TORCH_CUDA_CHECK(cudaMemset(device, 1, bytes));
        const double device_gbs = read_bandwidth(device, bytes, sink);
        TORCH_CUDA_CHECK(cudaFree(device));

        void* host = nullptr;
        void* host_view = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
        std::fill_n(static_cast<unsigned char*>(host), bytes, 1);  // written by the CPU, as a file read would
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&host_view, host, 0));
        const double mapped_gbs = read_bandwidth(host_view, bytes, sink);
        TORCH_CUDA_CHECK(cudaFreeHost(host));
        // Ordinary memory registered with the driver: O_DIRECT reads into it take the kernel's usual path.
        void* registered = std::aligned_alloc(4096, bytes);
        void* registered_view = nullptr;
        if (registered == nullptr)
            throw torch::Error("Out of memory for the registered buffer.");
        std::fill_n(static_cast<unsigned char*>(registered), bytes, 1);
        TORCH_CUDA_CHECK(cudaHostRegister(registered, bytes, cudaHostRegisterMapped));
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&registered_view, registered, 0));
        const double registered_gbs = read_bandwidth(registered_view, bytes, sink);
        TORCH_CUDA_CHECK(cudaHostUnregister(registered));
        std::free(registered);

        std::printf("GPU read bandwidth, %.2f GiB buffers, best of 10 passes\n", gib);
        std::printf("  device memory (cudaMalloc)            %7.1f GB/s\n", device_gbs);
        std::printf("  mapped pinned memory (cudaHostAlloc)  %7.1f GB/s  (%.1f%% of device memory)\n", mapped_gbs,
                    100.0 * mapped_gbs / device_gbs);
        std::printf("  registered memory (cudaHostRegister)  %7.1f GB/s  (%.1f%% of device memory)\n", registered_gbs,
                    100.0 * registered_gbs / device_gbs);
        cudaFree(sink);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_bench_memory: %s\n", e.what());
        return 1;
    }
    return 0;
}
