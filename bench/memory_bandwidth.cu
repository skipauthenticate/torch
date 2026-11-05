// How fast the GPU streams each kind of memory the engine can allocate: device memory (cudaMalloc), mapped pinned
// host memory (cudaHostAlloc with cudaHostAllocMapped) and ordinary memory registered as mapped (cudaHostRegister).
// The CPU can write the last two from a file read. The best read figure of the memory type the weights live in is the
// "streaming bandwidth" every kernel's gate is measured against (docs/roadmap.md, M3).
//
// Each launch shape is swept: 2 to 16 blocks per SM of 256 or 512 threads, each thread keeping 1 or 4 independent
// 16-byte loads in flight. Reads fold the words with XOR so nothing is optimized away; read+write copies one buffer
// into another and counts both directions. The GPU clock is sampled from sysfs while the timed passes run (the memory
// controller's clock is not readable without root, so `nvpmodel -q` is printed instead).
//
// usage: torch_bench_memory [GiB per buffer, default 1]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#include "core/cuda_check.h"

namespace {

template <int kLoads>
__global__ void stream_read(const uint4* data, size_t words, unsigned* sink) {
    unsigned acc = 0;
    const size_t stride = size_t{gridDim.x} * blockDim.x;
    size_t i = blockIdx.x * size_t{blockDim.x} + threadIdx.x;
    for (; i + (kLoads - 1) * stride < words; i += kLoads * stride) {
        uint4 v[kLoads];
#pragma unroll
        for (int k = 0; k < kLoads; ++k)
            v[k] = data[i + k * stride];
#pragma unroll
        for (int k = 0; k < kLoads; ++k)
            acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
    for (; i < words; i += stride) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u)
        *sink = acc;
}

template <int kLoads>
__global__ void stream_copy(const uint4* src, uint4* dst, size_t words) {
    const size_t stride = size_t{gridDim.x} * blockDim.x;
    size_t i = blockIdx.x * size_t{blockDim.x} + threadIdx.x;
    for (; i + (kLoads - 1) * stride < words; i += kLoads * stride) {
        uint4 v[kLoads];
#pragma unroll
        for (int k = 0; k < kLoads; ++k)
            v[k] = src[i + k * stride];
#pragma unroll
        for (int k = 0; k < kLoads; ++k)
            dst[i + k * stride] = v[k];
    }
    for (; i < words; i += stride)
        dst[i] = src[i];
}

struct Shape {
    int blocks_per_sm, threads, loads;
};

// Highest GPU clock seen in sysfs while `running` is set, in MHz.
class ClockSampler {
public:
    ClockSampler() {
        thread_ = std::thread([this] {
            while (!stop_) {
                std::ifstream f("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq");
                long hz = 0;
                if (f >> hz)
                    peak_ = std::max<long>(peak_, hz / 1000000);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~ClockSampler() {
        stop_ = true;
        thread_.join();
    }
    long peak_mhz() const { return peak_; }

private:
    std::atomic<bool> stop_{false};
    std::atomic<long> peak_{0};
    std::thread thread_;
};

// Best of ten timed passes after one warm-up pass, in GB/s (10^9 bytes per second, both directions for a copy).
double time_passes(const Shape& s, const void* a, void* b, size_t bytes, unsigned* sink, int sms) {
    cudaEvent_t start, stop;
    TORCH_CUDA_CHECK(cudaEventCreate(&start));
    TORCH_CUDA_CHECK(cudaEventCreate(&stop));
    const auto* src = static_cast<const uint4*>(a);
    auto* dst = static_cast<uint4*>(b);
    const size_t words = bytes / sizeof(uint4);
    const int grid = s.blocks_per_sm * sms;
    float best = 1e30f;
    for (int pass = 0; pass < 11; ++pass) {
        TORCH_CUDA_CHECK(cudaEventRecord(start));
        if (dst == nullptr) {
            if (s.loads == 4)
                stream_read<4><<<grid, s.threads>>>(src, words, sink);
            else
                stream_read<1><<<grid, s.threads>>>(src, words, sink);
        } else {
            if (s.loads == 4)
                stream_copy<4><<<grid, s.threads>>>(src, dst, words);
            else
                stream_copy<1><<<grid, s.threads>>>(src, dst, words);
        }
        TORCH_CUDA_CHECK(cudaEventRecord(stop));
        TORCH_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (pass > 0)
            best = std::min(best, ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    const double moved = static_cast<double>(bytes) * (dst == nullptr ? 1 : 2);
    return moved / (best * 1e-3) / 1e9;
}

// Sweeps the launch shapes and prints the best read and the best copy for one memory type.
void measure(const char* name, const void* a, void* b, size_t bytes, unsigned* sink, int sms) {
    double best_read = 0, best_copy = 0;
    Shape read_shape{}, copy_shape{};
    for (int bps : {2, 4, 8, 16}) {
        for (int threads : {256, 512}) {
            for (int loads : {1, 4}) {
                const Shape s{bps, threads, loads};
                const double r = time_passes(s, a, nullptr, bytes, sink, sms);
                if (r > best_read)
                    best_read = r, read_shape = s;
                const double c = time_passes(s, a, b, bytes, sink, sms);
                if (c > best_copy)
                    best_copy = c, copy_shape = s;
            }
        }
    }
    std::printf("  %-38s read %6.1f GB/s (%2d x %3d threads, %d loads)   copy %6.1f GB/s (%2d x %3d, %d)\n", name,
                best_read, read_shape.blocks_per_sm, read_shape.threads, read_shape.loads, best_copy,
                copy_shape.blocks_per_sm, copy_shape.threads, copy_shape.loads);
}

std::string command_output(const char* command) {
    std::string out;
    if (std::FILE* p = popen(command, "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, p))
            out += line;
        pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
        out.pop_back();
    return out.empty() ? "unknown" : out;
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
        int sms = 0;
        TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        unsigned* sink = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
        void* scratch = nullptr;  // the copy destination, device memory for every source type
        TORCH_CUDA_CHECK(cudaMalloc(&scratch, bytes));

        std::printf("GPU streaming, %.2f GiB buffers, best of 10 passes per launch shape, %d SMs\n", gib, sms);
        std::printf("  power mode: %s\n", command_output("nvpmodel -q 2>/dev/null | head -1").c_str());
        ClockSampler clock;

        void* device = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&device, bytes));
        TORCH_CUDA_CHECK(cudaMemset(device, 1, bytes));
        measure("device memory (cudaMalloc)", device, scratch, bytes, sink, sms);
        TORCH_CUDA_CHECK(cudaFree(device));

        void* host = nullptr;
        void* host_view = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
        std::fill_n(static_cast<unsigned char*>(host), bytes, 1);  // written by the CPU, as a file read would
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&host_view, host, 0));
        measure("mapped pinned memory (cudaHostAlloc)", host_view, scratch, bytes, sink, sms);
        TORCH_CUDA_CHECK(cudaFreeHost(host));

        void* registered = std::aligned_alloc(4096, bytes);
        if (registered == nullptr)
            throw torch::Error("Out of memory for the registered buffer.");
        std::fill_n(static_cast<unsigned char*>(registered), bytes, 1);
        void* registered_view = nullptr;
        TORCH_CUDA_CHECK(cudaHostRegister(registered, bytes, cudaHostRegisterMapped));
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&registered_view, registered, 0));
        measure("registered memory (cudaHostRegister)", registered_view, scratch, bytes, sink, sms);
        TORCH_CUDA_CHECK(cudaHostUnregister(registered));
        std::free(registered);

        std::printf("  highest GPU clock seen during the passes: %ld MHz\n", clock.peak_mhz());
        cudaFree(scratch);
        cudaFree(sink);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_bench_memory: %s\n", e.what());
        return 1;
    }
    return 0;
}
