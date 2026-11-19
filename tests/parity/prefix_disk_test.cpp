#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "core/cuda_check.h"
#include "runtime/prefix_storage.h"

int main() {
    using namespace torch::runtime;
    char dir[] = "prefix-disk-XXXXXX";
    if (!mkdtemp(dir))
        return 1;
    uint8_t* device = nullptr;
    void* aligned = nullptr;
    int result = 1;
    try {
        constexpr size_t count = 10003;
        std::vector<uint8_t> input(count), output(count);
        for (size_t i = 0; i < count; ++i)
            input[i] = static_cast<uint8_t>((i * 37 + 7) % 251);
        TORCH_CUDA_CHECK(cudaMalloc(&device, count));
        TORCH_CUDA_CHECK(cudaHostAlloc(&aligned, 4096, cudaHostAllocDefault));
        TORCH_CUDA_CHECK(cudaMemcpy(device, input.data(), count, cudaMemcpyHostToDevice));
        PrefixDisk disk(dir, nullptr);
        DeviceRanges parts{{device + 2, 7011}, {device + 7100, 2900}};
        auto file = disk.write(parts);
        if (!file || !std::filesystem::is_empty(dir))
            throw std::runtime_error("disk file is missing or visible");
        TORCH_CUDA_CHECK(cudaMemset(device, 0, count));
        if (!disk.read(*file, parts))
            throw std::runtime_error("disk restore failed");
        TORCH_CUDA_CHECK(cudaMemcpy(output.data(), device, count, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            const bool kept = (i >= 2 && i < 7013) || (i >= 7100 && i < 10000);
            if (output[i] != (kept ? input[i] : 0))
                throw std::runtime_error("restored bytes differ");
        }
        std::fill_n(static_cast<uint8_t*>(aligned), 4096, 0);
        if (::pwrite(file->fd, aligned, 4096, 0) != 4096 || disk.read(*file, parts))
            throw std::runtime_error("damaged disk entry passed its checksum");
        result = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
    }
    if (aligned)
        cudaFreeHost(aligned);
    if (device)
        cudaFree(device);
    ::rmdir(dir);
    return result;
}
