#include <cuda_runtime.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "runtime/prefix_storage.h"

int main() {
    using namespace torch::runtime;
    char dir[] = "prefix-disk-XXXXXX";
    if (!mkdtemp(dir))
        return 1;
    const auto alias = std::filesystem::path(dir) / "alias";
    uint8_t* device = nullptr;
    void* aligned = nullptr;
    int result = 1;
    try {
        const auto refused = [](const std::filesystem::path& path) {
            try {
                PrefixDisk disk(path, nullptr);
            } catch (const torch::Error&) {
                return true;
            }
            return false;
        };
        if (::chmod(dir, 0750) || !refused(dir) || ::chmod(dir, 0700))
            throw std::runtime_error("disk cache accepted group access");
        std::filesystem::create_directory_symlink(".", alias);
        if (!refused(alias))
            throw std::runtime_error("disk cache followed a directory symlink");
        std::filesystem::remove(alias);
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
        struct stat st{};
        if (!file || ::fstat(file->fd, &st) || (st.st_mode & 0777) != 0600 || st.st_nlink ||
            !std::filesystem::is_empty(dir))
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
        // A file system without unnamed direct-I/O files (tmpfs on many kernels) is refused at start with a message
        // that says what to do; one that is accepted must then park and restore.
        char shm[] = "/dev/shm/prefix-disk-XXXXXX";
        if (mkdtemp(shm)) {
            std::string refusal;
            bool restored = false;
            try {
                PrefixDisk tmpfs(shm, nullptr);
                auto f = tmpfs.write(parts);
                restored = f && tmpfs.read(*f, parts);
            } catch (const torch::Error& e) {
                refusal = e.what();
            }
            ::rmdir(shm);
            if (refusal.empty() ? !restored : refusal.find("give --prefix-disk a directory") == std::string::npos)
                throw std::runtime_error("a directory without direct-I/O files was accepted, or refused unclearly");
            std::printf("/dev/shm: %s\n",
                        refusal.empty() ? "accepted, and a checkpoint round trip works" : refusal.c_str());
        }
        result = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
    }
    if (aligned)
        cudaFreeHost(aligned);
    if (device)
        cudaFree(device);
    ::unlink(alias.c_str());
    ::rmdir(dir);
    return result;
}
