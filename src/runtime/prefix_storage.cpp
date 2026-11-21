#include "runtime/prefix_storage.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <future>
#include <limits>

#include "core/cuda_check.h"
#include "core/error.h"
#include "runtime/direct_io.h"

namespace torch::runtime {
namespace {
constexpr size_t kHalf = PrefixDisk::staging_bytes / 2;

uint32_t checksum(const uint8_t* data, size_t n) {
    static constexpr auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < values.size(); ++i) {
            uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit)
                c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0u);
            values[i] = c;
        }
        return values;
    }();
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; ++i)
        c = table[(c ^ data[i]) & 255] ^ (c >> 8);
    return ~c;
}

void copy_slice(const DeviceRanges& parts, size_t skip, size_t length, uint8_t* packed, bool save, cudaMemcpyKind kind,
                cudaStream_t stream) {
    for (auto part : parts) {
        if (skip >= part.size()) {
            skip -= part.size();
            continue;
        }
        const size_t n = std::min(length, part.size() - skip);
        auto* active = part.data() + skip;
        TORCH_CUDA_CHECK(cudaMemcpyAsync(save ? packed : active, save ? active : packed, n, kind, stream));
        packed += n;
        length -= n;
        if (!length)
            break;
        skip = 0;
    }
    TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool file_io(int fd, uint8_t* data, size_t bytes, uint64_t at, bool save) {
    while (bytes) {
        const size_t n = std::min<size_t>(bytes, kReadChunk);
        const ssize_t taken =
            save ? ::pwrite(fd, data, n, static_cast<off_t>(at)) : ::pread(fd, data, n, static_cast<off_t>(at));
        if (taken < 0 && errno == EINTR)
            continue;
        if (taken <= 0 || static_cast<size_t>(taken) % kPage != 0)
            return false;
        const auto moved = static_cast<size_t>(taken);
        bytes -= moved;
        at += moved;
        data += moved;
    }
    return true;
}
}  // namespace

uint64_t range_bytes(const DeviceRanges& parts) {
    uint64_t n = 0;
    for (auto p : parts)
        n += p.size();
    return n;
}

void copy_device(const DeviceRanges& parts, uint8_t* packed, bool save, cudaStream_t stream) {
    copy_slice(parts, 0, range_bytes(parts), packed, save, cudaMemcpyDeviceToDevice, stream);
}

PrefixFile::~PrefixFile() {
    if (fd >= 0)
        ::close(fd);
}

PrefixDisk::PrefixDisk(const std::filesystem::path& dir, cudaStream_t stream) : stream_(stream) {
    if (::mkdir(dir.c_str(), 0700) < 0 && errno != EEXIST)
        throw Error("Cannot create the private prefix cache directory.");
    directory_ = ::open(dir.c_str(), O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_RDONLY);
    struct stat st{};
    if (directory_ < 0 || ::fstat(directory_, &st) < 0 || st.st_uid != ::geteuid() || (st.st_mode & 0777) != 0700) {
        if (directory_ >= 0)
            ::close(directory_);
        throw Error("The prefix cache directory must belong to the server user and have mode 0700.");
    }
    const auto rc = cudaHostAlloc(&staging_, staging_bytes, cudaHostAllocDefault);
    if (rc != cudaSuccess) {
        ::close(directory_);
        TORCH_CUDA_CHECK(rc);
    }
    // Parked checkpoints are unnamed direct-I/O files, which some file systems refuse (tmpfs on many kernels, overlay,
    // network file systems): write and read one page now, so such a directory fails at start instead of every
    // checkpoint being dropped later.
    const int probe = ::openat(directory_, ".", O_TMPFILE | O_RDWR | O_CLOEXEC | O_DIRECT, 0600);
    bool ok = probe >= 0;
    if (ok) {
        std::memset(staging_, 0x5a, kPage);
        ok = ::pwrite(probe, staging_, kPage, 0) == static_cast<ssize_t>(kPage) &&
             ::pread(probe, staging_, kPage, 0) == static_cast<ssize_t>(kPage);
    }
    const int error = ok ? 0 : errno;
    if (probe >= 0)
        ::close(probe);
    if (!ok) {
        cudaFreeHost(staging_);
        ::close(directory_);
        throw Error("The prefix cache directory " + dir.string() + " cannot hold unnamed direct-I/O files (" +
                    std::strerror(error) +
                    "); give --prefix-disk a directory on a local ext4 or XFS file system, or leave it out.");
    }
}

PrefixDisk::~PrefixDisk() {
    if (staging_)
        cudaFreeHost(staging_);
    if (directory_ >= 0)
        ::close(directory_);
}

std::shared_ptr<PrefixFile> PrefixDisk::write(const DeviceRanges& parts) {
    auto file = std::make_shared<PrefixFile>();
    file->fd = ::openat(directory_, ".", O_TMPFILE | O_RDWR | O_CLOEXEC | O_DIRECT, 0600);
    if (file->fd < 0)
        return {};
    if (!transfer(*file, parts, true))
        return {};
    return file;
}

bool PrefixDisk::read(PrefixFile& file, const DeviceRanges& parts) {
    return transfer(file, parts, false);
}

void PrefixDisk::damage(PrefixFile& file) {
    std::memset(staging_, 0xa5, kPage);
    if (::pwrite(file.fd, staging_, kPage, 0) != static_cast<ssize_t>(kPage))
        throw Error("Cannot damage a prefix checkpoint file for the test.");
}

bool PrefixDisk::transfer(PrefixFile& file, const DeviceRanges& parts, bool save) {
    const size_t bytes = range_bytes(parts), chunks = (bytes + kHalf - 1) / kHalf;
    if (save)
        file.checksums.resize(chunks);
    if (file.checksums.size() != chunks)
        return false;
    std::array<std::future<bool>, 2> pending;
    const auto start_io = [&](size_t i) {
        const size_t offset = i * kHalf, n = std::min(kHalf, bytes - offset);
        uint8_t* buf = staging_ + (i % 2) * kHalf;
        return std::async(std::launch::async | std::launch::deferred, [&, i, offset, n, buf] {
            if (save) {
                std::memset(buf + n, 0, page_ceil(n) - n);
                file.checksums[i] = checksum(buf, n);
            }
            return file_io(file.fd, buf, page_ceil(n), offset, save) && (save || checksum(buf, n) == file.checksums[i]);
        });
    };
    if (!save)
        for (size_t i = 0; i < std::min<size_t>(2, chunks); ++i)
            pending[i] = start_io(i);
    for (size_t i = 0; i < chunks; ++i) {
        auto& io = pending[i % 2];
        if (io.valid() && !io.get())
            return false;
        const size_t at = i * kHalf, n = std::min(kHalf, bytes - at);
        copy_slice(parts, at, n, staging_ + (i % 2) * kHalf, save,
                   save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, stream_);
        if (save)
            io = start_io(i);
        else if (i + 2 < chunks)
            io = start_io(i + 2);
    }
    for (auto& io : pending)
        if (io.valid() && !io.get())
            return false;
    return true;
}

}  // namespace torch::runtime
