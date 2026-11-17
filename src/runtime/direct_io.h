// Reads from the model files that bypass the page cache, and a small thread pool that runs several at once.
//
// Weights are read with O_DIRECT (F_NOCACHE on macOS) into memory the engine owns, so the kernel's page cache never
// holds a second copy of them: on the Orin the page cache and the GPU share one pool of memory, and cached model
// pages would take memory the experts need (architecture decision 1). O_DIRECT reads must start and end on page
// boundaries and land in page-aligned memory, so every read covers the whole pages around the requested bytes and
// returns a pointer to the first requested byte inside its buffer.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace torch::gguf {
class File;
}

namespace torch::runtime {

inline constexpr uint64_t kPage = 4096;

// The most bytes one read call asks the device for. On the Orin's SSD (WD_BLACK SN850X, 128 KiB maximum transfer),
// single O_DIRECT reads of ~0.9 MiB left a request uncompleted until the driver's 30 s timeout about once in 300
// reads at queue depth 1; reads of 128 KiB never stalled in 9,000 tries at queue depths 1 and 8 (2026-11-04).
inline constexpr uint64_t kReadChunk = uint64_t{128} << 10;

constexpr uint64_t page_floor(uint64_t x) {
    return x / kPage * kPage;
}

constexpr uint64_t page_ceil(uint64_t x) {
    return (x + kPage - 1) / kPage * kPage;
}

// Bytes of buffer needed to read [offset, offset + n): the whole pages that contain it.
constexpr uint64_t span_bytes(uint64_t offset, uint64_t n) {
    return page_ceil(offset + n) - page_floor(offset);
}

// A file opened for direct reads, or a model's files read as one (gguf::File::parts(): file i's byte j is byte
// base_i + j). Move-only; reads are safe from several threads at once.
class DirectFile {
public:
    // Throws Error when the file cannot be opened.
    explicit DirectFile(const std::filesystem::path& path);
    // The files of a model from gguf::File::open (one) or open_model (its shards), at the bases it gives them, so a
    // tensor's data_offset() + offset is where this reads it. Throws Error when a file cannot be opened.
    explicit DirectFile(const gguf::File& model);
    DirectFile(DirectFile&& other) noexcept;
    DirectFile& operator=(DirectFile&& other) noexcept;
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;
    ~DirectFile();

    uint64_t size() const noexcept { return size_; }  // the end of the last file

    // Reads the pages around [offset, offset + n) into `buffer`, which is page-aligned and holds
    // span_bytes(offset, n) bytes, and returns the address of byte `offset` in it. Throws Error when the range is not
    // inside one file or a read fails.
    uint8_t* read(uint64_t offset, uint64_t n, uint8_t* buffer) const;

private:
    struct Part {
        int fd;
        uint64_t base, size;
        std::string path;
    };
    void add(const std::filesystem::path& path, uint64_t base);

    std::vector<Part> parts_;  // in order of base
    uint64_t size_ = 0;
};

// A fixed set of worker threads for blocking reads. One caller at a time.
class ReadPool {
public:
    explicit ReadPool(int threads);
    ReadPool(const ReadPool&) = delete;
    ReadPool& operator=(const ReadPool&) = delete;
    ~ReadPool();

    // Calls job(i) for every i < n on the workers and returns when all have finished. If any job throws, the first
    // exception is rethrown here after the rest have finished.
    void run(int n, const std::function<void(int)>& job);

private:
    void work();

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int next_ = 0, count_ = 0, finished_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

}  // namespace torch::runtime
