#include "runtime/direct_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#include "core/error.h"
#include "gguf/gguf.h"

namespace torch::runtime {
namespace {

int open_direct(const std::string& path) {
#if defined(__APPLE__)
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0 && fcntl(fd, F_NOCACHE, 1) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
#else
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
}

}  // namespace

DirectFile::DirectFile(const std::filesystem::path& path) {
    add(path, 0);
}

DirectFile::DirectFile(const gguf::File& model) {
    if (model.parts().empty())
        throw Error("A GGUF header parsed from memory has no file to read; open the model with gguf::File.");
    try {
        for (const gguf::Part& p : model.parts())
            add(p.path, p.base);
    } catch (...) {
        for (const Part& p : parts_)  // a constructor that throws runs no destructor
            ::close(p.fd);
        throw;
    }
}

void DirectFile::add(const std::filesystem::path& path, uint64_t base) {
    if (base % kPage != 0 || base < size_)
        throw Error("The files of a model must start on page boundaries, in order; this is a bug in the engine.");
    const int fd = open_direct(path.string());
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) != 0) {
        const int error = errno;
        if (fd >= 0)
            ::close(fd);
        throw Error("Cannot open " + path.string() + " for direct reads (" + std::strerror(error) +
                    "); check the path and that the file system supports O_DIRECT.");
    }
    try {
        parts_.push_back({fd, base, static_cast<uint64_t>(st.st_size), path.string()});
    } catch (...) {
        ::close(fd);
        throw;
    }
    size_ = base + parts_.back().size;
}

DirectFile::DirectFile(DirectFile&& other) noexcept
    : parts_(std::exchange(other.parts_, {})), size_(std::exchange(other.size_, 0)) {}

DirectFile& DirectFile::operator=(DirectFile&& other) noexcept {
    std::swap(parts_, other.parts_);  // other's destructor closes what this held
    std::swap(size_, other.size_);
    return *this;
}

DirectFile::~DirectFile() {
    for (const Part& p : parts_)
        ::close(p.fd);
}

uint8_t* DirectFile::read(uint64_t offset, uint64_t n, uint8_t* buffer) const {
    // The last file that starts at or before `offset`.
    auto it =
        std::upper_bound(parts_.begin(), parts_.end(), offset, [](uint64_t o, const Part& p) { return o < p.base; });
    const Part* p = it == parts_.begin() ? nullptr : &*std::prev(it);
    if (p == nullptr || offset - p->base > p->size || n > p->size - (offset - p->base))
        throw Error("Bytes [" + std::to_string(offset) + ", +" + std::to_string(n) + ") are not in " +
                    (p ? p->path : std::string("the model's files")) + "; check that the file is complete.");
    const uint64_t local = offset - p->base;  // in the file; its base is page-aligned, so the pages line up
    const uint64_t first = page_floor(local);
    const uint64_t need = local + n - first;  // bytes from `first` that must arrive; the last page may be short
    uint64_t done = 0;
    const uint64_t total = span_bytes(offset, n);
    while (done < need) {
        const ssize_t got =
            ::pread(p->fd, buffer + done, std::min(total - done, kReadChunk), static_cast<off_t>(first + done));
        const int error = errno;
        if (got < 0 && error == EINTR)
            continue;
        if (got <= 0)
            throw Error("Cannot read " + p->path + " at byte " + std::to_string(first + done) + " (" +
                        (got < 0 ? std::strerror(error) : "unexpected end of file") + "); check the disk.");
        done += static_cast<uint64_t>(got);
    }
    return buffer + (local - first);
}

ReadPool::ReadPool(int threads) {
    for (int i = 0; i < threads; ++i)
        threads_.emplace_back([this] { work(); });
}

ReadPool::~ReadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : threads_)
        t.join();
}

void ReadPool::run(int n, const std::function<void(int)>& job) {
    if (n <= 0)
        return;
    std::unique_lock lock(mutex_);
    job_ = &job;
    next_ = 0;
    count_ = n;
    finished_ = 0;
    error_ = nullptr;
    ++generation_;
    wake_.notify_all();
    done_.wait(lock, [&] { return finished_ == count_; });
    job_ = nullptr;
    if (error_)
        std::rethrow_exception(error_);
}

void ReadPool::work() {
    uint64_t seen = 0;
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || (generation_ != seen && next_ < count_); });
        if (stop_)
            return;
        while (next_ < count_) {
            const int i = next_++;
            lock.unlock();
            std::exception_ptr error;
            try {
                (*job_)(i);
            } catch (...) {
                error = std::current_exception();
            }
            lock.lock();
            if (error && !error_)
                error_ = error;
            if (++finished_ == count_)
                done_.notify_one();
        }
        seen = generation_;
    }
}

}  // namespace torch::runtime
