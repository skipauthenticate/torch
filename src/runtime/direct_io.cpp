#include "runtime/direct_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#include "core/error.h"

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

DirectFile::DirectFile(const std::filesystem::path& path) : path_(path.string()) {
    fd_ = open_direct(path_);
    struct stat st{};
    if (fd_ < 0 || fstat(fd_, &st) != 0) {
        const int error = errno;
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
        throw Error("Cannot open " + path_ + " for direct reads (" + std::strerror(error) +
                    "); check the path and that the file system supports O_DIRECT.");
    }
    size_ = static_cast<uint64_t>(st.st_size);
}

DirectFile::DirectFile(DirectFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), size_(other.size_), path_(std::move(other.path_)) {}

DirectFile& DirectFile::operator=(DirectFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = std::exchange(other.fd_, -1);
        size_ = other.size_;
        path_ = std::move(other.path_);
    }
    return *this;
}

DirectFile::~DirectFile() {
    if (fd_ >= 0)
        ::close(fd_);
}

uint8_t* DirectFile::read(uint64_t offset, uint64_t n, uint8_t* buffer) const {
    if (offset > size_ || n > size_ - offset)
        throw Error("Bytes [" + std::to_string(offset) + ", +" + std::to_string(n) + ") are not in " + path_ +
                    "; check that the file is complete.");
    const uint64_t first = page_floor(offset);
    const uint64_t need = offset + n - first;  // bytes from `first` that must arrive; the last page may be short
    uint64_t done = 0;
    const uint64_t total = span_bytes(offset, n);
    while (done < need) {
        const ssize_t got =
            ::pread(fd_, buffer + done, std::min(total - done, kReadChunk), static_cast<off_t>(first + done));
        const int error = errno;
        if (got < 0 && error == EINTR)
            continue;
        if (got <= 0)
            throw Error("Cannot read " + path_ + " at byte " + std::to_string(first + done) + " (" +
                        (got < 0 ? std::strerror(error) : "unexpected end of file") + "); check the disk.");
        done += static_cast<uint64_t>(got);
    }
    return buffer + (offset - first);
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
