// Checks the bytes from each file offset, buffer reuse, failed reads, and a batch with one spare buffer.
// The tests for a late read hold one file operation until the reader issues a second request.
//
// usage: hedged_reads_test DIR (a directory on a disk file system: tmpfs does not support O_DIRECT)
#include "runtime/hedged_reads.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

#include "check.h"

using namespace torch;
using runtime::DirectFile;
using runtime::HedgedReads;
using runtime::kPage;
using runtime::kReadChunk;
using runtime::span_bytes;

namespace {
std::atomic<const HedgedReads*> slow_read{nullptr};
}

// Only the test copy of DirectFile calls this function. A fast disk can finish every normal read before it needs a
// hedge.
extern "C" ssize_t torch_test_pread(int fd, void* buffer, size_t count, off_t offset) {
    if (const auto* reader = slow_read.exchange(nullptr)) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (reader->hedged() == 0 && std::chrono::steady_clock::now() < limit)
            std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return ::pread(fd, buffer, count, offset);
}

namespace {

void check(const DirectFile& f, const std::vector<uint8_t>& bytes, std::chrono::microseconds hedge_after, bool hedges) {
    constexpr int kReads = 6, kBuffers = kReads + 4;
    constexpr uint64_t kBuffer = 3 * kPage;
    auto* memory = static_cast<uint8_t*>(std::aligned_alloc(kPage, 2 * kBuffers * kBuffer));
    {
        HedgedReads reads(f, 4, hedge_after);
        const int small = reads.add_pool(memory, kBuffer, kBuffers);
        const int large = reads.add_pool(memory + kBuffers * kBuffer, kBuffer, kBuffers);
        CHECK(small == 0 && large == 1);
        if (hedges)
            slow_read.store(&reads);
        std::mt19937 rng(7);
        for (int round = 0; round < 200; ++round) {
            std::vector<HedgedReads::Request> requests;
            for (int i = 0; i < kReads; ++i) {
                const uint64_t n = 1 + rng() % (2 * kPage);
                requests.push_back({i % 2, rng() % (bytes.size() - n), n});
            }
            std::vector<const uint8_t*> out(kReads);
            reads.read_all(requests, out);
            for (int i = 0; i < kReads; ++i)
                CHECK(out[static_cast<size_t>(i)] != nullptr &&
                      std::memcmp(out[static_cast<size_t>(i)], bytes.data() + requests[static_cast<size_t>(i)].offset,
                                  requests[static_cast<size_t>(i)].n) == 0);
            reads.release();
        }
        CHECK(slow_read.exchange(nullptr) == nullptr);
        CHECK(hedges ? reads.hedged() > 0 : reads.hedged() == 0);
        // Past the end of the file: the read fails and its result is nullptr.
        const HedgedReads::Request bad{0, bytes.size() - 10, 100};
        const uint8_t* out = &bytes[0];
        reads.read_all(std::span(&bad, 1), std::span(&out, 1));
        CHECK(out == nullptr);
        reads.release();
    }
    std::free(memory);
}

// Requests of several chunks (128 KiB each) from unaligned offsets, one that ends at the end of the file, and the
// reads of one batch finishing in any order: the bytes must be exact, with and without the whole request being
// re-issued.
void check_chunks(const DirectFile& f, const std::vector<uint8_t>& bytes, std::chrono::microseconds hedge_after,
                  bool hedges) {
    constexpr int kReads = 4, kBuffers = kReads + 4;
    const uint64_t buffer_bytes = span_bytes(1, 150 * kPage);  // the largest request below
    auto* memory = static_cast<uint8_t*>(std::aligned_alloc(kPage, kBuffers * buffer_bytes));
    {
        HedgedReads reads(f, 8, hedge_after);
        const int pool = reads.add_pool(memory, buffer_bytes, kBuffers);
        if (hedges)
            slow_read.store(&reads);
        std::mt19937 rng(11);
        for (int round = 0; round < 60; ++round) {
            std::vector<HedgedReads::Request> requests;
            for (int i = 0; i < kReads; ++i) {
                const uint64_t n = 1 + rng() % (150 * kPage);
                // the last request of each round ends exactly at the end of the file
                const uint64_t offset = i == kReads - 1 ? bytes.size() - n : rng() % (bytes.size() - n);
                requests.push_back({pool, offset, n});
            }
            std::vector<const uint8_t*> out(kReads);
            reads.read_all(requests, out);
            for (size_t i = 0; i < kReads; ++i)
                CHECK(out[i] != nullptr && std::memcmp(out[i], bytes.data() + requests[i].offset, requests[i].n) == 0);
            reads.release();
        }
        CHECK(slow_read.exchange(nullptr) == nullptr);
        CHECK(hedges ? reads.hedged() > 0 : reads.hedged() == 0);
        // a multi-chunk request that runs past the end of the file fails as a whole
        const HedgedReads::Request bad{pool, bytes.size() - 3 * kReadChunk, 4 * kReadChunk};
        const uint8_t* out = &bytes[0];
        reads.read_all(std::span(&bad, 1), std::span(&out, 1));
        CHECK(out == nullptr);
        reads.release();
    }
    std::free(memory);
}

// A large batch has one spare buffer. A duplicate read that needs the spare must let the batch make progress.
void check_many_late(const DirectFile& f, const std::vector<uint8_t>& bytes) {
    constexpr int kReads = 48, kBuffers = kReads + 1;
    const uint64_t buffer_bytes = span_bytes(1, 40 * kPage);
    auto* memory = static_cast<uint8_t*>(std::aligned_alloc(kPage, kBuffers * buffer_bytes));
    {
        HedgedReads reads(f, 4, std::chrono::microseconds(0));
        const int pool = reads.add_pool(memory, buffer_bytes, kBuffers);
        slow_read.store(&reads);
        std::mt19937 rng(13);
        for (int round = 0; round < 20; ++round) {
            std::vector<HedgedReads::Request> requests;
            for (int i = 0; i < kReads; ++i) {
                const uint64_t n = 1 + rng() % (40 * kPage);
                requests.push_back({pool, rng() % (bytes.size() - n), n});
            }
            std::vector<const uint8_t*> out(kReads);
            reads.read_all(requests, out);
            for (size_t i = 0; i < kReads; ++i)
                CHECK(out[i] != nullptr && std::memcmp(out[i], bytes.data() + requests[i].offset, requests[i].n) == 0);
            reads.release();
        }
        CHECK(slow_read.exchange(nullptr) == nullptr);
        CHECK(reads.hedged() > 0);
    }
    std::free(memory);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: hedged_reads_test DIR\n");
        return 2;
    }
    std::vector<uint8_t> bytes(400 * kPage);  // 1.6 MB: several chunks
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(i * 7 % 251);
    const std::filesystem::path path = std::filesystem::path(argv[1]) / "hedged_reads_test.bin";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::ssize(bytes));
    {
        const DirectFile f(path);
        check(f, bytes, std::chrono::seconds(10), false);     // never hedged
        check(f, bytes, std::chrono::microseconds(0), true);  // one controlled late read
        check_chunks(f, bytes, std::chrono::seconds(10), false);
        check_chunks(f, bytes, std::chrono::microseconds(0), true);
        check_many_late(f, bytes);
    }
    std::filesystem::remove(path);
    return test::test_exit_code();
}
