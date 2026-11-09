// File reads that do not wait for a stalled SSD request.
//
// The development Orin's SSD occasionally leaves one read uncompleted until the NVMe driver's 30 s timeout (about once
// in 240,000 reads of 128 KiB, with O_DIRECT and through the page cache alike) while other reads keep completing; a
// 4 KiB read of the PLE rows stalled the same way. So every read that is still pending after a short delay is issued
// again into a spare buffer, and the first copy to arrive is used. The loser's buffer stays out of use until its read
// completes, so a late write never lands in memory the caller reads.
//
// A request is read as page-aligned chunks of kReadChunk (128 KiB) that different workers read at once, so one expert's
// slices keep many reads in flight on the SSD instead of one at a time. A request is done when its last chunk is; the
// hedge re-issues the whole request.
//
// The buffers are the caller's memory, in pools of equal-sized buffers; a pool needs at least one spare buffer more
// than the reads one read_all() puts into it.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "runtime/direct_io.h"

namespace torch::runtime {

class HedgedReads {
public:
    struct Request {
        int pool;         // the buffer pool the bytes land in (add_pool)
        uint64_t offset;  // in the file
        uint64_t n;
    };

    // The default delay: 10 concurrent expert reads take 2.6 ms median and 8.9 ms p99 (bench/miss_roundtrip.cu), 16
    // PLE rows 0.3 ms, and a stalled read 30 s.
    HedgedReads(const DirectFile& file, int threads,
                std::chrono::microseconds hedge_after = std::chrono::milliseconds(50));
    HedgedReads(const HedgedReads&) = delete;
    HedgedReads& operator=(const HedgedReads&) = delete;
    ~HedgedReads();  // waits for the reads still in flight

    // A pool of `count` buffers of `buffer_bytes` each, at base, base + buffer_bytes, ...; returns its id. Each buffer
    // must hold a read's page-aligned superset (DirectFile::read). Call before the first read.
    int add_pool(uint8_t* base, uint64_t buffer_bytes, int count);

    // Reads every request in parallel; out[i] is where request i's bytes are, or nullptr when its read failed. The
    // buffers stay reserved until release().
    void read_all(std::span<const Request> requests, std::span<const uint8_t*> out);
    // Frees the buffers of every read since the last release().
    void release();

    uint64_t hedged() const noexcept { return hedged_.load(); }  // reads issued a second time

private:
    struct Buffer;
    struct Task;
    struct Chunk {
        std::shared_ptr<Task> task;
        uint32_t j;  // the chunk's index in its read
    };
    void work();
    std::shared_ptr<Task> submit(const Request& r);
    Buffer* take(int pool);
    static void abandon(Task& t);

    const DirectFile& file_;
    const std::chrono::microseconds hedge_after_;
    std::vector<std::vector<std::unique_ptr<Buffer>>> pools_;
    std::vector<Buffer*> held_;  // the winners' buffers since the last release()
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Chunk> queue_;
    bool stop_ = false;
    std::atomic<uint64_t> hedged_{0};
    std::vector<std::thread> workers_;
};

}  // namespace torch::runtime
