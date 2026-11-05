#include "runtime/hedged_reads.h"

#include <exception>

namespace torch::runtime {
namespace {

// kFree: unused. kInUse: a read in flight that read_all() waits for, or bytes the caller may be reading (until
// release()). kLate: a read that lost to its hedge and is still in flight; its buffer is freed when it completes.
enum BufferState : int { kFree, kInUse, kLate };

using Clock = std::chrono::steady_clock;

}  // namespace

struct HedgedReads::Buffer {
    uint8_t* base = nullptr;
    std::atomic<int> state{kFree};
};

struct HedgedReads::Task {
    Buffer* buffer = nullptr;
    uint64_t offset = 0, n = 0;
    std::atomic<int> status{0};  // 0 pending, 1 done, 2 failed
    const uint8_t* result = nullptr;
};

HedgedReads::HedgedReads(const DirectFile& file, int threads, std::chrono::microseconds hedge_after)
    : file_(file), hedge_after_(hedge_after) {
    for (int i = 0; i < threads; ++i)
        workers_.emplace_back([this] { work(); });
}

HedgedReads::~HedgedReads() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_)
        t.join();
}

int HedgedReads::add_pool(uint8_t* base, uint64_t buffer_bytes, int count) {
    std::vector<std::unique_ptr<Buffer>> pool;
    for (int i = 0; i < count; ++i) {
        pool.push_back(std::make_unique<Buffer>());
        pool.back()->base = base + static_cast<uint64_t>(i) * buffer_bytes;
    }
    pools_.push_back(std::move(pool));
    return static_cast<int>(pools_.size()) - 1;
}

void HedgedReads::work() {
    for (;;) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty())
                return;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            task->result = file_.read(task->offset, task->n, task->buffer->base);
            task->status.store(1, std::memory_order_release);
        } catch (const std::exception&) {
            task->status.store(2, std::memory_order_release);
        }
        int late = kLate;  // a read that lost to its hedge frees its buffer only now
        task->buffer->state.compare_exchange_strong(late, kFree);
    }
}

std::shared_ptr<HedgedReads::Task> HedgedReads::submit(const Request& r) {
    auto task = std::make_shared<Task>();
    task->buffer = take(r.pool);
    task->offset = r.offset;
    task->n = r.n;
    {
        std::lock_guard lock(mutex_);
        queue_.push_back(task);
    }
    wake_.notify_one();
    return task;
}

// A free buffer of the pool, marked in use; waits for a late read to finish if all are taken.
HedgedReads::Buffer* HedgedReads::take(int pool) {
    for (;;) {
        for (const std::unique_ptr<Buffer>& b : pools_[static_cast<size_t>(pool)]) {
            int expected = kFree;
            if (b->state.compare_exchange_strong(expected, kInUse))
                return b.get();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// The loser of a hedge: free its buffer now if its read is done, otherwise once it completes.
void HedgedReads::abandon(Task& t) {
    int in_use = kInUse;
    t.buffer->state.compare_exchange_strong(in_use, kLate);
    if (t.status.load(std::memory_order_acquire) != 0) {  // it may have completed in between
        int late = kLate;
        t.buffer->state.compare_exchange_strong(late, kFree);
    }
}

void HedgedReads::read_all(std::span<const Request> requests, std::span<const uint8_t*> out) {
    const auto start = Clock::now();
    struct Pending {
        std::shared_ptr<Task> first, hedge;
    };
    std::vector<Pending> pending(requests.size());
    for (size_t i = 0; i < requests.size(); ++i)
        pending[i].first = submit(requests[i]);
    size_t open = requests.size();
    while (open > 0) {
        for (size_t i = 0; i < requests.size(); ++i) {
            Pending& p = pending[i];
            if (!p.first)
                continue;  // resolved
            std::shared_ptr<Task> winner, loser;
            if (p.first->status.load(std::memory_order_acquire) != 0) {
                winner = p.first, loser = p.hedge;
            } else if (p.hedge && p.hedge->status.load(std::memory_order_acquire) != 0) {
                winner = p.hedge, loser = p.first;
            } else if (!p.hedge && Clock::now() - start > hedge_after_) {
                p.hedge = submit(requests[i]);
                ++hedged_;
            }
            if (!winner)
                continue;
            out[i] = winner->status.load() == 1 ? winner->result : nullptr;
            held_.push_back(winner->buffer);
            if (loser)
                abandon(*loser);
            p.first.reset();
            p.hedge.reset();
            --open;
        }
        if (open > 0)
            std::this_thread::yield();
    }
}

void HedgedReads::release() {
    for (Buffer* b : held_) {
        int in_use = kInUse;
        b->state.compare_exchange_strong(in_use, kFree);
    }
    held_.clear();
}

}  // namespace torch::runtime
