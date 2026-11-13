#include "runtime/prefill_reads.h"

#include <algorithm>

#include "core/cuda_check.h"
#include "core/error.h"

namespace torch::runtime {
namespace {

constexpr int kReadThreads = 16;  // as the expert store's: 128 KiB reads reach ~5.4 GB/s at queue depth 16
constexpr int kSpare = 4;         // buffers per kind beyond one per slot, for reads still in flight after a hedge

const model::ExpertTensor& tensor(const model::Layer& L, int kind) {
    return kind == 0 ? L.gate_exps : kind == 1 ? L.up_exps : L.down_exps;
}

// A slot holds one matrix at the same offset within a page as in the file: its size rounded up to pages, plus the
// page the slice may start inside (as the expert store's).
uint64_t slot_bytes(uint64_t slice) {
    return page_ceil(slice) + kPage;
}

uint64_t largest_slice(const gguf::File& shard1, int n_layer, const char* kind) {
    uint64_t most = 0;
    for (int il = 0; il < n_layer; ++il)
        if (const gguf::TensorInfo* t = shard1.tensor("blk." + std::to_string(il) + ".ffn_" + kind + "_exps.weight"))
            most = std::max(most, t->slice_bytes());
    return most;
}

}  // namespace

uint64_t PrefillReads::bytes(const model::Config& config, const gguf::File& shard1, int slots) {
    uint64_t per_expert = 0;
    for (const char* kind : {"gate", "up", "down"})
        per_expert += slot_bytes(largest_slice(shard1, config.n_layer, kind));
    return 2 * static_cast<uint64_t>(slots + kSpare) * per_expert;
}

PrefillReads::PrefillReads(const std::vector<model::Layer>& layers, const DirectFile& file, int slots)
    : layers_(layers), slots_(slots) {
    uint64_t stride[3] = {};
    for (int kind = 0; kind < 3; ++kind)
        for (const model::Layer& L : layers)
            stride[kind] = std::max(stride[kind], slot_bytes(tensor(L, kind).slice_bytes));
    const uint64_t per_pool = static_cast<uint64_t>(slots + kSpare) * (stride[0] + stride[1] + stride[2]);
    void* p = nullptr;
    TORCH_CUDA_CHECK(cudaHostAlloc(&p, 2 * per_pool, cudaHostAllocMapped));
    memory_ = static_cast<uint8_t*>(p);
    for (int pool = 0; pool < 2; ++pool) {
        io_[pool] = std::make_unique<HedgedReads>(file, kReadThreads);
        uint8_t* at = memory_ + pool * per_pool;
        for (int kind = 0; kind < 3; ++kind) {
            io_[pool]->add_pool(at, stride[kind], slots + kSpare);  // pool id = kind
            at += static_cast<uint64_t>(slots + kSpare) * stride[kind];
        }
    }
    thread_ = std::thread([this] { work(); });
}

PrefillReads::~PrefillReads() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable())
        thread_.join();
    io_[0].reset(), io_[1].reset();  // their reads finish before the memory goes
    for (cudaEvent_t e : events_)
        cudaEventDestroy(e);
    if (memory_)
        cudaFreeHost(memory_);
}

int PrefillReads::request(int layer, std::vector<int32_t> experts) {
    if (static_cast<int>(experts.size()) > slots_)
        throw Error("A prefill read holds at most " + std::to_string(slots_) +
                    " experts; this is a bug in the engine.");
    std::lock_guard lock(mutex_);
    queue_.push_back({layer, std::move(experts), {}});
    wake_.notify_all();
    return next_++;
}

std::vector<const uint8_t*> PrefillReads::wait(int ticket) {
    std::unique_lock lock(mutex_);
    wake_.wait(lock, [&] { return queue_[static_cast<size_t>(ticket)].done; });
    const Request& r = queue_[static_cast<size_t>(ticket)];
    if (r.failed)
        throw Error("Reading an expert of layer " + std::to_string(r.layer) + " from the SSD failed.");
    return r.data;
}

void PrefillReads::release(int ticket, cudaStream_t stream) {
    std::lock_guard lock(mutex_);
    while (static_cast<int>(events_.size()) <= ticket) {
        cudaEvent_t e = nullptr;
        TORCH_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        events_.push_back(e);
    }
    Request& r = queue_[static_cast<size_t>(ticket)];
    TORCH_CUDA_CHECK(cudaEventRecord(events_[static_cast<size_t>(ticket)], stream));
    r.released = events_[static_cast<size_t>(ticket)];
    r.has_released = true;
    wake_.notify_all();
}

void PrefillReads::reset() {
    std::unique_lock lock(mutex_);
    queue_.resize(static_cast<size_t>(reading_ + (busy_ ? 1 : 0)));  // unread requests go
    next_ = static_cast<int>(queue_.size());
    for (Request& r : queue_)  // the caller has waited for the GPU: nothing is read any more
        r.has_released = true, r.released = nullptr;
    wake_.notify_all();
    wake_.wait(lock, [&] { return !busy_; });
    queue_.clear();
    next_ = reading_ = 0;
    io_[0]->release(), io_[1]->release();
}

void PrefillReads::work() {
    for (;;) {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
            return stop_ ||
                   (reading_ < next_ && (reading_ < 2 || queue_[static_cast<size_t>(reading_ - 2)].has_released));
        });
        if (stop_)
            return;
        const int t = reading_;
        const cudaEvent_t before = t >= 2 ? queue_[static_cast<size_t>(t - 2)].released : nullptr;
        const int layer = queue_[static_cast<size_t>(t)].layer;
        const std::vector<int32_t> experts = queue_[static_cast<size_t>(t)].experts;
        busy_ = true;
        lock.unlock();
        bool failed = false;
        std::vector<const uint8_t*> data(3 * experts.size());
        try {
            if (before)
                TORCH_CUDA_CHECK(cudaEventSynchronize(before));  // the GPU is done with this pool's last bytes
            io_[t % 2]->release();
            const model::Layer& L = layers_[static_cast<size_t>(layer)];
            std::vector<HedgedReads::Request> reads(data.size());
            for (size_t i = 0; i < experts.size(); ++i)
                for (int kind = 0; kind < 3; ++kind) {
                    const model::ExpertTensor& m = tensor(L, kind);
                    reads[3 * i + static_cast<size_t>(kind)] = {
                        kind, m.file_offset + static_cast<uint64_t>(experts[i]) * m.slice_bytes, m.slice_bytes};
                }
            io_[t % 2]->read_all(reads, data);
            for (const uint8_t* d : data)
                failed = failed || d == nullptr;
        } catch (...) {
            failed = true;
        }
        lock.lock();
        Request& r = queue_[static_cast<size_t>(t)];
        r.data = std::move(data);
        r.failed = failed;
        r.done = true;
        ++reading_;
        busy_ = false;
        wake_.notify_all();
    }
}

}  // namespace torch::runtime
