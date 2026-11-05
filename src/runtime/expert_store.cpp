#include "runtime/expert_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <span>
#include <string>

#include "core/cuda_check.h"
#include "core/error.h"

namespace torch::runtime {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kReadThreads = 16;  // 128 KiB reads reach ~5.4 GB/s at queue depth 16 (bench/ssd_reads.cpp)
constexpr int kSpareBuffers =
    4;  // staging buffers per kind beyond one per slot, for reads still in flight after a hedge

// A slot holds one expert matrix at the same offset within a page as in the file: its size rounded up to pages, plus
// the page the slice may start inside.
uint64_t slot_bytes(uint64_t slice) {
    return page_ceil(slice) + kPage;
}

uint64_t slice_of(const gguf::File& shard1, int layer, const char* kind) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight";
    const gguf::TensorInfo* t = shard1.tensor(name);
    if (t == nullptr)
        throw Error("The model file has no tensor " + name + "; use the supported model file.");
    return t->slice_bytes();
}

constexpr const char* kKinds[3] = {"gate", "up", "down"};

const model::ExpertTensor& tensor(const model::Layer& L, int kind) {
    return kind == 0 ? L.gate_exps : kind == 1 ? L.up_exps : L.down_exps;
}

template <class T>
T* device_alloc(size_t count) {
    void* p = nullptr;
    TORCH_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    TORCH_CUDA_CHECK(cudaMemset(p, 0, count * sizeof(T)));
    return static_cast<T*>(p);
}

template <class T>
T* mapped_alloc(size_t count) {
    void* p = nullptr;
    TORCH_CUDA_CHECK(cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocMapped));
    void* device = nullptr;
    TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&device, p, 0));
    if (device != p)
        throw Error(
            "The GPU maps pinned memory at a different address than the CPU; the engine needs unified "
            "addressing (a Jetson Orin with JetPack 7).");
    return static_cast<T*>(p);
}

}  // namespace

std::vector<uint64_t> ExpertStore::layer_expert_bytes(const model::Config& config, const gguf::File& shard1) {
    std::vector<uint64_t> bytes;
    for (int il = 0; il < config.n_layer; ++il) {
        uint64_t sum = 0;
        for (const char* kind : kKinds)
            sum += slot_bytes(slice_of(shard1, il, kind));
        bytes.push_back(sum);
    }
    return bytes;
}

uint64_t ExpertStore::fixed_bytes(const model::Config& config, const gguf::File& shard1) {
    uint64_t staging = 0, largest = 0;
    for (const char* kind : kKinds) {
        uint64_t most = 0;
        for (int il = 0; il < config.n_layer; ++il)
            most = std::max(most, slice_of(shard1, il, kind));
        staging += slot_bytes(most);
        largest = std::max(largest, most);
    }
    const uint64_t layers = static_cast<uint64_t>(config.n_layer);
    const uint64_t pointers =
        layers * (static_cast<uint64_t>(config.n_expert) + static_cast<uint64_t>(config.n_expert_used)) * 3;
    return static_cast<uint64_t>(config.n_expert_used + kSpareBuffers) * staging + pointers * sizeof(void*) + largest +
           page_ceil(sizeof(Mailbox));
}

ExpertStore::ExpertStore(const model::Config& config, const std::vector<model::Layer>& layers, const DirectFile& shard1,
                         const std::vector<std::vector<int32_t>>& resident)
    : layers_(layers), file_(shard1), slots_(config.n_expert_used), n_expert_(config.n_expert), pool_(kReadThreads) {
    if (static_cast<int>(resident.size()) != config.n_layer || slots_ > kGateSlots)
        throw Error("The expert placement does not match the model; this is a bug in the engine.");
    host_map_.assign(static_cast<size_t>(config.n_layer) * static_cast<size_t>(config.n_expert) * 3, nullptr);
    map_ = device_alloc<const uint8_t*>(host_map_.size());
    load(resident);

    uint64_t largest = 0, stride[3] = {};
    for (int kind = 0; kind < 3; ++kind) {
        uint64_t most = 0;
        for (const model::Layer& L : layers)
            most = std::max(most, tensor(L, kind).slice_bytes);
        stride[kind] = slot_bytes(most);
        largest = std::max(largest, most);
    }
    const int per_kind = slots_ + kSpareBuffers;
    staging_ = mapped_alloc<uint8_t>(static_cast<size_t>(per_kind) * (stride[0] + stride[1] + stride[2]));
    io_ = std::make_unique<HedgedReads>(file_, kReadThreads);
    uint8_t* at = staging_;
    for (int kind = 0; kind < 3; ++kind) {
        io_->add_pool(at, stride[kind], per_kind);  // pool id = kind
        at += static_cast<uint64_t>(per_kind) * stride[kind];
    }
    tables_ = device_alloc<const uint8_t*>(static_cast<size_t>(config.n_layer) * 3 * static_cast<size_t>(slots_));
    seq_ = device_alloc<uint32_t>(1);
    counters_ = device_alloc<unsigned long long>(2);
    n_route_counts_ = static_cast<size_t>(config.n_layer) * static_cast<size_t>(config.n_expert);
    route_counts_ = device_alloc<unsigned long long>(n_route_counts_);
    zeros_ = device_alloc<uint8_t>(largest);
    mailbox_ = mapped_alloc<Mailbox>(1);
    *mailbox_ = Mailbox{};
    thread_ = std::thread([this] { serve(); });
}

void ExpertStore::add(const std::vector<std::vector<int32_t>>& more) {
    if (more.size() != layers_.size())
        throw Error("The expert placement does not match the model; this is a bug in the engine.");
    load(more);
}

void ExpertStore::load(const std::vector<std::vector<int32_t>>& resident) {
    // One allocation for all of them: their slots layer after layer, gate, up and down.
    struct Read {
        int layer, expert, kind;
        uint64_t offset;  // in the arena
    };
    std::vector<Read> reads;
    uint64_t arena_bytes = 0;
    for (size_t il = 0; il < resident.size(); ++il) {
        for (const int32_t e : resident[il]) {
            if (e < 0 || e >= n_expert_)
                throw Error("The expert placement names expert " + std::to_string(e) +
                            ", which the model does not have.");
            if (host_map_[(il * static_cast<size_t>(n_expert_) + static_cast<size_t>(e)) * 3] != nullptr)
                throw Error("Expert " + std::to_string(e) + " of layer " + std::to_string(il) +
                            " is already resident; this is a bug in the engine.");
            for (int kind = 0; kind < 3; ++kind) {
                reads.push_back({static_cast<int>(il), e, kind, arena_bytes});
                arena_bytes += slot_bytes(tensor(layers_[il], kind).slice_bytes);
            }
        }
    }
    if (reads.empty())
        return;
    uint8_t* arena = mapped_alloc<uint8_t>(arena_bytes);
    arenas_.push_back(arena);
    pool_.run(static_cast<int>(reads.size()), [&](int i) {
        const Read& r = reads[static_cast<size_t>(i)];
        const model::ExpertTensor& t = tensor(layers_[static_cast<size_t>(r.layer)], r.kind);
        host_map_[(static_cast<size_t>(r.layer) * static_cast<size_t>(n_expert_) + static_cast<size_t>(r.expert)) * 3 +
                  static_cast<size_t>(r.kind)] =
            file_.read(t.file_offset + static_cast<uint64_t>(r.expert) * t.slice_bytes, t.slice_bytes,
                       arena + r.offset);
    });
    TORCH_CUDA_CHECK(cudaMemcpy(map_, host_map_.data(), host_map_.size() * sizeof(void*), cudaMemcpyHostToDevice));
}

ExpertStore::~ExpertStore() {
    stop_ = true;
    if (thread_.joinable())
        thread_.join();
    io_.reset();  // waits for reads still in flight before their buffers are freed
    cudaFreeHost(mailbox_);
    cudaFree(zeros_);
    cudaFree(route_counts_);
    cudaFree(counters_);
    cudaFree(seq_);
    cudaFree(tables_);
    cudaFree(map_);
    if (staging_)
        cudaFreeHost(staging_);
    for (uint8_t* arena : arenas_)
        cudaFreeHost(arena);
}

void ExpertStore::serve() {
    std::atomic_ref<uint32_t> request(mailbox_->request_seq);
    std::atomic_ref<uint32_t> done(mailbox_->done_seq);
    uint32_t served = 0;
    auto idle_since = Clock::now();
    while (!stop_) {
        const uint32_t seq = request.load(std::memory_order_acquire);
        if (seq == served) {
            // Poll hard while decode is running; back off when nothing has arrived for a while.
            if (Clock::now() - idle_since > std::chrono::milliseconds(2))
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            else
                asm volatile("yield");
            continue;
        }
        const auto start = Clock::now();
        const model::Layer& L = layers_[mailbox_->layer];
        const int count = static_cast<int>(std::min<uint32_t>(mailbox_->count, kGateSlots));
        // The GPU has finished the previous request's experts: their buffers are free again.
        io_->release();
        HedgedReads::Request reads[kGateSlots * 3];
        for (int i = 0; i < count; ++i) {
            for (int kind = 0; kind < 3; ++kind) {
                const model::ExpertTensor& t = tensor(L, kind);
                reads[i * 3 + kind] = {kind, t.file_offset + static_cast<uint64_t>(mailbox_->expert[i]) * t.slice_bytes,
                                       t.slice_bytes};
            }
        }
        const uint8_t* data[kGateSlots * 3];
        io_->read_all(std::span(reads, static_cast<size_t>(3 * count)),
                      std::span(data, static_cast<size_t>(3 * count)));
        for (int i = 0; i < 3 * count; ++i) {
            failed_ = failed_ || data[i] == nullptr;
            mailbox_->data[i / 3][i % 3] = data[i] != nullptr ? data[i] : zeros_;
        }
        read_ns_ +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        done.store(seq, std::memory_order_release);
        served = seq;
        idle_since = Clock::now();
    }
}

std::pair<uint64_t, uint64_t> ExpertStore::hits_and_misses() const {
    unsigned long long c[2] = {};
    TORCH_CUDA_CHECK(cudaMemcpy(c, counters_, sizeof c, cudaMemcpyDeviceToHost));
    return {c[0], c[1]};
}

std::vector<uint64_t> ExpertStore::read_route_counts() const {
    std::vector<uint64_t> out(n_route_counts_);
    TORCH_CUDA_CHECK(cudaMemcpy(out.data(), route_counts_, out.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return out;
}

void ExpertStore::check() const {
    if (failed_)
        throw Error("Reading a non-resident expert from the SSD failed; check the disk.");
    if (mailbox_->error)
        throw Error("A non-resident expert did not arrive within 2 s; the I/O thread stalled.");
}

}  // namespace torch::runtime
