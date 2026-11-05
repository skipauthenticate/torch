// Can the GPU ask the host for a missing expert and wait for it inside a kernel, instead of the host synchronizing
// after every layer's router? This probe measures the design that M3's one-graph-per-token decode depends on:
//
//   GPU: a block posts a request (file offset, sequence number) to mapped memory, fences, and spins on a mapped
//        completion word with __nanosleep until the host acknowledges, with a 2 s timeout that sets an error flag;
//   host: a thread polls the requests, reads the expert matrix from the model file with O_DIRECT into a mapped
//        staging slot (128 KiB preads), stores the slice's checksum, then publishes the sequence number.
//
// It reports the request-to-acknowledge time seen by the GPU (globaltimer) with no read, with 1 read and with 10
// reads at once; checks that every byte the DMA wrote is what the GPU reads (checksum per slot); checks that the
// timeout fires when the host does not answer; and, for comparison, times the alternative of a host round trip per
// layer (launch, cudaStreamSynchronize, launch).
//
// usage: torch_bench_miss_roundtrip MODEL-00001-of-00002.gguf [rounds, default 300]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "runtime/direct_io.h"

namespace {

constexpr int kMaxSlots = 10;
constexpr uint64_t kSliceBytes = 921600;  // one IQ4_NL down projection, 2560 x 640
constexpr uint64_t kSlotBytes = (kSliceBytes / 4096 + 2) * 4096;
constexpr uint64_t kTimeoutNs = 2'000'000'000;

// Shared between the GPU and the host, in mapped pinned memory. Sequence numbers only grow, so nothing is reset.
struct Mailbox {
    uint32_t request_seq[kMaxSlots];
    uint32_t done_seq[kMaxSlots];
    uint64_t offset[kMaxSlots];       // file offset of the slice, written by the GPU before request_seq
    uint32_t checksum[kMaxSlots];     // XOR of the slice's 32-bit words, written by the host before done_seq
    uint32_t data_offset[kMaxSlots];  // where the slice starts inside its staging slot
    uint32_t error;                   // set by the GPU on a timeout
    uint32_t mismatches;              // slots whose bytes did not match the host's checksum
};

__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One block per slot. Thread 0 posts and waits; then the block checks the staged bytes.
__global__ void request_and_wait(Mailbox* box, const uint8_t* staging, const uint64_t* offsets, uint32_t seq,
                                 bool check, uint64_t* wait_ns) {
    __shared__ bool ok;
    const int slot = blockIdx.x;
    if (threadIdx.x == 0) {
        volatile Mailbox* v = box;
        v->offset[slot] = offsets[slot];
        __threadfence_system();
        const uint64_t start = global_ns();
        v->request_seq[slot] = seq;
        __threadfence_system();
        ok = true;
        while (v->done_seq[slot] != seq) {
            if (global_ns() - start > kTimeoutNs) {
                v->error = 1;
                ok = false;
                break;
            }
            __nanosleep(500);
        }
        __threadfence_system();
        wait_ns[slot] = global_ns() - start;
    }
    __syncthreads();
    if (!ok || !check)
        return;
    const auto* words = reinterpret_cast<const uint32_t*>(staging + slot * kSlotBytes + box->data_offset[slot]);
    uint32_t x = 0;
    for (uint64_t i = threadIdx.x; i < kSliceBytes / 4; i += blockDim.x)
        x ^= words[i];
    __shared__ uint32_t partial[1024];
    partial[threadIdx.x] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t all = 0;
        for (unsigned t = 0; t < blockDim.x; ++t)
            all ^= partial[t];
        if (all != box->checksum[slot])
            atomicAdd(&box->mismatches, 1u);
    }
}

__global__ void touch(uint32_t* p) {
    *p += 1;
}

// The host side: one thread per slot polls its request word and serves it.
class Server {
public:
    Server(Mailbox* box, uint8_t* staging, const torch::runtime::DirectFile& file, int slots, bool read)
        : box_(box), staging_(staging), file_(file) {
        for (int s = 0; s < slots; ++s)
            threads_.emplace_back([this, s, read] { serve(s, read); });
    }
    ~Server() {
        stop_ = true;
        for (std::thread& t : threads_)
            t.join();
    }
    void pause(bool p) { paused_ = p; }

private:
    void serve(int slot, bool read) {
        std::atomic_ref<uint32_t> request(box_->request_seq[slot]);
        std::atomic_ref<uint32_t> done(box_->done_seq[slot]);
        uint32_t served = done.load();
        while (!stop_) {
            const uint32_t seq = request.load(std::memory_order_acquire);
            if (seq == served || paused_) {
                asm volatile("yield");
                continue;
            }
            if (read) {
                uint8_t* slot_base = staging_ + slot * kSlotBytes;
                const uint8_t* data = file_.read(box_->offset[slot], kSliceBytes, slot_base);
                box_->data_offset[slot] = static_cast<uint32_t>(data - slot_base);
                uint32_t x = 0;
                for (uint64_t i = 0; i < kSliceBytes / 4; ++i)
                    x ^= reinterpret_cast<const uint32_t*>(data)[i];
                box_->checksum[slot] = x;
            }
            done.store(seq, std::memory_order_release);
            served = seq;
        }
    }

    Mailbox* box_;
    uint8_t* staging_;
    const torch::runtime::DirectFile& file_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false}, paused_{false};
};

struct Stats {
    double median_us, p99_us, worst_us;
};

Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v[v.size() * 99 / 100], v.back()};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: torch_bench_miss_roundtrip MODEL-00001-of-00002.gguf [rounds, default 300]\n");
        return 2;
    }
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 300;
    try {
        const torch::runtime::DirectFile file(argv[1]);
        void* raw = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&raw, sizeof(Mailbox), cudaHostAllocMapped));
        auto* box = static_cast<Mailbox*>(raw);
        *box = Mailbox{};
        TORCH_CUDA_CHECK(cudaHostAlloc(&raw, kMaxSlots * kSlotBytes, cudaHostAllocMapped));
        auto* staging = static_cast<uint8_t*>(raw);
        uint64_t* offsets = nullptr;
        uint64_t* wait_ns = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&raw, kMaxSlots * sizeof(uint64_t), cudaHostAllocMapped));
        offsets = static_cast<uint64_t*>(raw);
        TORCH_CUDA_CHECK(cudaHostAlloc(&raw, kMaxSlots * sizeof(uint64_t), cudaHostAllocMapped));
        wait_ns = static_cast<uint64_t*>(raw);
        cudaStream_t stream;
        TORCH_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::mt19937_64 rng(7);
        uint32_t seq = 0;
        std::printf("GPU request -> host acknowledge, %d rounds each (microseconds, as the GPU measured them)\n",
                    rounds);
        std::printf("  case                         median      p99    worst   checksum mismatches\n");
        struct Case {
            const char* name;
            int slots;
            bool read;
        };
        for (const Case c : {Case{"no read, 1 slot", 1, false}, Case{"1 read of 0.9 MiB", 1, true},
                             Case{"10 reads of 0.9 MiB at once", 10, true}}) {
            Server server(box, staging, file, c.slots, c.read);
            std::vector<double> waits;
            box->mismatches = 0;
            for (int r = 0; r < rounds; ++r) {
                for (int s = 0; s < c.slots; ++s)
                    offsets[s] = rng() % (file.size() - kSlotBytes) / 32 * 32;  // tensors start 32-byte aligned
                ++seq;
                request_and_wait<<<c.slots, 256, 0, stream>>>(box, staging, offsets, seq, c.read, wait_ns);
                TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
                if (box->error)
                    throw torch::Error("A request timed out while the host was serving it.");
                waits.push_back(static_cast<double>(*std::max_element(wait_ns, wait_ns + c.slots)) / 1000.0);
            }
            const Stats st = summarize(waits);
            std::printf("  %-28s %8.1f %8.1f %8.1f   %u\n", c.name, st.median_us, st.p99_us, st.worst_us,
                        box->mismatches);
        }

        {  // the host never answers: the GPU must give up after the timeout and set the error flag
            Server server(box, staging, file, 1, false);
            server.pause(true);
            ++seq;
            const auto start = std::chrono::steady_clock::now();
            request_and_wait<<<1, 256, 0, stream>>>(box, staging, offsets, seq, false, wait_ns);
            TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("  timeout: the kernel returned after %.2f s with the error flag %s\n", s,
                        box->error ? "set" : "NOT set");
            box->error = 0;
            server.pause(false);
        }

        {  // the alternative: one host round trip per layer
            uint32_t* counter = nullptr;
            TORCH_CUDA_CHECK(cudaMalloc(&counter, sizeof(uint32_t)));
            std::vector<double> trips;
            for (int r = 0; r < rounds; ++r) {
                const auto start = std::chrono::steady_clock::now();
                touch<<<1, 1, 0, stream>>>(counter);
                TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
                trips.push_back(
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
            }
            const Stats st = summarize(trips);
            std::printf("  host round trip (launch + cudaStreamSynchronize): median %.1f us, p99 %.1f, worst %.1f\n",
                        st.median_us, st.p99_us, st.worst_us);
            cudaFree(counter);
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "torch_bench_miss_roundtrip: %s\n", e.what());
        return 1;
    }
    return 0;
}
