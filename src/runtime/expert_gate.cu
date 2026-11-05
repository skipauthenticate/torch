#include "runtime/expert_gate.h"

namespace torch::runtime {
namespace {

constexpr unsigned long long kTimeoutNs = 2'000'000'000ull;

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One warp. Lane s handles routed slot s.
__global__ void expert_gate_kernel(const int32_t* ids, int k, int layer, const uint8_t* const* map, int n_expert,
                                   const uint8_t** table, int slots, Mailbox* mailbox, uint32_t* seq,
                                   unsigned long long* counters, unsigned long long* route_counts,
                                   const uint8_t* zeros) {
    const int s = threadIdx.x;
    const uint8_t* p[3] = {nullptr, nullptr, nullptr};
    int32_t e = -1;
    if (s < k) {
        e = ids[s];
        atomicAdd(route_counts + static_cast<int64_t>(layer) * n_expert + e, 1ull);
        const uint8_t* const* entry = map + (static_cast<int64_t>(layer) * n_expert + e) * 3;
        for (int kind = 0; kind < 3; ++kind)
            p[kind] = entry[kind];
    }
    const bool miss = s < k && p[0] == nullptr;
    const unsigned misses = __ballot_sync(0xffffffffu, miss);
    if (s < k && !miss) {
        for (int kind = 0; kind < 3; ++kind)
            table[kind * slots + s] = p[kind];
    }
    if (s == 0) {
        counters[0] += static_cast<unsigned long long>(k - __popc(misses));
        counters[1] += static_cast<unsigned long long>(__popc(misses));
    }
    if (misses == 0)
        return;

    volatile Mailbox* box = mailbox;
    if (miss) {  // each missing lane writes its own entry, packed by its rank among the misses
        const int i = __popc(misses & ((1u << s) - 1));
        box->slot[i] = static_cast<uint32_t>(s);
        box->expert[i] = e;
    }
    __syncwarp();
    uint32_t request = 0;
    bool ok = true;
    if (s == 0) {
        request = ++*seq;
        box->layer = static_cast<uint32_t>(layer);
        box->count = static_cast<uint32_t>(__popc(misses));
        __threadfence_system();
        box->request_seq = request;  // published last
        __threadfence_system();
        const unsigned long long start = global_ns();
        while (box->done_seq != request) {
            if (global_ns() - start > kTimeoutNs) {
                box->error = 1;
                ok = false;
                break;
            }
            __nanosleep(500);
        }
        __threadfence_system();
    }
    ok = __shfl_sync(0xffffffffu, ok ? 1 : 0, 0) != 0;
    if (miss) {
        const int i = __popc(misses & ((1u << s) - 1));
        for (int kind = 0; kind < 3; ++kind)
            table[kind * slots + s] = ok ? box->data[i][kind] : zeros;
    }
}

}  // namespace

void expert_gate(const int32_t* ids, int k, int layer, const uint8_t* const* map, int n_expert, const uint8_t** table,
                 int slots, Mailbox* mailbox, uint32_t* seq, unsigned long long* counters,
                 unsigned long long* route_counts, const uint8_t* zeros, cudaStream_t stream) {
    expert_gate_kernel<<<1, 32, 0, stream>>>(ids, k, layer, map, n_expert, table, slots, mailbox, seq, counters,
                                             route_counts, zeros);
}

}  // namespace torch::runtime
