#include "runtime/expert_gate.h"

namespace torch::runtime {
namespace {

constexpr unsigned long long kTimeoutNs = 2'000'000'000ull;
constexpr int kWarpSize = 32;

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One warp. It numbers the window's routes' unique experts in order of first appearance (a short serial pass, at most
// 80 routes), looks each unique expert up in the map, posts all misses in one request, and writes the tables.
__global__ void expert_gate_kernel(const int32_t* ids, int k, int columns, int layer, const uint8_t* const* map,
                                   int n_expert, GateTables out, Mailbox* mailbox, uint32_t* seq,
                                   unsigned long long* counters, unsigned long long* route_counts, const uint8_t* zeros,
                                   const int32_t* pos, uint32_t* last_use) {
    __shared__ int32_t unique[kGateSlots];
    __shared__ uint8_t slot_of[kGateSlots];  // the unique index of route i
    __shared__ bool absent[kGateSlots];      // unique expert u is not resident
    const int lane = threadIdx.x;
    const int routes = columns * k;
    for (int i = lane; i < kGateSlots * kGateColumns; i += kWarpSize)
        out.rank[i] = -1;
    for (int i = lane; i < kGateSlots; i += kWarpSize)
        out.columns[i] = 0;
    __syncwarp();

    int n = 0;  // the same in every lane
    for (int i = 0; i < routes; ++i) {
        const int32_t e = ids[i];
        int found = -1;
        for (int base = 0; base < n; base += kWarpSize) {
            const unsigned hit = __ballot_sync(0xffffffffu, base + lane < n && unique[base + lane] == e);
            if (hit != 0) {
                found = base + __ffs(hit) - 1;
                break;
            }
        }
        if (found < 0) {
            if (lane == 0)
                unique[n] = e;
            found = n++;
        }
        if (lane == 0) {
            slot_of[i] = static_cast<uint8_t>(found);
            out.rank[found * kGateColumns + i / k] = static_cast<int8_t>(i % k);
            out.columns[found] |= static_cast<uint8_t>(1u << (i / k));
        }
        __syncwarp();
    }
    const uint32_t stamp = static_cast<uint32_t>(*pos) + 1u;
    for (int i = lane; i < routes; i += kWarpSize) {
        atomicAdd(route_counts + static_cast<int64_t>(layer) * n_expert + ids[i], 1ull);
        last_use[static_cast<int64_t>(layer) * n_expert + ids[i]] = stamp;  // lanes writing one entry write one value
    }

    // Look up every unique expert; lane l takes experts l, l + 32 and l + 64.
    volatile Mailbox* box = mailbox;
    int misses = 0;  // the same in every lane
    for (int round = 0; round < 3; ++round) {
        const int u = round * kWarpSize + lane;
        bool miss = false;
        if (u < n) {
            const uint8_t* const* entry = map + (static_cast<int64_t>(layer) * n_expert + unique[u]) * 3;
            miss = entry[0] == nullptr;
            absent[u] = miss;
            if (!miss)
                for (int kind = 0; kind < 3; ++kind)
                    out.table[kind * kGateSlots + u] = entry[kind];
        }
        const unsigned m = __ballot_sync(0xffffffffu, miss);
        if (miss) {  // packed by order of unique index
            const int at = misses + __popc(m & ((1u << lane) - 1));
            box->slot[at] = static_cast<uint32_t>(u);
            box->expert[at] = unique[u];
        }
        misses += __popc(m);
    }
    __syncwarp();
    // Hits and misses per route: a unique expert counts once for every column that routed it.
    int miss_routes = 0;
    for (int base = 0; base < routes; base += kWarpSize)
        miss_routes += __popc(__ballot_sync(0xffffffffu, base + lane < routes && absent[slot_of[base + lane]]));
    if (lane == 0) {
        counters[0] += static_cast<unsigned long long>(routes - miss_routes);
        counters[1] += static_cast<unsigned long long>(miss_routes);
        *out.count = n;
    }

    bool ok = true;
    if (misses > 0) {
        __syncwarp();
        if (lane == 0) {
            const uint32_t request = ++*seq;
            box->layer = static_cast<uint32_t>(layer);
            box->count = static_cast<uint32_t>(misses);
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
        for (int i = lane; i < misses; i += kWarpSize) {
            const int u = static_cast<int>(box->slot[i]);
            for (int kind = 0; kind < 3; ++kind)
                out.table[kind * kGateSlots + u] = ok ? box->data[i][kind] : zeros;
        }
    }
    __syncwarp();
    for (int i = lane; i < routes; i += kWarpSize) {
        out.gate_routes[i] = out.table[slot_of[i]];
        out.up_routes[i] = out.table[kGateSlots + slot_of[i]];
        out.down_routes[i] = out.table[2 * kGateSlots + slot_of[i]];
    }
}

// Bytes before the first 4-byte boundary and after the last one one by one, the 4-byte words between by every thread.
__global__ void copy_bytes_kernel(uint8_t* dst, const uint8_t* src, uint64_t n) {
    const uint64_t head =
        (4 - reinterpret_cast<uintptr_t>(src) % 4) % 4 < n ? (4 - reinterpret_cast<uintptr_t>(src) % 4) % 4 : n;
    const uint64_t words = (n - head) / 4, tail = head + words * 4;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t t = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < head)
        dst[t] = src[t];
    if (t < n - tail)
        dst[tail + t] = src[tail + t];
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + head);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst + head);
    for (uint64_t i = t; i < words; i += stride)
        d[i] = s[i];
}

}  // namespace

void copy_bytes(uint8_t* dst, const uint8_t* src, uint64_t n, cudaStream_t stream) {
    if (n > 0)
        copy_bytes_kernel<<<128, 256, 0, stream>>>(dst, src, n);
}

void expert_gate(const int32_t* ids, int k, int columns, int layer, const uint8_t* const* map, int n_expert,
                 const GateTables& out, Mailbox* mailbox, uint32_t* seq, unsigned long long* counters,
                 unsigned long long* route_counts, const uint8_t* zeros, const int32_t* pos, uint32_t* last_use,
                 cudaStream_t stream) {
    expert_gate_kernel<<<1, kWarpSize, 0, stream>>>(ids, k, columns, layer, map, n_expert, out, mailbox, seq, counters,
                                                    route_counts, zeros, pos, last_use);
}

}  // namespace torch::runtime
