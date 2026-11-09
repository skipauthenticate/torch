// The GPU side of expert placement (architecture decision 6): after a layer's router, one small kernel writes the
// layer's expert pointer table, so the expert kernels can run without the host.
//
// A device map gives every (layer, expert) the addresses of its gate, up and down matrices when the expert is
// resident, or null. The gate kernel copies the routed experts' addresses into the layer's table. When any routed
// expert is absent it posts the layer's misses to a mailbox in mapped pinned memory and waits, spinning with
// __nanosleep, until the host's I/O thread has read them into staging slots and published the request's sequence
// number; it then points the table at the staged bytes. The expert kernels read the same bytes through the same table
// either way, so output never depends on residency. If the host has not answered after 2 s the kernel sets an error
// flag and points the table at a zero buffer; the host reports the error after the token.
//
// Measured on the Orin (bench/miss_roundtrip.cu): request to acknowledge 7.6 us without a read; the staged bytes are
// what the GPU reads.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::runtime {

// Unique experts per layer a window can route: 8 columns x 10 routes (the mailbox, the staging and the table hold
// this many).
inline constexpr int kGateSlots = 80;
inline constexpr int kGateColumns = 8;

// Shared by the gate kernel and the host I/O thread, in mapped pinned memory. Sequence numbers only grow, so a CUDA
// graph can replay the kernel without anything being reset.
struct Mailbox {
    // Written by the GPU, request_seq last.
    uint32_t request_seq;
    uint32_t layer;
    uint32_t count;
    uint32_t slot[kGateSlots];  // the routed slot of each miss
    int32_t expert[kGateSlots];
    // Written by the host, done_seq last.
    uint32_t done_seq;
    const uint8_t* data[kGateSlots][3];  // where the staged gate, up and down bytes of each miss start
    // Written by the GPU when a request times out.
    uint32_t error;
};

// The window's experts of one layer, as the gate kernel writes them, in device memory. The routes of column t are
// ids[t * k + r] (rank r = 0..k-1). Unique experts are numbered in order of first appearance over (t, then r), so a
// one-column window numbers them by rank, exactly as one token's routes.
struct GateTables {
    const uint8_t**
        table;         // [3][kGateSlots]: gate, up and down matrix of unique expert u at table[kind * kGateSlots + u]
    uint8_t* columns;  // [kGateSlots]: bit t set when column t routed unique expert u
    int8_t* rank;      // [kGateSlots][kGateColumns]: the rank column t gave u, or -1
    int32_t* count;    // unique experts in the window
    const uint8_t** down_routes;  // [kGateColumns * k]: column t's rank-r down matrix at t * k + r
    const uint8_t** gate_routes;  // [kGateColumns * k]: likewise its gate matrix
    const uint8_t** up_routes;    // [kGateColumns * k]: and its up matrix
};

// Places a window's routed experts (columns x k routes, columns <= kGateColumns, columns * k <= kGateSlots) for
// `layer`. map holds n_layer x n_expert x 3 device addresses (null when absent); seq is a device counter the kernel
// advances once per request; counters[0] and [1] count hits and misses per route, and route_counts[layer * n_expert +
// e] how often expert e was routed (the input of a ranked placement, core/placement.h); last_use[layer * n_expert + e]
// becomes *pos + 1 for every routed e, pos being the window's first position (the recency swaps go by, core/swaps.h).
void expert_gate(const int32_t* ids, int k, int columns, int layer, const uint8_t* const* map, int n_expert,
                 const GateTables& out, Mailbox* mailbox, uint32_t* seq, unsigned long long* counters,
                 unsigned long long* route_counts, const uint8_t* zeros, const int32_t* pos, uint32_t* last_use,
                 cudaStream_t stream);

// Copies n bytes from src to dst on the GPU (both device-readable; mapped pinned memory included, which a
// cudaMemcpyAsync would copy on the CPU, synchronously), as one launch on `stream`. src and dst must have the same
// address modulo 4, as an expert and the slot it moves into do.
void copy_bytes(uint8_t* dst, const uint8_t* src, uint64_t n, cudaStream_t stream);

}  // namespace torch::runtime
