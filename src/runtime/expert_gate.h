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

inline constexpr int kGateSlots = 16;  // routed experts per layer the mailbox can hold (the model routes 10)

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

// Writes the pointer table of `layer` (table[kind * slots + s] for kind gate, up, down) for the experts in ids[0, k).
// map holds n_layer x n_expert x 3 device addresses (null when absent); seq is a device counter the kernel advances
// once per request; counters[0] and [1] count hits and misses, and route_counts[layer * n_expert + e] how often
// expert e was routed (the input of a ranked placement, core/placement.h). k <= kGateSlots.
void expert_gate(const int32_t* ids, int k, int layer, const uint8_t* const* map, int n_expert, const uint8_t** table,
                 int slots, Mailbox* mailbox, uint32_t* seq, unsigned long long* counters,
                 unsigned long long* route_counts, const uint8_t* zeros, cudaStream_t stream);

}  // namespace torch::runtime
