// Which routed experts are resident (architecture decision 8: "placement is a ranking").
//
// The experts do not all fit next to the operating system (46.8 GiB), and the model routes some far more often than
// others. A route-count file records how often each (layer, expert) was routed while decoding calibration text; the
// placement keeps the most-routed experts resident, across all layers, until the memory the budget leaves for experts
// is full. Without counts, every layer keeps its experts 0 to k - 1. Output never depends on the placement (the expert
// store computes every expert with the same kernel), only speed.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace torch {

// Routing counts, layer after layer: count(layer, expert) = counts[layer * n_expert + expert].
struct RouteCounts {
    int n_layer = 0, n_expert = 0;
    std::vector<uint64_t> counts;

    uint64_t at(int layer, int expert) const { return counts[static_cast<size_t>(layer * n_expert + expert)]; }

    // A text file: the line "torch route counts", a line with n_layer and n_expert, then one line of n_expert counts
    // per layer. Throws Error when the file is missing or malformed.
    static RouteCounts read(const std::filesystem::path& path);
    void write(const std::filesystem::path& path) const;
    // Adds another set of counts of the same shape. Throws Error when the shapes differ.
    void add(const RouteCounts& other);
};

struct Placement {
    std::vector<std::vector<int32_t>> resident;  // per layer, ascending expert ids
    uint64_t bytes = 0;                          // their memory, by expert_bytes
    int count() const;
};

// expert_bytes[layer] is the memory one expert of that layer takes (its three matrices).

// k experts in every layer: 0 to k - 1.
Placement place_uniform(int k, int n_expert, const std::vector<uint64_t>& expert_bytes);

// The largest k with every layer's k experts within `room` bytes.
int uniform_capacity(uint64_t room, int n_expert, const std::vector<uint64_t>& expert_bytes);

// The most-routed experts within `room` bytes, in order of count (ties: lower layer, then lower expert id); an expert
// that does not fit is skipped and smaller ones are still tried.
Placement place_ranked(const RouteCounts& counts, const std::vector<uint64_t>& expert_bytes, uint64_t room);

// The next experts in the same order that are not in `have`, within `room` more bytes: what to add once the memory
// actually left after loading is known.
Placement extend_ranked(const RouteCounts& counts, const std::vector<uint64_t>& expert_bytes, const Placement& have,
                        uint64_t room);

// Counts that rank in place_uniform's order: expert 0 of every layer, then expert 1, and so on.
RouteCounts uniform_ranking(int n_layer, int n_expert);

}  // namespace torch
