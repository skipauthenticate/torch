// Which resident expert makes room for another (architecture decision 8: residency follows use). The placement at
// start is a ranking of calibration counts; while a request runs, the experts that just missed in a window take the
// slots of resident experts of the same layer that have gone longest without a route, so memory stays constant. These
// are pure functions over the residency state; runtime::ExpertStore moves the bytes. Output never depends on them.
#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace torch {

struct Swap {
    int layer, in, out;  // `in` takes the slots of resident expert `out`
};

// Per (layer, expert), flat at layer * n_expert + expert: whether it is resident, the stamp of the last window that
// routed it (the gate kernel writes the window's first position + 1; 0 for none yet) and its calibration count.
struct Residency {
    int n_layer = 0, n_expert = 0;
    std::vector<uint8_t> resident;
    std::span<const uint32_t> last_use;
    std::vector<uint64_t> calibration;

    size_t at(int layer, int expert) const noexcept {
        return static_cast<size_t>(layer) * static_cast<size_t>(n_expert) + static_cast<size_t>(expert);
    }
};

// The resident expert of `layer` with the oldest stamp; ties go to the lower calibration count, then to the higher id.
// -1 when the layer has no resident expert.
int lru_victim(const Residency& r, int layer);

// For experts (layer, expert) that missed in the window stamped `now`, in order: each one not resident yet takes the
// slots of its layer's lru_victim, unless that victim was routed in this window too. At most cap swaps. Updates
// r.resident.
std::vector<Swap> online_swaps(Residency& r, const std::vector<std::pair<int, int>>& missed, uint32_t now, int cap);

}  // namespace torch
