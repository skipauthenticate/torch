#include "core/swaps.h"

#include <algorithm>
#include <tuple>

namespace torch {

int lru_victim(const Residency& r, int layer) {
    int best = -1;
    for (int e = 0; e < r.n_expert; ++e) {
        const size_t i = r.at(layer, e);
        if (!r.resident[i])
            continue;
        const size_t b = best < 0 ? 0 : r.at(layer, best);
        if (best < 0 || std::tie(r.last_use[i], r.calibration[i]) <= std::tie(r.last_use[b], r.calibration[b]))
            best = e;  // <= so that among equals the higher id wins
    }
    return best;
}

std::vector<Swap> online_swaps(Residency& r, const std::vector<std::pair<int, int>>& missed, uint32_t now, int cap) {
    std::vector<Swap> out;
    for (const auto& [layer, expert] : missed) {
        if (static_cast<int>(out.size()) >= cap)
            break;
        const size_t in = r.at(layer, expert);
        if (r.resident[in])
            continue;
        const int victim = lru_victim(r, layer);
        if (victim < 0 || r.last_use[r.at(layer, victim)] >= now)
            continue;
        r.resident[r.at(layer, victim)] = 0;
        r.resident[in] = 1;
        out.push_back({layer, expert, victim});
    }
    return out;
}

}  // namespace torch
