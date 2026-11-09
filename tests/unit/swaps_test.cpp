// Checks the swap policy (core/swaps.h): the LRU victim and its ties, online swaps that skip residents and stop at
// the cap.
#include "core/swaps.h"

#include <vector>

#include "check.h"

using namespace torch;

namespace {

// 2 layers of 6 experts; experts 0-3 of each layer resident; stamps in `use`.
Residency make(const std::vector<uint32_t>& use) {
    Residency r;
    r.n_layer = 2, r.n_expert = 6;
    r.resident = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0};
    r.last_use = use;
    r.calibration = {50, 40, 30, 30, 5, 5, 9, 8, 7, 6, 1, 1};
    return r;
}

void check_victim() {
    std::vector<uint32_t> use(12, 0);
    Residency r = make(use);
    CHECK(lru_victim(r, 0) == 3);  // all unused: lowest calibration (30, experts 2 and 3), then the higher id
    use[3] = 5, use[2] = 4, use[1] = 4;
    CHECK(lru_victim(r, 0) == 0);  // 0 is the only one never used
    use[0] = 6;
    CHECK(lru_victim(r, 0) == 2);  // 1 and 2 last used in window 4: lower calibration
    r.resident = std::vector<uint8_t>(12, 0);
    CHECK(lru_victim(r, 1) == -1);
}

void check_online() {
    // window 7 routed experts 0-2, 4 and 5 of layer 0 and 1, 4 of layer 1 (the gate stamps routes, resident or not)
    const std::vector<uint32_t> use = {7, 7, 7, 2, 7, 7, 3, 7, 3, 3, 7, 0};
    Residency r = make(use);
    // window 7: expert 4 of layer 0 missed (victim 3), then 5 (the next LRU is used in window 7: no swap), a resident
    // expert (ignored), and expert 4 of layer 1 (victim: all equally old, lowest calibration 6 = expert 3)
    const std::vector<Swap> s = online_swaps(r, {{0, 4}, {0, 5}, {1, 1}, {1, 4}}, 7, 16);
    CHECK(s.size() == 2);
    CHECK(s[0].layer == 0 && s[0].in == 4 && s[0].out == 3);
    CHECK(s[1].layer == 1 && s[1].in == 4 && s[1].out == 3);
    CHECK(r.resident[r.at(0, 4)] && !r.resident[r.at(0, 3)]);
    Residency c = make(use);
    CHECK(online_swaps(c, {{0, 4}, {1, 4}}, 7, 1).size() == 1);  // the cap
}

}  // namespace

int main() {
    check_victim();
    check_online();
    return test::test_exit_code();
}
