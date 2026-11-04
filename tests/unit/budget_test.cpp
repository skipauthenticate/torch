// Checks the memory budget: how many experts fit, the refusals, and reading MemAvailable.
#include "core/budget.h"

#include <string>

#include "check.h"
#include "core/error.h"

using namespace torch;

namespace {

bool throws(auto&& f) {
    try {
        f();
        return false;
    } catch (const Error&) {
        return true;
    }
}

void check_plan() {
    const std::vector<Reservation> r = {{"weights", 4 * kGiB}, {"state", kGiB}};
    // 30 GiB available, 6 kept free, 5 reserved: 19 GiB for experts of 100 MiB each (one per layer).
    const Budget b = plan_budget(30 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, std::nullopt);
    CHECK(b.resident == 194);
    CHECK(b.reserved() == 5 * kGiB);
    CHECK(b.left() == 19 * kGiB - 194 * 100 * kMiB);
    CHECK(b.table().find("resident experts: 194 of 512 per layer") != std::string::npos);

    // Never more than the layer has.
    CHECK(plan_budget(300 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, std::nullopt).resident == 512);
    // A forced count that fits, and ones that do not.
    CHECK(plan_budget(30 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, 0).resident == 0);
    CHECK(plan_budget(30 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, 194).resident == 194);
    CHECK(throws([&] { plan_budget(30 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, 195); }));
    CHECK(throws([&] { plan_budget(30 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, 513); }));
    // Below the floor before any expert.
    CHECK(throws([&] { plan_budget(10 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, std::nullopt); }));
    CHECK(plan_budget(11 * kGiB, 6 * kGiB, r, 100 * kMiB, 512, std::nullopt).resident == 0);
}

void check_meminfo() {
    CHECK(parse_mem_available("MemTotal:       64328812 kB\nMemFree:  1 kB\nMemAvailable:   29123456 kB\n") ==
          uint64_t{29123456} * 1024);
    CHECK(throws([] { parse_mem_available("MemTotal: 1 kB\n"); }));
    CHECK(throws([] { parse_mem_available("MemAvailable: lots\n"); }));
    CHECK(throws([] { parse_mem_available("MemAvailable: 12 MB\n"); }));
}

}  // namespace

int main() {
    check_plan();
    check_meminfo();
    return test::test_exit_code();
}
