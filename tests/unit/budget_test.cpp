// Checks the memory budget: the room it leaves for experts, the refusal below the floor, the table, and reading
// MemAvailable.
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
    // 30 GiB available, 6 kept free, 5 reserved: 19 GiB of room for experts.
    Budget b = plan_budget(30 * kGiB, 6 * kGiB, r);
    CHECK(b.reserved() == 5 * kGiB);
    CHECK(b.room() == 19 * kGiB);
    CHECK(b.placement_room() == 19 * kGiB - kTopUpMargin);
    b.resident = 100, b.experts = 400, b.resident_bytes = 10 * kGiB, b.placement = "test";
    CHECK(b.left() == 9 * kGiB);
    CHECK(b.table().find("resident experts: 100 of 400 (25.0%)") != std::string::npos);
    CHECK(b.table().find("after loading") == std::string::npos);
    b.available_after_load = 12 * kGiB, b.added = 20, b.added_bytes = 2 * kGiB;
    CHECK(b.table().find("floor + 1.5 GiB: 20 (30.0% resident)") != std::string::npos);
    // Below the floor before any expert; exactly at it leaves no room.
    CHECK(throws([&] { plan_budget(10 * kGiB, 6 * kGiB, r); }));
    CHECK(plan_budget(11 * kGiB, 6 * kGiB, r).room() == 0);
    CHECK(plan_budget(11 * kGiB, 6 * kGiB, r).placement_room() == 0);
    CHECK(plan_budget(12 * kGiB, 6 * kGiB, r).placement_room() == 0);  // room under the margin
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
