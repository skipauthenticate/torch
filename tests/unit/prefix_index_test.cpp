#include "core/prefix_index.h"

#include <array>
#include <cstdio>
#include <vector>

int main() {
    using namespace torch;
    PrefixIndex i;
    const std::vector<int32_t> a{1, 2, 3, 4, 5, 6}, b{1, 2, 9};
    const auto ids = std::span(a);
    const auto block = i.insert(ids.first(2), PrefixKind::block, 0, 0);
    const auto one = i.insert(ids.first(3), PrefixKind::turn, block, 0);
    const auto two = i.insert(ids.first(4), PrefixKind::turn, block, one);
    const auto three = i.insert(ids.first(5), PrefixKind::turn, block, two);
    const auto check = [&](bool ok) {
        if (!ok)
            std::fputs("prefix index mismatch\n", stderr);
        return ok;
    };
    bool ok =
        check(i.match(a, 5) == three && i.match(a, 4) == two && i.match(b, 2) == block && i.match(a, 5, true) == block);
    ok &= check(i.old_turn(two) == one && i.victim([](auto) { return true; }) == one);
    i.touch(one);
    ok &= check(i.victim([](auto) { return true; }) == two);
    i.insert(b, PrefixKind::turn, block, one);
    ok &= check(i.old_turn(two) == 0);
    const auto fork = i.match(b, 3);
    ok &= check(i.to_park(ids.first(3)) == std::vector<PrefixKey>{two, three, fork});
    i.entries.at(two).live = false;
    ok &= check(i.to_park(ids.first(4)) == std::vector<PrefixKey>{three, fork});
    ok &= check(i.to_park(std::span(b)) == std::vector<PrefixKey>{one, three});
    ok &= check(i.to_park({}) == std::vector<PrefixKey>{block, one, three, fork});
    const auto tied = i.insert(ids.first(2), PrefixKind::turn, block, 0);
    ok &= check(i.match(a, 2) == tied && i.match(a, 1) == 0);
    return ok ? 0 : 1;
}
