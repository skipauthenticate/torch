// A minimal test harness. CHECK records a failure and carries on, so one run reports every broken check;
// test_exit_code() ends main().
#pragma once

#include <cstdio>

namespace torch::test {

inline int failures = 0;

inline int test_exit_code() {
    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    return 0;
}

}  // namespace torch::test

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition); \
            ++torch::test::failures;                                                           \
        }                                                                                      \
    } while (0)
