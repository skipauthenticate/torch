// Parity of the GPU argmax with a scalar scan: random logits, ties (the lowest index must win), the largest value at
// the first and the last index, and a NaN that must never win.
#include "kernels/sampling/argmax.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "../unit/check.h"
#include "device.h"

using namespace torch;
using kernels::ArgmaxResult;
using test::DeviceBuffer;

namespace {

int want(const std::vector<float>& x) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(x.size()); ++i)
        if (x[i] > x[best])
            best = i;
    return best;
}

int got(const std::vector<float>& x) {
    DeviceBuffer<float> dx(x);
    DeviceBuffer<ArgmaxResult> scratch(kernels::kArgmaxBlocks), out(1);
    kernels::argmax(dx.get(), static_cast<int>(x.size()), scratch.get(), out.get(), nullptr);
    return out.download()[0].index;
}

}  // namespace

int main() {
    try {
        const int n = 248320;
        std::vector<float> x = test::random_floats(n, -20, 20, 1);
        CHECK(got(x) == want(x));
        x[100] = x[200000] = 50.0f;  // a tie: the lower index wins
        CHECK(got(x) == 100);
        x[0] = 60.0f;
        CHECK(got(x) == 0);
        x[n - 1] = 70.0f;
        CHECK(got(x) == n - 1);
        x[n - 1] = NAN;  // a NaN never wins
        CHECK(got(x) == 0);
        std::vector<float> same(n, 1.0f);  // every logit equal: index 0
        CHECK(got(same) == 0);
        std::printf("argmax: random, ties, ends and NaN %s\n", test::failures == 0 ? "ok" : "FAIL");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "argmax_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
