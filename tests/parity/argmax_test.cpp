// Parity of the GPU argmax with a scalar scan: random logits, ties (the lowest index must win), the largest value at
// the first and the last index, and a NaN that must never win. And argmax_prob against a double softmax.
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

// The softmax probability of the largest logit, against the double reference.
void check_prob(const std::vector<float>& x) {
    DeviceBuffer<float> dx(x), prob(1);
    DeviceBuffer<ArgmaxResult> scratch(kernels::kArgmaxBlocks), best(1);
    kernels::argmax(dx.get(), static_cast<int>(x.size()), scratch.get(), best.get(), nullptr);
    kernels::argmax_prob(dx.get(), static_cast<int>(x.size()), best.get(), prob.get(), nullptr);
    double sum = 0;
    for (float v : x)
        sum += std::exp(static_cast<double>(v) - x[static_cast<size_t>(want(x))]);
    CHECK(std::fabs(prob.download()[0] / (1.0 / sum) - 1.0) < 1e-5);
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
        check_prob(test::random_floats(n, -20, 20, 2));  // a flat-ish distribution
        check_prob(test::random_floats(40960, -5, 5, 3));
        std::vector<float> peaked = test::random_floats(n, -20, 0, 4);
        peaked[1234] = 30.0f;  // probability ~1
        check_prob(peaked);
        std::printf("argmax: random, ties, ends and NaN, and argmax_prob %s\n", test::failures == 0 ? "ok" : "FAIL");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "argmax_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
