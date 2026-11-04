// Parity of the mixture-of-experts kernels (src/kernels/moe) with a scalar double reference: router_top_k with
// random logits and with ties (lowest index first), swiglu and moe_combine at the model's sizes.
#include "kernels/moe/moe.h"

#include <algorithm>
#include <numeric>

#include "../unit/check.h"
#include "device.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

constexpr int kExperts = 512, kTopK = 10, kN = 2560;
// A softmax division and a few products: a few 1e-7 relative.
constexpr double kTolerance = 1e-5;

// The reference order: probability descending, index ascending for equal probabilities.
void check_router(const char* name, const std::vector<float>& logits) {
    double peak = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(kExperts);
    double sum = 0;
    for (int i = 0; i < kExperts; ++i)
        sum += p[i] = std::exp(logits[i] - peak);
    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    double chosen = 0;
    for (int r = 0; r < kTopK; ++r)
        chosen += p[order[r]] / sum;
    chosen = std::max(chosen, 6.103515625e-5);

    DeviceBuffer<float> dlogits(logits), dweights(kTopK);
    DeviceBuffer<int32_t> dids(kTopK);
    kernels::router_top_k(dlogits.get(), kExperts, kTopK, dids.get(), dweights.get(), nullptr);
    const std::vector<int32_t> ids = dids.download();
    bool same_ids = true;
    std::vector<double> want(kTopK);
    for (int r = 0; r < kTopK; ++r) {
        same_ids = same_ids && ids[r] == order[r];
        want[r] = p[order[r]] / sum / chosen;
    }
    CHECK(same_ids);
    CHECK(test::report(name, test::worst_relative(dweights.download(), want, want), kTolerance));
    std::printf("%-40s %s\n", "  chosen experts and order", same_ids ? "ok" : "FAIL");
}

void check_swiglu() {
    const std::vector<float> g = test::random_floats(kN, -5, 5, 4), u = test::random_floats(kN, -2, 2, 5);
    std::vector<double> want(kN), mag(kN);
    for (int i = 0; i < kN; ++i) {
        want[i] = test::silu(g[i]) * u[i];
        mag[i] = std::fabs(want[i]) + 1e-30;
    }
    DeviceBuffer<float> dg(g), du(u), dh(kN);
    kernels::swiglu(dg.get(), du.get(), dh.get(), kN, nullptr);
    CHECK(test::report("swiglu", test::worst_relative(dh.download(), want, mag), kTolerance));
}

void check_combine() {
    const std::vector<float> y = test::random_floats(static_cast<size_t>(kTopK) * kN, -2, 2, 6);
    const std::vector<float> w = test::random_floats(kTopK, 0, 0.3f, 7);
    const std::vector<float> ys = test::random_floats(kN, -2, 2, 8);
    const std::vector<float> sg = {0.7f};
    std::vector<double> want(kN), mag(kN);
    for (int e = 0; e < kN; ++e) {
        double acc = 0, abs_sum = 0;
        for (int r = 0; r < kTopK; ++r) {
            acc += double{w[r]} * y[r * kN + e];
            abs_sum += std::fabs(double{w[r]} * y[r * kN + e]);
        }
        const double shared = test::sigmoid(sg[0]) * ys[e];
        want[e] = acc + shared;
        mag[e] = abs_sum + std::fabs(shared) + 1e-30;
    }
    DeviceBuffer<float> dy(y), dw(w), dys(ys), dsg(sg), dout(kN);
    kernels::moe_combine(dy.get(), dw.get(), kTopK, dys.get(), dsg.get(), dout.get(), kN, nullptr);
    CHECK(test::report("moe_combine", test::worst_relative(dout.download(), want, mag), kTolerance));
}

}  // namespace

int main() {
    try {
        check_router("router_top_k 512 random logits", test::random_floats(kExperts, -4, 4, 1));
        check_router("router_top_k all equal", std::vector<float>(kExperts, 0.0f));
        std::vector<float> ties = test::random_floats(kExperts, -4, 0, 2);
        for (int i : {511, 300, 7, 4, 130})
            ties[i] = 2.0f;  // five equal maxima in different warps: indices must come out 4, 7, 130, 300, 511
        for (int i : {9, 260, 400})
            ties[i] = 1.0f;  // then three equal runners-up
        check_router("router_top_k ties", ties);
        check_swiglu();
        check_combine();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "moe_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
