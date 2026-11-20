// Parity of the mixture-of-experts kernels (src/kernels/moe) with a scalar double reference: router_top_k with
// random logits and with ties (lowest index first), swiglu and moe_combine at the model's sizes, and expert_table.
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
    const int experts = static_cast<int>(logits.size());
    double peak = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(experts);
    double sum = 0;
    for (int i = 0; i < experts; ++i)
        sum += p[i] = std::exp(logits[i] - peak);
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    double chosen = 0;
    for (int r = 0; r < kTopK; ++r)
        chosen += p[order[r]] / sum;
    chosen = std::max(chosen, 6.103515625e-5);

    DeviceBuffer<float> dlogits(logits), dweights(kTopK);
    DeviceBuffer<int32_t> dids(kTopK);
    kernels::router_top_k(dlogits.get(), experts, kTopK, dids.get(), dweights.get(), nullptr);
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

// The pointer table of resident experts: slot s of gate, up and down at base + ids[s] * slice.
void check_table() {
    const std::vector<int32_t> ids = {511, 0, 7, 300, 4, 130, 9, 260, 400, 1};
    const kernels::ResidentExperts e{reinterpret_cast<const uint8_t*>(0x100000),
                                     reinterpret_cast<const uint8_t*>(0x900000),
                                     reinterpret_cast<const uint8_t*>(0x2000000), 1740800, 870400};
    DeviceBuffer<int32_t> dids(ids);
    DeviceBuffer<const uint8_t*> table(3 * kTopK);
    kernels::expert_table(dids.get(), kTopK, e, table.get(), nullptr);
    const std::vector<const uint8_t*> got = table.download();
    bool same = true;
    for (int s = 0; s < kTopK; ++s) {
        const uint64_t id = static_cast<uint64_t>(ids[static_cast<size_t>(s)]);
        same &= got[static_cast<size_t>(s)] == e.gate + id * e.gate_up_slice &&
                got[static_cast<size_t>(kTopK + s)] == e.up + id * e.gate_up_slice &&
                got[static_cast<size_t>(2 * kTopK + s)] == e.down + id * e.down_slice;
    }
    std::printf("expert_table %s\n", same ? "ok" : "FAIL");
    CHECK(same);
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
        check_router("router_top_k partial warp", std::vector<float>(37, 0.0f));
        check_router("router_top_k one warp", test::random_floats(19, -4, 4, 8));
        check_swiglu();
        check_combine();
        check_table();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "moe_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
