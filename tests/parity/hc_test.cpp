// Parity of the hyper-connection kernels (src/kernels/hc) with a scalar double reference at the model's sizes.
#include "kernels/hc/hc.h"

#include <cstring>

#include "../unit/check.h"
#include "device.h"
#include "kernels/gemv/gemv.h"
#include "kernels/norm/norm.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

constexpr int kN = 2560, kHc = 4;
// Element-wise with at most a few roundings and one expf: errors are a few 1e-7 of the sum of the absolute terms.
constexpr double kTolerance = 1e-5;

void check_silu_scale() {
    const std::vector<float> x = test::random_floats(kN, -4, 4, 1);
    const float scale = 0.37f;
    std::vector<double> want(kN), mag(kN);
    for (int i = 0; i < kN; ++i) {
        want[i] = test::silu(double{x[i] * scale});
        mag[i] = std::fabs(want[i]) + 1e-30;
    }
    DeviceBuffer<float> dx(x);
    kernels::silu_scale(dx.get(), kN, scale, nullptr);
    CHECK(test::report("silu_scale", test::worst_relative(dx.download(), want, mag), kTolerance));
}

void check_collapse() {
    const std::vector<float> xn = test::random_floats(static_cast<size_t>(kHc) * kN, -2, 2, 2);
    const std::vector<float> gate = test::random_floats(static_cast<size_t>(kHc) * kN, -3, 3, 3);
    std::vector<double> want(kN, 0), mag(kN, 0);
    for (int e = 0; e < kN; ++e) {
        for (int s = 0; s < kHc; ++s) {
            const double term = double{xn[s * kN + e]} * test::sigmoid(gate[s * kN + e]);
            want[e] += term / kHc;
            mag[e] += std::fabs(term) / kHc;
        }
    }
    DeviceBuffer<float> dxn(xn), dgate(gate), dmixed(kN);
    DeviceBuffer<kernels::BlockQ8> q8(kN / 32), want_q8(kN / 32);
    kernels::hc_collapse(dxn.get(), dgate.get(), dmixed.get(), q8.get(), kN, kHc, nullptr);
    CHECK(test::report("hc_collapse", test::worst_relative(dmixed.download(), want, mag), kTolerance));
    // its Q8 blocks are bitwise those quantize_q8 makes of its output
    kernels::quantize_q8(dmixed.get(), kN, want_q8.get(), nullptr);
    const std::vector<kernels::BlockQ8> a = q8.download(), b = want_q8.download();
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(kernels::BlockQ8)) == 0;
    CHECK(same);
    std::printf("%-40s %s\n", "hc_collapse Q8 = quantize_q8", same ? "ok" : "FAIL");
}

void check_combine() {
    const std::vector<float> res = test::random_floats(static_cast<size_t>(kHc) * kN, -2, 2, 4);
    const std::vector<float> out = test::random_floats(kN, -2, 2, 5);
    const std::vector<float> inject = test::random_floats(kHc, -6, 6, 6);
    std::vector<double> want(res.size()), mag(res.size());
    for (int s = 0; s < kHc; ++s) {
        for (int e = 0; e < kN; ++e) {
            const double add = out[e] * 2.0 * test::sigmoid(double{inject[s]} / kHc);
            want[s * kN + e] = res[s * kN + e] + add;
            mag[s * kN + e] = std::fabs(res[s * kN + e]) + std::fabs(add) + 1e-30;
        }
    }
    DeviceBuffer<float> dres(res), dout(out), dinject(inject);
    kernels::hc_combine(dres.get(), dout.get(), dinject.get(), kN, kHc, nullptr);
    CHECK(test::report("hc_combine", test::worst_relative(dres.download(), want, mag), kTolerance));
}

// hc_combine_norm must equal hc_combine followed by rms_norm bit for bit (the model relies on it).
void check_combine_norm() {
    const std::vector<float> res = test::random_floats(static_cast<size_t>(kHc) * kN, -2, 2, 8);
    const std::vector<float> out = test::random_floats(kN, -2, 2, 9);
    const std::vector<float> inject = test::random_floats(kHc, -6, 6, 10);
    const std::vector<float> w = test::random_floats(static_cast<size_t>(kHc) * kN, 0.5, 1.5, 11);
    constexpr float kEps = 1e-6f;
    DeviceBuffer<float> r1(res), r2(res), dout(out), dinject(inject), dw(w), x1(res.size()), x2(res.size());
    kernels::hc_combine(r1.get(), dout.get(), dinject.get(), kN, kHc, nullptr);
    kernels::rms_norm(r1.get(), kN, dw.get(), kN, nullptr, 0, x1.get(), kN, kN, kHc, kEps, nullptr);
    kernels::hc_combine_norm(r2.get(), dout.get(), dinject.get(), dw.get(), x2.get(), kN, kHc, kEps, nullptr);
    const bool same = r1.download() == r2.download() && x1.download() == x2.download();
    CHECK(same);
    std::printf("%-40s %s\n", "hc_combine_norm = combine + rms_norm", same ? "ok" : "FAIL");
}

void check_broadcast() {
    const std::vector<float> x = test::random_floats(kN, -2, 2, 7);
    DeviceBuffer<float> dx(x), dres(static_cast<size_t>(kHc) * kN);
    kernels::hc_broadcast(dx.get(), dres.get(), kN, kHc, nullptr);
    const std::vector<float> got = dres.download();
    bool same = true;
    for (int s = 0; s < kHc; ++s)
        for (int e = 0; e < kN; ++e)
            same = same && got[s * kN + e] == x[e];
    CHECK(same);
    std::printf("%-40s %s\n", "hc_broadcast exact", same ? "ok" : "FAIL");
}

}  // namespace

int main() {
    try {
        check_silu_scale();
        check_collapse();
        check_combine();
        check_combine_norm();
        check_broadcast();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hc_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
