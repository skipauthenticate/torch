// Parity of the per-layer embedding kernels (src/kernels/ple) with a scalar double reference: ple_gate (including the
// sign(0) and floor cases), ple_gated_value, and ple_conv over 12 consecutive positions so the 9-row history ring
// wraps and the first positions read zero taps.
#include "kernels/ple/ple.h"

#include <algorithm>

#include "../unit/check.h"
#include "device.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

constexpr int kN = 2560, kHc = 4;
constexpr double kTolerance = 1e-5;

void check_gate() {
    std::vector<float> key = test::random_floats(static_cast<size_t>(kHc) * kN, -1, 1, 1);
    const std::vector<float> query = test::random_floats(static_cast<size_t>(kHc) * kN, -1, 1, 2);
    std::fill_n(key.begin() + 2 * kN, kN, 0.0f);  // stream 2: v = 0 exactly, so sign(0) = 0 and the gate is 0.5
    std::fill_n(key.begin() + 3 * kN, kN, 1e-9f);
    // stream 3: |v| ~ 1e-9 < 1e-6, so the floor makes the magnitude sqrt(1e-6) = 1e-3
    std::vector<double> want(kHc), mag(kHc);
    for (int s = 0; s < kHc; ++s) {
        double dot = 0;
        for (int i = 0; i < kN; ++i)
            dot += double{key[s * kN + i]} * query[s * kN + i];
        const double v = dot / std::sqrt(double{kN});
        const double sign = v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
        want[s] = test::sigmoid(sign * std::sqrt(std::max(std::fabs(v), 1e-6)));
        mag[s] = want[s];
    }
    DeviceBuffer<float> dkey(key), dquery(query), dgate(kHc);
    kernels::ple_gate(dkey.get(), dquery.get(), dgate.get(), kN, kHc, nullptr);
    // sqrt amplifies the error of v near zero, so the tolerance is looser than for a plain product
    CHECK(test::report("ple_gate", test::worst_relative(dgate.download(), want, mag), 2e-5));
}

void check_gated_value() {
    const std::vector<float> value = test::random_floats(kN, -2, 2, 3), gate = test::random_floats(kHc, 0, 1, 4);
    std::vector<double> want(static_cast<size_t>(kHc) * kN), mag(want.size());
    for (int s = 0; s < kHc; ++s)
        for (int e = 0; e < kN; ++e) {
            want[s * kN + e] = value[e] * gate[s];  // one float product: exact
            mag[s * kN + e] = std::fabs(want[s * kN + e]) + 1e-30;
        }
    DeviceBuffer<float> dvalue(value), dgate(gate), dgated(want.size());
    kernels::ple_gated_value(dvalue.get(), dgate.get(), dgated.get(), kN, kHc, nullptr);
    CHECK(test::report("ple_gated_value (single product)", test::worst_relative(dgated.download(), want, mag), 0));
}

void check_conv() {
    const int channels = kHc * kN, steps = 12, ring = 9;
    const std::vector<float> w = test::random_floats(static_cast<size_t>(4) * channels, -1, 1, 5);
    std::vector<float> res = test::random_floats(channels, -1, 1, 6);
    std::vector<double> res_want(res.begin(), res.end());
    std::vector<std::vector<float>> xs;  // x of every position so far, for the reference taps
    DeviceBuffer<float> dres(res), dgated(channels), dx(channels), dhist(static_cast<size_t>(ring) * channels), dw(w);
    dhist.upload(
        std::vector<float>(static_cast<size_t>(ring) * channels, 7.0f));  // stale values: the kernel must not read them
    DeviceBuffer<int32_t> dpos(1);
    double worst = 0;
    for (int pos = 0; pos < steps; ++pos) {
        xs.push_back(test::random_floats(channels, -2, 2, 50 + pos));
        const std::vector<float> gated = test::random_floats(channels, -1, 1, 80 + pos);
        dgated.upload(gated);
        dx.upload(xs[pos]);
        dpos.upload({pos});
        kernels::ple_conv(dres.get(), dgated.get(), dx.get(), dhist.get(), dw.get(), channels, dpos.get(), nullptr);
        std::vector<double> mag(channels);
        for (int c = 0; c < channels; ++c) {
            const auto back = [&](int b) { return pos - b < 0 ? 0.0 : double{xs[pos - b][c]}; };
            const double taps[4] = {back(9), back(6), back(3), xs[pos][c]};
            double pre = 0, abs_sum = 0;
            for (int j = 0; j < 4; ++j) {
                pre += w[4 * c + j] * taps[j];
                abs_sum += std::fabs(w[4 * c + j] * taps[j]);
            }
            const double add = gated[c] + test::silu(pre);
            res_want[c] += add;
            mag[c] = std::fabs(res_want[c]) + std::fabs(gated[c]) + abs_sum + 1e-30;
        }
        worst = std::max(worst, test::worst_relative(dres.download(), res_want, mag));
    }
    CHECK(test::report("ple_conv 10240 channels, positions 0-11", worst, kTolerance));
}

}  // namespace

int main() {
    try {
        check_gate();
        check_gated_value();
        check_conv();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ple_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
