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

// Windows that accept only some of their columns, one after another, against one-token ple_conv over the accepted
// tokens: the rejected columns' x (junk here) must never reach a later window's taps, including tap 9, whose ring row
// is the one a rejected column would have overwritten. Each window's accepted columns are compared bit for bit.
void check_windows() {
    const int channels = kHc * kN, ring = 9;
    // (columns run, columns accepted) per window, positions following on from each other; 8 columns wrap the ring
    const std::vector<std::pair<int, int>> plan{{8, 2}, {8, 3}, {4, 4}, {8, 8}, {8, 1}, {3, 3}, {8, 5}, {2, 1}, {8, 8}};
    int total = 0;
    for (auto [run, accepted] : plan)
        total += accepted;
    const std::vector<float> w = test::random_floats(static_cast<size_t>(4) * channels, -1, 1, 15);
    std::vector<std::vector<float>> xs, gateds, ress;  // the true token at each position
    for (int p = 0; p < total; ++p) {
        xs.push_back(test::random_floats(channels, -2, 2, 200 + p));
        gateds.push_back(test::random_floats(channels, -1, 1, 300 + p));
        ress.push_back(test::random_floats(channels, -1, 1, 400 + p));
    }
    // one token at a time
    DeviceBuffer<float> dw(w), hist_ref(static_cast<size_t>(ring) * channels), res_ref(channels), gated_ref(channels),
        x_ref(channels);
    DeviceBuffer<int32_t> pos_ref(1);
    std::vector<std::vector<float>> want;
    for (int p = 0; p < total; ++p) {
        res_ref.upload(ress[p]);
        gated_ref.upload(gateds[p]);
        x_ref.upload(xs[p]);
        pos_ref.upload({p});
        kernels::ple_conv(res_ref.get(), gated_ref.get(), x_ref.get(), hist_ref.get(), dw.get(), channels,
                          pos_ref.get(), nullptr);
        want.push_back(res_ref.download());
    }
    // windows
    DeviceBuffer<float> hist(std::vector<float>(static_cast<size_t>(ring) * channels, 7.0f));  // stale: never read
    DeviceBuffer<int32_t> pos(1);
    int p = 0;
    bool same = true;
    for (auto [run, accepted] : plan) {
        std::vector<float> x, gated, res;
        for (int t = 0; t < run; ++t) {
            const bool real = t < accepted;
            const std::vector<float> junk = test::random_floats(channels, -2, 2, 900 + 20 * p + t);
            const std::vector<float>& xt = real ? xs[p + t] : junk;
            x.insert(x.end(), xt.begin(), xt.end());
            const std::vector<float>& gt = real ? gateds[p + t] : junk;
            gated.insert(gated.end(), gt.begin(), gt.end());
            const std::vector<float>& rt = real ? ress[p + t] : junk;
            res.insert(res.end(), rt.begin(), rt.end());
        }
        DeviceBuffer<float> dx(x), dgated(gated), dres(res);
        pos.upload({p});
        kernels::ple_conv_window(dres.get(), dgated.get(), dx.get(), hist.get(), dw.get(), channels, pos.get(), run,
                                 nullptr);
        const std::vector<float> got = dres.download();
        for (int t = 0; t < accepted; ++t)
            same = same && std::equal(got.begin() + static_cast<size_t>(t) * channels,
                                      got.begin() + static_cast<size_t>(t + 1) * channels, want[p + t].begin());
        if (p % 2 == 0)
            kernels::ple_commit(hist.get(), dx.get(), channels, pos.get(), accepted, nullptr);
        else {  // the count in device memory
            DeviceBuffer<int32_t> count(std::vector<int32_t>{accepted});
            kernels::ple_commit(hist.get(), dx.get(), channels, pos.get(), count.get(), nullptr);
        }
        p += accepted;
    }
    CHECK(same);
    std::printf("%-40s %s\n", "ple_conv windows with rejections", same ? "ok" : "FAIL");
}

}  // namespace

int main() {
    try {
        check_gate();
        check_gated_value();
        check_conv();
        check_windows();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ple_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
