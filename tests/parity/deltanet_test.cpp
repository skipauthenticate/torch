// Parity of the Gated DeltaNet kernels (src/kernels/deltanet) with a scalar double reference: conv_silu_l2 and gdn_step
// (with its gates) over consecutive tokens, so state carries over.
#include "kernels/deltanet/deltanet.h"

#include <algorithm>

#include "../unit/check.h"
#include "device.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

// Each output is a sum of up to 128 (or 4) products rounded in float, then one expf: a few 1e-7 of the sum of the
// absolute terms; 1e-5 leaves room for the state error that carries over several tokens.
constexpr double kTolerance = 1e-5;

void check_conv(int tokens) {
    const int normed = 16 * 128 * 2, channels = normed + 48 * 128;  // q, k and v of the model
    constexpr float kEps = 1e-6f;
    std::vector<float> state = test::random_floats(static_cast<size_t>(3) * channels, -1, 1, 1);
    const std::vector<float> w = test::random_floats(static_cast<size_t>(4) * channels, -1, 1, 2);
    DeviceBuffer<float> dstate(state), dw(w), dx(channels), dout(channels);
    std::vector<double> state_want(state.begin(), state.end());
    double worst = 0;
    for (int t = 0; t < tokens; ++t) {
        const std::vector<float> x = test::random_floats(channels, -2, 2, 10 + t);
        dx.upload(x);
        kernels::conv_silu_l2(dstate.get(), dx.get(), dw.get(), dout.get(), channels, normed, kEps, nullptr);
        std::vector<double> want(channels), mag(channels);
        for (int c = 0; c < channels; ++c) {
            const double taps[4] = {state_want[c], state_want[channels + c], state_want[2 * channels + c], x[c]};
            double pre = 0, abs_sum = 0;
            for (int j = 0; j < 4; ++j) {
                pre += w[4 * c + j] * taps[j];
                abs_sum += std::fabs(w[4 * c + j] * taps[j]);
            }
            want[c] = test::silu(pre);
            mag[c] = abs_sum + 1e-30;  // |silu'| <= 1.1: the error of pre bounds the error of out
            state_want[c] = taps[1];
            state_want[channels + c] = taps[2];
            state_want[2 * channels + c] = taps[3];
        }
        // q and k: each head of 128 divided by its L2 norm; an input's error carries over scaled by 1 / norm, and the
        // norm's own rounding is relative to the output.
        for (int head = 0; head < normed / 128; ++head) {
            double sum = 0;
            for (int i = 0; i < 128; ++i)
                sum += want[head * 128 + i] * want[head * 128 + i];
            const double norm = std::sqrt(sum + kEps);
            for (int i = 0; i < 128; ++i) {
                const int c = head * 128 + i;
                want[c] /= norm;
                mag[c] = mag[c] / norm + std::fabs(want[c]) + 1e-30;
            }
        }
        worst = std::max(worst, test::worst_relative(dout.download(), want, mag));
    }
    CHECK(test::report("conv_silu_l2 10240 channels, 6 tokens", worst, kTolerance));
    const std::vector<float> got = dstate.download();
    bool same = true;
    for (size_t i = 0; i < got.size(); ++i)
        same = same && got[i] == static_cast<float>(state_want[i]);
    CHECK(same);
    std::printf("%-40s %s\n", "  conv state shifted exactly", same ? "ok" : "FAIL");
}

void check_step(int tokens) {
    constexpr int kD = 128, kKeyHeads = 16, kValueHeads = 48;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    std::vector<float> state = test::random_floats(static_cast<size_t>(kValueHeads) * kD * kD, -0.5f, 0.5f, 7);
    std::vector<double> want_state(state.begin(), state.end());
    DeviceBuffer<float> dstate(state), dq(static_cast<size_t>(kKeyHeads) * kD), dk(static_cast<size_t>(kKeyHeads) * kD),
        dv(static_cast<size_t>(kValueHeads) * kD), dalpha(kValueHeads), dbeta(kValueHeads), ddt(kValueHeads),
        da(kValueHeads), dout(static_cast<size_t>(kValueHeads) * kD);
    const std::vector<float> dt = test::random_floats(kValueHeads, -1, 0, 5),
                             a = test::random_floats(kValueHeads, -1, -0.1f, 6);
    ddt.upload(dt);
    da.upload(a);
    double worst_gates = 0;
    double worst = 0;
    for (int t = 0; t < tokens; ++t) {
        const uint32_t seed = 100 + 10 * t;
        const std::vector<float> q = test::random_floats(static_cast<size_t>(kKeyHeads) * kD, -0.1f, 0.1f, seed);
        const std::vector<float> k = test::random_floats(static_cast<size_t>(kKeyHeads) * kD, -0.1f, 0.1f, seed + 1);
        const std::vector<float> v = test::random_floats(static_cast<size_t>(kValueHeads) * kD, -1, 1, seed + 2);
        // Raw gates: g = softplus(alpha + dt) * a lands in [-0.7, -0.01], beta = sigmoid(raw) in [0.12, 0.88].
        const std::vector<float> alpha = test::random_floats(kValueHeads, -1, 0, seed + 3);
        const std::vector<float> beta_raw = test::random_floats(kValueHeads, -2, 2, seed + 4);
        dq.upload(q);
        dk.upload(k);
        dv.upload(v);
        dalpha.upload(alpha);
        dbeta.upload(beta_raw);
        kernels::gdn_step(dstate.get(), dq.get(), dk.get(), dv.get(), dalpha.get(), dbeta.get(), ddt.get(), da.get(),
                          dout.get(), kKeyHeads, kValueHeads, scale, nullptr);
        std::vector<double> g(kValueHeads), beta(kValueHeads), g_mag(kValueHeads);
        for (int h = 0; h < kValueHeads; ++h) {
            g[h] = test::softplus(double{alpha[h]} + dt[h]) * a[h];
            beta[h] = test::sigmoid(beta_raw[h]);
            g_mag[h] = std::fabs(g[h]);
        }
        worst_gates = std::max({worst_gates, test::worst_relative(dalpha.download(), g, g_mag),
                                test::worst_relative(dbeta.download(), beta, beta)});

        std::vector<double> want(static_cast<size_t>(kValueHeads) * kD), mag(want.size());
        for (int h = 0; h < kValueHeads; ++h) {
            const float* qh = q.data() + (h % kKeyHeads) * kD;  // tiled, not interleaved
            const float* kh = k.data() + (h % kKeyHeads) * kD;
            double* s = want_state.data() + static_cast<size_t>(h) * kD * kD;
            const double eg = std::exp(g[h]);
            for (int j = 0; j < kD; ++j) {
                double kv = 0;
                for (int i = 0; i < kD; ++i)
                    kv += s[i * kD + j] * kh[i];
                const double d = (v[h * kD + j] - eg * kv) * beta[h];
                double acc = 0, abs_sum = 0;
                for (int i = 0; i < kD; ++i) {
                    s[i * kD + j] = eg * s[i * kD + j] + kh[i] * d;
                    acc += s[i * kD + j] * qh[i];
                    abs_sum += std::fabs(s[i * kD + j] * qh[i]);
                }
                want[h * kD + j] = acc * scale;
                mag[h * kD + j] = abs_sum * scale + 1e-30;
            }
        }
        worst = std::max(worst, test::worst_relative(dout.download(), want, mag));
    }
    CHECK(test::report("gdn_step gates stored back", worst_gates, kTolerance));
    CHECK(test::report("gdn_step 48 x 128 x 128, 4 tokens", worst, kTolerance));
    // the state is compared against the largest magnitude of its head: its entries are sums of cancelling terms
    const std::vector<float> got = dstate.download();
    std::vector<double> mag(got.size());
    for (size_t h = 0; h < kValueHeads; ++h) {
        double peak = 0;
        for (size_t i = 0; i < static_cast<size_t>(kD) * kD; ++i)
            peak = std::max(peak, std::fabs(want_state[h * kD * kD + i]));
        std::fill_n(mag.begin() + h * kD * kD, kD * kD, peak);
    }
    CHECK(test::report("  gdn_step state after 4 tokens", test::worst_relative(got, want_state, mag), kTolerance));
}

}  // namespace

int main() {
    try {
        check_conv(6);
        check_step(4);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "deltanet_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
