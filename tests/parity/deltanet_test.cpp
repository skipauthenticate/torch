// Parity of the Gated DeltaNet kernels (src/kernels/deltanet) with a scalar double reference: conv_silu and gdn_step
// over consecutive tokens (state carries over), and gdn_gates.
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
    const int channels = 16 * 128 * 2 + 48 * 128;  // q, k and v of the model
    std::vector<float> state = test::random_floats(static_cast<size_t>(3) * channels, -1, 1, 1);
    const std::vector<float> w = test::random_floats(static_cast<size_t>(4) * channels, -1, 1, 2);
    DeviceBuffer<float> dstate(state), dw(w), dx(channels), dout(channels);
    std::vector<double> state_want(state.begin(), state.end());
    double worst = 0;
    for (int t = 0; t < tokens; ++t) {
        const std::vector<float> x = test::random_floats(channels, -2, 2, 10 + t);
        dx.upload(x);
        kernels::conv_silu(dstate.get(), dx.get(), dw.get(), dout.get(), channels, nullptr);
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
        worst = std::max(worst, test::worst_relative(dout.download(), want, mag));
    }
    CHECK(test::report("conv_silu 10240 channels, 6 tokens", worst, kTolerance));
    const std::vector<float> got = dstate.download();
    bool same = true;
    for (size_t i = 0; i < got.size(); ++i)
        same = same && got[i] == static_cast<float>(state_want[i]);
    CHECK(same);
    std::printf("%-40s %s\n", "  conv state shifted exactly", same ? "ok" : "FAIL");
}

void check_gates() {
    const int heads = 48;
    const std::vector<float> beta = test::random_floats(heads, -4, 4, 3), alpha = test::random_floats(heads, -3, 3, 4);
    const std::vector<float> dt = test::random_floats(heads, -1, 1, 5), a = test::random_floats(heads, -2, -0.1f, 6);
    std::vector<double> want_beta(heads), want_alpha(heads), mag_beta(heads), mag_alpha(heads);
    for (int h = 0; h < heads; ++h) {
        want_beta[h] = test::sigmoid(beta[h]);
        want_alpha[h] = test::softplus(double{alpha[h]} + dt[h]) * a[h];
        mag_beta[h] = want_beta[h];
        mag_alpha[h] = std::fabs(want_alpha[h]);
    }
    DeviceBuffer<float> dbeta(beta), dalpha(alpha), ddt(dt), da(a);
    kernels::gdn_gates(dbeta.get(), dalpha.get(), ddt.get(), da.get(), heads, nullptr);
    CHECK(test::report("gdn_gates beta", test::worst_relative(dbeta.download(), want_beta, mag_beta), kTolerance));
    CHECK(test::report("gdn_gates alpha", test::worst_relative(dalpha.download(), want_alpha, mag_alpha), kTolerance));
}

void check_step(int tokens) {
    constexpr int kD = 128, kKeyHeads = 16, kValueHeads = 48;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    std::vector<float> state = test::random_floats(static_cast<size_t>(kValueHeads) * kD * kD, -0.5f, 0.5f, 7);
    std::vector<double> want_state(state.begin(), state.end());
    DeviceBuffer<float> dstate(state), dq(static_cast<size_t>(kKeyHeads) * kD), dk(static_cast<size_t>(kKeyHeads) * kD),
        dv(static_cast<size_t>(kValueHeads) * kD), dg(kValueHeads), dbeta(kValueHeads),
        dout(static_cast<size_t>(kValueHeads) * kD);
    double worst = 0;
    for (int t = 0; t < tokens; ++t) {
        const uint32_t seed = 100 + 10 * t;
        const std::vector<float> q = test::random_floats(static_cast<size_t>(kKeyHeads) * kD, -0.1f, 0.1f, seed);
        const std::vector<float> k = test::random_floats(static_cast<size_t>(kKeyHeads) * kD, -0.1f, 0.1f, seed + 1);
        const std::vector<float> v = test::random_floats(static_cast<size_t>(kValueHeads) * kD, -1, 1, seed + 2);
        const std::vector<float> g = test::random_floats(kValueHeads, -0.5f, -0.05f, seed + 3);
        const std::vector<float> beta = test::random_floats(kValueHeads, 0.1f, 0.9f, seed + 4);
        dq.upload(q);
        dk.upload(k);
        dv.upload(v);
        dg.upload(g);
        dbeta.upload(beta);
        kernels::gdn_step(dstate.get(), dq.get(), dk.get(), dv.get(), dg.get(), dbeta.get(), dout.get(), kKeyHeads,
                          kValueHeads, scale, nullptr);

        std::vector<double> want(static_cast<size_t>(kValueHeads) * kD), mag(want.size());
        for (int h = 0; h < kValueHeads; ++h) {
            const float* qh = q.data() + (h % kKeyHeads) * kD;  // tiled, not interleaved
            const float* kh = k.data() + (h % kKeyHeads) * kD;
            double* s = want_state.data() + static_cast<size_t>(h) * kD * kD;
            const double eg = std::exp(double{g[h]});
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
        check_gates();
        check_step(4);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "deltanet_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
