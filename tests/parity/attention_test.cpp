// Parity of the attention kernels (src/kernels/attention) with a scalar double reference: rope_neox, store_kv (bit
// for bit against the CPU F16 conversion) and attention_decode at position 0, a middle position and the last one of
// the cache, with the model's grouped-query layout (24 query heads, 2 KV heads, 256 dimensions).
#include "kernels/attention/attention.h"

#include <algorithm>
#include <bit>

#include "../unit/check.h"
#include "device.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

constexpr int kHeads = 24, kKvHeads = 2, kD = 256, kMaxCtx = 2048;
constexpr int kRowLen = kKvHeads * kD;

uint16_t to_f16(float x) {
    return std::bit_cast<uint16_t>(static_cast<_Float16>(x));
}
double from_f16(uint16_t bits) {
    return static_cast<double>(std::bit_cast<_Float16>(bits));
}

// The angle is a float product with a float powf; its few-ulp error scales with theta, which is at most
// pos, so the tolerance grows with the position. The rotation itself is two products and a sum.
void check_rope(int pos_value) {
    const int n_rot = 64, stride = 512;
    const float base = 1e7f;
    const std::vector<float> x = test::random_floats(static_cast<size_t>(kHeads) * stride, -2, 2, 1 + pos_value);
    std::vector<double> want(x.begin(), x.end()), mag(x.size(), 1);
    for (int h = 0; h < kHeads; ++h) {
        for (int i = 0; i < n_rot / 2; ++i) {
            const double theta = pos_value * std::pow(double{base}, -2.0 * i / n_rot);
            const double x0 = x[h * stride + i], x1 = x[h * stride + i + n_rot / 2];
            want[h * stride + i] = x0 * std::cos(theta) - x1 * std::sin(theta);
            want[h * stride + i + n_rot / 2] = x0 * std::sin(theta) + x1 * std::cos(theta);
            mag[h * stride + i] = mag[h * stride + i + n_rot / 2] = std::fabs(x0) + std::fabs(x1) + 1e-30;
        }
    }
    DeviceBuffer<float> dx(x);
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{pos_value});
    kernels::rope_neox(dx.get(), stride, kHeads, n_rot, dpos.get(), base, nullptr);
    const std::string what = "rope_neox pos " + std::to_string(pos_value);
    CHECK(test::report(what.c_str(), test::worst_relative(dx.download(), want, mag), 2e-6 + 4e-7 * pos_value));
}

void check_store_kv() {
    const std::vector<float> k = test::random_floats(kRowLen, -3, 3, 20), v = test::random_floats(kRowLen, -3, 3, 21);
    DeviceBuffer<float> dk(k), dv(v);
    DeviceBuffer<uint16_t> kc(static_cast<size_t>(kRowLen) * 4), vc(static_cast<size_t>(kRowLen) * 4);
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{3});
    kernels::store_kv(dk.get(), dv.get(), kc.get(), vc.get(), kRowLen, dpos.get(), nullptr);
    const std::vector<uint16_t> kg = kc.download(), vg = vc.download();
    bool same = true;
    for (int i = 0; i < kRowLen; ++i)
        same = same && kg[3 * kRowLen + i] == to_f16(k[i]) && vg[3 * kRowLen + i] == to_f16(v[i]);
    CHECK(same);
    std::printf("%-40s %s\n", "store_kv F16 bits, row 3", same ? "ok" : "FAIL");
}

// Error is measured against the sum of |weight * v| times the gate: the output is a weighted sum that can cancel. The
// scores are 256-term dots (a few 1e-7 of the sum of |q k|), the softmax weights inherit that, and the weighted sum
// adds up to 2,048 roundings in a sequential order: 2e-5 bounds all three with margin.
void check_attention(int pos_value, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc,
                     const std::vector<float>& qg) {
    const float scale = 1.0f / 16.0f;
    DeviceBuffer<float> dqg(qg), dout(static_cast<size_t>(kHeads) * kD);
    DeviceBuffer<uint16_t> dkc(kc), dvc(vc);
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{pos_value});
    kernels::attention_decode(dqg.get(), dqg.get() + kD, 2 * kD, dkc.get(), dvc.get(), dout.get(), dpos.get(), kHeads,
                              kKvHeads, scale, kMaxCtx, nullptr);
    std::vector<double> want(static_cast<size_t>(kHeads) * kD), mag(want.size());
    for (int h = 0; h < kHeads; ++h) {
        const int kv = h / (kHeads / kKvHeads);
        const float* q = qg.data() + h * 2 * kD;
        std::vector<double> score(pos_value + 1);
        double peak = -1e300;
        for (int p = 0; p <= pos_value; ++p) {
            double dot = 0;
            for (int d = 0; d < kD; ++d)
                dot += q[d] * from_f16(kc[static_cast<size_t>(p) * kRowLen + kv * kD + d]);
            score[p] = dot * scale;
            peak = std::max(peak, score[p]);
        }
        double sum = 0;
        for (double& s : score)
            sum += s = std::exp(s - peak);
        for (int d = 0; d < kD; ++d) {
            double acc = 0, abs_sum = 0;
            for (int p = 0; p <= pos_value; ++p) {
                const double term = score[p] / sum * from_f16(vc[static_cast<size_t>(p) * kRowLen + kv * kD + d]);
                acc += term;
                abs_sum += std::fabs(term);
            }
            const double gate = test::sigmoid(q[kD + d]);
            want[h * kD + d] = acc * gate;
            mag[h * kD + d] = abs_sum * gate + 1e-30;
        }
    }
    const std::string what = "attention_decode pos " + std::to_string(pos_value);
    CHECK(test::report(what.c_str(), test::worst_relative(dout.download(), want, mag), 2e-5));
}

}  // namespace

int main() {
    try {
        for (int pos : {0, 11, 300})
            check_rope(pos);
        check_store_kv();
        std::vector<uint16_t> kc(static_cast<size_t>(kMaxCtx) * kRowLen), vc(kc.size());
        const std::vector<float> k = test::random_floats(kc.size(), -1, 1, 30),
                                 v = test::random_floats(kc.size(), -1, 1, 31);
        for (size_t i = 0; i < kc.size(); ++i) {
            kc[i] = to_f16(k[i]);
            vc[i] = to_f16(v[i]);
        }
        // [q | gate] per head, as the projection lays it out
        const std::vector<float> qg = test::random_floats(static_cast<size_t>(kHeads) * 2 * kD, -1, 1, 32);
        for (int pos : {0, 777, kMaxCtx - 1})
            check_attention(pos, kc, vc, qg);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "attention_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
