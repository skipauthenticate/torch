// Parity of the attention kernels (src/kernels/attention) with a scalar double reference: rope_neox, store_kv (bit
// for bit against the CPU F16 conversion) and attention_decode at positions around the 128-row chunk boundaries, a
// middle position and the last one of the cache, with the model's grouped-query layout (24 query heads, 2 KV heads, 256
// dimensions).
#include "kernels/attention/attention.h"

#include <algorithm>
#include <bit>

#include "../unit/check.h"
#include "device.h"
#include "kernels/norm/norm.h"
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
    DeviceBuffer<float> dscratch(static_cast<size_t>(kernels::attention_scratch_floats(kHeads, kMaxCtx, 1)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{pos_value});
    kernels::attention_decode(dqg.get(), dqg.get() + kD, 2 * kD, dkc.get(), dvc.get(), dout.get(), dscratch.get(),
                              dpos.get(), kHeads, kKvHeads, scale, kMaxCtx, 1, nullptr);
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

// The fused launch must equal rms_norm (q, then k), rope_neox (q, then k) and store_kv in sequence bit for bit.
void check_fused(int pos_value) {
    const int n_rot = 64;
    const float base = 1e7f, eps = 1e-6f;
    const std::vector<float> q = test::random_floats(static_cast<size_t>(kHeads) * 2 * kD, -2, 2, 40),
                             k = test::random_floats(kRowLen, -2, 2, 41), v = test::random_floats(kRowLen, -2, 2, 42),
                             wq = test::random_floats(kD, 0.5f, 1.5f, 43), wk = test::random_floats(kD, 0.5f, 1.5f, 44);
    const std::vector<uint16_t> cache(static_cast<size_t>(kMaxCtx) * kRowLen, 0x1234);
    DeviceBuffer<int32_t> pos(std::vector<int32_t>{pos_value});
    DeviceBuffer<float> dwq(wq), dwk(wk), q1(q), k1(k), v1(v), q2(q), k2(k);
    DeviceBuffer<uint16_t> kc1(cache), vc1(cache), kc2(cache), vc2(cache);
    kernels::rms_norm(q1.get(), 2 * kD, dwq.get(), 0, nullptr, 0, q1.get(), 2 * kD, kD, kHeads, eps, nullptr);
    kernels::rms_norm(k1.get(), kD, dwk.get(), 0, nullptr, 0, k1.get(), kD, kD, kKvHeads, eps, nullptr);
    kernels::rope_neox(q1.get(), 2 * kD, kHeads, n_rot, pos.get(), base, nullptr);
    kernels::rope_neox(k1.get(), kD, kKvHeads, n_rot, pos.get(), base, nullptr);
    kernels::store_kv(k1.get(), v1.get(), kc1.get(), vc1.get(), kRowLen, pos.get(), nullptr);
    kernels::qk_norm_rope_store(q2.get(), 2 * kD, dwq.get(), k2.get(), dwk.get(), v1.get(), kc2.get(), vc2.get(),
                                kHeads, kKvHeads, n_rot, pos.get(), base, eps, 1, nullptr);
    const bool same = q1.download() == q2.download() && k1.download() == k2.download() &&
                      kc1.download() == kc2.download() && vc1.download() == vc2.download();
    CHECK(same);
    std::printf("%-40s %s\n", ("qk_norm_rope_store bitwise pos " + std::to_string(pos_value)).c_str(),
                same ? "ok" : "FAIL");
}

// A window of `columns` tokens at *pos = first: qk_norm_rope_store and attention_decode over all columns must give,
// for every column t, bit for bit what the one-token pair gives at position first + t after the earlier columns' rows
// are in the cache.
void check_window(int first, int columns) {
    const int n_rot = 64;
    const float base = 1e7f, eps = 1e-6f, scale = 1.0f / 16.0f;
    const size_t q_floats = static_cast<size_t>(kHeads) * 2 * kD, row = static_cast<size_t>(kRowLen);
    const size_t n = static_cast<size_t>(columns);
    const std::vector<float> q = test::random_floats(q_floats * n, -2, 2, 50 + first),
                             k = test::random_floats(row * n, -2, 2, 51 + first),
                             v = test::random_floats(row * n, -2, 2, 52 + first),
                             wq = test::random_floats(kD, 0.5f, 1.5f, 53), wk = test::random_floats(kD, 0.5f, 1.5f, 54);
    std::vector<uint16_t> kc(static_cast<size_t>(kMaxCtx) * kRowLen), vc(kc.size());
    const std::vector<float> kh = test::random_floats(kc.size(), -1, 1, 55),
                             vh = test::random_floats(kc.size(), -1, 1, 56);
    for (size_t i = 0; i < kc.size(); ++i)
        kc[i] = to_f16(kh[i]), vc[i] = to_f16(vh[i]);
    DeviceBuffer<float> dwq(wq), dwk(wk),
        dscratch(static_cast<size_t>(kernels::attention_scratch_floats(kHeads, kMaxCtx, columns)));
    // the window
    DeviceBuffer<float> qw(q), kw(k), vw(v), outw(static_cast<size_t>(kHeads) * kD * n);
    DeviceBuffer<uint16_t> kcw(kc), vcw(vc);
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{first});
    kernels::qk_norm_rope_store(qw.get(), 2 * kD, dwq.get(), kw.get(), dwk.get(), vw.get(), kcw.get(), vcw.get(),
                                kHeads, kKvHeads, n_rot, dpos.get(), base, eps, columns, nullptr);
    kernels::attention_decode(qw.get(), qw.get() + kD, 2 * kD, kcw.get(), vcw.get(), outw.get(), dscratch.get(),
                              dpos.get(), kHeads, kKvHeads, scale, kMaxCtx, columns, nullptr);
    // one token at a time
    DeviceBuffer<float> qo(q), ko(k), vo(v), outo(static_cast<size_t>(kHeads) * kD * n);
    DeviceBuffer<uint16_t> kco(kc), vco(vc);
    for (size_t t = 0; t < n; ++t) {
        DeviceBuffer<int32_t> pt(std::vector<int32_t>{first + static_cast<int>(t)});
        kernels::qk_norm_rope_store(qo.get() + t * q_floats, 2 * kD, dwq.get(), ko.get() + t * row, dwk.get(),
                                    vo.get() + t * row, kco.get(), vco.get(), kHeads, kKvHeads, n_rot, pt.get(), base,
                                    eps, 1, nullptr);
        kernels::attention_decode(qo.get() + t * q_floats, qo.get() + t * q_floats + kD, 2 * kD, kco.get(), vco.get(),
                                  outo.get() + t * kHeads * kD, dscratch.get(), pt.get(), kHeads, kKvHeads, scale,
                                  kMaxCtx, 1, nullptr);
    }
    const bool same = outw.download() == outo.download() && qw.download() == qo.download() &&
                      kw.download() == ko.download() && kcw.download() == kco.download() &&
                      vcw.download() == vco.download();
    CHECK(same);
    std::printf("%-40s %s\n",
                ("window T=" + std::to_string(columns) + " at " + std::to_string(first) + " bitwise").c_str(),
                same ? "ok" : "FAIL");
}

}  // namespace

int main() {
    try {
        for (int pos : {0, 11, 300})
            check_rope(pos);
        check_store_kv();
        for (int first : {0, 5, 126, 127, 1800, 2040})
            for (int columns = 1; columns <= 8; ++columns)
                if (first + columns <= kMaxCtx)
                    check_window(first, columns);
        for (int pos : {0, 3, 1800, kMaxCtx - 1})
            check_fused(pos);
        std::vector<uint16_t> kc(static_cast<size_t>(kMaxCtx) * kRowLen), vc(kc.size());
        const std::vector<float> k = test::random_floats(kc.size(), -1, 1, 30),
                                 v = test::random_floats(kc.size(), -1, 1, 31);
        for (size_t i = 0; i < kc.size(); ++i) {
            kc[i] = to_f16(k[i]);
            vc[i] = to_f16(v[i]);
        }
        // [q | gate] per head, as the projection lays it out
        const std::vector<float> qg = test::random_floats(static_cast<size_t>(kHeads) * 2 * kD, -1, 1, 32);
        // 127, 128, 129 and 255, 256, 257 straddle the 128-row chunks; 1920 starts the last one
        for (int pos : {0, 1, 63, 127, 128, 129, 255, 256, 257, 777, 1800, 1920, kMaxCtx - 1})
            check_attention(pos, kc, vc, qg);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "attention_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
