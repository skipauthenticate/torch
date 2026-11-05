// Parity of rms_norm and l2_norm (src/kernels/norm) with a scalar double reference: per-group and shared weights, a
// gate, no weight, a strided layout and the in-place form.
#include "kernels/norm/norm.h"

#include <cstring>

#include "../unit/check.h"
#include "device.h"
#include "kernels/gemv/gemv.h"
#include "reference.h"

using namespace torch;
using test::DeviceBuffer;

namespace {

// Elementwise ops with a handful of roundings (the sum of squares, rsqrtf, two or three products, expf): a few
// 1e-7 each. 1e-5 leaves room and still catches a wrong formula, which is off by far more.
constexpr double kTolerance = 1e-5;

struct Case {
    const char* name;
    int n, groups;
    int64_t x_stride, w_stride, gate_stride;
    bool weight, gate, in_place;
};

void check_rms(const Case& c, uint32_t seed) {
    const float eps = 1e-6f;
    const size_t x_len = static_cast<size_t>(c.x_stride) * c.groups;
    const std::vector<float> x = test::random_floats(x_len, -2, 2, seed);
    const std::vector<float> w =
        test::random_floats(std::max<size_t>(c.w_stride * c.groups, c.n), 0.5f, 1.5f, seed + 1);
    const std::vector<float> gate =
        test::random_floats(static_cast<size_t>(c.gate_stride) * c.groups + c.n, -3, 3, seed + 2);

    std::vector<double> want(x_len, 0), scale(x_len, 1);
    for (int g = 0; g < c.groups; ++g) {
        double ss = 0;
        for (int i = 0; i < c.n; ++i)
            ss += double{x[g * c.x_stride + i]} * x[g * c.x_stride + i];
        const double inv = 1.0 / std::sqrt(ss / c.n + eps);
        for (int i = 0; i < c.n; ++i) {
            double v = x[g * c.x_stride + i] * inv;
            if (c.weight)
                v *= w[g * c.w_stride + i];
            if (c.gate)
                v *= test::sigmoid(gate[g * c.gate_stride + i]);
            want[g * c.x_stride + i] = v;
            scale[g * c.x_stride + i] = std::fabs(v) + 1e-30;
        }
    }
    DeviceBuffer<float> dx(x), dw(w), dgate(gate), dout(x_len);
    float* out = c.in_place ? dx.get() : dout.get();
    kernels::rms_norm(dx.get(), c.x_stride, c.weight ? dw.get() : nullptr, c.w_stride, c.gate ? dgate.get() : nullptr,
                      c.gate_stride, out, c.x_stride, c.n, c.groups, eps, nullptr);
    std::vector<float> got = c.in_place ? dx.download() : dout.download();
    // the padding between groups is not written: compare only the group elements
    for (int g = 0; g < c.groups; ++g)
        for (int i = c.n; i < c.x_stride; ++i)
            got[g * c.x_stride + i] = static_cast<float>(want[g * c.x_stride + i]);
    CHECK(test::report(c.name, test::worst_relative(got, want, scale), kTolerance));
}

// rms_norm_q8 must give rms_norm's output bit for bit and the Q8 blocks quantize_q8 makes of it (the model feeds them
// to ssm_out in place of quantizing again).
void check_rms_q8() {
    constexpr int kN = 128, kGroups = 48;
    const std::vector<float> x = test::random_floats(static_cast<size_t>(kN) * kGroups, -3, 3, 21);
    const std::vector<float> w = test::random_floats(kN, 0.5f, 1.5f, 22);
    const std::vector<float> gate = test::random_floats(static_cast<size_t>(kN) * kGroups, -4, 4, 23);
    DeviceBuffer<float> dx(x), dw(w), dgate(gate), out1(x.size()), out2(x.size());
    DeviceBuffer<kernels::BlockQ8> q1(x.size() / 32), q2(x.size() / 32);
    kernels::rms_norm(dx.get(), kN, dw.get(), 0, dgate.get(), kN, out1.get(), kN, kN, kGroups, 1e-6f, nullptr);
    kernels::quantize_q8(out1.get(), static_cast<int64_t>(x.size()), q1.get(), nullptr);
    kernels::rms_norm_q8(dx.get(), kN, dw.get(), 0, dgate.get(), kN, out2.get(), kN, q2.get(), kN, kGroups, 1e-6f,
                         nullptr);
    const std::vector<kernels::BlockQ8> a = q1.download(), b = q2.download();
    const bool same =
        out1.download() == out2.download() && std::memcmp(a.data(), b.data(), a.size() * sizeof(kernels::BlockQ8)) == 0;
    CHECK(same);
    std::printf("%-40s %s\n", "rms_norm_q8 = rms_norm + quantize_q8", same ? "ok" : "FAIL");
}

void check_l2() {
    const int n = 128, groups = 16;
    const float eps = 1e-6f;
    const std::vector<float> x = test::random_floats(static_cast<size_t>(n) * groups, -2, 2, 31);
    std::vector<double> want(x.size()), scale(x.size());
    for (int g = 0; g < groups; ++g) {
        double ss = 0;
        for (int i = 0; i < n; ++i)
            ss += double{x[g * n + i]} * x[g * n + i];
        const double inv = 1.0 / std::sqrt(ss / n + eps / n) / std::sqrt(double{n});
        for (int i = 0; i < n; ++i) {
            want[g * n + i] = x[g * n + i] * inv;
            scale[g * n + i] = std::fabs(want[g * n + i]) + 1e-30;
        }
    }
    DeviceBuffer<float> dx(x), dout(x.size());
    kernels::l2_norm(dx.get(), n, dout.get(), n, n, groups, eps, nullptr);
    CHECK(test::report("l2_norm 16 x 128", test::worst_relative(dout.download(), want, scale), kTolerance));
}

}  // namespace

int main() {
    try {
        check_rms({"rms_norm 4 x 2560, weights, gate", 2560, 4, 2560, 2560, 2560, true, true, false}, 1);
        check_rms({"rms_norm shared weight (w_stride 0)", 2560, 4, 2560, 0, 0, true, false, false}, 5);
        check_rms({"rms_norm in place, weights and gate", 2560, 4, 2560, 2560, 2560, true, true, true}, 9);
        check_rms({"rms_norm no weight", 2560, 2, 2560, 0, 0, false, false, false}, 13);
        check_rms({"rms_norm 48 x 128 strided, gate", 128, 48, 256, 0, 128, true, true, false}, 17);
        check_rms_q8();
        check_l2();
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "norm_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
