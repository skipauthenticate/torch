// Parity of the matrix-vector kernels (src/kernels/gemv) with the CPU reference: quantize_q8 bit for bit; gemv_q8
// on every quantized format against dequant_row (src/formats/dequant.h) times the same quantized activations, in
// double; gemv_bf16 against the reference and bit for bit against its canonical order, and the paired launch; the
// expert path, which must equal the dense kernel bit for bit and not depend on where an expert's bytes live; and the
// group launch of matrices that read one input, bit for bit against gemv_q8 on each; and the mixer's SiLU folded into
// the down projection, bit for bit against silu_scale.
#include "kernels/gemv/gemv.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <string>

#include "../unit/check.h"
#include "device.h"
#include "formats/dequant.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"

using namespace torch;
using kernels::BlockQ8;
using test::DeviceBuffer;

namespace {

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The CPU twin of kernels::quantize_q8.
BlockQ8 quantize_block(const float* x) {
    float amax = 0;
    for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0f;
    BlockQ8 b{};
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        b.q[i] = static_cast<int8_t>(amax == 0.0f ? 0 : static_cast<int>(std::round(x[i] / d)));
        sum += b.q[i];
    }
    b.d = std::bit_cast<uint16_t>(static_cast<_Float16>(d));
    b.sum = static_cast<int16_t>(sum);
    return b;
}

std::vector<BlockQ8> quantize(const std::vector<float>& x) {
    std::vector<BlockQ8> out(x.size() / 32);
    for (size_t b = 0; b < out.size(); ++b)
        out[b] = quantize_block(x.data() + 32 * b);
    return out;
}

// Activations with the cases quantization must handle: an all-zero block, tiny and large values.
std::vector<float> activations(int n, uint32_t seed) {
    std::vector<float> x = test::random_floats(n, -2.0f, 2.0f, seed);
    std::fill_n(x.begin() + 32, 32, 0.0f);
    x[70] = 1e-30f;
    x[100] = 300.0f;
    return x;
}

// Where each format keeps its F16 scales, so random blocks get finite, moderate ones: (block bytes, offsets).
struct Scales {
    int bytes;
    std::vector<int> offsets;
};
Scales scale_fields(Type t) {
    switch (t) {
        case Type::Q4_K:
            return {144, {0, 2}};
        case Type::Q5_K:
            return {176, {0, 2}};
        case Type::Q6_K:
            return {210, {208}};
        default:
            return {static_cast<int>(type_info(static_cast<uint32_t>(t))->block_bytes), {0}};
    }
}

std::vector<uint8_t> random_matrix(Type t, int rows, int cols, uint32_t seed) {
    const uint64_t rb = row_bytes(*type_info(static_cast<uint32_t>(t)), cols);
    std::vector<uint8_t> m(rb * rows);
    std::mt19937 rng(seed);
    for (uint8_t& b : m)
        b = static_cast<uint8_t>(rng());
    const Scales s = scale_fields(t);
    std::uniform_real_distribution<float> scale(0.001f, 0.02f);
    for (size_t block = 0; block < m.size() / s.bytes; ++block) {
        for (int offset : s.offsets) {
            const uint16_t h = std::bit_cast<uint16_t>(static_cast<_Float16>(scale(rng)));
            std::memcpy(m.data() + block * s.bytes + offset, &h, 2);
        }
    }
    return m;
}

// want[r] = W[r] . dequantized x in double; scale[r] = sum of |products|.
void reference(Type t, const std::vector<uint8_t>& m, int rows, int cols, const std::vector<BlockQ8>& xq,
               std::vector<double>& want, std::vector<double>& scale) {
    const uint64_t rb = row_bytes(*type_info(static_cast<uint32_t>(t)), cols);
    std::vector<float> w(cols);
    want.assign(rows, 0);
    scale.assign(rows, 0);
    for (int r = 0; r < rows; ++r) {
        CHECK(dequant_row(t, m.data() + r * rb, cols, w.data()));
        for (int i = 0; i < cols; ++i) {
            const double x = static_cast<double>(xq[i / 32].q[i % 32]) * fp16_to_f32(xq[i / 32].d);
            want[r] += w[i] * x;
            scale[r] += std::fabs(w[i] * x);
        }
    }
}

void check_quantize() {
    const std::vector<float> x = activations(2560, 1);
    const std::vector<BlockQ8> want = quantize(x);
    DeviceBuffer<float> dx(x);
    DeviceBuffer<BlockQ8> dq(want.size());
    kernels::quantize_q8(dx.get(), 2560, dq.get(), nullptr);
    const std::vector<BlockQ8> got = dq.download();
    const bool same = std::memcmp(got.data(), want.data(), got.size() * sizeof(BlockQ8)) == 0;
    CHECK(same);
    std::printf("%-40s %s\n", "quantize_q8 bit for bit", same ? "ok" : "FAIL");
}

void check_format(Type t, int cols) {
    const int rows = 37;  // not a multiple of the 8 rows per block
    const std::vector<uint8_t> m = random_matrix(t, rows, cols, 7 + static_cast<uint32_t>(t));
    const std::vector<BlockQ8> xq = quantize(activations(cols, 3));
    std::vector<double> want, scale;
    reference(t, m, rows, cols, xq, want, scale);

    DeviceBuffer<uint8_t> dm(m);
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> dy(rows);
    kernels::gemv_q8(t, dm.get(), rows, cols, dx.get(), dy.get(), nullptr);
    const std::string what =
        "gemv_q8 " + std::string(type_info(static_cast<uint32_t>(t))->name) + " x " + std::to_string(cols);
    CHECK(test::report(what.c_str(), test::worst_relative(dy.download(), want, scale), 1e-5));
}

// The canonical BF16 order (gemv.h) on the CPU: lane l of 32 sums the 8-element chunks l, l + 32, ... with one fused
// multiply-add per element, then the 32 lane sums are added in the warp_sum butterfly.
float bf16_canonical(const uint8_t* row, const float* x, int cols) {
    float lane[32] = {};
    for (int i = 0; i < cols; ++i) {
        uint16_t h = 0;
        std::memcpy(&h, row + 2 * i, 2);
        const float w = std::bit_cast<float>(static_cast<uint32_t>(h) << 16);
        lane[i / 8 % 32] = std::fma(w, x[i], lane[i / 8 % 32]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        float next[32];
        for (int l = 0; l < 32; ++l)
            next[l] = lane[l] + lane[l ^ offset];
        std::copy_n(next, 32, lane);
    }
    return lane[0];
}

std::vector<uint8_t> random_bf16(int rows, int cols, uint32_t seed) {
    std::vector<uint8_t> m(size_t{2} * rows * cols);
    const std::vector<float> values = test::random_floats(static_cast<size_t>(rows) * cols, -1, 1, seed);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t h = static_cast<uint16_t>(std::bit_cast<uint32_t>(values[i]) >> 16);
        std::memcpy(m.data() + 2 * i, &h, 2);
    }
    return m;
}

// Against the double reference within tolerance, and bit for bit against the canonical order: 320 columns take the
// short-row kernel, 2,560 the one-row-per-warp kernel.
void check_bf16(int cols) {
    const int rows = 37;
    const std::vector<uint8_t> m = random_bf16(rows, cols, 5);
    const std::vector<float> x = activations(cols, 9);
    std::vector<double> want(rows, 0), scale(rows, 0);
    std::vector<float> w(cols), canonical(rows);
    for (int r = 0; r < rows; ++r) {
        dequant_row(Type::BF16, m.data() + size_t{2} * r * cols, cols, w.data());
        for (int i = 0; i < cols; ++i) {
            want[r] += static_cast<double>(w[i]) * x[i];
            scale[r] += std::fabs(static_cast<double>(w[i]) * x[i]);
        }
        canonical[r] = bf16_canonical(m.data() + size_t{2} * r * cols, x.data(), cols);
    }
    DeviceBuffer<uint8_t> dm(m);
    DeviceBuffer<float> dx(x), dy(rows);
    kernels::gemv_bf16(dm.get(), rows, cols, dx.get(), dy.get(), nullptr);
    const std::vector<float> y = dy.download();
    const std::string what = "gemv_bf16 x " + std::to_string(cols);
    CHECK(test::report(what.c_str(), test::worst_relative(y, want, scale), 1e-5));
    const bool same = same_bits(y, canonical);
    CHECK(same);
    std::printf("%-40s %s\n", "  canonical order bit for bit", same ? "ok" : "FAIL");
}

// A mixer's down projection with its inject rows in one launch equals the two launches bit for bit.
void check_bf16_pair(int rows, int rows2, int cols) {
    const std::vector<uint8_t> m = random_bf16(rows, cols, 21), m2 = random_bf16(rows2, cols, 22);
    const std::vector<float> x = activations(cols, 23);
    DeviceBuffer<uint8_t> dm(m), dm2(m2);
    DeviceBuffer<float> dx(x), dy(rows), dy2(rows2), py(rows), py2(rows2);
    kernels::gemv_bf16(dm.get(), rows, cols, dx.get(), dy.get(), nullptr);
    kernels::gemv_bf16(dm2.get(), rows2, cols, dx.get(), dy2.get(), nullptr);
    kernels::gemv_bf16_pair(dm.get(), rows, dm2.get(), rows2, cols, dx.get(), py.get(), py2.get(), nullptr);
    const bool same = same_bits(dy.download(), py.download()) && same_bits(dy2.download(), py2.download());
    CHECK(same);
    const std::string what =
        "gemv_bf16_pair " + std::to_string(rows) + "+" + std::to_string(rows2) + " x " + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// Three slots: experts 0 and 1 in device memory, and expert 0 again in mapped pinned memory (a staging slot), each
// with its own activations (slot 2 gets slot 0's). With `swiglu`, gate and up as the routed gate and up projections,
// else gate alone as the down projection. Checks the CPU reference within tolerance, the dense kernel bit for bit
// (gemv_q8 per slot, and swiglu of its two results), and slot 2 against slot 0 bit for bit.
void check_experts(Type t, int rows, int cols, bool swiglu) {
    const uint64_t matrix = row_bytes(*type_info(static_cast<uint32_t>(t)), cols) * rows;
    const std::vector<uint8_t> gate = random_matrix(t, rows * 2, cols, 11), up = random_matrix(t, rows * 2, cols, 12);
    DeviceBuffer<uint8_t> dgate(gate), dup(up);
    uint8_t* staged = nullptr;
    TORCH_CUDA_CHECK(cudaHostAlloc(&staged, 2 * matrix, cudaHostAllocMapped));
    std::memcpy(staged, gate.data(), matrix);
    std::memcpy(staged + matrix, up.data(), matrix);
    uint8_t* staged_view = nullptr;
    TORCH_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&staged_view), staged, 0));

    const std::vector<const uint8_t*> gates = {dgate.get(), dgate.get() + matrix, staged_view};
    const std::vector<const uint8_t*> ups = {dup.get(), dup.get() + matrix, staged_view + matrix};
    DeviceBuffer<const uint8_t*> gate_table(gates), up_table(ups);
    std::vector<float> x = activations(cols * 3, 13);
    std::copy_n(x.begin(), cols, x.begin() + 2 * cols);
    const std::vector<BlockQ8> xq = quantize(x);
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> dy(size_t{3} * rows);
    kernels::gemv_q8_experts(t, gate_table.get(), swiglu ? up_table.get() : nullptr, 3, rows, cols, dx.get(), cols / 32,
                             dy.get(), nullptr);
    const std::vector<float> y = dy.download();

    // the dense kernel on the same bytes and activations, slot by slot
    DeviceBuffer<float> dense(size_t{3} * rows), dense_up(size_t{3} * rows), dense_h(size_t{3} * rows);
    for (int slot = 0; slot < 3; ++slot) {
        const BlockQ8* xs = dx.get() + slot * cols / 32;
        kernels::gemv_q8(t, gates[slot], rows, cols, xs, dense.get() + slot * rows, nullptr);
        if (swiglu)
            kernels::gemv_q8(t, ups[slot], rows, cols, xs, dense_up.get() + slot * rows, nullptr);
    }
    if (swiglu)
        kernels::swiglu(dense.get(), dense_up.get(), dense_h.get(), 3 * rows, nullptr);
    const bool same_dense = same_bits(y, (swiglu ? dense_h : dense).download());

    std::vector<double> want, scale;
    double worst = 0;
    for (int slot = 0; slot < 2; ++slot) {
        const std::vector<BlockQ8> xs(xq.begin() + slot * cols / 32, xq.begin() + (slot + 1) * cols / 32);
        std::vector<double> g, gs, u, us;
        const std::vector<uint8_t> gm(gate.begin() + slot * matrix, gate.begin() + (slot + 1) * matrix);
        reference(t, gm, rows, cols, xs, g, gs);
        if (swiglu) {
            const std::vector<uint8_t> um(up.begin() + slot * matrix, up.begin() + (slot + 1) * matrix);
            reference(t, um, rows, cols, xs, u, us);
        }
        for (int r = 0; r < rows; ++r) {
            const float got = y[slot * rows + r];
            if (!swiglu) {
                worst = std::max(worst, std::fabs(got - g[r]) / std::max(gs[r], 1e-30));
                continue;
            }
            // d(silu(g) u) is at most |u| dg + |g| du: the errors of both products, propagated
            const double want_r = g[r] / (1 + std::exp(-g[r])) * u[r];
            const double bound = std::fabs(u[r]) * gs[r] * 1.1 + std::fabs(g[r]) * us[r];
            worst = std::max(worst, std::fabs(got - want_r) / std::max(bound, 1e-30));
        }
    }
    const std::string what = "gemv_q8_experts " + std::string(swiglu ? "swiglu " : "down ") +
                             std::string(type_info(static_cast<uint32_t>(t))->name) + " " + std::to_string(rows) + "x" +
                             std::to_string(cols);
    CHECK(test::report(what.c_str(), worst, 1e-5));
    CHECK(same_dense);
    std::printf("%-40s %s\n", "  equals the dense kernel bit for bit", same_dense ? "ok" : "FAIL");
    const bool same = std::memcmp(y.data(), y.data() + 2 * rows, rows * sizeof(float)) == 0;
    CHECK(same);
    std::printf("%-40s %s\n", "  staged expert equals resident expert", same ? "ok" : "FAIL");
    cudaFreeHost(staged);
}

// A group launch (gemv_q8_group) bit for bit against gemv_q8 on each of its matrices, and with `swiglu` against swiglu
// of the two gemv_q8 results. Row counts not a multiple of the group kernel's 2 rows a warp.
void check_group(std::initializer_list<std::pair<Type, int>> shape, bool swiglu) {
    const int cols = 2560;
    std::vector<std::unique_ptr<DeviceBuffer<uint8_t>>> matrices;
    std::vector<kernels::Q8Part> parts;
    int total = 0;
    uint32_t seed = 31;
    for (auto [t, rows] : shape) {
        matrices.push_back(std::make_unique<DeviceBuffer<uint8_t>>(random_matrix(t, rows, cols, seed++)));
        total += rows;
    }
    const std::vector<BlockQ8> xq = quantize(activations(cols, 33));
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> got(static_cast<size_t>(total)), want(static_cast<size_t>(total)),
        h(static_cast<size_t>(total));
    int at = 0, i = 0;
    for (auto [t, rows] : shape) {
        const uint8_t* m = matrices[static_cast<size_t>(i)]->get();
        parts.push_back({t, m, rows, got.get() + at});
        kernels::gemv_q8(t, m, rows, cols, dx.get(), want.get() + at, nullptr);
        at += rows, ++i;
    }
    kernels::gemv_q8_group(parts.data(), static_cast<int>(parts.size()), cols, dx.get(), swiglu, nullptr);
    std::string what = swiglu ? "gemv_q8_group swiglu" : "gemv_q8_group";
    for (auto [t, rows] : shape)
        what += " " + std::string(type_info(static_cast<uint32_t>(t))->name) + "x" + std::to_string(rows);
    bool same = false;
    if (swiglu) {
        const int rows = parts[0].rows;
        kernels::swiglu(want.get(), want.get() + rows, h.get(), rows, nullptr);
        const std::vector<float> g = got.download(), w = h.download();
        same = std::memcmp(g.data(), w.data(), static_cast<size_t>(rows) * sizeof(float)) == 0;
    } else {
        same = same_bits(got.download(), want.download());
    }
    CHECK(same);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// A mixer's down projection storing its low-rank SiLU (gemv_bf16_pair's silu_scale) against gemv_bf16 followed by
// silu_scale, bit for bit; the inject rows unchanged.
void check_bf16_silu(int rows, int rows2, int cols) {
    const std::vector<uint8_t> m = random_bf16(rows, cols, 41), m2 = random_bf16(rows2, cols, 42);
    const std::vector<float> x = activations(cols, 43);
    DeviceBuffer<uint8_t> dm(m), dm2(m2);
    DeviceBuffer<float> dx(x), dy(rows), dy2(rows2), py(rows), py2(rows2);
    kernels::gemv_bf16(dm.get(), rows, cols, dx.get(), dy.get(), nullptr);
    kernels::silu_scale(dy.get(), rows, 0.25f, nullptr);
    kernels::gemv_bf16(dm2.get(), rows2, cols, dx.get(), dy2.get(), nullptr);
    kernels::gemv_bf16_pair(dm.get(), rows, dm2.get(), rows2, cols, dx.get(), py.get(), py2.get(), nullptr, 0.25f);
    const bool same = same_bits(dy.download(), py.download()) && same_bits(dy2.download(), py2.download());
    CHECK(same);
    std::printf("%-40s %s\n", "gemv_bf16_pair with silu_scale", same ? "ok" : "FAIL");
}

}  // namespace

int main() {
    try {
        kernels::upload_iq_tables();
        check_quantize();
        for (Type t : {Type::Q8_0, Type::Q2_0, Type::IQ4_NL, Type::IQ4_XS, Type::Q4_K, Type::Q5_K, Type::Q6_K,
                       Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S}) {
            CHECK(kernels::gemv_q8_supports(t));
            check_format(t, 2560);
            if (type_info(static_cast<uint32_t>(t))->block_elements <= 64)
                check_format(t, 640);
        }
        CHECK(!kernels::gemv_q8_supports(Type::BF16));
        for (int cols : {320, 2560})
            check_bf16(cols);
        check_bf16_pair(320, 4, 10240);  // a mixer's down projection and inject rows
        check_bf16_silu(320, 4, 10240);
        for (Type t : {Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S, Type::IQ4_XS}) {
            check_experts(t, 640, 2560, true);  // gate and up: 640 rows of 2,560
            check_experts(t, 640, 2560, false);
        }
        for (Type t : {Type::IQ4_NL, Type::Q2_0})
            check_experts(t, 2560, 640, false);  // down: 2,560 rows of 640
        for (Type t : {Type::Q4_K, Type::Q5_K, Type::Q6_K, Type::IQ4_XS, Type::IQ2_S})
            CHECK(kernels::gemv_q8_groups(t) == (t != Type::IQ2_S));
        // the shapes of a layer's matrices that read one input, in mixed formats
        check_group({{Type::IQ4_XS, 61}, {Type::Q6_K, 7}, {Type::Q5_K, 9}}, false);  // attention q, k, v
        check_group({{Type::Q6_K, 41}, {Type::Q4_K, 23}}, false);                    // DeltaNet qkv and z
        check_group({{Type::Q4_K, 37}, {Type::IQ4_XS, 37}}, true);                   // the shared expert's gate and up
        check_group({{Type::Q6_K, 37}, {Type::Q5_K, 37}}, true);
        check_group({{Type::Q6_K, 37}, {Type::Q6_K, 37}}, true);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemv_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
