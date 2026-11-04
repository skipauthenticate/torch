// Parity of the matrix-vector kernels (src/kernels/gemv) with the CPU reference: quantize_q8 bit for bit; gemv_q8
// on every quantized format against dequant_row (src/formats/dequant.h) times the same quantized activations, in
// double; gemv_bf16; and the expert path, whose result must not depend on where an expert's bytes live.
#include "kernels/gemv/gemv.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>

#include "../unit/check.h"
#include "device.h"
#include "formats/dequant.h"

using namespace torch;
using kernels::BlockQ8;
using test::DeviceBuffer;

namespace {

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

void check_bf16() {
    const int rows = 37, cols = 2560;
    std::vector<uint8_t> m(size_t{rows} * cols * 2);
    const std::vector<float> values = test::random_floats(size_t{rows} * cols, -1, 1, 5);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t h = static_cast<uint16_t>(std::bit_cast<uint32_t>(values[i]) >> 16);
        std::memcpy(m.data() + 2 * i, &h, 2);
    }
    const std::vector<float> x = activations(cols, 9);
    std::vector<double> want(rows, 0), scale(rows, 0);
    std::vector<float> w(cols);
    for (int r = 0; r < rows; ++r) {
        dequant_row(Type::BF16, m.data() + size_t{2} * r * cols, cols, w.data());
        for (int i = 0; i < cols; ++i) {
            want[r] += static_cast<double>(w[i]) * x[i];
            scale[r] += std::fabs(static_cast<double>(w[i]) * x[i]);
        }
    }
    DeviceBuffer<uint8_t> dm(m);
    DeviceBuffer<float> dx(x), dy(rows);
    kernels::gemv_bf16(dm.get(), rows, cols, dx.get(), dy.get(), nullptr);
    CHECK(test::report("gemv_bf16 x 2560", test::worst_relative(dy.download(), want, scale), 1e-5));
}

// Three slots: experts 0 and 1 in device memory, and expert 0 again in mapped pinned memory (a staging slot). The
// first and third results must be identical bit for bit.
void check_experts(Type t, int rows, int cols) {
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
    // each slot reads its own activations
    std::vector<float> x = activations(cols * 3, 13);
    std::copy_n(x.begin(), cols, x.begin() + 2 * cols);
    const std::vector<BlockQ8> xq = quantize(x);
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> dy(size_t{3} * rows);
    kernels::gemv_q8_experts(t, gate_table.get(), up_table.get(), 3, rows, cols, dx.get(), cols / 32, dy.get(),
                             nullptr);
    const std::vector<float> y = dy.download();

    std::vector<double> want, scale;
    double worst = 0;
    for (int slot = 0; slot < 2; ++slot) {
        const std::vector<BlockQ8> xs(xq.begin() + slot * cols / 32, xq.begin() + (slot + 1) * cols / 32);
        std::vector<double> g, gs, u, us;
        const std::vector<uint8_t> gm(gate.begin() + slot * matrix, gate.begin() + (slot + 1) * matrix);
        const std::vector<uint8_t> um(up.begin() + slot * matrix, up.begin() + (slot + 1) * matrix);
        reference(t, gm, rows, cols, xs, g, gs);
        reference(t, um, rows, cols, xs, u, us);
        for (int r = 0; r < rows; ++r) {
            // d(silu(g) u) is at most |u| dg + |g| du: the errors of both products, propagated
            const double want_r = g[r] / (1 + std::exp(-g[r])) * u[r];
            const double bound = std::fabs(u[r]) * gs[r] * 1.1 + std::fabs(g[r]) * us[r];
            worst = std::max(worst, std::fabs(y[slot * rows + r] - want_r) / std::max(bound, 1e-30));
        }
    }
    const std::string what = "gemv_q8_experts swiglu " + std::string(type_info(static_cast<uint32_t>(t))->name);
    CHECK(test::report(what.c_str(), worst, 1e-5));
    const bool same = std::memcmp(y.data(), y.data() + 2 * rows, rows * sizeof(float)) == 0;
    CHECK(same);
    std::printf("%-40s %s\n", "  staged expert equals resident expert", same ? "ok" : "FAIL");
    cudaFreeHost(staged);
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
        check_bf16();
        for (Type t : {Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S, Type::IQ4_XS})
            check_experts(t, 640, 2560);  // gate and up: 640 rows of 2,560
        // the down projection is the plain path: one matrix per slot, its own activations
        check_format(Type::Q2_0, 640);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemv_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
