// Parity of the matrix-vector kernels (src/kernels/gemv) with the CPU reference: quantize_q8 bit for bit; gemv_q8
// on every quantized format against dequant_row (src/formats/dequant.h) times the same quantized activations, in
// double; gemv_bf16 against the reference and bit for bit against its canonical order, and the paired launch; the
// expert path, which must equal the dense kernel bit for bit and not depend on where an expert's bytes live; and the
// group launch of matrices that read one input, bit for bit against gemv_q8 on each; the mixer's SiLU folded into the
// down projection, bit for bit against silu_scale; and every multi-column kernel (verify windows) at 1 to 8 columns,
// each column bit for bit the one-column launch on it, the window's routed experts against the per-column path.
#include "kernels/gemv/gemv.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

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

// ---- Columns: every column of a T-column launch equals the one-column launch on that column, bit for bit ----------

bool column_same(const std::vector<float>& got, const std::vector<float>& want, int column, int n) {
    return std::memcmp(got.data() + static_cast<size_t>(column) * n, want.data() + static_cast<size_t>(column) * n,
                       static_cast<size_t>(n) * sizeof(float)) == 0;
}

// gemv_q8 at T = 1..8 distinct columns against eight one-column launches.
void check_columns(Type t, int rows, int cols) {
    const int units = cols / 32;
    const std::vector<uint8_t> m = random_matrix(t, rows, cols, 71 + static_cast<uint32_t>(t));
    const std::vector<BlockQ8> xq = quantize(activations(kernels::kMaxColumns * cols, 73));
    DeviceBuffer<uint8_t> dm(m);
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> want(static_cast<size_t>(kernels::kMaxColumns) * rows), got(want.bytes() / sizeof(float));
    for (int c = 0; c < kernels::kMaxColumns; ++c)
        kernels::gemv_q8(t, dm.get(), rows, cols, dx.get() + c * units, want.get() + c * rows, nullptr);
    const std::vector<float> w = want.download();
    bool same = true;
    for (int T = 1; T <= kernels::kMaxColumns; ++T) {
        kernels::gemv_q8(t, dm.get(), rows, cols, dx.get(), got.get(), nullptr, T);
        const std::vector<float> g = got.download();
        for (int c = 0; c < T; ++c)
            same = same && column_same(g, w, c, rows);
    }
    CHECK(same);
    const std::string what = "columns 1-8 " + std::string(type_info(static_cast<uint32_t>(t))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// gemv_q8_group (plain and SwiGLU) at T = 1..8 against one-column group launches.
void check_group_columns(std::initializer_list<std::pair<Type, int>> shape, bool swiglu) {
    const int cols = 2560, units = cols / 32, T8 = kernels::kMaxColumns;
    std::vector<std::unique_ptr<DeviceBuffer<uint8_t>>> matrices;
    std::vector<std::pair<Type, int>> parts_shape(shape);
    int total = 0;
    for (auto [t, rows] : parts_shape) {
        matrices.push_back(std::make_unique<DeviceBuffer<uint8_t>>(random_matrix(t, rows, cols, 81 + total)));
        total += rows;
    }
    const std::vector<BlockQ8> xq = quantize(activations(T8 * cols, 83));
    DeviceBuffer<BlockQ8> dx(xq);
    DeviceBuffer<float> want(static_cast<size_t>(T8) * total), got(static_cast<size_t>(T8) * total);
    // want: part p's column c at its one-column launch; got: part p's [T][rows] block
    auto parts_at = [&](float* y, bool columns_layout, int c) {
        std::vector<kernels::Q8Part> parts;
        size_t at = 0;
        for (size_t p = 0; p < parts_shape.size(); ++p) {
            const int rows = parts_shape[p].second;
            parts.push_back({parts_shape[p].first, matrices[p]->get(), rows,
                             y + at + (columns_layout ? 0 : static_cast<size_t>(c) * rows)});
            at += static_cast<size_t>(rows) * T8;
        }
        return parts;
    };
    for (int c = 0; c < T8; ++c) {
        const std::vector<kernels::Q8Part> parts = parts_at(want.get(), false, c);
        kernels::gemv_q8_group(parts.data(), static_cast<int>(parts.size()), cols, dx.get() + c * units, swiglu,
                               nullptr);
    }
    const std::vector<float> w = want.download();
    bool same = true;
    for (int T = 1; T <= T8; ++T) {
        const std::vector<kernels::Q8Part> parts = parts_at(got.get(), true, 0);
        kernels::gemv_q8_group(parts.data(), static_cast<int>(parts.size()), cols, dx.get(), swiglu, nullptr, T);
        const std::vector<float> g = got.download();
        size_t at = 0;
        for (size_t p = 0; p < parts_shape.size(); ++p) {
            const int rows = parts_shape[p].second;
            if (!swiglu || p == 0)
                for (int c = 0; c < T; ++c)
                    same =
                        same && std::memcmp(g.data() + at + static_cast<size_t>(c) * rows,
                                            w.data() + at + static_cast<size_t>(c) * rows, rows * sizeof(float)) == 0;
            at += static_cast<size_t>(rows) * T8;
        }
    }
    CHECK(same);
    std::printf("%-40s %s\n", swiglu ? "columns 1-8 group swiglu" : "columns 1-8 group", same ? "ok" : "FAIL");
}

// gemv_bf16_pair at T = 1..8 (short rows, and long rows with inject rows and the SiLU) against one-column launches.
void check_bf16_columns(int rows, int rows2, int cols, float silu) {
    const int T8 = kernels::kMaxColumns;
    const std::vector<uint8_t> m = random_bf16(rows, cols, 91), m2 = random_bf16(std::max(rows2, 1), cols, 92);
    const std::vector<float> x = activations(T8 * cols, 93);
    DeviceBuffer<uint8_t> dm(m), dm2(m2);
    DeviceBuffer<float> dx(x), want(static_cast<size_t>(T8) * (rows + rows2)), got(want.bytes() / sizeof(float));
    for (int c = 0; c < T8; ++c)
        kernels::gemv_bf16_pair(dm.get(), rows, dm2.get(), rows2, cols, dx.get() + static_cast<size_t>(c) * cols,
                                want.get() + static_cast<size_t>(c) * rows,
                                want.get() + static_cast<size_t>(T8) * rows + static_cast<size_t>(c) * rows2, nullptr,
                                silu);
    const std::vector<float> w = want.download();
    bool same = true;
    for (int T = 1; T <= T8; ++T) {
        kernels::gemv_bf16_pair(dm.get(), rows, dm2.get(), rows2, cols, dx.get(), got.get(),
                                got.get() + static_cast<size_t>(T8) * rows, nullptr, silu, T);
        const std::vector<float> g = got.download();
        for (int c = 0; c < T; ++c) {
            same = same && column_same(g, w, c, rows);
            same = same && std::memcmp(g.data() + static_cast<size_t>(T8) * rows + static_cast<size_t>(c) * rows2,
                                       w.data() + static_cast<size_t>(T8) * rows + static_cast<size_t>(c) * rows2,
                                       static_cast<size_t>(rows2) * sizeof(float)) == 0;
        }
    }
    CHECK(same);
    const std::string what =
        "columns 1-8 bf16 " + std::to_string(rows) + "+" + std::to_string(rows2) + "x" + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// gemv_bf16_collapse at T = 1..8 against gemv_bf16 and hc_collapse one column at a time: mixed and its Q8 blocks bit
// for bit (a mixer's up projection, hc x n rows of `cols`).
void check_collapse(Type type, int n, int hc, int cols) {
    const int T8 = kernels::kMaxColumns, rows = hc * n;
    const auto m = type == Type::BF16 ? random_bf16(rows, cols, 95) : random_matrix(type, rows, cols, 95);
    const std::vector<float> x = activations(T8 * cols, 96), xn = activations(T8 * rows, 97);
    DeviceBuffer<uint8_t> dm(m);
    DeviceBuffer<BlockQ8> xq(static_cast<size_t>(T8 * cols / 32));
    DeviceBuffer<float> dx(x), dxn(xn), gate(static_cast<size_t>(rows)), want(static_cast<size_t>(T8) * n),
        got(static_cast<size_t>(T8) * n);
    DeviceBuffer<kernels::BlockQ8> want_q8(static_cast<size_t>(T8) * n / 32), got_q8(static_cast<size_t>(T8) * n / 32);
    kernels::quantize_q8(dx.get(), T8 * cols, xq.get(), nullptr);
    for (int c = 0; c < T8; ++c) {
        if (type == Type::BF16)
            kernels::gemv_bf16(dm.get(), rows, cols, dx.get() + static_cast<size_t>(c) * cols, gate.get(), nullptr);
        else
            kernels::gemv_q8(type, dm.get(), rows, cols, xq.get() + c * (cols / 32), gate.get(), nullptr);
        kernels::hc_collapse(dxn.get() + static_cast<size_t>(c) * rows, gate.get(),
                             want.get() + static_cast<size_t>(c) * n, want_q8.get() + static_cast<size_t>(c) * n / 32,
                             n, hc, nullptr);
    }
    const std::vector<float> w = want.download();
    const std::vector<kernels::BlockQ8> wq = want_q8.download();
    bool same = true;
    for (int T = 1; T <= T8; ++T) {
        TORCH_CUDA_CHECK(cudaMemset(got.get(), 0xff, got.bytes()));
        TORCH_CUDA_CHECK(cudaMemset(got_q8.get(), 0xff, got_q8.bytes()));
        if (type == Type::BF16)
            same &= kernels::gemv_bf16_collapse(dm.get(), n, hc, cols, dx.get(), dxn.get(), got.get(), got_q8.get(),
                                                nullptr, T);
        else
            kernels::mixer_up_q8(dm.get(), n, dx.get(), dxn.get(), got.get(), got_q8.get(), T, nullptr);
        const std::vector<float> g = got.download();
        const std::vector<kernels::BlockQ8> gq = got_q8.download();
        same = same && std::memcmp(g.data(), w.data(), static_cast<size_t>(T) * n * sizeof(float)) == 0 &&
               std::memcmp(gq.data(), wq.data(), static_cast<size_t>(T) * n / 32 * sizeof(kernels::BlockQ8)) == 0;
    }
    CHECK(same);
    std::printf("columns 1-8 %s up + collapse: %s\n", type == Type::BF16 ? "BF16" : "Q8_0", same ? "ok" : "FAIL");
}

void check_mixer_down_q8() {
    constexpr int kCols = 10240, kRows = 320, kInject = 4;
    DeviceBuffer<uint8_t> w(random_matrix(Type::Q8_0, kRows, kCols, 181)), wi(random_bf16(kInject, kCols, 182));
    DeviceBuffer<float> x(activations(8 * kCols, 183)), low(8 * kRows), inject(8 * kInject);
    DeviceBuffer<BlockQ8> q(8 * kCols / 32);
    kernels::quantize_q8(x.get(), 8 * kCols, q.get(), nullptr);
    for (int c = 0; c < 8; ++c) {
        kernels::gemv_q8(Type::Q8_0, w.get(), kRows, kCols, q.get() + c * (kCols / 32), low.get() + c * kRows, nullptr);
        kernels::silu_scale(low.get() + c * kRows, kRows, 0.25f, nullptr);
        kernels::gemv_bf16(wi.get(), kInject, kCols, x.get() + c * kCols, inject.get() + c * kInject, nullptr);
    }
    const auto expected = low.download(), expected_inject = inject.download();
    for (int t = 1; t <= 8; ++t) {
        for (int paired : {0, kInject}) {
            DeviceBuffer<float> result(t * kRows), result_inject(t * kInject);
            TORCH_CUDA_CHECK(cudaMemset(result.get(), 0xff, result.bytes()));
            TORCH_CUDA_CHECK(cudaMemset(result_inject.get(), 0xff, result_inject.bytes()));
            kernels::mixer_down_q8(w.get(), paired ? wi.get() : nullptr, kRows, paired, kCols, q.get(), x.get(),
                                   result.get(), paired ? result_inject.get() : nullptr, 0.25f, t, nullptr);
            CHECK(std::memcmp(result.download().data(), expected.data(), result.bytes()) == 0);
            if (paired)
                CHECK(std::memcmp(result_inject.download().data(), expected_inject.data(), result_inject.bytes()) == 0);
        }
    }
}

}  // namespace

// A window's routes as the window gate numbers them: 10 distinct routes a column out of 24 experts (so columns share
// experts), unique experts in order of first appearance over (column, rank), with their column masks and ranks.
struct WindowRoutes {
    static constexpr int kExperts = 24, kRoutes = 10, kSlots = 80;
    std::vector<int> ids;     // column c's rank-r expert at c * kRoutes + r
    std::vector<int> unique;  // unique expert u's expert id
    std::vector<uint8_t> mask = std::vector<uint8_t>(kSlots, 0);
    std::vector<int8_t> rank = std::vector<int8_t>(size_t{kSlots} * kernels::kMaxColumns, -1);
};

WindowRoutes window_routes(int columns, std::mt19937& rng) {
    WindowRoutes w;
    for (int c = 0; c < columns; ++c) {
        std::vector<int> e(WindowRoutes::kExperts);
        for (int i = 0; i < WindowRoutes::kExperts; ++i)
            e[i] = i;
        std::shuffle(e.begin(), e.end(), rng);
        w.ids.insert(w.ids.end(), e.begin(), e.begin() + WindowRoutes::kRoutes);
    }
    for (int c = 0; c < columns; ++c)
        for (int r = 0; r < WindowRoutes::kRoutes; ++r) {
            const int e = w.ids[c * WindowRoutes::kRoutes + r];
            auto it = std::find(w.unique.begin(), w.unique.end(), e);
            const int u = static_cast<int>(it - w.unique.begin());
            if (it == w.unique.end())
                w.unique.push_back(e);
            w.mask[u] |= static_cast<uint8_t>(1u << c);
            w.rank[u * kernels::kMaxColumns + c] = static_cast<int8_t>(r);
        }
    return w;
}

// Device pointers to expert e's matrix for each route and each unique expert of w.
void route_tables(const WindowRoutes& w, const uint8_t* base, uint64_t matrix, std::vector<const uint8_t*>& routes,
                  std::vector<const uint8_t*>& unique) {
    routes.clear();
    for (int e : w.ids)
        routes.push_back(base + e * matrix);
    unique.assign(WindowRoutes::kSlots, nullptr);
    for (size_t u = 0; u < w.unique.size(); ++u)
        unique[u] = base + w.unique[u] * matrix;
}

// gemv_q8_experts_window at T = 1..8 against the per-column path (gemv_q8_experts on each column's 10 routes), byte
// for byte.
void check_experts_window(Type t, int rows, int cols) {
    constexpr int kExperts = WindowRoutes::kExperts, kRoutes = WindowRoutes::kRoutes, kSlots = WindowRoutes::kSlots;
    const int units = cols / 32;
    const uint64_t matrix = row_bytes(*type_info(static_cast<uint32_t>(t)), cols) * rows;
    const std::vector<uint8_t> gate = random_matrix(t, rows * kExperts, cols, 81),
                               up = random_matrix(t, rows * kExperts, cols, 82);
    DeviceBuffer<uint8_t> dgate(gate), dup(up);
    const std::vector<BlockQ8> xq = quantize(activations(kernels::kMaxColumns * cols, 83));
    DeviceBuffer<BlockQ8> dx(xq);
    std::mt19937 rng(84);
    bool same = true;
    for (int T = 1; T <= kernels::kMaxColumns; ++T) {
        const WindowRoutes w = window_routes(T, rng);
        std::vector<const uint8_t*> g_routes, u_routes, g_unique, u_unique;
        route_tables(w, dgate.get(), matrix, g_routes, g_unique);
        route_tables(w, dup.get(), matrix, u_routes, u_unique);
        const std::vector<int32_t> count = {static_cast<int32_t>(w.unique.size())};
        DeviceBuffer<const uint8_t*> dgr(g_routes), dur(u_routes), dgu(g_unique), duu(u_unique);
        DeviceBuffer<uint8_t> dmask(w.mask);
        DeviceBuffer<int8_t> drank(w.rank);
        DeviceBuffer<int32_t> dcount(count);
        DeviceBuffer<float> want(static_cast<size_t>(T) * kRoutes * rows), got(want.bytes() / sizeof(float));
        for (int c = 0; c < T; ++c)
            kernels::gemv_q8_experts(t, dgr.get() + c * kRoutes, dur.get() + c * kRoutes, kRoutes, rows, cols,
                                     dx.get() + c * units, 0, want.get() + static_cast<size_t>(c) * kRoutes * rows,
                                     nullptr);
        TORCH_CUDA_CHECK(cudaMemset(got.get(), 0xff, got.bytes()));
        kernels::gemv_q8_experts_window(t, dgu.get(), duu.get(), dmask.get(), drank.get(), dcount.get(),
                                        std::min(T * kRoutes, kSlots), kRoutes, rows, cols, dx.get(), got.get(), T,
                                        nullptr);
        same = same && same_bits(got.download(), want.download());
    }
    CHECK(same);
    const std::string what = "experts window 1-8 " + std::string(type_info(static_cast<uint32_t>(t))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// gemv_q8_experts_down_window at T = 1..8 against gemv_q8_experts on each column's 10 routes (each (column, rank)
// slot with its own input), byte for byte.
void check_down_window(Type t, int rows, int cols) {
    constexpr int kExperts = WindowRoutes::kExperts, kRoutes = WindowRoutes::kRoutes, kSlots = WindowRoutes::kSlots;
    const int units = cols / 32;
    const uint64_t matrix = row_bytes(*type_info(static_cast<uint32_t>(t)), cols) * rows;
    DeviceBuffer<uint8_t> dm(random_matrix(t, rows * kExperts, cols, 93));
    DeviceBuffer<BlockQ8> dx(quantize(activations(kSlots * cols, 94)));
    std::mt19937 rng(95);
    bool same = true;
    for (int T = 1; T <= kernels::kMaxColumns; ++T) {
        const WindowRoutes w = window_routes(T, rng);
        std::vector<const uint8_t*> routes, unique;
        route_tables(w, dm.get(), matrix, routes, unique);
        const std::vector<int32_t> count = {static_cast<int32_t>(w.unique.size())};
        DeviceBuffer<const uint8_t*> droutes(routes), dunique(unique);
        DeviceBuffer<uint8_t> dmask(w.mask);
        DeviceBuffer<int8_t> drank(w.rank);
        DeviceBuffer<int32_t> dcount(count);
        DeviceBuffer<float> want(static_cast<size_t>(T) * kRoutes * rows), got(want.bytes() / sizeof(float));
        for (int c = 0; c < T; ++c)
            kernels::gemv_q8_experts(t, droutes.get() + c * kRoutes, nullptr, kRoutes, rows, cols,
                                     dx.get() + c * kRoutes * units, units,
                                     want.get() + static_cast<size_t>(c) * kRoutes * rows, nullptr);
        TORCH_CUDA_CHECK(cudaMemset(got.get(), 0xff, got.bytes()));
        kernels::gemv_q8_experts_down_window(t, dunique.get(), dmask.get(), drank.get(), dcount.get(),
                                             std::min(T * kRoutes, kSlots), kRoutes, rows, cols, dx.get(), got.get(), T,
                                             nullptr);
        same = same && same_bits(got.download(), want.download());
    }
    CHECK(same);
    const std::string what = "down window 1-8 " + std::string(type_info(static_cast<uint32_t>(t))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

// The routed down projection of a window, gemv_q8_experts on 8 x 10 slots each with its own activations (past one
// launch's 48 KB of shared memory), against eight launches of 10 slots, byte for byte.
void check_experts_down_window(Type t, int rows, int cols) {
    constexpr int kSlots = 80, kRoutes = 10;
    const int units = cols / 32;
    const uint64_t matrix = row_bytes(*type_info(static_cast<uint32_t>(t)), cols) * rows;
    const std::vector<uint8_t> down = random_matrix(t, rows * kSlots, cols, 91);
    DeviceBuffer<uint8_t> dm(down);
    std::vector<const uint8_t*> table;
    for (int s = 0; s < kSlots; ++s)
        table.push_back(dm.get() + (s * 37 % kSlots) * matrix);
    DeviceBuffer<const uint8_t*> dtable(table);
    DeviceBuffer<BlockQ8> dx(quantize(activations(kSlots * cols, 92)));
    DeviceBuffer<float> want(static_cast<size_t>(kSlots) * rows), got(want.bytes() / sizeof(float));
    for (int c = 0; c < kSlots / kRoutes; ++c)
        kernels::gemv_q8_experts(t, dtable.get() + c * kRoutes, nullptr, kRoutes, rows, cols,
                                 dx.get() + c * kRoutes * units, units,
                                 want.get() + static_cast<size_t>(c) * kRoutes * rows, nullptr);
    kernels::gemv_q8_experts(t, dtable.get(), nullptr, kSlots, rows, cols, dx.get(), units, got.get(), nullptr);
    const bool same = same_bits(got.download(), want.download());
    CHECK(same);
    const std::string what = "experts down 80 slots " + std::string(type_info(static_cast<uint32_t>(t))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols);
    std::printf("%-40s %s\n", what.c_str(), same ? "ok" : "FAIL");
}

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
        // verify-window columns
        for (Type t : {Type::Q8_0, Type::Q2_0, Type::IQ4_NL, Type::IQ4_XS, Type::Q4_K, Type::Q5_K, Type::Q6_K,
                       Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S}) {
            check_columns(t, 37, 2560);
            if (type_info(static_cast<uint32_t>(t))->block_elements <= 64)
                check_columns(t, 37, 640);
        }
        check_columns(Type::Q6_K, 37, 6144);  // eight columns of activations past 48 KB of shared memory
        // Q8_0 at the shapes of a file that stores them so (rows x cols): a hyper-connection mixer's down and up, the
        // PLE block's key and value, the shared expert's down
        for (const auto& [rows, cols] : {std::pair{320, 10240}, {10240, 320}, {10240, 2560}, {2560, 2560}, {2560, 640}})
            check_columns(Type::Q8_0, rows, cols);
        check_group_columns({{Type::IQ4_XS, 61}, {Type::Q6_K, 7}, {Type::Q5_K, 9}}, false);
        check_group_columns({{Type::Q4_K, 37}, {Type::IQ4_XS, 37}}, true);
        check_bf16_columns(37, 0, 320, 0.0f);
        check_bf16_columns(320, 4, 10240, 0.25f);
        check_collapse(Type::BF16, 2560, 4, 320);
        check_collapse(Type::Q8_0, 2560, 4, 320);
        check_mixer_down_q8();
        check_bf16_columns(37, 5, 2560, 0.0f);
        for (Type t : {Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S, Type::IQ4_XS})
            check_experts_window(t, 37, 2560);  // gate and up of the routed experts, 37 rows of 2,560
        for (Type t : {Type::IQ4_NL, Type::Q2_0})
            check_experts_down_window(t, 37, 640);
        for (Type t : {Type::IQ4_NL, Type::Q2_0})
            check_down_window(t, 37, 640);  // routed down: rows of 640
        check_down_window(Type::IQ4_NL, 2560, 640);
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemv_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
