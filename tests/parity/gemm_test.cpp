// Parity of the prefill matrix products (src/kernels/gemm) with the one-column gemv_q8: every column bit for bit, for
// each dense format, at column counts that fill and leave ragged the kernel's 64-column passes and its column ranges,
// at both widths the model's dense matrices have (2,560 and 6,144), through the re-layout into Q8Columns; and the
// routed experts grouped by expert for each expert format, every route bit for bit, with the buckets' books.
#include "kernels/gemm/gemm.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../unit/check.h"
#include "device.h"
#include "formats/types.h"

using namespace torch;
using kernels::BlockQ8;
using test::DeviceBuffer;

namespace {

// Random blocks with finite, moderate F16 scales where each format keeps them: (block bytes, offsets).
std::vector<uint8_t> random_matrix(Type t, int rows, int cols, uint32_t seed) {
    const TypeInfo& info = *type_info(static_cast<uint32_t>(t));
    std::vector<uint8_t> m(row_bytes(info, static_cast<uint64_t>(cols)) * static_cast<size_t>(rows));
    std::mt19937 rng(seed);
    for (uint8_t& b : m)
        b = static_cast<uint8_t>(rng());
    const size_t block = info.block_bytes;
    const std::vector<size_t> scales = t == Type::Q6_K                      ? std::vector<size_t>{208}
                                       : t == Type::Q4_K || t == Type::Q5_K ? std::vector<size_t>{0, 2}
                                                                            : std::vector<size_t>{0};
    std::uniform_real_distribution<float> scale(0.001f, 0.02f);
    for (size_t b = 0; b < m.size() / block; ++b)
        for (const size_t at : scales) {
            const uint16_t h = std::bit_cast<uint16_t>(static_cast<_Float16>(scale(rng)));
            std::memcpy(m.data() + b * block + at, &h, 2);
        }
    return m;
}

void check_shape(Type type, int rows, int cols, int columns, uint32_t seed) {
    const std::vector<uint8_t> w = random_matrix(type, rows, cols, seed);
    const int units = cols / 32;
    DeviceBuffer<uint8_t> dw(w);
    DeviceBuffer<float> x(test::random_floats(static_cast<size_t>(cols) * columns, -2.0f, 2.0f, seed + 1));
    DeviceBuffer<BlockQ8> xq(static_cast<size_t>(units) * columns);
    kernels::quantize_q8(x.get(), int64_t{cols} * columns, xq.get(), nullptr);
    DeviceBuffer<float> want(static_cast<size_t>(rows) * columns), got(static_cast<size_t>(rows) * columns);
    for (int c = 0; c < columns; ++c)
        kernels::gemv_q8(type, dw.get(), rows, cols, xq.get() + static_cast<size_t>(c) * units,
                         want.get() + static_cast<size_t>(c) * rows, nullptr, 1);
    DeviceBuffer<uint8_t> memory(kernels::q8_columns_bytes(units, columns));
    const kernels::Q8Columns xc = kernels::q8_columns_at(memory.get(), units, columns);
    kernels::q8_to_columns(xq.get(), xc, nullptr);
    kernels::gemm_q8(type, dw.get(), rows, cols, xc, got.get(), nullptr);
    const std::vector<float> a = want.download(), b = got.download();
    size_t differ = 0;
    for (size_t i = 0; i < a.size(); ++i)
        differ += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    const std::string what = "gemm_q8 " + std::string(type_info(static_cast<uint32_t>(type))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols) + " at " + std::to_string(columns) +
                             " columns";
    std::printf("%-44s %s (%zu of %zu differ)\n", what.c_str(), differ == 0 ? "ok" : "FAIL", differ, a.size());
    CHECK(differ == 0);
}

// Routed experts: `experts` random matrices of `type` (rows x cols), `columns` columns routing k of them each (some
// experts unrouted, one popular), grouped by expert; every route bitwise gemv_q8 of its expert on its input. For the
// gate/up shape the input is the column's (gathered by route), for the down shape each route's own (as h is), stored
// at the route's row through the slots' index.
void check_experts(Type type, int rows, int cols, bool per_route, int columns, uint32_t seed) {
    constexpr int kExperts = 12, k = 10;
    const int units = cols / 32, routes = columns * k;
    std::vector<std::vector<uint8_t>> host;
    std::vector<std::unique_ptr<DeviceBuffer<uint8_t>>> mats;
    std::vector<const uint8_t*> table(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        host.push_back(random_matrix(type, rows, cols, seed + static_cast<uint32_t>(e)));
        mats.push_back(std::make_unique<DeviceBuffer<uint8_t>>(host.back()));
        table[static_cast<size_t>(e)] = mats.back()->get();
    }
    std::mt19937 rng(seed);
    std::vector<int32_t> ids(static_cast<size_t>(routes));
    for (int t = 0; t < columns; ++t)
        for (int r = 0; r < k; ++r)  // experts 10 and 11 never routed; expert 3 often
            ids[static_cast<size_t>(t * k + r)] = r == 0 ? 3 : static_cast<int32_t>(rng() % 10);
    DeviceBuffer<int32_t> did(ids);
    DeviceBuffer<const uint8_t*> dtable(table);
    const int inputs = per_route ? routes : columns;
    DeviceBuffer<float> x(test::random_floats(static_cast<size_t>(cols) * inputs, -2.0f, 2.0f, seed + 99));
    DeviceBuffer<BlockQ8> xq(static_cast<size_t>(units) * inputs);
    kernels::quantize_q8(x.get(), int64_t{cols} * inputs, xq.get(), nullptr);
    // the window path: each route's product by gemv_q8
    DeviceBuffer<float> want(static_cast<size_t>(rows) * routes), got(static_cast<size_t>(rows) * routes);
    for (int r = 0; r < routes; ++r)
        kernels::gemv_q8(type, table[static_cast<size_t>(ids[static_cast<size_t>(r)])], rows, cols,
                         xq.get() + static_cast<size_t>(per_route ? r : r / k) * units,
                         want.get() + static_cast<size_t>(r) * rows, nullptr, 1);
    DeviceBuffer<uint8_t> bucket_memory(kernels::expert_buckets_bytes(kExperts, routes));
    const kernels::ExpertBuckets b = kernels::expert_buckets_at(bucket_memory.get(), kExperts, routes);
    DeviceBuffer<int32_t> pos(std::vector<int32_t>{100});
    DeviceBuffer<unsigned long long> counts(std::vector<unsigned long long>(kExperts, 5));
    DeviceBuffer<uint32_t> stamps(std::vector<uint32_t>(kExperts, 7));
    kernels::expert_buckets(did.get(), routes, k, kExperts, b, 0, pos.get(), counts.get(), stamps.get(), nullptr);
    DeviceBuffer<uint8_t> columns_memory(kernels::q8_columns_bytes(units, b.slots + 64));
    kernels::Q8Columns xc = kernels::q8_columns_at(columns_memory.get(), units, b.slots + 64);
    xc.columns = b.slots;
    if (per_route) {  // h in slot order: each slot's route's own input
        std::vector<int32_t> slot_route(static_cast<size_t>(b.slots));
        TORCH_CUDA_CHECK(
            cudaMemcpy(slot_route.data(), b.slot_route, sizeof(int32_t) * slot_route.size(), cudaMemcpyDeviceToHost));
        const std::vector<BlockQ8> hq = xq.download();
        std::vector<BlockQ8> slotted(static_cast<size_t>(b.slots) * units);
        for (int s2 = 0; s2 < b.slots; ++s2)
            if (slot_route[static_cast<size_t>(s2)] >= 0)
                std::copy_n(hq.begin() + static_cast<std::ptrdiff_t>(slot_route[static_cast<size_t>(s2)]) * units,
                            units, slotted.begin() + static_cast<std::ptrdiff_t>(s2) * units);
        DeviceBuffer<BlockQ8> dslotted(slotted);
        kernels::q8_to_columns(dslotted.get(), xc, nullptr);
        kernels::gemm_q8_experts(type, dtable.get(), b, kExperts, rows, cols, xc, b.slot_route, got.get(), nullptr);
    } else {
        kernels::q8_gather_columns(xq.get(), units, b, k, xc, nullptr);
        DeviceBuffer<float> slots_out(static_cast<size_t>(rows) * b.slots);
        kernels::gemm_q8_experts(type, dtable.get(), b, kExperts, rows, cols, xc, nullptr, slots_out.get(), nullptr);
        // scatter the slot results to route order on the host
        std::vector<int32_t> slot_route(static_cast<size_t>(b.slots));
        TORCH_CUDA_CHECK(
            cudaMemcpy(slot_route.data(), b.slot_route, sizeof(int32_t) * slot_route.size(), cudaMemcpyDeviceToHost));
        const std::vector<float> per_slot = slots_out.download();
        std::vector<float> scattered(static_cast<size_t>(rows) * routes);
        for (int s2 = 0; s2 < b.slots; ++s2)
            if (slot_route[static_cast<size_t>(s2)] >= 0)
                std::copy_n(
                    per_slot.begin() + static_cast<std::ptrdiff_t>(s2) * rows, rows,
                    scattered.begin() + static_cast<std::ptrdiff_t>(slot_route[static_cast<size_t>(s2)]) * rows);
        got.upload(scattered);
    }
    const std::vector<float> a = want.download(), g = got.download();
    size_t differ = 0;
    for (size_t i = 0; i < a.size(); ++i)
        differ += std::memcmp(&a[i], &g[i], sizeof(float)) != 0;
    const std::string what = "gemm_q8_experts " + std::string(type_info(static_cast<uint32_t>(type))->name) + " " +
                             std::to_string(rows) + "x" + std::to_string(cols) + ", " + std::to_string(routes) +
                             " routes";
    std::printf("%-44s %s (%zu of %zu differ)\n", what.c_str(), differ == 0 ? "ok" : "FAIL", differ, a.size());
    CHECK(differ == 0);
    // the buckets against a host count, the route counts and the window stamps
    const std::vector<unsigned long long> rc = counts.download();
    const std::vector<uint32_t> st = stamps.download();
    bool books = true;
    for (int e = 0; e < kExperts; ++e) {
        int n = 0, last = -1;
        for (int r = 0; r < routes; ++r)
            if (ids[static_cast<size_t>(r)] == e)
                ++n, last = r;
        books = books && rc[static_cast<size_t>(e)] == 5ull + static_cast<unsigned long long>(n);
        books =
            books && st[static_cast<size_t>(e)] == (n == 0 ? 7u : static_cast<uint32_t>(100 + last / k / 8 * 8 + 1));
    }
    std::printf("%-44s %s\n", "  route counts and window stamps", books ? "ok" : "FAIL");
    CHECK(books);
}

// BF16 weights x F32 activations: every column bitwise gemv_bf16 (and gemv_bf16_pair's SiLU of a scaled sum).
void check_bf16(int rows, int cols, int columns, float silu_scale, uint32_t seed, int rows2 = 0) {
    const int total = rows + rows2;
    std::vector<uint8_t> w(static_cast<size_t>(total) * static_cast<size_t>(cols) * 2);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.05f);
    for (size_t i = 0; i < w.size() / 2; ++i) {
        const uint32_t bits = std::bit_cast<uint32_t>(normal(rng));
        const uint16_t h = static_cast<uint16_t>(bits >> 16);
        std::memcpy(w.data() + 2 * i, &h, 2);
    }
    DeviceBuffer<uint8_t> dw(w);
    DeviceBuffer<float> x(test::random_floats(static_cast<size_t>(cols) * columns, -3.0f, 3.0f, seed + 1));
    const uint8_t* w2 = rows2 > 0 ? dw.get() + static_cast<size_t>(rows) * cols * 2 : nullptr;
    DeviceBuffer<float> want(static_cast<size_t>(rows) * columns), got(static_cast<size_t>(rows) * columns);
    DeviceBuffer<float> want2(static_cast<size_t>(std::max(rows2, 1)) * columns),
        got2(static_cast<size_t>(std::max(rows2, 1)) * columns);
    for (int c = 0; c < columns; ++c)
        kernels::gemv_bf16_pair(dw.get(), rows, w2, rows2, cols, x.get() + static_cast<size_t>(c) * cols,
                                want.get() + static_cast<size_t>(c) * rows,
                                want2.get() + static_cast<size_t>(c) * static_cast<size_t>(rows2), nullptr, silu_scale,
                                1);
    kernels::gemm_bf16(dw.get(), rows, cols, x.get(), got.get(), columns, silu_scale, nullptr, w2, rows2, got2.get());
    std::vector<float> a = want.download(), b = got.download();
    if (rows2 > 0) {
        const std::vector<float> a2 = want2.download(), b2 = got2.download();
        a.insert(a.end(), a2.begin(), a2.end()), b.insert(b.end(), b2.begin(), b2.end());
    }
    size_t differ = 0;
    for (size_t i = 0; i < a.size(); ++i)
        differ += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    const std::string what = "gemm_bf16 " + std::to_string(rows) + "x" + std::to_string(cols) + " at " +
                             std::to_string(columns) + " columns" + (silu_scale != 0.0f ? ", SiLU" : "") +
                             (rows2 > 0 ? " + " + std::to_string(rows2) + " paired" : "");
    std::printf("%-44s %s (%zu of %zu differ)\n", what.c_str(), differ == 0 ? "ok" : "FAIL", differ, a.size());
    CHECK(differ == 0);
}

}  // namespace

int main() {
    try {
        kernels::upload_iq_tables();
        for (const Type t : {Type::Q6_K, Type::Q5_K, Type::Q4_K, Type::IQ4_XS, Type::IQ4_NL, Type::Q8_0}) {
            CHECK(kernels::gemm_q8_supports(t));
            for (const int columns : {1, 7, 131, 600})
                check_shape(t, 48, 2560, columns, 11u + static_cast<uint32_t>(columns));
            for (const int columns : {65, 300})
                check_shape(t, 32, 6144, columns, 101u + static_cast<uint32_t>(columns));
        }
        for (const Type t : {Type::IQ4_NL, Type::Q8_0})  // K = 640 (the shared expert's down): a short last octet
            for (const int columns : {9, 200})
                check_shape(t, 32, 640, columns, 201u + static_cast<uint32_t>(columns));
        CHECK(!kernels::gemm_q8_supports(Type::IQ3_S));
        kernels::upload_gemm_tables();
        for (const Type t : {Type::IQ2_S, Type::IQ3_XXS, Type::IQ3_S, Type::IQ4_XS})
            check_experts(t, 640, 2560, false, 37, 300u);  // gate and up: the columns' inputs
        for (const Type t : {Type::IQ4_NL, Type::Q2_0})
            check_experts(t, 2560, 640, true, 23, 400u);  // down: each route's h
        check_bf16(320, 10240, 37, 0.25f, 500u);          // a mixer's down projection, its SiLU
        check_bf16(320, 10240, 70, 0.25f, 501u, 4);       // with its inject rows paired in
        check_bf16(10240, 320, 9, 0.0f, 502u);            // up (the short-row kernel's shape)
        check_bf16(512, 2560, 130, 0.0f, 503u);           // the router
        check_bf16(48, 264, 5, 0.0f, 504u);               // 33 chunks: lane 0's pass has two; rows short of a tile
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemm_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
