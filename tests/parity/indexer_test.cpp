// Parity of the sparse attention indexer (src/kernels/indexer) with a CPU reference: pooling (against a double
// reference within one F16 step), scoring and selection (exactly, ties included), the 2,051 / 2,052 boundary, the
// query's own block, window columns at every offset, and attention over cell lists (bitwise the dense kernel when
// every position is a cell). A prefill chunk in batches equals the same tokens fed in windows of 8, bit for bit.
#include "kernels/indexer/indexer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../unit/check.h"
#include "device.h"
#include "kernels/attention/attention.h"
#include "kernels/columns.h"

using namespace torch;
using namespace torch::kernels;
using test::DeviceBuffer;

namespace {

uint16_t to_f16(float x) {
    return std::bit_cast<uint16_t>(static_cast<_Float16>(x));
}
float from_f16(uint16_t bits) {
    return static_cast<float>(std::bit_cast<_Float16>(bits));
}

// The score of one pooled key against one column's query, in the order indexer.h gives.
float cpu_score(const float* q, const uint16_t* key) {
    const float weight = 1.0f / std::sqrt(128.0f);
    float score = 0.0f;
    for (int h = 0; h < kIndexHeads; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < kIndexDim; ++d)
            dot = std::fmaf(q[h * kIndexDim + d], from_f16(key[d]), dot);
        volatile float term = std::fmax(dot, 0.0f) * weight;  // volatile: the compiler may not fuse it into the sum
        score = score + term;
    }
    return score + 0.0f;
}

// The cell list a column of n visible tokens must get: blocks (ascending), first tail position, cell count.
struct List {
    std::vector<int> blocks;
    int tail, cells;
};

List cpu_select(const std::vector<float>& scores, int n) {
    const int visible = n / kIndexPool;
    std::vector<int> order(static_cast<size_t>(visible));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    List l;
    l.blocks.assign(order.begin(), order.begin() + std::min(visible, kIndexTopBlocks));
    std::sort(l.blocks.begin(), l.blocks.end());
    l.tail = kIndexPool * visible;
    l.cells = kIndexPool * static_cast<int>(l.blocks.size()) + n - l.tail;
    return l;
}

struct Selection {
    std::vector<std::vector<int>> blocks;
    std::vector<int> info;  // [t][4]
};

// Runs the score and select kernels for `columns` columns at position p over given pooled keys and queries.
Selection run_select(const std::vector<uint16_t>& pooled, const std::vector<float>& q, int p, int columns, int max_ctx,
                     std::vector<std::vector<float>>* scores_out = nullptr) {
    const int max_blocks = max_ctx / 4;
    DeviceBuffer<uint16_t> dpooled(pooled);
    DeviceBuffer<float> dq(q), dscores(static_cast<size_t>(index_score_floats(max_ctx, columns)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{p}), blocks(static_cast<size_t>(columns) * kIndexTopBlocks),
        info(static_cast<size_t>(columns) * 4);
    index_score(dq.get(), dpooled.get(), dscores.get(), dpos.get(), max_blocks, columns, nullptr);
    index_select(dscores.get(), {blocks.get(), info.get()}, dpos.get(), max_blocks, columns, nullptr);
    Selection s;
    const std::vector<int32_t> b = blocks.download();
    const std::vector<int32_t> info_host = info.download();
    s.info.assign(info_host.begin(), info_host.end());
    for (int t = 0; t < columns; ++t)
        s.blocks.emplace_back(b.begin() + static_cast<size_t>(t) * kIndexTopBlocks,
                              b.begin() + static_cast<size_t>(t) * kIndexTopBlocks + s.info[4 * t]);
    if (scores_out) {
        const std::vector<float> sc = dscores.download();
        for (int t = 0; t < columns; ++t)
            scores_out->emplace_back(sc.begin() + static_cast<size_t>(t) * max_blocks,
                                     sc.begin() + static_cast<size_t>(t + 1) * max_blocks);
    }
    return s;
}

// A window of `columns` columns at p against random pooled keys: every column's list equals the CPU's. With `ties`,
// the keys come from a few distinct patterns so many scores are equal (and many are exactly 0).
bool check_window(const std::string& name, int p, int columns, int max_ctx, bool ties, uint32_t seed) {
    const int max_blocks = max_ctx / 4;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> pooled(static_cast<size_t>(max_blocks) * kIndexDim);
    std::vector<std::vector<uint16_t>> patterns(6, std::vector<uint16_t>(kIndexDim));
    for (auto& pattern : patterns)
        for (uint16_t& v : pattern)
            v = to_f16(normal(rng));
    for (int b = 0; b < max_blocks; ++b)
        for (int d = 0; d < kIndexDim; ++d)
            pooled[static_cast<size_t>(b) * kIndexDim + d] =
                ties ? patterns[static_cast<size_t>(rng() % patterns.size())][d] : to_f16(normal(rng));
    std::vector<float> q(static_cast<size_t>(columns) * kIndexHeads * kIndexDim);
    for (size_t i = 0; i < q.size(); ++i) {
        const int t = static_cast<int>(i / (kIndexHeads * kIndexDim));
        q[i] = ties && t % 2 == 0 ? 0.0f : normal(rng);  // ties: every other column's query is all zero
    }
    std::vector<std::vector<float>> scores;
    const Selection got = run_select(pooled, q, p, columns, max_ctx, &scores);
    bool ok = true;
    for (int t = 0; t < columns; ++t) {
        const int n = p + t + 1, visible = n / kIndexPool;
        std::vector<float> want_scores(static_cast<size_t>(max_blocks), 0.0f);
        for (int b = 0; b < visible && visible > kIndexTopBlocks; ++b) {
            want_scores[static_cast<size_t>(b)] = cpu_score(q.data() + static_cast<size_t>(t) * kIndexHeads * kIndexDim,
                                                            pooled.data() + static_cast<size_t>(b) * kIndexDim);
            ok = ok && std::bit_cast<uint32_t>(want_scores[static_cast<size_t>(b)]) ==
                           std::bit_cast<uint32_t>(scores[static_cast<size_t>(t)][static_cast<size_t>(b)]);
        }
        const List want = cpu_select(want_scores, n);
        if (got.blocks[static_cast<size_t>(t)] != want.blocks || got.info[4 * t + 1] != want.tail ||
            got.info[4 * t + 2] != want.cells) {
            std::printf("  column %d: n %d nsel %zu/%zu tail %d/%d cells %d/%d\n", t, n,
                        got.blocks[static_cast<size_t>(t)].size(), want.blocks.size(), got.info[4 * t + 1], want.tail,
                        got.info[4 * t + 2], want.cells);
            size_t diff = 0;
            for (size_t i = 0; i < std::min(want.blocks.size(), got.blocks[static_cast<size_t>(t)].size()); ++i)
                if (want.blocks[i] != got.blocks[static_cast<size_t>(t)][i]) {
                    diff = i;
                    break;
                }
            std::printf(
                "  first difference at %zu: got %d want %d; score bits ok %d\n", diff,
                got.blocks[static_cast<size_t>(t)].size() > diff ? got.blocks[static_cast<size_t>(t)][diff] : -1,
                want.blocks.size() > diff ? want.blocks[diff] : -1, int(ok));
        }
        ok = ok && got.blocks[static_cast<size_t>(t)] == want.blocks && got.info[4 * t + 1] == want.tail &&
             got.info[4 * t + 2] == want.cells;
    }
    CHECK(ok);
    std::printf("%-62s %s\n", name.c_str(), ok ? "ok" : "FAIL");
    return ok;
}

// n = 2,051 is the last dense position (512 blocks and a tail of 3), 2,052 the first sparse one: 513 blocks, the
// query's own block among them, and exactly the lowest score dropped.
void check_boundary() {
    constexpr int kCtx = 4096, kBlocks = kCtx / 4;
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> pooled(static_cast<size_t>(kBlocks) * kIndexDim);
    for (uint16_t& v : pooled)
        v = to_f16(normal(rng));
    std::vector<float> q(kIndexHeads * kIndexDim);
    for (float& v : q)
        v = normal(rng);
    {  // n = 2,051: position 2,050, 512 blocks and a tail of 3 cells: every cell
        const Selection s = run_select(pooled, q, 2050, 1, kCtx);
        bool ok = s.info[0] == 512 && s.info[1] == 2048 && s.info[2] == 2051;
        for (int b = 0; b < 512; ++b)
            ok = ok && s.blocks[0][static_cast<size_t>(b)] == b;
        CHECK(ok);
        std::printf("%-62s %s\n", "boundary n = 2,051: all 512 blocks and a tail of 3", ok ? "ok" : "FAIL");
    }
    {  // n = 2,052: position 2,051 closes block 512, which competes; exactly the lowest score goes
        std::vector<std::vector<float>> scores;
        const Selection s = run_select(pooled, q, 2051, 1, kCtx, &scores);
        std::vector<float> sc(513);
        for (int b = 0; b < 513; ++b)
            sc[static_cast<size_t>(b)] = cpu_score(q.data(), pooled.data() + static_cast<size_t>(b) * kIndexDim);
        // many blocks score exactly 0 (all four heads negative): the one dropped is the highest index among the lowest
        const List expect = cpu_select(sc, 2052);
        const float lowest_score = *std::min_element(sc.begin(), sc.end());
        int lowest = -1;
        for (int b = 0; b < 513; ++b)
            if (sc[static_cast<size_t>(b)] == lowest_score)
                lowest = b;
        bool ok = s.info[0] == 512 && s.info[1] == 2052 && s.info[2] == 2048;
        std::vector<int> want;
        for (int b = 0; b < 513; ++b)
            if (b != lowest)
                want.push_back(b);
        ok = ok && want == expect.blocks;
        ok = ok && s.blocks[0] == want;
        if (!ok)
            std::printf("  info %d %d %d, %zu blocks (want %zu), lowest %d\n", s.info[0], s.info[1], s.info[2],
                        s.blocks[0].size(), want.size(), lowest);
        CHECK(ok);
        std::printf("%-62s %s\n", "boundary n = 2,052: 513 blocks, the lowest score (own block in) drops",
                    ok ? "ok" : "FAIL");
    }
    {  // the query's own block has the lowest score: it is not attended to
        std::vector<uint16_t> own = pooled;
        std::vector<float> positive(kIndexHeads * kIndexDim, 1.0f);
        for (int b = 0; b < 513; ++b)
            for (int d = 0; d < kIndexDim; ++d)
                own[static_cast<size_t>(b) * kIndexDim + d] =
                    to_f16(b == 512 ? -1.0f : 0.5f + 0.001f * static_cast<float>(b % 7));
        const Selection s = run_select(own, positive, 2051, 1, kCtx);
        bool ok =
            s.blocks[0].size() == 512 && std::find(s.blocks[0].begin(), s.blocks[0].end(), 512) == s.blocks[0].end();
        CHECK(ok);
        std::printf("%-62s %s\n", "own block (closed by the query, lowest score) is excluded", ok ? "ok" : "FAIL");
    }
}

// Pooling against a double reference: each pooled key within an F16 step of mean, norm, rotation.
void check_pool() {
    constexpr int kP = 1002, kColumns = 8, kCtx = 4096;
    std::mt19937 rng(9);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const std::vector<float> q_raw = test::random_floats(static_cast<size_t>(kColumns) * 512, -1, 1, 3),
                             k_raw = test::random_floats(static_cast<size_t>(kColumns) * 128, -2, 2, 4),
                             q_norm = test::random_floats(128, 0.5f, 1.5f, 5),
                             k_norm = test::random_floats(128, 0.5f, 1.5f, 6);
    // the ring already holds the three positions before the window (1,000..1,001 and 999)
    std::vector<uint16_t> ring(kIndexRing * kIndexDim);
    std::vector<std::vector<float>> raw(2000, std::vector<float>(128));
    for (int pos = kP - 3; pos < kP; ++pos)
        for (int d = 0; d < 128; ++d) {
            raw[static_cast<size_t>(pos)][d] = normal(rng);
            ring[static_cast<size_t>(pos % kIndexRing) * 128 + d] = to_f16(raw[static_cast<size_t>(pos)][d]);
        }
    DeviceBuffer<uint16_t> dring(ring), dpooled(static_cast<size_t>(kCtx / 4) * 128);
    DeviceBuffer<float> dq_raw(q_raw), dk_raw(k_raw), dq_norm(q_norm), dk_norm(k_norm),
        dq_out(static_cast<size_t>(kColumns) * 512);
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{kP});
    index_prepare(dq_raw.get(), dk_raw.get(), dq_norm.get(), dq_out.get(), dring.get(), dpos.get(), 64, 1e7f, 1e-6f,
                  kColumns, nullptr);
    index_pool(dring.get(), dk_norm.get(), dpooled.get(), dpos.get(), 64, 1e7f, 1e-6f, kColumns, nullptr);
    const std::vector<uint16_t> pooled = dpooled.download();
    for (int t = 0; t < kColumns; ++t)
        for (int d = 0; d < 128; ++d)
            raw[static_cast<size_t>(kP + t)][d] = from_f16(to_f16(k_raw[static_cast<size_t>(t) * 128 + d]));
    // blocks completed by positions 1,002 .. 1,009: 251 (1,004..1,007) and 252 (1,008..1,011 closes at 1,011: no)
    bool ok = true;
    double worst = 0;
    for (int b = 0; b < kCtx / 4; ++b) {
        const bool completed = 4 * b + 3 >= kP && 4 * b + 3 <= kP + kColumns - 1;
        if (!completed) {
            bool zero = true;
            for (int d = 0; d < 128; ++d)
                zero = zero && pooled[static_cast<size_t>(b) * 128 + d] == 0;
            ok = ok && zero;  // not written: a block the window did not complete
            continue;
        }
        double mean[128], sum = 0;
        for (int d = 0; d < 128; ++d) {
            float s = raw[static_cast<size_t>(4 * b)][d];
            for (int m = 1; m < 4; ++m)
                s += raw[static_cast<size_t>(4 * b + m)][d];
            mean[d] = static_cast<double>(s * 0.25f);
            sum += mean[d] * mean[d];
        }
        const double scale = 1.0 / std::sqrt(sum / 128 + 1e-6);
        double x[128];
        for (int d = 0; d < 128; ++d)
            x[d] = mean[d] * scale * k_norm[d];
        for (int i = 0; i < 32; ++i) {
            const double theta = 4.0 * b * std::pow(1e7, -2.0 * i / 64);
            const double x0 = x[i], x1 = x[i + 32];
            x[i] = x0 * std::cos(theta) - x1 * std::sin(theta);
            x[i + 32] = x0 * std::sin(theta) + x1 * std::cos(theta);
        }
        for (int d = 0; d < 128; ++d) {
            const double got = from_f16(pooled[static_cast<size_t>(b) * 128 + d]);
            worst = std::max(worst, std::fabs(got - x[d]) / (std::fabs(x[d]) * 1e-3 + 1e-3));
        }
    }
    ok = ok && worst < 3;  // within a few F16 steps (rotation angles are float products of large positions)
    CHECK(ok);
    std::printf("%-62s worst %.2f %s\n", "pooling: the blocks a window completes, nothing else", worst,
                ok ? "ok" : "FAIL");
}

// Attention over cell lists. With at most 2,051 visible tokens the list is every position in order and the output is
// bitwise attention_decode's; with more it equals a double-precision reference over the list's cells.
void check_attention(int p, int columns, int rows, bool sparse, uint32_t seed) {
    constexpr int kHeads = 24, kKv = 2, kD = 256, kRow = kKv * kD;
    const int max_blocks = rows / 4;
    const std::vector<float> kh = test::random_floats(static_cast<size_t>(rows) * kRow, -1, 1, seed),
                             vh = test::random_floats(static_cast<size_t>(rows) * kRow, -1, 1, seed + 1),
                             qg = test::random_floats(static_cast<size_t>(columns) * kHeads * 2 * kD, -1, 1, seed + 2);
    std::vector<uint16_t> kc(kh.size()), vc(vh.size());
    for (size_t i = 0; i < kc.size(); ++i)
        kc[i] = to_f16(kh[i]), vc[i] = to_f16(vh[i]);
    // pooled keys and queries only steer the selection
    std::mt19937 rng(seed + 3);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> pooled(static_cast<size_t>(max_blocks) * kIndexDim);
    for (uint16_t& v : pooled)
        v = to_f16(normal(rng));
    std::vector<float> iq(static_cast<size_t>(columns) * kIndexHeads * kIndexDim);
    for (float& v : iq)
        v = normal(rng);
    DeviceBuffer<uint16_t> dkc(kc), dvc(vc), dpooled(pooled);
    DeviceBuffer<float> dqg(qg), diq(iq), dscores(static_cast<size_t>(index_score_floats(rows, columns))),
        out_cells(static_cast<size_t>(columns) * kHeads * kD),
        cells_scratch(static_cast<size_t>(attention_cells_scratch_floats(kHeads, columns)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{p}), blocks(static_cast<size_t>(columns) * kIndexTopBlocks),
        info(static_cast<size_t>(columns) * 4);
    const IndexCells cells{blocks.get(), info.get()};
    index_score(diq.get(), dpooled.get(), dscores.get(), dpos.get(), max_blocks, columns, nullptr);
    index_select(dscores.get(), cells, dpos.get(), max_blocks, columns, nullptr);
    attention_decode_cells(dqg.get(), dqg.get() + kD, 2 * kD, dkc.get(), dvc.get(), out_cells.get(),
                           cells_scratch.get(), dpos.get(), kHeads, kKv, 1.0f / 16.0f, cells, columns, nullptr);
    const std::vector<float> got = out_cells.download();
    bool ok = true;
    std::string what;
    if (!sparse) {
        DeviceBuffer<float> out_dense(got.size()),
            scratch(static_cast<size_t>(attention_scratch_floats(kHeads, rows, columns)));
        attention_decode(dqg.get(), dqg.get() + kD, 2 * kD, dkc.get(), dvc.get(), out_dense.get(), scratch.get(),
                         dpos.get(), kHeads, kKv, 1.0f / 16.0f, rows, columns, nullptr);
        ok = out_dense.download() == got;
        what = "bitwise the dense kernel";
    } else {
        const std::vector<int32_t> b = blocks.download(), inf = info.download();
        double worst = 0;
        for (int t = 0; t < columns; ++t) {
            std::vector<int> cell;
            for (int i = 0; i < inf[4 * t]; ++i)
                for (int m = 0; m < 4; ++m)
                    cell.push_back(b[static_cast<size_t>(t) * kIndexTopBlocks + i] * 4 + m);
            for (int pos = inf[4 * t + 1]; pos <= p + t; ++pos)
                cell.push_back(pos);
            ok = ok && static_cast<int>(cell.size()) == inf[4 * t + 2];
            for (int h = 0; h < kHeads; ++h) {
                const float* q = qg.data() + (static_cast<size_t>(t) * kHeads + h) * 2 * kD;
                std::vector<double> score(cell.size());
                double peak = -1e300;
                for (size_t c = 0; c < cell.size(); ++c) {
                    double dot = 0;
                    for (int d = 0; d < kD; ++d)
                        dot += q[d] * from_f16(kc[static_cast<size_t>(cell[c]) * kRow + (h / 12) * kD + d]);
                    peak = std::max(peak, score[c] = dot / 16.0);
                }
                double sum = 0;
                for (double& x : score)
                    sum += x = std::exp(x - peak);
                for (int d = 0; d < kD; ++d) {
                    double acc = 0, mag = 0;
                    for (size_t c = 0; c < cell.size(); ++c) {
                        const double term =
                            score[c] / sum * from_f16(vc[static_cast<size_t>(cell[c]) * kRow + (h / 12) * kD + d]);
                        acc += term, mag += std::fabs(term);
                    }
                    const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(q[kD + d])));
                    const double want = acc * gate;
                    worst = std::max(worst, std::fabs(got[(static_cast<size_t>(t) * kHeads + h) * kD + d] - want) /
                                                (mag * gate + 1e-30));
                }
            }
        }
        ok = ok && worst < 2e-5;
        what = "double reference over the list (worst " + std::to_string(worst) + ")";
    }
    CHECK(ok);
    std::printf("attention cells at %d, %d columns: %-40s %s\n", p, columns, what.c_str(), ok ? "ok" : "FAIL");
}

}  // namespace

// An attention layer over a prefill chunk of n tokens at positions s .. s + n - 1, in batches of b columns, against
// the same tokens fed in windows of 8: attention outputs, cell lists, K/V rows, pooled keys and the ring, bit for bit.
void check_chunk(int s, int n, int b, int rows, uint32_t seed) {
    constexpr int kHeads = 24, kKv = 2, kD = 256, kRow = kKv * kD, kQ = kHeads * 2 * kD, kOut = kHeads * kD;
    const int max_blocks = rows / 4, nn = n;
    const float scale = 1.0f / 16.0f, base = 1e7f, eps = 1e-6f;
    const auto floats = [&](size_t count, float lo, float hi) { return test::random_floats(count, lo, hi, ++seed); };
    const auto halves = [&](size_t count) {
        std::vector<float> f = floats(count, -1, 1);
        std::vector<uint16_t> h(count);
        for (size_t i = 0; i < count; ++i)
            h[i] = to_f16(f[i]);
        return h;
    };
    // the state before the chunk: K/V rows, pooled keys and ring rows of the earlier positions (the rest is zero)
    std::vector<uint16_t> kc = halves(static_cast<size_t>(rows) * kRow), vc = halves(kc.size());
    std::vector<uint16_t> pooled = halves(static_cast<size_t>(max_blocks) * kIndexDim), ring = halves(16 * 128);
    std::fill(kc.begin() + static_cast<int64_t>(s) * kRow, kc.end(), 0), std::fill(vc.begin() + s * kRow, vc.end(), 0);
    std::fill(pooled.begin() + static_cast<int64_t>(s / 4) * kIndexDim, pooled.end(), 0);
    const std::vector<float> q = floats(static_cast<size_t>(nn) * kQ, -1, 1),
                             k = floats(static_cast<size_t>(nn) * kRow, -1, 1),
                             v = floats(static_cast<size_t>(nn) * kRow, -1, 1),
                             iq = floats(static_cast<size_t>(nn) * 512, -1, 1),
                             ik = floats(static_cast<size_t>(nn) * 128, -2, 2), qw = floats(kD, 0.5f, 1.5f),
                             kw = floats(kD, 0.5f, 1.5f), qn = floats(128, 0.5f, 1.5f), kn = floats(128, 0.5f, 1.5f);
    std::vector<int32_t> positions(static_cast<size_t>(nn));
    std::iota(positions.begin(), positions.end(), s);
    DeviceBuffer<int32_t> dpos(positions);
    DeviceBuffer<float> dqw(qw), dkw(kw), dqn(qn), dkn(kn), dv(v), diq(iq), dik(ik);
    struct Result {
        std::vector<float> out;
        std::vector<uint16_t> kc, vc, pooled, ring;
        std::vector<int32_t> blocks, info;  // per column: the list's selected blocks, then nsel, tail, cells
    };
    const auto run = [&](bool chunk) {
        DeviceBuffer<float> dq(q), dk(k), dout(static_cast<size_t>(nn) * kOut), dqi(static_cast<size_t>(nn) * 512);
        DeviceBuffer<uint16_t> dkc(kc), dvc(vc), dpooled(pooled), dring(ring), draw(static_cast<size_t>(nn) * 128);
        const int width = chunk ? b : kMaxColumns;
        DeviceBuffer<float> dscores(static_cast<size_t>(index_score_floats(rows, width))),
            dscratch(static_cast<size_t>(attention_cells_scratch_floats(kHeads, width)));
        DeviceBuffer<int32_t> blocks(static_cast<size_t>(width) * kIndexTopBlocks),
            info(static_cast<size_t>(width) * 4);
        const IndexCells cells{blocks.get(), info.get()};
        Result r;
        r.blocks.assign(static_cast<size_t>(nn) * kIndexTopBlocks, -1), r.info.assign(static_cast<size_t>(nn) * 4, 0);
        const auto batch = [&](int c0, int w) {  // scoring, selection and attention for columns c0 .. c0 + w - 1
            const int32_t* p = dpos.get() + c0;
            index_score(dqi.get() + int64_t{c0} * 512, dpooled.get(), dscores.get(), p, max_blocks, w, nullptr);
            index_select(dscores.get(), cells, p, max_blocks, w, nullptr);
            float* qc = dq.get() + int64_t{c0} * kQ;
            attention_decode_cells(qc, qc + kD, 2 * kD, dkc.get(), dvc.get(), dout.get() + int64_t{c0} * kOut,
                                   dscratch.get(), p, kHeads, kKv, scale, cells, w, nullptr);
            const std::vector<int32_t> bl = blocks.download(), in = info.download();
            for (int t = 0; t < w; ++t) {
                const int nsel = in[static_cast<size_t>(t) * 4];
                std::copy_n(bl.begin() + t * kIndexTopBlocks, nsel, r.blocks.begin() + (c0 + t) * kIndexTopBlocks);
                std::copy_n(in.begin() + t * 4, 3, r.info.begin() + (c0 + t) * 4);  // the fourth is unused
            }
        };
        if (chunk) {
            qk_norm_rope_store(dq.get(), 2 * kD, dqw.get(), dk.get(), dkw.get(), dv.get(), dkc.get(), dvc.get(), kHeads,
                               kKv, 64, dpos.get(), base, eps, nn, nullptr);
            index_prepare_chunk(diq.get(), dik.get(), dqn.get(), dqi.get(), draw.get(), dpos.get(), 64, base, eps, nn,
                                nullptr);
            index_pool_chunk(dring.get(), draw.get(), dkn.get(), dpooled.get(), dpos.get(), 64, base, eps, nn, nullptr);
            for (int c0 = 0; c0 < nn; c0 += b)
                batch(c0, std::min(b, nn - c0));
        } else {
            for (int c0 = 0; c0 < nn; c0 += kMaxColumns) {
                const int w = std::min(kMaxColumns, nn - c0);
                const int32_t* p = dpos.get() + c0;
                qk_norm_rope_store(dq.get() + int64_t{c0} * kQ, 2 * kD, dqw.get(), dk.get() + int64_t{c0} * kRow,
                                   dkw.get(), dv.get() + int64_t{c0} * kRow, dkc.get(), dvc.get(), kHeads, kKv, 64, p,
                                   base, eps, w, nullptr);
                index_prepare(diq.get() + int64_t{c0} * 512, dik.get() + int64_t{c0} * 128, dqn.get(),
                              dqi.get() + int64_t{c0} * 512, dring.get(), p, 64, base, eps, w, nullptr);
                index_pool(dring.get(), dkn.get(), dpooled.get(), p, 64, base, eps, w, nullptr);
                batch(c0, w);
            }
        }
        r.out = dout.download(), r.kc = dkc.download(), r.vc = dvc.download(), r.pooled = dpooled.download();
        r.ring = dring.download();
        return r;
    };
    const Result w = run(false), c = run(true);
    const bool same = w.out == c.out && w.kc == c.kc && w.vc == c.vc && w.pooled == c.pooled && w.ring == c.ring &&
                      w.blocks == c.blocks && w.info == c.info;
    CHECK(same);
    if (!same) {  // which part differs, and the first column or row
        const auto first = [](const auto& a, const auto& b2, size_t per) {
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i] != b2[i])
                    return static_cast<long>(i / per);
            return -1L;
        };
        std::printf(
            "  differs: out col %ld, k row %ld, v row %ld, pooled block %ld, ring row %ld, blocks col %ld, info col "
            "%ld\n",
            first(w.out, c.out, kOut), first(w.kc, c.kc, kRow), first(w.vc, c.vc, kRow),
            first(w.pooled, c.pooled, kIndexDim), first(w.ring, c.ring, kIndexDim),
            first(w.blocks, c.blocks, kIndexTopBlocks), first(w.info, c.info, 4));
    }
    std::printf("%-62s %s\n",
                ("chunk of " + std::to_string(nn) + " at " + std::to_string(s) + ", batches of " + std::to_string(b) +
                 ": bitwise windows of 8")
                    .c_str(),
                same ? "ok" : "FAIL");
}

int main() {
    try {
        check_pool();
        check_boundary();
        int seed = 100;
        for (int p : {2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 3001, 4087})
            for (int columns : {1, 3, 8})
                if (p + columns <= 4096)
                    check_window("select window at " + std::to_string(p) + ", " + std::to_string(columns) + " columns",
                                 p, columns, 4096, false, static_cast<uint32_t>(++seed));
        for (int p : {2051, 2600, 4000})
            check_window("select with ties and zero scores at " + std::to_string(p), p, 4, 4096, true,
                         static_cast<uint32_t>(++seed));
        check_window("select 262,144 context, position 200,000", 200000, 8, 262144, false, 9);
        for (int p : {0, 3, 127, 128, 1000, 2040})
            for (int columns : {1, 4, 8})
                check_attention(p, columns, 2048, false, static_cast<uint32_t>(++seed));
        for (int p : {2047, 2050, 3000})
            for (int columns : {1, 8})
                check_attention(p, columns, 4096, p + columns > 2051, static_cast<uint32_t>(++seed));
        // prefill chunks: the start, across 2,051 / 2,052, at every offset in a 4-token block, one column, big batches
        for (const auto& [s, n, b] : std::vector<std::array<int, 3>>{{0, 37, 16},
                                                                     {0, 300, 128},
                                                                     {2000, 100, 64},
                                                                     {2049, 13, 8},
                                                                     {2002, 70, 33},
                                                                     {3003, 64, 64},
                                                                     {4093, 300, 128},
                                                                     {5000, 1, 1},
                                                                     {5000, 2048, 128}})
            check_chunk(s, n, b, 8192, static_cast<uint32_t>(++seed));
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "indexer_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
