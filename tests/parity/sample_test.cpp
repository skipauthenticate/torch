// Tests of the window sampler (src/kernels/sampling/sample.h): exact cases (greedy with penalties and ties, top-k 1,
// drafts that always or never pass), chi-square tests of the sampled and accepted token distributions and of the
// acceptance rate against a double-precision reference over thousands of seeds, and the time per window at the model's
// vocabulary size.
#include "kernels/sampling/sample.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "../unit/check.h"
#include "device.h"
#include "reference.h"

using namespace torch;
using kernels::SamplingParams;
using test::DeviceBuffer;

namespace {

// The column's history counts: counts plus the drafts that precede it.
std::vector<int> column_counts(const std::vector<int>& counts, const std::vector<int>& drafts, int column) {
    std::vector<int> n = counts;
    for (int j = 0; j < column; ++j)
        ++n[drafts[j]];
    return n;
}

// Steps 1-2 of the header, in float like the kernel.
std::vector<float> adjusted(const std::vector<float>& logits, const std::vector<int>& n, const SamplingParams& p) {
    std::vector<float> x = logits;
    for (size_t v = 0; v < x.size(); ++v) {
        if (n[v] > 0) {
            if (p.repetition_penalty != 1.0f)
                x[v] = x[v] > 0.0f ? x[v] / p.repetition_penalty : x[v] * p.repetition_penalty;
            x[v] = x[v] - p.frequency_penalty * static_cast<float>(n[v]);
            x[v] = x[v] - p.presence_penalty;
        }
        if (p.temperature != 0.0f)
            x[v] = x[v] / p.temperature;
    }
    return x;
}

int argmax_lowest(const std::vector<float>& x) {
    int best = 0;
    for (size_t v = 1; v < x.size(); ++v)
        if (x[v] > x[best])
            best = static_cast<int>(v);
    return best;
}

// Steps 3-6 in double: the distribution q of a column.
std::vector<double> distribution(const std::vector<float>& logits, const std::vector<int>& n, const SamplingParams& p) {
    const std::vector<float> x = adjusted(logits, n, p);
    const size_t vocab = x.size();
    const double peak = *std::max_element(x.begin(), x.end());
    std::vector<double> prob(vocab);
    double sum = 0;
    for (size_t v = 0; v < vocab; ++v)
        sum += prob[v] = std::exp(x[v] - peak);
    for (double& q : prob)
        q /= sum;
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prob[a] > prob[b]; });
    std::vector<bool> keep(vocab, true);
    if (p.top_k > 0)
        for (size_t r = static_cast<size_t>(p.top_k); r < vocab; ++r)
            keep[order[r]] = false;
    if (p.top_p < 1.0f) {
        double cumulative = 0;
        size_t r = 0;
        for (; r < vocab; ++r) {
            cumulative += prob[order[r]];
            if (cumulative >= p.top_p)
                break;
        }
        for (++r; r < vocab; ++r)
            keep[order[r]] = false;
    }
    if (p.min_p > 0.0f)
        for (size_t v = 0; v < vocab; ++v)
            if (prob[v] < p.min_p * prob[order[0]])
                keep[v] = false;
    double kept = 0;
    for (size_t v = 0; v < vocab; ++v)
        kept += keep[v] ? prob[v] : 0.0;
    for (size_t v = 0; v < vocab; ++v)
        prob[v] = keep[v] ? prob[v] / kept : 0.0;
    return prob;
}

struct Window {
    int vocab, columns;
    std::vector<float> logits;  // [columns][vocab]
    std::vector<int> counts, drafts;
};

struct Outcome {
    std::vector<int32_t> sampled, tokens;
    int32_t accepted;
};

// Runs the window once on the device.
Outcome run(const Window& w, SamplingParams params, int position, DeviceBuffer<float>& dlogits,
            DeviceBuffer<int32_t>& dcounts, DeviceBuffer<int32_t>& ddrafts) {
    DeviceBuffer<float> scratch(static_cast<size_t>(kernels::sampling_scratch_floats(w.vocab, w.columns)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{position}), sampled(static_cast<size_t>(w.columns)), accepted(1),
        tokens(static_cast<size_t>(w.columns));
    kernels::sample_columns(dlogits.get(), w.vocab, w.columns, dcounts.get(), ddrafts.get(), params, dpos.get(),
                            scratch.get(), sampled.get(), nullptr);
    kernels::accept_drafts(sampled.get(), ddrafts.get(), w.columns, accepted.get(), tokens.get(), nullptr);
    return {sampled.download(), tokens.download(), accepted.download()[0]};
}

Window make_window(int vocab, int columns, uint32_t seed) {
    Window w{vocab, columns, test::random_floats(static_cast<size_t>(vocab) * columns, -4, 4, seed),
             std::vector<int>(static_cast<size_t>(vocab), 0),
             std::vector<int>(static_cast<size_t>(std::max(columns - 1, 1)))};
    for (int v = 0; v < vocab; v += 7)
        w.counts[static_cast<size_t>(v)] = 1 + v % 3;
    return w;
}

std::vector<float> column_logits(const Window& w, int c) {
    return std::vector<float>(w.logits.begin() + static_cast<size_t>(c) * w.vocab,
                              w.logits.begin() + static_cast<size_t>(c + 1) * w.vocab);
}

void upload_all(const Window& w, DeviceBuffer<float>& dl, DeviceBuffer<int32_t>& dc, DeviceBuffer<int32_t>& dd) {
    dl.upload(w.logits);
    dc.upload(std::vector<int32_t>(w.counts.begin(), w.counts.end()));
    dd.upload(std::vector<int32_t>(w.drafts.begin(), w.drafts.end()));
}

// ---- Exact cases -------------------------------------------------------------------------------------------------

void check_greedy() {
    Window w = make_window(3000, 3, 11);
    w.drafts = {5, 17};
    // a tie for the largest value of column 1: the lower index must win
    std::vector<float> col1 = column_logits(w, 1);
    col1[900] = col1[2500] = 50.0f;
    std::copy(col1.begin(), col1.end(), w.logits.begin() + w.vocab);
    SamplingParams p;
    p.temperature = 0.0f;
    p.repetition_penalty = 1.3f;
    p.presence_penalty = 0.4f;
    p.frequency_penalty = 0.2f;
    DeviceBuffer<float> dl(w.logits.size());
    DeviceBuffer<int32_t> dc(w.counts.size()), dd(w.drafts.size());
    upload_all(w, dl, dc, dd);
    const Outcome out = run(w, p, 100, dl, dc, dd);
    bool same = true;
    for (int c = 0; c < 3; ++c)
        same = same &&
               out.sampled[c] == argmax_lowest(adjusted(column_logits(w, c), column_counts(w.counts, w.drafts, c), p));
    CHECK(same);
    CHECK(out.sampled[1] == 900);
    std::printf("%-52s %s\n", "greedy: penalties, history of drafts, ties",
                same && out.sampled[1] == 900 ? "ok" : "FAIL");
}

void check_certain_and_impossible() {
    Window w = make_window(3000, 3, 12);
    SamplingParams p;
    p.top_k = 1;
    p.temperature = 0.8f;
    p.presence_penalty = 0.3f;
    // drafts that are each column's top token: always accepted, whatever the seed
    std::vector<int> top(3);
    w.drafts = {0, 0};
    for (int c = 0; c < 3; ++c) {
        top[c] = argmax_lowest(adjusted(column_logits(w, c), column_counts(w.counts, w.drafts, c), p));
        if (c < 2)
            w.drafts[c] = top[c];
    }
    DeviceBuffer<float> dl(w.logits.size());
    DeviceBuffer<int32_t> dc(w.counts.size()), dd(w.drafts.size());
    upload_all(w, dl, dc, dd);
    bool always = true;
    for (uint64_t seed = 1; seed <= 200; ++seed) {
        p.seed = seed;
        const Outcome out = run(w, p, 7, dl, dc, dd);
        always = always && out.accepted == 3 && out.tokens[0] == top[0] && out.tokens[1] == top[1] &&
                 out.tokens[2] == top[2];
    }
    CHECK(always);
    std::printf("%-52s %s\n", "top_k 1, drafts = the top tokens: always accepted", always ? "ok" : "FAIL");
    // a first draft outside the top 5 (probability 0): the window ends after column 0
    p.top_k = 5;
    w.drafts[0] = (top[0] + 1) % 3000;
    upload_all(w, dl, dc, dd);
    const std::vector<double> q = distribution(column_logits(w, 0), column_counts(w.counts, w.drafts, 0), p);
    CHECK(q[static_cast<size_t>(w.drafts[0])] == 0.0);
    bool never = true;
    for (uint64_t seed = 1; seed <= 200; ++seed) {
        p.seed = seed;
        const Outcome out = run(w, p, 7, dl, dc, dd);
        never = never && out.accepted == 1 && out.tokens[1] == -1 && q[static_cast<size_t>(out.tokens[0])] > 0.0;
    }
    CHECK(never);
    std::printf("%-52s %s\n", "top_k 5, a draft with p = 0: always rejected", never ? "ok" : "FAIL");
}

// ---- Statistics --------------------------------------------------------------------------------------------------

// Chi-square of observed counts against expected probabilities; bins with fewer than 5 expected are pooled. Returns
// the statistic minus its degrees of freedom, in units of the standard deviation sqrt(2 df) (about N(0, 1)).
double chi_square_z(const std::vector<double>& observed, const std::vector<double>& q, double total) {
    double stat = 0, pooled_observed = 0, pooled_expected = 0;
    int bins = 0;
    for (size_t v = 0; v < q.size(); ++v) {
        const double expected = q[v] * total;
        if (expected < 5) {
            pooled_observed += observed[v], pooled_expected += expected;
            continue;
        }
        stat += (observed[v] - expected) * (observed[v] - expected) / expected;
        ++bins;
    }
    if (pooled_expected >= 5) {
        stat += (pooled_observed - pooled_expected) * (pooled_observed - pooled_expected) / pooled_expected;
        ++bins;
    }
    if (bins < 2)
        return 0;  // a single outcome: nothing to test
    const double df = bins - 1;
    return (stat - df) / std::sqrt(2 * df);
}

void check_statistics(const std::string& name, const SamplingParams& base, int vocab, int seeds) {
    constexpr int kColumns = 3;
    Window w = make_window(vocab, kColumns, 33);
    // the drafts: each column's second most likely token, so the windows are cut often
    w.drafts = {0, 0};
    for (int c = 0; c + 1 < kColumns; ++c) {
        const std::vector<double> q = distribution(column_logits(w, c), column_counts(w.counts, w.drafts, c), base);
        std::vector<int> order(static_cast<size_t>(vocab));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return q[a] > q[b]; });
        w.drafts[static_cast<size_t>(c)] = order[1];
    }
    DeviceBuffer<float> dl(w.logits.size());
    DeviceBuffer<int32_t> dc(w.counts.size()), dd(w.drafts.size());
    upload_all(w, dl, dc, dd);
    DeviceBuffer<float> scratch(static_cast<size_t>(kernels::sampling_scratch_floats(vocab, kColumns)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{42}), sampled(static_cast<size_t>(kColumns) * seeds),
        accepted(static_cast<size_t>(seeds)), tokens(static_cast<size_t>(kColumns) * seeds);
    for (int s = 0; s < seeds; ++s) {
        SamplingParams p = base;
        p.seed = 1000 + static_cast<uint64_t>(s);
        // each seed writes its own slots of the outputs
        kernels::sample_columns(dl.get(), vocab, kColumns, dc.get(), dd.get(), p, dpos.get(), scratch.get(),
                                sampled.get() + static_cast<size_t>(s) * kColumns, nullptr);
        kernels::accept_drafts(sampled.get() + static_cast<size_t>(s) * kColumns, dd.get(), kColumns,
                               accepted.get() + s, tokens.get() + static_cast<size_t>(s) * kColumns, nullptr);
    }
    const std::vector<int32_t> all_sampled = sampled.download(), all_tokens = tokens.download(),
                               all_accepted = accepted.download();
    double worst_z = 0, worst_accept_z = 0;
    bool identical = true;
    for (int c = 0; c < kColumns; ++c) {
        const std::vector<double> q = distribution(column_logits(w, c), column_counts(w.counts, w.drafts, c), base);
        std::vector<double> plain(static_cast<size_t>(vocab), 0.0), spec(plain);
        double reached = 0, accepts = 0;
        for (int s = 0; s < seeds; ++s) {
            plain[static_cast<size_t>(all_sampled[static_cast<size_t>(s) * kColumns + c])] += 1;
            if (c < all_accepted[static_cast<size_t>(s)]) {  // the window got to column c
                const int token = all_tokens[static_cast<size_t>(s) * kColumns + c];
                spec[static_cast<size_t>(token)] += 1;
                identical = identical && token == all_sampled[static_cast<size_t>(s) * kColumns + c];
                reached += 1;
                accepts += c + 1 < kColumns && all_accepted[static_cast<size_t>(s)] > c + 1;
            }
        }
        worst_z =
            std::max({worst_z, std::fabs(chi_square_z(plain, q, seeds)), std::fabs(chi_square_z(spec, q, reached))});
        if (c + 1 < kColumns) {  // the acceptance rate is q_c(draft_c)
            const double rate = q[static_cast<size_t>(w.drafts[static_cast<size_t>(c)])];
            worst_accept_z =
                std::max(worst_accept_z, std::fabs(accepts - rate * reached) / std::sqrt(reached * rate * (1 - rate)));
        }
    }
    // at 6 sigma a correct sampler fails about once in 10^9 runs
    const bool ok = worst_z < 6 && worst_accept_z < 6 && identical;
    CHECK(ok);
    std::printf("%-52s worst chi-square z %.2f, acceptance z %.2f, tokens = plain: %s  %s\n", name.c_str(), worst_z,
                worst_accept_z, identical ? "yes" : "NO", ok ? "ok" : "FAIL");
}

// ---- Timing ------------------------------------------------------------------------------------------------------

void time_window(const std::string& name, const SamplingParams& p, int columns) {
    const int vocab = 248320;
    Window w = make_window(vocab, columns, 77);
    w.logits = test::random_floats(w.logits.size(), -8, 8, 78);
    for (int c = 0; c + 1 < columns; ++c)
        w.drafts[static_cast<size_t>(c)] = 100 * (c + 1);
    DeviceBuffer<float> dl(w.logits.size());
    DeviceBuffer<int32_t> dc(w.counts.size()), dd(w.drafts.size());
    upload_all(w, dl, dc, dd);
    DeviceBuffer<float> scratch(static_cast<size_t>(kernels::sampling_scratch_floats(vocab, columns)));
    DeviceBuffer<int32_t> dpos(std::vector<int32_t>{9}), sampled(static_cast<size_t>(columns)), accepted(1),
        tokens(static_cast<size_t>(columns));
    const auto once = [&] {
        kernels::sample_columns(dl.get(), vocab, columns, dc.get(), dd.get(), p, dpos.get(), scratch.get(),
                                sampled.get(), nullptr);
        kernels::accept_drafts(sampled.get(), dd.get(), columns, accepted.get(), tokens.get(), nullptr);
    };
    once();
    TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    constexpr int kReps = 50;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i)
        once();
    TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / kReps;
    std::printf("%-52s %8.1f us per window (T = %d, 248,320 logits per column)\n", name.c_str(), us, columns);
}

}  // namespace

int main() {
    try {
        check_greedy();
        check_certain_and_impossible();
        SamplingParams plain;
        plain.temperature = 1.3f;
        check_statistics("temperature 1.3", plain, 400, 30000);
        SamplingParams filtered;
        filtered.temperature = 0.8f;
        filtered.top_k = 40;
        filtered.top_p = 0.9f;
        filtered.min_p = 0.02f;
        filtered.repetition_penalty = 1.1f;
        filtered.presence_penalty = 0.3f;
        filtered.frequency_penalty = 0.1f;
        check_statistics("temperature 0.8, top-k 40, top-p 0.9, min-p, penalties", filtered, 400, 30000);
        SamplingParams nucleus;
        nucleus.temperature = 1.0f;
        nucleus.top_p = 0.7f;
        check_statistics("top-p 0.7", nucleus, 400, 30000);
        SamplingParams only_k;
        only_k.top_k = 25;
        check_statistics("top-k 25", only_k, 400, 30000);
        for (int columns : {1, 4, 8}) {
            time_window("full filters and penalties", filtered, columns);
            time_window("temperature only", plain, columns);
        }
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sample_test: %s\n", e.what());
        return 1;
    }
    return test::test_exit_code();
}
