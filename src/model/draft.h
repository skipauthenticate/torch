// The draft layer (architecture decision 12), one column at a time after the trunk's windows, as CUDA graphs: the
// drafter of milestone M4's speculative loop (torch run --spec) and of its acceptance measurement (R3).
//
// Draft owns everything it runs on: the draft file's weights in an arena of their own with all routed experts resident
// (DraftWeights), the output-head rows its variants score, a K/V cache per variant and its own activations. From the
// trunk it reads only the 4-stream residual after the head mixer's combine (H, Model::residual), the token embedding
// and the output head (Model::weights), and it runs on the trunk's stream, so the trunk's state is never written.
//
// The pass for (H_t, x_{t+1}) at position t, as the model's MTP layer defines it: S_s = fc_hidden(hnorm(H_t)_s) +
// fc_embedding(enorm(embedding of x_{t+1})) for each stream s, then one attention + mixture-of-experts block with its
// hyper-connection mixers, the head's mixer and the shared output head. Depth j >= 2 runs the same pass from the
// block's own state S' and the previous draft at position t + j - 1; those rows of the K/V cache are rewritten before
// they are read. At prompt positions the KV-only pass writes row t from S's attention mixer and K/V projections only.
//
// The draft's attention is dense over the last kDraftWindow positions at most: each K/V cache is a ring of
// min(context, kDraftWindow) rows (row = position % rows). Below kDraftWindow positions that is every earlier
// position, exactly as a full cache. Past it, a ring row may still hold a draft written ahead of the last accepted
// position (the oldest position of the window loses its row) until that position is computed again; drafts are
// verified, so only acceptance can change.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "core/budget.h"

namespace torch::model {

inline constexpr int kDraftWindow = 16384;  // positions the draft layer attends to at most

class Model;

// How the draft layer runs: how hnorm normalizes the 4-stream hidden state, and which rows of the output head the
// draft argmax scores. Each variant keeps its own K/V cache.
struct DraftVariant {
    bool pooled_hnorm = false;  // one RMS over all hc x n_embd values, instead of one per stream
    int head_rows = 0;          // the first head_rows ids of the draft vocabulary; 0: the full head
};

// A drafted token and the draft layer's softmax probability for it (over the rows its variant scores).
struct Drafted {
    int32_t token;
    float prob;
};

class Draft {
public:
    // The memory a Draft of these variants takes, as a budget line: pass it in Options::reserve so the trunk's top-up
    // leaves room for it. Throws Error when a file cannot be read or is not the supported one.
    static Reservation reservation(const std::filesystem::path& model, const std::filesystem::path& file,
                                   const std::vector<DraftVariant>& variants, int context);

    // Loads the draft file for `trunk`, whose sequence it drafts from. vocab: token ids, most frequent first, at least
    // as many as the largest head_rows. Throws Error when the file or the vocabulary is not usable, or memory is short.
    Draft(const Model& trunk, const std::filesystem::path& file, std::vector<DraftVariant> variants,
          std::vector<int32_t> vocab, int context);
    Draft(const Draft&) = delete;
    Draft& operator=(const Draft&) = delete;
    ~Draft();

    // Extends over the n = next.size() positions t to t + n - 1 that the trunk's last window computed in its columns
    // 0 to n - 1 (H_{t+i} is Model::residual(i)), next[i] being the token at t + i + 1. Depth 0: the KV-only pass at
    // every one of them, which writes the variant's K/V rows t to t + n - 1 and nothing else. Depth D >= 1: the KV-only
    // pass at the first n - 1, then the full pass at the last and D - 1 more from the block's own state, at positions
    // t + n - 1 to t + n + D - 2, returning the D drafted tokens. Rows above t + n - 1 hold drafts until a later call
    // at those positions rewrites them. Waits for the GPU. Throws Error on failure.
    // more(drafted, prob), when given, is called after each draft (drafted counts from 1) and ends the drafting by
    // returning false: fewer drafts are returned, and the K/V rows of the positions not reached keep their old
    // contents (a later call at those positions rewrites them).
    void run(int variant, int position, std::span<const int32_t> next, int depth, std::vector<Drafted>& out,
             const std::function<bool(int, float)>& more = nullptr);

    // The first of the first `rows` K/V ring rows (at most min(context, kDraftWindow)) in which variants a and b
    // differ (keys or values, bitwise), or -1 when they are equal: for tests. Waits for the GPU.
    int first_kv_difference(int a, int b, int rows, int first = 0) const;
    // The ring rows before position end-1. The last row depends on the next request's first new token.
    void checkpoint_rows(int end, int variants, std::vector<std::span<uint8_t>>& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch::model
