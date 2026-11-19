// The generation loop (milestone M7): requests over one Model and its draft layer, greedy.
//
// A request's prompt goes as prefill chunks (Options::prefill_columns) or as windows; generated tokens go one at a
// time, or speculatively as windows of the last token and the draft layer's drafts, keeping the longest prefix the
// model's own argmax agrees with, so the tokens are the same either way. A request whose prompt extends the sequence so
// far reuses it (sessions): with prefill it goes back to the end of the last prefill chunk (Model::rewind) and prefills
// the tokens fed since with its new ones; without, it feeds only its new tokens. A configured prefix cache can instead
// restore the deepest saved boundary across requests. Generation stops after max_tokens, after a stop id, or when the
// caller's on_tokens returns false; Model::cancel() stops it between windows from any thread. Tokens are the greedy
// argmax, or sampled (kernels/sampling/sample.h): a sampled request draws each token at its position with its seed, so
// speculation gives the same tokens as one token at a time, and its penalties count only the tokens this request
// generated.
//
// `torch run` (tools/run.cpp) is a command line over this class; tests/model and the M6 checks drive it there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "kernels/sampling/sample.h"
#include "model/draft.h"
#include "model/model.h"
#include "model/window_policy.h"
#include "runtime/prefix_cache.h"

namespace torch::runtime {

struct Request {
    std::vector<int32_t> prompt;
    int max_tokens = 16;
    std::vector<int32_t> stop_ids;  // generation stops after any of these (the stop id is kept)
    // Speculation: up to `spec` drafts per verify window (0: one token at a time; needs the draft layer) and how many a
    // window verifies: "fixed" (all), "adaptive", "adaptive-stop" or "adaptive-tight" (model/window_policy.h).
    int spec = 0;
    std::string policy = "adaptive-stop";
    // The adaptive policy keeps what it learned (acceptance by draft probability, window costs) from the last request
    // with the same spec on this generator, instead of starting from its seeds.
    bool carry_policy = false;
    int draft_depth = 3;          // drafts per generated token without speculation (acceptance measurements)
    int swaps_after_prompt = -1;  // Model::set_swaps_per_window once the prompt is in (-1: unchanged)
    // Sampling instead of the argmax: the parameters (their seed is ignored), and the seed (a random one when absent,
    // returned in Result::seed). The vocabulary's padding ids are never drawn.
    bool sample = false;
    kernels::SamplingParams sampling;
    std::optional<uint64_t> seed;
    // Without prefill the prompt goes in windows of `window` tokens; tests keep only `keep` of each window from
    // position `accept_from` on (Model::accept).
    int window = 1, keep = model::kMaxColumns, accept_from = 0;
    // Sessions for chat (M7 review 1.1-1.2): the prompt's first `stable` tokens go as prefill chunks ending exactly
    // there (0: the whole prompt), the rest as windows; a turn that renders the history the same way rewinds to it.
    // rewind_only: a request rewinds to the end of the last prefill or starts again, never extends.
    size_t stable = 0;
    std::vector<std::pair<size_t, PrefixKind>> checkpoints;
    bool rewind_only = false;
    int cancel_after = 0;         // tests: Model::cancel() after this many generated windows (0: never)
    int cancel_prompt_after = 0;  // tests: Model::cancel() after this many prompt windows or chunks (0: never)
};

// Where a request started: the first position it fed, the model's position before it, and whether it went back to
// the end of the last prefill (rewound) or started a new sequence (reset).
struct Resume {
    size_t at = 0, fed = 0;
    bool rewound = false, reset = false;
};

// One draft variant's passes over a request: the KV-only pass at prompt positions and the drafts made at generated
// ones without speculation (drafts[s] after the window of step s).
struct DraftStats {
    double kv_ms = 0, draft_ms = 0;
    int kv_positions = 0, draft_steps = 0;
    std::vector<std::vector<model::Drafted>> drafts;
};

struct Result {
    std::vector<int32_t> sequence;  // the prompt and the generated tokens
    Resume resume;
    PrefixStats prefix;
    bool cancelled = false;                 // Model::cancel() stopped the request between windows
    bool stopped = false;                   // a stop id or the caller's on_tokens ended it
    int stop_window = 0, stop_column = -1;  // the tokens of the window it stopped in, and the stop token's place
    uint64_t seed = 0;                      // a sampled request's seed
    double prompt_s = 0;                    // the prompt's windows, the draft layer's passes over them included
    double decode_ms = 0;                   // one token at a time: the generated tokens' windows
    int decoded = 0;
    // The speculative loop: its windows, their verify-and-accept and draft times, the tokens they kept, the drafts they
    // verified, and the last position the draft layer extended over (-1: none).
    double verify_ms = 0, extend_ms = 0, spec_ms = 0;
    int windows = 0, window_tokens = 0, verified_drafts = 0, extended = -1;
    // By draft depth j: the windows that verified draft j with every earlier draft accepted, and that accepted it.
    std::vector<int> depth_tried, depth_accepted;
    std::vector<DraftStats> variants;
};

// Callbacks during a request, on the calling thread.
struct Observer {
    std::function<void(const Resume&)> on_start;  // before the first window
    // After each prompt window or chunk and each one-token window, while Model::logits and argmax hold its results:
    // positions first .. first + kept - 1 of sequence(), of a window `width` wide (a chunk has its last position's).
    std::function<void(size_t first, size_t kept, size_t width, bool chunked, double ms)> on_window;
    std::function<void()> on_prompt_end;                      // once, when the prompt is in
    std::function<bool(std::span<const int32_t>)> on_tokens;  // new generated tokens; false stops the request
    // After each window and prefill chunk: the positions fed so far and the prompt's length; false cancels the request
    // before its next window.
    std::function<bool(size_t fed, size_t prompt)> on_progress;
};

class Generator {
public:
    // `draft` may be null (no speculation); its first `variants` variants run after every window (the caller may keep
    // more for its own checks). prefill_columns is the model's Options::prefill_columns.
    Generator(model::Model& model, model::Draft* draft, int variants, int prefill_columns,
              const PrefixOptions& options = {});
    void forget();

    // Runs one request. Throws Error when the model fails; a cancel returns with Result::cancelled.
    Result generate(const Request& request, const Observer& observer = {});

    // The sequence of the request running or last run: its prompt and the tokens generated so far.
    const std::vector<int32_t>& sequence() const noexcept { return sequence_; }

private:
    model::Model& m_;
    model::Draft* draft_;
    int variants_, prefill_columns_;
    std::vector<int32_t> sequence_;
    std::unique_ptr<PrefixCache> prefix_;
    std::optional<model::WindowPolicy> policy_;  // the last request's, for Request::carry_policy
};

}  // namespace torch::runtime
