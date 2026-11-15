#include "runtime/generator.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <random>

#include "core/error.h"
#include "model/window_policy.h"

namespace torch::runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

Generator::Generator(model::Model& model, model::Draft* draft, int variants, int prefill_columns)
    : m_(model), draft_(draft), variants_(draft ? variants : 0), prefill_columns_(prefill_columns) {}

Result Generator::generate(const Request& r, const Observer& o) {
    const std::vector<int32_t>& prompt = r.prompt;
    std::vector<int32_t>& sequence = sequence_;
    Result out;
    // A request whose prompt extends the sequence so far (the tokens fed to the model) feeds only its new tokens; any
    // other prompt starts a new sequence. Sessions, option B: with prefill, back to the end of the last prefill, so the
    // tokens fed since (generated ones, and a speculative window's columns past the sequence) are prefilled again with
    // the new ones, as a fresh prefill of the whole prompt would feed them. With rewind_only (chat) a request rewinds
    // or starts again, never extends: generated ids rarely equal the history a template renders.
    Resume& resume = out.resume;
    resume.fed = static_cast<size_t>(m_.position());
    const bool back = prefill_columns_ > 0 && m_.prefill_end() >= 0;
    const size_t from = back ? static_cast<size_t>(m_.prefill_end()) : r.rewind_only ? 0 : resume.fed;
    if (from <= sequence.size() && from < prompt.size() &&
        std::equal(sequence.begin(), sequence.begin() + static_cast<std::ptrdiff_t>(from), prompt.begin()))
        resume.at = from;
    if (resume.at == 0 && resume.fed > 0)
        m_.reset(), resume.reset = true;
    if (resume.at > 0 && back)
        m_.rewind(), resume.rewound = true;
    if (o.on_start)
        o.on_start(resume);
    sequence = prompt;  // until the prompt is in: a cancel then leaves only what was fed (below)

    // Sampling: the request's seed, the padding ids masked, and the penalties' history counts of this request's
    // generated tokens only, from zero (also after a rewind).
    kernels::SamplingParams params = r.sampling;
    if (r.sample) {
        out.seed = r.seed ? *r.seed : (uint64_t{std::random_device{}()} << 32) | std::random_device{}();
        params.seed = out.seed;
        params.vocab_used = m_.config().n_vocab_used;
    }
    const bool counting = r.sample && (params.repetition_penalty != 1.0f || params.presence_penalty != 0.0f ||
                                       params.frequency_penalty != 0.0f);
    const std::span<int32_t> counts = m_.history_counts();
    if (counting)
        std::fill(counts.begin(), counts.end(), 0);
    const auto generated = [&](std::span<const int32_t> tokens) {
        if (counting)
            for (const int32_t t : tokens)
                ++counts[static_cast<size_t>(t)];
    };

    const size_t n_variants = static_cast<size_t>(variants_);
    out.variants.resize(n_variants);
    std::vector<model::Drafted> scratch;
    size_t total = prompt.size() + static_cast<size_t>(r.max_tokens);  // lowered when a stop id or the caller stops
    // How many of the new tokens count when generation must stop after them (a stop id, which counts, or the caller's
    // on_tokens returning false); 0 to go on.
    const auto stops = [&](std::span<const int32_t> tokens) {
        for (size_t k = 0; k < tokens.size(); ++k)
            if (std::find(r.stop_ids.begin(), r.stop_ids.end(), tokens[k]) != r.stop_ids.end()) {
                if (o.on_tokens)
                    o.on_tokens(tokens.first(k + 1));
                return k + 1;
            }
        return o.on_tokens && !o.on_tokens(tokens) ? tokens.size() : size_t{0};
    };
    int prompt_steps = 0;
    // The prompt's positions [0, prefilled) go as prefill chunks, the last one ending exactly there, and the rest as
    // windows: with a caller's stable length, the end of the last prefill is where the next turn rewinds to.
    const size_t prefilled = r.stable > 0 ? std::min(r.stable, prompt.size()) : prompt.size();
    bool cancel_now = false;  // the caller's on_progress asked to stop: no window after the current one
    const auto progress = [&] {
        if (o.on_progress && !o.on_progress(static_cast<size_t>(m_.position()), prompt.size()))
            cancel_now = true;
    };
    const auto prompt_start = Clock::now();
    for (size_t i = resume.at; i < sequence.size();) {
        // Prompt tokens are known, so they go `window` at a time, keeping `keep` of each window (or in prefill chunks
        // of up to prefill_columns tokens, kept whole); generated tokens go one at a time.
        if (cancel_now) {
            out.cancelled = true;
            break;
        }
        const bool chunked = prefill_columns_ > 0 && i < prefilled;
        const size_t width = chunked ? std::min<size_t>(static_cast<size_t>(prefill_columns_), prefilled - i)
                             : i < prompt.size() ? std::min<size_t>(static_cast<size_t>(r.window), prompt.size() - i)
                                                 : 1;
        const size_t kept = chunked || i < static_cast<size_t>(r.accept_from)
                                ? width
                                : std::min<size_t>(width, static_cast<size_t>(r.keep));
        const auto start = Clock::now();
        const std::span<const int32_t> tokens = std::span<const int32_t>(sequence).subspan(i, width);
        if (!(chunked ? m_.prefill(tokens) : m_.verify(tokens))) {
            if (!m_.cancelled())
                throw Error(m_.error());
            out.cancelled = true;
            break;
        }
        const double ms = ms_since(start);
        if (i >= prompt.size()) {  // generated tokens only: the speed of decoding, not of feeding the prompt
            out.decode_ms += ms;
            if (++out.decoded == r.cancel_after)
                m_.cancel();
        } else if (++prompt_steps == r.cancel_prompt_after) {
            m_.cancel();
        }
        progress();
        if (o.on_window)
            o.on_window(i, kept, width, chunked, ms);
        int32_t next = -1;
        for (size_t t = chunked ? kept - 1 : 0; t < kept; ++t)  // a chunk has its last position's results
            if (i + t + 1 == sequence.size())                   // greedy: the GPU's argmax, ties to the lowest id
                next = r.sample ? m_.sample(params, chunked ? 0 : static_cast<int>(t), 1)[0]
                                : m_.argmax(chunked ? 0 : static_cast<int>(t));
        // The prompt's last prefill chunk: the chunk buffer's lend (Model::lend_chunk) reads its experts from the SSD
        // while the draft's KV-only passes over the chunk run, and is joined before anything else uses the model.
        std::future<int> early;
        if (chunked && i + kept == prefilled && sequence.size() < total)
            early = std::async(std::launch::async, [this] { return m_.lend_chunk(true); });
        // The draft layer over the window's kept positions i .. i + kept - 1: the KV-only pass at prompt positions
        // whose next token is known, then drafts from the argmax at the last position while tokens are still to be
        // generated (the KV-only pass at the window's other positions in the same call). A chunk's positions go
        // kMaxColumns at a time, each group's residuals staged into the window columns.
        for (size_t v = 0; v < n_variants; ++v)
            for (size_t g = 0; g < kept; g += static_cast<size_t>(model::kMaxColumns)) {
                const size_t at = i + g, count = std::min<size_t>(kept - g, model::kMaxColumns);
                const bool kv_only = at + count < prompt.size();
                if (!kv_only && sequence.size() >= total)
                    break;
                if (chunked)
                    m_.stage_residuals(static_cast<int>(g), static_cast<int>(count));
                const auto draft_start = Clock::now();
                const int position = static_cast<int>(at);
                DraftStats& st = out.variants[v];
                if (kv_only) {
                    draft_->run(static_cast<int>(v), position, std::span(&prompt[at + 1], count), 0, scratch);
                } else {
                    std::vector<int32_t> after(sequence.begin() + static_cast<std::ptrdiff_t>(at + 1),
                                               sequence.begin() + static_cast<std::ptrdiff_t>(at + count));
                    after.push_back(next);
                    draft_->run(static_cast<int>(v), position, after, r.draft_depth, st.drafts.emplace_back());
                }
                (kv_only ? st.kv_ms : st.draft_ms) += ms_since(draft_start);
                if (kv_only)
                    st.kv_positions += static_cast<int>(count);
                else
                    ++st.draft_steps;
            }
        if (early.valid())
            early.get();
        if (kept < width && !m_.accept(static_cast<int>(kept)))
            throw Error(m_.error());
        i += kept;
        if (i == prompt.size()) {  // the prompt is in (once): the caller counts the generation's misses from here
            if (sequence.size() < total)
                m_.lend_chunk();  // its reads count in the prompt's time
            out.prompt_s = ms_since(prompt_start) / 1000.0;
            if (o.on_prompt_end)
                o.on_prompt_end();
            if (r.swaps_after_prompt >= 0)
                m_.set_swaps_per_window(r.swaps_after_prompt);
        }
        if (next >= 0 && i == sequence.size() && sequence.size() >= prompt.size() && sequence.size() < total) {
            sequence.push_back(next);
            generated(std::span(&sequence.back(), 1));
            if (stops(std::span(&sequence.back(), 1)) > 0)
                total = sequence.size(), out.stopped = true, out.stop_window = 1, out.stop_column = 0;
        }
        if (r.spec > 0 && i >= prompt.size())
            break;  // the speculative loop below generates the rest
    }
    if (out.cancelled && static_cast<size_t>(m_.position()) < prompt.size())
        sequence.resize(static_cast<size_t>(m_.position()));  // cancelled mid-prompt: the sequence is what was fed

    // Speculative greedy decoding: a window of the last token and the drafts, keeping its first n tokens, where n - 1
    // drafts equal the model's own argmax before them (stopping at an end of sequence); the window's n argmaxes are the
    // new tokens. The draft layer then extends over the kept positions (KV-only but the last) and drafts again from
    // the last one.
    if (r.spec > 0 && sequence.size() > prompt.size() && !out.cancelled) {
        const auto spec_start = Clock::now();
        const bool adaptive = r.policy != "fixed", tight = r.policy == "adaptive-tight",
                   stop_early = r.policy == "adaptive-stop" || tight;
        // Seed costs of a window of d drafts (milliseconds, drafting included): about 16 ms per verified column after
        // ~50 ms (measured with D = 3 on the review prompts, 96 ms a window); the policy follows the observed times.
        std::vector<double> seed_costs;
        for (int k = 0; k <= r.spec; ++k)
            seed_costs.push_back(50.0 + 16.0 * k);
        if (!r.carry_policy || !policy_ || policy_->max_drafts() != r.spec || policy_->tight() != tight)
            policy_.emplace(r.spec, seed_costs, tight);
        model::WindowPolicy& policy = *policy_;
        const std::function<bool(int, float)> more = [&](int drafted, float prob) {
            return policy.more(drafted, prob);
        };
        std::vector<model::Drafted> d = out.variants[0].drafts.back();
        out.depth_tried.assign(static_cast<size_t>(r.spec), 0),
            out.depth_accepted.assign(static_cast<size_t>(r.spec), 0);
        std::vector<int32_t> win, a(static_cast<size_t>(r.spec) + 1);
        while (sequence.size() < total && !(out.cancelled = cancel_now)) {
            const int p = m_.position();           // the window's first position, that of sequence.back()
            int use = static_cast<int>(d.size());  // drafts to verify
            if (adaptive) {
                for (size_t j = 0; j < d.size(); ++j)
                    policy.more(static_cast<int>(j) + 1, d[j].prob);  // records the probabilities
                use = policy.width();
            }
            win.assign(1, sequence.back());
            for (int j = 0; j < use; ++j)
                win.push_back(d[static_cast<size_t>(j)].token);
            const auto verify_start = Clock::now();
            if (!m_.verify(win)) {
                if (!m_.cancelled())
                    throw Error(m_.error());
                out.cancelled = true;
                break;
            }
            int n = 1;
            if (r.sample) {  // accept_drafts: the drafts equal to the drawn tokens, then the first that differs
                const std::vector<int32_t> drawn = m_.sample(params, 0, use + 1);
                n = static_cast<int>(drawn.size());
                std::copy(drawn.begin(), drawn.end(), a.begin());
            } else {
                for (int t = 0; t <= use; ++t)
                    a[static_cast<size_t>(t)] = m_.argmax(t);
                while (n <= use && d[static_cast<size_t>(n - 1)].token == a[static_cast<size_t>(n - 1)] &&
                       a[static_cast<size_t>(n - 1)] != static_cast<int32_t>(m_.config().eos_token))
                    ++n;
            }
            for (int j = 0; j < std::min(use, n); ++j)  // draft j was verified with every earlier one accepted
                ++out.depth_tried[static_cast<size_t>(j)], out.depth_accepted[static_cast<size_t>(j)] += j < n - 1;
            if (n <= use && !m_.accept(n))
                throw Error(m_.error());
            const double this_verify_ms = ms_since(verify_start);
            out.verify_ms += this_verify_ms;
            if (++out.windows == r.cancel_after)
                m_.cancel();
            progress();
            out.window_tokens += n, out.verified_drafts += use;
            const size_t before = sequence.size();
            sequence.insert(sequence.end(), a.begin(), a.begin() + n);
            const size_t counted = std::min<size_t>(static_cast<size_t>(n), total - before);
            generated(std::span(sequence.data() + before, counted));
            if (const size_t stop = stops(std::span(sequence.data() + before, counted)); stop > 0)
                total = before + stop, out.stopped = true, out.stop_window = n,
                out.stop_column = static_cast<int>(stop) - 1;
            const auto extend_start = Clock::now();
            if (sequence.size() < total) {
                draft_->run(0, p, std::span(a.data(), static_cast<size_t>(n)), r.spec, d,
                            stop_early ? more : std::function<bool(int, float)>());
                out.extended = p + n - 1;
            } else {  // the request's end: the K/V-only pass over its last window, for the drafts of a request that
                      // follows
                draft_->run(0, p, std::span(a.data(), static_cast<size_t>(n)), 0, scratch);
            }
            const double this_extend_ms = ms_since(extend_start);
            out.extend_ms += this_extend_ms;
            if (adaptive)
                policy.observe(use, n - 1, this_verify_ms + this_extend_ms);
        }
        sequence.resize(std::min(sequence.size(), total));
        out.spec_ms = ms_since(spec_start);
    }
    out.sequence = sequence;
    return out;
}

}  // namespace torch::runtime
