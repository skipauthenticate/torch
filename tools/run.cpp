// `torch run`: feeds token ids through the model one at a time, then decodes greedily, printing each step's most
// likely next tokens. It is the engine's M2 test driver: the model parity test (tests/model/) reads its --json output,
// and the residency test compares its --logits output byte for byte.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/budget.h"
#include "core/error.h"
#include "core/placement.h"
#include "model/draft.h"
#include "model/model.h"
#include "model/window_policy.h"

namespace torch {
namespace {

constexpr const char* kUsage =
    "usage: torch run [options] MODEL-00001-of-00002.gguf --tokens ID,ID,...\n"
    "\n"
    "Feeds the tokens one at a time, then generates greedily, printing each step's top log-probabilities.\n"
    "Shard 2 (the n-gram table) is found next to shard 1 by its split name.\n"
    "\n"
    "  --tokens LIST   comma-separated prompt token ids, or @FILE to read them from FILE\n"
    "  -n N            tokens to generate after the prompt (default 16)\n"
    "  --then LIST     another request after the first (repeatable): when its prompt extends the sequence so far,\n"
    "                  only its new tokens are fed (with --prefill, from the end of the last prefill: the tokens\n"
    "                  generated since are prefilled again with them); otherwise the sequence starts again\n"
    "  --cancel-after N  cancel each request after N generated windows (tests; Ctrl-C cancels between windows)\n"
    "  --cancel-prompt-after N  cancel the first request after N of its prompt's windows or prefill chunks (tests)\n"
    "  --top K         log-probabilities to print per step (default 5)\n"
    "  --context N     tokens the KV cache holds (default 2048, maximum 262144); above 2048 attention is sparse\n"
    "  --resident K    keep exactly experts 0 to K - 1 of every layer resident (residency tests)\n"
    "  --placement F   keep the experts most routed in route-count file F resident (default: every layer's first\n"
    "                  experts, as many as fit in memory)\n"
    "  --route-counts F  after the run, add how often each expert was routed for the generated tokens (not the\n"
    "                  prompt) to route-count file F\n"
    "  --lend-kv       keep more experts in the KV cache's unused tail, from position 16,384 of the context; before\n"
    "                  a window writes there, the experts in the next 16,384 positions' bytes stop being resident\n"
    "                  (the output is unchanged)\n"
    "  --lend-from P   --lend-kv from position P in steps of P (a multiple of 64; tests)\n"
    "  --logits FILE   write every step's raw F32 logits to FILE\n"
    "  --json FILE     write every step's top log-probabilities to FILE, one JSON object per line\n"
    "  --profile       print where decode time went: PLE rows, the GPU, expert reads, launches\n"
    "  --window T      feed the prompt T tokens at a time (1-8) as verify windows; every position's results must\n"
    "                  equal --window 1's bit for bit (generated tokens still go one at a time)\n"
    "  --prefill N     feed the prompt in prefill chunks of up to N tokens (1-4096), each kept whole; every\n"
    "                  position's state must equal --window 8's bit for bit; a chunk prints (and writes to --logits\n"
    "                  and --json) its last position only\n"
    "  --accept K      keep only K tokens of each prompt window and run the next window from there (tests rollback)\n"
    "  --accept-from P apply --accept only to windows starting at position P or later (default 0)\n"
    "  --quiet         no per-step lines, and no logits copied to the host (for timing)\n"
    "  --swap N        after each window, up to N of the experts it read from the SSD become resident in place of the\n"
    "                  least recently routed ones of their layers (the output is unchanged)\n"
    "  --swap-prompt N at most N swaps after each of the prompt's windows (default: --swap's N)\n"
    "  --draft FILE    load the draft layer (tools/make_draft.py) and run it after every window: the KV-only\n"
    "                  pass at prompt positions, --draft-depth drafts at generated ones (M4 R3; the model's output\n"
    "                  is unchanged)\n"
    "  --draft-depth D drafts per generated token (default 3)\n"
    "  --spec D        generate speculatively: windows of the last token and up to D drafts (1-7) from the first\n"
    "                  draft variant, keeping the longest prefix the model agrees with (greedy; the same tokens as\n"
    "                  without)\n"
    "  --spec-policy P how many drafts a window verifies: fixed (all D), adaptive (the width that maximises\n"
    "                  expected tokens per millisecond, learned online; drafts all D first) or adaptive-stop (also\n"
    "                  stops drafting early; default adaptive-stop)\n"
    "  --draft-variants LIST  comma-separated HNORM:HEAD, HNORM stream or pooled, HEAD full or a number of\n"
    "                  --draft-vocab ids (default stream:full); each keeps its own draft K/V cache\n"
    "  --check-draft-kv  after --spec, decode the history again one token at a time and require the draft layer's\n"
    "                  K/V rows to equal those the windows left, bit for bit\n"
    "  --draft-vocab FILE  token ids, most frequent first, whitespace-separated (bench/draft_acceptance.py vocab)\n"
    "  --acceptance-report FILE  print each variant's acceptance per depth and write it to FILE (JSON lines)\n";

using Clock = std::chrono::steady_clock;

std::atomic<model::Model*> g_cancel{nullptr};  // the model Ctrl-C cancels

void on_interrupt(int) {
    if (model::Model* m = g_cancel.load())
        m->cancel();
}

// Ctrl-C cancels the running request between windows while the guard lives.
struct CancelOnInterrupt {
    explicit CancelOnInterrupt(model::Model& m) {
        g_cancel = &m;
        std::signal(SIGINT, on_interrupt);
    }
    ~CancelOnInterrupt() {
        std::signal(SIGINT, SIG_DFL);
        g_cancel = nullptr;
    }
    CancelOnInterrupt(const CancelOnInterrupt&) = delete;
    CancelOnInterrupt& operator=(const CancelOnInterrupt&) = delete;
};

std::vector<int32_t> parse_tokens(std::string_view list) {
    std::vector<int32_t> out;
    while (!list.empty()) {
        const size_t comma = list.find(',');
        const std::string item(list.substr(0, comma));
        char* end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || v < 0 || v > INT32_MAX)
            throw Error("--tokens takes comma-separated token ids, such as 760,6511,314.");
        out.push_back(static_cast<int32_t>(v));
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
    }
    if (out.empty())
        throw Error("--tokens needs at least one token id.");
    return out;
}

int parse_count(const char* text, const char* option, int lo) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (*end != '\0' || v < lo || v > INT32_MAX)
        throw Error(std::string(option) + " takes a whole number of at least " + std::to_string(lo) + ".");
    return static_cast<int>(v);
}

std::filesystem::path second_shard(const std::filesystem::path& first) {
    std::string name = first.filename().string();
    const size_t at = name.rfind("-00001-of-00002.gguf");
    if (at == std::string::npos)
        throw Error("Pass shard 1 of the model, the file whose name ends in -00001-of-00002.gguf.");
    name.replace(at, 6, "-00002");
    return first.parent_path() / name;
}

std::vector<model::DraftVariant> parse_variants(std::string_view list) {
    std::vector<model::DraftVariant> out;
    while (!list.empty()) {
        const size_t comma = list.find(',');
        const std::string item(list.substr(0, comma));
        const size_t colon = item.find(':');
        const std::string hnorm = item.substr(0, colon),
                          head = colon == std::string::npos ? "" : item.substr(colon + 1);
        model::DraftVariant v;
        v.pooled_hnorm = hnorm == "pooled";
        if ((hnorm != "stream" && hnorm != "pooled") || head.empty())
            throw Error("--draft-variants takes HNORM:HEAD items, such as stream:full,pooled:40960.");
        v.head_rows = head == "full" ? 0 : parse_count(head.c_str(), "A draft head", 1);
        out.push_back(v);
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
    }
    if (out.empty())
        throw Error("--draft-variants needs at least one variant.");
    return out;
}

std::string variant_name(const model::DraftVariant& v) {
    return std::string(v.pooled_hnorm ? "pooled" : "stream") + ":" +
           (v.head_rows ? std::to_string(v.head_rows) : std::string("full"));
}

// Acceptance of one draft variant over the generated positions whose drafts all have a true token to compare with.
struct Acceptance {
    static constexpr double kBounds[] = {0.5, 0.7, 0.8, 0.9, 0.95, 0.99};  // draft-probability buckets
    static constexpr size_t kBuckets = std::size(kBounds) + 1;
    int steps = 0;
    std::vector<int> tried, accepted, lengths;  // per depth: drafted with every earlier draft right, and right
    int bucket_tried[kBuckets] = {}, bucket_accepted[kBuckets] = {};
    double kv_ms = 0, draft_ms = 0;
    int kv_positions = 0, draft_steps = 0;

    explicit Acceptance(int depth) : tried(depth), accepted(depth), lengths(depth + 1) {}

    // drafts[s] were made after decoding position first + s, so drafts[s][j] predicts sequence[first + s + 2 + j].
    void count(const std::vector<std::vector<model::Drafted>>& drafts, const std::vector<int32_t>& sequence,
               size_t first) {
        const size_t depth = tried.size();
        for (size_t s = 0; s < drafts.size(); ++s) {
            if (first + s + 1 + depth >= sequence.size())
                break;
            ++steps;
            size_t run = 0;
            for (size_t j = 0; j < depth; ++j) {
                const bool right = drafts[s][j].token == sequence[first + s + 2 + j];
                const size_t bucket = static_cast<size_t>(
                    std::upper_bound(std::begin(kBounds), std::end(kBounds), drafts[s][j].prob) - std::begin(kBounds));
                ++tried[j], ++bucket_tried[bucket];
                if (!right)
                    break;
                ++accepted[j], ++bucket_accepted[bucket], ++run;
            }
            ++lengths[run];
        }
    }
    double mean_accepted(size_t cap) const {
        double sum = 0;
        for (size_t n = 0; n < lengths.size(); ++n)
            sum += static_cast<double>(std::min(n, cap)) * lengths[n];
        return steps ? sum / steps : 0;
    }
};

struct Top {
    int32_t token;
    double logprob;
};

// The k most likely tokens and their log-probabilities; ties go to the lower token id, like the engine's argmax.
std::vector<Top> top_k(std::span<const float> logits, int k) {
    const float max = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float x : logits)
        sum += std::exp(static_cast<double>(x) - max);
    const double log_sum = std::log(sum) + max;
    std::vector<int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    k = std::min<int>(k, static_cast<int>(ids.size()));
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int32_t a, int32_t b) {
        return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)] ||
               (logits[static_cast<size_t>(a)] == logits[static_cast<size_t>(b)] && a < b);
    });
    std::vector<Top> out;
    for (int i = 0; i < k; ++i)
        out.push_back(
            {ids[static_cast<size_t>(i)], logits[static_cast<size_t>(ids[static_cast<size_t>(i)])] - log_sum});
    return out;
}

}  // namespace

int run(int argc, char** argv) {
    std::optional<std::filesystem::path> shard1;
    std::vector<int32_t> prompt;
    std::vector<std::vector<int32_t>> turns;  // the requests after the first (--then)
    int cancel_after = 0;                     // generated windows before Model::cancel() (tests); 0: never
    int cancel_prompt_after = 0;              // the first request's prompt windows or chunks before it (tests)
    int generate = 16, top = 5, window = 1, keep = model::kMaxColumns, accept_from = 0;
    bool profile = false, quiet = false;
    std::filesystem::path route_counts_path;
    model::Options options;
    std::FILE* logits_file = nullptr;
    std::FILE* json_file = nullptr;
    int draft_depth = 3, spec = 0, swap_prompt = -1;
    std::string spec_policy = "adaptive-stop";
    bool check_kv = false;
    std::filesystem::path draft_path, report_path;
    std::vector<model::DraftVariant> variants{{}};
    std::vector<int32_t> vocab;
    try {
        for (int i = 0; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto value = [&]() -> const char* {
                if (i + 1 >= argc)
                    throw Error(std::string(arg) + " needs a value.");
                return argv[++i];
            };
            const auto tokens_of = [](const std::string& list) {
                if (!list.starts_with("@"))
                    return parse_tokens(list);
                std::ifstream in(list.substr(1));  // @FILE: the ids from a file, as bench/workloads/ writes them
                std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                while (!text.empty() && (text.back() == '\n' || text.back() == ' '))
                    text.pop_back();
                if (!in && text.empty())
                    throw Error("Cannot read token ids from " + list.substr(1) + ".");
                return parse_tokens(text);
            };
            if (arg == "--tokens") {
                prompt = tokens_of(value());
            } else if (arg == "--then") {
                turns.push_back(tokens_of(value()));
            } else if (arg == "--cancel-after") {
                cancel_after = parse_count(value(), "--cancel-after", 1);
            } else if (arg == "--cancel-prompt-after") {
                cancel_prompt_after = parse_count(value(), "--cancel-prompt-after", 1);
            } else if (arg == "-n") {
                generate = parse_count(value(), "-n", 0);
            } else if (arg == "--top") {
                top = parse_count(value(), "--top", 1);
            } else if (arg == "--context") {
                options.context = parse_count(value(), "--context", 1);
            } else if (arg == "--resident") {
                options.resident = parse_count(value(), "--resident", 0);
            } else if (arg == "--placement") {
                options.placement = value();
            } else if (arg == "--lend-kv") {
                options.lend_from = options.lend_from > 0 ? options.lend_from : 16384;
            } else if (arg == "--lend-from") {
                options.lend_from = parse_count(value(), "--lend-from", 64);
            } else if (arg == "--route-counts") {
                route_counts_path = value();
            } else if (arg == "--logits") {
                const char* path = value();
                if (!(logits_file = std::fopen(path, "wb")))
                    throw Error(std::string("Cannot write ") + path + ".");
            } else if (arg == "--json") {
                const char* path = value();
                if (!(json_file = std::fopen(path, "w")))
                    throw Error(std::string("Cannot write ") + path + ".");
            } else if (arg == "--window") {
                window = std::atoi(value());
                if (window < 1 || window > model::kMaxColumns)
                    throw Error("--window takes 1 to " + std::to_string(model::kMaxColumns) + " tokens.");
            } else if (arg == "--prefill") {
                options.prefill_columns = std::atoi(value());
                if (options.prefill_columns < 1 || options.prefill_columns > model::kMaxPrefillColumns)
                    throw Error("--prefill takes 1 to " + std::to_string(model::kMaxPrefillColumns) + " tokens.");
            } else if (arg == "--accept") {
                keep = std::atoi(value());
                if (keep < 1)
                    throw Error("--accept takes a number of tokens of at least 1.");
            } else if (arg == "--accept-from") {
                accept_from = std::atoi(value());
            } else if (arg == "--profile") {
                profile = true;
            } else if (arg == "--quiet") {
                quiet = true;
            } else if (arg == "--draft") {
                draft_path = value();
            } else if (arg == "--check-draft-kv") {
                check_kv = true;
            } else if (arg == "--spec") {
                spec = parse_count(value(), "--spec", 1);
                if (spec >= model::kMaxColumns)
                    throw Error("--spec takes 1 to " + std::to_string(model::kMaxColumns - 1) + " drafts.");
            } else if (arg == "--swap") {
                options.swaps_per_window = parse_count(value(), "--swap", 0);
            } else if (arg == "--swap-prompt") {
                swap_prompt = parse_count(value(), "--swap-prompt", 0);
            } else if (arg == "--spec-policy") {
                spec_policy = value();
                if (spec_policy != "fixed" && spec_policy != "adaptive" && spec_policy != "adaptive-stop")
                    throw Error("--spec-policy takes fixed, adaptive or adaptive-stop.");
            } else if (arg == "--draft-depth") {
                draft_depth = parse_count(value(), "--draft-depth", 1);
            } else if (arg == "--draft-variants") {
                variants = parse_variants(value());
            } else if (arg == "--draft-vocab") {
                const char* path = value();
                std::ifstream in(path);
                if (!in)
                    throw Error(std::string("Cannot read ") + path + ".");
                for (long id; in >> id;)
                    vocab.push_back(static_cast<int32_t>(id));
            } else if (arg == "--acceptance-report") {
                report_path = value();
            } else if (arg == "--help" || arg == "-h") {
                std::fputs(kUsage, stdout);
                return 0;
            } else if (arg.starts_with("-") || shard1) {
                std::fprintf(stderr, "torch run: unexpected argument %s\n\n%s", argv[i], kUsage);
                return 2;
            } else {
                shard1 = argv[i];
            }
        }
        if (!shard1 || prompt.empty()) {
            std::fputs(kUsage, stderr);
            return 2;
        }
        options.shard1 = *shard1;
        options.shard2 = second_shard(*shard1);
        if (quiet && (logits_file || json_file))
            throw Error("--quiet copies no logits, so it cannot be combined with --logits or --json.");
        options.logits = !quiet;
        const int swap_generate = options.swaps_per_window;  // --swap-prompt sets the prompt's windows apart
        if (swap_prompt >= 0)
            options.swaps_per_window = swap_prompt;
        const bool drafting = !draft_path.empty();
        if (spec > 0 && (!drafting || variants.size() != 1))
            throw Error("--spec needs --draft and one --draft-variants variant.");
        if (spec > 0)
            draft_depth = spec;
        if (check_kv && spec == 0)
            throw Error("--check-draft-kv checks the speculative loop; give --spec.");
        if (options.prefill_columns > 0 && (keep < model::kMaxColumns || accept_from > 0))
            throw Error("--prefill keeps whole chunks: it cannot be combined with --accept.");
        if (!turns.empty() && (check_kv || !route_counts_path.empty() || !report_path.empty()))
            throw Error("--then cannot be combined with --check-draft-kv, --route-counts or --acceptance-report.");
        turns.insert(turns.begin(), prompt);
        // with --check-draft-kv, a second copy of the variant takes the fresh pass
        const std::vector<model::DraftVariant> draft_variants =
            check_kv ? std::vector{variants[0], variants[0]} : variants;
        if (drafting)
            options.reserve.push_back(
                model::Draft::reservation(options.shard1, draft_path, draft_variants, options.context));

        const auto load_start = Clock::now();
        model::Model m(options);
        const CancelOnInterrupt interrupts(m);
        std::unique_ptr<model::Draft> draft;
        if (drafting)
            draft = std::make_unique<model::Draft>(m, draft_path, draft_variants, vocab, options.context);
        const double load_s = std::chrono::duration<double>(Clock::now() - load_start).count();
        std::printf("%s", m.budget().table().c_str());
        std::printf("loaded in %.1f s; MemAvailable now %.2f GiB\n\n", load_s,
                    static_cast<double>(mem_available()) / static_cast<double>(kGiB));

        std::vector<int32_t> sequence;
        RouteCounts routes_at_prompt_end{};
        for (size_t turn = 0; turn < turns.size(); ++turn) {
            // A request whose prompt extends the sequence so far (the tokens fed to the model) feeds only its new
            // tokens; any other prompt starts a new sequence.
            prompt = turns[turn];
            size_t resume_at = 0;  // the first position this request feeds
            if (turn > 0) {
                // Sessions, option B: with --prefill, back to the end of the last prefill, so the tokens fed since
                // (generated ones, and a speculative window's columns past the sequence) are prefilled again with the
                // new ones, as a fresh prefill of the whole prompt would feed them.
                const size_t fed = static_cast<size_t>(m.position());
                const bool back = options.prefill_columns > 0 && m.prefill_end() >= 0;
                const size_t from = back ? static_cast<size_t>(m.prefill_end()) : fed;
                if (from <= sequence.size() && from < prompt.size() &&
                    std::equal(sequence.begin(), sequence.begin() + static_cast<std::ptrdiff_t>(from), prompt.begin()))
                    resume_at = from;
                else
                    m.reset();
                if (resume_at > 0 && back) {
                    m.rewind();
                    std::printf(
                        "\nback to the end of the last prefill at position %zu: the %zu tokens fed since are "
                        "dropped\n",
                        resume_at, fed - resume_at);
                }
                if (resume_at == 0 && options.lend_from > 0)
                    std::printf("\nafter the reset, %d experts in the KV cache's lent tail\n",
                                m.expert_stats().tenants);
                std::printf("\nturn %zu: %zu new prompt tokens after %zu already in the sequence\n", turn + 1,
                            prompt.size() - resume_at, resume_at);
            }
            sequence = prompt;
            const model::ExpertStats at_turn_start = m.expert_stats();
            model::ExpertStats at_prompt_end{};  // expert hits and misses when the prompt was in
            model::DecodeTimes times_at_prompt_end{};
            // The prompt's routing predicts the generation's poorly (code in, English out), so --route-counts records
            // only the generated tokens: the counts when the prompt was in are subtracted at the end.
            double decode_ms = 0;
            int decoded = 0, prompt_steps = 0;
            const size_t n_variants = drafting ? variants.size() : 0;
            std::vector<Acceptance> acceptance(n_variants, Acceptance(draft_depth));
            std::vector<std::vector<std::vector<model::Drafted>>> drafts(n_variants);  // [variant][generated step]
            std::vector<model::Drafted> scratch;
            const auto prompt_start = Clock::now();
            double prompt_s = 0;     // the prompt's windows, the draft layer's passes over them included
            bool cancelled = false;  // Model::cancel() stopped the request between windows
            for (size_t i = resume_at; i < sequence.size();) {
                // Prompt tokens are known, so they go `window` at a time, keeping `keep` of each window (or in prefill
                // chunks of up to --prefill tokens, kept whole); generated tokens go one at a time.
                const bool chunked = options.prefill_columns > 0 && i < prompt.size();
                const size_t width =
                    chunked ? std::min<size_t>(static_cast<size_t>(options.prefill_columns), prompt.size() - i)
                    : i < prompt.size() ? std::min<size_t>(static_cast<size_t>(window), prompt.size() - i)
                                        : 1;
                const size_t kept = chunked || i < static_cast<size_t>(accept_from)
                                        ? width
                                        : std::min<size_t>(width, static_cast<size_t>(keep));
                const auto start = Clock::now();
                const std::span<const int32_t> tokens = std::span<const int32_t>(sequence).subspan(i, width);
                if (!(chunked ? m.prefill(tokens) : m.verify(tokens))) {
                    if (!m.cancelled())
                        throw Error(m.error());
                    cancelled = true;
                    break;
                }
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (i >= prompt.size()) {  // generated tokens only: the speed of decoding, not of feeding the prompt
                    decode_ms += ms;
                    if (++decoded == cancel_after)
                        m.cancel();
                } else if (turn == 0 && ++prompt_steps == cancel_prompt_after) {
                    m.cancel();
                }
                int32_t next = -1;
                for (size_t t = chunked ? kept - 1 : 0; t < kept; ++t) {  // a chunk has its last position's results
                    const size_t at = i + t;
                    const int column = chunked ? 0 : static_cast<int>(t);
                    if (!quiet) {
                        const std::span<const float> logits = m.logits(column);
                        const std::vector<Top> best = top_k(logits, top);
                        std::printf("pos %4zu  in %6d  ->", at, sequence[at]);
                        for (const Top& top_entry : best)
                            std::printf("  %6d %8.4f", top_entry.token, top_entry.logprob);
                        std::printf("   (%.1f ms%s)\n", ms,
                                    chunked     ? " for the chunk"
                                    : width > 1 ? " for the window"
                                                : "");
                        if (logits_file)
                            std::fwrite(logits.data(), sizeof(float), logits.size(), logits_file);
                        if (json_file) {
                            std::fprintf(json_file, "{\"pos\": %zu, \"token\": %d, \"top\": [", at, sequence[at]);
                            for (size_t j = 0; j < best.size(); ++j)
                                std::fprintf(json_file, "%s[%d, %.7g]", j ? ", " : "", best[j].token, best[j].logprob);
                            std::fprintf(json_file, "]}\n");
                        }
                    }
                    if (at + 1 == sequence.size())
                        next = m.argmax(column);  // greedy: the GPU's argmax, ties to the lowest id
                }
                // The draft layer over the window's kept positions i .. i + kept - 1: the KV-only pass at prompt
                // positions whose next token is known, then drafts from the argmax at the last position while tokens
                // are still to be generated (the KV-only pass at the window's other positions in the same call). A
                // chunk's positions go kMaxColumns at a time, each group's residuals staged into the window columns.
                for (size_t v = 0; v < n_variants; ++v)
                    for (size_t g = 0; g < kept; g += static_cast<size_t>(model::kMaxColumns)) {
                        const size_t at = i + g, count = std::min<size_t>(kept - g, model::kMaxColumns);
                        const bool kv_only = at + count < prompt.size();
                        if (!kv_only && sequence.size() >= prompt.size() + static_cast<size_t>(generate))
                            break;
                        if (chunked)
                            m.stage_residuals(static_cast<int>(g), static_cast<int>(count));
                        const auto draft_start = Clock::now();
                        const int position = static_cast<int>(at);
                        if (kv_only) {
                            draft->run(static_cast<int>(v), position, std::span(&prompt[at + 1], count), 0, scratch);
                        } else {
                            std::vector<int32_t> after(sequence.begin() + static_cast<std::ptrdiff_t>(at + 1),
                                                       sequence.begin() + static_cast<std::ptrdiff_t>(at + count));
                            after.push_back(next);
                            draft->run(static_cast<int>(v), position, after, draft_depth, drafts[v].emplace_back());
                        }
                        const double draft_ms =
                            std::chrono::duration<double, std::milli>(Clock::now() - draft_start).count();
                        Acceptance& a = acceptance[v];
                        (kv_only ? a.kv_ms : a.draft_ms) += draft_ms;
                        if (kv_only)
                            a.kv_positions += static_cast<int>(count);
                        else
                            ++a.draft_steps;
                    }
                if (kept < width && !m.accept(static_cast<int>(kept)))
                    throw Error(m.error());
                i += kept;
                if (i == prompt.size()) {  // the prompt is in (once): count the generation's misses and swaps from here
                    prompt_s = std::chrono::duration<double>(Clock::now() - prompt_start).count();
                    at_prompt_end = m.expert_stats();
                    times_at_prompt_end = m.decode_times();
                    if (!route_counts_path.empty())
                        routes_at_prompt_end = m.route_counts();
                    m.set_swaps_per_window(swap_generate);
                }
                if (next >= 0 && i == sequence.size() && sequence.size() >= prompt.size() &&
                    sequence.size() < prompt.size() + static_cast<size_t>(generate))
                    sequence.push_back(next);
                if (spec > 0 && i >= prompt.size())
                    break;  // the speculative loop below generates the rest
            }

            // Speculative greedy decoding: a window of the last token and the drafts, keeping its first n tokens, where
            // n - 1 drafts equal the model's own argmax before them (stopping at an end of sequence); the window's n
            // argmaxes are the new tokens. The draft layer then extends over the kept positions (KV-only but the last)
            // and drafts again from the last one.
            double verify_ms = 0, extend_ms = 0, spec_ms = 0;
            int windows = 0, window_tokens = 0, verified_drafts = 0,
                extended = -1;  // the last position the draft layer extended over
            if (spec > 0 && sequence.size() > prompt.size() && !cancelled) {
                const auto spec_start = Clock::now();
                const size_t total = prompt.size() + static_cast<size_t>(generate);
                const bool adaptive = spec_policy != "fixed", stop_early = spec_policy == "adaptive-stop";
                // Seed costs of a window of d drafts (milliseconds, drafting included): about 16 ms per verified column
                // after ~50 ms (measured with D = 3 on the review prompts, 96 ms a window); the policy follows the
                // observed times from there.
                std::vector<double> seed_costs;
                for (int k = 0; k <= spec; ++k)
                    seed_costs.push_back(50.0 + 16.0 * k);
                model::WindowPolicy policy(spec, seed_costs);
                const std::function<bool(int, float)> more = [&](int drafted, float prob) {
                    return policy.more(drafted, prob);
                };
                std::vector<model::Drafted> d = drafts[0].back();
                std::vector<int32_t> win, a(static_cast<size_t>(spec) + 1);
                while (sequence.size() < total) {
                    const int p = m.position();            // the window's first position, that of sequence.back()
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
                    if (!m.verify(win)) {
                        if (!m.cancelled())
                            throw Error(m.error());
                        cancelled = true;
                        break;
                    }
                    for (int t = 0; t <= use; ++t)
                        a[static_cast<size_t>(t)] = m.argmax(t);
                    int n = 1;
                    while (n <= use && d[static_cast<size_t>(n - 1)].token == a[static_cast<size_t>(n - 1)] &&
                           a[static_cast<size_t>(n - 1)] != static_cast<int32_t>(m.config().eos_token))
                        ++n;
                    if (n <= use && !m.accept(n))
                        throw Error(m.error());
                    const double this_verify_ms =
                        std::chrono::duration<double, std::milli>(Clock::now() - verify_start).count();
                    verify_ms += this_verify_ms;
                    if (++windows == cancel_after)
                        m.cancel();
                    window_tokens += n, verified_drafts += use;
                    sequence.insert(sequence.end(), a.begin(), a.begin() + n);
                    const auto extend_start = Clock::now();
                    if (sequence.size() < total) {
                        draft->run(0, p, std::span(a.data(), static_cast<size_t>(n)), spec, d,
                                   stop_early ? more : std::function<bool(int, float)>());
                        extended = p + n - 1;
                    } else if (turn + 1 < turns.size()) {  // the next turn's drafts see these positions too
                        draft->run(0, p, std::span(a.data(), static_cast<size_t>(n)), 0, scratch);
                    }
                    const double this_extend_ms =
                        std::chrono::duration<double, std::milli>(Clock::now() - extend_start).count();
                    extend_ms += this_extend_ms;
                    if (adaptive)
                        policy.observe(use, n - 1, this_verify_ms + this_extend_ms);
                }
                sequence.resize(std::min(sequence.size(), total));
                spec_ms = std::chrono::duration<double, std::milli>(Clock::now() - spec_start).count();
                const int ring = std::min(options.context, model::kDraftWindow);
                if (check_kv && extended >= 0 && extended + 1 + spec > ring) {
                    std::printf("draft K/V check skipped: %d positions pass the draft's ring of %d rows\n",
                                extended + 1 + spec, ring);
                } else if (check_kv && extended >= 0) {  // the K/V invariant: decode the history again, one at a time
                    m.reset();
                    for (int q = 0; q <= extended; ++q) {
                        if (!m.decode(sequence[static_cast<size_t>(q)]))
                            throw Error(m.error());
                        draft->run(1, q, std::span(&sequence[static_cast<size_t>(q) + 1], 1), 0, scratch);
                    }
                    const int same = draft->first_kv_difference(0, 1, extended + 1);
                    const int beyond = draft->first_kv_difference(0, 1, std::min(extended + 1 + spec, options.context));
                    std::printf(
                        "draft K/V after %d windows: rows 0-%d %s a fresh KV-only pass over the accepted history "
                        "(first difference including the %d rows after them: row %d)\n",
                        windows, extended, same < 0 ? "equal, bit for bit," : "DIFFER from", spec, beyond);
                    if (same >= 0)
                        throw Error("The draft K/V differs from a fresh pass at row " + std::to_string(same) + ".");
                }
            }

            if (quiet) {  // the generated ids, so a run's output can be compared and hashed
                std::printf("generated tokens:");
                for (size_t i = prompt.size(); i < sequence.size(); ++i)
                    std::printf("%s%d", i == prompt.size() ? " " : ",", sequence[i]);
                std::printf("\n");
            }
            const model::ExpertStats s = m.expert_stats();
            {
                const model::DecodeTimes t = m.decode_times();
                const int gen_windows = std::max(spec > 0 ? windows : decoded, 1);
                std::printf(
                    "\ngeneration: %llu routed hits, %llu misses; %llu swaps, %.1f ms (%.2f ms per window); the "
                    "prompt's: %llu swaps, %.1f ms\n",
                    static_cast<unsigned long long>(s.hits - at_prompt_end.hits),
                    static_cast<unsigned long long>(s.misses - at_prompt_end.misses),
                    static_cast<unsigned long long>(t.swaps - times_at_prompt_end.swaps),
                    t.swap_ms - times_at_prompt_end.swap_ms, (t.swap_ms - times_at_prompt_end.swap_ms) / gen_windows,
                    static_cast<unsigned long long>(times_at_prompt_end.swaps), times_at_prompt_end.swap_ms);
                if (prompt_s > 0)  // the prompt is in
                    std::printf(
                        "prompt: %llu routed hits, %llu misses; %llu experts read for prefill chunks\n",
                        static_cast<unsigned long long>(at_prompt_end.hits - at_turn_start.hits),
                        static_cast<unsigned long long>(at_prompt_end.misses - at_turn_start.misses),
                        static_cast<unsigned long long>(at_prompt_end.prefill_reads - at_turn_start.prefill_reads));
            }
            std::printf("\nprompt %zu tokens, generated %d%s", prompt.size(),
                        static_cast<int>(sequence.size() - prompt.size()), cancelled ? " (cancelled)" : "");
            if (decoded > 0)
                std::printf(": %.2f tok/s (%.1f ms per token)", 1000.0 * decoded / decode_ms, decode_ms / decoded);
            std::printf(
                "\nexperts: %d of %d resident; %llu routed hits, %llu misses (%.1f%% hits); %llu hedged reads\n",
                s.resident, s.experts, static_cast<unsigned long long>(s.hits),
                static_cast<unsigned long long>(s.misses),
                100.0 * static_cast<double>(s.hits) / static_cast<double>(std::max<uint64_t>(s.hits + s.misses, 1)),
                static_cast<unsigned long long>(s.hedged));
            if (options.lend_from > 0)
                std::printf("of them in the KV cache's lent tail: %d\n", s.tenants);
            std::printf("MemAvailable at the end %.2f GiB\nprompt in %.1f s (%s %d)\n",
                        static_cast<double>(mem_available()) / static_cast<double>(kGiB), prompt_s,
                        options.prefill_columns > 0 ? "prefill chunks of" : "window",
                        options.prefill_columns > 0 ? options.prefill_columns : window);
            std::FILE* report = report_path.empty() ? nullptr : std::fopen(report_path.string().c_str(), "w");
            if (!report_path.empty() && !report)
                throw Error("Cannot write " + report_path.string() + ".");
            if (spec > 0 && windows > 0) {
                const double first = acceptance[0].draft_ms;  // the drafts after the prompt's last token
                const double gen_ms = spec_ms + first;
                std::printf(
                    "speculative (%s): %zu tokens in %d windows of the last token and up to %d drafts, %.2f verified "
                    "on "
                    "average: %.2f tokens per window, "
                    "%.2f tok/s (%.1f ms per window: verify and accept %.1f, draft %.1f, host %.1f)\n",
                    spec_policy.c_str(), static_cast<size_t>(generate), windows, spec,
                    static_cast<double>(verified_drafts) / windows, static_cast<double>(window_tokens) / windows,
                    1000.0 * generate / gen_ms, spec_ms / windows, verify_ms / windows, (extend_ms + first) / windows,
                    (spec_ms - verify_ms - extend_ms) / windows);
            }
            for (size_t v = 0; v < n_variants && spec == 0; ++v) {
                Acceptance& a = acceptance[v];
                a.count(drafts[v], sequence, prompt.size() - 1);
                const std::string name = variant_name(variants[v]);
                std::printf("draft %s: %d steps of %d drafts; accepted at each depth, given the earlier ones:",
                            name.c_str(), a.steps, draft_depth);
                for (int j = 0; j < draft_depth; ++j)
                    std::printf(" %.1f%%", 100.0 * a.accepted[static_cast<size_t>(j)] /
                                               std::max(a.tried[static_cast<size_t>(j)], 1));
                std::printf(
                    "\n  mean accepted drafts %.2f of 3, %.2f of %d; KV-only pass %.2f ms per prompt position; "
                    "drafting %.1f ms per step\n",
                    a.mean_accepted(3), a.mean_accepted(static_cast<size_t>(draft_depth)), draft_depth,
                    a.kv_ms / std::max(a.kv_positions, 1), a.draft_ms / std::max(a.draft_steps, 1));
                if (report) {
                    const auto list = [&](const int* x, size_t n) {
                        std::string text = "[";
                        for (size_t k = 0; k < n; ++k)
                            text += (k ? ", " : "") + std::to_string(x[k]);
                        return text + "]";
                    };
                    std::fprintf(report,
                                 "{\"variant\": \"%s\", \"depth\": %d, \"prompt_tokens\": %zu, \"generated\": %zu, "
                                 "\"steps\": %d, \"tried\": %s, \"accepted\": %s, \"lengths\": %s, "
                                 "\"prob_tried\": %s, \"prob_accepted\": %s, \"kv_ms\": %.3f, \"kv_positions\": %d, "
                                 "\"draft_ms\": %.3f, \"draft_steps\": %d}\n",
                                 name.c_str(), draft_depth, prompt.size(), sequence.size() - prompt.size(), a.steps,
                                 list(a.tried.data(), a.tried.size()).c_str(),
                                 list(a.accepted.data(), a.accepted.size()).c_str(),
                                 list(a.lengths.data(), a.lengths.size()).c_str(),
                                 list(a.bucket_tried, Acceptance::kBuckets).c_str(),
                                 list(a.bucket_accepted, Acceptance::kBuckets).c_str(), a.kv_ms, a.kv_positions,
                                 a.draft_ms, a.draft_steps);
                }
            }
            if (report)
                std::fclose(report);
        }
        if (!route_counts_path.empty()) {
            RouteCounts counts = m.route_counts();
            if (!routes_at_prompt_end.counts.empty())
                for (size_t e = 0; e < counts.counts.size(); ++e)
                    counts.counts[e] -= routes_at_prompt_end.counts[e];
            if (std::filesystem::exists(route_counts_path))
                counts.add(RouteCounts::read(route_counts_path));
            counts.write(route_counts_path);
            std::printf("route counts added to %s\n", route_counts_path.string().c_str());
        }
        if (profile) {
            const model::DecodeTimes t = m.decode_times();
            const double steps = static_cast<double>(std::max<size_t>(sequence.size(), 1));
            std::printf(
                "decode time per token over %zu steps: %.2f ms; PLE rows %.2f (slowest %.1f, %llu over 5 ms, %llu "
                "hedged), "
                "waiting for the GPU %.2f (expert reads during it %.2f), launches and other host work %.2f\n",
                sequence.size(), t.total / steps, t.ple / steps, t.ple_max, static_cast<unsigned long long>(t.ple_slow),
                static_cast<unsigned long long>(t.ple_hedged), t.gpu / steps, t.expert_reads / steps,
                (t.total - t.ple - t.gpu) / steps);
        }
    } catch (const Error& e) {
        std::fflush(stdout);
        std::fprintf(stderr, "torch run: %s\n", e.what());
        return 1;
    }
    if (logits_file)
        std::fclose(logits_file);
    if (json_file)
        std::fclose(json_file);
    return 0;
}

}  // namespace torch
