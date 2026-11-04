// `torch run`: feeds token ids through the model one at a time, then decodes greedily, printing each step's most
// likely next tokens. It is the engine's M2 test driver: the model parity test (tests/model/) reads its --json output,
// and the residency test compares its --logits output byte for byte.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/budget.h"
#include "core/error.h"
#include "model/model.h"

namespace torch {
namespace {

constexpr const char* kUsage =
    "usage: torch run [options] MODEL-00001-of-00002.gguf --tokens ID,ID,...\n"
    "\n"
    "Feeds the tokens one at a time, then generates greedily, printing each step's top log-probabilities.\n"
    "Shard 2 (the n-gram table) is found next to shard 1 by its split name.\n"
    "\n"
    "  --tokens LIST   comma-separated prompt token ids\n"
    "  -n N            tokens to generate after the prompt (default 16)\n"
    "  --top K         log-probabilities to print per step (default 5)\n"
    "  --context N     tokens the KV cache holds (default and maximum 2048)\n"
    "  --resident K    experts per layer to keep resident (default: as many as fit in memory)\n"
    "  --logits FILE   write every step's raw F32 logits to FILE\n"
    "  --json FILE     write every step's top log-probabilities to FILE, one JSON object per line\n"
    "  --profile       print where decode time went: PLE rows, routing waits, expert reads, the rest\n";

using Clock = std::chrono::steady_clock;

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
    int generate = 16, top = 5;
    bool profile = false;
    model::Options options;
    std::FILE* logits_file = nullptr;
    std::FILE* json_file = nullptr;
    try {
        for (int i = 0; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto value = [&]() -> const char* {
                if (i + 1 >= argc)
                    throw Error(std::string(arg) + " needs a value.");
                return argv[++i];
            };
            if (arg == "--tokens") {
                prompt = parse_tokens(value());
            } else if (arg == "-n") {
                generate = parse_count(value(), "-n", 0);
            } else if (arg == "--top") {
                top = parse_count(value(), "--top", 1);
            } else if (arg == "--context") {
                options.context = parse_count(value(), "--context", 1);
            } else if (arg == "--resident") {
                options.resident = parse_count(value(), "--resident", 0);
            } else if (arg == "--logits") {
                const char* path = value();
                if (!(logits_file = std::fopen(path, "wb")))
                    throw Error(std::string("Cannot write ") + path + ".");
            } else if (arg == "--json") {
                const char* path = value();
                if (!(json_file = std::fopen(path, "w")))
                    throw Error(std::string("Cannot write ") + path + ".");
            } else if (arg == "--profile") {
                profile = true;
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

        const auto load_start = Clock::now();
        model::Model m(options);
        const double load_s = std::chrono::duration<double>(Clock::now() - load_start).count();
        std::printf("%s", m.budget().table().c_str());
        std::printf("loaded in %.1f s; MemAvailable now %.2f GiB\n\n", load_s,
                    static_cast<double>(mem_available()) / static_cast<double>(kGiB));

        std::vector<int32_t> sequence = prompt;
        double decode_ms = 0;
        int decoded = 0;
        for (size_t i = 0; i < sequence.size(); ++i) {
            const auto start = Clock::now();
            const std::span<const float> logits = m.decode(sequence[i]);
            if (logits.empty())
                throw Error(m.error());
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            const bool generating = i + 1 >= prompt.size();
            if (i >= prompt.size()) {  // generated tokens only: the speed of decoding, not of feeding the prompt
                decode_ms += ms;
                ++decoded;
            }
            const std::vector<Top> best = top_k(logits, top);
            std::printf("pos %4zu  in %6d  ->", i, sequence[i]);
            for (const Top& t : best)
                std::printf("  %6d %8.4f", t.token, t.logprob);
            std::printf("   (%.1f ms)\n", ms);
            if (logits_file)
                std::fwrite(logits.data(), sizeof(float), logits.size(), logits_file);
            if (json_file) {
                std::fprintf(json_file, "{\"pos\": %zu, \"token\": %d, \"top\": [", i, sequence[i]);
                for (size_t j = 0; j < best.size(); ++j)
                    std::fprintf(json_file, "%s[%d, %.7g]", j ? ", " : "", best[j].token, best[j].logprob);
                std::fprintf(json_file, "]}\n");
            }
            if (generating && sequence.size() < prompt.size() + static_cast<size_t>(generate))
                sequence.push_back(best[0].token);
        }

        const model::ExpertStats s = m.expert_stats();
        std::printf("\nprompt %zu tokens, generated %d", prompt.size(), decoded);
        if (decoded > 0)
            std::printf(": %.2f tok/s (%.1f ms per token)", 1000.0 * decoded / decode_ms, decode_ms / decoded);
        std::printf(
            "\nexperts: %d resident per layer; %llu routed hits, %llu misses (%.1f%% hits)\n", s.resident,
            static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.misses),
            100.0 * static_cast<double>(s.hits) / static_cast<double>(std::max<uint64_t>(s.hits + s.misses, 1)));
        std::printf("MemAvailable at the end %.2f GiB\n",
                    static_cast<double>(mem_available()) / static_cast<double>(kGiB));
        if (profile) {
            const model::DecodeTimes t = m.decode_times();
            const double steps = static_cast<double>(std::max<size_t>(sequence.size(), 1));
            std::printf(
                "decode time per token over %zu steps: %.2f ms; PLE rows %.2f, routing waits %.2f, expert reads "
                "%.2f, final wait %.2f, other host work %.2f\n",
                sequence.size(), t.total / steps, t.ple / steps, t.routing / steps, t.expert_reads / steps,
                t.finish / steps, (t.total - t.ple - t.routing - t.expert_reads - t.finish) / steps);
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
