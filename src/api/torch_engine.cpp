// The C API (include/torch_engine.h) over Model, Draft and runtime::Generator. No exception leaves these functions:
// each catches, records the message on the handle and returns a status.
#include "torch_engine.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/error.h"
#include "model/draft.h"
#include "model/model.h"
#include "runtime/generator.h"

using namespace torch;

struct torch_engine {
    model::Options options;
    std::unique_ptr<model::Model> model;
    std::unique_ptr<model::Draft> draft;
    std::unique_ptr<runtime::Generator> generator;
    int swaps = 0;
    std::mutex busy;                  // one request at a time
    std::atomic<bool> cancel{false};  // torch_engine_cancel(): read after each window and chunk
    std::string error;
    torch_engine_stats last{};
};

namespace {

constexpr int kSpecDrafts = 5;
constexpr size_t kOldParams = offsetof(torch_engine_params, checkpoints);
constexpr size_t kOldStats = offsetof(torch_engine_stats, restored_tier);
bool stats_size(uint32_t size) {
    return size == sizeof(torch_engine_stats) || size == kOldStats;
}
void write_stats(torch_engine_stats* out, const torch_engine_stats& s) {
    const auto n = out->struct_size;
    std::memcpy(out, &s, n);
    out->struct_size = n;
}

void copy_error(const std::string& message, char* out, size_t size) {
    if (out && size > 0) {
        const size_t n = std::min(message.size(), size - 1);
        std::memcpy(out, message.data(), n);
        out[n] = '\0';
    }
}

}  // namespace

extern "C" {

uint32_t torch_engine_abi_version(void) {
    return TORCH_ENGINE_ABI_VERSION;
}

torch_engine_status torch_engine_open(const torch_engine_options* o, torch_engine** engine, char* error,
                                      size_t error_size) {
    if (engine)
        *engine = nullptr;
    // ABI 1 callers end the struct before fast_prefill
    const bool abi1 = o && o->struct_size == offsetof(torch_engine_options, fast_prefill);
    const bool abi2 = o && o->struct_size == offsetof(torch_engine_options, prefix_memory_bytes);
    if (!o || !engine || (o->struct_size != sizeof(torch_engine_options) && !abi1 && !abi2) || !o->model) {
        copy_error("Pass torch_engine_options with struct_size = sizeof(torch_engine_options) and a model path.", error,
                   error_size);
        return TORCH_ENGINE_INVALID;
    }
    try {
        auto e = std::make_unique<torch_engine>();
        model::Options& opt = e->options;
        runtime::PrefixOptions prefix;
        if (!abi1 && !abi2) {
            prefix.memory_bytes = o->prefix_memory_bytes;
            if (o->prefix_disk_dir)
                prefix.directory = o->prefix_disk_dir;
            prefix.disk_bytes = o->prefix_disk_bytes;
        }
        prefix.reserve(opt.reserve);
        opt.model = o->model;
        opt.context = o->context;
        opt.prefill_columns = o->prefill_columns;
        if (opt.prefill_columns < 1)
            throw Error("The C API prefills prompts: give prefill_columns of 1 to 4,096.");
        if (o->placement)
            opt.placement = o->placement;
        opt.lend_from = o->lend_kv ? 16384 : 0;
        opt.fast_prefill = !abi1 && o->fast_prefill != 0;
        opt.swaps_per_window = 0;  // a prompt's windows swap nothing; generated ones up to swaps_per_window
        opt.logits = false;        // only the drawn tokens leave the GPU
        e->swaps = o->swaps_per_window;
        std::vector<model::DraftVariant> variants(1);
        std::vector<int32_t> vocab;
        if (o->draft) {
            variants[0].head_rows = o->draft_head_rows;
            if (o->draft_vocab) {
                std::ifstream in(o->draft_vocab);
                if (!in)
                    throw Error(std::string("Cannot read ") + o->draft_vocab + ".");
                for (long id; in >> id;)
                    vocab.push_back(static_cast<int32_t>(id));
            }
            opt.reserve.push_back(model::Draft::reservation(opt.model, o->draft, variants, opt.context));
        }
        e->model = std::make_unique<model::Model>(opt);
        if (o->draft)
            e->draft = std::make_unique<model::Draft>(*e->model, o->draft, variants, vocab, opt.context);
        e->generator = std::make_unique<runtime::Generator>(*e->model, e->draft.get(), 1, opt.prefill_columns, prefix);
        *engine = e.release();
        return TORCH_ENGINE_OK;
    } catch (const std::exception& x) {
        copy_error(x.what(), error, error_size);
        return TORCH_ENGINE_ERROR;
    }
}

torch_engine_status torch_engine_generate(torch_engine* e, const int32_t* ids, int32_t n_ids, int32_t stable,
                                          const torch_engine_params* p, torch_engine_on_tokens on_tokens,
                                          torch_engine_on_progress on_progress, void* user, torch_engine_stats* stats) {
    if (!e)
        return TORCH_ENGINE_INVALID;
    const std::unique_lock lock(e->busy, std::try_to_lock);
    if (!lock.owns_lock())
        return TORCH_ENGINE_BUSY;
    e->error.clear();
    const bool extended = p && p->struct_size == sizeof(torch_engine_params);
    if (!ids || n_ids < 1 || stable < 0 || stable > n_ids || !p || (!extended && p->struct_size != kOldParams) ||
        p->max_tokens < 1 || (stats && !stats_size(stats->struct_size)) || (p->n_stop_ids > 0 && !p->stop_ids) ||
        (p->speculative && !e->draft)) {
        e->error =
            "Pass ids, a stable length of 0 to n_ids, torch_engine_params and torch_engine_stats with their "
            "struct_size, max_tokens of at least 1, and speculative only with a draft file.";
        return TORCH_ENGINE_INVALID;
    }
    if (extended) {
        if (p->n_checkpoints < 0 || p->n_checkpoints > n_ids ||
            (p->n_checkpoints && (!p->checkpoints || !p->checkpoint_kinds))) {
            e->error = "Give one position and kind per checkpoint.";
            return TORCH_ENGINE_INVALID;
        }
        for (int i = 0; i < p->n_checkpoints; ++i) {
            if (p->checkpoints[i] < 1 || p->checkpoints[i] > (stable ? stable : n_ids) ||
                (p->checkpoint_kinds[i] != 1 && p->checkpoint_kinds[i] != 2)) {
                e->error = "The checkpoint position or kind is outside the stable prompt.";
                return TORCH_ENGINE_INVALID;
            }
        }
    }
    e->cancel = false;  // a cancel before this request does not stop it
    try {
        runtime::Request r;
        r.prompt.assign(ids, ids + n_ids);
        r.max_tokens = p->max_tokens;
        if (p->n_stop_ids > 0)
            r.stop_ids.assign(p->stop_ids, p->stop_ids + p->n_stop_ids);
        if (extended)
            for (int i = 0; i < p->n_checkpoints; ++i)
                r.checkpoints.emplace_back(static_cast<size_t>(p->checkpoints[i]),
                                           static_cast<PrefixKind>(p->checkpoint_kinds[i]));
        r.spec = p->speculative ? kSpecDrafts : 0;
        r.copy_depth = p->speculative ? model::kMaxColumns - 1 : 0;
        r.draft_depth = r.spec;  // without speculation the draft layer only keeps its K/V rows
        r.swaps_after_prompt = e->swaps;
        r.window = model::kMaxColumns;
        r.stable = static_cast<size_t>(stable);
        r.rewind_only = true;
        if ((r.sample = p->sample != 0)) {
            r.sampling.temperature = p->temperature;
            r.sampling.top_k = p->top_k;
            r.sampling.top_p = p->top_p;
            r.sampling.min_p = p->min_p;
            r.sampling.repetition_penalty = p->repetition_penalty;
            r.sampling.presence_penalty = p->presence_penalty;
            r.sampling.frequency_penalty = p->frequency_penalty;
            if (p->has_seed)
                r.seed = p->seed;
        }
        runtime::Observer o;
        if (on_tokens)
            o.on_tokens = [&](std::span<const int32_t> t) {
                return on_tokens(user, t.data(), static_cast<int32_t>(t.size())) == 0;
            };
        o.on_progress = [&](size_t fed, size_t prompt) {
            return !e->cancel.load() &&
                   !(on_progress && on_progress(user, static_cast<int32_t>(fed), static_cast<int32_t>(prompt)) != 0);
        };
        const runtime::Result res = e->generator->generate(r, o);
        const model::ExpertStats x = e->model->expert_stats();
        torch_engine_stats& s = e->last;
        s = {};
        s.struct_size = sizeof(torch_engine_stats);
        s.prompt_tokens = n_ids;
        s.cached_tokens = res.resume.rewound ? static_cast<int32_t>(res.resume.at) : 0;
        s.generated_tokens =
            res.sequence.size() > r.prompt.size() ? static_cast<int32_t>(res.sequence.size() - r.prompt.size()) : 0;
        s.finish = res.cancelled ? TORCH_ENGINE_FINISH_CANCELLED
                   : res.stopped ? TORCH_ENGINE_FINISH_STOP
                                 : TORCH_ENGINE_FINISH_LENGTH;
        s.seed = res.seed;
        s.prompt_seconds = res.prompt_s;
        s.decode_seconds =
            (r.spec > 0 ? res.spec_ms + (res.variants.empty() ? 0.0 : res.variants[0].draft_ms) : res.decode_ms) /
            1000.0;
        s.tokens_per_window = res.windows > 0 ? static_cast<double>(res.window_tokens) / res.windows : 1.0;
        s.resident_experts = x.resident, s.total_experts = x.experts;
        s.expert_hits = x.hits, s.expert_misses = x.misses;
        s.restored_tier = static_cast<int32_t>(res.prefix.tier);
        s.restore_seconds = res.prefix.restore_seconds;
        s.parked_bytes = res.prefix.parked_bytes;
        s.checkpoints_memory = res.prefix.in_memory;
        s.checkpoints_disk = res.prefix.on_disk;
        if (stats)
            write_stats(stats, s);
        return res.cancelled ? TORCH_ENGINE_CANCELLED : TORCH_ENGINE_OK;
    } catch (const std::exception& x) {
        e->error = x.what();
        return TORCH_ENGINE_ERROR;
    }
}

void torch_engine_cancel(torch_engine* e) {
    if (e)
        e->cancel = true;
}

torch_engine_status torch_engine_reset(torch_engine* e) {
    if (!e)
        return TORCH_ENGINE_INVALID;
    const std::unique_lock lock(e->busy, std::try_to_lock);
    if (!lock.owns_lock())
        return TORCH_ENGINE_BUSY;
    try {
        e->generator->forget();
        return TORCH_ENGINE_OK;
    } catch (const std::exception& x) {
        e->error = x.what();
        return TORCH_ENGINE_ERROR;
    }
}

torch_engine_status torch_engine_forget(torch_engine* e) {
    return torch_engine_reset(e);
}

const char* torch_engine_error(const torch_engine* e) {
    return e ? e->error.c_str() : "";
}

torch_engine_status torch_engine_last_stats(const torch_engine* e, torch_engine_stats* stats) {
    if (!e || !stats || !stats_size(stats->struct_size))
        return TORCH_ENGINE_INVALID;
    write_stats(stats, e->last);
    return TORCH_ENGINE_OK;
}

void torch_engine_close(torch_engine* e) {
    delete e;
}

}  // extern "C"
