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

std::filesystem::path second_shard(const std::filesystem::path& first) {
    std::string name = first.filename().string();
    const size_t at = name.rfind("-00001-of-00002.gguf");
    if (at == std::string::npos)
        throw Error("Pass shard 1 of the model, the file whose name ends in -00001-of-00002.gguf.");
    name.replace(at, 6, "-00002");
    return first.parent_path() / name;
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
    if (!o || !engine || o->struct_size != sizeof(torch_engine_options) || !o->model) {
        copy_error("Pass torch_engine_options with struct_size = sizeof(torch_engine_options) and a model path.", error,
                   error_size);
        return TORCH_ENGINE_INVALID;
    }
    try {
        auto e = std::make_unique<torch_engine>();
        model::Options& opt = e->options;
        opt.shard1 = o->model;
        opt.shard2 = second_shard(opt.shard1);
        opt.context = o->context;
        opt.prefill_columns = o->prefill_columns;
        if (opt.prefill_columns < 1)
            throw Error("The C API prefills prompts: give prefill_columns of 1 to 4,096.");
        if (o->placement)
            opt.placement = o->placement;
        opt.lend_from = o->lend_kv ? 16384 : 0;
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
            opt.reserve.push_back(model::Draft::reservation(opt.shard1, o->draft, variants, opt.context));
        }
        e->model = std::make_unique<model::Model>(opt);
        if (o->draft)
            e->draft = std::make_unique<model::Draft>(*e->model, o->draft, variants, vocab, opt.context);
        e->generator = std::make_unique<runtime::Generator>(*e->model, e->draft.get(), 1, opt.prefill_columns);
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
    if (!ids || n_ids < 1 || stable < 0 || stable > n_ids || !p || p->struct_size != sizeof(torch_engine_params) ||
        p->max_tokens < 1 || (stats && stats->struct_size != sizeof(torch_engine_stats)) ||
        (p->n_stop_ids > 0 && !p->stop_ids) || (p->speculative && !e->draft)) {
        e->error =
            "Pass ids, a stable length of 0 to n_ids, torch_engine_params and torch_engine_stats with their "
            "struct_size, max_tokens of at least 1, and speculative only with a draft file.";
        return TORCH_ENGINE_INVALID;
    }
    e->cancel = false;  // a cancel before this request does not stop it
    try {
        runtime::Request r;
        r.prompt.assign(ids, ids + n_ids);
        r.max_tokens = p->max_tokens;
        r.stop_ids.assign(p->stop_ids, p->stop_ids + std::max(p->n_stop_ids, 0));
        r.spec = p->speculative ? kSpecDrafts : 0;
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
        if (stats)
            *stats = s;
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
        e->model->reset();
        return TORCH_ENGINE_OK;
    } catch (const std::exception& x) {
        e->error = x.what();
        return TORCH_ENGINE_ERROR;
    }
}

const char* torch_engine_error(const torch_engine* e) {
    return e ? e->error.c_str() : "";
}

torch_engine_status torch_engine_last_stats(const torch_engine* e, torch_engine_stats* stats) {
    if (!e || !stats || stats->struct_size != sizeof(torch_engine_stats))
        return TORCH_ENGINE_INVALID;
    *stats = e->last;
    return TORCH_ENGINE_OK;
}

void torch_engine_close(torch_engine* e) {
    delete e;
}

}  // extern "C"
