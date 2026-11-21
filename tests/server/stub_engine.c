/* A stand-in for libtorch_engine.so with the same C API (include/torch_engine.h) and no model, for the server's tests
 * on any machine: deterministic tokens, the session rule, windows, chunks, callbacks and cancel as the engine has them.
 *
 * options.model names a text file of comma-separated token ids, read again at every request: the answer, cut at
 * max_tokens and after a stop id (greedy or sampled alike). Windows of TORCH_STUB_WINDOW tokens
 * when speculative (default 4), else 1; TORCH_STUB_DELAY_MS milliseconds after each prefill chunk and window (default
 * 0). The torch_stub_* functions let a test see the last request. */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "torch_engine.h"

#define MAX_IDS 300000

enum { MAX_PREFIXES = 12 };
struct saved_prefix {
    int32_t* ids;
    int32_t n;
    int kind;
    unsigned age;
};

struct torch_engine {
    char answer_path[4096];
    int has_draft, cache, disk_only;
    struct saved_prefix prefixes[MAX_PREFIXES];
    unsigned age;
    int32_t context, prefill, window, delay_ms;
    int32_t* sequence; /* the session: what was fed, and how much of it the next request may keep */
    int32_t n_sequence, stable;
    atomic_int cancel, busy;
    char error[256];
    torch_engine_stats last;
};

static atomic_int running; /* a request is in torch_engine_generate */

static struct {
    int32_t ids[MAX_IDS], n, stable, generated, stop_ids[64];
    int32_t checkpoints[MAX_IDS];
    uint8_t kinds[MAX_IDS];
    torch_engine_params params; /* its stop_ids point at the copy above: the caller's array is gone after the call */
    int cancelled_by_progress, stopped_by_tokens, requests;
} last_request;

static void pause_ms(int32_t ms) {
    if (ms > 0) {
        struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L};
        nanosleep(&t, NULL);
    }
}

static int32_t env_int(const char* name, int32_t fallback) {
    const char* v = getenv(name);
    return v && *v ? (int32_t)atoi(v) : fallback;
}

uint32_t torch_engine_abi_version(void) {
    return TORCH_ENGINE_ABI_VERSION;
}

static int last_fast_prefill = -1;

torch_engine_status torch_engine_open(const torch_engine_options* o, torch_engine** engine, char* error,
                                      size_t error_size) {
    const int abi1 = o && o->struct_size == offsetof(torch_engine_options, fast_prefill); /* ends before it */
    const int abi2 = o && o->struct_size == offsetof(torch_engine_options, prefix_memory_bytes);
    if (!o || !engine || (o->struct_size != sizeof(torch_engine_options) && !abi1 && !abi2) || !o->model ||
        o->context < 1 || o->prefill_columns < 1) {
        if (error && error_size)
            snprintf(error, error_size, "stub: bad options");
        return TORCH_ENGINE_INVALID;
    }
    torch_engine* e = calloc(1, sizeof *e);
    snprintf(e->answer_path, sizeof e->answer_path, "%s", o->model);
    e->has_draft = o->draft != NULL;
    if (!abi1 && !abi2) {
        e->cache = o->prefix_memory_bytes || (o->prefix_disk_dir && o->prefix_disk_bytes);
        e->disk_only = o->prefix_memory_bytes == 0 && o->prefix_disk_bytes;
    }
    e->context = o->context, e->prefill = o->prefill_columns;
    last_fast_prefill = abi1 ? 0 : o->fast_prefill != 0;
    e->window = env_int("TORCH_STUB_WINDOW", 4), e->delay_ms = env_int("TORCH_STUB_DELAY_MS", 0);
    e->sequence = malloc(sizeof(int32_t) * MAX_IDS);
    *engine = e;
    return TORCH_ENGINE_OK;
}

static int32_t read_answer(const char* path, int32_t* out, int32_t cap) {
    FILE* f = fopen(path, "r");
    int32_t n = 0;
    long v;
    if (!f)
        return 0;
    while (n < cap && fscanf(f, "%ld", &v) == 1) {
        out[n++] = (int32_t)v;
        if (fgetc(f) == EOF)
            break;
    }
    fclose(f);
    return n;
}

static void remember(torch_engine* e, const int32_t* ids, int32_t n, int kind) {
    struct saved_prefix* slot = NULL;
    for (int i = 0; i < MAX_PREFIXES; ++i) {
        struct saved_prefix* p = &e->prefixes[i];
        if (p->n == n && p->kind == kind && !memcmp(p->ids, ids, (size_t)n * sizeof(*ids))) {
            p->age = ++e->age;
            return;
        }
        if (!slot || (!p->n && slot->n) ||
            (p->n && slot->n && (p->kind > slot->kind || (p->kind == slot->kind && p->age < slot->age))))
            slot = p;
    }
    int32_t* copy = malloc((size_t)n * sizeof(*ids));
    if (!copy)
        return;
    memcpy(copy, ids, (size_t)n * sizeof(*ids));
    free(slot->ids);
    *slot = (struct saved_prefix){copy, n, kind, ++e->age};
}

static int valid_stats(uint32_t size) {
    return size == sizeof(torch_engine_stats) || size == offsetof(torch_engine_stats, restored_tier);
}
static void copy_stats(torch_engine_stats* out, const torch_engine_stats* in) {
    const uint32_t size = out->struct_size;
    memcpy(out, in, size);
    out->struct_size = size;
}

torch_engine_status torch_engine_generate(torch_engine* e, const int32_t* ids, int32_t n_ids, int32_t stable,
                                          const torch_engine_params* p, torch_engine_on_tokens on_tokens,
                                          torch_engine_on_progress on_progress, void* user, torch_engine_stats* stats) {
    if (!e)
        return TORCH_ENGINE_INVALID;
    if (atomic_exchange(&e->busy, 1))
        return TORCH_ENGINE_BUSY;
    torch_engine_status status = TORCH_ENGINE_OK;
    if (!ids || n_ids < 1 || n_ids > MAX_IDS || stable < 0 || stable > n_ids || !p ||
        (p->struct_size != sizeof(torch_engine_params) &&
         p->struct_size != offsetof(torch_engine_params, checkpoints)) ||
        p->max_tokens < 1 || (stats && !valid_stats(stats->struct_size)) || (p->n_stop_ids > 0 && !p->stop_ids) ||
        (p->speculative && !e->has_draft) || n_ids + p->max_tokens > e->context) {
        snprintf(e->error, sizeof e->error, "stub: invalid request");
        atomic_store(&e->busy, 0);
        return TORCH_ENGINE_INVALID;
    }
    const int marks = p->struct_size == sizeof(*p) ? p->n_checkpoints : 0;
    if (marks < 0 || marks > n_ids || (marks && (!p->checkpoints || !p->checkpoint_kinds))) {
        atomic_store(&e->busy, 0);
        return TORCH_ENGINE_INVALID;
    }
    for (int i = 0; i < marks; ++i)
        if (p->checkpoints[i] < 1 || p->checkpoints[i] > (stable ? stable : n_ids) ||
            (p->checkpoint_kinds[i] != 1 && p->checkpoint_kinds[i] != 2)) {
            atomic_store(&e->busy, 0);
            return TORCH_ENGINE_INVALID;
        }
    atomic_store(&e->cancel, 0);
    atomic_store(&running, 1);
    if (stable == 0)
        stable = n_ids;
    memcpy(last_request.ids, ids, sizeof(int32_t) * (size_t)n_ids);
    last_request.n = n_ids, last_request.stable = stable, last_request.generated = 0;
    memset(&last_request.params, 0, sizeof last_request.params);
    memcpy(&last_request.params, p, p->struct_size);
    if (marks) {
        memcpy(last_request.checkpoints, p->checkpoints, (size_t)marks * sizeof(int32_t));
        memcpy(last_request.kinds, p->checkpoint_kinds, (size_t)marks);
        last_request.params.checkpoints = last_request.checkpoints;
        last_request.params.checkpoint_kinds = last_request.kinds;
    }
    last_request.params.n_stop_ids = p->n_stop_ids < 0 ? 0 : p->n_stop_ids < 64 ? p->n_stop_ids : 64;
    if (last_request.params.n_stop_ids > 0)
        memcpy(last_request.stop_ids, p->stop_ids, sizeof(int32_t) * (size_t)last_request.params.n_stop_ids);
    last_request.params.stop_ids = last_request.stop_ids;
    last_request.cancelled_by_progress = 0, last_request.stopped_by_tokens = 0, ++last_request.requests;

    /* the session: keep the previous request's stable prefix when this prompt starts with it */
    int32_t cached = 0;
    if (e->stable > 0 && n_ids >= e->stable && memcmp(ids, e->sequence, sizeof(int32_t) * (size_t)e->stable) == 0)
        cached = e->stable;
    int tier = cached ? 1 : 0;
    if (e->cache)
        for (int i = 0; i < MAX_PREFIXES; ++i) {
            struct saved_prefix* entry = &e->prefixes[i];
            const int32_t n = entry->n;
            if (n <= cached || n >= n_ids || n > stable || memcmp(ids, entry->ids, (size_t)n * sizeof(*ids)))
                continue;
            cached = n;
            tier = (n <= e->stable && !memcmp(e->sequence, ids, (size_t)n * sizeof(*ids))) ? 1 : (e->disk_only ? 3 : 2);
            entry->age = ++e->age;
        }
    int32_t fed = cached, generated = 0;
    int finish = TORCH_ENGINE_FINISH_LENGTH;
    clock_t t0 = clock();
    /* the prompt: chunks up to stable, then windows of 8 */
    while (fed < n_ids) {
        fed += fed < stable ? (stable - fed < e->prefill ? stable - fed : e->prefill)
                            : (n_ids - fed < 8 ? n_ids - fed : 8);
        pause_ms(e->delay_ms);
        if (atomic_load(&e->cancel) || (on_progress && on_progress(user, fed, n_ids))) {
            last_request.cancelled_by_progress = 1;
            status = TORCH_ENGINE_CANCELLED, finish = TORCH_ENGINE_FINISH_CANCELLED;
            break;
        }
    }
    double prompt_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    static int32_t answer[MAX_IDS];
    const int32_t count = status == TORCH_ENGINE_OK ? read_answer(e->answer_path, answer, MAX_IDS) : 0;
    uint64_t seed = p->has_seed ? p->seed : (uint64_t)time(NULL) * 2654435761u;
    int32_t at = 0;
    const int32_t window = p->speculative ? e->window : 1;
    int32_t windows = 0;
    while (status == TORCH_ENGINE_OK && generated < p->max_tokens && count > 0) {
        int32_t out[64], n = 0, stop = 0;
        while (n < window && generated + n < p->max_tokens && !stop) {
            out[n] = answer[(at++) % count];
            for (int32_t i = 0; i < p->n_stop_ids; ++i)
                stop |= out[n] == p->stop_ids[i];
            ++n;
        }
        generated += n, ++windows;
        if (on_tokens && on_tokens(user, out, n)) {
            last_request.stopped_by_tokens = 1;
            finish = TORCH_ENGINE_FINISH_STOP;
            break;
        }
        if (stop) {
            finish = TORCH_ENGINE_FINISH_STOP;
            break;
        }
        pause_ms(e->delay_ms);
        if (generated < p->max_tokens &&
            (atomic_load(&e->cancel) || (on_progress && on_progress(user, n_ids + generated, n_ids)))) {
            last_request.cancelled_by_progress = 1;
            status = TORCH_ENGINE_CANCELLED, finish = TORCH_ENGINE_FINISH_CANCELLED;
        }
    }
    last_request.generated = generated;
    /* what the next request may keep: the stable prefix, when the whole prompt was fed */
    memcpy(e->sequence, ids, sizeof(int32_t) * (size_t)n_ids);
    e->stable = status == TORCH_ENGINE_OK ? stable : 0;
    torch_engine_stats s;
    memset(&s, 0, sizeof s);
    s.struct_size = sizeof s;
    s.prompt_tokens = n_ids, s.cached_tokens = cached, s.generated_tokens = generated, s.finish = finish;
    s.seed = seed, s.prompt_seconds = prompt_s > 0 ? prompt_s : 1e-6, s.decode_seconds = 1e-3 * (windows + 1);
    s.tokens_per_window = windows ? (double)generated / windows : 1.0;
    s.resident_experts = 100, s.total_experts = 24576;
    if (e->cache && status == TORCH_ENGINE_OK) {
        for (int i = 0; i < marks; ++i)
            remember(e, ids, p->checkpoints[i], p->checkpoint_kinds[i]);
        remember(e, ids, stable, 2);
    }
    s.restored_tier = tier;
    for (int i = 0; i < MAX_PREFIXES; ++i)
        if (e->prefixes[i].n) {
            if (e->disk_only)
                ++s.checkpoints_disk;
            else
                ++s.checkpoints_memory;
        }
    e->last = s;
    if (stats)
        copy_stats(stats, &s);
    atomic_store(&running, 0);
    atomic_store(&e->busy, 0);
    return status;
}

void torch_engine_cancel(torch_engine* e) {
    if (e)
        atomic_store(&e->cancel, 1);
}

torch_engine_status torch_engine_reset(torch_engine* e) {
    if (!e)
        return TORCH_ENGINE_INVALID;
    if (atomic_exchange(&e->busy, 1))
        return TORCH_ENGINE_BUSY;
    e->stable = 0;
    for (int i = 0; i < MAX_PREFIXES; ++i) {
        free(e->prefixes[i].ids);
        e->prefixes[i] = (struct saved_prefix){0};
    }
    atomic_store(&e->busy, 0);
    return TORCH_ENGINE_OK;
}

torch_engine_status torch_engine_forget(torch_engine* e) {
    return torch_engine_reset(e);
}

const char* torch_engine_error(const torch_engine* e) {
    return e ? e->error : "";
}

torch_engine_status torch_engine_last_stats(const torch_engine* e, torch_engine_stats* stats) {
    if (!e || !stats || !valid_stats(stats->struct_size))
        return TORCH_ENGINE_INVALID;
    copy_stats(stats, &e->last);
    return TORCH_ENGINE_OK;
}

void torch_engine_close(torch_engine* e) {
    if (e) {
        for (int i = 0; i < MAX_PREFIXES; ++i)
            free(e->prefixes[i].ids);
        free(e->sequence);
        free(e);
    }
}

/* For the tests: the last request as the stub saw it. */
TORCH_ENGINE_API int32_t torch_stub_last_ids(int32_t* out, int32_t cap) {
    const int32_t n = last_request.n < cap ? last_request.n : cap;
    memcpy(out, last_request.ids, sizeof(int32_t) * (size_t)n);
    return last_request.n;
}
TORCH_ENGINE_API int32_t torch_stub_last_stable(void) {
    return last_request.stable;
}
TORCH_ENGINE_API void torch_stub_last_params(torch_engine_params* p) {
    *p = last_request.params;
}
TORCH_ENGINE_API int32_t torch_stub_last_generated(void) {
    return last_request.generated;
}
TORCH_ENGINE_API int torch_stub_last_cancelled(void) {
    return last_request.cancelled_by_progress;
}
TORCH_ENGINE_API int torch_stub_last_stopped(void) {
    return last_request.stopped_by_tokens;
}
TORCH_ENGINE_API int torch_stub_running(void) {
    return atomic_load(&running);
}
TORCH_ENGINE_API int torch_stub_last_fast_prefill(void) { /* of the last torch_engine_open */
    return last_fast_prefill;
}
TORCH_ENGINE_API int torch_stub_requests(void) {
    return last_request.requests;
}
