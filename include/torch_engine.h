/* The engine's C API (milestone M7): token ids in, token ids out, for torch_server and other callers.
 *
 * One handle holds one loaded model and one sequence, and runs one request at a time. A request's prompt goes as
 * prefill chunks up to its `stable` length and as windows after it; when the next request's prompt starts with the same
 * tokens up to that length, the model goes back there (a session) instead of feeding the conversation again, and
 * stats.cached_tokens says how many tokens it kept. Greedy or sampled, with or without speculation: a sampled request
 * with a seed draws the same tokens either way. Calls return a status; nothing throws across this boundary, and
 * torch_engine_error() describes the last failure. Every struct starts with its own size, set by the caller with
 * sizeof, so a library and a caller built from different versions of this header refuse each other instead of
 * misreading.
 *
 * Thread safety: torch_engine_cancel() may be called from any thread; every other call on a handle must come from one
 * thread at a time. Callbacks run on the thread that called torch_engine_generate(). */
#ifndef TORCH_ENGINE_H
#define TORCH_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define TORCH_ENGINE_API __attribute__((visibility("default")))
#else
#define TORCH_ENGINE_API
#endif

#define TORCH_ENGINE_ABI_VERSION 2

typedef enum torch_engine_status {
    TORCH_ENGINE_OK = 0,
    TORCH_ENGINE_ERROR = 1,     /* the engine failed: see torch_engine_error() */
    TORCH_ENGINE_INVALID = 2,   /* an argument is out of range or a struct size does not match */
    TORCH_ENGINE_CANCELLED = 3, /* torch_engine_cancel() or the progress callback stopped the request */
    TORCH_ENGINE_BUSY = 4       /* another request is running on this handle */
} torch_engine_status;

typedef enum torch_engine_finish {
    TORCH_ENGINE_FINISH_LENGTH = 0, /* max_tokens generated */
    TORCH_ENGINE_FINISH_STOP = 1,   /* a stop id, or the token callback returned nonzero */
    TORCH_ENGINE_FINISH_CANCELLED = 2
} torch_engine_finish;

typedef struct torch_engine torch_engine;

typedef struct torch_engine_options {
    uint32_t struct_size;     /* sizeof(torch_engine_options) */
    const char* model;        /* shard 1, MODEL-00001-of-00002.gguf; shard 2 is found next to it */
    const char* draft;        /* the draft-layer file, or NULL: no speculation */
    const char* draft_vocab;  /* the draft head's token list (NULL: the full vocabulary) */
    int32_t draft_head_rows;  /* rows of the draft head taken from draft_vocab (0: all of them) */
    const char* placement;    /* a route-count file for the resident experts, or NULL */
    int32_t context;          /* tokens the KV cache holds, 1 to 262,144 */
    int32_t prefill_columns;  /* the widest prefill chunk, 1 to 4,096 */
    int32_t lend_kv;          /* nonzero: the KV cache's unused tail holds experts until the sequence reaches it */
    int32_t swaps_per_window; /* experts made resident after each generated window (0: none; torch_server uses 128).
                               * Without lend_kv, the misses a window holds when its staging fills are swapped in then,
                               * within this budget. */
    int32_t fast_prefill;     /* nonzero: prefill's BF16 products on the tensor cores with the activations rounded to
                               * BF16 (ABI 2): a shorter time to the first token, but prefill is then not bitwise the
                               * one-token path, so a greedy answer can differ. A caller built for ABI 1 passes
                               * struct_size = offsetof(torch_engine_options, fast_prefill) and gets the exact path. */
} torch_engine_options;

typedef struct torch_engine_params {
    uint32_t struct_size; /* sizeof(torch_engine_params) */
    int32_t max_tokens;   /* tokens to generate, at least 1 */
    int32_t speculative;  /* nonzero: windows of up to 5 drafts (needs a draft file); the tokens are the same */
    int32_t sample;       /* zero: greedy (the argmax); nonzero: sampled with the fields below */
    float temperature;    /* 0 is greedy; penalties still apply */
    int32_t top_k;        /* 0: off */
    float top_p;          /* 1: off */
    float min_p;          /* 0: off */
    float repetition_penalty, presence_penalty, frequency_penalty; /* over this request's generated tokens */
    int32_t has_seed;                                              /* zero: a random seed, returned in stats.seed */
    uint64_t seed;
    const int32_t* stop_ids; /* generation stops after any of these (kept in the output) */
    int32_t n_stop_ids;
} torch_engine_params;

typedef struct torch_engine_stats {
    uint32_t struct_size; /* sizeof(torch_engine_stats) */
    int32_t prompt_tokens;
    int32_t cached_tokens; /* prompt tokens kept from the previous request (the session), not fed again */
    int32_t generated_tokens;
    int32_t finish;           /* torch_engine_finish */
    uint64_t seed;            /* the seed a sampled request used */
    double prompt_seconds;    /* the prompt's chunks and windows: the time to the first token */
    double decode_seconds;    /* the generated tokens' windows */
    double tokens_per_window; /* speculative: generated tokens per verify window */
    int32_t resident_experts, total_experts;
    uint64_t expert_hits, expert_misses; /* routed experts found resident or read from the SSD, over the handle */
} torch_engine_stats;

/* New tokens of the running request; returning nonzero stops it after them (finish STOP). */
typedef int (*torch_engine_on_tokens)(void* user, const int32_t* tokens, int32_t n);
/* After each prefill chunk and each window: positions fed so far and the prompt's length; nonzero cancels. */
typedef int (*torch_engine_on_progress)(void* user, int32_t fed, int32_t prompt_tokens);

TORCH_ENGINE_API uint32_t torch_engine_abi_version(void);

/* Loads the model (tens of seconds). On failure returns a status, *engine stays NULL and error (when error_size > 0)
 * holds the reason. */
TORCH_ENGINE_API torch_engine_status torch_engine_open(const torch_engine_options* options, torch_engine** engine,
                                                       char* error, size_t error_size);

/* Runs one request: ids[0 .. n_ids) is the whole prompt (the conversation so far), stable (0 to n_ids; 0: n_ids) the
 * length a later turn's prompt will share (the template rendered without the generation prompt). Writes the
 * generated tokens through on_tokens (may be NULL) and the request's numbers to *stats (may be NULL). */
TORCH_ENGINE_API torch_engine_status torch_engine_generate(torch_engine* engine, const int32_t* ids, int32_t n_ids,
                                                           int32_t stable, const torch_engine_params* params,
                                                           torch_engine_on_tokens on_tokens,
                                                           torch_engine_on_progress on_progress, void* user,
                                                           torch_engine_stats* stats);

/* From any thread: the running request stops before its next window or chunk and returns TORCH_ENGINE_CANCELLED.
 * Without a running request it does nothing. */
TORCH_ENGINE_API void torch_engine_cancel(torch_engine* engine);

/* Starts a new, empty sequence: the next request feeds its whole prompt. */
TORCH_ENGINE_API torch_engine_status torch_engine_reset(torch_engine* engine);

/* The last failure on this handle, or "". Valid until the next call on it. */
TORCH_ENGINE_API const char* torch_engine_error(const torch_engine* engine);

/* The last request's numbers. */
TORCH_ENGINE_API torch_engine_status torch_engine_last_stats(const torch_engine* engine, torch_engine_stats* stats);

TORCH_ENGINE_API void torch_engine_close(torch_engine* engine);

#ifdef __cplusplus
}
#endif

#endif /* TORCH_ENGINE_H */
