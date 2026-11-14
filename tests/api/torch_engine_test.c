/* The C API end to end on the real model (the Orin only; not in ctest): written in C against include/torch_engine.h,
 * linked with libtorch_engine.so.
 *
 *   torch_engine_test MODEL DRAFT VOCAB PROMPT.tokens
 *     sessions: turn 2 rewinds to turn 1's stable length (cached_tokens), and its ids equal the same turn 2 on a fresh
 *     handle, greedy and seeded; cancel from another thread mid-prefill and mid-decode, then the same request equals an
 *     uncancelled run. PROMPT is a chat prompt ending in the 7-token generation prompt (bench/workloads/).
 *   torch_engine_test MODEL DRAFT VOCAB PROMPT.tokens --bench SPEC PLACEMENT
 *     one request of 256 greedy tokens at the race's settings (context 262,144, prefill 4,096, swaps 32): decode tok/s
 *     and resident experts, for A/B runs against torch run. */
#include "torch_engine.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_IDS 300000
#define GEN 32

static int failed = 0;
static void check(int ok, const char* what) {
    printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok)
        failed = 1;
}

static int32_t* read_ids(const char* path, int32_t* n) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    int32_t* ids = malloc(sizeof(int32_t) * MAX_IDS);
    long v;
    *n = 0;
    while (*n < MAX_IDS && fscanf(f, "%ld", &v) == 1) {
        ids[(*n)++] = (int32_t)v;
        if (fgetc(f) == EOF)
            break;
    }
    fclose(f);
    return ids;
}

/* A request's output and the cancel trigger: another thread cancels once `fed` passes cancel_fed or `count` passes
 * cancel_count. */
typedef struct {
    int32_t tokens[4096];
    int32_t count;
    int32_t fed;
    int32_t cancel_fed, cancel_count;
    int triggered;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    torch_engine* engine;
    int done;
} Run;

static int on_tokens(void* user, const int32_t* t, int32_t n) {
    Run* r = user;
    pthread_mutex_lock(&r->lock);
    for (int32_t i = 0; i < n && r->count < 4096; ++i)
        r->tokens[r->count++] = t[i];
    if (r->cancel_count > 0 && r->count >= r->cancel_count && !r->triggered)
        r->triggered = 1, pthread_cond_signal(&r->wake);
    pthread_mutex_unlock(&r->lock);
    return 0;
}

static int on_progress(void* user, int32_t fed, int32_t prompt) {
    Run* r = user;
    (void)prompt;
    pthread_mutex_lock(&r->lock);
    r->fed = fed;
    if (r->cancel_fed > 0 && fed >= r->cancel_fed && !r->triggered)
        r->triggered = 1, pthread_cond_signal(&r->wake);
    pthread_mutex_unlock(&r->lock);
    return 0;
}

static void* canceller(void* user) {
    Run* r = user;
    pthread_mutex_lock(&r->lock);
    while (!r->triggered && !r->done)
        pthread_cond_wait(&r->wake, &r->lock);
    pthread_mutex_unlock(&r->lock);
    if (r->triggered)
        torch_engine_cancel(r->engine);
    return NULL;
}

static const int32_t kStop[2] = {248044, 248046};

static torch_engine_params params(int sample, uint64_t seed, int speculative) {
    torch_engine_params p;
    memset(&p, 0, sizeof p);
    p.struct_size = sizeof p;
    p.max_tokens = GEN;
    p.speculative = speculative;
    p.sample = sample;
    p.temperature = 1.0f, p.top_k = 20, p.top_p = 0.95f, p.repetition_penalty = 1.0f;
    p.has_seed = 1, p.seed = seed;
    p.stop_ids = kStop, p.n_stop_ids = 2;
    return p;
}

/* Runs one request; with cancel_fed or cancel_count > 0 a second thread cancels it there. */
static torch_engine_status run(torch_engine* e, const int32_t* ids, int32_t n, int32_t stable,
                               const torch_engine_params* p, Run* r, torch_engine_stats* s, int32_t cancel_fed,
                               int32_t cancel_count) {
    memset(r, 0, sizeof *r);
    pthread_mutex_init(&r->lock, NULL);
    pthread_cond_init(&r->wake, NULL);
    r->engine = e, r->cancel_fed = cancel_fed, r->cancel_count = cancel_count;
    pthread_t thread;
    pthread_create(&thread, NULL, canceller, r);
    memset(s, 0, sizeof *s);
    s->struct_size = sizeof *s;
    const torch_engine_status status = torch_engine_generate(e, ids, n, stable, p, on_tokens, on_progress, r, s);
    pthread_mutex_lock(&r->lock);
    r->done = 1;
    pthread_cond_signal(&r->wake);
    pthread_mutex_unlock(&r->lock);
    pthread_join(thread, NULL);
    if (status != TORCH_ENGINE_OK && status != TORCH_ENGINE_CANCELLED)
        printf("generate: status %d: %s\n", (int)status, torch_engine_error(e));
    return status;
}

static int same(const Run* a, const Run* b) {
    return a->count == b->count && memcmp(a->tokens, b->tokens, sizeof(int32_t) * (size_t)a->count) == 0;
}

static torch_engine* open_engine(const char** argv, int32_t context, int32_t prefill, const char* placement,
                                 int32_t swaps) {
    torch_engine_options o;
    memset(&o, 0, sizeof o);
    o.struct_size = sizeof o;
    o.model = argv[1], o.draft = argv[2], o.draft_vocab = argv[3], o.draft_head_rows = 40960;
    o.placement = placement, o.context = context, o.prefill_columns = prefill, o.swaps_per_window = swaps;
    char error[512];
    torch_engine* e = NULL;
    if (torch_engine_open(&o, &e, error, sizeof error) != TORCH_ENGINE_OK) {
        fprintf(stderr, "open: %s\n", error);
        exit(1);
    }
    return e;
}

/* Turn 2's prompt: turn 1 up to its stable length, the answer as history, a second user message, the generation
 * prompt. */
static int32_t turn2(const int32_t* p1, int32_t stable1, const Run* answer, int32_t* out) {
    static const int32_t user2[] = {248045, 846, 198, 6820, 1910, 424, 1495, 303, 799, 2716, 11316, 13, 248046, 198};
    static const int32_t gen[] = {248045, 74455, 198, 248068, 271, 248069, 271};
    int32_t n = 0;
    memcpy(out, p1, sizeof(int32_t) * (size_t)stable1), n = stable1;
    out[n++] = 248045, out[n++] = 74455, out[n++] = 198;
    for (int32_t i = 0; i < answer->count; ++i)
        if (answer->tokens[i] != kStop[0] && answer->tokens[i] != kStop[1])
            out[n++] = answer->tokens[i];
    out[n++] = 248046, out[n++] = 198;
    memcpy(out + n, user2, sizeof user2), n += (int32_t)(sizeof user2 / sizeof user2[0]);
    memcpy(out + n, gen, sizeof gen), n += (int32_t)(sizeof gen / sizeof gen[0]);
    return n;
}

int main(int argc, const char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: torch_engine_test MODEL DRAFT VOCAB PROMPT.tokens [--bench SPEC PLACEMENT]\n");
        return 2;
    }
    if (torch_engine_abi_version() != TORCH_ENGINE_ABI_VERSION) {
        fprintf(stderr, "library ABI %u, header %d\n", torch_engine_abi_version(), TORCH_ENGINE_ABI_VERSION);
        return 1;
    }
    int32_t n1 = 0;
    int32_t* p1 = read_ids(argv[4], &n1);
    static Run a, b;
    torch_engine_stats s;

    if (argc >= 8 && strcmp(argv[5], "--bench") == 0) {
        torch_engine* e = open_engine(argv, 262144, 4096, argv[7], 32);
        torch_engine_params p = params(0, 0, atoi(argv[6]));
        p.max_tokens = 256, p.n_stop_ids = 0;
        if (run(e, p1, n1, n1 - 7, &p, &a, &s, 0, 0) != TORCH_ENGINE_OK)
            return 1;
        printf("bench: %d tokens, %.2f tok/s, %.2f tokens per window, prompt %.1f s, %d of %d resident\n",
               s.generated_tokens, s.generated_tokens / s.decode_seconds, s.tokens_per_window, s.prompt_seconds,
               s.resident_experts, s.total_experts);
        printf("generated tokens:");
        for (int32_t i = 0; i < a.count; ++i)
            printf("%s%d", i ? "," : " ", a.tokens[i]);
        printf("\n");
        torch_engine_close(e);
        return 0;
    }

    const int32_t stable1 = n1 - 7;
    int32_t* p2 = malloc(sizeof(int32_t) * MAX_IDS);
    int32_t* q2 = malloc(sizeof(int32_t) * MAX_IDS);
    static Run g1, g2, s1, s2;

    /* Handle A: two greedy turns, then two sampled turns (speculative). */
    torch_engine* e = open_engine(argv, 16384, 512, NULL, 0);
    torch_engine_params greedy = params(0, 0, 1), sampled1 = params(1, 5, 1), sampled2 = params(1, 6, 1);
    check(run(e, p1, n1, stable1, &greedy, &g1, &s, 0, 0) == TORCH_ENGINE_OK && g1.count > 0, "turn 1, greedy");
    const int32_t n2 = turn2(p1, stable1, &g1, p2);
    check(run(e, p2, n2, n2 - 7, &greedy, &g2, &s, 0, 0) == TORCH_ENGINE_OK, "turn 2, greedy");
    printf("  turn 2: %d prompt tokens, %d cached (turn 1's stable %d), %d generated\n", s.prompt_tokens,
           s.cached_tokens, stable1, s.generated_tokens);
    check(s.cached_tokens == stable1, "turn 2 rewinds to turn 1's stable length (cached_tokens)");
    check(torch_engine_reset(e) == TORCH_ENGINE_OK, "reset");
    check(run(e, p1, n1, stable1, &sampled1, &s1, &s, 0, 0) == TORCH_ENGINE_OK, "turn 1, seeded");
    const int32_t m2 = turn2(p1, stable1, &s1, q2);
    check(run(e, q2, m2, m2 - 7, &sampled2, &s2, &s, 0, 0) == TORCH_ENGINE_OK && s.cached_tokens == stable1,
          "turn 2, seeded, cached_tokens = turn 1's stable");
    torch_engine_close(e);

    /* Handle B, fresh: the same turn 2 requests, then cancels. */
    e = open_engine(argv, 16384, 512, NULL, 0);
    check(run(e, p2, n2, n2 - 7, &greedy, &a, &s, 0, 0) == TORCH_ENGINE_OK && s.cached_tokens == 0 && same(&a, &g2),
          "turn 2 greedy on a fresh handle = through the session");
    torch_engine_reset(e);
    check(run(e, q2, m2, m2 - 7, &sampled2, &a, &s, 0, 0) == TORCH_ENGINE_OK && same(&a, &s2),
          "turn 2 seeded on a fresh handle = through the session");
    torch_engine_reset(e);
    check(run(e, p1, n1, stable1, &greedy, &a, &s, 1, 0) == TORCH_ENGINE_CANCELLED && a.count == 0 && a.fed < stable1,
          "cancel from another thread mid-prefill");
    printf("  cancelled at position %d of %d\n", a.fed, n1);
    check(run(e, p1, n1, stable1, &greedy, &b, &s, 0, 0) == TORCH_ENGINE_OK && same(&b, &g1),
          "then the same request = an uncancelled run");
    printf("  it kept %d tokens\n", s.cached_tokens);
    check(run(e, p1, n1, stable1, &greedy, &a, &s, 0, 6) == TORCH_ENGINE_CANCELLED && a.count < g1.count,
          "cancel from another thread mid-decode (greedy)");
    check(run(e, p1, n1, stable1, &greedy, &b, &s, 0, 0) == TORCH_ENGINE_OK && same(&b, &g1),
          "then the same request = an uncancelled run");
    check(run(e, p1, n1, stable1, &sampled1, &a, &s, 0, 6) == TORCH_ENGINE_CANCELLED && a.count < s1.count,
          "cancel from another thread mid-decode (seeded)");
    check(run(e, p1, n1, stable1, &sampled1, &b, &s, 0, 0) == TORCH_ENGINE_OK && same(&b, &s1),
          "then the same request = an uncancelled run");
    torch_engine_close(e);
    free(p1), free(p2), free(q2);
    printf("%s\n", failed ? "FAILED" : "all ok");
    return failed;
}
