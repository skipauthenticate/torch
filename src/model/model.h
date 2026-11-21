// The model on the GPU: weights, the state of one sequence, and the one-token forward pass (milestone M2).
//
// decode() runs a token as one replayed CUDA graph (or kernel by kernel) and waits once, at the end. After each layer's
// router a gate kernel places the routed experts on the GPU; an expert that is not resident is read by the expert
// store's I/O thread while the gate waits (architecture decision 6), so the host never synchronizes inside a token.
// The layer math is docs/architecture.md's "The model"; tests/model/parity.py checks the result against a second
// implementation of the model.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/budget.h"
#include "core/placement.h"
#include "model/config.h"

namespace torch::kernels {
struct SamplingParams;  // kernels/sampling/sample.h
}

struct CUstream_st;  // cudaStream_t is CUstream_st*

namespace torch::model {

class Weights;

// The widest verify window: tokens run through the model in one pass (architecture decision 2).
inline constexpr int kMaxColumns = 8;
// The widest prefill chunk (Options::prefill_columns).
inline constexpr int kMaxPrefillColumns = 4096;

struct Options {
    std::filesystem::path model;  // shard 1 of the model; the others are found next to it (gguf::File::open_model)
    // Tokens the KV cache holds, up to 262,144, allocated at start; above indexer_top_k attention is sparse.
    int context = 2048;
    std::optional<int> resident;  // experts 0 to resident - 1 of every layer resident, exactly, with no top-up (tests)
    // With `placement`: the placement and the top-up after loading stop at this many experts in total, in the same
    // order, so two runs compare at equal bytes whatever memory the board has free (measurements).
    std::optional<int> resident_total;
    // Route counts (core/placement.h) ranking which experts to keep resident; without them, and without `resident`,
    // as many of every layer's first experts as the budget allows.
    std::filesystem::path placement;
    // Memory the caller allocates once the model is built (the draft layer, draft.h): budget lines, and room the
    // top-up after loading leaves free.
    std::vector<Reservation> reserve;
    // After each window, up to this many of the experts it read from the SSD take the slots of the least recently
    // routed resident experts of their layers (core/swaps.h); 0: residency stays as placed. Output does not change.
    int swaps_per_window = 0;
    // When a window's misses fill the staging, the ones held so far are swapped in then, within the window's swap
    // budget, instead of being dropped: a wide window's early layers can still be kept (M8: code review -3% to -6%
    // decode time with 128 swaps a window). The CPU copies those swaps, so it is off with lend_from (tenants are
    // device memory), and TORCH_SWAP_WHEN_FULL=0 turns it off. Output does not change.
    bool swap_when_full = true;
    // Between a prompt's end and the next prefill, the prefill chunk's device buffer (1.47 GiB at 4,096 columns) holds
    // the experts the prompt routed most that are not resident (then the placement's ranking), read when the prompt is
    // in (its time to first token, ~0.4 s); prefill() takes the buffer back first. M8: code review -4.1% decode time,
    // prose -1.6%, repetition -1.4% in one process. TORCH_LEND_CHUNK=0 turns it off. Output does not change.
    bool lend_chunk = true;
    // Prefill's BF16 products (the hyper-connection mixers, the router and the other BF16 projections) on the tensor
    // cores with the activations rounded to BF16 (kernels::gemm_bf16_tc): time to the first token -22% on a 4,096-token
    // code review prompt, but the prompt is then not computed bitwise as the default path computes it, so a greedy
    // answer can differ from the default's. Deterministic and independent of the chunk size either way. Off by default;
    // TORCH_FAST_PREFILL=1 turns it on.
    bool fast_prefill = false;
    // Positions from which the KV cache's unused tail holds more experts (0: none). Before a window writes K/V rows
    // there, the experts in the next lend_from positions' bytes stop being resident. Output does not change.
    int lend_from = 0;
    bool logits = true;  // copy every token's 248,320 logits to the host (per-step output, parity tests)
    // The widest prefill chunk (prefill()), 1 to kMaxPrefillColumns; 0: no prefill, prompts go through windows.
    int prefill_columns = 0;
    // Experts each of the two read-ahead pools holds (chunks of 256 tokens and more read every layer's non-resident
    // experts ahead, in passes of this many); 0: each layer waits for its reads.
    int prefill_staging = 64;
};

struct ExpertStats {
    int resident;     // experts resident, over all layers
    int experts;      // experts in the model, over all layers
    uint64_t hits;    // routed experts found resident
    uint64_t misses;  // routed experts read from the SSD
    uint64_t hedged;  // miss reads issued a second time because the first was late (expert_store.h)
    int tenants;      // of the resident ones, those in the KV cache's tail now (Options::lend_from)
    // experts read from the SSD for prefill chunks' grouped experts: each non-resident one once per chunk and layer
    // (all of a layer's when read ahead); their routes count in hits and misses as a window's do
    uint64_t prefill_reads;
};

// Where decode time went, summed over every decode() call (host wall-clock, milliseconds).
struct DecodeTimes {
    double total = 0;
    double ple = 0;                // reading and dequantizing the PLE rows
    double ple_max = 0;            // the slowest token's PLE rows
    uint64_t ple_slow = 0;         // tokens whose PLE rows took over 5 ms
    uint64_t ple_hedged = 0;       // PLE row reads issued a second time (hedged_reads.h)
    double gpu = 0;                // waiting for the GPU after the last launch: the work not yet done, including misses
    double expert_reads = 0;       // the I/O thread reading non-resident experts (overlaps `gpu`)
    uint64_t swaps = 0;            // experts made resident after windows (Options::swaps_per_window)
    uint64_t swaps_when_full = 0;  // of those, swapped in when the staging filled (Options::swap_when_full)
    double swap_ms = 0;            // choosing them and queueing their copies, between windows
};

class Model {
public:
    // Plans the memory budget, then loads the weights. Throws Error when a file is missing or not the supported
    // model, or when memory is short.
    explicit Model(const Options& options);
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    const Config& config() const noexcept;
    const Budget& budget() const noexcept;
    ExpertStats expert_stats() const;  // waits for the GPU
    DecodeTimes decode_times() const noexcept;
    RouteCounts route_counts() const;  // how often each (layer, expert) was routed so far (waits for the GPU)
    int position() const noexcept;     // tokens in the sequence so far

    // Appends `token` to the sequence and runs it through the model. The decode loop does not throw
    // (CONTRIBUTING.md): when the token is out of range, the context is full, or the GPU or a read fails, it returns
    // false and error() says why. After a GPU or read failure the sequence state is undefined; reset() before
    // decoding again. A token runs as one replayed CUDA graph, recorded at the first token; TORCH_EAGER=1 launches
    // kernel by kernel instead (the result is the same, bit for bit).
    bool decode(int32_t token) noexcept;
    // Runs 1 to kMaxColumns tokens at the next positions as one window (milestone M4): afterwards argmax(t) and
    // logits(t) are the results after tokens[0..t], equal bit for bit to decoding the tokens one at a time. All of
    // them join the sequence until accept() keeps fewer. decode(token) is a window of one. Errors as decode().
    bool verify(std::span<const int32_t> tokens) noexcept;
    // Keeps the first n tokens of the last window (1 <= n <= its width), as if only they had been decoded. The
    // window's results are not valid afterwards: read them first.
    bool accept(int n) noexcept;
    // Runs 1 to Options::prefill_columns prompt tokens at the next positions as one prefill chunk (milestone M6) and
    // keeps them all; every token's state is bitwise what feeding them in windows gives. Afterwards argmax(0) and
    // logits(0) are the last token's. Errors as decode().
    // A nonempty predictions span checks each position. A mismatch rewinds before the chunk.
    // A complete match keeps the chunk. Both outcomes preserve the preceding prompt checkpoint.
    bool prefill(std::span<const int32_t> tokens, std::span<int32_t> predictions = {}) noexcept;
    int copy_window_limit() const noexcept;  // largest proposal that fits the existing prompt workspace
    // Copies the 4-stream residual of columns first .. first + n - 1 (n <= kMaxColumns) of the last prefill chunk to
    // residual(0 .. n - 1), for the draft layer's passes over the chunk. Throws Error when out of range.
    void stage_residuals(int first, int n);
    // Column t's next-token logits after the last window (n_vocab floats), or empty when Options::logits is false.
    std::span<const float> logits(int column = 0) const noexcept;
    // Column t's most likely next token after the last window, computed on the GPU; ties go to the lowest id.
    int32_t argmax(int column = 0) const noexcept;
    // Sampling (kernels/sampling/sample.h): draws the next tokens of columns first .. first + n - 1 of the last window
    // (a prefill chunk: column 0, its last token's), the later columns conditional on the window's inputs after column
    // `first` as drafts, and returns the accepted tokens: the drafts that equal the drawn tokens, then the first drawn
    // token that differs, or the last column's. Column first draws at the position after its input token, so a request
    // draws the same tokens however its windows are cut. Penalties read history_counts(). Call before accept(); throws
    // Error when the GPU fails.
    std::vector<int32_t> sample(const kernels::SamplingParams& params, int first, int n);
    // The history counts the sampler's penalties read, n_vocab of them, in memory the GPU reads directly: the caller
    // keeps them, between windows.
    std::span<int32_t> history_counts() noexcept;
    const std::string& error() const noexcept;

    // Changes Options::swaps_per_window from the next window on (for example fewer during a prompt's one-token
    // windows).
    void set_swaps_per_window(int n) noexcept;
    // Options::lend_chunk: lends the prefill chunk's idle device buffer to experts; call when the prompt is in (the
    // reads count in its time). prefill() takes the buffer back before it writes there. Swaps never pick these experts,
    // so nothing else writes into the buffer. Returns how many; 0 when off, without prefill, or already lent.
    // With early, after the prompt's last prefill chunk and from another thread, while the draft layer's KV-only
    // passes over the chunk run (nothing else may use the model meanwhile): only the buffer after the chunk's
    // residuals, which those passes read; the call when the prompt is in then lends the residuals' part.
    int lend_chunk(bool early = false);

    // For the draft layer (draft.h), which runs on the model's stream and reads these, never writes them: the
    // weights (the token embedding and the output head), the stream every window runs on, and column t's 4-stream
    // residual after the last window on the GPU (hc x n_embd floats: the hidden state after the head mixer's
    // combine), valid until the next window.
    const Weights& weights() const noexcept;
    CUstream_st* stream() const noexcept;
    const float* residual(int column = 0) const noexcept;

    // Starts a new, empty sequence; with Options::lend_from, the KV cache's tail holds experts again from there.
    void reset();
    // Sessions (milestone M6, option B): the position after the last prefill chunk, whose sequence state is kept, or
    // -1 (none since reset(), or the last prefill failed).
    int prefill_end() const noexcept;
    // Goes back to prefill_end(): the tokens after it leave the sequence, so a request that extends the sequence
    // prefills the generated tokens again with its new ones, as a fresh prefill of the whole prompt would. Throws
    // Error when there is no prefill to go back to.
    void rewind();
    // Checkpoint data on the model stream. Only complete index blocks have a pooled key.
    int checkpoint_limit() const noexcept;
    std::span<uint8_t> checkpoint_state() const noexcept;
    std::span<uint8_t> checkpoint_residual() const noexcept;
    void checkpoint_rows(int first, int end, std::vector<std::span<uint8_t>>& out) const;
    void restore_checkpoint(std::span<const int32_t> ids);
    // From any thread: the next verify() or decode() returns false without running a window, error() says the request
    // was cancelled and cancelled() is true. The sequence is unchanged, so a later request can extend it.
    void cancel() noexcept;
    bool cancelled() const noexcept;  // the last verify() or decode() was cancelled

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch::model
