# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-11-20

### Added

- Saved prompts, opt-in with `--prefix-memory-gib N` ([Sessions](docs/server.md#sessions)). Conversations that share a
  system prompt and tools continue from a saved prefix, and a conversation resumes its last turn after a client's
  title request. On the Jetson AGX Orin 64 GB on 2026-11-19 (IQ3_S), a new conversation sharing 2,073 tokens showed
  its first output in 2.28 s instead of 8.73 s, and one sharing 10,000 tokens in 2.07 s instead of 38.35 s. A 2 GiB
  reservation keeps about 1,040 fewer experts in memory, and generation on fresh prompts was 0.44-0.86 tokens/s
  slower. Saving to disk needs an explicit directory on a local file system; the engine refuses one that cannot hold
  its files.
- Version 3 of the C API: checkpoint positions, the storage tier a request restored from, and a call that clears saved
  prompts. Callers built for versions 1 and 2 keep working, with their previous meaning of `speculative`.

### Changed

- Speculative greedy requests also check up to 7 tokens copied from earlier input. Only tokens the model itself
  picks are kept, so the output is unchanged. Repetitive text on 2026-11-19: 50.90 -> 70.17 tokens/s on IQ3_S and
  47.56 -> 69.54 on UD-IQ3_XXS.
- Faster exact prompt processing: wide 8-bit rows stream through the prompt products, and short verification windows
  reuse the mixer weights. A 4,096-token code-review prompt: 288 -> 304 tokens/s on IQ3_S and 288 -> 350 on
  UD-IQ3_XXS.

### Known regressions

- Both files keep fewer experts in memory than 0.4.0, and some pauses between output groups are longer (UD-IQ3_XXS
  repetition: the median longest pause went from 126 to 163 ms).

### Verified

- On both files, the 36 benchmark answers (9,216 generated tokens) are identical to 0.4.0, fixed-seed sampling gives
  the same tokens, and model parity passes (mean KL 0.00143 on IQ3_S, 0.00249 on UD-IQ3_XXS). Saved prompts restored
  from memory, from the KV cache and from disk give byte-identical output to a fresh run at 300 to 70,000 tokens.

## [0.4.0] - 2026-11-16

### Added

- The UD-IQ3_XXS model file (docs/model-files.md): four shards, read as one; its calibration folder; its parity
  reference; `scripts/install.sh --quant UD-IQ3_XXS`.
- `HF_ENDPOINT`, when set, replaces `https://huggingface.co` for the installer's shard downloads (a mirror).
- The installer's smoke test prints the model file it ran, and `smoke.log` starts with it.

### Changed

- Model parity has new references and thresholds (docs/development.md, "Model parity"). Each file's short-prompt
  reference puts the experts of 2 layers on the reference's CPU instead of 4, and the threshold is twice the largest of
  three spreads of the reference against itself: 2 vs 4 CPU layers, prompt batches of 512 vs 8, flash attention on vs
  off. IQ3_S: 0.0041 -> 0.00328 (Torch 0.00143); UD-IQ3_XXS: 0.00312 -> 0.00596 (Torch 0.00249). Long prompts get
  their own thresholds: 0.00292 and 0.00542 (Torch 0.00193 and 0.00345). The rule was set after Torch's numbers
  against the old references were known. The reason: the reference's CPU experts round their inputs per 256 values,
  its GPU kernels and Torch per 32, and the old spread mostly measured that difference. `--fast-prefill` passes on the
  IQ3_S file (0.00179) and fails the top-1 rule at one token on the UD-IQ3_XXS file.

### Fixed

- The installer remembers the model file it installed last (`PREFIX/share/torch/quant`) and uses it without
  `--quant`. Before, a plain rerun after `--quant UD-IQ3_XXS` started an 83.6 GB IQ3_S download. The models step
  also writes the server launcher, so the launcher names that file.
- The installer checks the disk once for all missing shards, less what their `.part` files hold, before the first
  download. Before, each shard was checked alone, so a disk with room for the first shards ran out partway.
- A `.part` that is already complete (a run stopped while verifying it) is verified without a new request.
- When a later shard of a split model fails to open, the shard files already opened are closed.
- A model whose F32 matrix holds values BF16 cannot store now gets an error that says to use a supported model file
  (docs/model-files.md).

## [0.3.1] - 2026-11-15

### Fixed

- When its memory estimates at start were exact (little page cache to free while loading), the engine placed
  experts down to its 6 GiB free-memory floor and could end just below it, skipping the 1.5 GiB margin the top-up
  keeps. The placement at start now keeps the margin too.

## [0.3.0] - 2026-11-15

### Added

- Tool calls in `torch_server` (docs/server.md, Tool calls): OpenAI `tools`, `tool_choice` `auto` and `none`,
  `parallel_tool_calls`, and assistant `tool_calls` and `tool` messages in the history. The model's calls come back as
  `tool_calls`, streamed whole (one chunk per call), with arguments typed by each parameter's schema. The server runs
  no tool, and text that is not a call comes back as text. A tool result that holds the text of a control token is
  encoded as plain text. `/v1/models` reports `context_length`. Measured on the Jetson AGX Orin 64 GB on 2026-11-14:
  15 tool requests with thinking off and on gave 15 parsed calls, all of their schema's types, none as text; the
  follow-up turn kept turn 1's prompt (`cached_tokens` 484 of 608); requests without tools give 0.2.0's ids (9 of 9
  benchmark bodies). `tool_choice` `required` or a named function is refused: it would need constrained decoding.
- `reasoning: {"enabled", "effort"}` in a request, as some agent clients send it: another spelling of
  `reasoning_effort` (`enabled: false` turns thinking off); `reasoning_effort` wins when both are given.

### Changed

- A prompt chunk reads its PLE rows on another thread while the GPU runs layer 0, instead of before its first kernel
  is queued. The rows and the output are the same. Through one `torch_server` process on the Jetson AGX Orin 64 GB on
  2026-11-14, with the nine benchmark requests in order, the mean time to the first token went from 14.47 to 14.29 s
  for code review, 15.47 to 15.19 s for prose and 15.43 to 15.11 s for repetition. The ids were unchanged and the decode
  speed was the same.
- A request without `max_tokens` may generate what the context has left, instead of at most 4,096 tokens, as agent
  clients that send none expect (thinking counts against it). `torch_server --max-tokens-default N` still caps it.

## [0.2.0] - 2026-11-14

### Changed

- Faster code review (M8). Measured on the Jetson AGX Orin 64 GB on 2026-11-14 with nine 4,096-token chat requests
  (three code review, three prose, three repetitive text, 256 greedy tokens each) sent in that order to one
  `torch_server` process, F16 KV for 262,144 tokens allocated up front: the code review median went from 32.6 to
  35.0 tok/s with the three new defaults below (31.9 for v0.1.0's own build in the same session), and to 36.0-37.3
  tok/s with the prefill chunk's buffer lent as well. The generated ids are unchanged on all nine requests.
  docs/performance.md, M8, has every measurement.
- New defaults, measured together in one process with 15,500 experts pinned (one ABBA pair): code review decode time
  -7.9%, prose -1.5%, repetition -9.6%.
  - When a window's misses fill the 80-slot staging, the misses held so far are swapped in at that point instead of
    being dropped (CPU copies inside the window; off with KV lending; `torch run --no-swap-when-full` or
    `TORCH_SWAP_WHEN_FULL=0`).
  - 128 swaps a window instead of 32 (`torch run --swap`, `torch_server --swaps`); none during the prompt.
  - The draft head has 81,920 rows instead of 40,960 (`calibration/ed59f920/draft-vocab.txt`).
- Between a prompt's end and the next prefill, the prefill chunk's idle device buffer (1.47 GiB at 4,096 columns)
  holds ~760 more experts: the ones the prompt routed most that are not resident. One server process, two ABBA pairs:
  code review decode time -4.1%, prose -1.6%, repetition -1.4%, time to the first token +0.43 s. `torch run
  --no-lend-chunk` or `TORCH_LEND_CHUNK=0` turns it off.
- Time to the first token through the server: code review 19.7 -> 17.1 s, prose and repetition 19.6 -> 18.7 s. Each
  distinct PLE row is read once per prompt, the draft's K/V-only pass over a prompt chunk runs 8 positions as columns,
  three prefill GEMM changes, and the chunk buffer's reads overlap the draft's last pass.
- Time to the first token on the default (exact) path, through the server in the benchmark's order on 2026-11-14:
  code review 17.3 -> 14.5 s, prose and repetition 18.8 -> 15.5 s, with the same answers and decode speed. In prefill
  the shared expert's gate and up run on the tensor cores, the routed experts' SwiGLU, Q8 and activation re-layouts read
  and write whole rows, the K = 640 down projections run each output's tree in one warp, the resident experts run while
  the SSD reads the rest of the layer's experts, and a product's few rows past its last 64-row tile no longer take a
  tile. Each output is bitwise what it was. `--fast-prefill` gains as much: code review 12.9 -> 10.6 s.
- Window kernels: IQ3_XXS gate and up with the grid's signs applied, short BF16 matrices reading their activations
  directly, and 192-unit dense rows on a whole warp at 3-4 columns: code review's GPU compute -0.5 ms a window at 2
  columns and -1.15 ms at 3.
- Mixers and dense products in verify windows: a mixer's up projection and collapse run as one launch, and the dense
  products at 3-6 columns stage their columns once per resident block. Code review's GPU time per window -0.4 ms at
  width 2 and -1.4 ms at width 4, its most common width; the generated ids are unchanged.
- Experts in lent memory (`--lend-kv` and the chunk buffer) are read through the hedged reads, so a read that stalls in
  the SSD is issued again after 50 ms instead of holding the lend for the driver's 30 s timeout.

### Added

- `--fast-prefill` (torch run and torch_server; `TORCH_FAST_PREFILL=1`; the C API's `fast_prefill`, ABI 2, with ABI 1
  callers' shorter struct still accepted): the prompt's BF16 products on the tensor cores with the activations rounded
  to BF16. Time to the first token on the nine 4,096-token benchmark prompts 16.9 -> 13.0 s (code review) and 18.4 ->
  14.4 s (prose, repetition); not bitwise the default path, so greedy answers can differ (7 of the 9 benchmark answers
  did), deterministic and independent of the chunk size, model parity mean KL 0.00158 (gate 0.0041). Off by default.
- `torch run --resident-total N`: the ranked placement and the top-up after loading stop at N experts in total, to
  compare runs at equal bytes. torch run's generation line also prints the misses' read time and the swaps made when
  the staging filled; `bench/run_workloads.py` records both.
- `TORCH_DECODE_STAGES=1` prints the GPU milliseconds per window by stage and window width.
- `torch run --carry-policy` (the window policy keeps what it learned across `--then` requests) and
  `--spec-policy adaptive-tight` (drafting stops once a draft would not be verified). Both measured slower on code
  review in one process (+8.4% and +11.6% decode time: a carried policy locks at one draft a window), so neither is a
  default.

## [0.1.0] - 2026-11-14

### Added

- `scripts/install.sh` (M7, docs/install.md): checks the Orin, builds into a prefix, makes the server's environment,
  downloads (resumable) or verifies the model shards against `scripts/models.sha256`, builds the draft layer locally
  with its BF16 input checked, and runs a smoke test; needs the model licenses accepted, never uses sudo.
- `calibration/ed59f920/`: the placement (route counts) and draft-head vocabulary for the pinned model files, made from
  the held-out calibration prompts; `torch_server` uses them by default, and the installer installs them.
- `torch_server` (M7, `server/torch_server`, docs/server.md): `POST /v1/chat/completions` streamed or not,
  `GET /v1/models`, `GET /health` over the C API; the model's own template in a sandbox; thinking as
  `reasoning_content`; stop strings across windows; sessions with `cached_tokens`; cancel when the client goes, also
  during the prompt; one engine thread and a queue. CI tests it against a stub of the C API on x86-64 and arm64.
- Milestone M7, first release:
  - The generation loop is a library class, `runtime::Generator` (`src/runtime/generator.h`): the prompt in prefill
    chunks or windows, one-token and speculative greedy decoding, sessions, cancel, and stops on max tokens, stop ids
    or a caller callback. `torch run` is a command line over it, with the same output; `--stop-ids` stops after any
    of the given ids.
  - Sampled decoding (`torch run --temperature ... --seed S`): the sampling kernels draw each token at its position
    with the request's seed, one token at a time or speculatively with the exact accept step (`accept_drafts`), so
    the same seed gives the same tokens either way. The 243 unused padding ids are never drawn; penalties count only
    the request's generated tokens. A request's last speculative window always gets the draft layer's K/V pass, and a
    request cancelled mid-prompt records only what was fed.
  - The C API (`include/torch_engine.h`, `libtorch_engine.so`): open, generate (token ids, a `stable` length, greedy
    or sampled parameters, token and progress callbacks), cancel from any thread, reset, error, stats, close. Status
    codes, no exceptions across the boundary, a struct size in every struct and `torch_engine_abi_version()`. Sessions
    rewind to the end of the previous prompt's stable part or start again (`cached_tokens`); a cancel stops before the
    next window or prefill chunk. The engine is built once, position-independent, for the command line and the
    library, and the library exports only `torch_engine_*`.
  - `torch inspect --export-tokenizer DIR` (M7): `tokenizer.json` and `chat_template.jinja` from shard 1's header,
    equal as parsed JSON to the base model's tokenizer; CI checks it on the real header (job `tokenizer-export`).
- Milestone M6, long context:
  - The model's sparse indexer above 2,048 tokens of context, to 262,144 (`src/kernels/indexer/`): pooled keys, scores
    in a fixed order checked bit for bit against a CPU reference, the top 512 blocks with ties to the lower index, and
    attention over each column's cell list. Up to 2,051 tokens the result is bitwise the dense kernel's.
  - `--context` up to 262,144, the KV cache, pooled keys and scratch allocated and zeroed at start; the budget prints
    the KV cache and indexer keys on their own line.
  - The draft layer attends to the last 16,384 positions from a K/V ring, and `--draft` prompts go in windows like
    any other.
  - `--lend-kv` (`--lend-from P` for tests): the KV cache's unused tail holds more experts until the sequence reaches
    it; output unchanged.
  - Sessions and cancel: `torch run --then` feeds a second request only its new tokens when it extends the sequence;
    `Model::cancel()` stops a request between windows (Ctrl-C, `--cancel-after` for tests); a reset lends the KV
    tail again.
  - Chunked prefill (`Model::prefill`, `torch run --prefill N`, chunks of 1 to 4,096 tokens), every column bitwise
    windows of 8. Dense quantized products on the int8 tensor cores (`kernels/gemm`); BF16 x F32 on the FP32 cores in
    the window order; routed experts grouped by expert, with each layer's non-resident experts read ahead into two
    pinned pools of 64. The chunk's DeltaNet recurrence runs in one launch, and its attention and indexer run in
    batches of 128 columns. `TORCH_PREFILL_STAGES=1` prints GPU time per stage. The race's 4,096-token review prompt
    takes 17.7 s (231 tok/s, 2026-11-13).
  - Sessions over prefill: the state after each chunk is kept (0.11 GiB); an extending request rewinds to it
    (`Model::rewind()`, `Model::prefill_end()`) and prefills the tokens generated since with its new ones, bitwise a
    fresh prefill. `--cancel-prompt-after N` for tests.
  - The run summary's `prompt:` line: the prompt's routed hits, misses and experts read for prefill chunks.
  - `bench/needle.py` (needle in a haystack of synthetic prose), `bench/sparse_attention.cu` (attention cost per
    window, dense against sparse), `tests/model/parity.py` for long prompts (`@FILE` ids, `--window`, `--placement`,
    `--gaps` for the selection's near-ties), `TORCH_DUMP_FROM`.
- Milestone M4, verify windows and speculative decoding:
  - `Model::verify(tokens)` runs 1 to 8 tokens as one window, one CUDA graph per width, and `accept(n)` keeps the
    first n. Every kernel takes the window's columns, and each column is bitwise the one-column result.
  - Window sequence state: attention writes the window's K/V rows and then attends per column in the one-token
    order. DeltaNet walks the window in registers from one state buffer into the other and replays the kept columns
    with the same kernel when a window is cut. The convolution history and the PLE ring are committed after the
    accept decision.
  - Routed experts across the window: one gate launch per layer numbers the window's unique experts. Gate and up run
    once per unique expert over its columns, down once per unique expert (Q2_0 per route), and misses are read once,
    in 128 KiB chunks in parallel.
  - The draft layer:
    - `tools/make_draft.py` builds it from the original checkpoint's MTP tensors with HTTP range reads, as an all-Q8_0
      file or (`--experts iq4`) with IQ4_XS/IQ4_NL routed experts, using this project's quantizers.
    - `model::Draft` runs it on the trunk's stream: a K/V-only pass at prompt and kept positions, then the full block
      and a 40,960-row head subset, as CUDA graphs.
  - `torch run --spec D` generates speculatively (greedy), with `--spec-policy` choosing the window width from
    learned acceptance and measured cost. `--check-draft-kv` checks the draft layer's K/V invariant, and
    `--draft-depth`/`--acceptance-report` measure acceptance per depth.
  - Exact speculative sampling kernels (`sample_columns`, `accept_drafts`) with a distribution test.
  - `--swap N` / `--swap-prompt N`: after each window, missed experts take the slots of the least recently routed
    resident ones. The GPU stamps recency and a GPU copy kernel moves the bytes.
  - Benchmarks: `bench/run_workloads.py --spec-args`, which also records the speculative summary and MemAvailable;
    `bench/draft_acceptance.py`; the kernel harness at 1 to 8 columns.
- Milestone M3, fast decode of one token at a time:
  - Each token runs as one CUDA graph. After each router a gate kernel writes the layer's expert pointer table on
    the GPU from a device map of resident experts; misses go through a mailbox in mapped memory to the expert store's
    I/O thread, so the host never synchronizes inside a token (`src/runtime/expert_gate.h`). Greedy argmax on the GPU.
  - Ranked expert placement from recorded route counts (`--route-counts`, `--placement`, `src/core/placement.h`),
    then topped up after loading from a fresh MemAvailable reading down to the floor plus 1.5 GiB.
  - Hedged reads for expert misses and PLE rows (`src/runtime/hedged_reads.h`): a read still pending after 50 ms is
    issued again, because the development SSD occasionally leaves one uncompleted for 30 s.
  - Kernels: BF16 short rows four per warp with the inject rows in the down projection's launch; one quantized GEMV
    for dense rows and experts alike; grouped launches for matrices that share an input, each input quantized once
    and written as Q8 by the kernel that produces it; split-K decode attention; the residual update fused into the
    next norm; the DeltaNet gates and q/k norm fused into their neighbors; the router ranked by counting.
  - Benchmarks: the three workloads as committed token ids and their driver (`bench/workloads/`,
    `bench/run_workloads.py`), the per-kernel harness (`bench/kernels.cu`) and its gates (`bench/kernel_gates.py`),
    and microbenchmarks for the miss round trip and graph launch costs.
  - `tools/check_orin.sh`: the checks a milestone passes on the Orin before it merges.
  - `torch run --tokens @FILE`; `--quiet` prints the generated ids; `--profile` reports the slowest PLE read.
- Milestone M2, a correct forward pass:
  - The model on the GPU, one token at a time: kernels for every layer operation (`src/kernels/`), each with a
    parity test against a scalar CPU reference; the forward pass (`src/model/model.cpp`).
  - The memory budget, planned from MemAvailable with a 6 GiB floor and printed at start (`src/core/budget.h`).
  - Weights and resident experts in one mapped pinned arena filled by `O_DIRECT` reads of at most 128 KiB; misses
    read synchronously into staging slots behind one pointer table, so output does not depend on residency
    (`src/runtime/`).
  - `torch run` (`--resident`, `--logits`, `--json`, `--profile`); `TORCH_DUMP` and `TORCH_DUMP_DIR` stage dumps.
  - Model parity against a committed reference (`tests/model/`), gated on greedy top-1 agreement and mean KL.
  - Microbenchmarks for GPU reads by memory type and SSD reads by queue depth (`bench/`).
- Milestone M1, foundations:
  - CMake build with the options `TORCH_CUDA`, `TORCH_TESTS`, `TORCH_WERROR`, `TORCH_SANITIZE` and `TORCH_FUZZ`;
    `torch --version` prints the version and git commit.
  - GGUF reader for the model's split files (`src/gguf/`): bounds-checked header parsing, typed metadata access,
    split information, tensor reads with `pread`, and header-only files such as the first 11 MiB of shard 1.
  - CPU reference dequantization of the 13 formats in the model's two shards (`src/formats/`), with the IQ formats'
    codebooks in `src/formats/codebooks.h` under their MIT license notice; [formats.md](docs/formats.md) writes out
    every block layout.
  - `torch inspect [--tensors] [--header-only] [--json] FILE...`.
  - Unit tests, a libFuzzer target for the reader, and synthetic fixtures generated by the project's own numpy
    reference and GGUF writer, with numpy pinned.
  - CI: formatting (clang-format 19.1.7), host tests (GCC 13; Clang 18 with AddressSanitizer and
    UndefinedBehaviorSanitizer; Apple Clang), fixture reproducibility and 60 seconds of fuzzing.
- Project documentation: architecture, performance model, benchmarking and development guides.
- Contribution guide, code of conduct, security policy, issue and pull request templates.
- `docs/model-files.md`: exact names, sizes, revision and verified sha256s of the two model shards; the definition
  of the draft-layer file.
- `NOTICE` (model licenses) and
  `.github/CODEOWNERS`.

### Fixed

- Hedged reads could hang a batch that takes longer than the hedge delay: reads queued behind the others counted as
  late, and once the spare buffers were taken a hedge waited for one while nothing freed the losers' buffers. A read
  is now late only after a worker has had it for the delay, and a hedge waits for a free spare without holding up the
  batch.

### Changed

- Expert placement from the routing of generated tokens: `--route-counts` records only the tokens generated after the
  prompt, and `run_workloads.py --calibrate` generates 256 tokens per held-out prompt. Code review +4.7% (median of
  three, one session) with the memory a 262,144-token context will take; generation misses fall 40-90%.
- Milestone M5, host time per window:
  - A verify window's PLE rows are read in one batch of parallel reads, and a single-chunk read wakes one read worker
    instead of all of them: PLE time per token 1.05 to 0.32 ms.
  - A window's commit launches as one CUDA graph per (state parity, kept columns, replay case): 0.30 to 0.02 ms of host
    time per window. `TORCH_HOST_STAGES=1` prints the host time per window by stage.
  - Code review 38.2, prose 41.3 and repetition 56.0 tok/s (2026-11-09); one token at a time back to within ~1% of M3.
- Architecture after the 2026-11-03 design review: a written memory budget with a KV cache that grows with the
  sequence; per-layer CUDA graph segments with a pointer table for resident and staged experts; 1-4 columns in
  registers plus an int8 tensor-core path for 5-16 with one canonical reduction order; double-buffered DeltaNet
  state with operand replay; F16 KV cache (int8 dropped from M6); prefill, session reuse and cancel; the draft-layer
  file; tokenizer, chat template, sampling and exact speculative sampling; a numerics and determinism policy.
- Corrected the model description from the file's header: the indexer selects 2,048 cells (512 pooled blocks), the
  format list includes Q8_0 and F16, the n-gram table is IQ4_NL, RoPE is stored as M-RoPE sections, the file has no
  MTP tensors. Vision is a non-goal.
- Milestone gates separated from speed goals; benchmark reports include the slowest window and a committed result file.
