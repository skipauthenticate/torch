# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
- Project documentation: architecture, performance model, roadmap, benchmarking and development guides.
- Contribution guide, code of conduct, security policy, issue and pull request templates.
- `docs/model-files.md`: exact names, sizes, revision and verified sha256s of the two model shards; the definition
  of the draft-layer file.
- `NOTICE` (model licenses) and
  `.github/CODEOWNERS`.

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
- Roadmap gates separated from speed goals; benchmark reports include the slowest window and a committed result file.
