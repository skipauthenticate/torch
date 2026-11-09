# Roadmap

The engine is built in milestones. Each one ends with something that runs, is tested, and is measured against
[performance.md](performance.md). A milestone is done when its **gates** pass on the Orin and its numbers, including
any speed goal it missed, are written in its pull request. Speed goals are listed in
[performance.md](performance.md#targets).

| Milestone | What works at the end | Gates |
| --- | --- | --- |
| **M0** Repository | Docs, contribution guide | Docs review |
| **M1** Foundations | GGUF reader for split files (header mapped, tensor bytes read with `pread`); CPU reference dequantization of the 13 formats in the two shards; `torch inspect`; CI for host-only code | Dequantization fixtures byte-exact against the project's numpy reference, regenerated in CI; the reader's fuzz target is clean; `torch inspect --header-only` on the real shards reproduces the type table in [model-files.md](model-files.md) |
| **M2** A correct forward pass | Eager single-token decode on the GPU; misses read synchronously into staging and computed by the same kernel; the memory budget printed at start; the SSD and the memory types measured | Next-token log-probabilities match the committed reference (`tests/model/reference.json`) on fixed prompts within the thresholds M2 fixes from the reference's own spread (fixed: greedy top-1 agreement at every position with a 0.38 near-tie margin, and a mean per-position KL divergence of at most 0.0041); identical logits whatever the residency (tested with 437, 256 and 0 experts resident per layer, since all 512 do not fit next to the operating system) |
| **M3** Fast autoregressive decode | Per-layer graph segments; fused hyper-connection mixers; the six expert kernels near bandwidth | Every kernel microbenchmark within 15% of streaming bandwidth; >= 27 tok/s on code review (goal 30) |
| **M4** Verify windows | Windows of 1-8 columns, each bitwise the one-column result; routed experts deduplicated across the window; DeltaNet state in two buffers, committed after accept and replayed by the same kernel; the draft-layer file, written and quantized by this project's tools, and the 4-bit MTP drafter; adaptive window; exact speculative sampling kernels; residency that follows use | Logits bitwise identical across window widths; greedy text identical with speculation on and off; sampled output passes a distribution test (goal 45 tok/s on code review) |
| **M5** Copy mode and residency | Prompt-lookup copies; 9-16 columns on int8 tensor cores; asynchronous misses with measured prefetch; PLE read-ahead; KV, indexer and draft-KV that grow with the sequence | Output independent of residency; logits bitwise identical across both column paths; predictor hit rates reported separately from residency (goals: prose 40, repetition 100) |
| **M6** Long context | Sparse indexer attention to 262,144 tokens with pooled keys; F16 KV cache; chunked prefill on int8 tensor cores, which verify windows also use where faster; sessions that extend the previous sequence; cancel | Needle retrieval at 32K, 128K and 256K; a second turn does not prefill the first (goal: 1,000 prompt tok/s at 4,096) |
| **M7** First release | `torch_server` (OpenAI-compatible: tokenizer, chat template, sampling and speculative sampling, streaming); `torch inspect --export-tokenizer`; install script; model-file verification; v0.1.0 | End-to-end install on a fresh Orin; every file in [model-files.md](model-files.md) verified by sha256 |

M3 merged on 2026-11-04 below two of its gates, by the maintainer's decision: 25.4 tok/s on code review against 27,
and 29 launch classes of 1 MiB or more below 85% of the streaming ceiling ([performance.md](performance.md#measured-so-far)
has the numbers). Both carry to M4 as goals, since its multi-column kernels rewrite the same paths.

M4 is done. Its gates, as measured on the Orin on 2026-11-05:

- **Logits bitwise identical across window widths**: windows of 1 to 8 columns and every rollback case equal one
  token at a time, byte for byte. Every kernel has a test that checks each column against the one-column launch.
- **Greedy text identical with speculation on and off**: the nine benchmark prompts, 256 generated tokens, with and
  without swaps.
- **Sampled output passes a distribution test**: `sample_test`, 30,000 seeds for each of four parameter sets,
  chi-square z-scores below 1.3. Sampling runs as GPU kernels with an exact accept step; sampled decoding end to end
  comes with the server in M7.
- **Speed**: 37.2 tok/s on code review (40.3 prose, 53.9 repetition) against the goal of 45 tok/s ([performance.md](performance.md#m4-decode)).

What carries over:

- Columns 9 to 16 on int8 tensor cores move to M5 with copy windows, and so does that gate.
- The M3 gates carried to M4 are still missed and carry to M5: one token at a time measures 24.15 tok/s on code
  review against 27, and the kernel harness has the same 29 one-column classes below 85% of the ceiling as M3
  (36.9 ms of kernels per token, 1,107 launches; launches under 1 MiB take 2.30 ms against 2).
- One-token decode is 3.3-3.6% slower than M3's in one session with alternating binaries (24.01 against 24.82 tok/s
  on code review): partly 170-280 fewer resident experts, and ~1 ms per token not yet attributed. M5 finds it.
- Prompt-informed warming from M5 is dropped: warming from the prompt's route counts ended with more misses than
  none, and M4's online swaps move residency after every window instead.

M5 is done, below its speed goals. Its gates, as measured on the Orin on 2026-11-09:

- **Output independent of residency**: 0, 256 and the budget's experts give byte-identical logits
  (`tools/check_orin.sh`), and the nine benchmark requests generate the same ids with and without speculation.
- **Logits bitwise identical across both column paths**: the int8 tensor-core prototype (Q6_K) equals the dp4a path
  byte for byte at 1 to 8 columns. The forward pass does not use it yet.
- **Predictor hit rates**: no predictor was built. Expert reads take 0.31 of 37.2 ms per token in the code review
  profile (99.92% hits), so asynchronous misses with prefetch were dropped.
- **Speed**: 38.2 tok/s on code review (goal 50), 41.3 on prose (goal 40) and 56.0 on repetition (goal 100)
  ([performance.md](performance.md#m5-decode)).

What changed from the plan:

- Copy proposals were built and measured: repetition fell from 53.7 to 39.6 tok/s, so they are not in the engine.
- Columns 9 to 16 and the int8 tensor cores move to M6. The tensor-core kernel is faster than dp4a from 3 columns,
  and M6's prefill needs the same kernels at many columns.
- KV, indexer and draft-layer KV that grow with the sequence are dropped. Device buffers are one allocation and
  experts sit in whole arenas, so growth would need virtual memory management for both. M6 allocates the context up
  front.
- PLE read-ahead became one batch of reads per window.

What carries over:

- The kernel gates carried from M3 are still missed: 32 launch classes of 1 MiB or more below 85% of the ceiling (27
  one-column, 5 multi-column), 36.9 ms of kernels per token, 1,107 launches, and launches under 1 MiB take 2.30 ms
  against 2. One token at a time measures 24.41 tok/s on code review against 27.

Contributions that move a milestone forward are welcome; see [CONTRIBUTING.md](../CONTRIBUTING.md).
