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
| **M4** Verify windows | 1-4 columns in registers and 5-16 on int8 tensor cores; grouped experts; double-buffered DeltaNet state with operand replay; the draft-layer file, written and quantized by this project's tools, and the 4-bit drafter; adaptive window; exact speculative sampling | Logits bitwise identical across window widths and both column paths; greedy text identical with speculation on and off; sampled output passes a distribution test (goal 45 tok/s on code review) |
| **M5** Copy mode and residency | Prompt-lookup copies; prompt-informed expert warming; asynchronous misses with measured prefetch; PLE read-ahead; KV, indexer and draft-KV that grow with the sequence | Output independent of residency; predictor hit rates reported separately from residency (goals: prose 40, repetition 100) |
| **M6** Long context | Sparse indexer attention to 262,144 tokens with pooled keys; F16 KV cache; chunked prefill with the chunked DeltaNet algorithm; sessions that extend the previous sequence; cancel | Needle retrieval at 32K, 128K and 256K; a second turn does not prefill the first (goal: 1,000 prompt tok/s at 4,096) |
| **M7** First release | `torch_server` (OpenAI-compatible: tokenizer, chat template, sampling, streaming); `torch inspect --export-tokenizer`; install script; model-file verification; v0.1.0 | End-to-end install on a fresh Orin; every file in [model-files.md](model-files.md) verified by sha256 |

Contributions that move a milestone forward are welcome; see [CONTRIBUTING.md](../CONTRIBUTING.md).
