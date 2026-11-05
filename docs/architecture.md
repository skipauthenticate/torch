# Architecture

Torch runs one model, Qwen3.8-Flash-Next, on one device, the NVIDIA Jetson AGX Orin 64 GB. Every design choice
below follows from that narrow target: we never trade speed on the Orin for portability.

This document explains what the engine does and why. [performance.md](performance.md) has the numbers it is built
around, [roadmap.md](roadmap.md) the order in which it is built, and [model-files.md](model-files.md) the files it
reads.

Numbers marked **[file]** were read from the GGUF headers at revision `ed59f920` on 2026-11-03. **[M2]** and **[M3]**
were measured on the owner's Orin by this project's benchmarks (2026-11-03 and 2026-11-04). **[est]** is arithmetic that a later milestone replaces with
a measurement.

## Goals

1. **Fastest single-request decode on the Orin**, measured as generated tokens per second on three workloads
   (code review, prose, repetition; see [benchmarking.md](benchmarking.md)).
2. **No quality loss.** The engine computes the model the quantized file describes, weights as stored, activations
   quantized to 8 bits per 32 values for the integer dot products, F32 everywhere else, KV cache in F16. It never skips
   or prunes experts and never changes routing, sampling or attention math. Speculative decoding is verified exactly:
   greedy output is identical with speculation on or off, and sampled output has the same distribution (decision 7).
3. **The full 262,144-token context**, on the same 64 GB.
4. **Same answer wherever a weight lives.** Output never depends on which experts happen to be resident. (An engine
   that computes cache misses with a different kernel changes its answer when free memory changes.)
5. **Clean, documented, tested code** that a new contributor can read in an afternoon.

Non-goals: other GPUs, other models, multi-GPU, batching many users for throughput, training, vision (the model has
image and video tokens, `ple.image_token_id` 248056 [file]; the engine serves text only), Windows.

## The device

| Property | Value | What it means for us |
| --- | --- | --- |
| GPU | Ampere, sm_87, 16 SMs, 2,048 CUDA cores, 4 MB L2 | Small L2: weights are always streamed, never cached |
| Memory | 64 GB LPDDR5, 204.8 GB/s peak, ~180 GB/s measured streaming | Decode is bandwidth-bound; bytes per token decide speed |
| CPU | 12 Arm Cortex-A78AE cores | Spare cores do I/O, prefetch and drafting bookkeeping |
| Storage | NVMe SSD | Weights that do not fit are read from here. 2 MiB reads take 1.40 ms mean at queue depth 4, 5.8 GB/s [M2] |

CPU and GPU share the same DRAM. There is no PCIe copy: "loading to the GPU" means choosing an allocation the GPU
reads at full speed. Staging slots for SSD reads must be mapped pinned memory (`cudaHostAlloc`), because the CPU
writes them. The GPU reads mapped pinned memory as fast as `cudaMalloc` memory: 158.2 against 155.4 GB/s with 1 GiB
buffers, 154.0 against 150.8 GB/s with 4 GiB (`bench/memory_bandwidth.cu`, 2026-11-03). So all weights, every
resident expert and every staging slot live in mapped pinned memory, filled by `O_DIRECT` reads with no second copy.

## The model

Qwen3.8-Flash-Next (`qwen4exp` in GGUF): 48 layers, hidden size 2,560, vocabulary 248,320 [file].

- **Mixer layers.** 36 layers are Gated DeltaNet (linear attention with a 128 x 128 state per head, 16 key heads,
  48 value heads, a 4-tap causal convolution). Every fourth layer (3, 7, ..., 47) is gated full attention: 24 query
  heads, 2 KV heads, head size 256, RoPE on 64 dims at base 1e7. The file stores M-RoPE sections [11, 11, 10, 0]; for
  text every position axis is equal, which is plain NeoX RoPE on those 64 dims. A sparse indexer (4 heads x 128)
  pools keys 4:1, and each query attends to the top 512 pooled blocks (`indexer.top_k` = 2,048 cells) plus the open
  tail; below 2,048 tokens that is exact dense attention.
- **Mixture of experts.** Every layer routes each token to 10 of 512 experts (softmax over the top 10, renormalized)
  plus a sigmoid-gated shared expert. Expert width 640.
- **Hyper-connections.** The residual stream is 4 parallel streams (4 x 2,560); each layer mixes them in and out
  through low-rank (320) BF16 projections (`hc_attn_*`, `hc_ffn_*`: norm, down, up, inject).
- **Per-layer embeddings (PLE).** Layer 1 adds rows looked up from a hashed 3-gram table: 16 heads (8 per n-gram
  order) of 160 dimensions, each head a prime-sized vocabulary of about 20M rows, 320,001,536 rows in all, stored
  as IQ4_NL in 90-byte rows (26.8 GiB) [file]. The table lives on the SSD.
- **Multi-token prediction (MTP).** One extra attention layer drafts the next tokens for speculative decoding.
  **The ISTA file has no MTP tensors** [file]; the draft layer is a separate file (decision 12).

The weights come from the ISTA-DASLab GSQ-RCO "IQ3_S" GGUF, a mixed-precision file in two shards [file]:

| Tensor class | Size | Formats | Read per token |
| --- | ---: | --- | --- |
| Routed experts | 46.84 GiB | gate/up: IQ2_S (20 layers), IQ3_XXS (17), IQ3_S (10), IQ4_XS (1); down: IQ4_NL (39), Q2_0 (9) | 10 per layer, ~1.0 GB |
| Attention and indexer | 1.46 GiB | Q6_K, Q5_K, Q4_K, IQ4_XS, BF16, F32 | all |
| DeltaNet | 0.44 GiB | Q6_K, Q5_K, Q4_K, BF16, F32 | all |
| Hyper-connections, router | 1.30 GiB | BF16, F32 | all |
| Shared experts | 0.14 GiB | IQ4_NL, Q6_K, Q4_K, IQ4_XS, Q5_K, Q8_0, BF16 | all |
| PLE layer | 0.06 GiB | BF16, F16, F32 | all |
| Output head | 0.49 GiB | Q6_K | all |
| Token embeddings | 0.31 GiB | IQ4_XS | one row |
| PLE table (shard 2) | 26.82 GiB | IQ4_NL | 16 rows |

Thirteen formats in all. One expert (gate, up and down) is 2.0 to 2.3 MB depending on the layer's formats.

## How a token is generated

```
prompt ──► prefill (chunked; decision 11) ──► KV, DeltaNet state, indexer keys, PLE history
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
  draft: MTP chain (adaptive length, up to 7 drafts) or a prompt-lookup copy (up to 15)
        ▼
  verify: one forward pass over the window of T = drafts + 1 tokens
        │  each weight is read once for the whole window
        ▼
  accept: greedy, the longest prefix equal to the model's own argmax;
          sampled, exact rejection sampling position by position (decision 7)
  roll back DeltaNet state, conv and PLE history, indexer tail and KV to the accepted length; repeat
```

Decode is a loop of **verify windows**. A window of T tokens reads every dense weight once and every expert used by
any of the T tokens once, then does T tokens' worth of arithmetic on data already on chip. A window is worth it when
the tokens it accepts outweigh what the extra columns cost; the engine measures both and chooses T per window.

## Design decisions

Each decision cites the measurement or the arithmetic that led to it; [performance.md](performance.md) has the
numbers.

### 1. A written memory budget, and a KV cache that grows with the sequence

On the owner's board about 47 GiB is available to the engine. [performance.md](performance.md#memory-budget) lists
every reservation: at full 262,144-token capacity the non-expert items take ~15.8 GiB and leave ~31 GiB for experts,
two thirds of them (~16,400 of 24,576) [est]. Residency decides prose and repetition speed (decision 8), so the KV
cache (24 MiB per 1,024 tokens for the 12 attention layers in F16), the indexer keys and the draft layer's KV are
allocated as the sequence grows, in chunks, and the memory they take is freed by evicting the lowest-ranked resident
experts between windows, never during one. A 4,096-token conversation then runs with ~38 GiB of experts resident
(82%, ~20,000) instead of 67% [est]. Output never changes with residency (decision 8), only speed.

At start the engine prints its budget: each reservation, the number of experts resident, and the free-memory floor it
keeps; it refuses to start below the floor, so the operating system never runs short of memory. Weights are read
with `O_DIRECT`, so the page cache does not compete for the same memory. The reservations are estimates, and loading
40 GiB also frees page cache and idle memory that MemAvailable did not count at start (on the development Orin
MemAvailable fell 37.5 GiB for a planned 45 GiB [M3]), so after loading the engine reads MemAvailable again and makes
the next-ranked experts resident until it is down to the floor plus 1.5 GiB (except with `--resident`, which tests use
to fix residency exactly).

### 2. Columns: 1 to 4 in registers, 5 to 16 on int8 tensor cores, one reduction order

A warp-per-row dp4a kernel over K = 2,560 holds 20 `int32` of Q8 activations per lane per column: 80 registers at
4 columns, 320 at 16. That wall is why register-resident activations stop paying at about four columns [est]; M4
measures the per-column cost.

So every matrix-vector kernel (dense, shared expert, hyper-connection, head, routed expert) takes 1 to 4 columns with
the activations in registers, and 5 to 16 columns (verify windows and copy windows) go through int8 tensor-core tiles
(`mma.sync.m16n8k16`, sm_87): 16 weight rows, dequantized to int8 in registers, share each activation fragment held
in shared memory. Both paths compute one canonical order per format. A row is split into 32-element units, the size
of an activation block: a unit's dot product is an exact `int32` sum, multiplied by the unit's integer sub-scales in
`int32`, then by the weight and activation scales in F32 (`src/kernels/gemv/dot_q8.cuh` has the formula per format).
The F32 unit results are added in one fixed order: lane l of a row's warp sums units l, l + 32, l + 64, ..., and the
32 lane sums are added in a fixed butterfly. Integer sums are exact in any order and every kernel adds the F32 unit
results in that order, so the two paths and every window width give bitwise-identical logits. That identity is a CI
gate on the Orin.

### 3. Experts grouped across the window

The T tokens of a window route to up to 10 x T experts per layer, with overlap. Experts are grouped so each one is
read once and applied to every token that chose it; the down projection is applied per token (M4 measures whether
grouping it pays). The file needs six expert kernels, each specialised at compile time:
gate/up for IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, down for IQ4_NL and Q2_0 [file]. The accumulation order of a token's
result never depends on which other tokens share its group (see Numerics).

### 4. Fused hyper-connection mixers

A hyper-connection mixer is norm, low-rank down projection, activation, up projection and a stream update: five
small kernels in a naive engine. We fuse each mixer into one launch whose phases are work items on atomic counters: a
block takes a phase-2 item only after the phase-1 counter is complete, and items are claimed only by running blocks,
so waiting cannot deadlock and no cooperative launch is needed. The reason is launch count and per-column compute, not
bandwidth: the 97 mixers move 1.29 GB per window [file], 8.3 ms at 155 GB/s [M2], and five launches each would be ~485
launches per window, 2.5-5 ms of launch overhead alone [est].

### 5. DeltaNet: double-buffered state, operand log, replay on rejection

The DeltaNet state is 113 MB (36 layers x 48 heads x 128 x 128 x F32). Writing it after every window position would
cost T x 113 MB of writes per window (+0.6 ms per column at 180 GB/s) and 1.8 GiB of buffers at T = 16 [est]. Instead
the recurrence kernel reads the pre-window state once, walks the T positions in registers, writes the post-window
state to the other of two buffers, and logs each position's update operands (key, decay, beta, error: ~33 KB per
layer). Accepting the whole window is a pointer swap. Accepting k < T tokens is one replay pass from the pre-window
state over k operand rows in the original FMA order, bitwise identical to a fresh run and paid only when a draft was
rejected (~1.3 ms [est]). The convolution history, the
PLE hash history and the indexer tail are snapshotted the same way.

### 6. One CUDA graph per token, with experts placed on the GPU

A token (and from M4 a verify window of each width) is recorded once as a CUDA graph and replayed. A replayed graph
cannot wait for the host, so expert placement happens on the GPU: a device map gives every (layer, expert) pair the
addresses of its three matrices, or null when the expert is not resident, and after each router a gate kernel writes
the layer's expert pointer table from it. When a routed expert is absent, the gate kernel posts the layer's misses to
a mailbox in mapped pinned memory and waits for the expert store's I/O thread, which reads them into staging slots
and publishes a sequence number; then it points the table at the staged bytes (decision 8). The expert kernels read
every expert through the table, so they run the same code on the same bytes whatever the residency (goal 4).
Sequence numbers only grow, so nothing needs resetting between replays.

Measured on the Orin [M3]: the GPU-to-host-and-back round trip is 7.6 us without a read (bench/miss_roundtrip.cu),
against 21.2 us median and 144 us p99 for a host synchronization per layer; a replayed graph costs about 1.1 us per
kernel (bench/graph_gaps.cu), so the token's host work fell from 64 ms (launching ~2,000 kernels one by one) to 0.6
ms. The host's work per token is the embedding row, the PLE rows, one graph launch and one wait.

### 7. Speculation the hardware can afford

- **MTP chain** with an adaptive length: drafting stops at the first draft whose probability is below a gate, and
  the window width comes from a cost model fed by the measured per-column cost.
- **Prompt-lookup copies**: when the recent output matches an earlier span of the context, the continuation of that span
  is proposed directly and replaces the neural draft (copy windows are accepted whole on repetitive text, up to T tokens
  per window [est]).
- The draft layer's experts run at 4 bits (IQ4_XS), built from the original checkpoint (decision 12) [est: 2 bits loses
  acceptance, 8 bits gains little]; M4 measures acceptance per precision before the file is published. Drafts are
  verified exactly, so draft precision affects speed only, never the output.
- **Exact speculative sampling.** Drafts are argmax tokens (point-mass proposals). Greedy requests accept the longest
  prefix equal to the target's argmax. Sampled requests go position by position: the drafted token x is accepted with
  probability p(x) under the target's distribution after temperature, top-k, top-p, min-p and penalties; otherwise a
  token is sampled from p with x removed and renormalised, and the window ends there. This needs the full
  248,320-logit distribution per window position on the GPU and a seeded counter-based random generator. It yields the
  target's distribution exactly, so sampled output has the same distribution with speculation on or off, not the same
  tokens. A window that runs past EOS or a stop string is cut at the first EOS in the accepted prefix.

### 8. Expert residency and the miss path

The experts (46.8 GiB) do not all fit next to the operating system, the KV cache and other software, so the engine
keeps as many resident as the budget allows and reads the rest from the SSD:

- **Placement is a ranking.** Route counts recorded while decoding held-out calibration prompts rank all 24,576
  (layer, expert) pairs, and the most-routed are resident at start, across all layers, until the budget's room is full
  (core/placement.h). Each resident matrix sits in a slot of its own layer's exact size. Measured [M3]: with the same
  85% of experts resident, ranking by counts from 800 calibration tokens raised a held-out prompt's hits from 85.4%
  to 95.0% and its decode from 8.65 to 12.48 tok/s.
- **Prompt-informed warming.** After prefill, the experts the prompt routed to most are swapped in before decoding:
  a summary or a copy of the prompt uses the same experts as the prompt. A profile learned on one workload serves the
  others worse [est]; the engine reports hit rate per workload. Misses dominate below ~99% hits.
- **Misses are computed by the same kernel.** The missing expert is read into a staging slot in mapped pinned memory
  (`O_DIRECT` reads of a 4 KiB-aligned superset in 128 KiB pieces, since GGUF aligns tensors to 32 bytes; 16 reads in
  flight) and the kernel reads the slot through the pointer table (decision 6). Output does not depend on residency
  (goal 4). The development Orin's SSD occasionally leaves one read uncompleted until the NVMe driver's 30 s timeout
  (about once in 240,000 reads of 128 KiB); a miss read still pending after 50 ms is therefore issued again into a
  spare buffer and the first copy is used, and the late read's buffer stays out of use until it completes.
- **Prefetch is lossless and measured.** An unpredicted miss costs its layer's segment the SSD latency. Reads start
  before the router runs, from predictors that only move bytes: prompt-informed warming, the draft chain's own routing
  for the drafted positions, and the previous windows' experts. A predictor that moves bytes without raising the hit
  rate only costs bandwidth, so the engine reports each predictor's hit rate separately from residency and keeps only
  what measures well.
- **The SSD**: random 2 MiB reads issued as 128 KiB pieces take 1.56 ms median at queue depth 1 and reach 5.4 GB/s at
  depth 16 [M2]; a miss of a whole expert (three matrices, 1.5-2.7 MB) costs ~1 ms on the critical path.

### 9. PLE rows from the SSD, read ahead

A token needs 16 rows of 90 bytes, one per head, each in a different 4 KiB page: 16 random reads. PLE lookups depend
only on token ids, so the rows for drafted tokens are read while the window is verified, and a RAM cache (0.25 GiB)
holds hot rows. A 4 KiB read is ~0.09 ms [M2]; sixteen on the critical path, with the occasional slow one, would stall a
window. Until then (M3) the 16 reads are issued in parallel before each token, 0.3 ms per token, and hedged like the
expert reads: one 4 KiB read stalled for 30 s in an M3 benchmark run, which a read issued again after 50 ms avoids. The token embedding (0.31 GiB, one 1,360-byte row per token) can later move to the same path; it is worth ~150
experts.

### 10. Long context at the reference's precision

The KV cache is F16 (24 MiB per 1,024 tokens; 6.0 GiB at 262,144 for the 12 attention layers, plus 0.5 GiB for the
draft layer). The indexer keeps pooled keys only, 64 MiB per layer at 262K (0.8 GiB for 13 layers), plus the open
tail of raw keys. Costs at 262K [est]: indexer scoring reads 0.87 GB
per window (+4.8 ms; 0.11 GB at 32K), the top-512 selection per query breaks ties to the lowest block index, and the
selected K/V gather is ~50 MB per token. Prefill above 2,048 tokens uses the same sparse path.

### 11. Prefill and sessions

Prefill runs in chunks. On this board it is SSD-bound, not compute-bound: every chunk touches all 512 experts of every
layer, so the non-resident 18-33% (8-15 GiB) is read from the SSD once per chunk, 3-5 s at ~3 GB/s against ~0.3 s of GPU
time per 4,096-token chunk [est]. Chunk size therefore sets prompt speed, bounded by the activation memory in the budget
([performance.md](performance.md#prefill) has the table). Prefill uses the chunked DeltaNet algorithm and, above 2,048
tokens, the sparse indexer path (decision 10).

A request that extends the previous sequence reuses it: KV, DeltaNet and convolution state, PLE hash history and the
indexer tail. Without that every chat turn would re-prefill. The C API cancels a running request between windows.

### 12. The draft-layer file

The draft layer ships as a separate GGUF whose tensor names follow the main file's convention: `blk.48.*` for the MTP
block's attention, hyper-connections, router, routed experts and shared expert, and `blk.48.nextn.{eh_proj, enorm,
hnorm, hc_head_norm, hc_head_down, hc_head_up}`. Routed experts are IQ4_XS, everything else Q8_0, about 1.3 GiB [est].
`tools/make_draft.py` builds it from the MTP shards of the original checkpoint with this project's writer and
quantizers (M4). [model-files.md](model-files.md) lists it; it is published
only once the base model's license allows redistribution (the Hugging Face card says "other").

### 13. Tokenizer, chat template and sampling

The tokenizer is `gpt2` BPE with the `qwen35` pre-tokenizer, 248,320 tokens, merges in the GGUF [file]. The C API
takes token ids. The server (`torch_server`) tokenizes with `tokenizers` and renders `tokenizer.chat_template`
(9,993 characters of Jinja, with `enable_thinking`) with `jinja2`, both built from the vocabulary, merges and template
that `torch inspect --export-tokenizer` writes from the GGUF, so no second copy of the tokenizer is downloaded.
Sampling takes temperature, top-k, top-p, min-p, seed, and repetition and presence penalties as the OpenAI API names
them; the file recommends top_k 20, top_p 0.95, temperature 1.0 [file]. Benchmarks run greedy.

### 14. Measured, not guessed

Every kernel has a microbenchmark that reports achieved GB/s against the bytes it must move, and CI fails a kernel
change that does not report it. Decode prints a per-stage, per-window-width time table on request
(`TORCH_PROFILE=1`), and `torch bench` writes a result file with every number [benchmarking.md](benchmarking.md)
asks for.

## Numerics and determinism

- Weights exactly as stored. Activations are quantized to 8 bits per 32 values for the integer dot products. F32 for
  accumulators, norms, softmax, RoPE, the router, the DeltaNet state and the hyper-connection mixers. F16 for the KV
  cache and indexer keys.
- No fast-math, no flush-to-zero; `__fmaf_rn` and `__fmul_rn` where the order of operations matters; no
  floating-point atomics. One canonical reduction order per format (decision 2). A token's accumulation order never
  depends on the window width, on which tokens share its expert group, or on which block ran a work item.
- Decode attention splits the positions into fixed 128-row chunks: each chunk's maximum score, sum of exponentials
  and weighted sum of values are merged in ascending chunk order (kernels/attention/attention.h), so the result does
  not depend on which block finished first.
- Ties in argmax, router top-10 and indexer top-512 go to the lowest index.
- The same model, tokens and settings give the same greedy output on every run, with any residency and any window
  policy. Sampled output is reproducible for a fixed seed and window policy, and has the same distribution for any.

## Correctness

- **Unit tests**: GGUF parsing (valid, corrupt, truncated and split files, plus a fuzz target); dequantization of
  all 13 formats against fixtures generated by the project's numpy reference and regenerated in CI.
- **Kernel parity**: every GPU kernel against a scalar CPU reference in float, within a stated relative tolerance;
  GPU against GPU **bitwise** across window widths 1 to 16, across the register and tensor-core paths, across expert
  grouping, and across residency (a run with every expert resident and one with half of them give identical logits:
  goal 4).
- **Model parity**: next-token log-probabilities on fixed prompts against the committed reference log-probabilities
  (a second implementation of the model on the same file): greedy top-1 agreement at every position with a near-tie
  rule, and a mean per-position KL divergence within twice the reference's own spread between two runs with
  different numerics. M2 fixed the values (development.md, "Model parity").
- **Speculation is invisible**: greedy output with speculation on equals greedy output with it off, token for token,
  on every benchmark prompt; M4 adds a distribution test for sampled output.
- **Long context**: needle retrieval at 32K, 128K and 256K.

## Source layout

```
src/
  core/        errors, logging, the memory budget, small utilities
  gguf/        GGUF reader: the header is mapped; tensor bytes are read with pread / O_DIRECT into engine memory
  formats/     quantization formats: block layouts, the IQ codebooks (see NOTICE), CPU reference dequantization
  kernels/     CUDA kernels, one folder per operation (gemv, experts, attention, deltanet, hc, norm, sampling)
  model/       model configuration, weight loading, the layer graph
  runtime/     decode loop, speculation, graph segments, expert residency and I/O, PLE reader, sessions
  api/         C API used by the server and the tools
server/torch_server/   OpenAI-compatible HTTP server (Python, over the C API; tokenizer and chat template live
                       here; the package is never named `torch`, a name a widely used Python package already takes)
tools/         inspect, bench, compare, make_draft
tests/         unit, fuzz, kernel parity, model parity
bench/         workloads, kernel microbenchmarks, results/<date>-<commit>.json
```
