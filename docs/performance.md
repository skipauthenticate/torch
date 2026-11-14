# Performance model

Decode on the Orin is limited by memory bandwidth: each generated token needs a fixed set of weights plus the
experts it routes to, and the GPU can stream about 180 GB/s. This page turns that into numbers, so every kernel and
every milestone has a target we can check.

Sizes on this page are in GB (10^9 bytes) so they divide by GB/s; file sizes elsewhere are in GiB. **[file]**,
**[M2]**, **[M3]**, **[M4]** and **[est]** mean what they mean in [architecture.md](architecture.md).

## Bytes per token

From the GSQ-RCO IQ3_S file (per decoded token, before speculation):

| What is read | Bytes | Notes |
| --- | ---: | --- |
| Attention and indexer, DeltaNet, shared experts, PLE layer | ~2.3 GB | every token; 1.46 + 0.44 + 0.14 + 0.06 GiB [file] |
| Hyper-connection mixers and router (BF16) | ~1.4 GB | every token; 1.30 GiB [file] |
| Output head (Q6_K) | ~0.5 GB | every token; 0.49 GiB [file] |
| **Fixed per pass** | **~4.2 GB** | read once per verify window, however many tokens it holds |
| Routed experts, 10 of 512 per layer | ~1.0 GB per token | 2.0-2.3 MB each [file]; grows with the window, minus overlap |
| Indexer scoring (pooled keys, F16) | 0.8 MB per 1,024 tokens of context, per window | 25 MB at 32K, 0.2 GB at 262K [est] |
| Selected K/V (2,048 cells, 12 layers, F16) | ~50 MB per token | flat with context [est] |
| PLE rows and token embedding | 1.4 KB + 1.4 KB per token | SSD latency, not bandwidth |

**Autoregressive ceiling**: ~5.2 GB a token at 180 GB/s is ~35 tok/s at short context, ~30 tok/s at 262K.

## What speculation buys

A verify window of T tokens reads the fixed 4.2 GB once, plus the union of the experts its tokens route to. With A
tokens accepted per window on average:

| Scenario | T | A accepted | GB per window | Ideal tok/s at 180 GB/s |
| --- | ---: | ---: | ---: | ---: |
| Autoregressive | 1 | 1.0 | 5.2 | 35 |
| MTP chain on code review (today's acceptance) | 4 | 2.7 | ~7.5 | ~65 |
| MTP chain, longer accepted run | 8 | 4.5 | ~10.5 | ~77 |
| Copy mode on repetitive text | 8 | 8.0 | ~9 | ~160 |

Each extra column is assumed to add ~0.76 GB of new experts (about 7.6 of a token's 10 experts per layer are new to the
window [est]). M4 measures the overlap on our workloads. Real kernels reach 80-90% of streaming bandwidth at best, and
the window also pays for drafting and bookkeeping, so the realistic targets are lower than the ideal column.

## Memory budget

The owner's board has about 47 GiB available to the engine (other software runs on it). Reservations at full
262,144-token capacity [est]:

| Item | Size | Note |
| --- | ---: | --- |
| Non-expert weights, including `token_embd` | 4.2 GiB | always resident |
| Draft layer (IQ4_XS experts, Q8_0 rest) | ~1.3 GiB | 2.5 GiB if all Q8_0 [est] |
| KV cache, 12 attention layers, F16 | 6.0 GiB | 24 MiB per 1,024 tokens; allocated up front for `--context` (M6) |
| Draft-layer KV, F16 | 32 MiB | a ring of the last 16,384 positions (M6) |
| Indexer pooled keys, F16, 12 layers | 0.19 GiB | 64 B per token per layer |
| DeltaNet state x2, conv history, operand log | 0.25 GiB | 113 MB per state copy |
| CUDA graph segments, 16 window shapes | ~0.5 GiB | ~30 MiB per window shape [est] |
| Miss staging (2 layers in flight x 160 experts x 2.3 MB) | 0.75 GiB | mapped pinned memory |
| PLE row cache | 0.25 GiB | |
| Activations, prefill chunk workspace, logits | 0.5-1.0 GiB | chunk-size dependent; prefill never materializes every token's logits |
| CUDA context, allocator slack | 0.5 GiB | |
| **Non-expert total at 262K** | **~15.8 GiB** | |
| **Experts resident at 262K** | **~31 GiB: 67%, ~16,400 of 24,576** | |
| **Experts resident at a 4,096-token context** | **~38 GiB: 82%, ~20,000** | the KV, indexer and draft-KV reservations are 0.1 GiB |

MemAvailable on the development board was 24-29 GiB on 2026-11-03 with other software running [M2]; the engine reads
it at start and prints its own version of this table.

## Prefill

Prefill is compute-bound on this board, not SSD-bound as first estimated: a 4,096-token chunk reads its ~4,950
non-resident experts (~10 GB) in the background while ~16.3 s of GPU work runs (architecture decision 11). The race's
review prompt 1 (4,096 tokens, `--context 262144`, `calib-g-spec.txt`, one chunk), GPU seconds per stage from CUDA
events (`TORCH_PREFILL_STAGES=1`), on the Orin, before and after the tuning round of 2026-11-13. Its three kept commits
measured, one run each: cfc67c8 20.3 s (GEMM tree nodes padded against bank conflicts), 51c7338 18.7 s (BF16 passes
staged as one sequence, so the 10240 x 320 mixer up projection leaves the one-column kernel; BF16 5.02 s), af4641e
17.8 s (two 256-thread int8 GEMM blocks an SM; experts 5.36 to 4.57 s). The experts stage varies by about 0.1 s
between runs (the read-ahead).

| Stage | 2026-11-12, ac29367 | 2026-11-13, af82794 (final) |
| --- | ---: | ---: |
| Dense int8 products | 4.38 | 3.54 |
| BF16 x F32 products | 6.58 | 5.04 |
| Routed experts | 6.91 | 4.48 |
| Attention and the indexer | 1.24 | 1.24 |
| DeltaNet | 0.91 | 0.91 |
| Everything else | 1.11 | 1.11 |
| **GPU total** | **21.13** | **16.32** |
| **Prompt, wall clock** | **22.6 s (181 tok/s)**, twice | **17.7 s (231 tok/s)**, twice |

Before prefill the race prompts took 76-82 s in windows of 8 (147.7 s for review prompt 1 with the draft layer's
pass). The goal was 1,000 prompt tok/s at 4,096; after the int8 kernels were measured it was reset in writing to ~350.
The measured 231 misses it.

Memory: the chunk's activations are 1.55 GiB at 4,096 columns, the read-ahead pools 0.34 GiB and the
session snapshot 0.11 GiB, each its own budget line. Long prompts run at about the same rate: 131,001 tokens took
854.7 s (153 tok/s) and 262,040 took 1,763.6 s (149 tok/s) on 2026-11-11, before the BF16 tuning (the needles below).

## Measured so far

On the development Orin (JetPack 7, MAXN, GPU at 1.3 GHz), 2026-11-03 and 2026-11-04 [M2]:

- GPU streaming (`torch_bench_memory`, 1 GiB buffers, the best launch shape of a sweep, GPU at 1,300 MHz, MAXN),
  2026-11-04: read 179.8 GB/s from device memory, 179.9 GB/s from mapped pinned memory and 181.4 GB/s from registered
  memory; copy (both directions counted) 185.3, 180.9 and 188.0 GB/s. The best reads keep 4 independent 16-byte loads
  in flight per thread in 2 blocks of 512 threads per SM; one load per thread reaches only ~155 GB/s. 180 GB/s is the
  streaming bandwidth the M3 kernel gates are measured against.
- SSD `O_DIRECT` random reads (`torch_bench_ssd`, issued as 128 KiB preads, as the engine reads), 2026-11-04:

  | Read | Depth 1 | Depth 4 | Depth 16 |
  | --- | ---: | ---: | ---: |
  | 2 MiB: median ms, GB/s | 1.56, 1.33 | 3.41, 2.62 | 5.98, 5.39 |
  | 4 KiB: median ms | 0.093 | 0.093 | 0.101 |

  Single reads larger than the SSD's 128 KiB transfer limit (WD_BLACK SN850X) sometimes left one request
  uncompleted until the driver's 30 s timeout (11 of 3,000 reads at depth 1). 128 KiB reads did not in 9,000 tries,
  but in M3's longer runs about one in 240,000 did, and once a 4 KiB PLE read; the engine therefore issues any read
  still pending after 50 ms a second time (`runtime/hedged_reads.h`) [M3].
- Free memory: MemAvailable was 24-29 GiB while other software used the board and 50-51 GiB with it idle. With 51
  GiB the budget keeps 437-439 of 512 experts per layer resident (85%) and loads in 16-20 s.
- M3 probes, 2026-11-04: a GPU-to-host request through mapped memory and back takes 7.6 us median without a read
  (`bench/miss_roundtrip.cu`), 0.97 ms with one 1.5-2.7 MB expert read and 2.6 ms (p99 8.9) with ten; a host
  synchronization per layer took 21.2 us median, 144 us p99. A replayed CUDA graph costs about 1.1 us per kernel
  (`bench/graph_gaps.cu`); launching a token's ~2,000 kernels one by one took 64 ms of host time.
- Free memory after loading: MemAvailable fell 37.5 GiB for a planned 45 GiB (loading also frees page cache and idle
  memory that MemAvailable did not count at start), so the engine measures it again and fills it with experts down
  to the floor plus 1.5 GiB: 24,000-24,200 of 24,576 resident (98%), runs ending with 7.2-7.7 GiB available [M3].
- **M3 decode**, 2026-11-04 at commit 2611c30 (`bench/results/2026-11-04-2611c30.json`; the three workloads of
  [benchmarking.md](benchmarking.md) on their M3 prompts of 1,455-1,520 tokens, 256 generated tokens, greedy, medians
  of three requests): **code review 25.41, prose 25.68, repetition 24.32 tok/s**, at 99.94-100% routed hits with
  ~23,900 of 24,576 experts resident; MemAvailable 6.7-8.6 GiB at the end of each request. A token is one CUDA graph
  of 1,107 kernels: 36.8 ms of kernel time (`bench/results/kernels-2026-11-04-2611c30.json`, mean 148.6 GB/s, 83%
  of the ceiling) of ~39.4 ms per generated token; the rest is ~1 us per graph node and the 0.35 ms PLE read before
  each token. Per kernel class, at the start of M3 (`kernels-2026-11-04-790ecbf.json`) and at its end:

  | Kernel | GB/s before | GB/s after | ms per token before | after |
  | --- | ---: | ---: | ---: | ---: |
  | BF16 mixer up projections, 10240 x 320 | 93 | 169 | 6.85 | 3.79 |
  | BF16 mixer down projections + inject | 119 | 164 | 5.49 | 3.95 |
  | Routed gate+up, IQ2_S / IQ3_XXS / IQ3_S | 71 / 79 / 83 | 122 / 142 / 142 | 7.40 | 4.24 |
  | Routed down, IQ4_NL / Q2_0 | 93 / 64 | 158 / 134 | 4.59 | 2.63 |
  | Output head, Q6_K | 140 | 165 | 3.73 | 3.17 |
  | Other quantized matrices (attention, DeltaNet, shared expert) | 133 | 150 | 16.01 | 14.10 |
  | Decode attention at position 1,800 | 8 | 90 | 5.50 | 0.50 |
  | Launches under 1 MiB (count) | 1,463 | 573 | 5.88 | 2.26 |
  | **Total** (launches) | 2,044 | 1,107 | **55.5** | **36.8** |

  A re-run of the starting point later the same day measured 53.1 ms (its BF16 up projections at 132 GB/s), so the
  first row's "before" is partly harness variance. M3's gates were not all met: 29 launch classes of 1 MiB or more
  stay below 85% of the ceiling (most at 150-152 GB/s; IQ2_S gate+up at 122 is limited by its codebook lookups, the
  shared expert's grouped gate+up at 113-128, attention at 90), the launches under 1 MiB take 2.26 ms (gate 2 ms),
  and code review runs at 25.4 tok/s (gate 27). M3 merged with these numbers by the maintainer's decision; both gates
  carry to M4, whose multi-column kernels rewrite these paths [M3].
- Eager decode (M2, one token per step, one host round trip per layer): 7.77 tok/s, 128.8 ms per token, generating
  64 tokens after a 5-token prompt at 85% expert hits. Per token: 41.5 ms waiting for the GPU up to each router,
  71.8 ms reading the ~71 missed experts (about 1 ms each), 12.7 ms other host work. With every expert read from the
  SSD, a token takes 0.42-0.55 s.

## M4: verify windows and the draft layer

Measured on the development Orin (JetPack 7, MAXN, GPU at 1.3 GHz) on 2026-11-05 [M4]. Prompts are the M3 prompts of
1,455-1,520 tokens and 256 generated tokens, unless stated otherwise.

### Window cost by width

The kernel harness (`bench/kernels.cu --columns T`), ms per window, at position 1,800 for attention:

| Part of a window | T = 1 | T = 2 | T = 4 | T = 8 |
| --- | ---: | ---: | ---: | ---: |
| Dense matrices (every matrix but the routed experts) | 26.3 | 28.4 | 34.3 | 52.5 |
| Routed experts, routes drawn uniformly from 256 (almost no sharing) | 6.97 | 16.1 | 31.8 | not measured |
| Attention, 12 layers (`attention_decode`) | 0.51 | 0.80 | 1.42 | 2.63 |
| DeltaNet recurrence, 36 layers (`gdn`) | 1.30 | 1.50 | 1.88 | 2.67 |
| The whole window, end to end (model, not harness) | 40.3 | 51.0 | 73.6 | 123.5 |

- The end-to-end row is the model itself, teacher-forced (`torch run --window T`, review prompt 1, `calib.txt`),
  measured with stage timestamps on 2026-11-05 before the window down projection and the two-row dense kernels.
- The routed-expert row was measured before the down projection read each unique expert once; real windows share
  experts between columns, which that row does not.
- Every dense kernel at 8 columns takes less than twice its 4-column time. The output head (Q6_K, 248,320 x 2,560)
  takes 3,156 us at T = 1, 3,277 at 2, 3,989 at 4 and 5,662 at 8.
- The 6,144-wide Q6_K projections take 85.2 us at T = 1 and 125.2 at T = 4, since a lane group takes two rows at 3
  and 4 columns.

### The draft layer

- **Acceptance**: eager, one column, from the 9 benchmark prompts, 128 generated tokens, at depth 7. Acceptance at
  depth j is given every earlier draft right; "mean (3)" counts the leading right drafts of 3. It was measured with the
  8-bit draft file and the full head (`bench/draft_acceptance.py`):

  | Workload | d1 | d2 | d3 | d7 | Mean (3) | Mean (7) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Code review | 85.1 | 80.6 | 78.7 | 75.6 | 2.08 | 3.18 |
  | Prose | 83.7 | 78.9 | 74.2 | 62.7 | 1.99 | 2.85 |
  | Repetition | 100.0 | 99.7 | 98.1 | 91.1 | 2.98 | 6.32 |

  - One RMS over all four streams instead of one per stream: depth 1 falls from 89.6% to 86.0% over all workloads.
  - The 40,960-row head subset: 2.33 against 2.35 accepted drafts of 3.
  - A draft's own probability predicts its acceptance: 65% of drafts have p >= 0.99, and 98% of those are accepted,
    against 29-31% below p = 0.5.
- **Tokens per window**, greedy speculation with 3 drafts on all of them: code review 2.99, prose 3.15, repetition
  3.97. The generated ids are the same as without speculation on all nine prompts.
- **Drafting time per window**, 3 drafts, the IQ4 draft file:
  - 7.5 ms eager.
  - 6.6 ms eager with fc_hidden read once per pass (the four streams as four columns).
  - 6.2 ms with the passes as CUDA graphs, measured against 6.6 eager in the same session.
  - With the adaptive-stop policy and up to 5 drafts: 4.6-5.8 ms on prose.
- **The IQ4 draft file** against the all-Q8_0 one, the same binary on the review prompts: 33.75 / 31.92 / 30.15 ->
  35.45 / 33.48 / 31.68 tok/s; 636 more trunk experts resident and about half the misses.

### Residency during generation

`--swap N`: after each window, up to N missed experts take the slots of the least recently routed ones. On the
review prompts, each prompt running every setting in one session, tok/s and misses during generation:

| Prompt | No swaps | --swap 8 | --swap 16 | --swap 32 | --swap 16 --swap-prompt 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code review 1 | 32.72 (1,674) | 32.89 (1,319) | 36.08 (756) | 36.32 (684) | 36.10 (622) |
| Code review 2 | 31.47 (1,616) | 29.69 (1,982) | 32.44 (957) | 33.26 (847) | 33.78 (729) |
| Code review 3 | 27.94 (2,546) | 29.68 (1,808) | 31.48 (1,134) | 31.85 (1,006) | 32.01 (924) |

- In a later session, `--swap 32 --swap-prompt 0` beat `--swap 16 --swap-prompt 0` on all three prompts (35.14
  against 34.39 tok/s on the mean), and became the benchmark setting.
- Swaps cost 0.25-0.40 ms per generated window. A swap was 0.4 ms while `cudaMemcpyAsync` copied between two mapped
  pinned buffers on the CPU, and is about 0.05 ms as a GPU copy kernel.
- The prompt's own swaps did not help.
- Trunk residency varies by up to ~1,300 experts between runs of a session, with whatever else holds memory at load,
  which is worth about +-2% tok/s.

### The sampler

`sample_columns` with penalties, top-k, top-p and min-p takes 1.11 ms per window at T = 1, 1.20 at T = 4 and 1.57 at
T = 8, and with temperature only 0.23, 0.33 and 0.46 ms. Each column is one block, and its sweeps over the 1 MB of
probabilities are latency-bound.

### M4 decode

The benchmark command is `bench/run_workloads.py` with `--placement calib-spec.txt` and `--spec-args "--draft DRAFT
--draft-variants stream:40960 --draft-vocab VOCAB --spec 5 --spec-policy adaptive-stop --swap 32 --swap-prompt 0"`.
Medians of three requests:

| Workload | Speculative tok/s | Tokens per window | One token at a time |
| --- | ---: | ---: | ---: |
| Code review | 37.23 | 2.63 | 24.15 |
| Prose | 40.32 | 2.93 | 24.42 |
| Repetition | 53.85 | 5.71 | 23.75 |

- Measured on 2026-11-08 at 9dd7fb8 (`bench/results/2026-11-08-9dd7fb8-spec.json` and `2026-11-08-9dd7fb8.json`).
  The generated ids of all nine requests are identical with and without speculation.
- Residency decides much of code review's speed. In these runs MemAvailable was 14.9 GiB after loading, the
  top-up filled it to ~22,800-22,900 resident experts, and 99.95-99.98% of routes hit. On 2026-11-05 the same
  configuration loaded ~20,700 experts and code review measured a 35.5 median. Compare runs within one session.
- The one-token column uses `calib.txt`; the speculative one uses `calib-spec.txt`, route counts from the held-out
  calibration prompts decoded speculatively (`run_workloads.py --calibrate`).

## M5: host time per window

M4 left about 1.1 ms of host work per token that M3 did not have: the bench profile showed PLE rows 1.05 ms and other
host work 0.48 ms per token, against M3's 0.32 and 0.13. M5 removed it:

- A window's PLE rows (16 per column) are read as one batch of parallel reads instead of column after column.
- A single-chunk read wakes one read worker instead of all of them. Waking all of them for every 4 KiB PLE row made
  them contend: the batched reads alone were slower until this change.
- A window's commit (convolution history, the DeltaNet replay when a window is cut, the PLE ring) launches as one CUDA
  graph per (state parity, kept columns, replay) instead of about 70 launches: 0.30 to 0.02 ms of host time per window.

Prompt 1 of each workload, the M4 binary against M5's in one board session, alternating, 2026-11-09 (ids identical):

| Workload | M4 tok/s | M5 tok/s | PLE rows, ms per token | Other host work, ms per token |
| --- | ---: | ---: | --- | --- |
| Code review | 40.37 | 40.17 | 1.07 to 0.32 | 0.49 to 0.21 |
| Prose | 40.09 | 41.37 | 1.05 to 0.35 | 0.50 to 0.22 |
| Repetition | 53.51 | 55.95 | 1.04 to 0.32 | 0.45 to 0.15 |

One token at a time, M3's binary (d0e0a13) against M5's in one session: code review 24.67 and 24.51, prose 25.24 and
24.97, repetition 23.84 and 24.00 tok/s. M4's one-token regression (3.3-3.6%) is down to ~1%, with ~165 fewer experts
resident in the M5 runs (23,194 against 23,358).

Measured and not kept (2026-11-09):

- **Copy proposals** (prompt lookup: propose the continuation of the longest earlier match of the last tokens):
  repetition fell from 53.70 to 39.59 tok/s. 46 of 49 windows verified a copy, but windows took 131.8 ms against
  105.7 and kept 5.27 tokens against 5.71. Reverted.
- **A draft-expert subset** (`--draft-experts 256`: only the most-routed draft experts resident): frees 0.64 GiB for
  ~308 trunk experts, but prose loses drafts; mean +0.4%, inside the run-to-run noise.
- **Int8 tensor cores for Q6_K at 1-8 columns** (prototype, bitwise equal to the one-column result): 2560x6144 at
  1 / 4 / 8 columns takes 107.4 / 98.2 / 104.3 us against dp4a's 88.5 / 113.0 / 157.3. Faster from 3 columns; it
  moves to M6 with the other formats, where prefill needs the same kernels.
- `--spec 7` on repetition: +1.3%.

### M5 decode

Same command and placements as M4 decode. Medians of three requests:

| Workload | Speculative tok/s | Tokens per window | One token at a time |
| --- | ---: | ---: | ---: |
| Code review | 38.23 | 2.63 | 24.41 |
| Prose | 41.31 | 2.87 | 24.85 |
| Repetition | 55.98 | 5.71 | 23.93 |

- Measured on 2026-11-09 at 090d839 (`bench/results/2026-11-09-090d839-spec.json` and `2026-11-09-090d839.json`).
  The generated ids of all nine requests are identical with and without speculation.
- 22,400-22,530 experts resident in the speculative runs (M4's: ~22,850), 99.92-99.97% of routes hit.
- With 6.7 GiB taken before loading (what a 262,144-token context's F16 KV, draft-layer KV and pooled indexer keys
  will need), the same binary measured code review at 35.43 tok/s (median of three; 19,100-19,400 resident) against
  38.70 without, in one session: about 1 tok/s per 1,000 resident experts.

### Placement from generated tokens

The placement ranks experts by a route-count file. Until 2026-11-09 the calibration counted every prompt position
as well as the generated tokens: `calib-spec.txt` held 14,729 positions, about 92% of them prompt tokens (9 held-out
prompts of ~1,500 tokens, 128 generated each). A prompt's routing predicts its generation's poorly: on the three code
review prompts, the review's English uses different experts from the code it reads. `--route-counts` now records only
the generated tokens, and `run_workloads.py --calibrate` generates 256 tokens per held-out prompt.

Static misses of the generated tokens of the three review benchmark prompts (offline, from per-request route counts):

| Placement | 19,400 resident | 22,950 resident |
| --- | --- | --- |
| Prompts and generated tokens (old) | 3,870 / 3,090 / 4,010 | 419 / 205 / 338 |
| Generated tokens only | 739 / 865 / 1,722 | 77 / 122 / 237 |
| Old counts blended half and half with the request's own prompt routing | 5,300 / 4,150 / 5,201 | 478 / 274 / 380 |

Measured with the tool's own files (`--calibrate` on the held-out prompts), the old placement against the new one,
alternating in one session on 2026-11-09 with 6.7 GiB taken first (~19,800 resident), ids identical:

| Review request | Old placement | Generation-only | Generation misses |
| --- | ---: | ---: | --- |
| 1 | 38.38 | 39.51 | 1,074 to 336 |
| 2 | 35.93 | 37.62 | 1,133 to 399 |
| 3 | 34.98 | 36.49 | 1,241 to 678 |

The median rises 4.7%. An earlier A/B, with a placement built offline from the same kind of counts and without the
6.7 GiB, measured review +2.8% (median of three), prose 1 +1.7% and repetition 1 +3.2%. The full benchmark with the
tool's files (`bench/results/2026-11-09-calibgen-spec.json` and `2026-11-09-calibgen.json`, a different session from
M5's with ~700 fewer experts resident at the same MemAvailable) measured speculative review 38.26, prose 42.05 and
repetition 56.85 tok/s, and one token at a time 24.99, 25.08 and 25.15; generation misses fell 40-90% on every
request except review 3 (speculative 255 to 281). The prompt misses more, which lengthens prefill.

6.7 GiB is what a 262,144-token context's F16 KV, draft-layer KV and pooled indexer keys will take. Also measured
there and not kept: `--swap 64` and `--swap 128` (medians 35.65 and 35.70 against 35.58 with 32).

## M6: long context

Attention above 2,048 tokens of context is sparse: the indexer keeps each column's 512 best 4-token blocks plus the open
tail (decision 10). One layer's attention per window, from `bench/sparse_attention.cu` (CUDA events, graphs over 12
layers' caches, random inputs), times 12 layers, ms per window, measured on 2026-11-10:

| Columns | Dense at position 2,040 | Sparse at 3,848 | Difference | Sparse at 32,000 | Sparse at 262,000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.55 | 1.22 | +0.67 | 1.51 | 4.10 |
| 4 | 1.54 | 2.48 | +0.94 | 3.34 | 8.74 |
| 8 | 2.93 | 4.12 | +1.19 | 5.26 | 11.02 |

Per layer, us, the same runs:

| Tokens, columns | BF16 projections | Score | Select | Attention over the cells |
| --- | ---: | ---: | ---: | ---: |
| 3,848, 1 / 4 / 8 | 20.9 / 34.9 / 51.0 | 4.8 / 13.9 / 18.9 | 10.6-10.8 | 53.9 / 133.1 / 242.8 |
| 32,000, 1 / 4 / 8 | 20.9 / 35.0 / 50.0 | 16.4 / 50.1 / 56.4 | 18.4-18.6 | 58.8 / 164.4 / 296.3 |
| 262,000, 1 / 4 / 8 | 20.9 / 34.7 / 50.3 | 112.9 / 350.7 / 365.6 | 124.1-124.9 | 67.6 / 201.8 / 351.0 |

- Gate K4 (a sparse window at 4,096 within 1 ms of the dense window at 2,048) passes at 1 and 4 columns. At 8 it is
  0.1-0.4 ms over (this bench, and same-token model runs plus the bench's dense 4K-2K difference), because of the
  indexer's BF16 projections at 8 columns: the multi-column BF16 kernel, shared with the mixers.
- Scoring reads the pooled keys: 16.8 MB per layer at 262,144 tokens, ~112 us at 150 GB/s, which the 1-column score
  reaches. Windows of 3 or more columns use a kernel that reads each key once for all columns (444 -> 351 us at 4
  columns); the F32 arithmetic of 8 columns is ~100 us per layer at the GPU's peak [est]. Select is 124 us per layer at
  262,144 (186 before the change of 2026-11-10).
- The same 4,096-token prompt, positions 3,600-4,096, sparse against dense attention: 115.22 against 116.25 ms per
  window at 8 columns, 72.59 against 72.96 at 4 (one session, 22,900 experts resident).
- `--context 262144` reserves 6.19 GiB for the KV cache and indexer keys and leaves 19,303 of 24,576 experts resident
  (78.5%; 22,466 at `--context 4096`); MemAvailable 7.42 GiB at the end. A 4,096-token prompt's logits are bitwise
  those of `--context 4096` at all 4,096 positions, and window 8 is bitwise window 1.
- Parity with the reference runtime (F16 KV, teacher forcing, 32 greedy steps each): 4,046 tokens 32/32 top-1 and mean
  KL 0.00255, 4,028 tokens 32/32 and 0.00115, 32,700 tokens 31/32 plus one near-tie (Torch's choice 0.10 nats
  below the reference's own in the reference's ranking) and 0.00210 (gate 0.0041).
- The draft layer attends to the last 16,384 positions from a K/V ring (`src/model/draft.h`): its budget line is
  1.49 GiB at `--context 262144` (2.00 GiB with a full cache). On the three race review prompts (4,096 tokens, 256
  greedy tokens, `--context 262144`, placement `calib-g-spec.txt`) the ids with speculation equal the one-token run's:
  29.84 / 29.33 / 30.74 tok/s at 2.34 / 1.98 / 2.76 tokens per window, against 22.62 / 22.59 / 22.55 one token at a
  time, 18,025-18,168 experts resident. Past the ring's wrap (a 16,500-token review prompt, 64 tokens, `calib.txt`)
  the ids also agree, at 2.79 tokens per window and 21.71 tok/s (17,881 resident).
- With the draft layer the prompt now goes in windows of 8 too: review prompt 1 in 147.7 s instead of 319.0 at
  window 1, the draft's K/V rows bitwise those of a one-token pass (`--check-draft-kv`). The race's prose and
  repetition prompts (4,096 tokens, 256 greedy tokens, `--context 262144`; speculation with `calib-g-spec.txt`, one
  token at a time with `calib-g.txt`), ids identical in all six:

| Prompt | Speculative tok/s | Tokens per window | Generation misses | One token at a time | Its misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose 1 / 2 / 3 | 32.85 / 32.30 / 30.45 | 2.11 / 2.18 / 2.09 | 495 / 543 / 621 | 23.81 / 23.86 / 23.85 | 450 / 418 / 427 |
| Repetition 1 / 2 / 3 | 42.62 / 44.90 / 43.62 | 4.71 / 4.98 / 4.02 | 1,218 / 1,315 / 1,223 | 23.36 / 23.30 / 23.30 | 758 / 783 / 710 |

  18,877-19,616 experts resident in the speculative runs, 19,520-19,531 one token at a time; prompts in 76-82 s.
- `--lend-kv` (a second number, not the headline): the bytes of positions 16,384-262,143 in every attention layer's K
  rows, V rows and pooled keys hold ~2,990 more experts (5.80 GiB) until a window writes there. Race review prompts,
  `--context 262144`, speculation, `calib-g-spec.txt`, without and with it back to back (ids identical in all six):

| Review | Without: tok/s | Resident | Generation misses | With: tok/s | Resident (lent) | Generation misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.94 | 17,994 | 3,051 | 32.51 | 20,929 (2,990) | 796 |
| 2 | 29.16 | 17,875 | 3,011 | 31.98 | 20,805 (2,988) | 835 |
| 3 | 29.03 | 17,366 | 4,360 | 34.12 | 20,338 (2,993) | 1,020 |

  Median 29.03 to 32.51 tok/s (+12%). The top-up after loading varied from 514 to 1,167 experts between runs
  (MemAvailable after loading 9.98-11.24 GiB). A prompt that crosses the lend mark (16,500 tokens, `--lend-from 2048`,
  the mark moved 8 times) gives bitwise the logits of the run without lending at every position.
- Needle in a haystack (`bench/needle.py`: the workloads' synthetic prose, one sentence with a 7-digit number at a
  depth, a question): 32,700 prompt tokens at `--context 32768`, window 8, `calib-g.txt`, 32 greedy tokens. The answer
  comes first at depths 10, 50 and 90%; prompts in 792.7, 781.0 and 775.9 s (about 41 tokens per second). With
  prefill (chunks of 4,096, `calib-g.txt`, 32 greedy tokens, 2026-11-11, before the BF16 tuning), the answer comes
  first at depth 50% at 131,001 tokens (`--context 131072`, prompt 854.7 s) and at 262,040 tokens (`--context
  262144`, prompt 1,763.6 s). GPU seconds at 262,040: dense 280, BF16 618, experts 448, attention 198, DeltaNet 58,
  other 70. MemAvailable stayed at 7.35 GiB or above (the floor is 6). Depths 10 and 90% were not run at 128K and
  256K.
- Prefill against windows of 8 (review prompt 1, `--context 4096`, 8 generated tokens, logits byte for byte at every
  chunk end and every generated step): 300 tokens in chunks of 300, 64 and 37, and 64 with `--resident 256`; 2,200
  tokens in chunks of 512, 333 and 4,096 (across the 2,051/2,052 edge). All equal, and so are: `--spec` with
  `--check-draft-kv` (ids and draft K/V); `--lend-from 1024` at `--context 16384`, the mark crossed inside chunks
  with 173 experts lent; `--resident 0`; `--then` against windows of 8 with `--then`.
- The race prompts with prefill (2026-11-13 at 51c7338, chunks of 4,096, `--context 262144`, `calib-g-spec.txt`, 256 greedy
  tokens, `--swap 32 --swap-prompt 0`): in all nine, the ids one token at a time and with speculation equal the ids
  of the window-path runs of 2026-11-09. Prompts in 18.2-18.8 s one token at a time and 19.4-20.0 s with the draft
  layer's pass. Decode, one token at a time: prose 23.67-23.73, repetition 22.97-23.02, review 22.44-22.68 tok/s.
  Speculative: prose 32.13 / 31.56 / 30.11, repetition 40.05 / 42.89 / 43.25, review 28.75 / 29.72 / 31.29 tok/s.
- Race round 2 (2026-11-13, af82794, the race's rules: the nine 4,096-token prompts, 256 greedy tokens, `--context
  262144`, one process per request with the load excluded, prefill in one chunk of 4,096): median decode 30.25 tok/s on
  code review, 31.37 on prose and 42.38 on repetition; time to first token 18.8-19.0 s (216-218 prompt tok/s; 92-242 s
  on the window path in round 1); the speculative ids equal the one-token ids in all nine. With `--lend-kv`, a labeled
  second number: 32.20 / 32.67 / 46.99 tok/s, time to first token 18.3-18.6 s.
- Sessions (`torch run --then`): on review prompt 1, a second turn of 29 new tokens after 4,128 takes 0.6 s of prompt
  against 131.7 s for a fresh run of the same 4,157 tokens, and generates the same 32 ids; so do the cancelled
  (`--cancel-after 3`) and speculative variants. With prefill, a second turn rewinds to the end of the last chunk and
  prefills the generated tokens again with its new ones. Turn 1 is 1,024 tokens and 8 generated; turn 2 adds 500
  and prefills 508 in 4.2 s, against 12.5 s for a fresh prefill of the 1,532 tokens. Turn 2's rows equal the fresh
  run's byte for byte, at chunks of 512 and 37, with `--spec` (ids), with lending (the mark crossed in turn 2), and
  after a request cancelled between chunks (`--cancel-prompt-after`).
- Memory at `--context 262144`: MemAvailable falls by the device buffers' 5.9 GiB at cudaMalloc already, and they are
  zeroed once at allocation, so the top-up after loading counts no untouched KV. During a 32,700-token prompt
  MemAvailable stayed at 6.73 GiB or above.
- Measured from 2026-11-09 to 2026-11-13; the final numbers are at commit af82794.

## M7: the C API, the server and sampling

Measured on 2026-11-13 at f1e1b8b, in one session on the Orin, with the race's settings (the nine 4,096-token race
prompts, `--context 262144`, `--prefill 4096`, `calib-g-spec.txt`, `--swap 32 --swap-prompt 0`, 256 tokens). Each
request is its own process: per prompt, the frozen M6 binary (af82794), this build's `torch run` and a `torch_server`
started for that one request, in turn. **The board's memory moved during the session**: another user's session was on
the board, and the experts added after loading went from 0 (the M6 binary and `torch run` on review 1 and 2) to
2,500-3,400 (every later run). Decode follows the resident count, about 1 tok/s per 1,000 experts, so compare within a
row. Resident experts are at the end of the request (the server's from its `torch` field).

Greedy and speculative, tok/s (resident experts):

| Prompt | M6 binary | `torch run` | `torch_server` |
| --- | ---: | ---: | ---: |
| Review 1 | 26.09 (15,593) | 25.88 (15,616) | 25.64 (15,478) |
| Review 2 | 25.77 (15,617) | 25.45 (15,610) | 29.93 (18,735) |
| Review 3 | 31.33 (18,930) | 29.82 (18,020) | 30.23 (18,382) |
| Prose 1 | 31.97 (18,901) | 31.07 (18,439) | 30.84 (17,726) |
| Prose 2 | 31.23 (18,771) | 30.64 (18,505) | 30.80 (17,994) |
| Prose 3 | 29.93 (18,922) | 30.07 (18,719) | 30.54 (18,598) |
| Repetition 1 | 41.80 (18,504) | 41.06 (18,550) | 42.05 (18,810) |
| Repetition 2 | 39.32 (18,931) | 43.61 (18,557) | 38.09 (17,998) |
| Repetition 3 | 42.92 (18,629) | 43.16 (18,044) | 43.72 (18,067) |
| **Median** review / prose / repetition | 26.09 / 31.23 / 41.80 | 25.88 / 30.64 / 43.16 | 29.93 / 30.80 / 42.05 |

The ids of all 27 requests are equal, prompt by prompt, and equal the window path's of 2026-11-09. Where the resident
counts are close the three are within 2% (prose 3, repetition 1 and 3); the medians differ by the residency swings
(review). Prompts took 18.6-19.8 s.

Sampled at the model file's defaults (temperature 1.0, top_k 20, top_p 0.95), `torch run`, seeds 1 and 2, speculative
and one token at a time in turn; median of the 3 prompts x 2 seeds per workload, tok/s:

| Workload | Speculative | One token at a time |
| --- | ---: | ---: |
| Review | 29.48 | 18.00 |
| Prose | 28.93 | 21.85 |
| Repetition | 42.11 | 19.43 |

In every pair the speculative ids equal the one-token ids (18 pairs); 2,102-3,226 experts were added after loading
(one run 1,155). The repetition pairs ran an hour later in the session, after a run of them was stopped at the 6 GiB
MemAvailable floor. Sampled speculation is 1.3-2.2x one token at a time. Against greedy speculation in the same session
at similar residency it is 1-7% lower on review (29.48 against 29.82-31.33) and prose (28.93 against 30.07-31.07) and
about equal on repetition (42.11 against 41.06-43.61).

Thinking on (the template's default, `reasoning_effort` left out of the race request bodies), through the server,
seed 1, 256 tokens, all of them reasoning: speculative 24.81 / 28.11 / 28.02 tok/s against one token at a time 23.30 /
24.51 / 24.60 on review 1, prose 1 and repetition 1 (17,410 resident), ids equal; 2.52 / 1.95 / 1.84 tokens per
window. The gain (6-15%) is smaller than with thinking off.

`tools/check_orin.sh` at f1e1b8b: unit and kernel parity tests, model parity (mean KL 0.00211), graph = eager and the
residency identity pass; the kernel gates carried from M3 are missed (35 launch classes of 1 MiB or more below 85% of
the ceiling, launches under 1 MiB 2.30 ms against 2). `ctest`: every unit and parity test passes.

## Targets

Speed numbers are **goals**. The **gates** a milestone must pass are its correctness checks and bandwidth checks in
[roadmap.md](roadmap.md). A milestone may close below a speed goal with the measured number written in its pull
request; it may not close with a failed gate.

| Milestone | Code review | Prose | Repetition | Prompt tok/s at 4,096 |
| --- | ---: | ---: | ---: | ---: |
| M3, autoregressive | 30 (gate: >= 27) | | | |
| M4, verify windows + MTP | 45 | | | |
| M5, copy mode + residency | 50 | 40 | 100 | |
| M6, long context | | | | 1,000 (about 4 s to the first token); reset to ~350 |

M4 measured 37.2 on code review against its goal of 45; the goal carries to M5.

M5 measured 38.2 on code review against 50, 41.3 on prose against 40 and 56.0 on repetition against 100. The
review and repetition goals carry forward with the int8 tensor-core kernels, which move to M6.

M6's prompt goal of 1,000 tok/s at 4,096 was reset in writing to ~350 once the int8 kernels were measured. M6
measured 231 (17.7 s for the race's 4,096-token review prompt, 2026-11-13): missed.

M3's 30 tok/s is 86% of the physical ceiling across roughly a thousand launches, which is why its gate is 27:
that is ambitious [est]. M3 measured 25.4 on code review (25.7 prose, 24.3 repetition) and merged below its gate by
the maintainer's decision; the gate carries to M4.
