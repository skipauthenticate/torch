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
| Indexer scoring (pooled keys, F16) | 3.3 MB per 1,024 tokens of context, per window | 0.11 GB at 32K, 0.87 GB at 262K [est] |
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
| KV cache, 12 attention layers, F16 | 6.0 GiB | 24 MiB per 1,024 tokens; allocated as the sequence grows |
| Draft-layer KV, F16 | 0.5 GiB | grows with the sequence |
| Indexer pooled keys, F16, 13 layers | 0.8 GiB | grows with the sequence |
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

Prefill is SSD-bound on this board. Every chunk touches all 512 experts of every layer, so the non-resident experts are
read once per chunk; GPU time is ~0.3 s per 4,096-token chunk (47 GiB of weights streamed once, plus compute) [est].
With an SSD at 5.8 GB/s at queue depth 4 [M2] (the table below keeps the conservative 3 GB/s until M6 measures prefill):

| Chunk | Non-resident experts read per chunk | Prompt tok/s at 67% resident (262K capacity) | at 82% resident (short context) |
| ---: | ---: | ---: | ---: |
| 4,096 | 15.5 GiB / 8.4 GiB | ~700 | ~1,500 |
| 8,192 | same | ~1,300 | ~2,500 |
| 16,384 | same | ~2,200 | ~3,700 |

Chunk size is bounded by the activation memory in the budget. A 262,144-token prompt takes minutes at any chunk size;
sessions that extend the previous sequence do not prefill again (architecture decision 11).

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

## Targets

Speed numbers are **goals**. The **gates** a milestone must pass are its correctness checks and bandwidth checks in
[roadmap.md](roadmap.md). A milestone may close below a speed goal with the measured number written in its pull
request; it may not close with a failed gate.

| Milestone | Code review | Prose | Repetition | Prompt tok/s at 4,096 |
| --- | ---: | ---: | ---: | ---: |
| M3, autoregressive | 30 (gate: >= 27) | | | |
| M4, verify windows + MTP | 45 | | | |
| M5, copy mode + residency | 50 | 40 | 100 | |
| M6, long context | | | | 1,000 (about 4 s to the first token) |

M4 measured 37.2 on code review against its goal of 45; the goal carries to M5.

M3's 30 tok/s is 86% of the physical ceiling across roughly a thousand launches, which is why its gate is 27:
that is ambitious [est]. M3 measured 25.4 on code review (25.7 prose, 24.3 repetition) and merged below its gate by
the maintainer's decision; the gate carries to M4.
