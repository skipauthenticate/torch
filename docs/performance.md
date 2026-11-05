# Performance model

Decode on the Orin is limited by memory bandwidth: each generated token needs a fixed set of weights plus the
experts it routes to, and the GPU can stream about 180 GB/s. This page turns that into numbers, so every kernel and
every milestone has a target we can check.

Sizes on this page are in GB (10^9 bytes) so they divide by GB/s; file sizes elsewhere are in GiB. **[file]**,
**[M2]**, **[M3]** and **[est]** mean what they mean in [architecture.md](architecture.md).

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

M3's 30 tok/s is 86% of the physical ceiling across roughly a thousand launches, which is why its gate is 27:
that is ambitious [est]. M3 measured 25.4 on code review (25.7 prose, 24.3 repetition) and merged below its gate by
the maintainer's decision; the gate carries to M4.
