# Performance model

Decode on the Orin is limited by memory bandwidth: each generated token needs a fixed set of weights plus the
experts it routes to, and the GPU can stream about 180 GB/s. This page turns that into numbers, so every kernel and
every milestone has a target we can check.

Sizes on this page are in GB (10^9 bytes) so they divide by GB/s; file sizes elsewhere are in GiB. **[file]**,
**[M2]** and **[est]** mean what they mean in [architecture.md](architecture.md).

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

On the development Orin, 2026-11-03 (M2 microbenchmarks):

- GPU streaming read: 155.4 GB/s from device memory and 158.2 GB/s from mapped pinned memory at 1 GiB; 150.8 and
  154.0 GB/s at 4 GiB.
- SSD `O_DIRECT` reads of 2 MiB at queue depth 4: 5.81 GB/s, 1.40 ms mean. Queue depths 1 and 2 were unreliable and
  are to be re-measured.
- SSD reads of 4 KiB at queue depth 1: 0.084 ms mean.

M3 adds the first decode numbers.

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
that is ambitious [est].
