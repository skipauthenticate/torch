# Architecture

Torch runs one model, Qwen3.8-Flash-Next, on one device, the NVIDIA Jetson AGX Orin 64 GB. Every design choice
below follows from that narrow target: we never trade speed on the Orin for portability.

This document explains what the engine does and why. [performance.md](performance.md) has the numbers it is built
around and [model-files.md](model-files.md) the files it reads.

Numbers marked **[file]** were read from the GGUF headers at revision `ed59f920` on 2026-11-03. **[M2]**, **[M3]** and
**[M4]** were measured on the owner's Orin (JetPack 7, MAXN, GPU at 1.3 GHz) by this project's benchmarks
(2026-11-03, 2026-11-04 and 2026-11-05). **[est]** is arithmetic that a later milestone replaces with a measurement.

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
prompt ──► prefill (chunks of up to 4,096 tokens; decision 11) ──► KV, DeltaNet state, PLE history,
                                                                    the draft layer's K/V
        ┌───────────────────────────────────────────────────────────────┘
        ▼
  draft: MTP chain from the trunk's hidden state, stopped when more drafts cannot pay (decision 7);
         from M5 also a prompt-lookup copy
        ▼
  verify: one forward pass over the window of T = drafts + 1 tokens, 1 to 8 columns
        │  each dense weight row and each unique routed expert read once for the whole window
        ▼
  accept: greedy, the longest prefix equal to the model's own argmax, cut at the first EOS;
          sampled, an exact accept step per column (decision 7)
  commit the kept columns: DeltaNet state (replayed when the window was cut), convolution and PLE history;
  K/V rows past them are rewritten before they are read
  swap the window's missed experts into residency (decision 8); extend the draft layer over the kept positions
  repeat
```

Decode is a loop of **verify windows**. A window of T tokens reads every dense weight once and every expert used by
any of the T tokens once, then does T tokens' worth of arithmetic on data already on chip. A window is worth it when
the tokens it accepts outweigh what the extra columns cost; the engine measures both and chooses T per window.

## Design decisions

Each decision cites the measurement or the arithmetic that led to it; [performance.md](performance.md) has the
numbers.

### 1. A written memory budget, and the context allocated up front

On the owner's board about 47 GiB is available to the engine. [performance.md](performance.md#memory-budget) lists
every reservation. The KV cache (24 MiB per 1,024 tokens for the 12 attention layers in F16), the indexer's pooled keys
(64 B per token per layer) and the attention scratch are allocated in full for `--context` at start and zeroed once,
so the memory a request uses never changes while it runs: at 262,144 tokens they take 6.19 GiB, and 17,400-19,300
experts stay resident on the development board against 22,466 at 4,096 tokens (measured, M6). Memory that grows with
the sequence was dropped in M5: the device buffers are one allocation and experts sit in whole arenas, so growth would
need virtual memory management for both. `--lend-kv` lends the cache's unused tail to more experts until the sequence
reaches it instead (decision 10). Output never changes with residency (decision 8), only speed.

At start the engine prints its budget: each reservation, the number of experts resident, and the free-memory floor it
keeps; it refuses to start below the floor, so the operating system never runs short of memory. Weights are read
with `O_DIRECT`, so the page cache does not compete for the same memory. The reservations are estimates, and loading
40 GiB also frees page cache and idle memory that MemAvailable did not count at start (on the development Orin
MemAvailable fell 37.5 GiB for a planned 45 GiB [M3]), so after loading the engine reads MemAvailable again and makes
the next-ranked experts resident until it is down to the floor plus 1.5 GiB (except with `--resident`, which tests use
to fix residency exactly). The placement at start keeps that 1.5 GiB free as well, so a start whose estimates were
exact also ends at the floor plus 1.5 GiB rather than on the floor.

### 2. Columns: 1 to 8, each bitwise the one-column result

A verify window multiplies every matrix with T columns, 1 to 8 (`kMaxColumns`, `kernels/columns.h`). A row's weights
are read once for all its columns, and each column is computed with the one-column kernel's integer sums, F32 formula,
lane placement and reduction, so it is byte for byte the one-column result.

The canonical order, per format: a row is split into 32-element units, the size of an activation block. A unit's dot
product is an exact `int32` sum, multiplied by the unit's integer sub-scales in `int32`, then by the weight and
activation scales in F32 (`src/kernels/gemv/dot_q8.cuh` has the formula per format). Lane l of a row's warp sums units
l, l + 32, l + 64, ... in that order, and the 32 lane sums are added in a fixed butterfly. Every kernel and every
window width adds the F32 unit results in that order, so the logits do not depend on the width. `gemv_test` checks
every format, the grouped launches and the BF16 kernels at 1 to 8 columns against the one-column launch, byte for
byte, on the Orin.

The column kernels decode a weight unit once and apply it to each column; groups stage each column's activations
once; at 3 and 4 columns a lane group takes two rows, so one read of a staged activation block feeds both. Measured
[M4]: every dense kernel at 8 columns takes less than twice its 4-column time, and the dense matrices of a window
take 26.3 ms at one column, 28.4 at two, 34.3 at four and 52.5 at eight (performance.md, "Window cost by width").

Columns 9 to 16, and int8 tensor-core tiles for them, come with copy windows in M5. At the register wall of the
warp-per-row kernel (20 `int32` of Q8 activations per lane per column at K = 2,560) the staged column kernels
already reach 8 columns.

### 3. Experts grouped across the window

The T columns of a window route to up to 10 x T experts per layer, with overlap. One gate launch per layer places
them all (decision 6): it numbers the window's unique experts (at most 80) in order of first appearance, over
columns then ranks, so a one-column window numbers them by rank, as one token does. For each unique expert it writes
the pointers of its three matrices, a mask of the columns that routed it and the rank each one gave it, and it posts
every missing expert of the layer in one request.

Gate and up then run once per unique expert over the columns that chose it (`gemv_q8_experts_window`), and the down
projection reads each unique expert's matrix once for its columns, except Q2_0, which stays per route. The combine
sums each column's ten experts in its own rank order, so a column's result never depends on which other columns share
its experts. The file needs six expert kernels, each specialised at compile time: gate/up for IQ2_S, IQ3_XXS, IQ3_S
and IQ4_XS, down for IQ4_NL and Q2_0 [file].

### 4. Fused hyper-connection mixers

A hyper-connection mixer is norm, low-rank down projection, activation, up projection and a stream update: five
small kernels in a naive engine. We fuse each mixer into one launch whose phases are work items on atomic counters: a
block takes a phase-2 item only after the phase-1 counter is complete, and items are claimed only by running blocks,
so waiting cannot deadlock and no cooperative launch is needed. The reason is launch count and per-column compute, not
bandwidth: the 97 mixers move 1.29 GB per window [file], 8.3 ms at 155 GB/s [M2], and five launches each would be ~485
launches per window, 2.5-5 ms of launch overhead alone [est].

### 5. DeltaNet: two state buffers, commit after accept, replay by the same kernel

The DeltaNet state is 113 MB (36 layers x 48 heads x 128 x 128 x F32). Writing it after every window position would
cost T x 113 MB of writes per window. Instead `gdn_window` keeps each state column in registers while it walks the T
columns, reading the state before the window (buffer A) and writing the state after it (buffer B), and writes the
gates to a buffer of their own rather than over its inputs. The two buffers swap with every window, so there is one
graph per width and buffer parity.

A window commits only what it keeps (`accept(n)`):

- **All T kept**: nothing to redo; B is the state.
- **n < T**: the same kernel runs again over the first n columns from A, bitwise the state of n one-token steps. Each
  DeltaNet layer keeps its window's inputs for this, so there is no operand log and no state copy before a window.
- **The convolution history** (3 rows per layer) and **the PLE ring** (9 rows) are read, never written, during the
  window. In-window taps read the column buffers, and commit kernels write the kept columns' rows afterwards. So a
  rejected column never sits where a later window reads.
- **Attention K/V**: rows of rejected columns stay in the cache and are rewritten before any read.

`deltanet_test` and `ple_test` check every (T, kept) case against one-token steps bit for bit. Measured at position
1,800 [M4]: `gdn` 36.0 us per launch at T = 1, 41.6 at 2, 52.1 at 4 and 74.3 at 8; each commit 1.6-2.7 us.

### 6. One CUDA graph per window, with experts placed on the GPU

A verify window of each width (and DeltaNet buffer parity, decision 5) is recorded once as a CUDA graph and
replayed; a one-token decode is a window of one. A replayed graph
cannot wait for the host, so expert placement happens on the GPU: a device map gives every (layer, expert) pair the
addresses of its three matrices, or null when the expert is not resident, and after each router a gate kernel writes
the layer's expert pointer table from it. When a routed expert is absent, the gate kernel posts the layer's misses to
a mailbox in mapped pinned memory and waits for the expert store's I/O thread, which reads them into staging slots
and publishes a sequence number; then it points the table at the staged bytes (decision 8). The expert kernels read
every expert through the table, so they run the same code on the same bytes whatever the residency (goal 4).
Sequence numbers only grow, so nothing needs resetting between replays. The gate runs once per layer for the whole
window (decision 3): a missing expert is read once however many columns route to it, and all of a layer's misses go
out in one request. The mailbox and the table hold 80 experts; the staging buffers hold 84 of each matrix kind
(about 234 MB of mapped memory, within the budget's fixed reservation).

Measured on the Orin [M3]: the GPU-to-host-and-back round trip is 7.6 us without a read (bench/miss_roundtrip.cu),
against 21.2 us median and 144 us p99 for a host synchronization per layer; a replayed graph costs about 1.1 us per
kernel (bench/graph_gaps.cu), so the token's host work fell from 64 ms (launching ~2,000 kernels one by one) to 0.6
ms. The host's work per token is the embedding row, the PLE rows, one graph launch and one wait.

### 7. Speculation the hardware can afford

- **The MTP draft layer** (decision 12) drafts from the trunk's own state. Its input at position t is the trunk's
  4-stream residual after the head mixer's combine and the next token's embedding: S_s = fc_hidden(hnorm(H_t)_s) +
  fc_embedding(enorm(embedding)), with hnorm one RMS per stream. Then one attention and mixture-of-experts block, the
  head's mixer and the shared output head. A deeper draft runs the same block from its own state and the previous
  draft, one position further.
  - At prompt positions, and at every kept position of a window but the last, it runs a K/V-only pass: the attention
    mixer and the K/V projections, about 15 MB of weights.
  - Its K/V rows of accepted positions always come from the trunk's own states. `--check-draft-kv` checks that after
    N windows they equal a fresh K/V-only pass over the accepted history, bit for bit.
  - Its argmax scores an 81,920-row subset of the output head: the ids generated from the held-out calibration
    prompts, by frequency, then the rest by id. Measured [M8]: against 40,960 rows, depth-1 acceptance on the
    code-review benchmark's answers rose from 79.8% to 83.8% for +0.55 ms a draft step.
  - Its passes run as CUDA graphs, one per extend width and one per depth step.
  - Measured [M4]: one RMS per stream accepted 89.6% at depth 1 against 86.0% for one RMS over all four. The 40,960
    rows lost 0.02 accepted drafts of 3 against the full head, and cut a draft step from 4.8 to 2.1 ms (eager).
- **The window policy** chooses how many drafts a window verifies. It picks the d that maximises expected tokens per
  millisecond, E(d) / cost(d):
  - E(d) = 1 + q_1 + ... + q_d, with q_i the product of the learned acceptance rates of the drafts' probability
    buckets: 8 buckets over [0, 1], with priors.
  - The cost of each width follows the measured times as a moving average.
  - With `adaptive-stop`, drafting stops once one or two more drafts could not beat the best window so far.
  - It is deterministic and changes only how many drafts are verified. The benchmarks use `--spec 5 --spec-policy
    adaptive-stop`: at most 5 drafts, 2.0-2.8 verified on average [M4].
- **Prompt-lookup copies** (M5): when the recent output matches an earlier span of the context, the continuation of
  that span is proposed directly, up to T tokens per window [est].
- **Draft precision**: the draft layer's routed experts are 4-bit. On the review prompts they took the same number
  of windows as the 8-bit file on two prompts, and one more (81 against 80) on the third [M4]. Drafts are verified
  exactly, so draft precision affects speed only, never the output.
- **Exact speculative sampling** (kernels, M4; end to end with the server, M7):
  - Each column's full 248,320-token distribution is built on the GPU (`sample_columns`): penalties, temperature,
    exact top-k and top-p, min-p. The token is drawn by an inverse transform with a Philox-4x32-10 number keyed by
    (seed, position).
  - `accept_drafts` keeps the longest prefix of drafts equal to the drawn tokens, plus the first token that differs
    (or the bonus token). A draft x is then accepted with probability q(x), and otherwise the token is a draw from q
    with x removed and renormalised. So a sampled window gives the very tokens the same seed gives without
    speculation, for any drafts.
  - `sample_test` checks this over 30,000 seeds for each of four parameter sets: chi-square z-scores below 1.3 against
    a double-precision reference.
  - A window that runs past EOS or a stop string is cut at the first EOS in the accepted prefix.

### 8. Expert residency and the miss path

The experts (46.8 GiB) do not all fit next to the operating system, the KV cache and other software, so the engine
keeps as many resident as the budget allows and reads the rest from the SSD:

- **Placement is a ranking.** Route counts recorded while decoding held-out calibration prompts rank all 24,576
  (layer, expert) pairs, and the most-routed are resident at start, across all layers, until the budget's room is full
  (core/placement.h). Each resident matrix sits in a slot of its own layer's exact size. Measured [M3]: with the same
  85% of experts resident, ranking by counts from 800 calibration tokens raised a held-out prompt's hits from 85.4%
  to 95.0% and its decode from 8.65 to 12.48 tok/s.
- **Residency follows use.** After each window, up to 128 of the experts it read from the SSD take the slots of the
  least recently routed resident experts of their layers (`--swap 128`, `core/swaps.h`). Ties go to the lower
  calibration count, and memory stays constant.
  - The staging holds 80 experts for a whole window. When a miss does not fit, the I/O thread first swaps in the misses
    it holds (their layers are done for this window), within the same budget, with CPU copies: no GPU work may be queued
    while the window waits on that thread. Measured [M8, one process, the nine benchmark requests in order, equal
    bytes]: code review decode time -3.3% and -5.8% against 32 swaps without it, generation misses -27% to -56%; 128
    swaps without it changed nothing. Off with KV lending (`--no-swap-when-full` turns it off).
  - The gate kernel stamps every routed (layer, expert) with its window.
  - After the window, the newcomer's bytes are still in the staging buffers, so a GPU copy kernel moves them into
    the victim's slots on the store's own stream, while the drafter runs, and the map entries change in place. The
    next window waits for the copies.
  - None run during the prompt's one-token windows (`--swap-prompt 0`), which measured better.
  - Measured [M4]: generation misses fell 40-76% on the nine benchmark prompts. A swap costs about 0.05 ms, 0.25-0.40
    ms per generated window.
  - Warming from the prompt's own route counts, before generation, ended with more misses than no swaps, and the
    online swaps subsume it.
  - Misses still dominate below ~99% hits.
- **Misses are computed by the same kernel.** The missing expert is read into a staging slot in mapped pinned memory
  (`O_DIRECT` reads of a 4 KiB-aligned superset in 128 KiB chunks, since GGUF aligns tensors to 32 bytes; the chunks
  of a request are read in parallel, up to 16 in flight) and the kernel reads the slot through the pointer table
  (decision 6). Output does not depend on residency (goal 4). The development Orin's SSD occasionally leaves one read
  uncompleted until the NVMe driver's 30 s timeout (about once in 240,000 reads of 128 KiB); a miss read still pending
  after 50 ms is therefore issued again into a spare buffer and the first copy is used, and the late read's buffer
  stays out of use until it completes.
- **Prefetch is lossless and measured** (M5). An unpredicted miss costs its layer's segment the SSD latency. Reads
  start before the router runs, from predictors that only move bytes: the draft chain's own routing for the drafted
  positions, and the previous windows' experts. A predictor that moves bytes without raising the hit
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

The KV cache is F16: 24 MiB per 1,024 tokens, 6.0 GiB at 262,144 for the 12 attention layers. Up to 2,048 tokens of
context attention is dense. Above that, each trunk attention layer runs the model's sparse indexer
(`src/kernels/indexer/`):

- **Pooled keys**: the mean of each 4-token block's raw F16 keys, normed and rotated at the block's first position,
  kept as F16 (16 MiB per layer at 262K, 0.19 GiB for 12 layers). Raw keys live in a 16-row ring.
- **Scores**: 4 query heads of 128 against every visible block, in a fixed order (a sequential fmaf per head, ReLU,
  times 1/sqrt(128), heads summed in order, then + 0.0 so no score is -0.0). A CPU reference checks them bit for bit
  (`tests/parity/indexer_test.cpp`).
- **Selection**: each column keeps its 512 best blocks, ties to the lower block index, in ascending order, plus the
  0-3 open tail cells. Attention reads those 2,051 cells at most, in 128-cell chunks of the column's own list.
- **The 2,051-token edge**: with at most 2,051 visible tokens every block is selected and attention is bitwise the
  dense kernel's; the first sparse position is the 2,052nd token.
- **Windows and rollback**: everything the indexer keeps is indexed by position, so a rejected column needs no undo.

The draft layer attends densely to the last 16,384 positions from a K/V ring of that many rows (32 MiB instead of
0.5 GiB at 262K). Drafts are verified, so this changes acceptance only, and below 16,384 positions nothing.

`--lend-kv` keeps the rule that the context is allocated up front but lends the bytes of positions 16,384 and up, in
every K and V array and the pooled keys, to ~2,990 more experts (5.80 GiB at 262K). Before a window writes there, the
experts in the next 16,384 positions' bytes stop being resident, so the run is the same bit for bit; it is a second,
labeled number, not the default.

Costs, measured per layer with `bench/sparse_attention.cu` (performance.md has the tables): at 3,848 tokens a window
of 8 columns costs 1.2 ms more than dense attention at 2,040, mostly the BF16 indexer projections; at 262,144 tokens
scoring reads the 16.8 MB of pooled keys (113 us at 1 column, 365 at 8), selection takes 124 us and attention over
the cells 351 us at 8 columns. Against the reference runtime past 2,051 tokens, 95 of 96 greedy steps agree, the other is a near-tie, and
the mean KL is 0.0012-0.0026 at 4K and 32K.

### 11. Prefill and sessions

Prompt tokens run as prefill chunks of up to 4,096 tokens (`torch run --prefill N`); generated tokens always run as
windows. A chunk goes through the model op by op: each block runs over all of the chunk's columns before the next
block starts. Every column is bitwise what feeding the prompt in windows of 8 gives, so the path never changes the
output and chunk size is free to follow memory. Per stage:

- **Dense quantized products**: the int8 tensor cores (`kernels/gemm`, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q8_0). Each
  format's formula and its lane order are the one-column kernel's, so each column is bitwise `gemv_q8`.
- **BF16 x F32 products** (the hyper-connection mixers, the router and the other BF16 matrices): FP32 cores in the
  window order
  (`gemm_bf16`), bitwise `gemv_bf16`. An exact split onto the BF16 tensor cores measured too slow (82-2,695 us per
  input for the split alone), and accumulating in the tensor cores erred 14-90 times more than the one-column
  kernel.
- **Routed experts**: the chunk's N x 10 routes are bucketed by expert in route order. Each expert's gate, up and down
  rows then run once over its bucket on the int8 tensor cores, and the combine reads the results in rank order.
- **Non-resident experts**: a chunk of 256 or more columns routes nearly every expert, so all of a layer's
  non-resident experts are read ahead. A background thread fills two pinned pools of 64 experts (a 0.34 GiB budget
  line) while the GPU runs the layers before. There is no wait for the layer's router: the host waits only for a pass's
  reads and its pointer-table slot. Pools of 88 and 120 measured no faster and
  cost resident experts. Smaller chunks read a layer's misses after its router, 80 at a time.
- **DeltaNet**: the window recurrence over all columns in one launch, state in registers, the next column's inputs
  loaded during the current one.
- **Attention**: all K/V rows, raw indexer keys and pooled keys first; then scores, selection and attention over each
  column's cells in batches of 128 columns (decision 10).
- **The head** runs for the chunk's last column only. The draft layer's K/V-only pass reads the chunk's residuals 8
  columns at a time.

The chunk's activations (1.55 GiB at 4,096 columns) are one allocation with its own budget line.
Each block's temporaries share one region in turn.

A request that extends the previous sequence reuses it: KV, DeltaNet and convolution state, PLE hash history, the
indexer's ring and pooled keys and the draft layer's ring all continue, and only the new tokens are fed (`torch run
--then`). A prompt that does not extend the sequence starts a new one. With prefill, the model also keeps the state
that is not indexed by position (each DeltaNet state, the convolution and PLE rows, the indexer's raw-key rings) as it
was after the last chunk: 0.11 GiB, its own budget line, outside the lent tail. An extending request rewinds to that
point (`Model::rewind()`) and prefills the tokens generated since together with its new ones. Its state is then what
a fresh prefill of the whole prompt gives, even once prefill and windows stop being bitwise equal.
`Model::cancel()` stops a request between windows or chunks from any thread and leaves the sequence intact; the C API
(M7) exposes it.

### 12. The draft-layer file

The draft layer is a separate GGUF whose tensor names follow the main file's convention: `blk.48.*` for the MTP
block's attention, hyper-connections, router, routed experts and shared expert, and `blk.48.nextn.{eh_proj, enorm,
hnorm, hc_head_norm, hc_head_down, hc_head_up}`. `tools/make_draft.py` builds it, with this project's writer and
quantizers.

- **The source**: the 31 BF16 `mtp.*` tensors of the original checkpoint at a pinned revision, read with HTTP range
  requests (5.2 GB of the 360 GB; each read must return 206 with the asked Content-Range).
- **The layout**: the main file's names, `eh_proj` = [fc_embedding | fc_hidden] per row, and the indexer projection and
  the experts' gate_up split. The nine Gemma-style norm gammas are stored as 1 + w.
- **A check**: the BF16 intermediate equals an independently converted BF16 file in all 32 tensors, names, shapes,
  types and every byte [M4].
- **The file the engine runs** (`--experts iq4`):
  - routed gate and up are IQ4_XS, and routed down IQ4_NL (its 640-wide rows do not divide into IQ4_XS's 256-value
    blocks);
  - attention, shared experts and `eh_proj` are Q8_0;
  - the router, hyper-connection mixers and indexer are BF16, and the norms F32;
  - 1,480,786,592 bytes (1.38 GiB), a budget line of 1.46 GiB against 2.61 GiB for the all-Q8_0 file.
- **The IQ4 quantizers**: a per-block search for the scale that minimises squared error over nearest-level
  assignments, with exact nearest levels for the F16 scale it picks. IQ4_XS also has 6-bit sub-block scales. Round
  trip on the real experts: 7.5-7.6% RMS relative error.
- **Speed** [M4]: against the Q8_0 draft on the review prompts, the IQ4 draft left 636 more trunk experts resident and
  about half the misses, for 5% more tok/s.

[model-files.md](model-files.md) lists it. It is published only once the base model's license allows redistribution
(the Hugging Face card says "other").

### 13. Tokenizer, chat template and sampling

The tokenizer is `gpt2` BPE with the `qwen35` pre-tokenizer, 248,320 tokens, merges in the GGUF [file]: 248,044
BPE tokens, 33 added tokens and 243 unused padding ids, which sampling masks ([model-files.md](model-files.md)). The C
API takes token ids. The server (`torch_server`) tokenizes with `tokenizers` and renders `tokenizer.chat_template`
(9,993 characters of Jinja, with `enable_thinking`) with `jinja2`, both built from the vocabulary, merges and template
that `torch inspect --export-tokenizer` writes from the GGUF, so no second copy of the tokenizer is downloaded.
Sampling takes temperature, top-k, top-p, min-p, seed, and repetition, presence and frequency penalties as the OpenAI
API names them; the file recommends top_k 20, top_p 0.95, temperature 1.0 [file]. The penalties count only the tokens
the request has generated, from zero on every request, as the API penalises the text generated so far: counting a
4,096-token prompt would make common tokens impossible at frequency_penalty 0.5. Benchmarks run greedy.

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
  policy. Sampled output is the same for a fixed seed with any window policy, with or without speculation: each token
  is drawn at its position with the request's seed (measured on the 9 bench prompts x 3 seeds, M7).

## Correctness

- **Unit tests**: GGUF parsing (valid, corrupt, truncated and split files, plus a fuzz target); dequantization of
  all 13 formats against fixtures generated by the project's numpy reference and regenerated in CI.
- **Kernel parity**: every GPU kernel against a scalar CPU reference in float, within a stated relative tolerance;
  GPU against GPU **bitwise** across window widths 1 to 8 (16 and the tensor-core path from M5), across expert
  grouping, and across residency (a run with every expert resident and one with half of them give identical logits:
  goal 4).
- **Model parity**: next-token log-probabilities on fixed prompts against the committed reference log-probabilities
  (a second implementation of the model on the same file): greedy top-1 agreement at every position with a near-tie
  rule, and a mean per-position KL divergence within twice the reference's own spread between two runs with
  different numerics. M2 fixed the values (development.md, "Model parity").
- **Speculation is invisible**: greedy output with speculation on equals greedy output with it off, token for token,
  on every benchmark prompt, with and without swaps (M4: the nine prompts, 256 tokens). The draft layer's K/V after
  N windows equals a fresh pass over the accepted history, bit for bit (`--check-draft-kv`). The sampler passes a
  distribution test (`sample_test`).
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
  api/         C API used by the server and the tools (include/torch_engine.h, libtorch_engine.so)
server/torch_server/   OpenAI-compatible HTTP server (Python, over the C API; tokenizer and chat template live
                       here; the package is never named `torch`, a name a widely used Python package already takes)
tools/         inspect, bench, compare, make_draft
tests/         unit, fuzz, kernel parity, model parity
bench/         workloads, kernel microbenchmarks, results/<date>-<commit>.json
```
