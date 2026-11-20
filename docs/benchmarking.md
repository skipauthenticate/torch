# Benchmarking

Speed claims in this project are measured the same way every time, so numbers from different changes and different
people can be compared.

## Workloads

Each benchmark run sends three kinds of requests to the engine's HTTP server, three fresh prompts of each kind, 4,096
prompt tokens and 256 generated tokens per request, through the chat template with thinking off, greedy (temperature
0, seed recorded):

| Workload | Prompt | Why |
| --- | --- | --- |
| **Code review** | A synthetic Python module and "Write a detailed explanation of the code above..." | Long technical output; the benchmark used for this model |
| **Prose** | Synthetic prose and "Summarize the passage above in detail, then write a short story inspired by it." | Ordinary chat-like text, where drafts are accepted less often |
| **Repetition** | The same kind of prose and "Repeat the passage above word for word, from the beginning." | Fully predictable output: shows what copy-mode speculation can do |

The prompts are written by `bench/workloads/make_workloads.py` (synthetic Python modules for code review, synthetic
passages from a fixed word pool for prose and repetition, seeds 1-3) and committed as token ids, tokenized with the
base model's tokenizer at a pinned revision. Seeds 101-103 of the same generator are held out for calibrating the
expert placement (`bench/run_workloads.py --calibrate`); benchmark prompts are never used for calibration.

**Until M6** (sparse attention and prefill) the prompts are 1,455-1,529 tokens instead of 4,096, so prompt plus
generated tokens stay below the 2,048 positions where dense attention is exact, and the prompt is fed one token at a
time. Attention reads ~55 MB less per token at that length than at 4,096 (~0.3 ms at 180 GB/s, about 1% of a token),
so decode numbers carry over within ~1%. Until M7 (the server) `bench/run_workloads.py` runs the requests through
`torch run` and writes `bench/results/<date>-<commit>.json`.

## What is reported

- **Decode tok/s**: generated tokens divided by decode time, as the engine reports it (prompt time excluded). The
  median of the three requests of each workload, and each request's own value.
- **Prompt tok/s** and **time to first token**.
- **Expert residency and hit rate** per workload, the prefetch predictors' hit rates, and the minimum free memory
  during the run.
- **The slowest verify window** in ms and the number of windows above 200 ms per request (a few stalled windows per run
  can decide a median).
- **Output text**, and its sha256, so a speedup can be checked for identical output.

`torch bench` writes all of it, with the engine commit, JetPack release (`/etc/nv_tegra_release`), `nvpmodel -q`,
`jetson_clocks` state, the model and draft file sha256s and the exact command, to `bench/results/<date>-<commit>.json`.
Each milestone pull request commits its result file.

Report each workload separately. Averages hide the difference between predictable and ordinary text.

## Check a release comparison

Run the saved comparison after the board tests. Use the report for the engine revision under test.

```sh
python3 bench/release_gates.py bench/results/2026-11-19-90136a0-qualification.json
```

The command checks both model files against v0.4.0. It checks each request, the output hashes, and the GPU operation times. Add `--speed-targets` to also require 200 generated tokens/s and 2,000 uncached prompt tokens/s for each workload. Exit code 0 means all checks passed. Exit code 1 means at least one result failed. Exit code 2 means the report is incomplete. The board test still runs the hardware and model checks.

## Rules

- Same device state for before and after: same JetPack, same power mode (`nvpmodel -q`), same clocks, nothing else
  using the GPU or a large share of memory.
- Fresh prompts (the generator adds a nonce), so prefix caching cannot help.
- One run is a screening result; confirm a gain with a repeat run before claiming it.
- Report outliers (a request much slower than the others) instead of dropping them.
- For v0.5.0, each reported metric must meet the v0.4.0 result. Compare each model file and workload separately.
  Repeat lower results with the run order reversed. Keep the release in draft while a regression or a required check
  remains open.
