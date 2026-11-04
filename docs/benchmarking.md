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

Prose and repetition use the project's prompt generator (`bench/workloads/`, M3): a fixed word pool and seeds, with a
nonce per prompt.

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

## Rules

- Same device state for before and after: same JetPack, same power mode (`nvpmodel -q`), same clocks, nothing else
  using the GPU or a large share of memory.
- Fresh prompts (the generator adds a nonce), so prefix caching cannot help.
- One run is a screening result; confirm a gain with a repeat run before claiming it.
- Report outliers (a request much slower than the others) instead of dropping them.
