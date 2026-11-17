# Development

How to build the engine, run its tests, and find your way around the code.

## Toolchain

| Tool | Version | Notes |
| --- | --- | --- |
| JetPack | 7.x (L4T R39) | On the Orin. Earlier JetPack releases are not supported |
| CUDA | 13.x | Comes with JetPack 7; the engine compiles for `sm_87` only |
| CMake | 3.24 or newer | |
| Ninja | any recent | Recommended generator |
| C++ compiler | GCC 13 or newer, Clang 18, or Apple Clang | C++20 |
| Python | 3.10 or newer | Server, tools and test fixtures; CI uses 3.12 |

The host-only parts (the GGUF reader, the format reference code, `torch inspect` and their tests) build on any Linux
or macOS machine. CI builds them with GCC 13 and Clang 18 on Ubuntu 24.04 and with Apple Clang on macOS 14. Windows
is not supported: the reader uses POSIX `pread` and `mmap`.

## Building

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On a machine without CUDA (a laptop, CI), the same commands build the host-only parts: the GGUF reader, the format
reference code, the memory budget, the direct-I/O reader, the PLE hash and `torch inspect`. When CMake finds a CUDA
compiler it also builds the GPU engine for `sm_87` only: the kernels, `torch run`, the parity tests and the
microbenchmarks. On the Orin, put CUDA 13 on the path first (`export PATH=/usr/local/cuda-13.2/bin:$PATH`).

Useful options:

| Option | Default | Effect |
| --- | --- | --- |
| `-DTORCH_CUDA=ON/OFF` | ON when CUDA is found | Build the GPU engine |
| `-DTORCH_TESTS=ON/OFF` | ON | Build the tests |
| `-DTORCH_WERROR=ON/OFF` | OFF | Treat compiler warnings as errors (CI turns it on) |
| `-DTORCH_SANITIZE=ON/OFF` | OFF | Build with AddressSanitizer and UndefinedBehaviorSanitizer; use with a Debug build |
| `-DTORCH_FUZZ=ON/OFF` | OFF | Build the fuzz target `gguf_fuzz`; needs LLVM Clang |
| `-DCMAKE_BUILD_TYPE=Debug` | Release | Debug build |

`build/torch --version` prints the version and the git commit the build was configured from.

## Tests

```sh
ctest --test-dir build --output-on-failure      # everything that can run on this machine
ctest --test-dir build -L unit                   # host-only unit tests (what CI runs)
ctest --test-dir build -L parity                 # GPU kernels against their CPU references (Orin only, from M2)
```

The unit tests:

| Test | What it checks |
| --- | --- |
| `dequant` | The CPU reference dequantization of all 13 formats against the numpy reference's values in `tests/fixtures/quant_blocks.bin` |
| `gguf` | The reader on `tests/fixtures/tiny.gguf`, and on truncated and corrupt headers the test builds |
| `tokenizer_export` | `torch inspect --export-tokenizer` on small headers the test builds: JSON escapes, the added tokens' flags, unused ids left out, and the files it refuses |
| `inspect_json`, `inspect_header_only` | `torch inspect` on the fixture, and on a copy of it cut inside the tensor data |

With the sanitizers:

```sh
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTORCH_SANITIZE=ON
cmake --build build-asan
ctest --test-dir build-asan -L unit --output-on-failure
```

On macOS 26 with Apple Clang 17, AddressSanitizer hangs before `main` even for an empty program (seen on
2026-11-03), so run this build on Linux. UndefinedBehaviorSanitizer alone works on macOS.

The unit tests also cover the memory budget (`budget`), direct reads and the read pool (`direct_io`, which reads a
file it writes in the build directory, because tmpfs does not support `O_DIRECT`) and the PLE n-gram hash
(`ple_rows`). The parity tests (Orin only) compare each GPU kernel with a scalar CPU reference on random data:
`gemv` (every quantized format, BF16, and experts in mapped and device memory), `norm`, `hc`, `deltanet`,
`attention`, `moe` and `ple`.

### Running the model

`torch run` feeds token ids through the model one at a time, then decodes greedily and prints each step's top
log-probabilities. It finds shard 2 next to shard 1 by its split name:

```sh
build/torch run Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --tokens 48,25,3437,369,220,16,22,2942,220,17,18,30,198,32,25 -n 16
```

It prints the memory budget first (MemAvailable, every reservation, how many experts per layer are resident) and
refuses to start when less than the 6 GiB floor would be left. `--resident K` keeps exactly K experts per layer
resident (0 reads every routed expert from the SSD), `--logits FILE` writes every step's raw logits, and `--json
FILE` writes every step's top log-probabilities for the parity script. The context is limited to 2,048 tokens until
milestone M6: below that, the model's sparse attention selects every cell, so dense attention is exact.

`--tokens @FILE` reads the ids from a file (as `bench/workloads/` stores them), `--placement COUNTS` keeps the
most-routed experts of a route-count file resident (with `--resident-total N`, the placement and the top-up stop at N
experts in total, so two runs compare at equal bytes whatever memory is free) and `--route-counts COUNTS` adds the
routing of this run's generated tokens to one (the prompt's routing predicts the generation's poorly), and `--quiet
--profile` prints only the speed, the generated ids and where decode time went. Each token runs as one CUDA graph;
`TORCH_EAGER=1` launches kernel by kernel instead (the output is byte-identical), and `TORCH_DUMP_DIR=dir
TORCH_DUMP_STAGES=router,...` writes the named stages' values per position and layer.

### Verify windows and speculative decoding

- `--window T [--accept K]` feeds the prompt in windows of T tokens (1 to 8), keeping K of each. Every position's
  results must equal `--window 1`'s bit for bit, which tests windows and their rollback.
- `--draft FILE` loads the draft layer (`tools/make_draft.py OUT --experts iq4` builds it). `--draft-variants
  stream:40960 --draft-vocab VOCAB` picks one RMS per stream and a 40,960-row head subset; `bench/draft_acceptance.py
  vocab` writes VOCAB.
- `--spec D` then generates speculatively, with windows of the last token and up to D drafts (1 to 7).
  `--spec-policy fixed|adaptive|adaptive-stop` chooses how many a window verifies (default adaptive-stop).
- `--swap N` makes up to N missed experts resident after each window, and `--swap-prompt M` caps it during the
  prompt (M4 runs `--spec 5 --swap 32 --swap-prompt 0`).
- `--check-draft-kv` decodes the history again after a speculative run and requires the draft layer's K/V to match,
  bit for bit.
- `--draft-depth D --acceptance-report FILE` measures the draft layer's acceptance per depth, one token at a time.
- The generated ids are the same with and without `--draft`, `--spec` and `--swap`; only speed changes.

### Benchmarks

`bench/run_workloads.py --torch build/torch --model MODEL --out RESULT.json --placement COUNTS` runs the three
workloads of [benchmarking.md](benchmarking.md) and writes the result file; `--calibrate COUNTS` records the route
counts from the held-out calibration prompts. `--spec-args "ARGS"` passes the speculative-decoding arguments to `torch
run` in both modes, and the result then records each request's tokens per window, drafts verified and the split of a
window's time. The M4 runs use `--placement calib-spec.txt` and ARGS `--draft DRAFT --draft-variants stream:40960
--draft-vocab VOCAB --spec 5 --spec-policy adaptive-stop --swap 32 --swap-prompt 0`; calib-spec.txt is `--calibrate
calib-spec.txt --placement calib.txt` with ARGS up to `adaptive-stop`. `build/torch_bench_kernels MODEL --json FILE`
times every kernel of a token in GB/s against the 180 GB/s ceiling, and `bench/kernel_gates.py FILE` checks that result
against the decode gates.

### Model parity

`tests/model/parity.py` (standard library only) compares Torch's next-token log-probabilities with those of a second
implementation of the model on the same file. `tests/model/reference.json` holds that implementation's output for
five fixed prompts: the token ids, 32 greedy continuation tokens and the top-5 log-probabilities at every generated
position.

```sh
python3 tests/model/parity.py compare --torch build/torch --model MODEL-00001-of-00002.gguf --ref tests/model/reference.json
```

`compare` feeds each prompt followed by the reference's own continuation (teacher forcing), so one near-tie does not
end the comparison. The gate: at every generated position Torch's most likely token is the reference's greedy token,
or the reference ranks Torch's choice within `--tie` of its own; and the mean over all positions of KL(reference ||
Torch), on the reference's top-5 tokens plus the remaining mass, is at most `--max-mean-kl`. The largest log-probability
gap on tokens both top-5 lists share is printed but not gated: it falls on improbable tokens and varies as much between
two runs of the reference itself.

`spread` measures the same numbers between two runs of the reference with different numerics. Each file's committed
reference is that implementation with the experts of the first 2 of 48 layers on the CPU. That is the fewest that
leave 8 GiB free on the Orin: with every expert on the GPU, the weights alone would leave about 6.3 GiB (projected
from the server's memory breakdown at 2 layers). Three pairs
measure the reference's own spread, each against the committed run:

- the experts of 4 layers on the CPU instead of 2;
- the prompt in batches of 8 tokens instead of 512, which uses other GPU kernels;
- flash attention off instead of on.

`--max-mean-kl` is twice the largest of the three, and `--tie` stays 0.38. Measured on the Jetson AGX Orin 64 GB on
2026-11-16, IQ3_S file (`tests/model/reference.json`):

| Pair | Greedy tokens agreeing | Mean KL | Largest top-5 gap |
| --- | ---: | ---: | ---: |
| Reference, 2 vs 4 CPU layers | 155 of 160 | 0.00146 | 0.98 |
| Reference, prompt batches of 512 vs 8 | 160 of 160 | 0.00087 | 0.97 |
| Reference, flash attention on vs off | 122 of 160 | 0.00164 | 0.95 |
| Torch vs the committed reference | 160 of 160 | 0.00143 | 1.27 |
| Torch with `--fast-prefill` | 160 of 160 | 0.00179 | 1.05 |

The thresholds are the defaults: `--tie 0.38` and `--max-mean-kl 0.00328`.

The UD-IQ3_XXS file has its own reference, `tests/model/reference-ud-iq3_xxs.json`: the same five prompts computed
the same way on that file. Never compare one file's logits with the other file's reference.

| Pair | Greedy tokens agreeing | Mean KL | Largest top-5 gap |
| --- | ---: | ---: | ---: |
| Reference, 2 vs 4 CPU layers | 110 of 160 | 0.00258 | 0.88 |
| Reference, prompt batches of 512 vs 8 | 129 of 160 | 0.00094 | 0.76 |
| Reference, flash attention on vs off | 160 of 160 | 0.00298 | 1.10 |
| Torch vs the committed reference | 158 of 160 (2 near-ties) | 0.00249 | 1.16 |
| Torch with `--fast-prefill` | 157 of 160 (2 near-ties, 1 miss) | 0.00430 | 1.73 |

```sh
python3 tests/model/parity.py compare --torch build/torch --model Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf \
  --ref tests/model/reference-ud-iq3_xxs.json --max-mean-kl 0.00596
TORCH_REF=tests/model/reference-ud-iq3_xxs.json TORCH_MAX_KL=0.00596 tools/check_orin.sh   # the same in the checks
```

Its threshold is 0.00596. The file stores the hyper-connection mixers as Q8_0, so both implementations round those
mixers' inputs to 8 bits per 32 values, where the IQ3_S file's BF16 mixers take F32 inputs. Those 8-bit inputs amplify
any change in rounding upstream, so the reference also spreads more against itself: flash attention on vs off moves
it by 0.00298 on this file, against 0.00164 on the IQ3_S file.

With `--fast-prefill` (every position its own prefill chunk, so the prompt's products run on the tensor cores), the
UD-IQ3_XXS file fails the top-1 rule at one position. In "Q: What is 17 times 23?", at the 19th generated token, the
reference's greedy token is a newline (log-probability -0.52; Torch -1.09) and Torch's is the end of the turn (Torch
-0.62; the reference -1.43). The reference ranks them 0.91 apart, above the 0.38 near-tie rule. So UD-IQ3_XXS with
`--fast-prefill` fails model parity, although its mean KL (0.00430) is under the threshold. The default path passes
both conditions on both files, and so does `--fast-prefill` on the IQ3_S file.

**Long prompts.** The three prompts of 4,046, 4,028 and 32,700 tokens use a reference with the experts of 4 layers on
the CPU: with 2, its 33,000-token context would leave about 7.1 GiB. Torch reads each prompt in prefill chunks of 4,096
and then generates 32 greedy tokens; `spread` compares the two runs up to the first token where they differ. The pairs
are the same three, with two limits:

- prompt batches of 8 for the first two prompts only: at the first prompt's 28.3 tokens/s, the third would take over
  19 minutes, the board's limit for one job;
- flash attention on vs off with both runs in batches of 128: off in batches of 512 needs a compute buffer 1.1 GiB
  larger, which does not fit.

| File | 4 vs 8 CPU layers | Batches of 512 vs 8 | Flash attention | Threshold | Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | 0.00086 | 0.00127 | 0.00146 | 0.00292 | 0.00193 over 73 positions, 1 near-tie |
| UD-IQ3_XXS | 0.00098 | 0.00271 | 0.00199 | 0.00542 | 0.00345 over 80 positions, 3 near-ties |

**Why these references and thresholds changed (2026-11-16).** Until then the committed references put the experts of
4 layers on the CPU, and the spread came from more CPU layers (4 vs 8, 8 vs 32, 32 vs 40), with thresholds 0.0041
(IQ3_S) and 0.00312 (UD-IQ3_XXS). An op-by-op comparison on 2026-11-15 showed that the reference's CPU experts compute
a different function: they round their inputs to 8 bits per 256 values, where its GPU kernels and Torch round per 32.
On layer 0, that rounding alone accounts for the whole difference between the two implementations on both files
(the first prompt, 36 positions). Each of Torch's operations checked in layers 0 to 2 matched a numpy computation of
its own definition to within 1e-7 (relative). So the old spread mostly measured
the reference's CPU path, which the committed reference used in its first 4 layers. The new rule moves the reference as
close to all-GPU as memory allows and adds two spreads that stay on the GPU. **The rule was set after Torch's numbers
against the old references were known**: short prompts 0.00196 (IQ3_S) and 0.00272 (UD-IQ3_XXS), long prompts 0.00192
and 0.00345, both above the thresholds the old rule gives long prompts (0.00172 and 0.00196).

### Residency independence

The output must not depend on which experts are resident. Run the same tokens with different residency and compare
the logits byte for byte:

```sh
build/torch run MODEL-00001-of-00002.gguf --tokens ... -n 16 --logits logits-max.bin   # as many as fit
build/torch run MODEL-00001-of-00002.gguf --tokens ... -n 16 --resident 256 --logits logits-256.bin
build/torch run MODEL-00001-of-00002.gguf --tokens ... -n 16 --resident 0 --logits logits-0.bin
cmp logits-max.bin logits-256.bin && cmp logits-max.bin logits-0.bin
```

### On the shared Orin

The development board is shared with other work. Every build or run that uses the GPU or more than ~2 GB of memory
holds the board's GPU lock (a `flock` on a shared lock file); builds run at `nice -n 10` with `-j 4`, or with `-j 1`
outside the lock. Copy the tree to the board with `rsync -rlpD --checksum` rather than `-a`: `-a` keeps the source
machine's modification times, and the build on the board can then skip files that changed.

`tools/check_orin.sh` runs everything a milestone must pass on the board, from the development machine:
`TORCH_ORIN=user@host TORCH_MODEL=/path/on/board/MODEL-00001-of-00002.gguf TORCH_GPU_LOCK=/path/to/lock
tools/check_orin.sh` copies the tree, builds it, and under the lock runs the unit, kernel parity and model parity
tests, the graph-versus-eager and residency checks, and the kernel harness against its gates.

### Fuzzing

The reader parses untrusted files, so it has a libFuzzer target. It needs LLVM Clang and its runtime libraries (on
Ubuntu 24.04, the `clang-18` and `libclang-rt-18-dev` packages):

```sh
CC=clang-18 CXX=clang++-18 cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTORCH_FUZZ=ON -DTORCH_SANITIZE=ON
cmake --build build-fuzz --target gguf_fuzz
mkdir -p build-fuzz/corpus && cp tests/fixtures/tiny.gguf build-fuzz/corpus/
build-fuzz/tests/gguf_fuzz -max_total_time=60 build-fuzz/corpus
```

CI runs it for 60 seconds on every pull request. A crash, a sanitizer report or an exception other than
`torch::Error` fails the run.

### Test fixtures

The files in `tests/fixtures/` are small and synthetic, never model data. `make_fixtures.py` writes them with the
project's numpy reference of the formats (`reference.py`, written from [formats.md](formats.md)) and its GGUF writer
(`gguf_writer.py`). `requirements.txt` pins numpy, the only dependency, so each run writes the same bytes. CI
regenerates them and fails if a byte differs from the committed files, and runs `tests/tools/make_draft_test.py`,
which checks the draft-layer builder's Q8_0 blocks and writer on synthetic tensors. After changing the script:

```sh
python3 -m venv .venv
.venv/bin/pip install -r tests/fixtures/requirements.txt
.venv/bin/python tests/fixtures/make_fixtures.py
```

## Formatting

C++ is formatted with clang-format 19.1.7 and the repository's `.clang-format`; CI checks every file. Install it in
the same virtual environment:

```sh
.venv/bin/pip install clang-format==19.1.7
git ls-files '*.h' '*.cpp' | xargs .venv/bin/clang-format -i
```

## Inspecting a model file

`torch inspect` prints a GGUF file's split, metadata and bytes per format; `--tensors` adds every tensor's format,
shape, size and file offset, and `--json` prints the same as one JSON object per file. With `--header-only` it reads
a file that stops after its tensor table, so the first 11 MiB of shard 1 shows the whole model without downloading
51 GiB:

```sh
curl -L -r 0-11534335 -o shard1-head.gguf \
  'https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
build/torch inspect --header-only shard1-head.gguf
```

`--export-tokenizer DIR` writes the tokenizer and chat template from the same file ([model-files.md](model-files.md),
"The tokenizer"); `tests/tools/tokenizer_export_test.py DIR` checks them (`pip install --no-deps --require-hashes -r
server/requirements.txt`).

The format table it prints matches [model-files.md](model-files.md) (checked on 2026-11-03):

```
  formats
    format    tensors              bytes       GiB
    IQ4_NL         86     18,445,824,000    17.179
    IQ3_XXS        34     10,918,297,600    10.168
    IQ2_S          40     10,747,904,000    10.010
    IQ3_S          20      7,208,960,000     6.714
    Q2_0            9      2,123,366,400     1.978
    Q6_K          129      1,877,836,800     1.749
    BF16          484      1,527,889,920     1.423
    IQ4_XS         45      1,398,906,880     1.303
    Q5_K           35        346,931,200     0.323
    Q4_K           47        198,696,960     0.185
    F32           292         10,063,360     0.009
    Q8_0            1          1,740,800     0.002
    F16             1             81,920     0.000
    total       1,223     54,806,499,840    51.043
```

## Where things are

See [architecture.md](architecture.md#source-layout) for the source layout. What exists so far:

| Path | Contents |
| --- | --- |
| `src/core/` | `torch::Error`, CUDA error checks, the memory budget |
| `src/formats/types.h` | GGUF type ids and block sizes; the 13 formats the engine dequantizes |
| `src/formats/dequant.h` | Every block layout, and the CPU reference dequantization (`dequant8`, `dequant_row`) |
| `src/formats/codebooks.h` | The IQ formats' codebooks, with their MIT license notice (see `NOTICE`) |
| `src/gguf/` | The GGUF reader |
| `src/kernels/` | CUDA kernels, one folder per operation: `gemv`, `norm`, `hc`, `deltanet`, `attention`, `moe`, `ple` |
| `src/model/` | The model's configuration (`config`), its weights in engine memory (`weights`), the forward pass (`model`) |
| `src/runtime/` | Direct reads and the read pool, the expert store (residency and the miss path), the PLE row reader, the generation loop (`generator.h`) |
| `src/api/`, `include/torch_engine.h` | The C API: `libtorch_engine.so` exports only the `torch_engine_*` functions; `tests/api/torch_engine_test.c` runs it on the model (sessions, seeds, cancel from another thread) |
| `tools/` | The `torch` command: `torch inspect`, `torch run`, `torch --version`; `tools/check_wording.py`; `tools/make_draft.py` (the draft-layer file) |
| `bench/` | Microbenchmarks: GPU memory bandwidth by memory type, SSD reads by queue depth; `draft_acceptance.py` (the draft layer's acceptance per depth on the workloads, M4) |
| `tests/` | Unit tests, kernel parity tests, model parity, the fuzz target, the fixtures and the script that writes them |

When adding code:

- A new **quantization format** goes in `src/formats/` (block layout, CPU reference) with a byte-exact fixture test.
- A new **kernel** goes in `src/kernels/<operation>/` with a parity test in `tests/parity/` and a microbenchmark in
  `bench/kernels/` that reports achieved GB/s.
- A **decode change** is measured with `bench/workloads/` on all three workloads.

## Profiling

Nsight Systems does not run reliably on every JetPack 7 build. The engine has its own timers: set
`TORCH_PROFILE=1` to print a per-stage time table for each verify-window width.
