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

`spread` measures the same numbers between two reference runs with different numerics. The reference was run with the
experts of 4, 8, 32 and 40 of the 48 layers computed on the CPU (its other kernels round activations differently); the
committed file is the 4-layer run, the closest to an all-GPU computation. Measured on 2026-11-04:

| Pair | Greedy tokens agreeing | Mean KL | Largest top-5 gap |
| --- | ---: | ---: | ---: |
| Reference, 4 vs 8 CPU layers | 155 of 160 | 0.00204 | 1.13 |
| Reference, 8 vs 32 CPU layers | 155 of 160 | 0.00196 | 1.43 |
| Reference, 32 vs 40 CPU layers | 160 of 160 | 0.00029 | 0.38 |
| Torch vs the committed reference | 160 of 160 | 0.00196 | 1.51 |

The thresholds are the defaults: `--tie 0.38` and `--max-mean-kl 0.0041`, twice the largest mean KL between two
reference runs.

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
regenerates them and fails if a byte differs from the committed files. After changing the script:

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
| `src/runtime/` | Direct reads and the read pool, the expert store (residency and the miss path), the PLE row reader |
| `tools/` | The `torch` command: `torch inspect`, `torch run`, `torch --version`; `tools/check_wording.py` |
| `bench/` | Microbenchmarks: GPU memory bandwidth by memory type, SSD reads by queue depth |
| `tests/` | Unit tests, kernel parity tests, model parity, the fuzz target, the fixtures and the script that writes them |

When adding code:

- A new **quantization format** goes in `src/formats/` (block layout, CPU reference) with a byte-exact fixture test.
- A new **kernel** goes in `src/kernels/<operation>/` with a parity test in `tests/parity/` and a microbenchmark in
  `bench/kernels/` that reports achieved GB/s.
- A **decode change** is measured with `bench/workloads/` on all three workloads.

## Profiling

Nsight Systems does not run reliably on every JetPack 7 build. The engine has its own timers: set
`TORCH_PROFILE=1` to print a per-stage time table for each verify-window width.
