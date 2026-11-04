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

On a machine without CUDA (a laptop, CI), the same commands build the host-only parts. There is no CUDA code yet
(milestone M2 adds it); when CMake finds a CUDA compiler, the build only enables CUDA for `sm_87`.

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

Model parity tests (from M2) need the model files; point them at the files with `TORCH_MODEL_DIR`.

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
| `src/core/error.h` | `torch::Error`, thrown by loading and setup |
| `src/formats/types.h` | GGUF type ids and block sizes; the 13 formats the engine dequantizes |
| `src/formats/dequant.h` | Every block layout, and the CPU reference dequantization (`dequant8`, `dequant_row`) |
| `src/formats/codebooks.h` | The IQ formats' codebooks, with their MIT license notice (see `NOTICE`) |
| `src/gguf/` | The GGUF reader |
| `tools/` | The `torch` command: `torch inspect`, `torch --version`; `tools/check_wording.py` |
| `tests/` | Unit tests, the fuzz target, the fixtures and the script that writes them |

When adding code:

- A new **quantization format** goes in `src/formats/` (block layout, CPU reference) with a byte-exact fixture test.
- A new **kernel** goes in `src/kernels/<operation>/` with a parity test in `tests/parity/` and a microbenchmark in
  `bench/kernels/` that reports achieved GB/s.
- A **decode change** is measured with `bench/workloads/` on all three workloads.

## Profiling

Nsight Systems does not run reliably on every JetPack 7 build. The engine has its own timers: set
`TORCH_PROFILE=1` to print a per-stage time table for each verify-window width.
