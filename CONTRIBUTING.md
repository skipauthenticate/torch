# Contributing

Thank you for helping. This guide covers how to build, test and measure a change, and what a pull request needs.

## Before you start

- **Bugs and ideas**: open an issue first for anything larger than a small fix, so we can agree on the approach.
- **Performance work**: say what you expect to gain and why (which bytes or launches it removes). Every speed claim
  needs a measurement on an Orin 64 GB.
- **Scope**: the engine supports one model on one device. Changes that add other hardware or models are out of scope.

## Build

On a Jetson AGX Orin 64 GB with JetPack 7:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The CPU parts (GGUF reader, format reference code, their tests) also build on any Linux or macOS machine without
CUDA, which is what CI runs. See [docs/development.md](docs/development.md) for details.

## The rules every change follows

1. **No quality loss.** Never go below the precision policy in
   [docs/architecture.md](docs/architecture.md#numerics-and-determinism) (weights as stored, Q8_1 activations in the
   dot products, F16 KV cache, F32 elsewhere), skip experts, or change routing, sampling or attention math. A faster
   kernel must compute the same result as the reference kernel, and the parity test must say so; a multi-column kernel
   must be bitwise identical to the one-column kernel. Speculative decoding must not change greedy output.
2. **Measure it.** A change to a kernel reports its achieved GB/s before and after (`bench/` has the
   microbenchmarks). A change to decode reports tok/s on the three benchmark workloads
   ([docs/benchmarking.md](docs/benchmarking.md)) and commits the result file `torch bench` writes under
   `bench/results/`.
3. **Test it.** New code comes with tests: unit tests for host code, a parity test against the CPU reference for
   every kernel.
4. **Keep it readable.** Follow the code style below. A reviewer should understand a kernel from its header comment
   and its reference implementation.

## Code style

- C++20 and CUDA 13. Format with the repository's `.clang-format` (CI checks it).
- Names: `snake_case` for functions and variables, `PascalCase` for types, `kPascalCase` for constants.
- One idea per file. Every kernel has a header that states its inputs, layouts, the bytes it moves and its parity
  test.
- Errors: loading and setup throw `torch::Error` with a message that says what to do; the C API catches them
  and returns an error code. The decode loop does not throw.
- Comments explain *why*. Don't restate what the code says.
- No new runtime dependencies beyond the CUDA toolkit without discussion.
- Write every line yourself. The one exception is the IQ codebooks in `src/formats/codebooks.h`, which keep their MIT
  license notice; nothing else is ported, and the project does not cite other projects. State the math or the reason
  in the comment instead. CI checks the wording with `tools/check_wording.py`; if it fails on your change, reword.

## Pull requests

- One change per pull request, with a title that says what it does ("Fuse the hyper-connection mixer into one
  launch").
- The description says what changed, why, how it was tested, and the measurements (before and after).
- Commits have clear messages; we squash on merge.
- Model weights, raw benchmark outputs and other large files are never committed; `bench/results/*.json` is the
  exception.

## Conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
