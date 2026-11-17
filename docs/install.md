# Installing

`scripts/install.sh` installs the engine, the server and the model files on a Jetson AGX Orin 64 GB. It never runs
`sudo`, can be run again at any time (each step keeps what is done; a download resumes) and prints every path it
writes.

```sh
git clone https://github.com/skipauthenticate/torch.git && cd torch
scripts/install.sh --check                    # the system checks only
scripts/install.sh                            # prints the model licenses and asks you to accept them
scripts/install.sh --quant UD-IQ3_XXS         # the other supported model file instead (model-files.md)
~/.local/bin/torch-server --context 16384     # then http://127.0.0.1:8080/v1 (docs/server.md)
```

Read the script before you run it; do not pipe it from the network into a shell.

Without `--quant`, the installer uses the model file it installed last (it writes the name to
`PREFIX/share/torch/quant`), else IQ3_S. So after `--quant UD-IQ3_XXS`, a plain `scripts/install.sh` keeps that file.

## What it does

1. **Checks** JetPack 7 (`nvidia-jetpack` 7.x), a Jetson AGX Orin (tegra234, sm_87), CUDA 13.x (`CUDACXX`, `nvcc`
   on the path or `/usr/local/cuda`), 64 GB of memory, cmake 3.24 or later, ninja, a C++20 compiler, Python 3.12 with
   `venv`, curl and sha256sum. When something is missing it prints the one `sudo apt-get install` line to run.
2. **Licenses**: prints both model repositories' licenses and links and needs `--accept-model-licenses` or a typed
   `yes` before anything is downloaded. This project's code is MIT; the model files are not part of it (NOTICE).
3. **build**: CMake Release into `PREFIX/share/torch/build` (`nice`, 4 jobs; `--jobs N`), then installs
   `PREFIX/bin/torch`, `PREFIX/lib/libtorch_engine.so` and `PREFIX/include/torch_engine.h`.
4. **server**: a virtual environment in `PREFIX/share/torch/venv` with `server/requirements.txt` (`--no-deps
   --require-hashes`), the `torch_server` package in `PREFIX/share/torch/server`, the shipped calibration in
   `PREFIX/share/torch/calibration` (the placement and draft vocabulary the server uses by default,
   [calibration/](../calibration/), one folder per model file) and the launcher `PREFIX/bin/torch-server`, which
   passes the installed library, model and draft file. The models step writes the launcher again, so it always names
   the model file installed last.
5. **models**: the shards of the model `--quant` names (IQ3_S, the default: 2 shards; UD-IQ3_XXS: 4) in `--models
   DIR` (default `PREFIX/share/torch/models`). A shard already there is verified in place; a missing one is
   downloaded from the repository and revision [model-files.md](model-files.md) gives for it with `curl --fail
   --location --continue-at -` into `NAME.part`, which a later run resumes. Before the first download, one check
   makes sure the disk has room for every missing shard (less what their `.part` files hold). A `.part` that is
   already complete (a run stopped while verifying it) is verified without asking the server again.
   Every file is checked against `scripts/models.sha256` (the source of the hashes in
   [model-files.md](model-files.md); CI checks they agree), once: `PREFIX/share/torch/verified/` records each
   checked file's hash, size and time, and the server never hashes the 83.6 GB again. `HF_TOKEN`, when set, goes to
   Hugging Face as a bearer token through a file descriptor; it is never printed or put on a command line.
   `HF_ENDPOINT`, when set, replaces `https://huggingface.co` for the shards (a mirror).
6. **draft**: builds the draft-layer file on this machine with `tools/make_draft.py` (numpy pinned with hashes in
   `scripts/draft-requirements.txt`, its own environment in `PREFIX/share/torch/draft-venv`): it reads the 31 `mtp.*`
   tensors of `Qwen/Qwen3.8-Flash-Next` by range requests (5.2 GB), checks that BF16 intermediate's sha256 before
   quantizing, writes `PREFIX/share/torch/draft/draft-iq4.gguf`, checks its sha256 and deletes the intermediate. The
   file is never uploaded. A different final sha256 (another machine's floating point) is a warning, not a failure:
   the draft layer changes speed, never output.
7. **smoke**: 16 greedy tokens of a fixed prompt must equal the ids recorded in the script for that model. It prints
   the model file it ran, and `PREFIX/share/torch/smoke.log` starts with it.

`--steps build,server,models,draft,smoke` runs some of these; `--wrap CMD` runs the build and the smoke test through
CMD (a shared machine's GPU lock).

## Disk space

About 92 GB at the peak: 83.6 GB of shards, 6.7 GB while the draft layer is built (5.2 GB of it freed after), and
about 2 GB of build files. The script checks free space for all the missing shards before the first download, and
before the draft build.

## Prerequisites

What the script used on the Orin where it was tested (JetPack 7.2.1, L4T R39.2.1, Ubuntu 24.04), from `dpkg -S` on
each tool:

| Tool | Package |
| --- | --- |
| nvcc | `cuda-nvcc-13-2` 13.2.86-1 (JetPack) |
| cmake | `cmake` 3.28.3-1build7 |
| ninja | `ninja-build` 1.11.1-2 |
| g++ | `g++-13-aarch64-linux-gnu` 13.3.0-6ubuntu2~24.04.1 |
| python3 | `python3.12-minimal` 3.12.3-1ubuntu0.17, with `python3.12-venv` (`venv`, `ensurepip`) |
| curl | `curl` 8.5.0-2ubuntu10.13 |
| sha256sum, install | `coreutils` 9.4-3ubuntu6.3 |

On a board without them: `sudo apt-get install -y cmake ninja-build g++ python3-venv curl`.

## What was tested

On the shared development Orin (JetPack 7.2.1), 2026-11-13, a clean copy of the commit run with `env -i`, an empty
home and an empty prefix:

- With `--models` pointing at the existing shards (verified in place), the first run took 19 minutes: checks, the
  build (78 s), both environments, the shards' sha256, the draft layer built from nothing (its BF16 input and the final
  file both matched `scripts/models.sha256`) and the smoke test (the recorded 16 ids). A second run took 3 s up to the
  smoke test: nothing hashed or built again.
- Shard 2 downloaded for real into an empty folder (shard 1 verified in place), interrupted at 6.1 GB and run again:
  it resumed from there, finished the 28,800,138,432 bytes in about 11 minutes in all and matched its sha256.
- Both shards downloaded for real into an empty folder: 83.6 GB in 27 minutes with the hashing (1,620 s), both
  matching `scripts/models.sha256`.
- `torch-server` from the prefix answered a chat request with speculative decoding on.

On a shared board the smoke test can find too little memory when another job has just released the GPU (the engine
says so in `smoke.log`); `--steps smoke` runs it again. Not verified on a newly flashed board.

## Uninstalling

Everything is under the prefix and the models folder:

```sh
rm -f ~/.local/bin/torch ~/.local/bin/torch-server ~/.local/lib/libtorch_engine.so ~/.local/include/torch_engine.h
rm -rf ~/.local/share/torch          # build, environments, server, draft layer, the default models folder, stamps
```

A `--models DIR` outside the prefix stays until you delete it.
