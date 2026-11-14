# Torch

An inference engine for one model on one device: **Qwen3.8-Flash-Next** on the **NVIDIA Jetson AGX Orin 64 GB**.

Supporting a single device lets every kernel, memory decision and default be tuned for the Orin's unified memory,
small L2 cache and 16 streaming multiprocessors, with no compromises for other hardware.

> **Status: early development (milestones M0-M7 done; v0.1.0 not yet tagged).** The engine decodes speculatively: the model's own draft layer
> proposes up to 5 tokens, one verify window of up to 8 columns checks them, and the generated tokens are the same as
> one at a time. On the Orin it runs at 38.2 tok/s on code review, 41.3 on prose and 56.0 on repetitive text (medians of three
> requests, 2026-11-09), with
> output that does not depend on which experts are resident and matches the reference within the parity gate.
> Contexts reach 262,144 tokens with the model's sparse attention above 2,048, and prompts prefill in chunks of up to
> 4,096 tokens: 231 tok/s on the 4,096-token code-review prompt (M6, 2026-11-13).
> M7 adds a C API and `torch_server`, an OpenAI-compatible server. At the race's settings (4,096-token prompts, a
> 262,144-token context, 2026-11-13) it decodes greedily at 29.9 / 30.8 / 42.1 tok/s on code review, prose and
> repetitive text, and sampled at the model file's defaults at 29.5 / 28.9 / 42.1 tok/s through `torch run`, with the same
> tokens as without speculation (medians; the board's free memory varied during the session, see the
> [performance model](docs/performance.md#m7-the-c-api-the-server-and-sampling)).
> The [roadmap](docs/roadmap.md) lists what each milestone delivers;
> the [performance model](docs/performance.md) lists what has been measured and what we expect.

## Goals

- **Fast**: the highest single-request decode speed on the Orin, measured on code review, prose and repetitive text.
- **No quality loss**: the engine computes the model exactly as the quantized file describes it, at the file's own
  precision (F16 KV cache, 8-bit activations in the dot products). Speculative decoding is verified
  exactly, so it changes speed, never the output.
- **Full context**: 262,144 tokens.
- **Readable**: small, documented modules with tests, so the engine is easy to learn from and contribute to.

## Requirements

- Jetson AGX Orin 64 GB developer kit or module
- JetPack 7 (L4T R39, CUDA 13)
- ~85 GB of free NVMe space for the model files: 51.0 GiB of weights, 26.8 GiB of n-gram table, 1.38 GiB of draft
  layer ([model files](docs/model-files.md) has exact sizes and sha256s)
- The model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf` from
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  at revision `ed59f920` (see [the model](docs/architecture.md#the-model)), plus the draft-layer file, which
  `tools/make_draft.py` builds from the original checkpoint (not yet published)

## Install

On the Orin, as your own user (no sudo): clone the repository, read [docs/install.md](docs/install.md) and
`scripts/install.sh`, then run `scripts/install.sh`. It checks the system, builds into `~/.local`, downloads and
verifies the model files, builds the draft layer locally, runs a smoke test and installs `torch-server`
([docs/server.md](docs/server.md)).

## Documentation

| Document | What it covers |
| --- | --- |
| [Architecture](docs/architecture.md) | How the engine works and why each design decision was made |
| [Performance model](docs/performance.md) | Bytes per token, the bandwidth ceiling, reference measurements, targets |
| [Roadmap](docs/roadmap.md) | Milestones, their gates and what each one delivers |
| [Benchmarking](docs/benchmarking.md) | The three workloads and what every speed claim reports |
| [Install](docs/install.md) | `scripts/install.sh`: checks, build, model files, draft layer, smoke test, uninstall |
| [Server](docs/server.md) | `torch_server`: the OpenAI chat API over the engine, its options, limits and security |
| [Model files](docs/model-files.md) | Exact file names, sizes, revision and sha256s; the draft-layer file |
| [Development](docs/development.md) | Building, testing, fuzzing, fixtures and `torch inspect` |
| [Contributing](CONTRIBUTING.md) | How to build, test, measure and send a change |

## License

The code is [MIT](LICENSE). The model files are not part of this project; they keep their own licenses, which
the installer prints and asks you to accept ([NOTICE](NOTICE), [model files](docs/model-files.md)).
