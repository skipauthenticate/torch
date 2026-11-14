<h1 align="center">Torch</h1>

<h3 align="center">A fast, exact inference engine for one model on one device.</h3>
<p align="center">Qwen3.8-Flash-Next on the NVIDIA Jetson AGX Orin 64 GB.</p>

<p align="center">
  <a href="#speed">Speed</a> · <a href="#install"><b>Install</b></a> · <a href="#use-it">Use it</a> ·
  <a href="#hardware">Hardware</a> · <a href="#how-it-works">How it works</a> · <a href="#help">Help</a>
</p>

<p align="center">
  <a href="https://github.com/skipauthenticate/torch/releases/latest"><img src="https://img.shields.io/github/v/release/skipauthenticate/torch?label=release&color=e8590c" alt="Latest release"></a>
  <a href="https://github.com/skipauthenticate/torch/actions/workflows/ci.yml"><img src="https://github.com/skipauthenticate/torch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

**Free, open source, and built for one board.** Torch runs Qwen3.8-Flash-Next with its full 262,144-token context
on a Jetson AGX Orin 64 GB and serves it over the OpenAI chat API. Every kernel and default is tuned for this one
device, and the output is exactly what the model file computes: speculative decoding changes the speed, never the
tokens.

## Speed

Answer speed on the Orin, in tokens per second. Each workload is three requests with 4,096-token prompts, a
262,144-token context and 256 generated tokens, all nine sent to one server process. Medians, greedy, measured on
2026-11-13.

| Workload | Answer speed | Reading the prompt |
| --- | ---: | ---: |
| Code review | 31.9 tokens/s | about 215 tokens/s |
| Writing | 32.5 tokens/s | about 215 tokens/s |
| Repetitive text | 46.2 tokens/s | about 215 tokens/s |

<details>
<summary><b>More numbers</b></summary>

| Setting | Code review | Writing | Repetitive text |
| --- | ---: | ---: | ---: |
| Sampled at the model's defaults (temperature 1.0, top-k 20, top-p 0.95), one process per request, 2026-11-13 | 29.5 | 28.9 | 42.1 |
| Greedy, short prompts (about 1,500 tokens) and a 2,048-token context, 2026-11-09 | 38.2 | 41.3 | 56.0 |

A long context takes memory that would otherwise hold more of the model's experts, so speed depends on the context you
ask for. A sampled request with a fixed seed gives the same tokens with speculation on or off.
[How it is measured](docs/benchmarking.md) · [All measurements](docs/performance.md)

</details>

## Install

On the Orin, as your own user (no sudo needed):

```sh
git clone https://github.com/skipauthenticate/torch.git && cd torch
scripts/install.sh
```

The installer checks your system, builds Torch into `~/.local`, downloads and verifies the model files (83.6 GB) and
builds the model's draft layer on your machine. It shows the model licenses and asks you to accept them before
anything is downloaded. A run from scratch takes about 20 minutes plus the download, and you can run it again at any
time: it resumes where it stopped. Read the script first; don't pipe it from the network into a shell.

[Install guide](docs/install.md) · [Model files and checksums](docs/model-files.md) · [Uninstall](docs/install.md#uninstalling)

## Use it

Start the server:

```sh
~/.local/bin/torch-server --context 16384
```

Then talk to it like any OpenAI-compatible server at `http://127.0.0.1:8080/v1`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
  "messages": [{"role": "user", "content": "Write a haiku about the Orin."}],
  "stream": true
}'
```

Streaming, thinking (returned as `reasoning_content`), stop strings, seeds and multi-turn conversations that don't
read the earlier turns again are all supported. The server answers one request at a time and listens only on your
machine unless you give it an API key. [Server guide](docs/server.md)

For scripts and C programs there is also `torch run` and a small C API (`include/torch_engine.h`).
[Development guide](docs/development.md)

## Hardware

| Supported hardware | Needs |
| --- | --- |
| **NVIDIA Jetson AGX Orin 64 GB** (developer kit or module) | JetPack 7 (CUDA 13), about 85 GB of free NVMe space |

That is the only device. Supporting one board is what lets every kernel, memory decision and default fit the Orin's
unified memory, small L2 cache and 16 streaming multiprocessors.

## How it works

- **The whole model, mostly in memory.** The model's experts that are used most stay resident; the rest are read from
  the NVMe in 128 KiB pieces when they are needed, and the set adapts while it runs.
- **Speculative decoding with the model's own draft layer.** It proposes up to 5 tokens and one pass checks them, so
  each pass usually yields more than one token. Checking is exact.
- **Long context.** The model's sparse attention reaches 262,144 tokens, and prompts are read in chunks of 4,096
  tokens on the Orin's tensor cores.
- **Exact by design.** Weights as stored, 8-bit activations in the dot products, F16 KV cache, F32 accumulators.
  Output does not depend on which experts are resident, and it matches the reference implementation within the
  parity gate.

[Architecture](docs/architecture.md) · [Performance model](docs/performance.md) · [Roadmap](docs/roadmap.md)

## Help

[All guides](docs) · [Report a bug](https://github.com/skipauthenticate/torch/issues/new/choose) ·
[Security](SECURITY.md) · [Contribute](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

The code is [MIT](LICENSE). The model files are not part of this project and keep their own licenses, which the
installer shows and asks you to accept ([NOTICE](NOTICE)).
