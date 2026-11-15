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

Measured on the Orin on 2026-11-15. Each workload is three requests with 4,096-token prompts, a 262,144-token context
and 256 generated tokens, all nine sent to one server process. Medians, greedy.

| Workload | Answer speed | Prompt processing | 4,096-token prompt |
| --- | ---: | ---: | ---: |
| Code review | 35.7 tokens/s | about 285 tokens/s | 14.3 s |
| Writing | 33.6 tokens/s | about 270 tokens/s | 15.1 s |
| Repetitive text | 52.3 tokens/s | about 275 tokens/s | 15.0 s |

With the opt-in `--fast-prefill`, the 4,096-token prompt takes about 10.6 s for code review and 11.5 s for the other
two; answers can then differ slightly from the default exact path.

<details>
<summary><b>More numbers</b></summary>

| Same settings | Code review | Writing | Repetitive text |
| --- | ---: | ---: | ---: |
| This release, answer speed | 35.7 | 33.6 | 52.3 |
| v0.1.0, answer speed (2026-11-13) | 31.9 | 32.5 | 46.2 |
| This release, 4,096-token prompt | 14.3 s | 15.1 s | 15.0 s |
| v0.1.0, 4,096-token prompt | about 19 s | about 19 s | about 19 s |

A long context takes memory that would otherwise hold more of the model's experts, so speed depends on the context you
ask for and on how much memory is free on the board (one request's speed moves by about 1 token/s between
loads). v0.1.0's prompts took 18.6-19.8 s on 2026-11-13. A sampled
request with a fixed seed gives the same tokens with speculation on or off.
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

Streaming, thinking (returned as `reasoning_content`), stop strings, seeds, multi-turn conversations that don't
read the earlier turns again, and tool calls for agent clients (the server runs no tool; your client does) are all
supported. The server answers one request at a time and listens only on your
machine unless you give it an API key. [Server guide](docs/server.md)

`--fast-prefill` cuts the wait for the first token of a 4,096-token prompt by about a quarter (14.5 -> 10.6 s on code
review) by rounding the prompt's activations to BF16 on the tensor cores; answers can then differ from the default
exact path. Off by default. [Server guide](docs/server.md)

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
- **Exact by design.** Weights as stored, 8-bit activations in the dot products, F16 KV cache, F32 accumulators. Output
  does not depend on which experts are resident, and it matches the reference implementation within the parity gate. The
  opt-in `--fast-prefill` gives up the bitwise match with the default path for a shorter wait for the first token.

[Architecture](docs/architecture.md) · [Performance model](docs/performance.md)

## Help

[All guides](docs) · [Report a bug](https://github.com/skipauthenticate/torch/issues/new/choose) ·
[Security](SECURITY.md) · [Contribute](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

The code is [MIT](LICENSE). The model files are not part of this project and keep their own licenses, which the
installer shows and asks you to accept ([NOTICE](NOTICE)).
