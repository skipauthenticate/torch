<h1 align="center">Torch</h1>

<h3 align="center">A fast, exact local LLM engine for the NVIDIA Jetson.</h3>
<p align="center">Qwen3.8-Flash-Next on the Jetson AGX Orin 64 GB, offline, with an OpenAI-compatible API.</p>

<p align="center">
  <a href="#speed">Speed</a> · <a href="#install"><b>Install</b></a> · <a href="#use-it">Use it</a> ·
  <a href="#hardware">Hardware</a> · <a href="#how-it-works">How it works</a> · <a href="#help">Help</a>
</p>

<p align="center">
  <a href="https://github.com/skipauthenticate/torch/releases/latest"><img src="https://img.shields.io/github/v/release/skipauthenticate/torch?label=release&color=e8590c" alt="Latest release"></a>
  <a href="https://github.com/skipauthenticate/torch/actions/workflows/ci.yml"><img src="https://github.com/skipauthenticate/torch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

**Free, open source, and built for one board.** Torch runs Qwen3.8-Flash-Next, a large mixture-of-experts LLM,
locally on an NVIDIA Jetson AGX Orin 64 GB with its full 262,144-token context. It serves the model over an
OpenAI-compatible chat API, so your apps, agents and coding tools get a private, offline LLM at the edge. Every CUDA
kernel and default is tuned for this one device, and the output is exactly what the GGUF model file computes:
speculative decoding changes the speed, never the tokens.

## Speed

**v0.4.0 is the current release. The v0.5.0 draft is on hold.**
On 2026-11-19, one Jetson AGX Orin 64 GB ran v0.4.0 and an experimental build with a 2 GiB prompt cache.
The build checked up to 256 positions at once when an answer copied an earlier passage.
Prompt processing (PP) measures input tokens per second. Generation measures output tokens per second.
Each value is the median of three fresh requests. Each request had 4,096 input tokens and 256 output tokens.
The server used the full 262,144-token context and reported zero cached input tokens for each timed request.

| Model file | Workload | v0.4.0 PP | Trial PP | v0.4.0 generation | Trial generation |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ3_S | Code review | 285.28 | 299.96 | 36.18 | 33.84 |
| IQ3_S | Prose | 268.66 | 281.41 | 32.50 | 32.19 |
| IQ3_S | Repetition | 269.45 | 282.20 | 49.43 | 99.07 |
| UD-IQ3_XXS | Code review | 286.86 | 289.70 | 33.53 | 32.88 |
| UD-IQ3_XXS | Prose | 270.46 | 271.59 | 31.39 | 31.91 |
| UD-IQ3_XXS | Repetition | 270.60 | 272.16 | 46.78 | 103.69 |

On repetition, generation speed increased by 100% on IQ3_S and 122% on UD-IQ3_XXS.
All 4,608 benchmark output token IDs matched v0.4.0. Large GPU operations on both model files still miss
the release bandwidth requirement.

With the extra cache off, a second IQ3_S test measured 35.52 generated tokens/s for code review, 32.45 for prose,
and 105.56 for repetition. The paired v0.4.0 rates were 34.11, 32.22, and 49.15. During repetition, the median
longest interval between output groups rose from 0.14 to 2.13 seconds. The draft now uses short checks by default.
[Rates, output checks, and GPU measurements](docs/performance.md#fresh-prompt-and-generation-tests)

In an earlier cache test, a new conversation reused a 2,073-token system prompt and tool definitions. Time to first visible output
fell from 8.73 to 2.28 seconds. With 10,000 shared tokens, it fell from 38.35 to 2.07 seconds. The first conversation
with the 2,073-token prompt took 1.64 seconds longer to start with the cache. Disk caching requires an explicit
directory. [Paired measurements](docs/performance.md#saved-prompts)

Earlier v0.4.0 tests measured both model files with the same context and request lengths.

| Model file | Generation | PP | PP with `--fast-prefill` |
| --- | ---: | ---: | ---: |
| IQ3_S (default) | 33.7–53.0 tokens/s | 273–289 tokens/s | 359–390 tokens/s |
| UD-IQ3_XXS | 33.9–49.0 tokens/s | 272–288 tokens/s | 283–299 tokens/s |

In those tests, fresh 4,096-token prompts took 14.2–15.1 seconds in the default mode. With `--fast-prefill`,
they took 10.5–11.4 seconds on IQ3_S and 13.7–14.5 seconds on UD-IQ3_XXS.
[Benchmark method](docs/benchmarking.md) · [All measurements](docs/performance.md)

### Why `--fast-prefill` is off by default

`--fast-prefill` reads the prompt about a quarter faster by running its BF16 matrix products on the GPU's tensor
cores. To do that it rounds the prompt's activations to BF16 first, and that rounding has costs:

- **Answers can change.** The numbers inside the model come out slightly different. In the earlier v0.4.0
  benchmark, 7 of 9 answers differed from the default's at some token, often one where two choices were nearly tied.
- **No exact match with the default path.** By default, reading a prompt in chunks gives bit for bit the same result
  as feeding it one token at a time, at the model file's own precision. With `--fast-prefill` it does not. Its
  results are still the same from run to run and at any chunk size.
- **Long prompts.** On IQ3_S, fast-mode cache restores matched fresh fast-mode runs through 70,000 tokens.
  Those checks used greedy decoding. Sampled answers in fast mode remain unverified.

On the IQ3_S file, quality stays inside the project's parity limit against the reference implementation either way
(mean KL 0.00179 with the flag and 0.00143 without, against a limit of 0.00328). Answer speed is the same
with or without the flag.

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

For the UD-IQ3_XXS file instead of the default, run `scripts/install.sh --quant UD-IQ3_XXS`
([model files](docs/model-files.md)).

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
supported. Conversations can share a system prompt and resume after a title request. The server answers one request at a time and listens only on your
machine unless you give it an API key. [Server guide](docs/server.md)

`--fast-prefill` cuts the wait for the first token of a 4,096-token prompt by about a quarter (14.2 -> 10.6 s on code
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
