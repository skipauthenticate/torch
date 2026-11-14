# The server

`torch_server` serves the model over the OpenAI chat API, on top of the engine's C API (`include/torch_engine.h`,
`libtorch_engine.so`). It is Python 3.12 with the standard library plus `tokenizers`, `jinja2` and `markupsafe`
(architecture decision 13), installed with hashes from `server/requirements.txt`:

```sh
python3.12 -m venv ~/.local/torch-server
~/.local/torch-server/bin/pip install --no-deps --require-hashes -r server/requirements.txt
PYTHONPATH=server ~/.local/torch-server/bin/python -m torch_server \
  --library build/libtorch_engine.so --torch-bin build/torch \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --draft draft-iq4.gguf --context 262144
```

(`scripts/install.sh` installs `torch-server`, which passes the installed paths.) It exports the tokenizer and chat
template from the model file at start (`torch inspect --export-tokenizer`, or `--tokenizer DIR` for files exported
before), loads the model and listens on `http://127.0.0.1:8080/v1`. For the pinned model files it uses the shipped
calibration in `calibration/ed59f920/` unless told otherwise: the placement for speculative decoding (or for one token
at a time without a draft file) and the draft head's 40,960-row vocabulary; `--placement none` turns the placement
off. `--swaps` (experts made resident after each window) defaults to 32 and `--prefill` to 4,096.
`python -m torch_server --help` lists every option.

## Endpoints

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | Streamed (server-sent events) or not |
| `GET /v1/models` | The one model, by the id `--model-name` gives (default: from the file name) |
| `GET /health` | `{"status": "ok", "queued": N}` |

There is no `/v1/completions`.

## Requests

| Field | |
| --- | --- |
| `messages` | Roles `system`, `developer` (merged into the system message by the template), `user` and `assistant`; content a string, a list of text parts, or null for an assistant. An assistant's `reasoning_content` goes back into the template. |
| `max_tokens`, `max_completion_tokens` | Default: `--max-tokens-default` (4,096), or what the context has left |
| `temperature` | 0 to 2, default 1.0; 0 is greedy |
| `top_p` | Above 0 to 1, default 0.95 |
| `top_k` (extension) | Default 20; 0 is off |
| `min_p` (extension) | 0 to 1, default 0 |
| `presence_penalty`, `frequency_penalty` | -2 to 2, default 0 |
| `repetition_penalty` (extension) | Above 0 to 10, default 1 |
| `seed` | 0 to 2^64 - 1. Without one, a sampled request gets a new random seed, returned as `torch.seed` |
| `stop` | A string or up to 16; matched on the answer's text, across tokens and windows |
| `stream`, `stream_options.include_usage` | |
| `reasoning_effort` | `none` turns thinking off; `minimal` and `low` give the template's low, `medium` medium, `high` and `xhigh` its xhigh |
| `chat_template_kwargs.enable_thinking` | true or false; wins over `reasoning_effort` |
| `torch.speculative` (extension) | false: decode one token at a time for this request (default: speculative when the server has a draft file; the tokens are the same either way) |
| `torch.token_ids` (extension) | true: the answer's token ids in `torch.token_ids` |

The defaults are the model file's recommended sampling (temperature 1.0, top_k 20, top_p 0.95). Thinking is on unless
a request turns it off, as in the model's template. Penalties count only the tokens this request has generated.

Ignored: `model` (any name works; answers carry the real id), `user`, `store`, `metadata`, `service_tier`,
`parallel_tool_calls`, an empty `logit_bias`. Refused with 400 and a sentence that says why: `tools`, `functions`,
`tool_choice`, `n` above 1, `response_format` other than text, `logprobs`, `top_logprobs`, a non-empty `logit_bias`,
image or audio parts, `tool` messages, and a prompt plus `max_tokens` longer than the context
(`context_length_exceeded`), all before any GPU work.

## Answers

- `message.content` is the answer and `message.reasoning_content` the thinking, split on the token ids of `<think>`
  and `</think>`, not on text. With thinking on, the generation prompt ends with `<think>`, and the model writes the
  reasoning and `</think>`.
- `finish_reason` is `stop` (an end token, `<|im_end|>` or `<|endoftext|>`, or a stop string) or `length`.
- `usage`: `prompt_tokens`, `completion_tokens` (up to the token that completed a stop string),
  `prompt_tokens_details.cached_tokens` (prompt tokens the session kept), `completion_tokens_details.reasoning_tokens`.
- `torch`: `cached_tokens`, `stable_tokens` (what the next turn can keep), `speculative`, `seed`, `prompt_seconds`
  (the time to the first token), `decode_seconds`, `prompt_tokens_per_second`, `decode_tokens_per_second`,
  `tokens_per_window`, `resident_experts`, `total_experts`.
- Streams: a first chunk with `delta: {"role": "assistant", "content": ""}`, deltas of `content` and
  `reasoning_content`, a chunk with `finish_reason` and the `torch` field, one with `choices: []` and `usage` when
  `include_usage` is set, then `data: [DONE]`. During the prompt the server sends a comment per prefill chunk
  (`: prefill 4096/131072`) so client and proxy read timeouts do not fire.
- Errors are `{"error": {"message", "type", "param", "code"}}`, also as an event once a stream has started.

Text leaves the server through a table of each token's bytes and an incremental UTF-8 decoder, so a character split
across tokens is never cut; control tokens never appear as text.

## Sessions

The engine holds one sequence. Each request sends the whole conversation; the server also sends `stable`, the
length its ids share with the same messages rendered without the generation prompt. When the next request's ids start
with those `stable` tokens, the engine goes back there instead of feeding the conversation again, and
`cached_tokens` says how many it kept. The previous answer is fed again with the new message. Any other request (a
title or summary a client sends between turns, for example) starts the sequence again, and the next turn pays for
the whole conversation: watch `cached_tokens`.

## Limits and cancelling

- One request runs at a time, on one engine thread; `--queue` more may wait (default 4). Beyond that: 429 with
  `Retry-After: 1`.
- The body is at most 16 MiB, and a client has 60 s to send its request.
- Between windows and between prefill chunks the engine thread checks the client's connection and stops when it has
  closed, streaming or not. During the prompt the check comes once per prefill chunk, so a cancel there takes effect
  at the end of the current chunk: about 19 s at the default 4,096-token chunks. A request that does not stream
  sends nothing until its answer, so prompts above about 100K tokens need streaming or a client timeout above the
  usual 10 minutes.

## Security

- The server listens on 127.0.0.1. Another `--host` needs `--api-key` (`Authorization: Bearer KEY`, compared in
  constant time). There is no TLS.
- Against web pages posting to a local server: requests must be `Content-Type: application/json` (a browser has to
  ask first, and the server answers no such question), the server sends no CORS headers, and on a loopback address
  the `Host` header must name localhost, 127.0.0.1 or ::1 (DNS rebinding).
- Logs hold the method, path and status, never prompt text.
- Message text is not isolated from the chat structure: `<|im_start|>` typed inside a message becomes the control
  token, as with the model's own renderer.

## Measured

On the Jetson AGX Orin 64 GB, 2026-11-13, at the race settings (`--context 262144 --prefill 4096`, the
`calib-g-spec.txt` placement, the draft file, `--swaps 32`), with `tests/server/board_gates.py`:

- Greedy ids through the server equal `torch run`'s on the 9 race request bodies (4,096 prompt tokens, 256 generated),
  speculative and one-token: 18 of 18.
- Seeded sampled requests (temperature 1.0, top_k 20, top_p 0.95, 128 tokens) give the same ids speculative and
  one-token: 3 prompts x 2 seeds.
- Decode speed, review prompt 1, server and `torch run` alternating in one session: 25.57 and 25.41 tok/s against
  25.61 and 25.40.
- A stream closed by its client stopped the engine 2 tokens and 72 ms later. A request closed 2 s into its 4,096-token
  prompt stopped when that prefill chunk ended (19.6 s per chunk), with no token generated.
- Turn 2 kept turn 1's stable prefix (`cached_tokens` 12 of a 19-token first prompt). The same 4,096-token request
  again answered its first token in 0.1-0.2 s instead of 19.6-19.8 s.
- The official `openai` client (3.50.0, `server/requirements-test.txt`) worked streamed and not, with thinking on and
  off.

## Tests

`tests/server/test_server.py` runs the server against `tests/server/stub_engine.c` (the C API without a model:
deterministic tokens, sessions, windows, chunks and cancel) and the real exported tokenizer; CI runs it on x86-64 and
arm64 (job `server`):

```sh
cmake --build build --target torch torch_engine_stub
build/torch inspect --header-only --export-tokenizer /tmp/tok shard1-head.gguf
TORCH_ENGINE_LIBRARY=build/tests/libtorch_engine_stub.so TORCH_TOKENIZER_DIR=/tmp/tok \
  python -m unittest discover -s tests/server -v
```
