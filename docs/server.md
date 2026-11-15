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
at a time without a draft file) and the draft head's 81,920-row vocabulary; `--placement none` turns the placement
off. `--swaps` (experts made resident after each window) defaults to 128 and `--prefill` to 4,096. A window whose
misses fill the expert staging swaps the ones it holds in at that point; `TORCH_SWAP_WHEN_FULL=0` in the environment
turns that off (and `--lend-kv` does).
`python -m torch_server --help` lists every option.

`--fast-prefill` (off by default) runs the prompt's BF16 products on the Orin's tensor cores with the activations
rounded to BF16, instead of the default exact kernels. On the nine 4,096-token benchmark prompts (2026-11-14) the time
to the first token falls from 16.9 to 13.0 s on code review and from 18.4 to 14.4 s on prose and repetition. What it
trades: the prompt is then not computed bitwise as the one-token path computes it, so a greedy answer can differ from
the default's (7 of the 9 benchmark answers did, from a near-tie onwards). It stays deterministic and gives the same
result at any `--prefill` chunk size, and it stays within the model parity gate against the reference implementation
(mean KL 0.00158 against the default's 0.00211; gate 0.0041). Decoding is unchanged. `TORCH_FAST_PREFILL=1` in the
environment turns it on where no option reaches (C API callers set `fast_prefill`, ABI 2).

## Endpoints

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | Streamed (server-sent events) or not |
| `GET /v1/models` | The one model, by the id `--model-name` gives (default: from the file name), with `context_length` (`--context`) |
| `GET /health` | `{"status": "ok", "queued": N}` |

There is no `/v1/completions`.

## Requests

| Field | |
| --- | --- |
| `messages` | Roles `system`, `developer` (merged into the system message by the template), `user`, `assistant` and `tool`; content a string, a list of text parts, or null for an assistant. An assistant's `reasoning_content` and `tool_calls` go back into the template. |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function tools; `tool_choice` `auto` (the default) or `none`. See [Tool calls](#tool-calls) |
| `max_tokens`, `max_completion_tokens` | Default: what the context has left, or `--max-tokens-default` when that is given and smaller |
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
| `reasoning.enabled`, `reasoning.effort` | Another spelling some clients send: `enabled: false` or `effort: "none"` turns thinking off, and an effort maps as `reasoning_effort`'s does; `reasoning_effort` wins when both are given |
| `chat_template_kwargs.enable_thinking` | true or false; wins over `reasoning_effort` and `reasoning` |
| `torch.speculative` (extension) | false: decode one token at a time for this request (default: speculative when the server has a draft file; the tokens are the same either way) |
| `torch.token_ids` (extension) | true: the answer's token ids in `torch.token_ids` |

The defaults are the model file's recommended sampling (temperature 1.0, top_k 20, top_p 0.95). Thinking is on unless
a request turns it off, as in the model's template. Penalties count only the tokens this request has generated.

Ignored: `model` (any name works; answers carry the real id), `user`, `store`, `metadata`, `service_tier`, an empty
`logit_bias`. Refused with 400 and a sentence that says why: the legacy `functions` and `function_call`, `tool_choice`
`required` or a named function, `n` above 1, `response_format` other than text, `logprobs`, `top_logprobs`, a
non-empty `logit_bias`, image or audio parts, and a prompt plus `max_tokens` longer than the context
(`context_length_exceeded`), all before any GPU work.

## Answers

- `message.content` is the answer and `message.reasoning_content` the thinking, split on the token ids of `<think>`
  and `</think>`, not on text. With thinking on, the generation prompt ends with `<think>`, and the model writes the
  reasoning and `</think>`.
- `finish_reason` is `stop` (an end token, `<|im_end|>` or `<|endoftext|>`, or a stop string), `length` or
  `tool_calls`.
- `usage`: `prompt_tokens`, `completion_tokens` (up to the token that completed a stop string),
  `prompt_tokens_details.cached_tokens` (prompt tokens the session kept), `completion_tokens_details.reasoning_tokens`.
- `torch`: `cached_tokens`, `stable_tokens` (what the next turn can keep), `speculative`, `seed`, `prompt_seconds`
  (the time to the first token), `decode_seconds`, `prompt_tokens_per_second`, `decode_tokens_per_second`,
  `tokens_per_window`, `resident_experts`, `total_experts`, and `expert_hits` and `expert_misses` (routed experts found
  resident or read from the SSD since the server started; a request's are the difference from the one before).
- Streams: a first chunk with `delta: {"role": "assistant", "content": ""}`, deltas of `content` and
  `reasoning_content`, a chunk with `finish_reason` and the `torch` field, one with `choices: []` and `usage` when
  `include_usage` is set, then `data: [DONE]`. During the prompt the server sends a comment per prefill chunk
  (`: prefill 4096/131072`) so client and proxy read timeouts do not fire.
- Errors are `{"error": {"message", "type", "param", "code"}}`, also as an event once a stream has started.

## Tool calls

The server passes `tools` to the model file's chat template, which lists them in the system message and asks the model
to answer a call as XML: a `<tool_call>` block holding `<function=NAME>` and one `<parameter=KEY>` block per argument.
The server reads those blocks back into OpenAI `tool_calls` and runs nothing itself: your client decides what to
execute, and should ask before anything destructive.

Requests:

- `tools`: objects with `type` `function` and `function.name` (1 to 256 characters without whitespace, `<`, `>` or
  `=`; no two alike), optional `description` and `parameters` (a JSON schema object). They reach the template as
  sent. `strict` has no effect: there is no constrained decoding, and the server does not validate arguments against
  the schema. `tools: []` is no tools.
- `tool_choice`: `auto` (the default) or `none`. With `none` the tools stay in the prompt, so it keeps its cached
  prefix, and no call is read: a call the model writes anyway comes back as text in `content`. `required` and a named
  function would need constrained decoding and are refused.
- `parallel_tool_calls: false` ends the answer at the first complete call.
- History: an assistant message's `tool_calls` with `function.arguments` as JSON text (as the API sends it), as an
  object, or empty; `role: "tool"` messages after the assistant message that made the calls, in the same order (the
  template pairs results with calls by order; `tool_call_id` is accepted and not used).

Answers:

- Content before the first call streams as usual, without the whitespace before the call. From the first
  `<tool_call>` token on the server holds the text, and at the end sends each call whole, as the API's
  `delta.tool_calls` with `index`, `id` (`call_` and 24 hex digits), `type` and `function.name` and
  `function.arguments` (a JSON object as text, `{}` without arguments), then `finish_reason: "tool_calls"`. While a
  call is held, a stream sends a comment every 5 s (`: tool call N tokens`) so read timeouts do not fire. Without
  streaming, `message.tool_calls`, with `content` null when the model wrote nothing before the calls.
- Argument types follow each parameter's schema, since the template writes strings as they are and other values as
  JSON: a value is parsed as JSON when its schema allows a type other than string and the JSON is of such a type; a
  parameter of no stated type (or one the schema does not declare) is parsed as JSON unless that gives a string;
  everything else stays the text as written.
- A value ends at the first `</parameter>` that ends the call or comes before the next parameter the tool declares,
  so a value may hold that text, or a whole call, itself.
- A call to a function the request did not define is returned as a call: the client can answer it with an error.
- Nothing is invented from text that is not a call: when a block does not parse (or none completes), the text from
  the first `<tool_call>` on comes back as `content`, as the model wrote it, with `finish_reason` `stop` (or `length`).
  A block cut off by `max_tokens` after complete calls is dropped and the complete calls are returned.
- After a call the template allows only another call: any other text ends the answer there, and is not returned.
- Stop strings match the content before the first call, never inside a call.
- A `<tool_call>` the model writes inside its reasoning stays reasoning text.

Sessions: the tools sit at the top of the system message, so they are part of the stable prefix, and the follow-up
turn with the tool results keeps the previous request's prompt (`cached_tokens`); the answer with its calls and the
results are fed again. What changes the prompt before that point costs the whole conversation again: other tools, another `reasoning_effort` or
`enable_thinking` (the template puts the reasoning instruction at the top of the system message), an edited earlier
turn, or a side request (a title or a summary) in between.

For agent clients: give the server `--context 65536` or more (some agents refuse a smaller window, and read it from
`/v1/models`). Many send no `max_tokens`; the answer, thinking included, may then use what the context has left. A
client's side requests (titles, summaries, compaction) reset the session like any other request, so the next turn
pays for the whole conversation again: route them to another model where the client allows it.

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
  token, as with the model's own renderer. Tool results are the exception, because nobody typed them (a fetched page,
  a file): a result that holds the text of any of the tokenizer's added tokens (`<|im_end|>`, `<tool_call>`,
  `</tool_response>`, ...) is encoded as plain text, so it cannot end its turn, open a system message or forge a call.
  Results without such text are encoded exactly as the template renders them.
- Tool definitions, arguments and results are prompt text: they are never logged, and the server executes nothing.

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

Tool calls, on the same board on 2026-11-14 (`--context 65536 --max-tokens-default 16384`, the draft file, the default
sampling with fixed seeds), with `tests/server/board_gates.py URL tools`:

- Three tools (weather, calculator, file search), five prompts, thinking off with two seeds and on with one: 15
  requests, and each of the 15 calls the prompts asked for parsed, two parallel calls where two cities were asked
  for, all with arguments of their schema's types (an integer, a boolean, an array of strings, an enum value). None
  came back as text, and none was written inside the reasoning. The prompt that needed no tool was answered in text.
- The same seeded request streamed and not gave the same calls.
- The follow-up turn with the tool results kept turn 1's prompt: `cached_tokens` 484 of 608 (thinking off) and 522
  of 741 (on), with 1.95 and 2.43 s to the first token.
- Requests without tools: the 9 race request bodies gave the same greedy ids through this server as through 0.2.0's,
  9 of 9, with the same library in one session.
- A smoke test with an open-source agent client from another machine through an SSH tunnel, four file tools, thinking
  on: two tasks (read a file and write one value to a new file; write a three-line file and edit one line) finished in
  4 and 5 requests (29 and 39 s), each follow-up keeping the previous request's prompt (`cached_tokens` 2,068 to
  2,743), the first prompt (2,073 tokens: the client's system prompt and tools) in 8.5 s.

## Tests

`tests/server/test_server.py` and `tests/server/test_tools.py` run the server against `tests/server/stub_engine.c`
(the C API without a model: deterministic tokens, sessions, windows, chunks and cancel) and the real exported
tokenizer; CI runs them on x86-64 and arm64 (job `server`), with the official `openai` client from
`server/requirements-test.txt`:

```sh
cmake --build build --target torch torch_engine_stub
build/torch inspect --header-only --export-tokenizer /tmp/tok shard1-head.gguf
TORCH_ENGINE_LIBRARY=build/tests/libtorch_engine_stub.so TORCH_TOKENIZER_DIR=/tmp/tok \
  python -m unittest discover -s tests/server -v
```
