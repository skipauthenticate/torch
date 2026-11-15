#!/usr/bin/env python3
"""Checks a tokenizer exported by `torch inspect --header-only --export-tokenizer DIR` (docs/model-files.md) against
goldens made with the base model's tokenizer.json:

- Hugging Face tokenizers loads DIR/tokenizer.json, and every id up to 248,319 decodes (the unused padding ids to "");
- it gives the committed ids of the benchmark prompts (bench/workloads, rebuilt as make_workloads.py builds them)
  and of tests/fixtures/tokenizer_goldens.json's texts (CJK, emoji, code, digits, whitespace runs), and decodes them
  back to the goldens' text;
- DIR/chat_template.jinja, rendered in the sandbox the server uses (`render` below), gives the goldens' ids for each
  conversation: system and developer messages, several turns, thinking on, off and unset, reasoning sent back,
  content given as text parts, with and without the generation prompt, and tools: their definitions, calls in
  history (arguments as mappings, as the server passes them), and tool results.

usage: tokenizer_export_test.py DIR
       tokenizer_export_test.py --write-goldens BASE_TOKENIZER_JSON DIR   (regenerates the goldens' ids from the base
                                                                          tokenizer; the texts and chats are below)
"""
import datetime
import hashlib
import json
import random
import sys
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests" / "fixtures" / "tokenizer_goldens.json"
sys.path.insert(0, str(ROOT / "bench" / "workloads"))
import make_workloads as W  # noqa: E402

TEXTS = [
    "",
    "Hello, world! It's a test; they'll see. I'D SAY WE'RE DONE.",
    "今天天气很好，我们去公园散步吧。东京是日本的首都。カタカナとひらがなと漢字。",
    "안녕하세요, 반갑습니다. 오늘은 2026년 11월 13일입니다.",
    "Привет, мир! Γειά σου κόσμε. مرحبا بالعالم. שלום עולם. नमस्ते दुनिया. สวัสดีครับ",
    "Emoji: 👍🏽 ok 🎉🎉🎉 family 👨‍👩‍👧‍👦 flags 🇯🇵🇫🇷 heart ❤️ math ∑∫√≤≥ money €£¥₹",
    "def fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i] * b[i];  // dot\n}\nreturn sum;\n",
    '{"id": 42, "name": "x", "tags": ["a", "b"], "ok": true, "ratio": 0.5}',
    "1234567890 3.14159 1,000,000 0x1F 1e-9 2026-11-13 12:34:56 #7 $19.99 100%",
    "a  b   c\t\td\n\n\ne    \n  \r\n x",
    "   leading spaces",
    "trailing spaces   ",
    "   \n\t \n",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa !!!!!!!!!! ---------- ==========",
    "é café ñ",
    "control \x01\x02\x07 bytes and a tab\tand DEL \x7f",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|> <tool_call>x</tool_call> <|endoftext|>",
    "Mixed: naïve façade coöperate Straße ﬁ ligature ½ ² Ⅻ",
]

U1, A1 = "What is the capital of France? Answer in one sentence.", "The capital of France is Paris."
R1, U2 = "The user asks for the capital of France. It is Paris.", "And of Germany?"
CHATS = [
    ("one user turn, thinking off", [{"role": "user", "content": U1}], {"enable_thinking": False}),
    ("system and user, thinking off",
     [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": U1}],
     {"enable_thinking": False}),
    ("developer and user, thinking off",
     [{"role": "developer", "content": "Answer briefly."}, {"role": "user", "content": U1}],
     {"enable_thinking": False}),
    ("two turns, thinking off",
     [{"role": "user", "content": U1}, {"role": "assistant", "content": A1}, {"role": "user", "content": U2}],
     {"enable_thinking": False}),
    ("one user turn, thinking on", [{"role": "user", "content": U1}], {"enable_thinking": True}),
    ("two turns, thinking on, reasoning sent back",
     [{"role": "user", "content": U1}, {"role": "assistant", "content": A1, "reasoning_content": R1},
      {"role": "user", "content": U2}], {"enable_thinking": True}),
    ("thinking unset, low effort", [{"role": "user", "content": U1}], {"reasoning_effort": "low"}),
    ("content as text parts, thinking off",
     [{"role": "user", "content": [{"type": "text", "text": "First part."}, {"type": "text", "text": "Second."}]}],
     {"enable_thinking": False}),
    ("no generation prompt, thinking off", [{"role": "user", "content": U1}],
     {"enable_thinking": False, "add_generation_prompt": False}),
]

# Tool conversations (tests/server/test_server.py sends the same ones in the API's shape and expects these ids)
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "The current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                        "units": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Finds files whose text matches a pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"},
                                                        "paths": {"type": "array", "items": {"type": "string"}},
                                                        "case_sensitive": {"type": "boolean"}},
                       "required": ["pattern"]}}},
]
U3 = "What's the weather in Paris and in Tokyo, and which notes mention them?"
CALLS = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}},
         {"type": "function", "function": {"name": "search_files", "arguments": {
             "pattern": "Paris|Tokyo", "paths": ["notes/", "travel.md"], "case_sensitive": False}}}]
RESULTS = [{"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": 14, "sky": "cloudy"}'},
           {"role": "tool", "tool_call_id": "call_2", "content": "notes/trip.md:3: Paris in May\ntravel.md:9: Tokyo"}]
CHATS += [
    ("tools, one user turn, thinking off", [{"role": "user", "content": U1}], {"enable_thinking": False,
                                                                              "tools": TOOLS}),
    ("tools and a system message, thinking on", [{"role": "system", "content": "You are a travel assistant."},
                                                 {"role": "user", "content": U3}], {"enable_thinking": True,
                                                                                    "tools": TOOLS}),
    ("tools, a call and its result, thinking off",
     [{"role": "user", "content": U3}, {"role": "assistant", "content": "", "tool_calls": CALLS[:1]}, RESULTS[0]],
     {"enable_thinking": False, "tools": TOOLS}),
    ("tools, two calls after text and reasoning, two results, thinking on",
     [{"role": "system", "content": "You are a travel assistant."}, {"role": "user", "content": U3},
      {"role": "assistant", "content": "Let me check both.", "reasoning_content": "Two lookups are needed.",
       "tool_calls": CALLS}, *RESULTS], {"enable_thinking": True, "tools": TOOLS}),
    ("tools, results then an answer and a new question, low effort",
     [{"role": "user", "content": U3}, {"role": "assistant", "content": None, "tool_calls": CALLS}, *RESULTS,
      {"role": "assistant", "content": "Paris is cloudy at 14 C; trip.md mentions it."},
      {"role": "user", "content": U2}], {"reasoning_effort": "low", "tools": TOOLS}),
    ("tools, a call and its result, no generation prompt",
     [{"role": "user", "content": U3}, {"role": "assistant", "content": "", "tool_calls": CALLS[:1]}, RESULTS[0]],
     {"enable_thinking": False, "tools": TOOLS, "add_generation_prompt": False}),
]


def raise_exception(message):
    raise TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def strftime_now(fmt):
    return datetime.datetime.now().strftime(fmt)


def renderer(template_path):
    """The chat template in the sandbox the server renders it in; the settings change token ids, so they are pinned
    here and by the goldens."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    template = env.from_string(template_path.read_text(encoding="utf-8"))

    def render(messages, add_generation_prompt=True, **kwargs):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)

    return render


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def chat_ids(tok, render):
    return [encode(tok, render(messages, **kwargs)) for _, messages, kwargs in CHATS]


def write_goldens(base, out_dir):
    tok = Tokenizer.from_file(str(base))
    render = renderer(out_dir / "chat_template.jinja")
    texts = [{"text": t, "ids": encode(tok, t), "decoded": tok.decode(encode(tok, t), skip_special_tokens=False)}
             for t in TEXTS]
    chats = [{"name": name, "ids": ids} for (name, _, _), ids in zip(CHATS, chat_ids(tok, render))]
    goldens = {"base_tokenizer_sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "texts": texts, "chats": chats}
    GOLDENS.write_text(json.dumps(goldens, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"wrote {GOLDENS}: {len(texts)} texts, {len(chats)} chats")


def check(out_dir):
    goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    render = renderer(out_dir / "chat_template.jinja")
    failures = []

    def expect(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what)

    expect(tok.get_vocab_size() == 248077 and tok.get_vocab_size(with_added_tokens=False) == 248044,
           f"vocabulary: {tok.get_vocab_size(with_added_tokens=False):,} BPE + added = {tok.get_vocab_size():,}")
    decoded = [tok.decode([i], skip_special_tokens=False) for i in range(248320)]  # raises if an id cannot decode
    expect(all(d == "" for d in decoded[248077:]) and all(decoded[i] for i in range(248077)),
           "every id decodes; the unused ids 248,077-248,319 to the empty string")

    manifest = json.loads((ROOT / "bench" / "workloads" / "manifest.json").read_text())
    for p in manifest["prompts"]:
        base = p["seed"] * 1000 + {"review": 1, "prose": 2, "repetition": 3}[p["workload"]]
        gen = W.code_text if p["workload"] == "review" else W.prose_text
        ask = {"review": W.REVIEW_ASK, "prose": W.PROSE_ASK, "repetition": W.REPEAT_ASK}[p["workload"]]
        ids = W.fit(tok, lambda n, g=gen, b=base: g(random.Random(b), n), ask)
        want = [int(t) for t in (ROOT / "bench" / "workloads" / p["file"]).read_text().split(",")]
        expect(ids == want, f"bench prompt {p['file']} ({len(want)} ids)")

    for g in goldens["texts"]:
        ids = encode(tok, g["text"])
        expect(ids == g["ids"] and tok.decode(ids, skip_special_tokens=False) == g["decoded"],
               f"text {g['text'][:32]!r} ({len(g['ids'])} ids)")
    for g, ids in zip(goldens["chats"], chat_ids(tok, render)):
        expect(ids == g["ids"], f"chat '{g['name']}' ({len(g['ids'])} ids)")
    try:
        render([{"role": "user", "content": "x"}], reasoning_effort="none")
        expect(False, "reasoning_effort 'none' reaching the template raises (the server maps it to thinking off)")
    except TemplateError:
        expect(True, "reasoning_effort 'none' reaching the template raises (the server maps it to thinking off)")
    print(f"{len(failures)} failed" if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--write-goldens":
        write_goldens(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 2:
        sys.exit(check(Path(sys.argv[1])))
    else:
        sys.exit(__doc__)
