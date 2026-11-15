"""Text side of a chat request: the chat template, token ids, and the generated ids back to text.

The template comes from the model file (`torch inspect --export-tokenizer`) and runs in jinja2's immutable sandbox with
the settings its ids depend on (tests/tools/tokenizer_export_test.py pins them). Text goes out through an id -> bytes
table and an incremental UTF-8 decoder, so a character split across tokens is never cut, and the token ids of
`<think>` and `</think>` split reasoning from the answer, as those of `<tool_call>` and `</tool_call>` find tool calls.
"""
import codecs
import datetime
import functools
import json
import re
import secrets
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

THINK, END_THINK = 248068, 248069
TOOL_CALL, END_TOOL_CALL = 248058, 248059
END_IDS = (248044, 248046)  # <|endoftext|>, <|im_end|>: both end a turn
# A call as the template asks for it and writes it back: <function=NAME>, then <parameter=KEY>\nVALUE\n</parameter> each
NAME = r"[^\s<>=]{1,256}"  # what <function=NAME> and <parameter=KEY> can carry
FUNCTION = re.compile(rf"\s*<function=({NAME})>(.*)</function>\s*", re.S)
PARAMETER = re.compile(rf"<parameter=({NAME})>")
JSON_TYPES = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "boolean": lambda v: isinstance(v, bool), "null": lambda v: v is None,
              "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list)}


class ChatError(ValueError):
    """A request the template cannot render; the message says why (HTTP 400)."""


def _raise(message):
    raise TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _byte_level_alphabet():
    """The byte-level encoding's 256 characters, byte by byte (each printable byte stands for itself)."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[chr(b)] = b
        else:
            chars[chr(256 + extra)] = b
            extra += 1
    return chars


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Chat:
    def __init__(self, tokenizer_dir):
        d = Path(tokenizer_dir)
        self.tokenizer_file = d / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        source = (d / "chat_template.jinja").read_text(encoding="utf-8")
        self.template = env.from_string(source)
        self.tool_format = "<function=" in source  # the call format the parser reads
        # id -> bytes of text: BPE tokens through the byte-level alphabet, added tokens as their UTF-8 unless special
        # (control tokens never become text), unused padding ids as nothing
        data = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        alphabet = _byte_level_alphabet()
        vocab = data["model"]["vocab"]
        self.bytes = [b""] * (max(vocab.values()) + 1)
        for token, i in vocab.items():
            self.bytes[i] = bytes(alphabet[c] for c in token)
        for t in data["added_tokens"]:
            self.bytes.extend([b""] * (t["id"] + 1 - len(self.bytes)))
            self.bytes[t["id"]] = b"" if t["special"] else t["content"].encode()
        self.added = [t["content"] for t in data["added_tokens"]]

    @functools.cached_property
    def plain(self):
        """The tokenizer without its added tokens, for tool results that hold one's text (prompt()); made on first use
        (about 0.1 GB)."""
        data = json.loads(self.tokenizer_file.read_text(encoding="utf-8"))
        return Tokenizer.from_str(json.dumps(data | {"added_tokens": []}))

    def render(self, messages, enable_thinking, reasoning_effort, add_generation_prompt, tools=None):
        # preserve_thinking stays unset: every assistant turn keeps its think block, so a conversation renders the
        # same ids up to each turn's stable point (false would re-render earlier turns once a user message comes)
        kwargs = {"enable_thinking": enable_thinking}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
        try:
            return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
        except TemplateError as e:
            raise ChatError(f"The chat template cannot render these messages: {e}") from None

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def prompt(self, messages, enable_thinking, reasoning_effort, tools=None):
        """(ids, stable): the conversation with the generation prompt, and how many of its ids a later turn will share:
        the common prefix with the same messages rendered without the generation prompt.

        A tool result holding the text of an added token (<|im_end|>, <tool_call>, ...) is encoded as plain text, so
        a fetched page cannot open a turn or forge a call; other results encode as the template renders them."""
        mark, quoted, sent = "\ue000" + secrets.token_hex(8) + "\ue000", [], []
        for m in messages:
            text = m["content"] if isinstance(m["content"], str) else "".join(p["text"] for p in m["content"])
            if m["role"] == "tool" and any(t in text for t in self.added):
                quoted.append(self.plain.encode(text.strip(), add_special_tokens=False).ids)  # the template trims it
                m = m | {"content": mark}
            sent.append(m)

        def encode(text):
            pieces = text.split(mark)
            return self.encode(pieces[0]) + [i for q, p in zip(quoted, pieces[1:]) for i in q + self.encode(p)]

        ids = encode(self.render(sent, enable_thinking, reasoning_effort, True, tools))
        bare = encode(self.render(sent, enable_thinking, reasoning_effort, False, tools))
        return ids, common_prefix(ids, bare)

    def decode(self, ids):
        return b"".join(self.bytes[i] if 0 <= i < len(self.bytes) else b"" for i in ids).decode("utf-8", "replace")


def _types(schema):
    """The JSON types a parameter's schema allows: its type (a name or a list), else its anyOf/oneOf alternatives'."""
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, (str, list)):
        return {t} if isinstance(t, str) else {x for x in t if isinstance(x, str)}
    alternatives = [a for k in ("anyOf", "oneOf") if isinstance(schema.get(k), list) for a in schema[k]]
    return set().union(*map(_types, alternatives))


def _value(raw, schema):
    """A parameter's text as a JSON value. The template writes strings as they are and other values as JSON, so: JSON
    of a type the schema allows other than string, any JSON but a string when the schema gives no type, else the
    text as written."""
    try:
        v = json.loads(raw, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        return raw
    types = _types(schema)
    if isinstance(v, str) or types and not any(JSON_TYPES[t](v) for t in types if t in JSON_TYPES):
        return raw
    return v


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")


def parse_call(text, tools):
    """One block's text (between <tool_call> and </tool_call>) to (name, arguments), or None when it is not a call in
    the template's format. A name the request did not define passes (the client answers it); its values are typed
    as values of no stated type. A value ends at the first </parameter> followed by the end of the call or by the
    next parameter the tool declares (any, for a tool that declares none), so a value may hold that text itself."""
    m = FUNCTION.fullmatch(text)
    if not m:
        return None
    properties = (tools.get(m[1]) or {}).get("properties")
    properties = properties if isinstance(properties, dict) else {}
    args, rest = {}, m[2].strip()
    while rest:
        p = PARAMETER.match(rest)
        if not p:
            return None
        for end in re.finditer("</parameter>", rest):
            after = rest[end.end() :].lstrip()
            q = PARAMETER.match(after)
            if not after or q and (not properties or q[1] in properties):
                break
        else:
            return None
        value = rest[p.end() : end.start()]
        value = value.removeprefix("\n").removesuffix("\n")  # the newlines the format puts around a value
        args[p[1]] = _value(value, properties.get(p[1]))
        rest = after
    return m[1], args


class Output:
    """Generated ids to text as they arrive: reasoning (between <think> and </think>) and content, stop strings found
    across tokens and windows with the text that could still start one held back. feed() takes one token at a time
    and returns the new text per channel and whether a stop string ended the answer.

    With tools (name -> parameters schema) the calls are parsed: from the first <tool_call> of the answer on nothing
    is sent until finish(), which sets `calls` when each complete block is a call (a block cut off after a complete
    one is dropped), and otherwise returns that text, as written, as content. Content keeps back trailing whitespace
    (a call drops it). After a call only whitespace, another call or an end token may come: anything else ends the
    answer, as does the first call without parallel calls."""

    def __init__(self, chat, thinking, stop_strings, tools=None, parallel=True):
        self.chat = chat
        self.reasoning = thinking  # thinking on: the generation prompt ends with <think>, the model writes the rest
        self.stops = [s for s in stop_strings if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.decoders = {"reasoning": codecs.getincrementaldecoder("utf-8")("replace"),
                         "content": codecs.getincrementaldecoder("utf-8")("replace")}
        self.pending = ""  # content not yet sent: could be the start of a stop string
        self.started = False  # content seen past its leading newlines
        self.text = {"reasoning": "", "content": ""}
        self.tokens = 0  # tokens that count: up to and including the one that completed a stop string
        self.reasoning_tokens = 0
        self.stopped = False
        self.tools, self.parallel = tools, parallel
        self.tail = None  # ids from the first <tool_call> on
        self.dropped = ""  # the whitespace before it
        self.block, self.depth, self.blocks = None, 0, []  # the open block's ids, <tool_call>s open in it, the texts
        self.calls = None  # [{"id", "type", "function": {"name", "arguments"}}] when the tail parsed

    def feed(self, token):
        """One generated id: {"reasoning": text, "content": text} to send now (possibly empty)."""
        out = {"reasoning": "", "content": ""}
        if self.stopped:
            return out
        self.tokens += 1
        if self.tail is not None:
            self._call_token(token)
            return out
        if token == TOOL_CALL and self.tools and not self.reasoning:
            out["content"] = self._content(self.decoders["content"].decode(b"", final=True))
            if not self.stopped:  # the content before the call cannot start a stop string any more: all of it goes
                kept = self.pending.rstrip()
                out["content"] += kept
                self.text["content"] += kept
                self.dropped, self.pending, self.tail, self.block = self.pending[len(kept) :], "", [token], []
            return out
        if token == THINK:
            self.reasoning = True
            self.reasoning_tokens += 1
            return out
        if token == END_THINK:
            out["reasoning"] = self.decoders["reasoning"].decode(b"", final=True)
            self.reasoning = False
            self.reasoning_tokens += 1
            self.text["reasoning"] += out["reasoning"]
            return out
        data = self._bytes(token)
        if self.reasoning:
            self.reasoning_tokens += 1
            out["reasoning"] = self.decoders["reasoning"].decode(data)
            self.text["reasoning"] += out["reasoning"]
            return out
        out["content"] = self._content(self.decoders["content"].decode(data))
        return out

    def _content(self, text):
        if not self.started:
            text = text.lstrip("\n")  # the newlines after </think> (or after the empty think block)
            if not text:
                return ""
            self.started = True
        self.pending += text
        cut = min((i for i in (self.pending.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            sent, self.pending, self.stopped = self.pending[:cut], "", True
        else:  # all but what could start a stop string, and with tools the whitespace a call would drop
            keep = len(self.pending) - self.hold
            if self.tools:
                keep = min(keep, len(self.pending.rstrip()))
            keep = max(keep, 0)
            sent, self.pending = self.pending[:keep], self.pending[keep:]
        self.text["content"] += sent
        return sent

    def _bytes(self, token):
        return self.chat.bytes[token] if 0 <= token < len(self.chat.bytes) else b""

    def _call_token(self, token):
        if self.block is None:  # after a call
            if token == TOOL_CALL:
                self.block, self.depth = [], 0
            elif token not in END_IDS and self._bytes(token).strip():
                self.stopped = True  # text after a call, which the template asks the model not to write
                return
        elif token == END_TOOL_CALL and not self.depth and self.chat.decode(self.block).rstrip().endswith("</function>"):
            self.blocks.append(self.chat.decode(self.block))
            self.block, self.stopped = None, not self.parallel
        else:  # a <tool_call> or </tool_call> that is part of a value
            self.depth += (token == TOOL_CALL) - (token == END_TOOL_CALL and self.depth > 0)
            self.block.append(token)
        self.tail.append(token)

    def finish(self):
        """The text still held back, at the end of the request."""
        out = {"reasoning": self.decoders["reasoning"].decode(b"", final=True), "content": ""}
        if self.tail is not None:
            out["content"] = self._calls()
        elif not self.stopped:
            tail = self.decoders["content"].decode(b"", final=True)
            out["content"] = self.pending + (tail.lstrip("\n") if not self.started else tail)
            self.pending = ""
        self.text["reasoning"] += out["reasoning"]
        self.text["content"] += out["content"]
        return out

    def _calls(self):
        """Parses the complete blocks into `calls`; returns the content that adds: nothing, or when no block is complete
        or one is not a call, the text from the first <tool_call> on as written."""
        calls = [parse_call(b, self.tools) for b in self.blocks]
        if not calls or None in calls:
            return self.dropped + self.chat.decode(self.tail)  # never a call from text that is not one
        self.calls = [{"id": "call_" + secrets.token_hex(12), "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                      for name, args in calls]
        return ""
