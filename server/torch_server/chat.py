"""Text side of a chat request: the chat template, token ids, and the generated ids back to text.

The template comes from the model file (`torch inspect --export-tokenizer`) and runs in jinja2's immutable sandbox with
the settings its ids depend on (tests/tools/tokenizer_export_test.py pins them). Text goes out through an id -> bytes
table and an incremental UTF-8 decoder, so a character split across tokens is never cut, and the token ids of
`<think>` and `</think>` split reasoning from the answer.
"""
import codecs
import datetime
import json
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

THINK, END_THINK = 248068, 248069
END_IDS = (248044, 248046)  # <|endoftext|>, <|im_end|>: both end a turn


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
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        self.template = env.from_string((d / "chat_template.jinja").read_text(encoding="utf-8"))
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

    def render(self, messages, enable_thinking, reasoning_effort, add_generation_prompt):
        kwargs = {"enable_thinking": enable_thinking}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
        except TemplateError as e:
            raise ChatError(f"The chat template cannot render these messages: {e}") from None

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def prompt(self, messages, enable_thinking, reasoning_effort):
        """(ids, stable): the conversation with the generation prompt, and how many of its ids a later turn will share:
        the common prefix with the same messages rendered without the generation prompt."""
        ids = self.encode(self.render(messages, enable_thinking, reasoning_effort, True))
        bare = self.encode(self.render(messages, enable_thinking, reasoning_effort, False))
        return ids, common_prefix(ids, bare)

    def decode(self, ids):
        return b"".join(self.bytes[i] if 0 <= i < len(self.bytes) else b"" for i in ids).decode("utf-8", "replace")


class Output:
    """Generated ids to text as they arrive: reasoning (between <think> and </think>) and content, stop strings found
    across tokens and windows with the text that could still start one held back. feed() takes one token at a time
    and returns the new text per channel and whether a stop string ended the answer."""

    def __init__(self, chat, thinking, stop_strings):
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

    def feed(self, token):
        """One generated id: {"reasoning": text, "content": text} to send now (possibly empty)."""
        out = {"reasoning": "", "content": ""}
        if self.stopped:
            return out
        self.tokens += 1
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
        data = self.chat.bytes[token] if 0 <= token < len(self.chat.bytes) else b""
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
        elif self.hold and len(self.pending) > self.hold:
            sent, self.pending = self.pending[: -self.hold], self.pending[-self.hold :]
        elif self.hold:
            sent = ""
        else:
            sent, self.pending = self.pending, ""
        self.text["content"] += sent
        return sent

    def finish(self):
        """The text still held back, at the end of the request."""
        out = {"reasoning": self.decoders["reasoning"].decode(b"", final=True), "content": ""}
        if not self.stopped:
            tail = self.decoders["content"].decode(b"", final=True)
            out["content"] = self.pending + (tail.lstrip("\n") if not self.started else tail)
            self.pending = ""
        self.text["reasoning"] += out["reasoning"]
        self.text["content"] += out["content"]
        return out
