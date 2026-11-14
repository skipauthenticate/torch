"""torch_server against tests/server/stub_engine.c (the C API without a model) and the real exported tokenizer.

    TORCH_ENGINE_LIBRARY=build/tests/libtorch_engine_stub.so TORCH_TOKENIZER_DIR=DIR \
        python -m unittest discover -s tests/server -v

DIR comes from `torch inspect --header-only --export-tokenizer DIR` on the first 11 MiB of shard 1 (CI job server).
"""
import ctypes
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
from torch_server import engine as E  # noqa: E402
from torch_server.chat import END_THINK, Chat, Output, common_prefix  # noqa: E402
from torch_server.http import Server  # noqa: E402

LIBRARY = os.environ.get("TORCH_ENGINE_LIBRARY", "")
TOKENIZER = os.environ.get("TORCH_TOKENIZER_DIR", "")
IM_END = 248046
USER = [{"role": "user", "content": "What is the capital of France?"}]


def setUpModule():
    if not (Path(LIBRARY).is_file() and (Path(TOKENIZER) / "tokenizer.json").is_file()):
        message = "set TORCH_ENGINE_LIBRARY (the stub) and TORCH_TOKENIZER_DIR (an exported tokenizer)"
        if os.environ.get("CI"):
            raise RuntimeError(message)  # in CI a skip would pass silently
        raise unittest.SkipTest(message)
    global CHAT, STUB, ANSWER
    CHAT = Chat(TOKENIZER)
    STUB = ctypes.CDLL(LIBRARY)
    ANSWER = Path(tempfile.mkdtemp()) / "answer.txt"


def answer(*parts):
    """The stub's next answer: text parts are encoded, ints are ids; <|im_end|> last."""
    ids = []
    for p in parts:
        ids += CHAT.encode(p) if isinstance(p, str) else [p]
    ANSWER.write_text(",".join(map(str, ids + [IM_END])))
    return ids + [IM_END]


def last_ids():
    out = (ctypes.c_int32 * 300000)()
    return list(out[: STUB.torch_stub_last_ids(out, 300000)])


def last_params():
    p = E.Params()
    STUB.torch_stub_last_params(ctypes.byref(p))
    return p


class Running:
    def __init__(self, delay_ms=0, window=4, queue=4, api_key=None, context=4096, prefill=64):
        os.environ["TORCH_STUB_DELAY_MS"], os.environ["TORCH_STUB_WINDOW"] = str(delay_ms), str(window)
        self.engine = E.Engine(LIBRARY, ANSWER, draft="draft", context=context, prefill_columns=prefill)
        self.server = Server(("127.0.0.1", 0), CHAT, self.engine, "test-model", api_key, queue, 1000)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.worker.join(5)
        self.engine.close()

    def request(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"Content-Type": "application/json"} | (headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        c.request(method, path, data, h)
        r = c.getresponse()
        text = r.read().decode()
        c.close()
        return r.status, text

    def chat(self, body, **kw):
        status, text = self.request("POST", "/v1/chat/completions", {"messages": USER} | body, **kw)
        return status, json.loads(text) if text.startswith("{") else text


def events(text):
    """SSE data lines as JSON (and the [DONE] marker), comments apart."""
    data = [line[6:] for line in text.split("\n") if line.startswith("data: ")]
    comments = [line for line in text.split("\n") if line.startswith(":")]
    return [d if d == "[DONE]" else json.loads(d) for d in data], comments


class Basic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Running()

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_health_and_models(self):
        self.assertEqual(self.s.request("GET", "/health")[0], 200)
        status, text = self.s.request("GET", "/v1/models")
        self.assertEqual((status, json.loads(text)["data"][0]["id"]), (200, "test-model"))

    def test_answer_and_usage(self):
        ids = answer("Hello, world! 你好 👍🏽 and café.")
        status, r = self.s.chat({"reasoning_effort": "none"})
        self.assertEqual(status, 200, r)
        m = r["choices"][0]
        self.assertEqual(m["message"]["content"], "Hello, world! 你好 👍🏽 and café.")
        self.assertEqual(m["finish_reason"], "stop")
        prompt, stable = CHAT.prompt(USER, False, None)
        self.assertEqual(last_ids(), prompt)
        self.assertEqual(STUB.torch_stub_last_stable(), stable)
        self.assertEqual(r["usage"]["prompt_tokens"], len(prompt))
        self.assertEqual(r["usage"]["completion_tokens"], len(ids))
        self.assertNotIn("reasoning_content", m["message"])

    def test_race_shape_render(self):
        # reasoning_effort "none" is thinking off: the prompt ends with the empty think block
        answer("ok")
        self.s.chat({"reasoning_effort": "none", "temperature": 0})
        self.assertEqual(last_ids()[-7:], CHAT.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(last_params().sample, 0)

    def test_stream_equals_complete(self):
        answer("Streaming ", "text with ", "emoji 🎉🎉 and 中文 characters.")
        _, whole = self.s.chat({"reasoning_effort": "none", "seed": 3})
        status, text = self.s.chat({"reasoning_effort": "none", "seed": 3, "stream": True,
                                    "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        evs, _ = events(text)
        self.assertEqual(evs[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertEqual(evs[-1], "[DONE]")
        self.assertEqual(evs[-2]["choices"], [])
        self.assertEqual(evs[-2]["usage"], whole["usage"])
        self.assertEqual(evs[-3]["choices"][0]["finish_reason"], "stop")
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in evs[1:-2] if e["choices"])
        self.assertEqual(text, whole["choices"][0]["message"]["content"])

    def test_thinking(self):
        reasoning = CHAT.encode("Let me think about it.\n")
        answer(*reasoning, END_THINK, "\n\nThe answer is Paris.")
        status, r = self.s.chat({})  # thinking on by default
        self.assertEqual(status, 200, r)
        m = r["choices"][0]["message"]
        self.assertEqual((m["reasoning_content"], m["content"]), ("Let me think about it.", "The answer is Paris."))
        self.assertEqual(r["usage"]["completion_tokens_details"]["reasoning_tokens"], len(reasoning) + 1)
        self.assertTrue(CHAT.decode(last_ids()).endswith("assistant\n<think>\n"))  # the model writes the rest

    def test_stop_string_across_windows(self):
        ids = answer("one two three STOP_HERE four five six")
        status, r = self.s.chat({"reasoning_effort": "none", "stop": ["STOP_HERE", "never"]})
        self.assertEqual(status, 200, r)
        self.assertEqual(r["choices"][0]["message"]["content"], "one two three ")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        upto = next(i for i in range(len(ids)) if "STOP_HERE" in CHAT.decode(ids[: i + 1])) + 1
        self.assertEqual(r["usage"]["completion_tokens"], upto)
        self.assertTrue(STUB.torch_stub_last_stopped())
        self.assertLess(STUB.torch_stub_last_generated(), len(ids))

    def test_length(self):
        answer("a b c d e f g h i j k l m n o p")
        _, r = self.s.chat({"reasoning_effort": "none", "max_tokens": 3, "temperature": 0})
        self.assertEqual((r["choices"][0]["finish_reason"], r["usage"]["completion_tokens"]), ("length", 3))

    def test_sampling_parameters(self):
        answer("x")
        _, r = self.s.chat({"reasoning_effort": "none"})
        p = last_params()
        self.assertEqual((p.sample, p.top_k, round(p.top_p, 2), p.temperature), (1, 20, 0.95, 1.0))
        self.assertTrue(p.has_seed and r["torch"]["seed"] == p.seed)  # a new seed, returned
        _, r = self.s.chat({"reasoning_effort": "none", "seed": 42, "temperature": 0.7, "top_k": 5, "min_p": 0.1,
                            "presence_penalty": 1.5, "frequency_penalty": 0.5, "repetition_penalty": 1.1})
        p = last_params()
        self.assertEqual((p.seed, p.top_k, round(p.min_p, 2), round(p.repetition_penalty, 2)), (42, 5, 0.1, 1.1))
        self.assertEqual(r["torch"]["seed"], 42)
        self.s.chat({"reasoning_effort": "none", "temperature": 0, "presence_penalty": 1.0})
        self.assertEqual((last_params().sample, last_params().temperature), (1, 0.0))  # greedy with penalties
        self.assertEqual(list(last_params().stop_ids[:2]), [248044, 248046])

    def test_speculative_switch(self):
        answer("x")
        self.s.chat({"reasoning_effort": "none"})
        self.assertEqual(last_params().speculative, 1)
        _, r = self.s.chat({"reasoning_effort": "none", "torch": {"speculative": False, "token_ids": True}})
        self.assertEqual(last_params().speculative, 0)
        self.assertEqual(r["torch"]["token_ids"], CHAT.encode("x") + [IM_END])

    def test_session(self):
        answer("Paris.")
        _, r1 = self.s.chat({"reasoning_effort": "none"})
        history = USER + [{"role": "assistant", "content": "Paris."}, {"role": "user", "content": "And Germany?"}]
        _, r2 = self.s.chat({"reasoning_effort": "none", "messages": history})
        self.assertEqual(r2["usage"]["prompt_tokens_details"]["cached_tokens"], r1["torch"]["stable_tokens"])
        self.assertGreater(r1["torch"]["stable_tokens"], 0)

    def test_message_shapes(self):
        answer("ok")
        parts = [{"role": "developer", "content": "Be brief."},
                 {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "there"}]},
                 {"role": "assistant", "content": None}, {"role": "user", "content": "Again"}]
        status, r = self.s.chat({"messages": parts, "max_completion_tokens": 2, "user": "u", "store": False,
                                 "metadata": {}, "logit_bias": {}, "model": "anything"})
        self.assertEqual(status, 200, r)
        self.assertEqual(r["model"], "test-model")

    def test_refused(self):
        for body, param in [({"tools": [{"type": "function"}]}, "tools"), ({"n": 2}, "n"),
                            ({"response_format": {"type": "json_object"}}, "response_format"),
                            ({"logprobs": True}, "logprobs"), ({"temperature": 3}, "temperature"),
                            ({"top_p": 0}, "top_p"), ({"seed": -1}, "seed"), ({"stop": [""]}, "stop"),
                            ({"reasoning_effort": "huge"}, "reasoning_effort"),
                            ({"max_tokens": 5000}, "max_tokens"),
                            ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "messages[0].content"),
                            ({"messages": [{"role": "tool", "content": "x"}]}, "messages[0]")]:
            status, r = self.s.chat(body)
            self.assertEqual((status, r["error"]["param"]), (400, param), body)
        status, r = self.s.chat({"max_tokens": 5000})
        self.assertEqual(r["error"]["code"], "context_length_exceeded")
        self.assertEqual(self.s.request("POST", "/v1/chat/completions", {"messages": USER},
                                        {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.s.request("POST", "/v1/chat/completions", {"messages": USER},
                                        {"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.s.request("POST", "/v1/completions", {"prompt": "x"})[0], 404)
        self.assertEqual(self.s.request("POST", "/v1/chat/completions", raw=b"{not json")[0], 400)

    def test_detokenizer_equals_library(self):
        goldens = json.loads((ROOT / "tests" / "fixtures" / "tokenizer_goldens.json").read_text(encoding="utf-8"))
        for g in goldens["texts"]:
            out = Output(CHAT, False, [])
            text = "".join(out.feed(t)["content"] for t in g["ids"]) + out.finish()["content"]
            want = CHAT.tokenizer.decode(g["ids"], skip_special_tokens=True).lstrip("\n")
            if any(t in (248068, 248069) for t in g["ids"]):  # think ids switch channels; compare the rest
                continue
            self.assertEqual(text, want, g["text"][:40])


class Limits(unittest.TestCase):
    def test_api_key(self):
        s = Running(api_key="k1")
        try:
            answer("x")
            self.assertEqual(s.chat({"reasoning_effort": "none"})[0], 401)
            self.assertEqual(s.chat({"reasoning_effort": "none"}, headers={"Authorization": "Bearer k1"})[0], 200)
        finally:
            s.close()

    def test_queue_full(self):
        s = Running(delay_ms=40, queue=1)
        try:
            answer(*["word "] * 40)
            results = []
            threads = [threading.Thread(target=lambda: results.append(s.chat({"reasoning_effort": "none"})[0]))
                       for _ in range(4)]
            for t in threads:
                t.start()
                time.sleep(0.05)
            for t in threads:
                t.join()
            self.assertIn(429, results)
            self.assertIn(200, results)
        finally:
            s.close()


class Disconnects(unittest.TestCase):
    def setUp(self):
        self.s = Running(delay_ms=30, window=2, prefill=16)

    def tearDown(self):
        self.s.close()

    def _post(self, body):
        data = json.dumps(body).encode()
        sock = socket.create_connection(("127.0.0.1", self.s.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n%s" % (len(data), data))
        return sock

    def _wait_done(self, before):
        """Until the request reached the engine and the engine is idle again."""
        for _ in range(500):
            if STUB.torch_stub_requests() > before and not STUB.torch_stub_running():
                return
            time.sleep(0.01)
        self.fail("the request did not finish")

    def test_stream_disconnect_stops_within_a_window(self):
        ids = answer(*["token "] * 200)
        before = STUB.torch_stub_requests()
        sock = self._post({"messages": USER, "reasoning_effort": "none", "stream": True})
        got = b""
        while got.count(b"data: ") < 4:
            got += sock.recv(65536)
        sock.close()
        self._wait_done(before)
        self.assertTrue(STUB.torch_stub_last_cancelled() or STUB.torch_stub_last_stopped())
        self.assertLess(STUB.torch_stub_last_generated(), len(ids) // 4)

    def test_prefill_disconnect_without_streaming(self):
        long = [{"role": "user", "content": "word " * 600}]
        answer("x")
        before = STUB.torch_stub_requests()
        sock = self._post({"messages": long, "reasoning_effort": "none"})
        time.sleep(0.1)
        sock.close()
        self._wait_done(before)
        self.assertTrue(STUB.torch_stub_last_cancelled())
        self.assertEqual(STUB.torch_stub_last_generated(), 0)

    def test_prefill_keepalive_comments(self):
        long = [{"role": "user", "content": "word " * 300}]
        answer("x")
        status, text = self.s.request("POST", "/v1/chat/completions",
                                      {"messages": long, "reasoning_effort": "none", "stream": True})
        evs, comments = events(text)
        self.assertEqual(status, 200)
        self.assertTrue(comments and comments[0].startswith(": prefill "), comments[:2])
        self.assertEqual(evs[-1], "[DONE]")


if __name__ == "__main__":
    unittest.main()
