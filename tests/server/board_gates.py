"""torch_server's gates on the real model (M7 P4; the Orin, by hand): a client of a running server.

  board_gates.py URL race OUT.json BODIES...       greedy ids of race request bodies, speculative and one-token (the
                                                   second from the session), to compare with `torch run --quiet`
  board_gates.py URL sampled BODIES...             seeded sampled requests: speculative ids == one-token ids
  board_gates.py URL openai                        the official client, streamed and not (needs `openai`)
  board_gates.py URL session                       turn 2's cached_tokens == turn 1's stable_tokens
  board_gates.py URL disconnect BODY BODY2         close a stream mid-answer (BODY), and a request mid-prompt (BODY2)
  board_gates.py URL speed BODY                    one speculative greedy request: decode tok/s
  board_gates.py compare OUT.json LOGS...          the server's ids against `torch run --quiet` logs
"""
import http.client
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


def post(url, body, timeout=1200):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {data}")
    return data


def body_of(path, **extra):
    b = json.loads(Path(path).read_text())
    b.pop("stream", None), b.pop("stream_options", None)
    return b | extra


def race(url, out, bodies):
    results = {}
    for p in bodies:
        for mode in ("spec", "one"):
            r = post(url, body_of(p, torch={"token_ids": True, "speculative": mode == "spec"}))
            t = r["torch"]
            results[f"{Path(p).stem}-{mode}"] = {"ids": t["token_ids"], "finish": r["choices"][0]["finish_reason"],
                                                 "torch": t}
            print(f"{Path(p).stem} {mode}: {len(t['token_ids'])} ids, {r['choices'][0]['finish_reason']}, prompt "
                  f"{t['prompt_seconds']:.1f} s (cached {t['cached_tokens']}), {t['decode_tokens_per_second']} tok/s, "
                  f"{t['tokens_per_window']} per window", flush=True)
    Path(out).write_text(json.dumps(results))


def compare(out, logs):
    results, bad = json.loads(Path(out).read_text()), 0
    for log in logs:
        name = Path(log).stem.removeprefix("run-")
        line = next(x for x in Path(log).read_text().splitlines() if x.startswith("generated tokens:"))
        want = [int(t) for t in line.split(":", 1)[1].split(",")]
        for mode in ("spec", "one"):
            got = results[f"{name}-{mode}"]["ids"]
            ok = got == want
            bad += not ok
            print(f"{name} {mode}: server {len(got)} ids, torch run {len(want)}: {'equal' if ok else 'DIFFER'}")
    sys.exit(1 if bad else 0)


def sampled(url, bodies):
    bad = 0
    for p in bodies:
        for seed in (1, 2):
            ids = {}
            for spec in (True, False):
                r = post(url, body_of(p, temperature=1.0, seed=seed, max_tokens=128,
                                      torch={"token_ids": True, "speculative": spec}))
                ids[spec] = r["torch"]["token_ids"]
            ok = ids[True] == ids[False]
            bad += not ok
            print(f"{Path(p).stem} seed {seed}: {len(ids[True])} ids speculative, {len(ids[False])} one-token: "
                  f"{'equal' if ok else 'DIFFER'}", flush=True)
    sys.exit(1 if bad else 0)


def openai_client(url):
    from openai import OpenAI

    client = OpenAI(base_url=url + "/v1", api_key="unused")
    messages = [{"role": "user", "content": "Name three primary colors, comma-separated."}]
    r = client.chat.completions.create(model="any", messages=messages, temperature=0, max_tokens=64,
                                       reasoning_effort="none")
    text = r.choices[0].message.content
    print(f"openai non-stream: {text!r}, finish {r.choices[0].finish_reason}, usage {r.usage.prompt_tokens}/"
          f"{r.usage.completion_tokens}, cached {r.usage.prompt_tokens_details.cached_tokens}")
    parts, usage = [], None
    stream = client.chat.completions.create(model="any", messages=messages, temperature=0, max_tokens=64,
                                            reasoning_effort="none", stream=True,
                                            stream_options={"include_usage": True})
    for chunk in stream:
        if chunk.choices:
            parts.append(chunk.choices[0].delta.content or "")
        if chunk.usage:
            usage = chunk.usage
    print(f"openai stream: {''.join(parts)!r}, usage {usage.prompt_tokens}/{usage.completion_tokens}")
    thinking = client.chat.completions.create(model="any", messages=messages, max_tokens=256, seed=7)
    m = thinking.choices[0].message
    print(f"openai thinking on: reasoning {len(getattr(m, 'reasoning_content', '') or '')} chars, content "
          f"{(m.content or '')[:60]!r}, reasoning tokens {thinking.usage.completion_tokens_details.reasoning_tokens}")
    sys.exit(0 if "".join(parts) == text else 1)


def session(url):
    m1 = [{"role": "user", "content": "Write one sentence about the sea."}]
    r1 = post(url, {"messages": m1, "temperature": 0, "max_tokens": 64, "reasoning_effort": "none"})
    m2 = m1 + [r1["choices"][0]["message"], {"role": "user", "content": "Now one about the mountains."}]
    r2 = post(url, {"messages": m2, "temperature": 0, "max_tokens": 64, "reasoning_effort": "none"})
    s1, c2 = r1["torch"]["stable_tokens"], r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    print(f"session: turn 1 prompt {r1['usage']['prompt_tokens']}, stable {s1}; turn 2 prompt "
          f"{r2['usage']['prompt_tokens']}, cached {c2}, prompt {r2['torch']['prompt_seconds']:.2f} s: "
          f"{'equal' if c2 == s1 else 'DIFFER'}")
    sys.exit(0 if c2 == s1 else 1)


def disconnect(url, body_path, prompt_body_path):
    u = urlparse(url)
    for stream, wait, path in ((True, None, body_path), (False, 2.0, prompt_body_path)):
        b = json.dumps(body_of(path, temperature=0, max_tokens=256, stream=stream)).encode()
        s = socket.create_connection((u.hostname, u.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n%s" % (len(b), b))
        t0, got = time.time(), b""
        if stream:  # wait for a few answer chunks, then close
            while got.count(b'"content": "') < 6:
                got += s.recv(65536)
        else:  # close during the prompt
            time.sleep(wait)
        s.close()
        print(f"{'stream' if stream else 'non-stream'}: closed at {time.strftime('%H:%M:%S')}."
              f"{int(time.time() * 1000) % 1000:03d}, {time.time() - t0:.1f} s after sending; the server log says how "
              "many tokens and ms the engine ran after it saw that", flush=True)
        time.sleep(25)  # the engine finishes the chunk or window it is in


def speed(url, body_path):
    r = post(url, body_of(body_path, temperature=0, max_tokens=256, torch={"speculative": True}))
    t = r["torch"]
    print(f"server: {r['usage']['completion_tokens']} tokens, {t['decode_tokens_per_second']} tok/s, "
          f"{t['tokens_per_window']} per window, prompt {t['prompt_seconds']:.1f} s, {t['resident_experts']} of "
          f"{t['total_experts']} resident")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "compare":
        compare(a[1], a[2:])
    elif a[1] == "race":
        race(a[0], a[2], a[3:])
    else:
        {"sampled": lambda: sampled(a[0], a[2:]), "openai": lambda: openai_client(a[0]),
         "session": lambda: session(a[0]), "disconnect": lambda: disconnect(a[0], a[2], a[3]),
         "speed": lambda: speed(a[0], a[2])}[a[1]]()
