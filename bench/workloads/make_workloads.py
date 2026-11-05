#!/usr/bin/env python3
"""Writes the benchmark prompts as token ids: three workloads (docs/benchmarking.md), three benchmark prompts each
(seeds 1-3) and three held-out calibration prompts each (seeds 101-103, for the expert placement's route counts).

  review      a synthetic Python module, then a request to explain it
  prose       a synthetic passage, then a request to summarize it and write a story inspired by it
  repetition  a synthetic passage, then a request to repeat it word for word

Every prompt is one user message in the model's chat format with thinking off, trimmed so prompt plus 256 generated
tokens stay below 2,048 positions (dense attention is exact there until sparse attention lands in M6). Token ids come
from the base model's tokenizer at a pinned revision, checked by sha256; they reproduce the reference tokenization of
tests/model/reference.json. Output: bench/workloads/<workload>-<bench|calib>-<seed>.tokens (comma-separated ids)
and manifest.json.

usage: make_workloads.py [--tokenizer tokenizer.json]   (downloaded to the system temp folder when not given)
"""
import argparse
import hashlib
import json
import random
import tempfile
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_URL = ("https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
                 "de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json")
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
PROMPT_TOKENS = 1536  # with 256 generated tokens, every position stays below 2,048
OUT = Path(__file__).resolve().parent

REVIEW_ASK = ("Write a detailed explanation of the code above: what each function and class does, how they work "
              "together, and any bugs, edge cases or improvements you see.")
PROSE_ASK = "Summarize the passage above in detail, then write a short story inspired by it."
REPEAT_ASK = "Repeat the passage above word for word, from the beginning."


def chat(content):
    # One user message, thinking off, as the model's chat template renders it.
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# --- synthetic Python ------------------------------------------------------------------------------------------------

NOUNS = ["record", "order", "invoice", "sensor", "reading", "account", "packet", "shipment", "ticket", "sample",
         "batch", "event", "session", "item", "job", "metric", "user", "file", "chunk", "route"]
VERBS = ["load", "parse", "merge", "filter", "score", "group", "validate", "normalize", "summarize", "export",
         "retry", "schedule", "index", "dedupe", "rank", "split", "encode", "decode", "audit", "resolve"]
ATTRS = ["id", "name", "size", "total", "weight", "timestamp", "status", "priority", "region", "owner", "price",
         "count", "limit", "offset", "level"]


def code_function(rng):
    noun, verb = rng.choice(NOUNS), rng.choice(VERBS)
    a, b = rng.sample(ATTRS, 2)
    n = rng.randint(2, 9)
    kind = rng.randrange(6)
    name = f"{verb}_{noun}s"
    if kind == 0:
        return (f"def {name}(items, min_{a}={n}):\n"
                f'    """Return the {noun}s whose {a} is at least min_{a}, sorted by {b}."""\n'
                f"    kept = [x for x in items if x.get('{a}', 0) >= min_{a}]\n"
                f"    kept.sort(key=lambda x: x['{b}'])\n"
                f"    return kept\n")
    if kind == 1:
        return (f"def {name}(rows):\n"
                f'    """Group {noun}s by {a} and sum their {b}."""\n'
                f"    totals = {{}}\n"
                f"    for row in rows:\n"
                f"        key = row['{a}']\n"
                f"        totals[key] = totals.get(key, 0) + row['{b}']\n"
                f"    return totals\n")
    if kind == 2:
        return (f"def {name}(fn, attempts={n}, delay=0.{n}):\n"
                f'    """Call fn until it succeeds, at most `attempts` times, doubling the delay."""\n'
                f"    for i in range(attempts):\n"
                f"        try:\n"
                f"            return fn()\n"
                f"        except IOError:\n"
                f"            if i == attempts:\n"
                f"                raise\n"
                f"            time.sleep(delay)\n"
                f"            delay *= 2\n")
    if kind == 3:
        return (f"def {name}(text):\n"
                f'    """Parse lines of key=value pairs into a list of {noun} dicts."""\n'
                f"    result, current = [], {{}}\n"
                f"    for line in text.splitlines():\n"
                f"        line = line.strip()\n"
                f"        if not line:\n"
                f"            result.append(current)\n"
                f"            current = {{}}\n"
                f"            continue\n"
                f"        key, _, value = line.partition('=')\n"
                f"        current[key.strip()] = value.strip()\n"
                f"    return result\n")
    if kind == 4:
        return (f"def {name}(values, window={n}):\n"
                f'    """Moving average of the {noun} {a} over `window` points."""\n'
                f"    out = []\n"
                f"    for i in range(len(values) - window):\n"
                f"        out.append(sum(values[i:i + window]) / window)\n"
                f"    return out\n")
    return (f"def {name}(a, b):\n"
            f'    """Merge two sorted lists of {noun}s by {a}, keeping duplicates once."""\n'
            f"    i = j = 0\n"
            f"    merged = []\n"
            f"    while i < len(a) and j < len(b):\n"
            f"        if a[i]['{a}'] < b[j]['{a}']:\n"
            f"            merged.append(a[i]); i += 1\n"
            f"        elif a[i]['{a}'] > b[j]['{a}']:\n"
            f"            merged.append(b[j]); j += 1\n"
            f"        else:\n"
            f"            merged.append(a[i]); i += 1; j += 1\n"
            f"    return merged + a[i:] + b[j:]\n")


def code_class(rng):
    noun = rng.choice(NOUNS)
    a, b = rng.sample(ATTRS, 2)
    cap = rng.randint(8, 128)
    cls = noun.capitalize() + rng.choice(["Cache", "Queue", "Store", "Index", "Tracker"])
    return (f"class {cls}:\n"
            f'    """Keeps at most {cap} {noun}s, evicting the one with the lowest {a}."""\n\n'
            f"    def __init__(self, capacity={cap}):\n"
            f"        self.capacity = capacity\n"
            f"        self.items = {{}}\n\n"
            f"    def put(self, key, {noun}):\n"
            f"        if len(self.items) >= self.capacity:\n"
            f"            victim = min(self.items, key=lambda k: self.items[k]['{a}'])\n"
            f"            del self.items[victim]\n"
            f"        self.items[key] = {noun}\n\n"
            f"    def get(self, key, default=None):\n"
            f"        return self.items.get(key, default)\n\n"
            f"    def total_{b}(self):\n"
            f"        return sum(x['{b}'] for x in self.items.values())\n")


def code_text(rng, parts):
    blocks = ["import time\nimport json\n"]
    for _ in range(parts):
        blocks.append(code_class(rng) if rng.random() < 0.25 else code_function(rng))
    return "\n\n".join(blocks)


# --- synthetic prose -------------------------------------------------------------------------------------------------

PLACES = ["the harbor", "the old mill", "the market square", "the river bend", "the hill road", "the library",
          "the bakery", "the train station", "the orchard", "the lighthouse", "the bridge", "the workshop"]
PEOPLE = ["a fisherman", "the baker's daughter", "an old teacher", "two travelers", "a young engineer", "the mayor",
          "a street musician", "the night watchman", "a painter", "three children", "a carpenter", "the postmaster"]
ACTIONS = ["waited for the rain to stop", "counted the boats coming in", "argued about the price of flour",
           "repaired a broken wheel", "read the letters aloud", "watched the lamps being lit",
           "carried crates up the steep stairs", "planted rows of young trees", "sang an old song about the sea",
           "traded stories about the winter", "measured the height of the water", "painted the shutters blue"]
TIMES = ["Early in the morning", "Before noon", "Late in the afternoon", "At dusk", "Long after midnight",
         "On the first warm day of spring", "During the long autumn", "When the fog lifted"]
DETAILS = ["The air smelled of salt and wet stone.", "Nobody remembered when the clock had last been right.",
           "A dog slept in the doorway and refused to move.", "The wind carried the sound of bells from far away.",
           "Smoke rose straight up from every chimney.", "The paint was still wet on the new sign.",
           "Gulls circled above, waiting for scraps.", "Someone had left a lantern burning by the gate."]


def prose_text(rng, paragraphs):
    out = []
    for _ in range(paragraphs):
        sentences = []
        for _ in range(rng.randint(4, 7)):
            if rng.random() < 0.3:
                sentences.append(rng.choice(DETAILS))
            else:
                sentences.append(f"{rng.choice(TIMES)}, {rng.choice(PEOPLE)} near {rng.choice(PLACES)} "
                                 f"{rng.choice(ACTIONS)}.")
        out.append(" ".join(sentences))
    return "\n\n".join(out)


# --- assembly --------------------------------------------------------------------------------------------------------

def fit(tok, make_body, ask):
    """The longest body (by part count) whose chat prompt stays within PROMPT_TOKENS tokens."""
    lo, hi, best = 1, 400, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = tok.encode(chat(make_body(mid) + "\n\n" + ask), add_special_tokens=False).ids
        if len(ids) <= PROMPT_TOKENS:
            best, lo = ids, mid + 1
        else:
            hi = mid - 1
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path)
    args = p.parse_args()
    path = args.tokenizer or Path(tempfile.gettempdir()) / "torch-tokenizer.json"
    if not path.exists():
        urllib.request.urlretrieve(TOKENIZER_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit(f"{path} is not the pinned tokenizer (sha256 differs)")
    tok = Tokenizer.from_file(str(path))

    manifest = {"tokenizer": TOKENIZER_URL, "tokenizer_sha256": TOKENIZER_SHA256, "prompt_tokens_max": PROMPT_TOKENS,
                "prompts": []}
    for workload, ask, kind in [("review", REVIEW_ASK, "code"), ("prose", PROSE_ASK, "prose"),
                                ("repetition", REPEAT_ASK, "prose")]:
        for use, seeds in [("bench", [1, 2, 3]), ("calib", [101, 102, 103])]:
            for seed in seeds:
                # A different stream per workload, so prose and repetition do not share a passage.
                base = seed * 1000 + {"review": 1, "prose": 2, "repetition": 3}[workload]
                gen = code_text if kind == "code" else prose_text
                ids = fit(tok, lambda n: gen(random.Random(base), n), ask)
                name = f"{workload}-{use}-{seed}.tokens"
                (OUT / name).write_text(",".join(map(str, ids)) + "\n")
                manifest["prompts"].append({"file": name, "workload": workload, "use": use, "seed": seed,
                                            "tokens": len(ids)})
                print(f"{name}: {len(ids)} tokens")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
