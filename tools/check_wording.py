#!/usr/bin/env python3
"""Fails when a tracked text file contains a word from a list the project does not use.

The list is stored as SHA-256 digests of lower-case words (or two-word phrases), so the words themselves are
not in the repository. `--hash WORD` prints the digest to add. GGUF metadata keys (`tokenizer.ggml.*`) are the
file format's own names and are skipped, and so is the copyright line that the MIT license of the lookup tables in
src/formats/codebooks.h requires. `--stdin` checks text read from standard input instead (a commit message or a log).
"""
import hashlib, re, subprocess, sys
DIGESTS = {
    "82ba7e1f6f4171d3648bb4f785ca31992efe32df6e024acb5bccd297feb2f263",
    "fc5a1047f5919892fcdf8aa79ea5d6bb6531b5c176939ef0110906cb225941c1",
    "1b4438300add4533ab49153689e43805c78bea7a667b3bbf88241e732922a49c",
    "20dacd925aeceadc9a0aa77d7869bd4904efd399b3571b5c464d13191adadccb",
    "a0bd261e3e648c372d2a4f89fa1da226db5a16923127394b027b950eb7dc5cba",
    "f0eb19b4423b083af77f3212d8ca94997d1315501c1d3b70dc1ed779d93a8961",
    "4f5b224e8a003c6f75f990197c2d5a8f23e9c1dc073db7fcc9771f588deb3b5d",
}
ALLOW = re.compile(r"tokenizer\.ggml\.[a-z_]+")
LICENSE_LINE = ("src/formats/codebooks.h", re.compile(r"^// Copyright \(c\) "))
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def tokens(text):
    words = re.findall(r"[a-z0-9]+", ALLOW.sub(" ", text.lower()))
    return set(words) | {a + " " + b for a, b in zip(words, words[1:])}
def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--hash":
        print(digest(sys.argv[2].lower())); return
    if len(sys.argv) == 2 and sys.argv[1] == "--stdin":
        bad = sum(1 for line in sys.stdin if any(digest(t) in DIGESTS for t in tokens(line)))
        print(f"{bad} lines" if bad else "ok"); sys.exit(1 if bad else 0)
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
    bad = 0
    for name in filter(None, files):
        if name == "tools/check_wording.py" or name.endswith((".gguf", ".bin")): continue
        try: text = open(f"{root}/{name}", encoding="utf-8").read()
        except UnicodeDecodeError: continue
        for i, line in enumerate(text.splitlines(), 1):
            if name == LICENSE_LINE[0] and LICENSE_LINE[1].match(line): continue
            hits = [t for t in tokens(line) if digest(t) in DIGESTS]
            if hits: bad += 1; print(f"{name}:{i}: uses a word the project does not use")
    print(f"{bad} lines" if bad else "ok"); sys.exit(1 if bad else 0)
if __name__ == "__main__": main()
