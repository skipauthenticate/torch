"""python -m torch_server: load the model and serve the OpenAI chat API on 127.0.0.1 (docs/server.md)."""
import argparse
import ipaddress
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .chat import Chat
from .engine import Engine, EngineError
from .http import Server

# The calibration shipped for the pinned model files (calibration/ed59f920/README.md), next to the package both in the
# source tree and where scripts/install.sh puts it.
CALIBRATION = Path(__file__).resolve().parents[2] / "calibration" / "ed59f920"
PINNED_SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m torch_server", description=__doc__)
    p.add_argument("--model", required=True, help="shard 1 of the model (MODEL-00001-of-00002.gguf)")
    p.add_argument("--library", default=os.environ.get("TORCH_ENGINE_LIBRARY", "libtorch_engine.so"),
                   help="the engine's shared library (default: $TORCH_ENGINE_LIBRARY or libtorch_engine.so)")
    p.add_argument("--tokenizer", type=Path, help="a folder from `torch inspect --export-tokenizer` (default: export "
                                                  "it from --model at start with --torch-bin)")
    p.add_argument("--torch-bin", default="torch", help="the torch command, for the export at start")
    p.add_argument("--draft", help="the draft-layer file: speculative decoding (per request: torch.speculative)")
    p.add_argument("--draft-vocab", help="the draft head's token list (default: the shipped one, with 81,920 rows)")
    p.add_argument("--draft-head-rows", type=int, default=0)
    p.add_argument("--placement", help="a route-count file for the resident experts (default: the shipped one for "
                                       "the pinned model; 'none' for none)")
    p.add_argument("--context", type=int, default=16384, help="tokens the KV cache holds (1 to 262,144)")
    p.add_argument("--prefill", type=int, default=4096, help="the widest prefill chunk (1 to 4,096)")
    p.add_argument("--lend-kv", action="store_true", help="the KV cache's unused tail holds experts until needed")
    p.add_argument("--fast-prefill", action="store_true",
                   help="prompts' BF16 products on the tensor cores, the activations rounded to BF16: a shorter time to "
                        "the first token, but a greedy answer can differ from the default's")
    p.add_argument("--swaps", type=int, default=128, help="experts made resident after each generated window")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--api-key", default=os.environ.get("TORCH_SERVER_API_KEY"),
                   help="require 'Authorization: Bearer KEY' (default: $TORCH_SERVER_API_KEY)")
    p.add_argument("--model-name", help="the model id the API reports (default: from --model's file name)")
    p.add_argument("--queue", type=int, default=4, help="requests that may wait for the engine (more: HTTP 429)")
    p.add_argument("--max-tokens-default", type=int,
                   help="the most tokens a request that gives no max_tokens may generate (default: what the context "
                        "has left)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if Path(a.model).name == PINNED_SHARD1 and CALIBRATION.is_dir():  # the published configuration
        if a.placement is None:
            a.placement = str(CALIBRATION / ("calib-g-spec.txt" if a.draft else "calib-g.txt"))
        if a.draft and a.draft_vocab is None:
            a.draft_vocab, a.draft_head_rows = str(CALIBRATION / "draft-vocab.txt"), a.draft_head_rows or 81920
    if a.placement == "none":
        a.placement = None
    loopback = a.host == "localhost" or ipaddress.ip_address(a.host).is_loopback
    if not loopback and not a.api_key:
        p.error(f"--host {a.host} is reachable from other machines: give --api-key as well (there is no TLS).")
    with tempfile.TemporaryDirectory(prefix="torch-tokenizer-") as tmp:
        tokenizer = a.tokenizer
        if tokenizer is None:  # from the model file itself, so it cannot drift from it
            tokenizer = Path(tmp)
            r = subprocess.run([a.torch_bin, "inspect", "--export-tokenizer", str(tokenizer), a.model],
                               capture_output=True, text=True)
            if r.returncode != 0:
                p.error(f"exporting the tokenizer failed: {r.stderr.strip() or r.stdout.strip()}")
        chat = Chat(tokenizer)
        name = a.model_name or re.sub(r"-0+1-of-0+\d+\.gguf$|\.gguf$", "", Path(a.model).name)
        try:
            engine = Engine(a.library, a.model, a.draft, a.draft_vocab, a.draft_head_rows, a.placement, a.context,
                            a.prefill, a.lend_kv, a.swaps, a.fast_prefill)
        except (EngineError, OSError) as e:
            sys.exit(f"torch_server: {e}")
        server = Server((a.host, a.port), chat, engine, name, a.api_key, a.queue, a.max_tokens_default)
        logging.info("serving %s on http://%s:%d/v1 (context %d, speculative %s, placement %s, draft vocabulary %s)",
                     name, a.host, server.server_address[1], a.context, "on" if engine.has_draft else "off",
                     a.placement or "none", f"{a.draft_vocab} ({a.draft_head_rows} rows)" if a.draft_vocab else "all")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            engine.cancel()  # the running request stops at its next window or chunk
            server.server_close()
            server.worker.join(timeout=60)
            engine.close()


if __name__ == "__main__":
    main()
