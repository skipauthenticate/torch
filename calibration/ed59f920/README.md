# Calibration for the pinned model files

For `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` at revision `ed59f920` and the draft layer `draft-iq4.gguf`
([model-files.md](../../docs/model-files.md)). They change speed only, never output. `torch_server` uses them by
default for these files; `torch run` takes them with `--placement` and `--draft-vocab`.

All three come from the nine held-out calibration prompts in `bench/workloads/` (`review`, `prose` and
`repetition`, seeds 101-103), never from the benchmark or race prompts.

| File | Bytes | sha256 | What it is |
| --- | ---: | --- | --- |
| `calib-g.txt` | 67,778 | `d71eae9162366e5380b59543d0c35b7bf3aa6f7722cc88b0ebd7d3b1571da920` | Route counts of the 256 greedy tokens generated from each prompt (core/placement.h): the resident experts for decoding one token at a time |
| `calib-g-spec.txt` | 69,127 | `3e1e3a9b9f648ccc149d851ca2d8786bac07e179e3030f6f661103556a62970e` | The same, decoded speculatively with the draft layer and this vocabulary: the resident experts for speculative decoding (the server's default with a draft file) |
| `draft-vocab.txt` | 234,651 | `56bd282a1e59ad958be77996f380feb57445aadf537cfb5f03f0feb7ca744723` | The draft head's 40,960 rows: the 634 distinct ids of the 2,304 tokens generated from the nine prompts, by frequency, then the other ids in ascending order |

Made on the Jetson AGX Orin: the vocabulary on 2026-11-05, the route counts on 2026-11-09. To make them again, with
`M` shard 1 and `D` the draft file:

```sh
python3 bench/draft_acceptance.py vocab --torch build/torch --model M --ids ids --out vocab-all.txt
head -n 40960 vocab-all.txt > draft-vocab.txt
python3 bench/run_workloads.py --torch build/torch --model M --calibrate calib-g.txt
python3 bench/run_workloads.py --torch build/torch --model M --calibrate calib-g-spec.txt --placement calib-g.txt \
  --spec-args "--draft D --draft-variants stream:40960 --draft-vocab draft-vocab.txt --spec 5 --spec-policy adaptive-stop"
```

With the current build (M7, 2026-11-13) these commands give byte-identical `draft-vocab.txt` and `calib-g.txt` (they
depend only on the greedy tokens). They give a near-identical `calib-g-spec.txt`, not a byte-identical one: its
counts include the drafts each window verified, which depend on the draft layer and the window policy, both changed
since 2026-11-09. The counts differ in 5,534 of the 24,576 cells (0.5% more routes in all), yet 99.9% of the 16,000
most-routed experts (the ones a 262K context leaves resident) and 99.8% of the top 330 per layer are the same. The
`calib-g-spec.txt` shipped here is the 2026-11-09 file, the one the published numbers used.
