# Calibration for the UD-IQ3_XXS model file

For `Qwen3.8-Flash-Next-UD-IQ3_XXS` at revision `f5d175de` and the draft layer `draft-iq4.gguf`
([model-files.md](../../docs/model-files.md)). They change speed only, never output. `torch_server` uses them by
default for this file (it picks the folder by the name of `--model`); `torch run` takes them with `--placement` and
`--draft-vocab`.

All three come from the nine held-out calibration prompts in `bench/workloads/` (`review`, `prose` and
`repetition`, seeds 101-103), never from the benchmark or race prompts. They follow the IQ3_S file's
([../ed59f920/README.md](../ed59f920/README.md)).

| File | Bytes | sha256 | What it is |
| --- | ---: | --- | --- |
| `calib-g.txt` | 67,791 | `1cabb236bdd8d6b383ad954a1afd078a9d9979cd7952d4599074fce2a24b48b9` | Route counts of the 256 greedy tokens generated from each prompt (core/placement.h): the resident experts for decoding one token at a time |
| `calib-g-spec.txt` | 69,063 | `c87d39b1c5a36a7494a3cb685079119af91b195dbe490048d66fec8a4a6db817` | The same, decoded speculatively with the draft layer and this vocabulary's first 81,920 rows: the resident experts for speculative decoding (the server's default with a draft file) |
| `draft-vocab.txt` | 480,411 | `ec222dd2f0b3489f936c1269333c8a3a9f3eb8bce45fb90dd19e328ef6bd6c4f` | The draft head's 81,920 rows: the 653 distinct ids of the 2,304 tokens generated from the nine prompts, by frequency, then the other ids in ascending order |

Made on the Jetson AGX Orin on 2026-11-15. The routes do not depend on which experts are resident, so the runs held
19,200 experts resident (`--resident 400`, or the IQ3_S file's placement with `--resident-total 19200`), well away from
the free-memory floor. Each workload's counts went to a file of their own, and the three were summed.

The routing is close to the IQ3_S file's: 95.3% of the 16,000 most-routed experts (96.0% of the top 19,200) are the
same in both files' `calib-g.txt`. 451 of the 634 ids in the IQ3_S file's draft vocabulary are also among this one's
653.

To make them again, with `M` shard 1 and `D` the draft file (the commands of the IQ3_S file's README, per workload `W`):

```sh
for seed in 101 102 103; do   # what draft_acceptance.py vocab runs, at a fixed residency
  build/torch run M --tokens @bench/workloads/W-calib-$seed.tokens -n 256 --quiet --resident 400 |
    sed -n 's/^generated tokens: //p' > ids/W-calib-$seed.generated
done
python3 bench/draft_acceptance.py vocab --torch build/torch --model M --ids ids --out vocab-all.txt --workload W
python3 bench/run_workloads.py --torch build/torch --model M --calibrate calib-g.W.txt --workload W \
  --placement ../ed59f920/calib-g.txt --spec-args "--resident-total 19200"
# once all three workloads are done: sum the three calib-g.W.txt files into calib-g.txt, then
head -n 81920 vocab-all.txt > draft-vocab.txt
python3 bench/run_workloads.py --torch build/torch --model M --calibrate calib-g-spec.W.txt --workload W \
  --placement calib-g.txt --spec-args "--draft D --draft-variants stream:81920 --draft-vocab draft-vocab.txt --spec 5 \
  --spec-policy adaptive-stop --resident-total 19200"
# and sum the three calib-g-spec.W.txt files into calib-g-spec.txt
```
