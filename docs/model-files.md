# Model files

The engine reads three files. Sizes are exact byte counts; "verified" means the value was read from the Hugging Face
repository's LFS metadata or from the file's GGUF header on 2026-11-03.

## The quantized model

Repository [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
revision `ed59f920` (`ed59f92082b1e93c0e96d60a8b11aab089b52f09`), folder `IQ3_S/`. Repository license: Apache-2.0
(card metadata). Base model `Qwen/Qwen3.8-Flash-Next`: license "other" on its card.

| File | Bytes | sha256 (verified) |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Shard 2 is the PLE n-gram table and is byte-identical in the repository's `IQ2_XS/` and `Q2_0/` folders (same
sha256). Together 83.6 GB (77.9 GiB) on disk.

What the headers say (verified):

| | Shard 1 | Shard 2 |
| --- | --- | --- |
| GGUF version | 3 | 3 |
| `split.no` / `split.count` / `split.tensors.count` | 0 / 2 / 1,224 | 1 / 2 / 1,224 |
| Tensors | 1,223 | 1 (`per_layer_token_embd.weight`, IQ4_NL, [160, 320,001,536]) |
| Header size | 11,024,355 bytes | 173 bytes |
| Metadata keys | 67 | 3 |
| Formats | IQ4_NL 17.18 GiB, IQ3_XXS 10.17, IQ2_S 10.01, IQ3_S 6.71, Q2_0 1.98, Q6_K 1.75, BF16 1.42, IQ4_XS 1.30, Q5_K 0.32, Q4_K 0.19, F32 0.009, Q8_0 (1 tensor), F16 (1 tensor); 51.04 GiB | IQ4_NL 26.82 GiB |

`torch inspect --header-only` reads a file that stops before its tensor data, so the first 11 MiB of shard 1
(for example, fetched with an HTTP range request) shows the full tensor table without downloading the file.

## The draft layer

The quantized model has no multi-token-prediction tensors. The engine's draft layer is a separate GGUF that this
project defines (architecture decision 12):

| Property | Value |
| --- | --- |
| File name | `draft-iq4.gguf` (what `tools/make_draft.py OUT --experts iq4` writes) |
| Tensor names | `blk.48.*` for the MTP block (attention, indexer, hyper-connections, router, routed and shared experts) and `blk.48.nextn.{eh_proj, enorm, hnorm, hc_head_norm, hc_head_down, hc_head_up}`, following the main file's naming |
| Formats | routed gate and up IQ4_XS, routed down IQ4_NL; attention, shared experts and `eh_proj` Q8_0; router, hyper-connection mixers and indexer BF16; norms F32 |
| Size | 1,480,786,592 bytes (1.38 GiB) |
| Built by | `tools/make_draft.py` from the 31 `mtp.*` tensors of `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, with this project's writer and quantizers (M4) |
| sha256 | `20a21ef3b118cc2e277938f7da64935368f483a19a36fcd93bc8f57c8ea8d164` (built on 2026-11-05) |
| Published | not yet; only once the base model's license has been checked for redistribution of derived weights |

## Disk space

About 85 GB of free NVMe space: the two shards (83.6 GB) and the draft layer (1.48 GB). Building the draft layer
also needs 5.2 GB for its BF16 intermediate, which make_draft.py keeps to resume and to quantize from.

## Verifying

```sh
sha256sum Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000*-of-00002.gguf   # compare with the table above
torch inspect Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000*-of-00002.gguf  # prints the split, formats and tensor table
```

The engine checks `split.tensors.count`, the architecture (`qwen4exp`) and every tensor's type, shape and bounds at
load, and refuses a file that does not match.
