# Reproducing the main results

This page reproduces the SAGE rows of Table 1 in the paper:

- **Backbones:** Qwen2.5-1.5B-Instruct and Ministral-3-3B-Instruct-2512.
- **Benchmarks:** six, evaluated with xFinder + xVerify.
- **Seeds:** 2025, 2026 and 2027.

## 1. Environments

The client (`pip install -e .`) and the vLLM servers can live in separate environments.
The paper used:

| Component | Version |
|---|---|
| Qwen2.5-1.5B-Instruct, xFinder, xVerify servers | vLLM 0.6.6.post1, torch 2.5.1, transformers 4.47.1 ([`envs/serve-qwen-paper.txt`](../envs/serve-qwen-paper.txt)) |
| Ministral-3-3B-Instruct-2512 server | vLLM 0.13.0, torch 2.9.0, mistral-common 1.11.7 ([`envs/serve-ministral-paper.txt`](../envs/serve-ministral-paper.txt)) |
| Client | Python 3.11 ([`envs/client-paper.txt`](../envs/client-paper.txt)) |
| Hardware | NVIDIA A100 80GB; one GPU holds all three servers |

Model revisions are pinned in [`configs/models/`](../configs/models/) and in
[`scripts/serve_vllm.sh`](../scripts/serve_vllm.sh):

| Model | Hugging Face revision |
|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | `b6d637bef2393152b3da2b2fde72eecdee30557e` |
| IAAR-Shanghai/xFinder-qwen1505 | `74710b225ed6b7655701d0540d868edc5466e350` |
| IAAR-Shanghai/xVerify-0.5B-I | `7ddfe002f965f9474c524c17f331263604b8c2da` |

Recent vLLM releases also work. They may shift results slightly, because generation
is not bit-identical across vLLM versions, GPUs or batch compositions.

## 2. Data

Five benchmarks ship in [`data/`](../data/): the first 500 of the paper's 1,000
sampled questions for each. GPQA-Diamond is downloaded:

```bash
pip install "sage-mas[data]"
python scripts/fetch_gpqa.py
```

## 3. Run, evaluate, aggregate

```bash
scripts/serve_vllm.sh                                    # Qwen2.5-1.5B + xFinder + xVerify on GPU 0

MODEL=configs/models/qwen2.5-1.5b-instruct.yaml
for seed in 2025 2026 2027; do
  for dataset in math gsm8k aqua_rat gsm_hard mmlu gpqa_diamond; do
    sage-run --model $MODEL --dataset $dataset --seed $seed
  done
done
sage-eval results/qwen2.5-1.5b-instruct/seed_*/*.jsonl
sage-aggregate results/qwen2.5-1.5b-instruct
```

For Ministral, serve with `scripts/serve_vllm.sh --task ministral` and run with
`--model configs/models/ministral-3-3b-instruct-2512.yaml`.

**Cost.** One run of all six benchmarks makes about 100k model calls. On one A100 it
takes a few hours per seed, and xFinder adds a similar number of short extraction
calls.

Serve with `OMP_NUM_THREADS=1`, as `serve_vllm.sh` does. vLLM samples per-request-seeded
generations on the CPU, and without the limit thread oversubscription can slow it
several times.

## Other settings

A rerun samples new generations, so expect agreement with the paper within run-to-run
variation, not identical numbers.

Other settings from the paper use the same commands with different configs:

- **Nine-agent teams:** `--set num_agents=9 --set routing.top_k=3`.
- **Other Qwen2.5 sizes:** a model config per size.
- **Mixed-backbone teams:** `backbones` in the config (see [configuration.md](configuration.md)).
