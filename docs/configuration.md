# Configuration

## SAGE settings

[`configs/sage.yaml`](../configs/sage.yaml) maps one-to-one onto `SageConfig` in
[`sage_mas/config.py`](../sage_mas/config.py). A missing file or key falls back to the
paper default, and unknown keys are rejected.

| Key | Paper | Meaning |
|---|---|---|
| `num_agents` | 4 | N, agents per question |
| `roles` | 9 roles | role pool; each question samples `num_agents` distinct roles |
| `backbones` | `null` | per-agent model names (see below) |
| `temperature` | 0.5 | initial answers, peer reviews and revisions |
| `prefix.tau` | 0.6 | τ, fraction of the response given as prefix |
| `prefix.weight` | 0.5 | λ, weight of prefix consistency in ρ |
| `prefix.temperature` | 0.6 | prefix completions |
| `review.sample_size` | 2 | m, agents sampled from each score group |
| `rewrite.temperature` | 0.2 | prompt rewriting |
| `rewrite.max_attempts` | 3 | rewrite attempts before keeping the original prompt |
| `rewrite.max_chars`, `rewrite.min_chars` | 6000, 80 | length limits of a rewritten prompt |
| `rewrite.materiality_threshold` | 0.98 | rewrites at least this similar to the original are rejected |
| `routing.top_k` | 2 | K, maximum parents per agent and round |
| `routing.max_rounds` | 3 | T, maximum collaboration rounds |
| `vote.leader_bonus` | 0.5 | β, leaders vote with weight 1 + β |
| `answer_extractor` | `xfinder` | κ(y): `xfinder` or `regex` |
| `max_workers` | 8 | concurrent model calls within one question |

Override any key from the command line with `--set`, for example
`--set num_agents=9 --set routing.top_k=3`. In Python, pass
`load_sage_config(path, ["routing.top_k=3"])` or construct `SageConfig(...)` directly.

## Models

Each model gets a small YAML file in [`configs/models/`](../configs/models/):

```yaml
name: qwen2.5-1.5b-instruct            # served model name (vllm serve --served-model-name)
hf_repo: Qwen/Qwen2.5-1.5B-Instruct    # documentation and run manifest
hf_revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
endpoints: [http://127.0.0.1:8000/v1]  # one or more replicas, used round-robin
max_tokens: 2048
timeout: 1200
max_concurrency: 128                   # concurrent requests per endpoint
sampling: {top_p: 0.8, top_k: 20, repetition_penalty: 1.1}
api_key_env: null                      # name of an env variable holding an API key, if needed
```

Endpoints can also be set per model through an environment variable named after the
model, for example
`SAGE_ENDPOINTS_QWEN2_5_1_5B_INSTRUCT="http://host1:8000/v1 http://host2:8000/v1"`.

## Mixed-backbone teams

List one model name per agent in `backbones`, and pass one `--model` config per
distinct model:

```bash
sage-run --model configs/models/qwen2.5-1.5b-instruct.yaml \
         --model configs/models/ministral-3-3b-instruct-2512.yaml \
         --set 'backbones=[qwen2.5-1.5b-instruct, ministral-3-3b-instruct-2512-bf16, qwen2.5-1.5b-instruct, ministral-3-3b-instruct-2512-bf16]' \
         --dataset gsm8k --name mixed
```

Every call that belongs to agent i uses agent i's backbone: its answers, reviews,
prefix completions and its own prompt rewrite.

## Output format

`sage-run` writes `results/<name>/seed_<seed>/<dataset>.jsonl`, one line per question:

| Field | Content |
|---|---|
| `answer` | final response (weighted pool vote); ends with a boxed answer when one was found |
| `answer_key` | normalized answer key of `answer` |
| `donor` | index of the strategy donor |
| `trace` | roles, initial answers and scores, reviews, donor, rewrites (with adapted prompts), every round (graph, update order, answers, scores), vote |
| `usage` | number of model calls and tokens |
| `calls` | every model call: stage, agent, round, completion, token counts, prompt hashes |

Pass `--trace-prompts` to store full prompts as well.

Next to the results file:

- `<dataset>.manifest.json` records the resolved configuration, model revisions,
  seed, package version and git revision.
- `sage-eval` adds `<dataset>.eval.jsonl` with one grade per question.
