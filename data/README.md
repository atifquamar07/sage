# Evaluation data

| File | Benchmark | Questions | Source (Hugging Face) |
|---|---|---|---|
| `gsm8k.json` | GSM8K | 500 | `openai/gsm8k`, config `main`, split `test` |
| `math.json` | MATH (algebra subset) | 500 | `EleutherAI/hendrycks_math`, config `algebra`, split `test` |
| `aqua_rat.json` | AQuA-RAT | 500 | `deepmind/aqua_rat`, config `raw`, split `train` |
| `gsm_hard.json` | GSM-Hard | 500 | `reasoning-machines/gsm-hard`, split `train` |
| `mmlu.json` | MMLU | 500 | `cais/mmlu`, config `all`, split `test` |
| `gpqa_diamond.json` | GPQA-Diamond | 198 | `fingertap/GPQA-Diamond@68be7564`, split `test`. Not shipped: run `python scripts/fetch_gpqa.py` |

These are the questions evaluated in the paper. For the first five benchmarks, the
paper drew 1,000 questions at random with seed 2025 and evaluated the first 500; these
files contain those 500 rows. GPQA-Diamond is used in full. `manifest.json` records
each file's source and SHA-256, which `sage-run` checks before running.

Rows are copied unchanged from the paper runs. Besides the question (`query`, plus
`choices` for multiple choice) and the reference (`gt`, and `gt_answer` where
available), every row has frozen identity fields:

- `sample_id`
- `dataset_name` (e.g. `GSM8K_1000`, the name of the original 1,000-question sample)
- `dataset_manifest_hash`
- `dataset_row_hash`
- `source_row_index`

SAGE hashes these fields to sample each question's roles, so they must stay unchanged
for the paper's role assignments to be reproduced.

Reference fields are never shown to SAGE: `sage-run` passes only `query`, `choices` and
the identity fields.

GPQA's authors ask that its questions not be published in plain text, so only
`gpqa_diamond.manifest.json` (sample ids and row hashes) is included. The fetch script
rebuilds the rows from the pinned revision and verifies every hash.
