# Source-Derived Evaluation Code

Complete source-derived perception, shard-merge, and SSFRQ-5D evaluators.
Original prompts, parsing/fallback rules, label mappings, scoring, and score
schemas are retained. Only imports, credentials, filesystem defaults, and
transport resource cleanup are adapted. Predictions and annotations are not
bundled by this package.

An accepted paper has been provided, but this cleanup does **not** establish
complete official reproduction or validate paper-aligned model outputs.
Rendering, inference, training, and paper-table assembly are outside scope.
Data and licensing notices are managed separately by the repository.

## Setup (Windows PowerShell)

Run from the repository root with Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r evaluation/requirements.txt
python -m evaluation.run_main_judge --help
python -m evaluation.run_reasoning_ssfrq5d --help
python -m evaluation.merge_judge_shards --help
python -m unittest discover -s tests -v
```

Set `OPENAI_API_KEY` through your secure workflow; there is no embedded key.
Prefer the environment over `--api-key`, which can expose secrets in history.
`OPENAI_BASE_URL` selects an OpenAI-compatible endpoint (default:
`https://api.openai.com/v1`). Historical judge defaults are `qwen3.5-flash`
(perception) and `o3-mini` (reasoning). Choose a model supported by your endpoint
and declare changes to the evaluation protocol. Perception retains Qwen/gateway
`enable_thinking` and `chat_template_kwargs` hints; a stock OpenAI endpoint may
reject them. Endpoint compatibility is not validated by offline tests.

## Perception

`--csv-root` or `POINTQ_CSV_ROOT` supplies three UTF-8 CSVs:
`yes_no_questions.csv`, `what_questions.csv`, `how_questions.csv`.
Required columns: `index,question,gt`; optional `is_boundary` is `true/false`.
Default location: repository-relative `data/csv`. CSV `index` must exactly
match string `pcqa_index` values in predictions.

Labels: yes/no `A=yes,B=no`; how `A=good,B=usable,C=bad`; what `A..H=S1..S8`,
`I=NONE` (comma-separated labels). Canonical labels are also accepted.
Prediction JSON contains optional `meta` and a `results` array. Synthetic example:

```json
{"results":[{"pcqa_index":"toy-001","sample_id":"synthetic::object_1","dataset":"synthetic","perception":{"yesno":{"answer":"A"},"what":{"answer":"I"},"how":{"answer":"A"}}}]}
```

```powershell
python -m evaluation.run_main_judge --result-json predictions.json --csv-root reference-csv --judge-model YOUR_SUPPORTED_JUDGE --output outputs/perception/evaluation.json
```

Optional per-question `prompt` overrides the CSV question. Missing question
blocks or missing GT stay outside their metric denominator; empty/null answers
count as `INVALID`. Invalid judge responses, including API errors, can use the
original local answer fallback. Review `method`, invalid rates, and usage failures.
Metrics: yes/no accuracy, mean per-sample what F1 (both-empty sets score 1), and
how macro-F1 over three fixed classes, all in percent. No overall aggregate.
The optional `evaluation.input_validation.validate_perception_inputs` helper
checks malformed inputs/joins; it is not automatically invoked by the original
runner. Cache defaults to `outputs/perception/judged_<input-stem>.json`;
override with `--cache-path`. Cost summary is configurable via `--cost-summary`.

## Reasoning (SSFRQ-5D)

`--final-protocol-dir` or `POINTQ_FINAL_PROTOCOL_DIR` supplies a tree of
`final-*.json` with `sample_id`, `ai_summary.summary_text`, and
`stats.final_level` (`good/usable/bad`). Default: `data/final_protocol` under
the repository. References are **AI-aggregated descriptions**, not independent
expert explanations; do not silently substitute expert comments.

Predictions accept JSON arrays, objects containing `results/predictions/items/data`,
single records with `sample_id`, or JSONL. Reasoning fields include
`reasoning: {"answer":"..."}` and `reasoning_text`. Missing IDs can be derived
from dataset-relative `rel_path/relative_path`: `synthetic/object_1.ply` maps to
`synthetic::object_1`. Inherited loaders skip malformed JSONL/reference files;
review `build_stats`. Zero matched pairs can finish without any scores.

```powershell
python -m evaluation.run_reasoning_ssfrq5d --predictions predictions.json --final-protocol-dir reference-json --output-dir outputs/reasoning-check --dry-run
python -m evaluation.run_reasoning_ssfrq5d --predictions predictions.json --final-protocol-dir reference-json --output-dir outputs/reasoning --judge-model YOUR_SUPPORTED_JUDGE
```

`--dry-run` only writes cleaned pairs, a manifest, and a summary; no API calls
or judge scores. Actual judges score structural sufficiency, specificity,
faithfulness, reasoning coherence, and quality accuracy on 0/1/2. Total: 0..10;
normalized total: 0..100. Original numeric JSON clamping, cleaning/canonical-tail,
retry, and resume behavior are retained. Incomplete five-dimension runs are
excluded from successful aggregates and recorded in invalid-response audits.

## Merging and Limits

```powershell
python -m evaluation.merge_judge_shards --manifest-json manifest.json --shards-root outputs/shards --output-json outputs/merged/judge_output.json
```

Manifest: `selected_sample_ids,num_shards,shard_to_sample_ids` (string shard
keys), optional `shard_tags`. Each shard contains `judge_output.json` with
`per_sample`, and optional `judge_cost_summary.json`. Merge only matching
model/GT/judge/protocol runs. The merger checks IDs/completeness/duplicates and
recomputes metrics; parallel launchers and SSFRQ shard merging are not included.

Outputs default under working-directory-relative `outputs/`; API usage logs
use `outputs/logs` or `POINTQ_LOG_DIR`. Original optional price-table loading
from `benchmark/config/token_price_config.json` and a
`POINTQ_WEBAPP_DIR/token_price_config.json` overlay is retained. Without matching
rates costs are unknown; no prices are bundled. Leave `VLM_RATE_LIMIT_RPM` unset
on Windows: the inherited optional cross-process limiter requires POSIX `fcntl`.

Use fresh caches/output directories after changes to inputs, GT, endpoint, or
protocol: inherited cache checks do not fingerprint input contents/endpoints.
Logs, caches, cleaned pairs, and outputs can contain input text, local paths,
and server error messages. Keep them private and redact before publication.
Offline tests use synthetic temporary inputs and mocked transports only; no
paid API, GPU, real benchmark scores, or official results are validated.
