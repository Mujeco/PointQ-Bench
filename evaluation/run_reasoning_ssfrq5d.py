#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PointQ-Bench SSFRQ-5D reasoning evaluator.

This script evaluates model-generated reasoning text against
`final_protocol/*/final-*.json::ai_summary.summary_text` using an LLM judge
(default: `o3-mini`) under the SSFRQ-5D protocol:

  S1: Structural Sufficiency  (0/1/2)
  S2: Specificity             (0/1/2)
  F : Faithfulness            (0/1/2)
  R : Reasoning Coherence     (0/1/2)
  Q : Quality Accuracy        (0/1/2)

Main workflow:
1) Load model output JSON and extract reasoning answers.
2) Clean reasoning text with deterministic rules.
3) Match sample_id to final_protocol GT summary_text.
4) Ask judge model for SSFRQ-5D scores.
5) Save per-sample and aggregate metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.utils.api_client import VLMClient, append_usage_log, build_usage_record, describe_api_error
from benchmark.utils.pointq_paths import final_protocol_dir as configured_final_protocol_dir

DEFAULT_FINAL_PROTOCOL_DIR = configured_final_protocol_dir()
DEFAULT_API_KEY = ""
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = "o3-mini"
DEFAULT_OUTPUT_ROOT = Path("outputs/ssfrq5d")
CACHE_SCHEMA_VERSION = 1
DIMENSION_PARSE_RETRIES = 0
INCOMPLETE_PAIR_REPAIR_ROUNDS = 0
RUN_MANIFEST_NAME = "ssfrq5d_run_manifest.json"
INVALID_RESPONSES_NAME = "ssfrq5d_invalid_responses.jsonl"
DEFAULT_ADAPTIVE_COMPLETION_RETRY_BUDGETS = [512, 1024]
DEFAULT_TRUNCATE_RETRY_MAX_ATTEMPTS = 2
TRUNCATION_RESPONSE_PREFIXES = {"", "score", "score:"}

DIMENSIONS = ["s1", "s2", "f", "r", "q"]
DERIVED_METRICS = ["ssfrq5d_total", "ssfrq5d_norm100"]
PREDICTION_META_KEYS = [
    "model",
    "view_setting",
    "task_type",
    "subset",
    "source",
    "manifest_json",
]
PREDICTION_RECORD_KEYS = [
    "pointcloud_file",
    "rel_path",
    "dataset",
    "sample_status",
    "pcqa_index",
    "view_setting",
    "source",
    "timestamp",
]

SSFRQ5D_JUDGE_SYSTEM = """\
You are a careful evaluator for point cloud quality descriptions.
"""

COMMON_CONTEXT_PREFIX_TEMPLATE = """\
待评估输出：[MODEL DESC]
{model_desc}

参考描述：[GOLD DESC]
{gold_desc}
"""

DIMENSION_PROMPTS = {
    "s1": """Evaluate the structural sufficiency of [MODEL DESC] with respect to the reference description [GOLD DESC].

A structurally sufficient description should include the key functional components of a point cloud quality assessment:
(1) an overall summary of the point cloud quality or geometric completeness,
(2) a description of the main visible quality issues or defects,
and (3) a final quality judgment.

Focus on whether [MODEL DESC] contains these essential components in a reasonably complete way, rather than whether it uses the same wording as [GOLD DESC].

Please rate:
- Score 2 if [MODEL DESC] includes all or almost all essential components in a complete and well-formed manner.
- Score 1 if [MODEL DESC] includes only part of the essential components, or some components are weak or incomplete.
- Score 0 if [MODEL DESC] misses most of the essential components or is structurally inadequate.

Please only provide the result in the following format:
Score:""",
    "s2": """Evaluate the specificity of [MODEL DESC] compared with the reference description [GOLD DESC].

A specific description should mention concrete and observable quality cues of the point cloud, such as incompleteness, sparsity, non-uniform density, noise, outliers, local structural defects, broken boundaries, or other visible artifacts. Generic statements such as "the quality is poor" without supporting details should be rated low.

Focus on whether [MODEL DESC] provides detailed, low-level, and evidence-based observations, rather than vague or template-like comments.

Please rate:
- Score 2 if [MODEL DESC] is highly specific, detailed, and supported by clear observable evidence.
- Score 1 if [MODEL DESC] contains some useful details but is still partly generic or insufficiently specific.
- Score 0 if [MODEL DESC] is vague, generic, or lacks concrete evidence.

Please only provide the result in the following format:
Score:""",
    "f": """Evaluate whether [MODEL DESC] is faithful to the reference description [GOLD DESC].

Minor wording differences are acceptable. Focus on whether [MODEL DESC] is semantically consistent with [GOLD DESC]. Only penalize clear mismatches, hallucinated defects, major exaggeration or downplaying, or statements that contradict the reference description.

Please rate:
- Score 2 if [MODEL DESC] is overall faithful to [GOLD DESC] and contains no major contradictions.
- Score 1 if [MODEL DESC] is partly faithful but includes a few minor mismatches or slight distortions.
- Score 0 if [MODEL DESC] contains obvious contradictions, hallucinations, or seriously misleading statements.

Please only provide the result in the following format:
Score:""",
    "r": """Evaluate the reasoning coherence of [MODEL DESC].

The described quality issues in [MODEL DESC] should logically support its final quality judgment. For example, if the description mentions severe incompleteness, strong noise, serious structural defects, or multiple obvious problems, the final quality should tend to be bad. If the description mentions only mild issues, the final quality should tend to be usable. If the description reports little to no noticeable problem, the final quality should tend to be good.

Use [GOLD DESC] only to understand the expected relationship between the described issues and the final judgment, but focus on whether the reasoning and conclusion in [MODEL DESC] are logically coherent.

Please rate:
- Score 2 if the described issues and the final quality judgment in [MODEL DESC] are highly coherent.
- Score 1 if they are mostly coherent but contain minor logical gaps.
- Score 0 if the final judgment is poorly supported or clearly inconsistent with the preceding description.

Please only provide the result in the following format:
Score:""",
    "q": """Evaluate the accuracy of the final quality judgment in [MODEL DESC] compared to the reference description [GOLD DESC].

The final judgment in [MODEL DESC] should follow the format:
"Overall, the quality of this point cloud is [good/usable/bad]".

The quality levels have an ordinal relationship:
bad < usable < good

Please compare the final quality judgment in [MODEL DESC] with the reference quality implied by [GOLD DESC], and rate:
- Score 2 if the predicted quality level exactly matches the reference level.
- Score 1 if the predicted quality level differs by one adjacent level.
- Score 0 if the predicted quality level differs by two levels or is completely incorrect.

Please only provide the result in the following format:
Score:""",
}


ANSWER_KEYS = {
    "answer",
    "text",
    "content",
    "rationale",
    "reasoning",
    "rationale_pred",
    "reasoning_pred",
    "reasoning_text",
}
SKIP_RECURSE_KEYS = {
    "prompt",
    "prompt_key",
    "latency",
    "usage",
    "estimated_cost",
    "timestamp",
}

EMPTY_USAGE: Dict[str, Optional[int]] = {
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
    "cached_tokens": None,
    "reasoning_tokens": None,
    "web_search_requests": None,
}


def _clean_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _resolve_cli_or_env(
    cli_value: Optional[str],
    env_names: Sequence[str],
    default: str = "",
) -> str:
    direct = _clean_str(cli_value)
    if direct:
        return direct
    for env_name in env_names:
        val = _clean_str(os.getenv(env_name, ""))
        if val:
            return val
    return default


def _safe_path_token(value: Any, *, default: str) -> str:
    text = _clean_str(value)
    if not text:
        return default
    text = re.sub(r"[\\/:\s]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or default


def _build_default_output_dir(
    *,
    predictions_path: Path,
    judge_model: str,
    prediction_meta: Dict[str, Any],
) -> Path:
    pred_model = _safe_path_token(
        prediction_meta.get("model") or predictions_path.parent.parent.name,
        default="unknown_model",
    )
    view_setting = _safe_path_token(
        prediction_meta.get("view_setting"),
        default="unknown_view",
    )
    task_type = _safe_path_token(
        prediction_meta.get("task_type") or "reasoning",
        default="reasoning",
    )
    subset = _safe_path_token(
        prediction_meta.get("subset") or "unknown_subset",
        default="unknown_subset",
    )
    pred_stem = _safe_path_token(predictions_path.stem, default="predictions")
    judge_tag = _safe_path_token(judge_model, default="judge")
    config_tag = f"{view_setting}_{task_type}_{subset}"
    return DEFAULT_OUTPUT_ROOT / pred_model / judge_tag / config_tag / pred_stem


def _build_run_instance_output_dir(
    *,
    base_output_dir: Path,
    dry_run: bool,
    max_samples: Optional[int],
    num_judge_runs: int,
) -> Path:
    max_tag = f"max{max_samples}" if max_samples is not None else "all"
    if dry_run:
        return base_output_dir / f"dryrun_{max_tag}"
    return base_output_dir / f"judge_{max_tag}_runs{num_judge_runs}"


def _build_run_meta(
    *,
    predictions_path: Path,
    final_protocol_dir: Path,
    judge_model: str,
    num_judge_runs: int,
    max_samples: Optional[int],
    use_all_reasoning_runs: bool,
    reasoning_effort: str,
    max_completion_tokens: int,
    dry_run: bool,
    max_clean_chars: int,
    append_canonical_tail: bool,
    adaptive_completion_retry_budgets: Sequence[int],
    truncate_retry_max_attempts: int,
) -> Dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "predictions_path": str(predictions_path),
        "final_protocol_dir": str(final_protocol_dir),
        "judge_model": judge_model,
        "num_judge_runs": int(num_judge_runs),
        "max_samples": max_samples,
        "use_all_reasoning_runs": bool(use_all_reasoning_runs),
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": int(max_completion_tokens),
        "dry_run": bool(dry_run),
        "max_clean_chars": int(max_clean_chars),
        "append_canonical_tail": bool(append_canonical_tail),
        "adaptive_completion_retry_budgets": [int(v) for v in adaptive_completion_retry_budgets],
        "truncate_retry_max_attempts": int(truncate_retry_max_attempts),
    }


def _is_cache_meta_compatible(existing_meta: Dict[str, Any], expected_meta: Dict[str, Any]) -> bool:
    if not isinstance(existing_meta, dict):
        return False
    if int(existing_meta.get("cache_schema_version") or 0) != CACHE_SCHEMA_VERSION:
        return False
    compare_keys = [
        "predictions_path",
        "final_protocol_dir",
        "judge_model",
        "num_judge_runs",
        "max_samples",
        "use_all_reasoning_runs",
        "reasoning_effort",
        "max_completion_tokens",
        "dry_run",
        "max_clean_chars",
        "append_canonical_tail",
        "adaptive_completion_retry_budgets",
        "truncate_retry_max_attempts",
    ]
    return all(existing_meta.get(key) == expected_meta.get(key) for key in compare_keys)


def _pair_cache_key(sample_id: Any, reasoning_index: Any) -> str:
    sid = _clean_str(sample_id)
    if not sid:
        return ""
    try:
        ridx = int(reasoning_index or 0)
    except (TypeError, ValueError):
        ridx = 0
    return f"{sid}::{ridx}"


def _is_reusable_success_row(row: Dict[str, Any], num_judge_runs: int) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("skipped"):
        return False
    avg_scores = row.get("avg_scores")
    counts = row.get("dimension_judgment_counts")
    if not isinstance(avg_scores, dict) or not isinstance(counts, dict):
        return False
    if any(dim not in avg_scores for dim in DIMENSIONS):
        return False
    return all(int(counts.get(dim) or 0) == int(num_judge_runs) for dim in DIMENSIONS)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_existing_results(
    output_dir: Path,
    *,
    expected_meta: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    scores_json_path = output_dir / "ssfrq5d_judge_scores.json"
    scores_jsonl_path = output_dir / "ssfrq5d_judge_scores.jsonl"
    manifest_path = output_dir / RUN_MANIFEST_NAME

    payload = _load_json(scores_json_path) if scores_json_path.exists() else None
    manifest = _load_json(manifest_path) if manifest_path.exists() else None

    meta_candidates: List[Dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        meta_candidates.append(payload["meta"])
    if isinstance(manifest, dict) and isinstance(manifest.get("run_meta"), dict):
        meta_candidates.append(manifest["run_meta"])

    compatible = any(_is_cache_meta_compatible(meta, expected_meta) for meta in meta_candidates)
    if not compatible:
        return {}, False

    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        payload_rows = payload.get("results")
        if isinstance(payload_rows, list):
            rows = [row for row in payload_rows if isinstance(row, dict)]
    elif scores_jsonl_path.exists():
        with open(scores_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)

    cache: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _pair_cache_key(row.get("sample_id"), row.get("reasoning_index"))
        if key:
            cache[key] = row
    return cache, True


def _write_run_manifest(
    path: Path,
    *,
    run_meta: Dict[str, Any],
    prediction_meta: Dict[str, Any],
    build_stats: Dict[str, Any],
    cached_success_count: int,
    rerun_invalid_count: int,
    new_jobs_count: int,
    final_target_pairs: int,
    completed_pairs: int,
) -> None:
    payload = {
        "run_meta": run_meta,
        "prediction_meta": prediction_meta,
        "build_stats": build_stats,
        "cached_success_count": cached_success_count,
        "rerun_invalid_count": rerun_invalid_count,
        "new_jobs_count": new_jobs_count,
        "final_target_pairs": final_target_pairs,
        "completed_pairs": completed_pairs,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_score_value(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        i = int(v)
        return max(0, min(2, i))
    if isinstance(v, str):
        m = re.search(r"-?\d+", v)
        if m:
            i = int(m.group(0))
            return max(0, min(2, i))
    return None


def parse_dimension_score(text: str) -> Optional[int]:
    """Parse a single-dimension judge response into score 0/1/2."""
    obj = _try_parse_json(text)
    if obj:
        for key in ("score", "Score", "result"):
            if key in obj:
                return _parse_score_value(obj[key])

    match = re.search(r"(?im)\bscore\s*[:：]\s*([0-2])\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"(?m)^\s*([0-2])\s*$", text)
    if match:
        return int(match.group(1))

    return None


def parse_retry_budgets(value: Any) -> List[int]:
    if value is None:
        return list(DEFAULT_ADAPTIVE_COMPLETION_RETRY_BUDGETS)
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
        if not parts:
            return []
        return [int(part) for part in parts]
    if isinstance(value, Sequence):
        out: List[int] = []
        for item in value:
            if item is None:
                continue
            out.append(int(item))
        return out
    return [int(value)]


def normalize_retry_budgets(
    *,
    initial_budget: int,
    retry_budgets: Sequence[int],
    max_extra_attempts: int,
) -> List[int]:
    if max_extra_attempts <= 0:
        return []
    seen = set()
    normalized: List[int] = []
    for budget in retry_budgets:
        try:
            budget_int = int(budget)
        except (TypeError, ValueError):
            continue
        if budget_int <= 0 or budget_int <= int(initial_budget) or budget_int in seen:
            continue
        seen.add(budget_int)
        normalized.append(budget_int)
        if len(normalized) >= max_extra_attempts:
            break
    return normalized


def detect_truncation_suspected(
    *,
    raw_response: Any,
    finish_reason: Optional[str],
    usage: Optional[Dict[str, Optional[int]]],
    max_completion_tokens: Optional[int],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    raw_text = "" if raw_response is None else str(raw_response)
    stripped = raw_text.strip().lower()
    if raw_response is None or stripped == "":
        reasons.append("empty_content")
    if stripped and stripped in TRUNCATION_RESPONSE_PREFIXES:
        reasons.append("truncated_score_prefix")
    if isinstance(finish_reason, str) and finish_reason.lower() == "length":
        reasons.append("finish_reason_length")
    output_tokens = None
    if isinstance(usage, dict):
        output_tokens = usage.get("output_tokens")
    if (
        output_tokens is not None
        and max_completion_tokens is not None
        and int(output_tokens) >= int(max_completion_tokens)
    ):
        reasons.append("output_tokens_hit_cap")
    deduped = list(dict.fromkeys(reasons))
    return bool(deduped), deduped


def _attach_derived_metrics(scores: Dict[str, Any]) -> Dict[str, Any]:
    total = float(sum(float(scores[d]) for d in DIMENSIONS))
    scores["ssfrq5d_total"] = round(total, 4)
    scores["ssfrq5d_norm100"] = round(total / 10.0 * 100.0, 4)
    return scores


def _extract_text_from_node(node: Any, depth: int = 0) -> List[str]:
    if depth > 4:
        return []
    out: List[str] = []

    if isinstance(node, str):
        t = node.strip()
        if t:
            out.append(t)
        return out

    if isinstance(node, list):
        for item in node:
            out.extend(_extract_text_from_node(item, depth + 1))
        return out

    if isinstance(node, dict):
        for key, val in node.items():
            key_l = str(key).strip().lower()
            if key_l in SKIP_RECURSE_KEYS:
                continue
            if key_l in ANSWER_KEYS and isinstance(val, str):
                t = val.strip()
                if t:
                    out.append(t)
                continue
            if isinstance(val, (dict, list)):
                out.extend(_extract_text_from_node(val, depth + 1))
        return out

    return out


def _dedup_keep_order(texts: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in texts:
        key = re.sub(r"\s+", " ", t).strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def extract_reasoning_candidates(record: Dict[str, Any]) -> List[str]:
    """Extract reasoning text candidates from multiple possible formats."""
    candidates: List[str] = []

    if "reasoning" in record:
        candidates.extend(_extract_text_from_node(record["reasoning"]))

    for key in (
        "reasoning_text",
        "reasoning_pred",
        "rationale_pred",
        "rationale",
        "reasoning_answer",
    ):
        t = _clean_str(record.get(key))
        if t:
            candidates.append(t)

    for key in ("reasoning_runs", "rationale_runs"):
        runs = record.get(key)
        if isinstance(runs, list):
            for item in runs:
                candidates.extend(_extract_text_from_node(item))

    return _dedup_keep_order(candidates)


def derive_sample_id(record: Dict[str, Any]) -> str:
    sid = _clean_str(record.get("sample_id"))
    if sid:
        return sid

    rel = _clean_str(record.get("rel_path") or record.get("relative_path"))
    if rel:
        rel = rel.replace("\\", "/").lstrip("./")
        parts = [p for p in rel.split("/") if p]
        if len(parts) >= 2:
            dataset = parts[0]
            rest = "/".join(parts[1:])
            if rest.lower().endswith(".ply"):
                rest = rest[:-4]
            return f"{dataset}::{rest}"
    return ""


def _extract_prediction_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return {
        key: meta[key]
        for key in PREDICTION_META_KEYS
        if key in meta and meta[key] not in (None, "")
    }


def _extract_record_context(record: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    rel_path = _clean_str(record.get("rel_path") or record.get("relative_path"))
    dataset = _clean_str(record.get("dataset"))
    if not dataset and sample_id and "::" in sample_id:
        dataset = sample_id.split("::", 1)[0]

    context = {
        "pointcloud_file": _clean_str(record.get("pointcloud_file")),
        "rel_path": rel_path,
        "dataset": dataset,
        "sample_status": _clean_str(record.get("sample_status")),
        "pcqa_index": _clean_str(record.get("pcqa_index")),
        "view_setting": _clean_str(record.get("view_setting")),
        "source": _clean_str(record.get("source")),
        "timestamp": _clean_str(record.get("timestamp")),
    }
    return {
        key: value
        for key, value in context.items()
        if key in PREDICTION_RECORD_KEYS and value not in (None, "")
    }


def load_prediction_bundle(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records, {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], {}

    if isinstance(data, dict):
        meta = _extract_prediction_meta(data.get("meta"))
        for key in ("results", "predictions", "items", "data"):
            vals = data.get(key)
            if isinstance(vals, list):
                return [x for x in vals if isinstance(x, dict)], meta
        if "sample_id" in data:
            return [data], meta
    return [], {}


def load_gt_map(final_protocol_dir: Path) -> Dict[str, Dict[str, str]]:
    gt_map: Dict[str, Dict[str, str]] = {}
    for jf in sorted(final_protocol_dir.rglob("final-*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        sid = _clean_str(obj.get("sample_id"))
        if not sid:
            continue
        ai_summary = obj.get("ai_summary") or {}
        stats = obj.get("stats") or {}
        gt_map[sid] = {
            "summary_text": _clean_str(ai_summary.get("summary_text")),
            "final_level": _clean_str(stats.get("final_level")).lower(),
            "json_path": str(jf),
        }
    return gt_map


QUALITY_PATTERNS = [
    re.compile(
        r"overall\s*,?\s*the quality of this point cloud is\s*[*_`\"]*(good|usable|bad)[*_`\"]*",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:quality level|quality)\s*(?:is|:)?\s*(good|usable|bad)\b", re.IGNORECASE),
    re.compile(r"\b(?:final answer|therefore|thus|conclusion)\b[^.]{0,80}\b(good|usable|bad)\b", re.IGNORECASE),
]


def extract_quality_level(text: str) -> Optional[str]:
    found: List[str] = []
    for pat in QUALITY_PATTERNS:
        for m in pat.finditer(text):
            found.append(m.group(1).lower())
    if found:
        return found[-1]

    words = re.findall(r"\b(good|usable|bad)\b", text.lower())
    if words:
        return words[-1]
    return None


def _dedup_sentences(text: str) -> str:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    seen = set()
    keep: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        key = re.sub(r"[^a-z0-9]+", "", c.lower())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        keep.append(c)
    return " ".join(keep) if keep else text.strip()


def clean_reasoning_text(
    raw_text: str,
    *,
    max_chars: int = 1200,
    append_canonical_tail: bool = True,
) -> str:
    text = unicodedata.normalize("NFKC", raw_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```(?:[\w+-]+)?", "", text)
    text = text.replace("```", "")
    text = text.replace("\u200b", "")

    lines: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*+]|>\s*|\d+\.)\s*", "", line)
        line = re.sub(r"^(?:final answer|answer|conclusion|最终答案|结论)\s*[:：\-]\s*", "", line, flags=re.IGNORECASE)
        line = line.replace("✅", "").replace("✔", "").replace("☑", "")
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = line.strip()
        if line:
            lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = _dedup_sentences(text)

    if max_chars > 0 and len(text) > max_chars:
        cut = text[:max_chars].rstrip()
        boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if boundary > int(max_chars * 0.6):
            text = cut[: boundary + 1].strip()
        else:
            text = cut

    if append_canonical_tail:
        level = extract_quality_level(text)
        if level in {"good", "usable", "bad"}:
            text = re.sub(
                r"[^.?!]*overall\s*,?\s*the quality of this point cloud is\s*(?:good|usable|bad)[^.?!]*[.?!]?",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            text = re.sub(r"\s+", " ", text).strip()
            if text and text[-1] not in ".!?":
                text += "."
            text = f"{text} Overall, the quality of this point cloud is {level}."

    return text.strip()


def build_dimension_prompt(dimension: str, model_desc: str, gold_desc: str) -> str:
    context_prefix = COMMON_CONTEXT_PREFIX_TEMPLATE.format(
        model_desc=model_desc,
        gold_desc=gold_desc,
    )
    question_template = DIMENSION_PROMPTS[dimension]
    return f"{context_prefix}\n\n{question_template}"


def _count_failed_reasons(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("skipped"):
            reason = _clean_str(row.get("reason")) or "unknown"
            counts[reason] += 1
    return dict(sorted(counts.items()))


def _count_error_types(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        all_scores = row.get("all_scores") or []
        if not isinstance(all_scores, list):
            continue
        for score_row in all_scores:
            if not isinstance(score_row, dict):
                continue
            for dim in DIMENSIONS:
                err = _clean_str(score_row.get(f"{dim}_error"))
                if err:
                    counts[err] += 1
    return dict(sorted(counts.items()))


def _build_results_bundle_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_samples = len(rows)
    skipped_samples = sum(1 for row in rows if row.get("skipped"))
    scored_samples = total_samples - skipped_samples
    return {
        "scored_samples": scored_samples,
        "total_samples": total_samples,
        "skipped_samples": skipped_samples,
        "failed_reason_counts": _count_failed_reasons(rows),
        "error_type_counts": _count_error_types(rows),
    }


def _collect_invalid_response_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        all_scores = row.get("all_scores") or []
        if not isinstance(all_scores, list):
            continue
        base = {
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "rel_path": row.get("rel_path"),
            "pcqa_index": row.get("pcqa_index"),
            "reasoning_index": row.get("reasoning_index", 0),
            "skip_reason": row.get("reason"),
            "row_skipped": bool(row.get("skipped")),
            "dimension_judgment_counts": row.get("dimension_judgment_counts"),
            "partial_avg_scores": row.get("partial_avg_scores"),
        }
        for score_row in all_scores:
            if not isinstance(score_row, dict):
                continue
            judge_run_idx = int(score_row.get("judge_run_idx", 0) or 0)
            for dim in DIMENSIONS:
                attempt_trace = score_row.get(f"{dim}_attempt_trace")
                if isinstance(attempt_trace, list) and attempt_trace:
                    for attempt in attempt_trace:
                        if not isinstance(attempt, dict):
                            continue
                        parsed_score = attempt.get("parsed_score")
                        attempt_error = _clean_str(attempt.get("error"))
                        if parsed_score is not None and not attempt_error:
                            continue
                        raw_response = attempt.get("raw_response")
                        records.append(
                            {
                                **base,
                                "judge_run_idx": judge_run_idx,
                                "dimension": dim,
                                "score": parsed_score,
                                "error": attempt_error or "missing_or_parse_failed",
                                "attempt_count": score_row.get(f"{dim}_attempt_count"),
                                "attempt_index": attempt.get("attempt_index"),
                                "raw_response": raw_response,
                                "raw_response_repr": repr(raw_response),
                                "raw_response_is_empty": raw_response is None or str(raw_response).strip() == "",
                                "finish_reason": attempt.get("finish_reason"),
                                "output_tokens": attempt.get("output_tokens"),
                                "total_tokens": attempt.get("total_tokens"),
                                "max_completion_tokens_used": attempt.get("max_completion_tokens_used"),
                                "truncation_suspected": bool(attempt.get("truncation_suspected")),
                                "truncation_reasons": attempt.get("truncation_reasons") or [],
                            }
                        )
                    continue

                score = score_row.get(dim)
                err = _clean_str(score_row.get(f"{dim}_error"))
                if score is not None and not err:
                    continue
                raw_response = score_row.get(f"{dim}_raw_response")
                records.append(
                    {
                        **base,
                        "judge_run_idx": judge_run_idx,
                        "dimension": dim,
                        "score": score,
                        "error": err or "missing_or_parse_failed",
                        "attempt_count": score_row.get(f"{dim}_attempt_count"),
                        "attempt_index": None,
                        "raw_response": raw_response,
                        "raw_response_repr": repr(raw_response),
                        "raw_response_is_empty": raw_response is None or str(raw_response).strip() == "",
                        "finish_reason": score_row.get(f"{dim}_finish_reason"),
                        "output_tokens": score_row.get(f"{dim}_output_tokens"),
                        "total_tokens": score_row.get(f"{dim}_total_tokens"),
                        "max_completion_tokens_used": score_row.get(f"{dim}_max_completion_tokens_used"),
                        "truncation_suspected": bool(score_row.get(f"{dim}_truncation_suspected")),
                        "truncation_reasons": score_row.get(f"{dim}_truncation_reasons") or [],
                    }
                )
    return records


def _write_invalid_response_audit(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    invalid_records = _collect_invalid_response_records(rows)
    with open(path, "w", encoding="utf-8") as f:
        for record in invalid_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(invalid_records)


def _preview_raw_response(raw_response: Any, *, max_chars: int = 60) -> str:
    if raw_response is None:
        return "<empty>"
    text = str(raw_response).strip()
    if not text:
        return "<empty>"
    text = text.replace("\n", "\\n")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def _format_invalid_row_debug(row: Dict[str, Any], *, max_items: int = 3) -> str:
    invalid_records = _collect_invalid_response_records([row])
    if not invalid_records:
        return ""
    parts = [
        f"{item['dimension']}={_preview_raw_response(item.get('raw_response'))}"
        for item in invalid_records[:max_items]
    ]
    if len(invalid_records) > max_items:
        parts.append(f"+{len(invalid_records) - max_items} more")
    return " invalid_raw[" + "; ".join(parts) + "]"


async def judge_dimension(
    client: VLMClient,
    *,
    judge_model: str,
    sample_id: str,
    dimension: str,
    model_desc: str,
    gold_desc: str,
    run_idx: int,
    run_id: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    adaptive_completion_retry_budgets: Sequence[int],
    truncate_retry_max_attempts: int,
) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SSFRQ5D_JUDGE_SYSTEM},
        {"role": "user", "content": build_dimension_prompt(dimension, model_desc, gold_desc)},
    ]

    retry_budgets = normalize_retry_budgets(
        initial_budget=max_completion_tokens,
        retry_budgets=adaptive_completion_retry_budgets,
        max_extra_attempts=truncate_retry_max_attempts,
    )
    attempt_budgets = [int(max_completion_tokens), *retry_budgets]
    last_text = None
    last_error = ""
    last_finish_reason = None
    last_usage: Dict[str, Optional[int]] = dict(EMPTY_USAGE)
    last_budget = int(max_completion_tokens)
    last_truncation_suspected = False
    last_truncation_reasons: List[str] = []
    attempt_trace: List[Dict[str, Any]] = []

    for attempt_idx, budget in enumerate(attempt_budgets):
        attempts_used = attempt_idx + 1
        request_t0 = time.time()
        try:
            details, latency = await client.call_async_detailed(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=budget,
            )
        except Exception as e:
            error_info = describe_api_error(e)
            latency = time.time() - request_t0
            attempt_trace.append(
                {
                    "attempt_index": attempts_used,
                    "max_completion_tokens_used": budget,
                    "parsed_score": None,
                    "raw_response": None,
                    "finish_reason": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "truncation_suspected": False,
                    "truncation_reasons": [],
                    "error": error_info["error_kind"],
                }
            )
            print(
                f"    [Judge ERROR] {sample_id} dim={dimension} run={run_idx}: "
                f"{error_info['message']}"
            )
            record = build_usage_record(
                run_id=run_id,
                model=judge_model,
                task_name=f"ssfrq5d_reasoning_judge_{dimension}",
                sample_id=sample_id,
                view_setting="text_only",
                run_idx=run_idx,
                usage=dict(EMPTY_USAGE),
                latency_sec=latency,
                success=False,
                error_msg=error_info["message"],
                error_kind=error_info["error_kind"],
                http_status=error_info["http_status"],
                response_preview=None,
                requested_max_completion_tokens=budget,
            )
            append_usage_log(record)
            return {
                "dimension": dimension,
                "score": None,
                "raw_response": None,
                "run_idx": run_idx,
                "attempt_count": attempts_used,
                "error": error_info["error_kind"],
                "error_message": error_info["message"],
                "http_status": error_info["http_status"],
                "finish_reason": None,
                "output_tokens": None,
                "total_tokens": None,
                "max_completion_tokens_used": budget,
                "truncation_suspected": False,
                "truncation_reasons": [],
                "attempt_trace": attempt_trace,
            }

        text = str(details.get("text") or "")
        usage = dict(details.get("usage") or EMPTY_USAGE)
        finish_reason = details.get("finish_reason")
        raw_response = None if details.get("raw_content_is_none") else text
        score = parse_dimension_score(text)
        truncation_suspected, truncation_reasons = detect_truncation_suspected(
            raw_response=raw_response,
            finish_reason=finish_reason,
            usage=usage,
            max_completion_tokens=budget,
        )
        attempt_trace.append(
            {
                "attempt_index": attempts_used,
                "max_completion_tokens_used": budget,
                "parsed_score": score,
                "raw_response": raw_response,
                "finish_reason": finish_reason,
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "truncation_suspected": truncation_suspected,
                "truncation_reasons": truncation_reasons,
                "error": None if score is not None else "parse_failed",
            }
        )
        record = build_usage_record(
            run_id=run_id,
            model=judge_model,
            task_name=f"ssfrq5d_reasoning_judge_{dimension}",
            sample_id=sample_id,
            view_setting="text_only",
            run_idx=run_idx,
            usage=usage,
            latency_sec=latency,
            success=score is not None,
            error_msg=None if score is not None else "parse_failed",
            error_kind=None if score is not None else "parse_failed",
            http_status=None,
            response_preview=text,
            finish_reason=finish_reason,
            requested_max_completion_tokens=budget,
            raw_content_preview=details.get("raw_content_preview"),
            content_was_empty=details.get("content_was_empty"),
            truncation_suspected=truncation_suspected,
        )
        append_usage_log(record)

        last_text = raw_response
        last_error = "parse_failed"
        last_finish_reason = finish_reason
        last_usage = usage
        last_budget = budget
        last_truncation_suspected = truncation_suspected
        last_truncation_reasons = truncation_reasons

        if score is not None:
            return {
                "dimension": dimension,
                "score": score,
                "raw_response": raw_response,
                "run_idx": run_idx,
                "attempt_count": attempts_used,
                "finish_reason": finish_reason,
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "max_completion_tokens_used": budget,
                "truncation_suspected": truncation_suspected,
                "truncation_reasons": truncation_reasons,
                "attempt_trace": attempt_trace,
            }

        if not truncation_suspected:
            break

    return {
        "dimension": dimension,
        "score": None,
        "raw_response": last_text,
        "run_idx": run_idx,
        "attempt_count": len(attempt_trace),
        "error": last_error or "parse_failed",
        "finish_reason": last_finish_reason,
        "output_tokens": last_usage.get("output_tokens"),
        "total_tokens": last_usage.get("total_tokens"),
        "max_completion_tokens_used": last_budget,
        "truncation_suspected": last_truncation_suspected,
        "truncation_reasons": last_truncation_reasons,
        "attempt_trace": attempt_trace,
    }


async def _judge_job(
    client: VLMClient,
    *,
    pair_idx: int,
    pair: Dict[str, Any],
    judge_model: str,
    dimension: str,
    judge_run_idx: int,
    run_idx: int,
    run_id: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    adaptive_completion_retry_budgets: Sequence[int],
    truncate_retry_max_attempts: int,
) -> Dict[str, Any]:
    scored = await judge_dimension(
        client,
        judge_model=judge_model,
        sample_id=pair["sample_id"],
        dimension=dimension,
        model_desc=pair["reasoning_clean"],
        gold_desc=pair["gold_summary_text"],
        run_idx=run_idx,
        run_id=run_id,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        adaptive_completion_retry_budgets=adaptive_completion_retry_budgets,
        truncate_retry_max_attempts=truncate_retry_max_attempts,
    )
    return {
        "pair_idx": pair_idx,
        "sample_id": pair["sample_id"],
        "reasoning_index": int(pair.get("reasoning_index", 0)),
        "dimension": dimension,
        "judge_run_idx": judge_run_idx,
        "score": scored,
    }


def _mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    mean_v = statistics.fmean(vals)
    std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return round(mean_v, 4), round(std_v, 4)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _progress_suffix(completed: int, total: int, *, elapsed_sec: float) -> str:
    remaining = max(0, total - completed)
    eta_sec = (elapsed_sec / completed * remaining) if completed > 0 else None
    return (
        f" done={completed}/{total}"
        f" remaining={remaining}"
        f" elapsed={_format_duration(elapsed_sec)}"
        f" eta={_format_duration(eta_sec)}"
    )


def _base_row_from_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": pair.get("sample_id", ""),
        "dataset": pair.get("dataset", ""),
        "pointcloud_file": pair.get("pointcloud_file", ""),
        "rel_path": pair.get("rel_path", ""),
        "sample_status": pair.get("sample_status", ""),
        "pcqa_index": pair.get("pcqa_index", ""),
        "view_setting": pair.get("view_setting", ""),
        "source": pair.get("source", ""),
        "timestamp": pair.get("timestamp", ""),
        "reasoning_index": int(pair.get("reasoning_index", 0)),
    }


def _build_row_from_collected(
    *,
    pair: Dict[str, Any],
    collected: Sequence[Dict[str, Any]],
    num_judge_runs: int,
) -> Dict[str, Any]:
    row = _base_row_from_pair(pair)
    per_dimension_scores: Dict[str, List[int]] = {dim: [] for dim in DIMENSIONS}
    raw_by_run: Dict[int, Dict[str, Any]] = {}

    latest_items: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for item in collected:
        dim = str(item.get("dimension", ""))
        judge_run_idx = int(item.get("judge_run_idx", 0))
        if dim not in DIMENSIONS:
            continue
        latest_items[(judge_run_idx, dim)] = item

    for (_, dim), item in sorted(latest_items.items()):
        judge_run_idx = int(item.get("judge_run_idx", 0))
        scored = item.get("score")
        raw_by_run.setdefault(judge_run_idx, {})
        if isinstance(scored, dict):
            raw_by_run[judge_run_idx][f"{dim}_raw_response"] = scored.get("raw_response")
            raw_by_run[judge_run_idx][f"{dim}_attempt_count"] = scored.get("attempt_count", 1)
            raw_by_run[judge_run_idx][f"{dim}_error"] = scored.get("error")
            raw_by_run[judge_run_idx][f"{dim}_finish_reason"] = scored.get("finish_reason")
            raw_by_run[judge_run_idx][f"{dim}_output_tokens"] = scored.get("output_tokens")
            raw_by_run[judge_run_idx][f"{dim}_total_tokens"] = scored.get("total_tokens")
            raw_by_run[judge_run_idx][f"{dim}_max_completion_tokens_used"] = scored.get("max_completion_tokens_used")
            raw_by_run[judge_run_idx][f"{dim}_truncation_suspected"] = bool(scored.get("truncation_suspected"))
            raw_by_run[judge_run_idx][f"{dim}_truncation_reasons"] = scored.get("truncation_reasons") or []
            raw_by_run[judge_run_idx][f"{dim}_attempt_trace"] = scored.get("attempt_trace") or []
            if dim in DIMENSIONS and scored.get("score") is not None:
                raw_by_run[judge_run_idx][dim] = int(scored["score"])
                per_dimension_scores[dim].append(int(scored["score"]))
            else:
                raw_by_run[judge_run_idx][dim] = None
        else:
            raw_by_run[judge_run_idx][dim] = None
            raw_by_run[judge_run_idx][f"{dim}_raw_response"] = None
            raw_by_run[judge_run_idx][f"{dim}_attempt_count"] = 0
            raw_by_run[judge_run_idx][f"{dim}_error"] = "missing_result"
            raw_by_run[judge_run_idx][f"{dim}_finish_reason"] = None
            raw_by_run[judge_run_idx][f"{dim}_output_tokens"] = None
            raw_by_run[judge_run_idx][f"{dim}_total_tokens"] = None
            raw_by_run[judge_run_idx][f"{dim}_max_completion_tokens_used"] = None
            raw_by_run[judge_run_idx][f"{dim}_truncation_suspected"] = False
            raw_by_run[judge_run_idx][f"{dim}_truncation_reasons"] = []
            raw_by_run[judge_run_idx][f"{dim}_attempt_trace"] = []

    counts = {dim: len(per_dimension_scores[dim]) for dim in DIMENSIONS}
    row["num_judgments"] = num_judge_runs
    row["dimension_judgment_counts"] = counts
    row["all_scores"] = [
        {
            "judge_run_idx": run_idx_key,
            **raw_by_run[run_idx_key],
        }
        for run_idx_key in sorted(raw_by_run)
    ]

    total_successes = sum(counts.values())
    if total_successes == 0:
        row["skipped"] = True
        row["reason"] = "all judge calls failed or parse failed"
        return row

    complete = all(counts[dim] == int(num_judge_runs) for dim in DIMENSIONS)
    avg_scores = {
        dim: round(statistics.fmean(per_dimension_scores[dim]), 4)
        for dim in DIMENSIONS
        if per_dimension_scores[dim]
    }

    if not complete:
        row["skipped"] = True
        row["reason"] = "incomplete_result"
        row["partial_avg_scores"] = avg_scores
        return row

    row["skipped"] = False
    row["avg_scores"] = _attach_derived_metrics(avg_scores)
    return row


def _collect_missing_run_slots(
    collected: Sequence[Dict[str, Any]],
    num_judge_runs: int,
) -> List[Tuple[str, int]]:
    success_by_dim: Dict[str, set[int]] = {dim: set() for dim in DIMENSIONS}
    for item in collected:
        dim = str(item.get("dimension", ""))
        if dim not in DIMENSIONS:
            continue
        judge_run_idx = int(item.get("judge_run_idx", 0))
        scored = item.get("score")
        if isinstance(scored, dict) and scored.get("score") is not None:
            success_by_dim[dim].add(judge_run_idx)

    missing: List[Tuple[str, int]] = []
    for dim in DIMENSIONS:
        for judge_run_idx in range(num_judge_runs):
            if judge_run_idx not in success_by_dim[dim]:
                missing.append((dim, judge_run_idx))
    return missing


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_results_bundle(
    path: Path,
    *,
    rows: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    extra_top_level: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {
        "meta": meta,
        "total_count": len(rows),
        "results": list(rows),
    }
    if extra_top_level:
        payload.update(extra_top_level)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_checkpoint(
    *,
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    final_target_pairs: int,
    build_stats: Dict[str, Any],
    prediction_meta: Dict[str, Any],
    run_meta: Dict[str, Any],
    cached_success_count: int,
    rerun_invalid_count: int,
    new_jobs_count: int,
    elapsed_sec: float,
) -> Dict[str, Any]:
    valid_rows = [r for r in rows if not r.get("skipped")]
    bundle_summary = _build_results_bundle_summary(rows)
    invalid_audit_path = output_dir / INVALID_RESPONSES_NAME
    invalid_response_record_count = _write_invalid_response_audit(invalid_audit_path, rows)
    summary = _build_global_summary(
        all_rows=rows,
        valid_rows=valid_rows,
        build_stats=build_stats,
        prediction_meta=prediction_meta,
        judge_model=str(run_meta.get("judge_model") or ""),
        num_judge_runs=int(run_meta.get("num_judge_runs") or 0),
        predictions_path=Path(str(run_meta.get("predictions_path") or "")),
        elapsed_sec=elapsed_sec,
    )
    summary["meta"].update(run_meta)
    summary["meta"]["invalid_response_audit_jsonl"] = str(invalid_audit_path)
    summary["meta"]["invalid_response_record_count"] = invalid_response_record_count

    summary_path = output_dir / "ssfrq5d_judge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    bundle_meta = {
        **run_meta,
        "elapsed_sec": round(elapsed_sec, 2),
        "source": "all_datasets",
    }
    bundle_meta.update(prediction_meta)
    scores_json_path = output_dir / "ssfrq5d_judge_scores.json"
    _write_results_bundle(
        scores_json_path,
        rows=rows,
        meta=bundle_meta,
        extra_top_level={
            "build_stats": build_stats,
            "summary": bundle_summary,
            "invalid_response_audit_jsonl": str(invalid_audit_path),
            "invalid_response_record_count": invalid_response_record_count,
        },
    )

    _write_run_manifest(
        output_dir / RUN_MANIFEST_NAME,
        run_meta=run_meta,
        prediction_meta=prediction_meta,
        build_stats=build_stats,
        cached_success_count=cached_success_count,
        rerun_invalid_count=rerun_invalid_count,
        new_jobs_count=new_jobs_count,
        final_target_pairs=final_target_pairs,
        completed_pairs=len(rows),
    )
    return summary


def _write_sample_table(path: Path, rows: Sequence[Dict[str, Any]]):
    header = [
        "sample_id",
        "pcqa_index",
        "dataset",
        "rel_path",
        "source",
        "view_setting",
        "sample_status",
        "reasoning_index",
        "s1",
        "s2",
        "f",
        "r",
        "q",
        "ssfrq5d_total",
        "ssfrq5d_norm100",
        "num_judgments",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            if row.get("skipped"):
                continue
            avg = row.get("avg_scores", {})
            writer.writerow(
                [
                    row.get("sample_id", ""),
                    row.get("pcqa_index", ""),
                    row.get("dataset", ""),
                    row.get("rel_path", ""),
                    row.get("source", ""),
                    row.get("view_setting", ""),
                    row.get("sample_status", ""),
                    row.get("reasoning_index", 0),
                    avg.get("s1"),
                    avg.get("s2"),
                    avg.get("f"),
                    avg.get("r"),
                    avg.get("q"),
                    avg.get("ssfrq5d_total"),
                    avg.get("ssfrq5d_norm100"),
                    row.get("num_judgments", 0),
                ]
            )


def _build_dataset_summary(valid_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid_rows:
        ds = str(row.get("dataset", "") or "unknown")
        grouped.setdefault(ds, []).append(row)

    out: Dict[str, Any] = {}
    metrics = DIMENSIONS + DERIVED_METRICS
    for ds, rows in grouped.items():
        entry: Dict[str, Any] = {"count": len(rows)}
        for metric in metrics:
            vals = [float(r["avg_scores"][metric]) for r in rows if metric in r.get("avg_scores", {})]
            mean_v, std_v = _mean_std(vals)
            entry[f"{metric}_mean"] = mean_v
            entry[f"{metric}_std"] = std_v
        out[ds] = entry
    return out


def _build_global_summary(
    *,
    all_rows: Sequence[Dict[str, Any]],
    valid_rows: Sequence[Dict[str, Any]],
    build_stats: Dict[str, Any],
    prediction_meta: Dict[str, Any],
    judge_model: str,
    num_judge_runs: int,
    predictions_path: Path,
    elapsed_sec: float,
) -> Dict[str, Any]:
    unique_samples_scored = len({r["sample_id"] for r in valid_rows if r.get("sample_id")})
    bundle_summary = _build_results_bundle_summary(all_rows)
    summary: Dict[str, Any] = {
        "meta": {
            "judge_model": judge_model,
            "num_judge_runs": num_judge_runs,
            "predictions_path": str(predictions_path),
            "elapsed_sec": round(elapsed_sec, 2),
            "scored_count": len(valid_rows),
            "scored_pairs": len(valid_rows),
            "total_pairs": len(all_rows),
            "unique_samples_scored": unique_samples_scored,
        },
        "build_stats": build_stats,
        "by_dataset": _build_dataset_summary(valid_rows),
        **bundle_summary,
    }
    summary["meta"].update(prediction_meta)

    metrics = DIMENSIONS + DERIVED_METRICS
    for metric in metrics:
        vals = [float(r["avg_scores"][metric]) for r in valid_rows if metric in r.get("avg_scores", {})]
        mean_v, std_v = _mean_std(vals)
        summary[f"{metric}_mean"] = mean_v
        summary[f"{metric}_std"] = std_v
    return summary


def _collect_pairs(
    records: Sequence[Dict[str, Any]],
    gt_map: Dict[str, Dict[str, str]],
    *,
    max_samples: Optional[int],
    use_all_reasoning_runs: bool,
    max_clean_chars: int,
    append_canonical_tail: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    seen_samples = 0
    stats = {
        "records_total": len(records),
        "records_missing_sample_id": 0,
        "records_missing_gt": 0,
        "records_missing_gt_summary_text": 0,
        "records_missing_reasoning": 0,
        "cleaned_too_short": 0,
        "pairs_built": 0,
    }

    for rec_idx, rec in enumerate(records):
        sid = derive_sample_id(rec)
        if not sid:
            stats["records_missing_sample_id"] += 1
            continue
        record_context = _extract_record_context(rec, sid)

        gt = gt_map.get(sid)
        if not gt:
            stats["records_missing_gt"] += 1
            continue
        gold_desc = _clean_str(gt.get("summary_text"))
        if not gold_desc:
            stats["records_missing_gt_summary_text"] += 1
            continue

        candidates = extract_reasoning_candidates(rec)
        if not candidates:
            stats["records_missing_reasoning"] += 1
            continue
        if not use_all_reasoning_runs:
            candidates = candidates[:1]

        for ridx, raw in enumerate(candidates):
            cleaned = clean_reasoning_text(
                raw,
                max_chars=max_clean_chars,
                append_canonical_tail=append_canonical_tail,
            )
            if len(cleaned) < 10:
                stats["cleaned_too_short"] += 1
                continue
            pairs.append(
                {
                    "sample_id": sid,
                    "record_index": rec_idx,
                    "reasoning_index": ridx,
                    "reasoning_raw": raw,
                    "reasoning_clean": cleaned,
                    "gold_summary_text": gold_desc,
                    "gold_level": gt.get("final_level", ""),
                    **record_context,
                }
            )

        seen_samples += 1
        if max_samples is not None and seen_samples >= max_samples:
            break

    stats["pairs_built"] = len(pairs)
    stats["records_used"] = seen_samples
    return pairs, stats


async def run_ssfrq5d_eval(
    *,
    predictions_path: Path,
    final_protocol_dir: Path,
    judge_model: str,
    api_key: str,
    api_base: str,
    output_dir: Path,
    num_judge_runs: int,
    max_concurrent: int,
    max_samples: Optional[int],
    use_all_reasoning_runs: bool,
    max_clean_chars: int,
    append_canonical_tail: bool,
    reasoning_effort: str,
    max_completion_tokens: int,
    adaptive_completion_retry_budgets: Sequence[int],
    truncate_retry_max_attempts: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    dry_run: bool,
    force_rerun: bool,
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SSFRQ-5D] predictions={predictions_path}")
    print(f"[SSFRQ-5D] final_protocol={final_protocol_dir}")
    print(f"[SSFRQ-5D] judge_model={judge_model}  runs_per_pair={num_judge_runs}")

    records, prediction_meta = load_prediction_bundle(predictions_path)
    print(f"[SSFRQ-5D] loaded prediction records: {len(records)}")

    gt_map = load_gt_map(final_protocol_dir)
    print(f"[SSFRQ-5D] loaded GT files: {len(gt_map)}")

    pairs, build_stats = _collect_pairs(
        records,
        gt_map,
        max_samples=max_samples,
        use_all_reasoning_runs=use_all_reasoning_runs,
        max_clean_chars=max_clean_chars,
        append_canonical_tail=append_canonical_tail,
    )
    print(f"[SSFRQ-5D] matched pairs: {len(pairs)}")

    cleaned_path = output_dir / "cleaned_reasoning_pairs.jsonl"
    _write_jsonl(cleaned_path, pairs)
    print(f"[SSFRQ-5D] cleaned pairs saved: {cleaned_path}")

    run_meta = _build_run_meta(
        predictions_path=predictions_path,
        final_protocol_dir=final_protocol_dir,
        judge_model=judge_model,
        num_judge_runs=num_judge_runs,
        max_samples=max_samples,
        use_all_reasoning_runs=use_all_reasoning_runs,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        dry_run=dry_run,
        max_clean_chars=max_clean_chars,
        append_canonical_tail=append_canonical_tail,
        adaptive_completion_retry_budgets=adaptive_completion_retry_budgets,
        truncate_retry_max_attempts=truncate_retry_max_attempts,
    )

    if dry_run:
        summary = {
            "meta": {
                "mode": "dry_run",
                "predictions_path": str(predictions_path),
                "final_protocol_dir": str(final_protocol_dir),
                "pairs_built": len(pairs),
            },
            "build_stats": build_stats,
        }
        summary["meta"].update(run_meta)
        summary["meta"].update(prediction_meta)
        summary_path = output_dir / "ssfrq5d_judge_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _write_run_manifest(
            output_dir / RUN_MANIFEST_NAME,
            run_meta=run_meta,
            prediction_meta=prediction_meta,
            build_stats=build_stats,
            cached_success_count=0,
            rerun_invalid_count=0,
            new_jobs_count=0,
            final_target_pairs=len(pairs),
            completed_pairs=0,
        )
        return summary

    cached_rows: Dict[str, Dict[str, Any]] = {}
    cache_compatible = False
    if not force_rerun:
        cached_rows, cache_compatible = _load_existing_results(
            output_dir,
            expected_meta=run_meta,
        )

    cached_success_rows: Dict[str, Dict[str, Any]] = {}
    rerun_invalid_count = 0
    for pair in pairs:
        key = _pair_cache_key(pair.get("sample_id"), pair.get("reasoning_index"))
        existing = cached_rows.get(key)
        if not existing:
            continue
        if _is_reusable_success_row(existing, num_judge_runs):
            cached_success_rows[key] = existing
        else:
            rerun_invalid_count += 1

    cached_success_count = len(cached_success_rows)
    new_pairs_count = max(0, len(pairs) - cached_success_count)
    print(
        "[SSFRQ-5D] resume "
        f"cache_compatible={cache_compatible} "
        f"cached_success_count={cached_success_count} "
        f"rerun_invalid_count={rerun_invalid_count} "
        f"new_jobs_count={new_pairs_count} "
        f"final_target_pairs={len(pairs)}"
    )

    client = VLMClient(
        api_key=api_key,
        api_base=api_base,
        max_concurrent=max_concurrent,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )

    results_path = output_dir / "ssfrq5d_judge_scores.jsonl"
    results_path.write_text("", encoding="utf-8")
    sample_rows: List[Dict[str, Any]] = []

    t0 = time.time()
    chunk_size = max(32, max_concurrent * 4)
    jobs: List[Tuple[int, Dict[str, Any], str, int, int]] = []
    for pair_idx, pair in enumerate(pairs):
        pair_key = _pair_cache_key(pair.get("sample_id"), pair.get("reasoning_index"))
        if pair_key in cached_success_rows:
            continue
        for jr in range(num_judge_runs):
            for dimension in DIMENSIONS:
                jobs.append((
                    pair_idx,
                    pair,
                    dimension,
                    jr,
                    (pair_idx * num_judge_runs * len(DIMENSIONS))
                    + (jr * len(DIMENSIONS))
                    + DIMENSIONS.index(dimension),
                ))

    pending_results: Dict[int, List[Dict[str, Any]]] = {}
    next_pair_to_flush = 0
    repair_run_cursor = len(jobs)
    summary: Optional[Dict[str, Any]] = None

    async def _flush_ready_rows() -> None:
        nonlocal next_pair_to_flush, summary, repair_run_cursor
        expected_jobs = num_judge_runs * len(DIMENSIONS)
        while next_pair_to_flush < len(pairs):
            pair = pairs[next_pair_to_flush]
            key = _pair_cache_key(pair.get("sample_id"), pair.get("reasoning_index"))
            if key in cached_success_rows:
                row = dict(cached_success_rows[key])
                sample_rows.append(row)
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_pairs = next_pair_to_flush + 1
                progress_text = _progress_suffix(
                    completed_pairs,
                    len(pairs),
                    elapsed_sec=time.time() - t0,
                )
                print(
                    f"  [{next_pair_to_flush + 1}/{len(pairs)}] "
                    f"{pair['sample_id']} (r={pair.get('reasoning_index', 0)})  [CACHE]"
                    f"{progress_text}",
                    flush=True,
                )
                summary = _write_checkpoint(
                    output_dir=output_dir,
                    rows=sample_rows,
                    final_target_pairs=len(pairs),
                    build_stats=build_stats,
                    prediction_meta=prediction_meta,
                    run_meta=run_meta,
                    cached_success_count=cached_success_count,
                    rerun_invalid_count=rerun_invalid_count,
                    new_jobs_count=new_pairs_count,
                    elapsed_sec=time.time() - t0,
                )
                next_pair_to_flush += 1
                continue

            collected = list(pending_results.get(next_pair_to_flush, []))
            if len(collected) < expected_jobs:
                break

            row = _build_row_from_collected(
                pair=pair,
                collected=collected,
                num_judge_runs=num_judge_runs,
            )
            repair_round = 0
            while (
                row.get("reason") == "incomplete_result"
                and repair_round < INCOMPLETE_PAIR_REPAIR_ROUNDS
            ):
                missing_slots = _collect_missing_run_slots(collected, num_judge_runs)
                if not missing_slots:
                    break
                repair_results = await asyncio.gather(*[
                    _judge_job(
                        client,
                        pair_idx=next_pair_to_flush,
                        pair=pair,
                        judge_model=judge_model,
                        dimension=dim,
                        judge_run_idx=judge_run_idx,
                        run_idx=repair_run_cursor + offset,
                        run_id=run_id,
                        reasoning_effort=reasoning_effort,
                        max_completion_tokens=max_completion_tokens,
                        adaptive_completion_retry_budgets=adaptive_completion_retry_budgets,
                        truncate_retry_max_attempts=truncate_retry_max_attempts,
                    )
                    for offset, (dim, judge_run_idx) in enumerate(missing_slots)
                ])
                repair_run_cursor += len(missing_slots)
                collected.extend(repair_results)
                row = _build_row_from_collected(
                    pair=pair,
                    collected=collected,
                    num_judge_runs=num_judge_runs,
                )
                repair_round += 1

            ridx = int(pair.get("reasoning_index", 0))
            completed_pairs = next_pair_to_flush + 1
            progress_text = _progress_suffix(
                completed_pairs,
                len(pairs),
                elapsed_sec=time.time() - t0,
            )
            if row.get("skipped"):
                print(
                    f"  [{next_pair_to_flush + 1}/{len(pairs)}] "
                    f"{pair['sample_id']} (r={ridx})  [SKIP] {row.get('reason', '')}"
                    f"{_format_invalid_row_debug(row)}"
                    f"{progress_text}",
                    flush=True,
                )
            else:
                avg_scores = row.get("avg_scores", {})
                print(
                    f"  [{next_pair_to_flush + 1}/{len(pairs)}] {pair['sample_id']} (r={ridx})"
                    f"  [OK] total={avg_scores.get('ssfrq5d_total', 0.0):.3f}"
                    f"{progress_text}",
                    flush=True,
                )

            sample_rows.append(row)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = _write_checkpoint(
                output_dir=output_dir,
                rows=sample_rows,
                final_target_pairs=len(pairs),
                build_stats=build_stats,
                prediction_meta=prediction_meta,
                run_meta=run_meta,
                cached_success_count=cached_success_count,
                rerun_invalid_count=rerun_invalid_count,
                new_jobs_count=new_pairs_count,
                elapsed_sec=time.time() - t0,
            )
            if next_pair_to_flush in pending_results:
                del pending_results[next_pair_to_flush]
            next_pair_to_flush += 1

    await _flush_ready_rows()

    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start : start + chunk_size]
        chunk_results = await asyncio.gather(*[
            _judge_job(
                client,
                pair_idx=pair_idx,
                pair=pair,
                judge_model=judge_model,
                dimension=dimension,
                judge_run_idx=judge_run_idx,
                run_idx=job_run_idx,
                run_id=run_id,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens,
                adaptive_completion_retry_budgets=adaptive_completion_retry_budgets,
                truncate_retry_max_attempts=truncate_retry_max_attempts,
            )
            for pair_idx, pair, dimension, judge_run_idx, job_run_idx in chunk
        ])

        for result in chunk_results:
            pending_results.setdefault(result["pair_idx"], []).append(result)
        await _flush_ready_rows()

    await _flush_ready_rows()

    elapsed = time.time() - t0
    if summary is None:
        summary = _write_checkpoint(
            output_dir=output_dir,
            rows=sample_rows,
            final_target_pairs=len(pairs),
            build_stats=build_stats,
            prediction_meta=prediction_meta,
            run_meta=run_meta,
            cached_success_count=cached_success_count,
            rerun_invalid_count=rerun_invalid_count,
            new_jobs_count=new_pairs_count,
            elapsed_sec=elapsed,
        )

    sample_table_path = output_dir / "ssfrq5d_sample_table.csv"
    _write_sample_table(sample_table_path, sample_rows)

    global_table_path = output_dir / "ssfrq5d_global_table.csv"
    with open(global_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"{d}_mean" for d in DIMENSIONS + DERIVED_METRICS]
        writer.writerow(header)
        writer.writerow([summary.get(h) for h in header])

    valid_rows = [r for r in sample_rows if not r.get("skipped")]
    print("\n[SSFRQ-5D] Completed")
    print(f"  Scored samples: {len(valid_rows)} / {len(sample_rows)}")
    for d in DIMENSIONS + DERIVED_METRICS:
        mean_v = summary.get(f"{d}_mean")
        std_v = summary.get(f"{d}_std")
        if mean_v is not None:
            print(f"  {d:16s} mean={mean_v:.4f} std={std_v:.4f}")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PointQ-Bench SSFRQ-5D reasoning judge"
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="Model output JSON/JSONL. Supports dict(results), dict(predictions), or list format.",
    )
    p.add_argument(
        "--final-protocol-dir",
        default=str(DEFAULT_FINAL_PROTOCOL_DIR),
        help="Reference directory (env: POINTQ_FINAL_PROTOCOL_DIR; default: data/final_protocol)",
    )
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--api-key", default=None, help="Prefer environment: OPENAI_API_KEY")
    p.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint (env: OPENAI_BASE_URL)")
    p.add_argument("--num-judge-runs", type=int, default=1)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional explicit output directory. Default: "
            "outputs/ssfrq5d/<pred_model>/<judge_model>/"
            "<view>_<task>_<subset>/<pred_stem>/<run_instance>"
        ),
    )
    p.add_argument("--use-all-reasoning-runs", action="store_true")
    p.add_argument("--max-clean-chars", type=int, default=1200)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument(
        "--adaptive-completion-retry-budgets",
        default="512,1024",
        help="Comma-separated extra completion caps to try when a response looks truncated.",
    )
    p.add_argument(
        "--truncate-retry-max-attempts",
        type=int,
        default=DEFAULT_TRUNCATE_RETRY_MAX_ATTEMPTS,
        help="Maximum extra truncation-repair attempts after the initial completion cap.",
    )
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--disable-canonical-tail", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)
    final_protocol_dir = Path(args.final_protocol_dir)

    api_key = _resolve_cli_or_env(
        args.api_key,
        ["OPENAI_API_KEY"],
        default=DEFAULT_API_KEY,
    )
    api_base = _resolve_cli_or_env(
        args.api_base,
        ["OPENAI_BASE_URL"],
        default=DEFAULT_API_BASE,
    )

    if not predictions_path.exists():
        raise SystemExit(f"--predictions not found: {predictions_path}")
    if not final_protocol_dir.exists():
        raise SystemExit(
            "final_protocol directory not found. "
            f"Checked: {final_protocol_dir}. "
            "Pass --final-protocol-dir explicitly if needed."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        _, prediction_meta = load_prediction_bundle(predictions_path)
        base_output_dir = _build_default_output_dir(
            predictions_path=predictions_path,
            judge_model=args.judge_model,
            prediction_meta=prediction_meta,
        )
        output_dir = _build_run_instance_output_dir(
            base_output_dir=base_output_dir,
            dry_run=args.dry_run,
            max_samples=args.max_samples,
            num_judge_runs=args.num_judge_runs,
        )

    if not args.dry_run and not api_key:
        raise SystemExit("Missing --api-key (or set OPENAI_API_KEY).")

    if args.num_judge_runs <= 0:
        raise SystemExit("--num-judge-runs must be >= 1")
    if args.max_completion_tokens <= 0:
        raise SystemExit("--max-completion-tokens must be >= 1")
    if args.truncate_retry_max_attempts < 0:
        raise SystemExit("--truncate-retry-max-attempts must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be >= 1")
    if args.retry_backoff <= 0:
        raise SystemExit("--retry-backoff must be > 0")

    adaptive_completion_retry_budgets = parse_retry_budgets(args.adaptive_completion_retry_budgets)

    print(f"[SSFRQ-5D] output_dir={output_dir}")

    asyncio.run(
        run_ssfrq5d_eval(
            predictions_path=predictions_path,
            final_protocol_dir=final_protocol_dir,
            judge_model=args.judge_model,
            api_key=api_key,
            api_base=api_base,
            output_dir=output_dir,
            num_judge_runs=args.num_judge_runs,
            max_concurrent=args.max_concurrent,
            max_samples=args.max_samples,
            use_all_reasoning_runs=args.use_all_reasoning_runs,
            max_clean_chars=args.max_clean_chars,
            append_canonical_tail=not args.disable_canonical_tail,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
            adaptive_completion_retry_budgets=adaptive_completion_retry_budgets,
            truncate_retry_max_attempts=args.truncate_retry_max_attempts,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )
    )


if __name__ == "__main__":
    main()
