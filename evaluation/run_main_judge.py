#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PointQ-Bench Main-Table Perception Judge

Judge-only evaluation pipeline:
  - All predictions are normalized by judge API (async).

Main-table metrics (per 主表指标简洁版.md):
  - YesNo:  Accuracy
  - What:   Sample-F1
  - How:    Macro-F1
  - No overall aggregate score in this version

Usage:
    # Judge-only normalization
    python run_main_judge.py --result-json path/to/result.json

    # Full options
    python run_main_judge.py --result-json path/to/result.json \\
        --csv-root data/csv --output eval_main.json \\
        --api-key KEY --judge-model gpt-4o-mini --max-concurrent 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import os

import numpy as np

# ── Import VLMClient from benchmark utils ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.utils.api_client import VLMClient, append_usage_log, build_usage_record
from benchmark.utils.pointq_paths import csv_root_default, webapp_dir
from benchmark.utils.price_config import estimate_cost, load_merged_price_config

# ═══════════════════════════════════════════════════════════════
#  Token price config (shared with benchmark)
# ═══════════════════════════════════════════════════════════════


def _load_price_config() -> Dict[str, Dict[str, Dict[str, float]]]:
    return load_merged_price_config(webapp_dir=webapp_dir())


def _init_cost_acc() -> Dict[str, Any]:
    return {
        "calls": 0,
        "success_calls": 0,
        "failed_calls": 0,
        "latency_sec_sum": 0.0,
        "input_tokens_sum": 0,
        "output_tokens_sum": 0,
        "total_tokens_sum": 0,
        "cached_tokens_sum": 0,
        "reasoning_tokens_sum": 0,
        "estimated_cost_sum": 0.0,
        "estimated_cost_known_calls": 0,
    }


def _acc_cost(
    acc: Dict[str, Any],
    *,
    success: bool,
    latency_sec: Optional[float],
    usage: Dict[str, Optional[int]],
    estimated_cost: Optional[float],
) -> None:
    acc["calls"] += 1
    if success:
        acc["success_calls"] += 1
    else:
        acc["failed_calls"] += 1

    if latency_sec is not None:
        acc["latency_sec_sum"] += float(latency_sec)

    it = int(usage.get("input_tokens") or 0)
    ot = int(usage.get("output_tokens") or 0)
    tt = int(usage.get("total_tokens") or 0)
    ct = int(usage.get("cached_tokens") or 0)
    rt = int(usage.get("reasoning_tokens") or 0)

    acc["input_tokens_sum"] += it
    acc["output_tokens_sum"] += ot
    acc["total_tokens_sum"] += tt
    acc["cached_tokens_sum"] += ct
    acc["reasoning_tokens_sum"] += rt

    if estimated_cost is not None:
        acc["estimated_cost_sum"] += float(estimated_cost)
        acc["estimated_cost_known_calls"] += 1

# ═══════════════════════════════════════════════════════════════
#  Constants & Mappings
# ═══════════════════════════════════════════════════════════════

YESNO_LETTER = {"A": "yes", "B": "no"}

WHAT_LETTER_TO_S = {
    "A": "S1", "B": "S2", "C": "S3", "D": "S4",
    "E": "S5", "F": "S6", "G": "S7", "H": "S8", "I": "NONE",
}

HOW_LETTER = {"A": "good", "B": "usable", "C": "bad"}

VALID_YESNO = {"yes", "no"}
VALID_HOW = {"good", "usable", "bad"}
VALID_S = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}

INVALID = "INVALID"

CSV_FILES = {
    "yesno": "yes_no_questions.csv",
    "what": "what_questions.csv",
    "how": "how_questions.csv",
}

DEFAULT_JUDGE_MODEL = "qwen3.5-flash"
CACHE_SCHEMA_VERSION = 2


# ═══════════════════════════════════════════════════════════════
#  Judge API Prompt Templates  (from judge_prompt.md §4)
# ═══════════════════════════════════════════════════════════════

JUDGE_SYSTEM = "You are a strict answer normalizer for benchmark evaluation."

JUDGE_USER_YESNO = """\
Normalize the raw model response for a binary question.

Question:
{question}

Raw response:
{raw_response}

Valid canonical outputs:
yes
no
INVALID

Rules:
1. Return yes only if the response clearly chooses Yes or A.
2. Return no only if the response clearly chooses No or B.
3. If the response is empty, refuses to answer, says it is unsure without a clear final choice, gives both yes and no, or gives any other option such as C, return INVALID.
4. If the response discusses multiple options but clearly commits to one final answer, use the final committed answer.
5. Ignore extra explanation.

Please only provide the result in the following format:
Prediction: yes
or
Prediction: no
or
Prediction: INVALID"""

JUDGE_USER_WHAT = """\
Normalize the raw model response for a multi-label defect classification task.

Question:
{question}

Raw response:
{raw_response}

Valid defect labels:
S1 = Missing / Incompleteness
S2 = Sampling and Density Abnormalities
S3 = Surface Noise
S4 = Outliers and Clutter
S5 = Geometric Structure Abnormalities
S6 = Quantization and Compression Artifacts
S7 = Coordinate, Scale, and Pose Errors
S8 = Attribute Abnormalities or Misalignment
NONE = no defect

Rules:
1. Return a comma-separated label set using only S1-S8, ordered as S1,S2,...,S8.
2. Return NONE only if the response clearly indicates that no defect applies.
3. If the response includes NONE together with one or more of S1-S8, remove NONE and return only S1-S8.
4. If the response is empty, refuses to answer, has no clear final label set, or cannot be mapped to any valid label, return INVALID.
5. Deduplicate repeated labels.
6. If the response discusses candidates but clearly commits to a final label set, use the final committed set.
7. Ignore all extra explanation.

Please only provide the result in one of the following formats:
Prediction: S1,S3
Prediction: S2
Prediction: NONE
Prediction: INVALID"""

JUDGE_USER_HOW = """\
Normalize the raw model response for a three-way quality classification task.

Question:
{question}

Raw response:
{raw_response}

Valid canonical outputs:
good
usable
bad
INVALID

Rules:
1. Return good only if the response clearly chooses good or A.
2. Return usable only if the response clearly chooses usable or B.
3. Return bad only if the response clearly chooses bad or C.
4. If the response is empty, refuses to answer, says it is unsure without a clear final choice, gives multiple final choices, or gives an invalid class, return INVALID.
5. If the response discusses multiple options but clearly commits to one final answer, use the final committed answer.
6. Ignore extra explanation.

Please only provide the result in the following format:
Prediction: good
or
Prediction: usable
or
Prediction: bad
or
Prediction: INVALID"""

_JUDGE_TEMPLATES = {
    "yesno": JUDGE_USER_YESNO,
    "what": JUDGE_USER_WHAT,
    "how": JUDGE_USER_HOW,
}


# ═══════════════════════════════════════════════════════════════
#  Label Normalization Helpers
# ═══════════════════════════════════════════════════════════════

def _normalize_what(labels: Set[str]) -> Set[str]:
    """NONE → empty; NONE + S_k → keep only S_k."""
    s = labels & VALID_S
    if s:
        return s
    if "NONE" in labels:
        return set()
    return s


# ═══════════════════════════════════════════════════════════════
#  Judge API Normalization
# ═══════════════════════════════════════════════════════════════

def _parse_prediction_line(text: str) -> Optional[str]:
    """Extract the value after ``Prediction:`` from judge output."""
    for line in text.strip().split("\n"):
        m = re.match(r"^\s*Prediction:\s*(.+)", line, re.I)
        if m:
            return m.group(1).strip()
    return None


def _clean_judge_text(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = s.replace("\u3000", " ")
    s = s.replace("，", ",").replace("、", ",").replace("；", ";")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fallback_prediction_text(text: str) -> str:
    s = _clean_judge_text(text)
    if not s:
        return INVALID
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return s


def _normalize_raw_answer_source(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")
    s = s.replace("，", ",").replace("、", ",").replace("；", ";").replace("：", ":")
    s = s.replace("\\,", ",")
    for _ in range(4):
        updated = re.sub(r"\\(?:text|mathrm|operatorname|mathbf)\s*\{([^{}]*)\}", r"\1", s)
        if updated == s:
            break
        s = updated
    return s


def _clean_raw_answer_fragment(text: str) -> str:
    s = _normalize_raw_answer_source(text)
    s = s.replace("**", " ").replace("__", " ").replace("`", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_choice_letters(text: str, valid_letters: Set[str]) -> List[str]:
    seen: List[str] = []
    # Only accept explicit uppercase option letters to avoid mistaking prose
    # articles like "a" for choice labels in fallback parsing.
    for match in re.finditer(r"(?<![A-Za-z0-9])([A-I])(?![A-Za-z0-9])", str(text or "")):
        letter = match.group(1)
        if letter in valid_letters and letter not in seen:
            seen.append(letter)
    return seen


def _collect_local_parse_candidates(text: str) -> List[str]:
    source = _normalize_raw_answer_source(text)
    if not source.strip():
        return []

    candidates: List[str] = []
    seen = set()

    def add(chunk: str) -> None:
        cleaned = _clean_raw_answer_fragment(chunk)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    boxed_re = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.I)
    for match in reversed(list(boxed_re.finditer(source))):
        add(match.group(1))

    marker_re = re.compile(
        r"(?:final|best|correct)\s+(?:answer|choice|selection|options?)\b"
        r"|selected\s+options?\b"
        r"|classified\s+as\b"
        r"|quality\s+(?:rating|level)(?:\s+of\s+[^.\n:]{1,80})?\s+(?:would\s+be|is)\b"
        r"|the\s+answer\s+is\b"
        r"|answer\s+is\b",
        re.I,
    )
    for match in reversed(list(marker_re.finditer(source))):
        tail_lines = [ln.strip() for ln in source[match.end():].split("\n") if ln.strip()]
        if tail_lines:
            add(" ".join(tail_lines[:3]))
            for line in tail_lines[:3]:
                add(line)

    tail = source[-600:] if len(source) > 600 else source
    for bold in reversed(re.findall(r"\*\*([^*]{1,200})\*\*", tail)):
        add(bold)

    lines = [ln.strip() for ln in source.split("\n") if ln.strip()]
    for line in reversed(lines[-5:]):
        add(line)
    if lines:
        add(" ".join(lines[-3:]))

    return candidates


def _local_parse_yesno(raw_answer: str) -> Optional[str]:
    for candidate in _collect_local_parse_candidates(raw_answer):
        upper = candidate.upper()
        lower = candidate.lower()
        words = []
        for word in ("YES", "NO"):
            if re.search(rf"\b{word}\b", upper):
                words.append(word)
        if len(set(words)) == 1:
            return "yes" if words[0] == "YES" else "no"
        if re.search(
            r"\b(?:no|none)\b[^.\n]{0,80}\b(?:noticeable|obvious|clear)?\s*(?:quality\s+)?"
            r"(?:issues?|problems?|defects?)\b",
            lower,
        ):
            return "no"
        if re.search(
            r"\b(?:several|multiple|noticeable|obvious|significant|clear)\s+"
            r"(?:quality\s+)?(?:issues?|problems?|defects?)\b",
            lower,
        ):
            return "yes"
        if re.search(r"\bissues?\s+(?:could|can|would|may)\s+affect\b", lower):
            return "yes"
        if re.search(
            r"\b(?:quality\s+issues?|quality\s+problems?|noise|outliers?|distortion|"
            r"artifacts?|abnormalities|missing|incompleteness|sparse\s+coverage)\b",
            lower,
        ) and re.search(r"\b(?:issue|problem|affect)\b", lower):
            return "yes"
        letters = _extract_choice_letters(candidate, {"A", "B"})
        if len(letters) == 1:
            return YESNO_LETTER[letters[0]]
    return None


def _local_parse_how(raw_answer: str) -> Optional[str]:
    for candidate in _collect_local_parse_candidates(raw_answer):
        upper = candidate.upper()
        lower = candidate.lower()
        explicit = re.search(
            r"\b(?:quality\s+(?:rating|level)[^.\n]{0,80}?\b(?:is|would\s+be)\s*"
            r"|classif\w*[^.\n]{0,40}?\bas\s*"
            r"|considered\s+)"
            r"[\"'“”‘’\s]*"
            r"(good|usable|bad)\b",
            lower,
        )
        if explicit:
            return explicit.group(1)
        words = []
        for level in ("GOOD", "USABLE", "BAD"):
            if re.search(rf"\b{level}\b", upper):
                words.append(level)
        uniq_words = list(dict.fromkeys(words))
        if len(uniq_words) == 1:
            return uniq_words[0].lower()
        letters = _extract_choice_letters(candidate, {"A", "B", "C"})
        if len(letters) == 1:
            return HOW_LETTER[letters[0]]
    return None


def _local_parse_what(raw_answer: str) -> Optional[Set[str]]:
    for candidate in _collect_local_parse_candidates(raw_answer):
        upper = candidate.upper()
        labels: List[str] = []
        for match in re.findall(r"\bS\s*([1-8])\b", upper):
            label = f"S{match}"
            if label not in labels:
                labels.append(label)
        for letter in _extract_choice_letters(candidate, set(WHAT_LETTER_TO_S)):
            mapped = WHAT_LETTER_TO_S[letter]
            if mapped not in labels:
                labels.append(mapped)

        normalized = _normalize_what(set(labels))
        if normalized or "NONE" in labels or re.search(r"\bNO\s+DEFECT\b", upper):
            return normalized
    return None


def _local_parse_raw_answer(qtype: str, raw_answer: str) -> Optional[Union[str, Set[str]]]:
    if qtype == "yesno":
        return _local_parse_yesno(raw_answer)
    if qtype == "what":
        return _local_parse_what(raw_answer)
    if qtype == "how":
        return _local_parse_how(raw_answer)
    return None


def _canonicalize_judge_prediction(qtype: str, raw: str) -> str:
    text = _parse_prediction_line(raw) or _fallback_prediction_text(raw)
    text_clean = _clean_judge_text(text)
    upper = text_clean.upper()

    if qtype == "yesno":
        if "INVALID" in upper:
            return INVALID
        if re.search(r"\bYES\b", upper) or re.fullmatch(r"A", upper):
            return "yes"
        if re.search(r"\bNO\b", upper) or re.fullmatch(r"B", upper):
            return "no"
        return INVALID

    if qtype == "how":
        if "INVALID" in upper:
            return INVALID
        if re.search(r"\bGOOD\b", upper) or re.fullmatch(r"A", upper):
            return "good"
        if re.search(r"\bUSABLE\b", upper) or re.fullmatch(r"B", upper):
            return "usable"
        if re.search(r"\bBAD\b", upper) or re.fullmatch(r"C", upper):
            return "bad"
        return INVALID

    if qtype == "what":
        if "INVALID" in upper:
            return INVALID
        labels = []
        for match in re.findall(r"\bS\s*([1-8])\b", upper):
            lab = f"S{match}"
            if lab not in labels:
                labels.append(lab)
        if labels:
            return ",".join(sorted(labels, key=lambda x: int(x[1:])))
        if re.search(r"\bNONE\b", upper) or re.search(r"\bNO DEFECT\b", upper):
            return "NONE"
        return INVALID

    return text_clean or INVALID


async def _judge_call(
    client: VLMClient,
    judge_model: str,
    qtype: str,
    question: str,
    raw_response: str,
    *,
    provider: str,
    price_config: Dict[str, Dict[str, Dict[str, float]]],
    run_id: str,
    run_idx: int,
    sample_id: str,
    cost_acc: Optional[Dict[str, Any]] = None,
    request_extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call judge API and return the raw ``Prediction: xxx`` value."""
    user_msg = _JUDGE_TEMPLATES[qtype].format(
        question=question, raw_response=raw_response,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        text, usage, lat = await client.call_async(
            model=judge_model,
            messages=messages,
            temperature=0.0,
            extra_body=request_extra_body,
        )
        estimated_cost = estimate_cost(
            provider=provider,
            model=judge_model,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cached_tokens"),
            web_search_requests=usage.get("web_search_requests"),
            price_config=price_config,
        )

        rec = build_usage_record(
            run_id=run_id,
            model=judge_model,
            task_name=f"judge_{qtype}",
            sample_id=sample_id,
            view_setting="judge",
            run_idx=run_idx,
            usage=usage,
            latency_sec=lat,
            success=True,
            response_preview=text,
            estimated_cost=estimated_cost,
        )
        append_usage_log(rec)
        if cost_acc is not None:
            _acc_cost(
                cost_acc,
                success=True,
                latency_sec=lat,
                usage=usage,
                estimated_cost=estimated_cost,
            )
        return _canonicalize_judge_prediction(qtype, text)
    except Exception as e:
        # Keep parity with benchmark logging
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }
        rec = build_usage_record(
            run_id=run_id,
            model=judge_model,
            task_name=f"judge_{qtype}",
            sample_id=sample_id,
            view_setting="judge",
            run_idx=run_idx,
            usage=usage,
            latency_sec=0.0,
            success=False,
            error_msg=str(e),
            response_preview=None,
            estimated_cost=None,
        )
        append_usage_log(rec)
        if cost_acc is not None:
            _acc_cost(
                cost_acc,
                success=False,
                latency_sec=None,
                usage=usage,
                estimated_cost=None,
            )
        print(f"    [judge] API error: {e}")
        return INVALID


def _parse_judge_yesno(raw: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_YESNO else INVALID


def _parse_judge_how(raw: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_HOW else INVALID


def _parse_judge_what(raw: str) -> Union[Set[str], str]:
    """Parse ``S1,S3`` / ``NONE`` / ``INVALID``."""
    v = raw.strip().upper()
    if v == INVALID:
        return INVALID
    if v == "NONE":
        return set()
    parts = {p.strip() for p in v.split(",")}
    s = parts & VALID_S
    has_none = "NONE" in parts
    if s:
        return s
    if has_none:
        return set()
    return INVALID


_JUDGE_PARSERS = {
    "yesno": _parse_judge_yesno,
    "what": _parse_judge_what,
    "how": _parse_judge_how,
}


# ═══════════════════════════════════════════════════════════════
#  Ground-Truth Loading  (CSV letter-labels → semantic labels)
# ═══════════════════════════════════════════════════════════════

def load_gt(csv_root: Path) -> Dict[str, Dict[str, dict]]:
    """Load GT from the three CSVs.

    Returns ``{qtype: {pcqa_index: {"gt": ..., "is_boundary": bool, "question": str}}}``.
    """
    all_gt: Dict[str, Dict[str, dict]] = {}
    for qtype, fname in CSV_FILES.items():
        path = csv_root / fname
        if not path.exists():
            print(f"  [WARN] GT CSV not found: {path}")
            continue
        records: Dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx = row["index"].strip()
                raw = row["gt"].strip()
                boundary = row.get("is_boundary", "false").strip().lower() == "true"

                if qtype == "yesno":
                    gt = YESNO_LETTER.get(raw, raw)
                elif qtype == "what":
                    letters = {c.strip() for c in raw.split(",")}
                    gt = _normalize_what(
                        {WHAT_LETTER_TO_S.get(l, l) for l in letters}
                    )
                elif qtype == "how":
                    gt = HOW_LETTER.get(raw, raw)
                else:
                    gt = raw

                records[idx] = {
                    "gt": gt,
                    "is_boundary": boundary,
                    "question": row["question"].strip(),
                }
        all_gt[qtype] = records
    return all_gt


# ═══════════════════════════════════════════════════════════════
#  Per-Sample Extraction  (judge-only → INVALID on parse failure)
# ═══════════════════════════════════════════════════════════════


async def _extract_sample(
    sample: dict,
    gt_data: Dict[str, Dict[str, dict]],
    client: VLMClient,
    judge_model: str,
    *,
    provider: str,
    price_config: Dict[str, Dict[str, Dict[str, float]]],
    run_id: str,
    run_idx_base: int,
    cost_acc: Optional[Dict[str, Any]] = None,
    request_extra_body: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    pcqa_idx = sample.get("pcqa_index")
    perception = sample.get("perception") or {}
    if not pcqa_idx or not perception:
        return None

    record: dict = {
        "pcqa_index": pcqa_idx,
        "sample_id": sample.get("sample_id", ""),
        "dataset": sample.get("dataset", ""),
        "source": sample.get("source") or sample.get("dataset", ""),
        "sample_status": sample.get("sample_status", ""),
        "view_setting": sample.get("view_setting", ""),
    }

    judge_call_idx = 0
    for qtype in ("yesno", "what", "how"):
        q_block = perception.get(qtype)
        if not q_block:
            continue
        gt_map = gt_data.get(qtype, {})
        gt_info = gt_map.get(pcqa_idx)
        if gt_info is None:
            continue

        raw_answer = str(q_block.get("answer") or "").strip()
        question = str(q_block.get("prompt") or gt_info.get("question") or "").strip()

        if not raw_answer:
            raw_result = INVALID
            pred = INVALID
            method = "empty_answer"
        else:
            raw_result = await _judge_call(
                client, judge_model, qtype, question, raw_answer,
                provider=provider,
                price_config=price_config,
                run_id=run_id,
                run_idx=run_idx_base + judge_call_idx,
                sample_id=str(pcqa_idx),
                cost_acc=cost_acc,
                request_extra_body=request_extra_body,
            )
            pred = _JUDGE_PARSERS[qtype](raw_result)
            method = "judge"
            if pred == INVALID:
                fallback_pred = _local_parse_raw_answer(qtype, raw_answer)
                if fallback_pred is not None:
                    pred = fallback_pred
                    method = "judge+local_fallback"
            judge_call_idx += 1

        if pred is None:
            pred = INVALID

        gt_val = gt_info["gt"]
        record[qtype] = {
            "gt": sorted(gt_val) if isinstance(gt_val, set) else gt_val,
            "pred": sorted(pred) if isinstance(pred, set) else pred,
            "method": method,
            "is_boundary": gt_info["is_boundary"],
            "raw_answer": raw_answer,
            "judge_raw_pred": raw_result,
        }

    record["_judge_calls"] = judge_call_idx
    return record


# ═══════════════════════════════════════════════════════════════
#  Metric Computation  (主表指标简洁版.md)
# ═══════════════════════════════════════════════════════════════

def _yesno_accuracy(records: List[dict]) -> Tuple[float, dict]:
    n = correct = inv = 0
    for r in records:
        info = r.get("yesno")
        if not info:
            continue
        n += 1
        p = info["pred"]
        if p == INVALID:
            inv += 1
        elif p == info["gt"]:
            correct += 1
    acc = round(correct / n * 100, 2) if n else 0.0
    return acc, {"n": n, "correct": correct,
                 "invalid": inv,
                 "invalid_rate": round(inv / n * 100, 2) if n else 0.0}


def _what_sample_f1(records: List[dict]) -> Tuple[float, dict]:
    scores: list[float] = []
    n = inv = 0
    for r in records:
        info = r.get("what")
        if not info:
            continue
        n += 1
        pred_raw = info["pred"]
        gt = set(info["gt"]) if isinstance(info["gt"], list) else info["gt"]

        if pred_raw == INVALID:
            scores.append(0.0)
            inv += 1
            continue

        pred = set(pred_raw) if isinstance(pred_raw, list) else pred_raw
        if not isinstance(gt, set) or not isinstance(pred, set):
            scores.append(0.0)
            continue

        if len(gt) == 0 and len(pred) == 0:
            scores.append(1.0)
        elif len(gt) == 0 or len(pred) == 0:
            scores.append(0.0)
        else:
            scores.append(2 * len(gt & pred) / (len(gt) + len(pred)))

    f1 = round(float(np.mean(scores)) * 100, 2) if scores else 0.0
    return f1, {"n": n, "invalid": inv,
                "invalid_rate": round(inv / n * 100, 2) if n else 0.0}


def _how_macro_f1(records: List[dict]) -> Tuple[float, dict]:
    classes = ["good", "usable", "bad"]
    y_true: list[str] = []
    y_pred: list[str] = []
    n = inv = 0
    for r in records:
        info = r.get("how")
        if not info:
            continue
        n += 1
        y_true.append(info["gt"])
        p = info["pred"]
        if p == INVALID:
            inv += 1
        y_pred.append(p)

    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class[cls] = {
            "f1": round(f1 * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "support": sum(1 for t in y_true if t == cls),
        }

    macro = round(float(np.mean(f1s)) * 100, 2) if f1s else 0.0
    return macro, {"n": n, "invalid": inv,
                   "invalid_rate": round(inv / n * 100, 2) if n else 0.0,
                   "per_class": per_class}


def _compute_three_metrics(records: List[dict]) -> dict:
    yn_acc, yn_diag = _yesno_accuracy(records)
    wh_f1, wh_diag = _what_sample_f1(records)
    hw_f1, hw_diag = _how_macro_f1(records)
    return {
        "main_table": {
            "yesno_accuracy": yn_acc,
            "what_sample_f1": wh_f1,
            "how_macro_f1": hw_f1,
        },
        "diagnostics": {"yesno": yn_diag, "what": wh_diag, "how": hw_diag},
    }


def _dataset_key(record: dict) -> str:
    source = str(record.get("source") or "").strip()
    if source:
        return source
    dataset = str(record.get("dataset") or "").strip()
    if dataset:
        return dataset
    return "unknown"


def compute_main_table(records: List[dict]) -> dict:
    all_metrics = _compute_three_metrics(records)

    by_dataset_records: Dict[str, List[dict]] = {}
    for r in records:
        key = _dataset_key(r)
        by_dataset_records.setdefault(key, []).append(r)

    by_dataset = {
        k: _compute_three_metrics(by_dataset_records[k])
        for k in sorted(by_dataset_records)
    }

    # Keep top-level aliases for backward compatibility.
    return {
        "main_table": all_metrics["main_table"],
        "diagnostics": all_metrics["diagnostics"],
        "all_datasets": all_metrics,
        "by_dataset": by_dataset,
    }


# ═══════════════════════════════════════════════════════════════
#  Cache  (skip already-judged samples on re-run)
# ═══════════════════════════════════════════════════════════════

def _load_cache(
    path: Path,
    *,
    result_json: str,
    judge_model: str,
    enable_thinking: bool,
    thinking_budget: Optional[int],
) -> Dict[str, dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta") or {}
        if int(meta.get("cache_schema_version") or 0) != CACHE_SCHEMA_VERSION:
            return {}
        if str(meta.get("result_json") or "") != str(result_json):
            return {}
        if str(meta.get("judge_model") or "") != str(judge_model):
            return {}
        if meta.get("enable_thinking") is not bool(enable_thinking):
            return {}
        if meta.get("thinking_budget") != thinking_budget:
            return {}
        return {r["pcqa_index"]: r for r in data.get("records", [])}
    return {}


def _save_cache(path: Path, records: List[dict], meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records},
                  f, indent=2, ensure_ascii=False)


def _cache_meta(
    *,
    model: str,
    view: str,
    result_json: str,
    judge_model: str,
    enable_thinking: bool,
    thinking_budget: Optional[int],
    mode: str,
    run_id: str,
    elapsed_sec: float,
    processed_samples: int,
    total_samples: int,
) -> dict:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "view_setting": view,
        "result_json": str(result_json),
        "judge_model": judge_model,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
        "mode": mode,
        "run_id": run_id,
        "judged_at": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed_sec, 1),
        "processed_samples": processed_samples,
        "total_samples": total_samples,
    }


def _build_judge_extra_body(*, enable_thinking: bool, thinking_budget: Optional[int]) -> Dict[str, Any]:
    if thinking_budget is not None:
        return {
            "enable_thinking": True,
            "thinking_budget": int(thinking_budget),
        }
    # Local vLLM Qwen3/Qwen3.5 uses the chat template kwargs path.  Keep the
    # top-level field as a harmless compatibility hint for gateways that read it.
    return {
        "enable_thinking": bool(enable_thinking),
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }


def _record_has_retryable_issue(record: dict) -> bool:
    for qtype in ("yesno", "what", "how"):
        info = record.get(qtype)
        if not isinstance(info, dict):
            return True
        pred = info.get("pred")
        method = str(info.get("method") or "")
        if pred == INVALID:
            return True
        if method == "empty_answer":
            return True
    return False


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest JSON: {path}")
    return data


def _filter_samples_by_manifest(
    samples: List[dict],
    manifest: Dict[str, Any],
    *,
    num_shards: Optional[int] = None,
    shard_id: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    selected_ids = manifest.get("selected_sample_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("Manifest missing non-empty selected_sample_ids")

    sample_by_id = {}
    for sample in samples:
        sid = sample.get("sample_id")
        if sid:
            sample_by_id[sid] = sample

    target_ids = list(selected_ids)
    if shard_id is not None:
        shard_map = manifest.get("shard_to_sample_ids") or {}
        target_ids = shard_map.get(str(shard_id)) or []

    missing = [sid for sid in target_ids if sid not in sample_by_id]
    if missing:
        raise ValueError(f"Manifest sample_ids missing from result_json: {len(missing)}")

    filtered = [sample_by_id[sid] for sid in target_ids]
    info = {
        "manifest_selected_total": len(selected_ids),
        "manifest_effective_total": len(filtered),
        "shard_id": shard_id,
        "num_shards": num_shards,
    }
    return filtered, info


# ═══════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════

_SEP = "=" * 56


def _print_report(result: dict, model: str, view: str):
    all_block = result.get("all_datasets", {})
    mt = all_block.get("main_table", result.get("main_table", {}))
    diag = all_block.get("diagnostics", result.get("diagnostics", {}))

    print(f"\n{_SEP}")
    print(f"  PointQ-Bench Perception — Main Table")
    print(f"  Model: {model}  |  View: {view}")
    print(_SEP)
    print()
    print(f"  | {'Metric':<22} | {'Score':>8} |")
    print(f"  |{'-' * 24}|{'-' * 10}|")
    print(f"  | {'YesNo (Accuracy)':<22} | {mt['yesno_accuracy']:>7}% |")
    print(f"  | {'What (Sample-F1)':<22} | {mt['what_sample_f1']:>7}% |")
    print(f"  | {'How (Macro-F1)':<22} | {mt['how_macro_f1']:>7}% |")
    print()

    for qtype in ("yesno", "what", "how"):
        d = diag.get(qtype, {})
        if d:
            print(f"  [{qtype:<5}] n={d['n']}  "
                  f"invalid={d['invalid']} ({d['invalid_rate']}%)")

    by_dataset = result.get("by_dataset", {})
    if by_dataset:
        print(f"\n  Per-dataset metrics:")
        print(f"  | {'Dataset':<20} | {'YesNo':>7} | {'What':>7} | {'How':>7} |")
        print(f"  |{'-' * 22}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")
        for ds, block in by_dataset.items():
            ds_mt = block.get("main_table", {})
            print(
                f"  | {ds:<20} | "
                f"{float(ds_mt.get('yesno_accuracy', 0.0)):>6.2f}% | "
                f"{float(ds_mt.get('what_sample_f1', 0.0)):>6.2f}% | "
                f"{float(ds_mt.get('how_macro_f1', 0.0)):>6.2f}% |"
            )
    print()


# ═══════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════

async def run_judge(args):
    csv_root = Path(args.csv_root) if args.csv_root else csv_root_default()

    with open(args.result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("results", [])
    meta = data.get("meta", {})
    model = meta.get("model", "unknown")
    view = meta.get("view_setting", "?")
    manifest_info = {}
    if args.manifest_json:
        manifest = _load_manifest(args.manifest_json)
        samples, manifest_info = _filter_samples_by_manifest(
            samples,
            manifest,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )

    mode = "judge-shard" if args.manifest_json else "judge-only"
    print(f"\n  PointQ-Bench Main-Table Judge")
    print(f"  Model: {model}  |  View: {view}  |  Samples: {len(samples)}")
    print(f"  Mode:  {mode}")
    print(f"  Judge: {args.judge_model}  |  Concurrency: {args.max_concurrent}")
    if manifest_info:
        print(
            f"  Manifest: selected_total={manifest_info.get('manifest_selected_total')} "
            f"effective_total={manifest_info.get('manifest_effective_total')} "
            f"shard={manifest_info.get('shard_id')}/{manifest_info.get('num_shards')}"
        )

    gt_data = load_gt(csv_root)

    basename = Path(args.result_json).stem
    cache_path = Path(args.cache_path) if args.cache_path else (Path("outputs/perception") / f"judged_{basename}.json")
    cached = _load_cache(
        cache_path,
        result_json=str(args.result_json),
        judge_model=args.judge_model,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )
    print(f"  Cache: {len(cached)} entries -> {cache_path.name}")

    if not (args.api_key or "").strip():
        raise ValueError(
            "Judge-only mode requires API key. Pass --api-key or set OPENAI_API_KEY."
        )
    client = VLMClient(
        api_key=args.api_key,
        api_base=args.api_base,
        max_concurrent=args.max_concurrent,
        timeout=60.0,
    )
    request_extra_body = _build_judge_extra_body(
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )

    all_records: list[dict] = []
    judge_calls = 0
    run_id = str(uuid.uuid4())[:8]
    price_config = _load_price_config()
    cost_acc = _init_cost_acc()
    t0 = time.time()
    judge_call_idx = 0

    for i, sample in enumerate(samples):
        idx = sample.get("pcqa_index")
        if idx and idx in cached and not (args.retry_invalid_samples and _record_has_retryable_issue(cached[idx])):
            cached_rec = dict(cached[idx])
            if not cached_rec.get("dataset"):
                cached_rec["dataset"] = sample.get("dataset", "")
            if not cached_rec.get("source"):
                cached_rec["source"] = sample.get("source") or sample.get("dataset", "")
            if not cached_rec.get("sample_status"):
                cached_rec["sample_status"] = sample.get("sample_status", "")
            if not cached_rec.get("view_setting"):
                cached_rec["view_setting"] = sample.get("view_setting", "")
            all_records.append(cached_rec)
            continue

        record = await _extract_sample(
            sample, gt_data, client, args.judge_model,
            provider=args.provider,
            price_config=price_config,
            run_id=run_id,
            run_idx_base=judge_call_idx,
            cost_acc=cost_acc,
            request_extra_body=request_extra_body,
        )
        if record is None:
            continue

        n_calls = int(record.get("_judge_calls") or 0)
        judge_calls += n_calls
        judge_call_idx += n_calls

        all_records.append(record)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed", flush=True)
        if args.cache_save_every > 0 and (i + 1) % args.cache_save_every == 0:
            _save_cache(
                cache_path,
                all_records,
                _cache_meta(
                    model=model,
                    view=view,
                    result_json=str(args.result_json),
                    judge_model=args.judge_model,
                    enable_thinking=args.enable_thinking,
                    thinking_budget=args.thinking_budget,
                    mode=mode,
                    run_id=run_id,
                    elapsed_sec=time.time() - t0,
                    processed_samples=i + 1,
                    total_samples=len(samples),
                ),
            )

    elapsed = time.time() - t0

    cache_meta = _cache_meta(
        model=model,
        view=view,
        result_json=str(args.result_json),
        judge_model=args.judge_model,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
        mode=mode,
        run_id=run_id,
        elapsed_sec=elapsed,
        processed_samples=len(all_records),
        total_samples=len(samples),
    )
    cache_meta["manifest_json"] = args.manifest_json
    cache_meta["num_shards"] = args.num_shards
    cache_meta["shard_id"] = args.shard_id
    cache_meta["shard_tag"] = args.shard_tag
    _save_cache(cache_path, all_records, cache_meta)

    result = compute_main_table(all_records)
    _print_report(result, model, view)

    if judge_calls:
        print(f"  Judge API calls: {judge_calls}  |  Elapsed: {elapsed:.1f}s")

    if args.output:
        out = {**{"meta": cache_meta}, **result, "per_sample": all_records}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {args.output}")

    # ── 额外输出 cost/token/latency 汇总 JSON ──
    success_calls = int(cost_acc.get("success_calls") or 0)
    known_cost_calls = int(cost_acc.get("estimated_cost_known_calls") or 0)
    cost_summary = {
        "ts": datetime.now().isoformat(),
        "run_id": run_id,
        "provider": args.provider,
        "model": model,
        "view_setting": view,
        "judge_model": args.judge_model,
        "result_json": str(args.result_json),
        "cache_path": str(cache_path),
        "judge_api_calls": judge_calls,
        "elapsed_sec": round(elapsed, 3),
        "overall": cost_acc,
        "derived": {
            "avg_latency_sec_per_success_call": (
                round(float(cost_acc["latency_sec_sum"]) / success_calls, 6)
                if success_calls else None
            ),
            "avg_estimated_cost_usd_per_known_call": (
                round(float(cost_acc["estimated_cost_sum"]) / known_cost_calls, 9)
                if known_cost_calls else None
            ),
        },
    }
    summary_path = (
        Path(args.cost_summary)
        if args.cost_summary
        else (Path("outputs/logs") / f"judge_cost_summary_{run_id}.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, ensure_ascii=False, indent=2)
    print(f"  Cost summary saved to: {summary_path}")

    print()


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="PointQ-Bench Main-Table Perception Judge",
    )
    p.add_argument("--result-json", required=True,
                   help="run_2dvlm_qa.py output JSON path")
    p.add_argument("--csv-root", default=None,
                   help="GT CSV directory (env: POINTQ_CSV_ROOT; default: data/csv under repository)")
    p.add_argument("--output", default=None,
                   help="Save full evaluation results to JSON")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="Judge API key (prefer environment: OPENAI_API_KEY)")
    p.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        help="OpenAI-compatible base URL (env: OPENAI_BASE_URL)",
    )
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Model for answer normalization (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--provider", default="openai",
                   help="Provider key for token_price_config.json (default: openai)")
    p.add_argument("--cost-summary", default=None,
                   help="Save cost/token/latency summary JSON (default: outputs/logs/judge_cost_summary_{run_id}.json)")
    p.add_argument("--cache-path", default=None,
                   help="Override judged cache JSON path (default: outputs/perception/judged_{basename}.json)")
    p.add_argument("--manifest-json", default=None,
                   help="Optional manifest JSON for sample-id based judge sharding")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Total shard count when using manifest-based judge sharding")
    p.add_argument("--shard-id", type=int, default=None,
                   help="Zero-based shard id when using manifest-based judge sharding")
    p.add_argument("--shard-tag", default=None,
                   help="Optional shard tag for logging/reporting")
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="Max concurrent judge API calls")
    p.add_argument("--enable-thinking", action="store_true", default=False,
                   help="Enable judge-model thinking mode (default: disabled)")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Optional judge-model thinking budget; implies thinking enabled")
    p.add_argument("--cache-save-every", type=int, default=50,
                   help="Checkpoint judged cache every N processed samples (default: 50)")
    p.add_argument("--retry-invalid-samples", action="store_true", default=False,
                   help="Re-judge cached samples whose current pred is INVALID or method=empty_answer")
    return p.parse_args()


def main():
    args = parse_args()
    if platform.system() == "Windows" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
