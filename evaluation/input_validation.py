"""Fail before paid calls when perception inputs cannot join to their GT."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


def validate_perception_inputs(data: Any, csv_root: Path, csv_files: Mapping[str, str]) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Perception input must be an object with a results array.")
    if not data["results"]:
        raise ValueError("Perception results array is empty.")
    if not isinstance(data.get("meta", {}), dict):
        raise ValueError("Perception meta must be an object.")

    indices = set()
    sample_ids = set()
    gt_indices = {}
    valid_gt = {
        "yesno": {"A", "B", "yes", "no"},
        "how": {"A", "B", "C", "good", "usable", "bad"},
        "what": set("ABCDEFGHI") | {f"S{i}" for i in range(1, 9)} | {"NONE"},
    }
    for qtype, filename in csv_files.items():
        path = csv_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required GT CSV not found: {path}")
        seen = set()
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"index", "gt", "question"}.issubset(reader.fieldnames or []):
                raise ValueError(f"GT CSV requires index, gt, question columns: {path}")
            for row in reader:
                idx = (row.get("index") or "").strip()
                raw = (row.get("gt") or "").strip()
                labels = {part.strip() for part in raw.split(",")} if qtype == "what" else {raw}
                if not idx or idx in seen:
                    raise ValueError(f"Empty or duplicate GT index in {path}")
                if not labels or not labels.issubset(valid_gt[qtype]):
                    raise ValueError(f"Invalid GT label in {path} for index {idx}")
                if not (row.get("question") or "").strip():
                    raise ValueError(f"Empty GT question in {path} for index {idx}")
                seen.add(idx)
        if not seen:
            raise ValueError(f"GT CSV has no records: {path}")
        gt_indices[qtype] = seen

    for record in data["results"]:
        if not isinstance(record, dict):
            raise ValueError("Each perception result must be an object.")
        idx = record.get("pcqa_index")
        if not isinstance(idx, str) or not idx.strip() or idx in indices:
            raise ValueError("Perception pcqa_index must be a unique non-empty string.")
        indices.add(idx)
        sid = record.get("sample_id")
        if sid is not None:
            if not isinstance(sid, str) or not sid.strip() or sid in sample_ids:
                raise ValueError("Provided sample_id values must be unique non-empty strings.")
            sample_ids.add(sid)
        perception = record.get("perception")
        if not isinstance(perception, dict) or not any(key in perception for key in csv_files):
            raise ValueError(f"No supported perception question blocks for index {idx}")
        for qtype in csv_files:
            if qtype not in perception:
                continue
            block = perception[qtype]
            if not isinstance(block, dict) or "answer" not in block:
                raise ValueError(f"{qtype} block requires an answer field for index {idx}")
            if block["answer"] is not None and not isinstance(block["answer"], str):
                raise ValueError(f"{qtype} answer must be a string or null for index {idx}")
            if idx not in gt_indices[qtype]:
                raise ValueError(f"Prediction index {idx} is missing from {qtype} GT")
