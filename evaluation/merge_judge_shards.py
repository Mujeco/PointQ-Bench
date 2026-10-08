# -*- coding: utf-8 -*-
"""Merge shard outputs from run_main_judge.py into a standard judge JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import run_main_judge as judge_base


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sum_numeric(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value


def merge_judge_shards(
    *,
    manifest_json: Path,
    shards_root: Path,
    output_json: Path,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_json)
    selected_ids = manifest.get("selected_sample_ids") or []
    num_shards = int(manifest.get("num_shards") or 0)
    shard_tags = manifest.get("shard_tags") or {}
    shard_map = manifest.get("shard_to_sample_ids") or {}
    if not selected_ids or not num_shards:
        raise ValueError("Manifest missing selected_sample_ids or num_shards")

    records_by_sample_id: Dict[str, dict] = {}
    shard_summaries: List[Dict[str, Any]] = []
    merged_cost: Dict[str, Any] = {}
    meta_ref: Dict[str, Any] | None = None

    for shard_id in range(num_shards):
        shard_tag = shard_tags.get(str(shard_id)) or f"shard_{shard_id:02d}"
        judge_path = shards_root / shard_tag / "judge_output.json"
        if not judge_path.exists():
            raise FileNotFoundError(f"Missing judge shard output: {judge_path}")

        data = _load_json(judge_path)
        meta = data.get("meta") or {}
        per_sample = data.get("per_sample") or []
        expected_ids = shard_map.get(str(shard_id)) or []
        got_ids = [rec.get("sample_id") for rec in per_sample]
        if len(got_ids) != len(expected_ids) or set(got_ids) != set(expected_ids):
            raise ValueError(
                f"Judge shard mismatch for {shard_tag}: expected={len(expected_ids)} got={len(got_ids)}"
            )

        for rec in per_sample:
            sid = rec.get("sample_id")
            if not sid:
                raise ValueError(f"Judge shard {shard_tag} contains record without sample_id")
            if sid in records_by_sample_id:
                raise ValueError(f"Duplicate sample_id across judge shards: {sid}")
            records_by_sample_id[sid] = rec

        cost_path = shards_root / shard_tag / "judge_cost_summary.json"
        if cost_path.exists():
            cost_data = _load_json(cost_path)
            _sum_numeric(merged_cost, cost_data.get("overall") or {})

        if meta_ref is None:
            meta_ref = meta
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "shard_tag": shard_tag,
                "judge_path": str(judge_path),
                "count": len(per_sample),
            }
        )

    if set(records_by_sample_id) != set(selected_ids):
        missing = [sid for sid in selected_ids if sid not in records_by_sample_id]
        extra = [sid for sid in records_by_sample_id if sid not in set(selected_ids)]
        raise ValueError(
            f"Merged judge shard ids do not match manifest. missing={len(missing)} extra={len(extra)}"
        )

    ordered_records = [records_by_sample_id[sid] for sid in selected_ids]
    result = judge_base.compute_main_table(ordered_records)
    out = {
        "meta": {
            **(meta_ref or {}),
            "manifest_json": str(manifest_json),
            "merge_mode": "judge_shard_merge",
            "merged_at": datetime.now().isoformat(),
            "processed_samples": len(ordered_records),
            "total_samples": len(ordered_records),
        },
        **result,
        "per_sample": ordered_records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    summary = {
        "ts": datetime.now().isoformat(),
        "manifest_json": str(manifest_json),
        "shards_root": str(shards_root),
        "output_json": str(output_json),
        "merged_total": len(ordered_records),
        "overall_cost": merged_cost,
        "shards": shard_summaries,
    }
    summary_path = output_json.parent / "merge_judge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[JUDGE-MERGE] merged_total={len(ordered_records)} -> {output_json}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded judge outputs")
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--shards-root", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_judge_shards(
        manifest_json=Path(args.manifest_json),
        shards_root=Path(args.shards_root),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()
