"""Merge token price tables: repo benchmark config + optional webapp overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_BENCH_CFG = Path(__file__).resolve().parent.parent / "config" / "token_price_config.json"


def _merge_price_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for prov, models in overlay.items():
        if not isinstance(models, dict):
            out[prov] = models
            continue
        if prov not in out or not isinstance(out.get(prov), dict):
            out[prov] = dict(models)
            continue
        merged_models: Dict[str, Any] = dict(out[prov])
        for model_name, rates in models.items():
            if isinstance(rates, dict) and isinstance(merged_models.get(model_name), dict):
                m = dict(merged_models[model_name])
                m.update(rates)
                merged_models[model_name] = m
            else:
                merged_models[model_name] = rates
        out[prov] = merged_models
    return out


def load_merged_price_config(*, webapp_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load benchmark/config/token_price_config.json, then overlay webapp_dir/token_price_config.json."""
    acc: Dict[str, Any] = {"openai": {}}
    if _BENCH_CFG.exists():
        try:
            with open(_BENCH_CFG, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                acc = _merge_price_dict(acc, data)
        except Exception:
            pass
    wpath = webapp_dir / "token_price_config.json"
    if wpath.exists():
        try:
            with open(wpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                acc = _merge_price_dict(acc, data)
        except Exception:
            pass
    return acc  # type: ignore[return-value]


def estimate_cost(
    *,
    provider: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_tokens: Optional[int],
    price_config: Dict[str, Dict[str, Dict[str, float]]],
    web_search_requests: Optional[int] = None,
) -> Optional[float]:
    provider_cfg = (price_config or {}).get(provider, {})
    mcfg = provider_cfg.get(model)
    if not mcfg:
        return None

    it = int(input_tokens or 0)
    ot = int(output_tokens or 0)
    ct = int(cached_tokens or 0)

    in_price = float(mcfg.get("input_per_1m", 0.0))
    out_price = float(mcfg.get("output_per_1m", 0.0))
    cached_price = float(mcfg.get("cached_input_per_1m", in_price))
    ws_unit = float(mcfg.get("web_search_per_request", 0.0))
    ws_n = int(web_search_requests or 0)

    non_cached_input = max(0, it - ct)
    cost = (
        (non_cached_input / 1_000_000.0) * in_price
        + (ct / 1_000_000.0) * cached_price
        + (ot / 1_000_000.0) * out_price
        + ws_n * ws_unit
    )
    return float(cost)
