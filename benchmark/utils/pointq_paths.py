"""Portable data roots. Explicit POINTQ_* environment variables take precedence.

Defaults name expected local directories; data are not bundled with the code.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configured_root(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


def pcqa_inner_root() -> Path:
    return _configured_root("POINTQ_PCQA_ROOT", project_root() / "data" / "pcqa")


def final_protocol_dir() -> Path:
    return _configured_root("POINTQ_FINAL_PROTOCOL_DIR", project_root() / "data" / "final_protocol")


def webapp_dir() -> Path:
    return _configured_root("POINTQ_WEBAPP_DIR", pcqa_inner_root() / "webapp")


def screenshot_root() -> Path:
    return _configured_root("POINTQ_SCREENSHOT_ROOT", project_root() / "data" / "screenshots")


def csv_root_default() -> Path:
    return _configured_root("POINTQ_CSV_ROOT", project_root() / "data" / "csv")


def datasets_root() -> Path:
    return _configured_root("POINTQ_DATASETS_ROOT", project_root() / "data")
