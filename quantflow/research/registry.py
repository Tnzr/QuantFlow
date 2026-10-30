"""Append-only experiment registry (JSONL). Every run records its config
hash, git revision, data sources and metrics so trial counts are known when
deflating Sharpe ratios."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def registry_path() -> Path:
    return Path(os.getenv("QUANTFLOW_EXPERIMENT_REGISTRY", "data/experiments.jsonl"))


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _clean(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        return _clean(obj.item())
    return obj


def log_experiment(family: str, name: str, config: dict, metrics: dict, notes: str = "", path: Path | None = None) -> dict:
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "family": family,
        "name": name,
        "config_hash": config_hash(config),
        "git_rev": _git_rev(),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(_clean(rec), default=str) + "\n")
    return rec


def load_experiments(family: str | None = None, path: Path | None = None) -> list[dict]:
    p = path or registry_path()
    if not p.exists():
        return []
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in recs if family is None or r.get("family") == family]


def n_trials(family: str, path: Path | None = None) -> int:
    """Distinct configurations tried in a family (for deflated Sharpe)."""
    return len({r["config_hash"] for r in load_experiments(family, path)})
