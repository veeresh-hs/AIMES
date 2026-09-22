"""Model versioning.

models/
  registry.json      {"active": "v3", "versions": {...metrics...}}
  v1/ v2/ v3/        artefacts for each trained version
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings

REGISTRY = "registry.json"


def registry_path(model_dir: Path | None = None) -> Path:
    return Path(model_dir or settings.model_dir) / REGISTRY


def load_registry(model_dir: Path | None = None) -> dict:
    path = registry_path(model_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"active": None, "versions": {}}


def save_registry(data: dict, model_dir: Path | None = None) -> None:
    path = registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def next_version(model_dir: Path | None = None) -> str:
    data = load_registry(model_dir)
    used = [int(v[1:]) for v in data["versions"] if v.startswith("v") and v[1:].isdigit()]
    return f"v{max(used) + 1 if used else 1}"


def version_dir(version: str, model_dir: Path | None = None) -> Path:
    path = Path(model_dir or settings.model_dir) / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def register(version: str, metrics: dict, meta: dict | None = None,
             model_dir: Path | None = None) -> dict:
    data = load_registry(model_dir)
    data["versions"][version] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "meta": meta or {},
    }
    save_registry(data, model_dir)
    return data


def active_version(model_dir: Path | None = None) -> str | None:
    return load_registry(model_dir).get("active")


def active_dir(model_dir: Path | None = None) -> Path | None:
    version = active_version(model_dir)
    if not version:
        return None
    path = Path(model_dir or settings.model_dir) / version
    return path if path.exists() else None


def promote(version: str, model_dir: Path | None = None) -> None:
    data = load_registry(model_dir)
    if version not in data["versions"]:
        raise KeyError(f"unknown model version {version}")
    data["active"] = version
    save_registry(data, model_dir)


def should_promote(candidate: dict, incumbent: dict | None,
                   min_f1_gain: float = 0.002, max_fpr_increase: float = 0.005) -> tuple[bool, str]:
    """Promotion gate: a new version ships only if it is measurably better and
    does not make false positives worse."""
    if not incumbent:
        return True, "no active model yet"
    f1_new = candidate.get("macro_f1", 0.0)
    f1_old = incumbent.get("macro_f1", 0.0)
    fpr_new = candidate.get("fpr", 1.0)
    fpr_old = incumbent.get("fpr", 1.0)
    if f1_new < f1_old + min_f1_gain:
        return False, f"macro-F1 {f1_new:.4f} does not beat {f1_old:.4f} by {min_f1_gain}"
    if fpr_new > fpr_old + max_fpr_increase:
        return False, f"false-positive rate rose from {fpr_old:.4f} to {fpr_new:.4f}"
    return True, f"macro-F1 {f1_old:.4f} -> {f1_new:.4f}, FPR {fpr_old:.4f} -> {fpr_new:.4f}"
