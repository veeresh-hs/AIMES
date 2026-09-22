"""Risk bands.

The default cut points are starting values, not universal truths.  They are
re-derived from validation data by scripts/run_experiments.py (cost-sensitive
sweep) and stored in the model registry.
"""
from __future__ import annotations

from ..config import settings

BANDS = ("SAFE", "SUSPICIOUS", "HIGH_RISK", "CRITICAL")


def band_for(risk_score: int) -> str:
    if risk_score >= settings.band_critical:
        return "CRITICAL"
    if risk_score >= settings.band_high:
        return "HIGH_RISK"
    if risk_score >= settings.band_suspicious:
        return "SUSPICIOUS"
    return "SAFE"


def band_colour(band: str) -> str:
    return {"SAFE": "safe", "SUSPICIOUS": "watch",
            "HIGH_RISK": "high", "CRITICAL": "critical"}.get(band, "safe")
