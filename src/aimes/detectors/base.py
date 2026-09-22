"""Detector base class.  Every layer returns a LayerResult on a 0-1 scale."""
from __future__ import annotations

import time
from typing import Optional

from ..logging_conf import get_logger
from ..schema import LayerResult, ParsedEmail

log = get_logger(__name__)


class Detector:
    name: str = "base"
    #: True once a trained artefact has been loaded from models/<version>/
    model_backed: bool = False

    def load(self, model_dir) -> "Detector":  # pragma: no cover - overridden
        return self

    def _analyze(self, email: ParsedEmail) -> LayerResult:  # pragma: no cover
        raise NotImplementedError

    def analyze(self, email: ParsedEmail) -> LayerResult:
        start = time.perf_counter()
        try:
            result = self._analyze(email)
        except Exception as exc:  # a broken layer must never kill the pipeline
            log.exception("layer %s failed", self.name)
            result = LayerResult(name=self.name, score=None, available=False,
                                 error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = (time.perf_counter() - start) * 1000
        if result.score is not None:
            result.score = float(min(1.0, max(0.0, result.score)))
        return result


def unavailable(name: str, reason: str) -> LayerResult:
    return LayerResult(name=name, score=None, available=False, error=reason, confidence=0.0)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def logistic(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
