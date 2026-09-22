"""The analysis pipeline: parsed email in, AnalysisResult out."""
from __future__ import annotations

import time
from pathlib import Path

from .config import LAYER_NAMES, settings
from .detectors.anomaly import AnomalyDetector
from .detectors.attachment import AttachmentDetector
from .detectors.header import HeaderDetector
from .detectors.impersonation import ImpersonationDetector
from .detectors.lexical import LexicalDetector
from .detectors.semantic import SemanticDetector
from .detectors.social_engineering import SocialEngineeringDetector
from .detectors.transformer import TransformerDetector
from .detectors.url import URLDetector
from .ensemble.risk_engine import RiskEngine
from .logging_conf import get_logger
from .ml.registry import active_dir, active_version
from .schema import AnalysisResult, ParsedEmail

log = get_logger(__name__)


class Pipeline:
    """Holds one loaded copy of every layer.  Build it once and reuse it."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else active_dir()
        self.version = active_version() if model_dir is None else Path(model_dir).name
        self.detectors = {
            "lexical": LexicalDetector(),
            "transformer": TransformerDetector(),
            "semantic": SemanticDetector(),
            "anomaly": AnomalyDetector(),
            "url": URLDetector(),
            "header": HeaderDetector(),
            "attachment": AttachmentDetector(),
            "impersonation": ImpersonationDetector(),
            "social_engineering": SocialEngineeringDetector(),
        }
        self.engine = RiskEngine()
        # load() is called even when no version is active: every detector checks
        # for its own artefact and reports an accurate reason when it is absent.
        load_dir = Path(self.model_dir) if self.model_dir else Path(settings.model_dir) / "none"
        for detector in self.detectors.values():
            detector.load(load_dir)
        if self.model_dir and Path(self.model_dir).exists():
            self.engine.load(self.model_dir)
            log.info("pipeline loaded model version %s (mode=%s); trained layers: %s",
                     self.version, settings.mode,
                     [n for n, d in self.detectors.items() if d.model_backed])
        else:
            log.warning("no trained model found - rule-based layers only. "
                        "Run: python scripts/train_all.py")

    @property
    def active_layers(self) -> list[str]:
        return [n for n in LAYER_NAMES if n in self.detectors]

    def analyze(self, email: ParsedEmail) -> AnalysisResult:
        start = time.perf_counter()
        layers = {name: detector.analyze(email) for name, detector in self.detectors.items()}
        result = self.engine.combine(layers, message_id=email.message_id,
                                     model_version=self.version or "untrained")
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def layer_scores(self, email: ParsedEmail) -> dict[str, float | None]:
        return {name: detector.analyze(email).score for name, detector in self.detectors.items()}


_PIPELINE: Pipeline | None = None


def get_pipeline(reload: bool = False) -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None or reload:
        _PIPELINE = Pipeline()
    return _PIPELINE
