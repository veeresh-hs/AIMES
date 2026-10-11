"""Layer 3 - semantic similarity.

Embeds the message and compares it with class centroids computed on the
training split only.  Backend is MiniLM via sentence-transformers when that is
installed; otherwise a locally fitted TF-IDF + truncated-SVD (LSA) encoder is
used so the layer still works on a machine with no model downloads.  The
backend actually used is recorded in the layer features and must be reported in
the results section.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector, clamp01
from ..config import settings
from ..schema import LayerResult, ParsedEmail

BUNDLE = "semantic_centroids.joblib"
LSA_BUNDLE = "semantic_lsa.joblib"


class EmbeddingBackend:
    """Uniform interface over MiniLM and the LSA fallback."""

    def __init__(self, kind: str, obj=None) -> None:
        self.kind = kind
        self.obj = obj

    @property
    def name(self) -> str:
        return settings.semantic_model if self.kind == "minilm" else "tfidf-svd-lsa"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "minilm":
            vecs = self.obj.encode(texts, batch_size=16, show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        vecs = self.obj.transform(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


def load_backend(model_dir) -> EmbeddingBackend | None:
    """MiniLM if available locally, else a saved LSA encoder, else None."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(settings.semantic_model)
        return EmbeddingBackend("minilm", model)
    except Exception:
        pass
    path = Path(model_dir) / LSA_BUNDLE
    if path.exists():
        import joblib

        return EmbeddingBackend("lsa", joblib.load(path)["encoder"])
    return None


class SemanticDetector(Detector):
    name = "semantic"

    def __init__(self) -> None:
        self.backend: EmbeddingBackend | None = None
        self.centroids: dict[str, np.ndarray] = {}

    def load(self, model_dir) -> "SemanticDetector":
        path = Path(model_dir) / BUNDLE
        if not path.exists():
            return self
        import joblib

        bundle = joblib.load(path)
        expected = bundle.get("backend", "")
        self.centroids = {k: np.asarray(v, dtype=np.float32) for k, v in bundle["centroids"].items()}
        self.backend = load_backend(model_dir)
        if self.backend is None:
            return self
        if self.backend.name != expected:
            # centroid space must match the encoder that produced it
            self.backend = None
            return self
        self.model_backed = True
        return self

    def similarities(self, text: str) -> dict[str, float]:
        vec = self.backend.encode([text])[0]
        return {cls: float(np.dot(vec, centroid)) for cls, centroid in self.centroids.items()}

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if self.backend is None or not self.centroids:
            return LayerResult(name=self.name, score=None, available=False, confidence=0.0,
                               error="No semantic centroids for the active embedding backend.")
        text = email.combined_text
        if not text.strip():
            return LayerResult(name=self.name, score=None, available=False,
                               confidence=0.0, error="Empty message body")

        sims = self.similarities(text)
        ham = sims.get("HAM", 0.0)
        threat = max((v for k, v in sims.items() if k != "HAM"), default=0.0)
        # margin between the closest threat centroid and the ham centroid
        score = clamp01(0.5 + (threat - ham) / 2.0)
        nearest = max(sims, key=sims.get)

        return LayerResult(
            name=self.name,
            score=score,
            confidence=clamp01(abs(threat - ham) * 2),
            features={"backend": self.backend.name,
                      "similarities": {k: round(v, 4) for k, v in sims.items()},
                      "nearest_class": nearest},
            evidence=[f"Closest to the {nearest} centroid "
                      f"(cosine {sims[nearest]:.3f}; ham {ham:.3f}) using {self.backend.name}"],
        )