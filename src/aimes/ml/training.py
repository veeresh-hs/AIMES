"""Training for every trainable layer, plus the ensemble fusion models.

Leakage control:
  * vectorisers, centroids, scalers and the anomaly model see the TRAIN split only
  * the fusion models are fitted on the VALIDATION split, where the base layers
    are already frozen, so no base model ever scores its own training data for
    the meta-learner (blending rather than in-fold stacking)
  * the TEST split is touched once, at evaluation time
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..config import BASE_CLASSES, settings
from ..logging_conf import get_logger
from .datasets import Sample
from .evaluation import multiclass_metrics
from .features import FEATURE_NAMES, feature_row

log = get_logger(__name__)


# ------------------------------------------------------------ layer 1
def build_vectoriser() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=120_000, sublinear_tf=True,
                                 strip_accents="unicode", lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=180_000, sublinear_tf=True,
                                 lowercase=True)),
    ])


def candidate_models() -> dict[str, object]:
    return {
        "MultinomialNB": MultinomialNB(alpha=0.2),
        "LogisticRegression": LogisticRegression(max_iter=2000, C=4.0,
                                                 class_weight="balanced"),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid"),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=42),
    }


def train_lexical(train: Sequence[Sample], val: Sequence[Sample], out_dir: Path) -> dict:
    """Trains all four baselines and keeps the best by validation macro-F1."""
    x_train = [s.email.combined_text for s in train]
    y_train = [s.label for s in train]
    x_val = [s.email.combined_text for s in val]
    y_val = [s.label for s in val]

    vectoriser = build_vectoriser()
    log.info("fitting TF-IDF on %d documents", len(x_train))
    xt = vectoriser.fit_transform(x_train)
    xv = vectoriser.transform(x_val)

    per_model: dict[str, dict] = {}
    best_name, best_score, best_clf = "", -1.0, None
    for name, clf in candidate_models().items():
        dense_needed = name == "RandomForest"
        start = time.perf_counter()
        try:
            clf.fit(xt.toarray() if dense_needed and xt.shape[1] < 8000 else xt, y_train)
        except (TypeError, ValueError) as exc:
            log.warning("%s could not be trained: %s", name, exc)
            continue
        train_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = clf.predict(xv)
        latency = (time.perf_counter() - start) / max(1, len(x_val)) * 1000
        metrics = multiclass_metrics(y_val, pred, BASE_CLASSES)
        metrics["train_seconds"] = round(train_s, 2)
        metrics["latency_ms"] = round(latency, 4)
        per_model[name] = metrics
        log.info("%-20s val macro-F1 %.4f (%.1fs)", name, metrics["macro_f1"], train_s)
        if metrics["macro_f1"] > best_score:
            best_name, best_score, best_clf = name, metrics["macro_f1"], clf

    if best_clf is None:
        raise RuntimeError("no lexical model could be trained")

    pipeline = Pipeline([("features", vectoriser), ("clf", best_clf)])
    joblib.dump({"pipeline": pipeline, "classes": list(best_clf.classes_),
                 "algorithm": best_name}, out_dir / "lexical_model.joblib")
    log.info("selected %s as the lexical layer (val macro-F1 %.4f)", best_name, best_score)
    return {"selected": best_name, "per_model": per_model}


# ------------------------------------------------------------ layer 3
def train_semantic(train: Sequence[Sample], out_dir: Path) -> dict:
    """MiniLM centroids if sentence-transformers is installed locally, else an
    LSA encoder fitted here so the layer still functions offline."""
    texts = [s.email.combined_text for s in train]
    labels = [s.label for s in train]
    backend_name, encode = None, None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(settings.semantic_model, device="cpu")
        backend_name = settings.semantic_model

        def encode(batch):
            return np.asarray(model.encode(batch, batch_size=16, convert_to_numpy=True,
                                           normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)
    except Exception as exc:
        log.warning("sentence-transformers unavailable (%s); fitting LSA encoder instead",
                    type(exc).__name__)
        encoder = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=60_000, min_df=2, sublinear_tf=True,
                                      ngram_range=(1, 2), strip_accents="unicode")),
            ("svd", TruncatedSVD(n_components=min(256, max(16, len(texts) // 4)), random_state=42)),
        ])
        encoder.fit(texts)
        joblib.dump({"encoder": encoder}, out_dir / "semantic_lsa.joblib")
        backend_name = "tfidf-svd-lsa"

        def encode(batch):
            vecs = np.asarray(encoder.transform(batch), dtype=np.float32)
            return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)

    vectors = encode(texts)
    centroids = {}
    for label in BASE_CLASSES:
        rows = vectors[[i for i, l in enumerate(labels) if l == label]]
        if len(rows) == 0:
            continue
        centroid = rows.mean(axis=0)
        centroids[label] = centroid / max(1e-9, float(np.linalg.norm(centroid)))
    joblib.dump({"centroids": centroids, "backend": backend_name},
                out_dir / "semantic_centroids.joblib")
    log.info("semantic centroids built with %s for %s", backend_name, list(centroids))
    return {"backend": backend_name, "classes": list(centroids)}


# ------------------------------------------------------------ layer 4
def train_anomaly(train: Sequence[Sample], out_dir: Path) -> dict:
    ham = [s for s in train if s.label == "HAM"]
    if len(ham) < 30:
        log.warning("only %d ham samples - anomaly layer skipped", len(ham))
        return {"trained": False, "reason": "too few legitimate samples"}

    matrix = np.array([feature_row(s.email) for s in ham], dtype=float)
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    model = IsolationForest(n_estimators=300, contamination=0.05,
                            random_state=42, n_jobs=-1).fit(scaled)
    scores = model.decision_function(scaled)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES,
                 "score_p01": float(np.percentile(scores, 1)),
                 "score_p99": float(np.percentile(scores, 99))},
                out_dir / "anomaly_model.joblib")
    log.info("anomaly model fitted on %d legitimate emails", len(ham))
    return {"trained": True, "n_ham": len(ham), "n_features": len(FEATURE_NAMES)}


# ------------------------------------------------------------ layer 5
def train_url_model(urls: Sequence[str], labels: Sequence[int], out_dir: Path) -> dict:
    if urls is None or len(urls) < 200:
        return {"trained": False, "reason": "no URL corpus in data/raw/url_dataset"}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    from ..detectors.url import FEATURE_NAMES as URL_FEATURES, url_features

    matrix = np.array([[url_features(u)[k] for k in URL_FEATURES] for u in urls], dtype=float)
    y = np.asarray(labels, dtype=int)
    x_tr, x_te, y_tr, y_te = train_test_split(matrix, y, test_size=0.2,
                                              stratify=y, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           random_state=42).fit(x_tr, y_tr)
    from .evaluation import binary_metrics

    probs = model.predict_proba(x_te)[:, 1]
    metrics = binary_metrics(["PHISHING" if v else "HAM" for v in y_te], probs)
    joblib.dump({"model": model, "features": URL_FEATURES}, out_dir / "url_model.joblib")
    log.info("URL model trained on %d urls, held-out ROC-AUC %.4f", len(urls),
             metrics.get("roc_auc", float("nan")))
    return {"trained": True, "n_urls": len(urls), "metrics": metrics}


# ------------------------------------------------------------ fusion
def fit_fusion(layer_matrix: np.ndarray, mask: np.ndarray, y_binary: np.ndarray,
               layer_names: list[str], out_dir: Path) -> dict:
    """Fits both fusion strategies on the validation split.

    layer_matrix  (n_samples, n_layers) layer scores, 0 where unavailable
    mask          (n_samples, n_layers) 1 where the layer produced a score
    """
    features = np.hstack([layer_matrix, mask])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(features, y_binary)

    # weighted variant: non-negative, normalised coefficients on the score block
    coefs = meta.coef_[0][: len(layer_names)]
    positive = np.clip(coefs, 0, None)
    if positive.sum() <= 0:
        weights = {name: 1.0 / len(layer_names) for name in layer_names}
        note = "all fitted coefficients were non-positive; fell back to uniform weights"
    else:
        weights = {name: float(w / positive.sum()) for name, w in zip(layer_names, positive)}
        note = "weights are normalised non-negative logistic-regression coefficients"

    joblib.dump({"meta": meta, "layer_names": layer_names, "weights": weights,
                 "note": note}, out_dir / "fusion.joblib")
    log.info("fusion fitted on %d validation emails: %s", len(y_binary),
             {k: round(v, 3) for k, v in weights.items()})
    return {"weights": weights, "note": note, "n_val": int(len(y_binary))}


def train_base_class_head(layer_matrix: np.ndarray, mask: np.ndarray,
                          y_labels: Sequence[str], lexical_probs: np.ndarray,
                          out_dir: Path) -> dict:
    """Small multinomial head mapping layer evidence to HAM/PROMO/PHISH.

    Used only when the lexical layer is unavailable or low-confidence; the
    lexical model remains the primary source of the base class.
    """
    features = np.hstack([layer_matrix, mask, lexical_probs])
    head = LogisticRegression(max_iter=2000, class_weight="balanced").fit(features, list(y_labels))
    joblib.dump({"head": head, "classes": list(head.classes_)}, out_dir / "class_head.joblib")
    return {"classes": list(head.classes_)}
