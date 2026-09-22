"""Train every trainable layer, fit the fusion models, evaluate, and register
the result as a new model version.

    python scripts/train_all.py                 # use whatever is in data/raw
    python scripts/train_all.py --synthetic     # smoke test with generated data
    python scripts/train_all.py --no-promote    # train but do not activate

The promotion gate is in aimes.ml.registry.should_promote: a new version only
becomes active if macro-F1 improves and the false-positive rate does not get
worse.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from aimes.config import BASE_CLASSES, LAYER_NAMES, settings
from aimes.logging_conf import get_logger
from aimes.ml import registry, training
from aimes.ml.datasets import generate_synthetic, is_synthetic, load_all, load_url_dataset
from aimes.ml.evaluation import (binary_metrics, comparison_table, fpr_at_recall,
                                 multiclass_metrics, plot_results, save_report)
from aimes.ml.preprocessing import build_split
from aimes.pipeline import Pipeline

log = get_logger("train_all")


def layer_matrix(pipeline: Pipeline, samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every layer over a split once and cache the scores."""
    names = list(pipeline.detectors)
    scores = np.zeros((len(samples), len(names)))
    mask = np.zeros_like(scores)
    lexical_probs = np.zeros((len(samples), len(BASE_CLASSES)))
    for i, sample in enumerate(samples):
        for j, name in enumerate(names):
            result = pipeline.detectors[name].analyze(sample.email)
            if result.score is not None:
                scores[i, j] = result.score
                mask[i, j] = 1.0
            if name == "lexical" and result.features.get("probs"):
                probs = result.features["probs"]
                lexical_probs[i] = [probs.get(c, 0.0) for c in BASE_CLASSES]
        if (i + 1) % 500 == 0:
            log.info("  scored %d/%d", i + 1, len(samples))
    return scores, mask, lexical_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="force generated smoke-test data")
    ap.add_argument("--n-synthetic", type=int, default=1200)
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap the corpus size for a quick run")
    args = ap.parse_args()

    settings.ensure_dirs()
    samples = generate_synthetic(args.n_synthetic) if args.synthetic else load_all()
    if args.limit:
        samples = samples[: args.limit]
    synthetic = is_synthetic(samples)
    if synthetic:
        log.warning("=" * 72)
        log.warning("SYNTHETIC DATA: metrics below verify the plumbing only.")
        log.warning("They are NOT experimental results and must not go in the report.")
        log.warning("=" * 72)

    split = build_split(samples)
    version = registry.next_version()
    out_dir = registry.version_dir(version)
    log.info("training model version %s into %s", version, out_dir)

    started = time.perf_counter()

    lexical_report = training.train_lexical(split.train, split.val, out_dir)
    semantic_report = training.train_semantic(split.train, out_dir)
    anomaly_report = training.train_anomaly(split.train, out_dir)

    urls, url_labels = load_url_dataset(Path(settings.data_dir) / "raw" / "url_dataset")
    url_report = training.train_url_model(urls, url_labels, out_dir) if urls else \
        {"trained": False, "reason": "no URL corpus in data/raw/url_dataset"}

    # base layers are frozen from here on
    pipeline = Pipeline(model_dir=out_dir)

    log.info("scoring validation split for fusion (%d emails)", len(split.val))
    val_scores, val_mask, val_lex = layer_matrix(pipeline, split.val)
    y_val_labels = [s.label for s in split.val]
    y_val = np.array([0 if l == "HAM" else 1 for l in y_val_labels])
    fusion_report = training.fit_fusion(val_scores, val_mask, y_val,
                                        list(pipeline.detectors), out_dir)
    training.train_base_class_head(val_scores, val_mask, y_val_labels, val_lex, out_dir)

    # reload so the engine picks up the fitted fusion model
    pipeline = Pipeline(model_dir=out_dir)

    log.info("evaluating on the held-out test split (%d emails)", len(split.test))
    results, y_true, risk = [], [], []
    for sample in split.test:
        result = pipeline.analyze(sample.email)
        results.append(result)
        y_true.append(sample.label)
        risk.append(result.risk_score / 100.0)

    metrics = binary_metrics(y_true, risk, threshold=settings.band_high / 100.0)
    metrics["latency_ms"] = float(np.mean([r.total_latency_ms for r in results]))
    metrics["latency_p95_ms"] = float(np.percentile([r.total_latency_ms for r in results], 95))
    class_metrics = multiclass_metrics(y_true, [r.base_class for r in results], BASE_CLASSES)
    metrics["macro_f1"] = class_metrics["macro_f1"]
    operating = fpr_at_recall(y_true, risk, 0.95)

    summary = {
        "version": version,
        "synthetic": synthetic,
        "split_strategy": split.strategy,
        "class_counts": split.counts(),
        "lexical": lexical_report,
        "semantic": semantic_report,
        "anomaly": anomaly_report,
        "url": url_report,
        "fusion": fusion_report,
        "ensemble_test_metrics": metrics,
        "base_class_metrics": class_metrics,
        "operating_point_recall95": operating,
        "train_seconds": round(time.perf_counter() - started, 1),
        "mode": settings.mode,
    }
    (out_dir / "training_report.json").write_text(json.dumps(summary, indent=2, default=float),
                                                  encoding="utf-8")
    save_report({"ensemble": metrics}, Path(settings.data_dir) / "reports" / f"{version}_test.json",
                note="SYNTHETIC SMOKE TEST" if synthetic else "held-out test split")

    incumbent = registry.load_registry().get("versions", {}).get(registry.active_version() or "", {})
    registry.register(version, metrics, {"synthetic": synthetic,
                                         "split": split.strategy,
                                         "lexical_algorithm": lexical_report["selected"],
                                         "semantic_backend": semantic_report["backend"]})
    ok, why = registry.should_promote(metrics, incumbent.get("metrics"))
    if args.no_promote:
        log.info("--no-promote given: %s stays inactive", version)
    elif ok:
        registry.promote(version)
        log.info("promoted %s to active (%s)", version, why)
    else:
        log.warning("%s NOT promoted: %s", version, why)

    print()
    print(comparison_table({f"lexical:{k}": {"f1": v["macro_f1"], "accuracy": v["accuracy"],
                                             "latency_ms": v["latency_ms"]}
                            for k, v in lexical_report["per_model"].items()}
                           | {"ENSEMBLE(test)": metrics}))
    print()
    print(f"Ensemble risk-score ROC-AUC {metrics.get('roc_auc', float('nan')):.4f} | "
          f"FPR {metrics['fpr']:.4f} | recall {metrics['recall']:.4f}")
    print(f"At 95% recall the threshold would be {operating['threshold']:.3f} "
          f"with FPR {operating['fpr']:.4f}")
    if synthetic:
        print("\n*** SYNTHETIC DATA - plumbing check only, not a result ***")
    plot_results({"ENSEMBLE": metrics}, Path(settings.data_dir) / "reports")


if __name__ == "__main__":
    main()
