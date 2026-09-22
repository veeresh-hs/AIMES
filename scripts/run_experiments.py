"""The research experiments.

Research question:
  Does fusing lexical, semantic, structural, URL, header and anomaly evidence
  improve spam and phishing detection over the best single model, at an
  acceptable false-positive rate and under cross-corpus shift?

Experiments produced by this script:
  E1  baselines A-G, individually, on the same held-out split
  E2  fusion: weighted vs logistic stacking vs the best single model
  E3  leave-one-layer-out ablation over all nine layers
  E4  cross-corpus generalisation (train on A+B, test on unseen corpus C)
  E5  operating point: FPR at 95% recall and a cost-sensitive threshold sweep
  E6  latency per layer, p50 and p95

Outputs: data/reports/experiments.json, experiments.md and PNG figures.
Nothing here is written by hand; if a section is missing it is because the
corresponding data was not available, and the report says so.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from aimes.config import BASE_CLASSES, settings
from aimes.logging_conf import get_logger
from aimes.ml import registry
from aimes.ml.datasets import generate_synthetic, is_synthetic, load_all
from aimes.ml.evaluation import (binary_metrics, comparison_table, fpr_at_recall,
                                 multiclass_metrics, plot_results, to_binary)
from aimes.ml.preprocessing import build_split
from aimes.pipeline import Pipeline

log = get_logger("experiments")
LAYER_ORDER = ["lexical", "transformer", "semantic", "anomaly", "url",
               "header", "attachment", "impersonation", "social_engineering"]


def score_split(pipeline: Pipeline, samples) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    names = list(pipeline.detectors)
    scores = np.zeros((len(samples), len(names)))
    mask = np.zeros_like(scores)
    latency = {name: [] for name in names}
    labels = []
    for i, sample in enumerate(samples):
        labels.append(sample.label)
        for j, name in enumerate(names):
            result = pipeline.detectors[name].analyze(sample.email)
            latency[name].append(result.latency_ms)
            if result.score is not None:
                scores[i, j] = result.score
                mask[i, j] = 1.0
        if (i + 1) % 500 == 0:
            log.info("  scored %d/%d", i + 1, len(samples))
    return scores, mask, labels, latency


def fuse_weighted(scores: np.ndarray, mask: np.ndarray, weights: dict, names: list[str]) -> np.ndarray:
    w = np.array([weights.get(n, 0.0) for n in names])
    numerator = (scores * mask * w).sum(axis=1)
    denominator = (mask * w).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=1200)
    ap.add_argument("--model", default=None, help="model version (default: active)")
    args = ap.parse_args()

    model_dir = Path(settings.model_dir) / args.model if args.model else registry.active_dir()
    if not model_dir or not Path(model_dir).exists():
        raise SystemExit("No trained model. Run: python scripts/train_all.py")
    pipeline = Pipeline(model_dir=model_dir)

    samples = generate_synthetic(args.n_synthetic) if args.synthetic else load_all()
    synthetic = is_synthetic(samples)
    split = build_split(samples)
    names = list(pipeline.detectors)

    log.info("E1/E2: scoring the test split (%d emails)", len(split.test))
    scores, mask, labels, latency = score_split(pipeline, split.test)
    y = to_binary(labels)
    results: dict[str, dict] = {}

    # ---------------------------------------------------------------- E1
    for j, name in enumerate(names):
        if mask[:, j].sum() < 0.5 * len(labels):
            log.info("skipping %s in E1: available for only %d/%d emails",
                     name, int(mask[:, j].sum()), len(labels))
            continue
        metrics = binary_metrics(labels, scores[:, j], threshold=0.5)
        metrics["latency_ms"] = float(np.mean(latency[name]))
        metrics["coverage"] = float(mask[:, j].mean())
        results[f"single:{name}"] = metrics

    # lexical multiclass baselines recorded during training, for the A-D table
    training_report = Path(model_dir) / "training_report.json"
    baseline_table = {}
    if training_report.exists():
        baseline_table = json.loads(training_report.read_text())["lexical"]["per_model"]

    # ---------------------------------------------------------------- E2
    weights = pipeline.engine.weights
    weighted = fuse_weighted(scores, mask, weights, names)
    results["fusion:weighted"] = binary_metrics(labels, weighted,
                                                threshold=settings.band_high / 100)
    stacked = np.array([pipeline.engine.stacked_score(
        {n: type("L", (), {"score": (scores[i, j] if mask[i, j] else None)})()
         for j, n in enumerate(names)}) or 0.0 for i in range(len(labels))])
    results["fusion:stacking"] = binary_metrics(labels, stacked,
                                                threshold=settings.band_high / 100)

    full_results, full_risk = [], []
    started = time.perf_counter()
    for sample in split.test:
        result = pipeline.analyze(sample.email)
        full_results.append(result)
        full_risk.append(result.risk_score / 100)
    end_to_end = binary_metrics(labels, full_risk, threshold=settings.band_high / 100)
    end_to_end["latency_ms"] = float(np.mean([r.total_latency_ms for r in full_results]))
    end_to_end["latency_p95_ms"] = float(np.percentile([r.total_latency_ms for r in full_results], 95))
    end_to_end["macro_f1"] = multiclass_metrics(
        labels, [r.base_class for r in full_results], BASE_CLASSES)["macro_f1"]
    results["ENSEMBLE:end_to_end"] = end_to_end
    log.info("E2 done in %.1fs", time.perf_counter() - started)

    # ---------------------------------------------------------------- E3
    ablation = {}
    baseline_auc = results["fusion:stacking"].get("roc_auc", float("nan"))
    for j, name in enumerate(names):
        reduced_mask = mask.copy()
        reduced_mask[:, j] = 0.0
        fused = fuse_weighted(scores, reduced_mask, weights, names)
        metrics = binary_metrics(labels, fused, threshold=settings.band_high / 100)
        ablation[name] = {
            "roc_auc_without": metrics.get("roc_auc"),
            "f1_without": metrics["f1"],
            "fpr_without": metrics["fpr"],
            "delta_roc_auc": (results["fusion:weighted"].get("roc_auc", 0) or 0) - (metrics.get("roc_auc") or 0),
        }

    # ---------------------------------------------------------------- E4
    cross = {}
    if split.cross_corpus:
        log.info("E4: cross-corpus test on %d unseen-source emails", len(split.cross_corpus))
        c_scores, c_mask, c_labels, _ = score_split(pipeline, split.cross_corpus)
        cross["weighted"] = binary_metrics(c_labels, fuse_weighted(c_scores, c_mask, weights, names),
                                           threshold=settings.band_high / 100)
        best_single = max((n for n in names if c_mask[:, names.index(n)].sum() > 0),
                          key=lambda n: results.get(f"single:{n}", {}).get("roc_auc", 0),
                          default=None)
        if best_single:
            cross[f"single:{best_single}"] = binary_metrics(
                c_labels, c_scores[:, names.index(best_single)], threshold=0.5)
    else:
        cross["note"] = ("No held-out corpus available. Add a second source under data/raw "
                         "(for example the Nazario phishing corpus) to run this experiment.")

    # ---------------------------------------------------------------- E5
    operating = {f"recall_{int(r * 100)}": fpr_at_recall(labels, full_risk, r)
                 for r in (0.90, 0.95, 0.99)}
    sweep = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        metrics = binary_metrics(labels, full_risk, threshold=float(threshold))
        # cost model: one missed threat is treated as five times worse than one
        # false alarm.  The ratio is a parameter, not a fact - state it in the report.
        metrics["cost"] = 5 * metrics["fn"] + metrics["fp"]
        sweep.append({"threshold": round(float(threshold), 2), "fpr": metrics["fpr"],
                      "recall": metrics["recall"], "cost": metrics["cost"]})
    best_threshold = min(sweep, key=lambda row: row["cost"])

    # ---------------------------------------------------------------- E6
    timing = {name: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
              for name, v in latency.items()}

    payload = {
        "synthetic": synthetic,
        "model_version": Path(model_dir).name,
        "mode": settings.mode,
        "split": {"strategy": split.strategy, "counts": split.counts()},
        "E1_single_layers": {k: v for k, v in results.items() if k.startswith("single:")},
        "E1_lexical_baselines": baseline_table,
        "E2_fusion": {k: v for k, v in results.items() if not k.startswith("single:")},
        "E3_ablation": ablation,
        "E4_cross_corpus": cross,
        "E5_operating_points": {"fpr_at_recall": operating, "sweep": sweep,
                                "cost_optimal": best_threshold,
                                "cost_model": "5 x false negatives + 1 x false positives"},
        "E6_latency": timing,
    }

    out_dir = Path(settings.data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments.json").write_text(json.dumps(payload, indent=2, default=float),
                                              encoding="utf-8")
    write_markdown(payload, out_dir / "experiments.md")
    plot_results(results, out_dir)

    print()
    print(comparison_table(results))
    print()
    print("Ablation (drop in ROC-AUC when a layer is removed; larger = more useful):")
    for name, row in sorted(ablation.items(), key=lambda t: -(t[1]["delta_roc_auc"] or 0)):
        print(f"  {name:22s} {row['delta_roc_auc']:+.4f}")
    print()
    print(f"Cost-optimal threshold {best_threshold['threshold']:.2f} "
          f"(recall {best_threshold['recall']:.3f}, FPR {best_threshold['fpr']:.3f})")
    if synthetic:
        print("\n*** SYNTHETIC DATA - plumbing check only, not a result ***")
    print(f"\nWritten to {out_dir}")


def write_markdown(payload: dict, path: Path) -> None:
    lines = ["# Experimental results", ""]
    if payload["synthetic"]:
        lines += ["> Generated from SYNTHETIC smoke-test data. These numbers verify that the",
                  "> pipeline runs; they are not experimental results and must not be reported.", ""]
    lines += [f"Model version `{payload['model_version']}`, runtime mode {payload['mode']}, "
              f"split strategy {payload['split']['strategy']}.", "",
              "## E1 - individual layers", "",
              "| Layer | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Latency (ms) | Coverage |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, m in payload["E1_single_layers"].items():
        lines.append(f"| {name.split(':')[1]} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                     f"{m['recall']:.4f} | {m['f1']:.4f} | {m.get('roc_auc', float('nan')):.4f} | "
                     f"{m['fpr']:.4f} | {m['latency_ms']:.2f} | {m['coverage']:.2f} |")

    if payload["E1_lexical_baselines"]:
        lines += ["", "## E1b - lexical baselines (validation split, multiclass)", "",
                  "| Algorithm | Accuracy | Macro F1 | Train (s) | Inference (ms) |", "|---|---|---|---|---|"]
        for name, m in payload["E1_lexical_baselines"].items():
            lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                         f"{m['train_seconds']} | {m['latency_ms']} |")

    lines += ["", "## E2 - fusion", "",
              "| Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |", "|---|---|---|---|---|---|---|---|"]
    for name, m in payload["E2_fusion"].items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {m.get('roc_auc', float('nan')):.4f} | "
                     f"{m.get('pr_auc', float('nan')):.4f} | {m['fpr']:.4f} |")

    lines += ["", "## E3 - leave-one-layer-out ablation", "",
              "Positive delta means removing the layer made the ensemble worse.", "",
              "| Layer removed | ROC-AUC without | F1 without | FPR without | Delta ROC-AUC |", "|---|---|---|---|---|"]
    for name, row in sorted(payload["E3_ablation"].items(), key=lambda t: -(t[1]["delta_roc_auc"] or 0)):
        auc = row["roc_auc_without"]
        lines.append(f"| {name} | {auc:.4f} | {row['f1_without']:.4f} | "
                     f"{row['fpr_without']:.4f} | {row['delta_roc_auc']:+.4f} |")

    lines += ["", "## E4 - cross-corpus generalisation", ""]
    cross = payload["E4_cross_corpus"]
    if "note" in cross:
        lines.append(cross["note"])
    else:
        lines += ["| System | Accuracy | Recall | F1 | ROC-AUC | FPR |", "|---|---|---|---|---|---|"]
        for name, m in cross.items():
            lines.append(f"| {name} | {m['accuracy']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                         f"{m.get('roc_auc', float('nan')):.4f} | {m['fpr']:.4f} |")

    operating = payload["E5_operating_points"]
    lines += ["", "## E5 - operating points", "",
              f"Cost model: {operating['cost_model']}.", "",
              "| Target recall | Threshold | Achieved recall | FPR |", "|---|---|---|---|"]
    for name, row in operating["fpr_at_recall"].items():
        lines.append(f"| {row['target_recall']:.2f} | {row['threshold']:.3f} | "
                     f"{row['recall']:.4f} | {row['fpr']:.4f} |")
    best = operating["cost_optimal"]
    lines += ["", f"Cost-optimal threshold: {best['threshold']:.2f} "
                  f"(recall {best['recall']:.4f}, FPR {best['fpr']:.4f}).", "",
              "## E6 - latency per layer", "", "| Layer | Mean (ms) | p95 (ms) |", "|---|---|---|"]
    for name, row in sorted(payload["E6_latency"].items(), key=lambda t: -t[1]["mean_ms"]):
        lines.append(f"| {name} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
