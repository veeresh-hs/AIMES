"""Metrics and comparison tables.

Every number the dashboard and the report show comes from this module running
over a held-out split.  No metric is ever written by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

THREAT = "threat"


def to_binary(labels: Iterable[str]) -> np.ndarray:
    """1 = anything the user should be warned about, 0 = legitimate mail."""
    return np.array([0 if label == "HAM" else 1 for label in labels], dtype=int)


def binary_metrics(y_true_labels: Sequence[str], y_score: Sequence[float],
                   threshold: float = 0.5) -> dict:
    y_true = to_binary(y_true_labels)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
               fnr=float(fn / (fn + tp)) if (fn + tp) else 0.0)
    return out


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                       labels: Sequence[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            label: float(score) for label, score in
            zip(labels, f1_score(y_true, y_pred, average=None, labels=list(labels), zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels": list(labels),
    }


def fpr_at_recall(y_true_labels: Sequence[str], y_score: Sequence[float],
                  target_recall: float = 0.95) -> dict:
    """Operating point that matters in production: how much legitimate mail do
    we flag if we insist on catching 95% of threats?"""
    y_true = to_binary(y_true_labels)
    y_score = np.asarray(y_score, dtype=float)
    best = {"target_recall": target_recall, "threshold": 1.0, "recall": 0.0, "fpr": 0.0}
    for threshold in np.unique(np.round(y_score, 4)):
        pred = (y_score >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= target_recall:
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            if fpr <= best["fpr"] or best["recall"] < target_recall:
                best = {"target_recall": target_recall, "threshold": float(threshold),
                        "recall": float(recall), "fpr": float(fpr)}
    return best


def comparison_table(results: dict[str, dict]) -> str:
    """Plain-text table for the report and the console."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "latency_ms"]
    width = max((len(k) for k in results), default=10) + 2
    head = "Model".ljust(width) + "".join(c.replace("_", " ").title().rjust(12) for c in cols)
    lines = [head, "-" * len(head)]
    for name, metrics in results.items():
        row = name.ljust(width)
        for col in cols:
            value = metrics.get(col)
            row += ("-" if value is None else f"{value:.4f}").rjust(12)
        lines.append(row)
    return "\n".join(lines)


def save_report(results: dict, path: Path, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": note, "results": results}
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def plot_results(results: dict[str, dict], out_dir: Path) -> list[Path]:
    """Bar charts for the report.  Skipped silently if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric in ("f1", "roc_auc", "fpr", "latency_ms"):
        names = [n for n in results if results[n].get(metric) is not None]
        values = [results[n][metric] for n in names]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(names, values, color="#2f6f6b")
        ax.set_title(f"{metric.replace('_', ' ').title()} by model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"compare_{metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written
