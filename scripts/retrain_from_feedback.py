"""Feedback-driven retraining.

Deliberately NOT triggered by every correction: feedback is collected, checked,
merged with the base corpora, and a candidate model is trained and evaluated.
The candidate replaces the active model only if the promotion gate passes.

    python scripts/retrain_from_feedback.py --dry-run
    python scripts/retrain_from_feedback.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from aimes import database as db
from aimes.config import BASE_CLASSES, settings
from aimes.logging_conf import get_logger
from aimes.ml import registry

log = get_logger("retrain")
MIN_FEEDBACK = 20


def collect() -> list[dict]:
    rows = db.unused_feedback()
    usable = []
    for row in rows:
        if row["verdict"] == "correct":
            continue  # confirmations do not add a new training signal
        label = row["corrected_label"]
        if label not in BASE_CLASSES:
            log.info("skipping feedback %s: no valid corrected label", row["id"])
            continue
        if not row.get("raw_body"):
            log.info("skipping feedback %s: body not retained (AIMES_STORE_RAW_BODY is off)",
                     row["id"])
            continue
        usable.append(row)
    return usable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min", type=int, default=MIN_FEEDBACK)
    args = ap.parse_args()

    db.init_db()
    usable = collect()
    print(f"{len(usable)} usable corrections since the last retraining run "
          f"(minimum {args.min}).")
    if not usable:
        print("Nothing to learn from yet. Corrections need AIMES_STORE_RAW_BODY=true "
              "so the text is available for retraining.")
        return
    if len(usable) < args.min and not args.dry_run:
        print("Below the threshold; collect more feedback before retraining.")
        return

    corrections = Path(settings.data_dir) / "raw" / "feedback_csv"
    corrections.mkdir(parents=True, exist_ok=True)
    import csv

    path = corrections / "user_feedback.csv"
    existing = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            existing = {row["message_id"] for row in csv.DictReader(fh)}
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["message_id", "subject", "body", "label"])
        if not existing:
            writer.writeheader()
        for row in usable:
            if row["message_id"] in existing:
                continue
            writer.writerow({"message_id": row["message_id"], "subject": row["subject"] or "",
                             "body": row["raw_body"] or "", "label": row["corrected_label"]})
    print(f"Feedback written to {path}; it is picked up as a CSV corpus on the next run.")

    if args.dry_run:
        print("Dry run: stopping before training.")
        return

    print("Training a candidate model...")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("train_all.py"))],
                            check=False)
    if result.returncode != 0:
        raise SystemExit("Training failed; the active model is unchanged.")

    db.mark_feedback_used([row["id"] for row in usable], registry.active_version() or "candidate")
    print(f"Done. Active model version is now {registry.active_version()}.")
    print("If the candidate did not clear the promotion gate, the previous model is still live "
          "- that is the intended behaviour, not an error.")


if __name__ == "__main__":
    main()
