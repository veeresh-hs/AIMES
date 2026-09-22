"""Analyse a .eml file from the command line - the demo that needs no mailbox.

    python scripts/analyze_eml.py tests/fixtures/phishing_sample.eml
    python scripts/analyze_eml.py samples\\*.eml --json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aimes import database as db
from aimes.connector.parser import parse_file
from aimes.explain.evidence import summarise
from aimes.pipeline import get_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help=".eml files (globs allowed)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--save", action="store_true", help="store the result in the database")
    args = ap.parse_args()

    files = [Path(p) for pattern in args.paths for p in glob.glob(pattern)]
    if not files:
        raise SystemExit("No files matched.")
    if args.save:
        db.init_db()

    pipeline = get_pipeline()
    for path in files:
        email = parse_file(path)
        result = pipeline.analyze(email)
        if args.save:
            db.save_analysis(email, result)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
            continue

        print("\n" + "=" * 72)
        print(f"{path.name}")
        print(f"From:    {email.sender_name} <{email.sender}>")
        print(f"Subject: {email.subject}")
        print("-" * 72)
        print(f"{result.base_class}  risk {result.risk_score}/100  "
              f"band {result.band}  confidence {result.confidence * 100:.0f}%")
        if result.tags:
            print("Tags:    " + ", ".join(result.tags))
        print(f"Fusion:  {result.fusion} (model {result.model_version})")
        print("\nWhy:")
        for reason in result.reasons:
            print(f"  - {reason}")
        print("\nLayer scores:")
        for name, layer in result.layers.items():
            value = "n/a " if layer.score is None else f"{layer.score * 100:5.1f}"
            note = f"  ({layer.error})" if layer.score is None else ""
            print(f"  {name:20s} {value}{note}")
        print(f"\nAnalysed in {result.total_latency_ms:.1f} ms")
