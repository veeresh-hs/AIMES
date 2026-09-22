"""Splitting and de-duplication.

Order matters: de-duplicate BEFORE splitting, otherwise a near-copy of a test
message can sit in the training set and every metric is inflated.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

from ..logging_conf import get_logger
from .datasets import Sample

log = get_logger(__name__)


@dataclass
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    strategy: str = "random"
    cross_corpus: list[Sample] | None = None

    def counts(self) -> dict[str, dict[str, int]]:
        return {name: dict(Counter(s.label for s in part))
                for name, part in (("train", self.train), ("val", self.val),
                                   ("test", self.test),
                                   ("cross_corpus", self.cross_corpus or []))}


def _norm_key(sample: Sample) -> str:
    text = re.sub(r"\W+", " ", sample.email.combined_text.lower()).strip()
    return text[:2000]


def deduplicate(samples: list[Sample]) -> list[Sample]:
    seen: set[str] = set()
    out: list[Sample] = []
    for sample in samples:
        if not sample.email.combined_text.strip():
            continue
        key = _norm_key(sample)
        if len(key) < 20 or key in seen:
            continue
        seen.add(key)
        out.append(sample)
    log.info("de-duplication: %d -> %d samples", len(samples), len(out))
    return out


def stratified_split(samples: list[Sample], ratios=(0.7, 0.15, 0.15), seed: int = 42):
    rng = random.Random(seed)
    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    train, val, test = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * (ratios[0] + ratios[1]))
        train += group[:n_train]
        val += group[n_train:n_val]
        test += group[n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def time_split(samples: list[Sample], ratios=(0.7, 0.15, 0.15)):
    """Older mail trains, newer mail tests - the realistic deployment setting."""
    dated = [s for s in samples if s.date is not None]
    if len(dated) < 0.8 * len(samples):
        return None
    dated.sort(key=lambda s: s.date.replace(tzinfo=None))
    n = len(dated)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return dated[:a], dated[a:b], dated[b:]


def build_split(samples: list[Sample], strategy: str = "auto",
                holdout_source: str | None = "nazario", seed: int = 42) -> Split:
    """holdout_source is kept out of train/val/test entirely and used as the
    cross-corpus generalisation test (train on sources A+B, test on C)."""
    samples = deduplicate(samples)

    cross: list[Sample] = []
    if holdout_source:
        cross = [s for s in samples if s.source == holdout_source]
        if cross and len({s.source for s in samples}) > 1:
            samples = [s for s in samples if s.source != holdout_source]
        else:
            cross = []

    chosen = strategy
    parts = None
    if strategy in ("auto", "time"):
        parts = time_split(samples)
        chosen = "time" if parts else "random"
    if parts is None:
        parts = stratified_split(samples, seed=seed)
        chosen = "random"

    split = Split(*parts, strategy=chosen, cross_corpus=cross or None)
    log.info("split strategy=%s counts=%s", chosen, split.counts())
    return split
