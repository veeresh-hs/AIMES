"""Feature extraction, fusion arithmetic and threshold behaviour."""
from conftest import FIXTURES

from aimes.config import LAYER_NAMES
from aimes.connector.parser import from_plain_text, parse_file
from aimes.ensemble.risk_engine import RiskEngine
from aimes.ensemble.thresholds import band_for
from aimes.ml.features import FEATURE_NAMES, feature_row, tabular_features
from aimes.schema import LayerResult


def layers(**scores):
    return {name: LayerResult(name=name, score=scores.get(name), confidence=0.8,
                              available=scores.get(name) is not None)
            for name in LAYER_NAMES}


def test_feature_row_length_matches_names():
    row = feature_row(parse_file(FIXTURES / "phishing_sample.eml"))
    assert len(row) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in row)


def test_features_capture_structure():
    features = tabular_features(parse_file(FIXTURES / "attachment_sample.eml"))
    assert features["n_attachments"] == 1
    assert features["has_exec_attach"] == 1.0


def test_weighted_fusion_renormalises_over_available_layers():
    engine = RiskEngine()
    both = engine.weighted_score(layers(lexical=1.0, url=1.0))
    one = engine.weighted_score(layers(lexical=1.0))
    assert abs(both - 1.0) < 1e-9
    assert abs(one - 1.0) < 1e-9  # a single available layer is not diluted by absent ones


def test_missing_layers_do_not_count_as_zero():
    engine = RiskEngine()
    absent = engine.weighted_score(layers(lexical=1.0))
    explicit_zero = engine.weighted_score(layers(lexical=1.0, url=0.0, header=0.0))
    assert absent > explicit_zero


def test_combine_produces_band_and_tags():
    engine = RiskEngine()
    result = engine.combine(layers(lexical=0.95, url=0.9, header=0.8,
                                   impersonation=0.9, social_engineering=0.7))
    assert result.risk_score > 60
    assert result.band in ("HIGH_RISK", "CRITICAL")
    assert "SUSPICIOUS_URL" in result.tags and "IMPERSONATION" in result.tags
    assert result.reasons


def test_combine_handles_all_layers_abstaining():
    engine = RiskEngine()
    result = engine.combine(layers())
    assert result.risk_score == 0 and result.band == "SAFE"


def test_bands_are_monotonic():
    assert band_for(5) == "SAFE"
    assert band_for(45) == "SUSPICIOUS"
    assert band_for(70) == "HIGH_RISK"
    assert band_for(95) == "CRITICAL"


def test_confidence_falls_when_layers_disagree():
    engine = RiskEngine()
    agree = engine.combine(layers(lexical=0.9, url=0.9, header=0.9))
    disagree = engine.combine(layers(lexical=0.9, url=0.1, header=0.9))
    assert agree.confidence > disagree.confidence
