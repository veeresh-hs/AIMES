"""Integration: file -> parser -> layers -> ensemble -> database -> alert.

Runs against whatever model version is active; if none is trained the rule
layers alone must still produce a coherent result.
"""
import tempfile
from pathlib import Path

from conftest import FIXTURES

from aimes import database as db
from aimes.alerts.dispatcher import STREAM, build_message, push_stream
from aimes.config import settings
from aimes.connector.parser import parse_file
from aimes.pipeline import Pipeline


def setup_module(module):
    module._tmp = tempfile.TemporaryDirectory()
    settings.db_url = f"sqlite:///{Path(module._tmp.name) / 'integration.db'}"
    db._LOCAL.__dict__.pop("conn", None)
    db.init_db()
    module.pipeline = Pipeline()


def teardown_module(module):
    db._LOCAL.__dict__.pop("conn", None)
    module._tmp.cleanup()


def test_phishing_scores_above_legitimate():
    phishing = pipeline.analyze(parse_file(FIXTURES / "phishing_sample.eml"))
    legitimate = pipeline.analyze(parse_file(FIXTURES / "legit_sample.eml"))
    assert phishing.risk_score > legitimate.risk_score
    assert phishing.risk_score >= 50
    assert legitimate.risk_score < 50


def test_every_layer_reports_something():
    result = pipeline.analyze(parse_file(FIXTURES / "phishing_sample.eml"))
    assert set(result.layers) == set(pipeline.detectors)
    for layer in result.layers.values():
        assert layer.score is not None or layer.error


def test_result_persists_with_layers():
    email = parse_file(FIXTURES / "attachment_sample.eml")
    result = pipeline.analyze(email)
    email_id = db.save_analysis(email, result)
    stored = db.get_email(email_id)
    assert stored["base_class"] == result.base_class
    assert len(stored["layers"]) == len(result.layers)


def test_alert_message_is_built_from_real_result():
    email = parse_file(FIXTURES / "phishing_sample.eml")
    result = pipeline.analyze(email)
    message = build_message(email, result)
    assert str(result.risk_score) in message
    assert email.sender in message
    push_stream(email, result, 1)
    assert not STREAM.empty()


def test_latency_is_measured():
    result = pipeline.analyze(parse_file(FIXTURES / "legit_sample.eml"))
    assert result.total_latency_ms > 0
    assert all(layer.latency_ms >= 0 for layer in result.layers.values())
