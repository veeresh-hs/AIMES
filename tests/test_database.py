"""Database, feedback and model-registry behaviour (uses a temporary DB)."""
import os
import tempfile
from pathlib import Path

import conftest  # noqa: F401

from aimes import database as db
from aimes.config import settings


def setup_module(module):
    module._tmp = tempfile.TemporaryDirectory()
    settings.db_url = f"sqlite:///{Path(module._tmp.name) / 'test.db'}"
    db._LOCAL.__dict__.pop("conn", None)
    db.init_db()


def teardown_module(module):
    db._LOCAL.__dict__.pop("conn", None)
    module._tmp.cleanup()


def _analysed():
    from aimes.connector.parser import parse_file
    from aimes.ensemble.risk_engine import RiskEngine
    from aimes.schema import LayerResult
    from conftest import FIXTURES

    email = parse_file(FIXTURES / "phishing_sample.eml")
    layers = {"url": LayerResult("url", 0.9, 0.8, evidence=["bad link"]),
              "header": LayerResult("header", 0.7, 0.7, evidence=["spf fail"]),
              "lexical": LayerResult("lexical", None, 0.0, available=False,
                                     error="no model")}
    return email, RiskEngine().combine(layers, message_id=email.message_id)


def test_password_hashing_round_trip():
    db.create_user("tester", "correct horse battery")
    assert db.verify_user("tester", "correct horse battery")
    assert not db.verify_user("tester", "wrong password")
    assert not db.verify_user("nobody", "whatever")


def test_save_and_read_analysis():
    email, result = _analysed()
    email_id = db.save_analysis(email, result)
    record = db.get_email(email_id)
    assert record["risk_score"] == result.risk_score
    assert len(record["layers"]) == 3
    assert db.email_exists(email.message_id)


def test_body_not_stored_by_default():
    settings.store_raw_body = False
    email, result = _analysed()
    email_id = db.save_analysis(email, result)
    assert db.get_email(email_id)["raw_body"] is None


def test_feedback_lifecycle():
    email, result = _analysed()
    email_id = db.save_analysis(email, result)
    db.add_feedback(email_id, "incorrect", "HAM", result.base_class, "false alarm")
    pending = db.unused_feedback()
    assert any(row["email_id"] == email_id for row in pending)
    db.mark_feedback_used([row["id"] for row in pending], "v9")
    assert not db.unused_feedback()


def test_stats_are_computed_not_hardcoded():
    before = db.stats()["total"]
    email, result = _analysed()
    db.save_analysis(email, result)
    assert db.stats()["total"] >= before


def test_promotion_gate():
    from aimes.ml.registry import should_promote

    ok, _ = should_promote({"macro_f1": 0.90, "fpr": 0.02}, None)
    assert ok
    ok, _ = should_promote({"macro_f1": 0.90, "fpr": 0.02}, {"macro_f1": 0.91, "fpr": 0.02})
    assert not ok
    ok, _ = should_promote({"macro_f1": 0.95, "fpr": 0.09}, {"macro_f1": 0.90, "fpr": 0.02})
    assert not ok  # better F1 but worse false positives must not ship
    ok, _ = should_promote({"macro_f1": 0.95, "fpr": 0.02}, {"macro_f1": 0.90, "fpr": 0.02})
    assert ok
