"""API tests.  Skipped automatically when FastAPI is not installed."""
import tempfile
from pathlib import Path

import conftest  # noqa: F401

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

import pytest

pytestmark = pytest.mark.skipif(not HAVE_FASTAPI,
                                reason="fastapi/httpx not installed")


@pytest.fixture(scope="module")
def client():
    from aimes import database as db
    from aimes.config import settings

    tmp = tempfile.TemporaryDirectory()
    settings.db_url = f"sqlite:///{Path(tmp.name) / 'api.db'}"
    db._LOCAL.__dict__.pop("conn", None)
    db.init_db()
    db.create_user("tester", "hunter2hunter2")

    from aimes.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    db._LOCAL.__dict__.pop("conn", None)
    tmp.cleanup()


def test_health_is_public(client):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_dashboard_requires_login(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_api_returns_401_json_when_signed_out(client):
    assert client.get("/api/stats").status_code == 401


def test_login_and_dashboard(client):
    response = client.post("/login", data={"username": "tester", "password": "hunter2hunter2"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert client.get("/").status_code == 200
    assert client.get("/api/stats").json()["total"] >= 0


def test_upload_and_feedback_round_trip(client):
    from conftest import FIXTURES

    client.post("/login", data={"username": "tester", "password": "hunter2hunter2"})
    with open(FIXTURES / "phishing_sample.eml", "rb") as fh:
        response = client.post("/api/analyze", files={"file": ("p.eml", fh, "message/rfc822")},
                               follow_redirects=False)
    assert response.status_code == 303
    email_id = int(response.headers["location"].rsplit("/", 1)[1])
    detail = client.get(f"/api/email/{email_id}").json()
    assert detail["risk_score"] > 0

    response = client.post("/api/feedback", data={"email_id": email_id, "verdict": "incorrect",
                                                  "corrected_label": "HAM", "note": "test"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert client.get(f"/api/email/{email_id}").json()["feedback"]


def test_bad_feedback_is_rejected(client):
    client.post("/login", data={"username": "tester", "password": "hunter2hunter2"})
    response = client.post("/api/feedback", data={"email_id": 1, "verdict": "maybe"})
    assert response.status_code == 400
