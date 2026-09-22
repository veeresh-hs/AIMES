"""SQLite persistence layer (stdlib sqlite3 - no ORM, no external database).

Privacy: raw message bodies are stored only when AIMES_STORE_RAW_BODY=true and
are purged after AIMES_BODY_RETENTION_DAYS.  By default the database keeps
metadata, extracted features, scores and a SHA-256 of the body.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .config import settings
from .logging_conf import get_logger

log = get_logger(__name__)
_LOCAL = threading.local()

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY,
    message_id TEXT UNIQUE NOT NULL,
    thread_id TEXT,
    received_at TEXT,
    analyzed_at TEXT NOT NULL,
    sender TEXT,
    sender_domain TEXT,
    sender_name TEXT,
    subject TEXT,
    n_urls INTEGER DEFAULT 0,
    n_attachments INTEGER DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    body_sha256 TEXT,
    raw_body TEXT,
    risk_score INTEGER NOT NULL,
    band TEXT NOT NULL,
    base_class TEXT NOT NULL,
    tags TEXT,
    confidence REAL,
    fusion TEXT,
    model_version TEXT,
    latency_ms REAL,
    reasons TEXT
);
CREATE INDEX IF NOT EXISTS idx_emails_analyzed ON emails(analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_emails_band ON emails(band);

CREATE TABLE IF NOT EXISTS layer_results (
    id INTEGER PRIMARY KEY,
    email_id INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    layer TEXT NOT NULL,
    score REAL,
    confidence REAL,
    available INTEGER,
    latency_ms REAL,
    features TEXT,
    evidence TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_layer_email ON layer_results(email_id);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    delivered INTEGER DEFAULT 0,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    email_id INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    verdict TEXT NOT NULL,            -- correct | incorrect
    corrected_label TEXT,             -- HAM | PROMOTIONAL_SPAM | PHISHING
    predicted_label TEXT,
    note TEXT,
    used_in_version TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    active INTEGER DEFAULT 0,
    metrics TEXT,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS monitor_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def db_path() -> Path:
    url = settings.db_url
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///"):])
    return Path(url)


def connect() -> sqlite3.Connection:
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=20, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _LOCAL.conn = conn
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    conn = connect()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    log.info("database ready at %s", db_path())


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ auth
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 240_000)
    return digest.hex(), salt


def create_user(username: str, password: str) -> None:
    digest, salt = hash_password(password)
    with cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO users(username, password_hash, salt, created_at)"
            " VALUES (?,?,?,?)", (username, digest, salt, now()))


def verify_user(username: str, password: str) -> bool:
    with cursor() as cur:
        cur.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    if not row:
        hash_password(password)  # constant-ish work so timing does not leak
        return False
    digest, _ = hash_password(password, row["salt"])
    return hmac.compare_digest(digest, row["password_hash"])


def user_count() -> int:
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM users")
        return int(cur.fetchone()["n"])


# ---------------------------------------------------------------- emails
def email_exists(message_id: str) -> bool:
    with cursor() as cur:
        cur.execute("SELECT 1 FROM emails WHERE message_id=?", (message_id,))
        return cur.fetchone() is not None


def save_analysis(email, result) -> int:
    """Persist one analysed email and its per-layer results."""
    raw_body = email.clean_text[:100_000] if settings.store_raw_body else None
    with cursor() as cur:
        cur.execute(
            """INSERT OR REPLACE INTO emails
               (message_id, thread_id, received_at, analyzed_at, sender, sender_domain,
                sender_name, subject, n_urls, n_attachments, size_bytes, body_sha256,
                raw_body, risk_score, band, base_class, tags, confidence, fusion,
                model_version, latency_ms, reasons)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (email.message_id, email.thread_id,
             email.date.isoformat() if email.date else None, now(),
             email.sender, email.sender_domain, email.sender_name, email.subject,
             len(email.urls), len(email.attachments), email.size_bytes, email.body_sha256,
             raw_body, result.risk_score, result.band, result.base_class,
             json.dumps(result.tags), result.confidence, result.fusion,
             result.model_version, result.total_latency_ms, json.dumps(result.reasons)))
        email_id = int(cur.lastrowid)
        cur.execute("DELETE FROM layer_results WHERE email_id=?", (email_id,))
        for name, layer in result.layers.items():
            cur.execute(
                """INSERT INTO layer_results
                   (email_id, layer, score, confidence, available, latency_ms,
                    features, evidence, error)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (email_id, name, layer.score, layer.confidence, int(layer.available),
                 layer.latency_ms, json.dumps(layer.features, default=str),
                 json.dumps(layer.evidence), layer.error))
    return email_id


def recent_emails(limit: int = 50, band: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM emails"
    params: list[Any] = []
    if band:
        query += " WHERE band=?"
        params.append(band)
    query += " ORDER BY analyzed_at DESC, id DESC LIMIT ?"
    params.append(limit)
    with cursor() as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def get_email(email_id: int) -> Optional[dict]:
    with cursor() as cur:
        cur.execute("SELECT * FROM emails WHERE id=?", (email_id,))
        row = cur.fetchone()
        if not row:
            return None
        record = dict(row)
        cur.execute("SELECT * FROM layer_results WHERE email_id=? ORDER BY score DESC", (email_id,))
        record["layers"] = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM feedback WHERE email_id=? ORDER BY id DESC", (email_id,))
        record["feedback"] = [dict(r) for r in cur.fetchall()]
    return record


def stats() -> dict:
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) n, AVG(risk_score) avg_risk, AVG(latency_ms) avg_latency FROM emails")
        base = dict(cur.fetchone())
        cur.execute("SELECT base_class, COUNT(*) n FROM emails GROUP BY base_class")
        by_class = {r["base_class"]: r["n"] for r in cur.fetchall()}
        cur.execute("SELECT band, COUNT(*) n FROM emails GROUP BY band")
        by_band = {r["band"]: r["n"] for r in cur.fetchall()}
        cur.execute("SELECT COUNT(*) n FROM alerts")
        alerts = int(cur.fetchone()["n"])
        cur.execute("SELECT verdict, COUNT(*) n FROM feedback GROUP BY verdict")
        feedback = {r["verdict"]: r["n"] for r in cur.fetchall()}
        cur.execute("""SELECT substr(analyzed_at,1,10) d, COUNT(*) n,
                              SUM(CASE WHEN band IN ('HIGH_RISK','CRITICAL') THEN 1 ELSE 0 END) threats
                       FROM emails GROUP BY d ORDER BY d DESC LIMIT 14""")
        timeline = [dict(r) for r in cur.fetchall()][::-1]
        cur.execute("""SELECT CAST(risk_score/10 AS INT)*10 bucket, COUNT(*) n
                       FROM emails GROUP BY bucket ORDER BY bucket""")
        histogram = {int(r["bucket"]): r["n"] for r in cur.fetchall()}
    return {"total": base["n"] or 0,
            "avg_risk": round(base["avg_risk"] or 0, 1),
            "avg_latency_ms": round(base["avg_latency"] or 0, 1),
            "by_class": by_class, "by_band": by_band, "alerts": alerts,
            "feedback": feedback, "timeline": timeline, "risk_histogram": histogram}


# ---------------------------------------------------------------- alerts
def record_alert(email_id: int, channel: str, delivered: bool, detail: str = "") -> None:
    with cursor() as cur:
        cur.execute("INSERT INTO alerts(email_id, created_at, channel, delivered, detail)"
                    " VALUES (?,?,?,?,?)", (email_id, now(), channel, int(delivered), detail))


def recent_alerts(limit: int = 25) -> list[dict]:
    with cursor() as cur:
        cur.execute("""SELECT a.*, e.subject, e.sender, e.risk_score, e.base_class
                       FROM alerts a LEFT JOIN emails e ON e.id=a.email_id
                       ORDER BY a.id DESC LIMIT ?""", (limit,))
        return [dict(r) for r in cur.fetchall()]


# -------------------------------------------------------------- feedback
def add_feedback(email_id: int, verdict: str, corrected_label: Optional[str],
                 predicted_label: str, note: str = "") -> None:
    with cursor() as cur:
        cur.execute("""INSERT INTO feedback
                       (email_id, created_at, verdict, corrected_label, predicted_label, note)
                       VALUES (?,?,?,?,?,?)""",
                    (email_id, now(), verdict, corrected_label, predicted_label, note))


def unused_feedback() -> list[dict]:
    with cursor() as cur:
        cur.execute("""SELECT f.*, e.subject, e.body_sha256, e.raw_body, e.message_id
                       FROM feedback f JOIN emails e ON e.id=f.email_id
                       WHERE f.used_in_version IS NULL""")
        return [dict(r) for r in cur.fetchall()]


def mark_feedback_used(ids: Iterable[int], version: str) -> None:
    with cursor() as cur:
        cur.executemany("UPDATE feedback SET used_in_version=? WHERE id=?",
                        [(version, i) for i in ids])


# --------------------------------------------------------------- monitor
def get_state(key: str, default: str = "") -> str:
    with cursor() as cur:
        cur.execute("SELECT value FROM monitor_state WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def set_state(key: str, value: str) -> None:
    with cursor() as cur:
        cur.execute("INSERT OR REPLACE INTO monitor_state(key, value) VALUES (?,?)", (key, value))


def sync_model_versions() -> None:
    from .ml.registry import load_registry

    data = load_registry()
    with cursor() as cur:
        for version, info in data.get("versions", {}).items():
            cur.execute("""INSERT OR REPLACE INTO model_versions
                           (version, created_at, active, metrics, meta) VALUES (?,?,?,?,?)""",
                        (version, info.get("created", now()),
                         int(data.get("active") == version),
                         json.dumps(info.get("metrics", {})),
                         json.dumps(info.get("meta", {}))))


def model_versions() -> list[dict]:
    sync_model_versions()
    with cursor() as cur:
        cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def purge_old_bodies() -> int:
    if settings.body_retention_days <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.body_retention_days)).isoformat()
    with cursor() as cur:
        cur.execute("UPDATE emails SET raw_body=NULL WHERE raw_body IS NOT NULL AND analyzed_at < ?",
                    (cutoff,))
        return cur.rowcount
