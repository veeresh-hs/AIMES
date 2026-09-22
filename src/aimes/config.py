"""Central configuration for AIMES.

All settings come from environment variables (optionally loaded from a .env
file at the project root).  Nothing here requires a paid service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parents[1]


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------- runtime profile
def detect_mode() -> str:
    """Pick FAST or ADVANCED based on available RAM unless pinned by env."""
    pinned = os.environ.get("AIMES_MODE", "").strip().upper()
    if pinned in {"FAST", "ADVANCED"}:
        return pinned
    try:  # psutil is optional
        import psutil

        gb = psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        try:
            gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        except (ValueError, AttributeError, OSError):
            gb = 8.0
    return "ADVANCED" if gb >= 8 else "FAST"


@dataclass
class Settings:
    # runtime
    mode: str = field(default_factory=detect_mode)
    app_name: str = "AIMES - AI Multi-layer Email Security"

    # storage
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    log_dir: Path = PROJECT_ROOT / "logs"
    secrets_dir: Path = PROJECT_ROOT / "secrets"
    db_url: str = _env("AIMES_DB_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'aimes.db').as_posix()}")

    # dashboard / api
    host: str = _env("AIMES_HOST", "127.0.0.1")
    port: int = _env_int("AIMES_PORT", 8000)
    session_secret: str = _env("AIMES_SESSION_SECRET", "change-me-in-.env")
    admin_user: str = _env("AIMES_ADMIN_USER", "admin")
    admin_password: str = _env("AIMES_ADMIN_PASSWORD", "")

    # gmail
    gmail_credentials: Path = Path(_env("AIMES_GMAIL_CREDENTIALS", str(PROJECT_ROOT / "secrets" / "credentials.json")))
    gmail_token: Path = Path(_env("AIMES_GMAIL_TOKEN", str(PROJECT_ROOT / "secrets" / "token.enc")))
    token_key: str = _env("AIMES_TOKEN_KEY", "")
    gmail_allow_modify: bool = _env_bool("AIMES_GMAIL_ALLOW_MODIFY", False)
    gmail_label_high_risk: bool = _env_bool("AIMES_GMAIL_LABEL_HIGH_RISK", False)
    poll_seconds: int = _env_int("AIMES_POLL_SECONDS", 20)
    max_messages_per_poll: int = _env_int("AIMES_MAX_MESSAGES_PER_POLL", 25)

    # privacy
    store_raw_body: bool = _env_bool("AIMES_STORE_RAW_BODY", False)
    body_retention_days: int = _env_int("AIMES_BODY_RETENTION_DAYS", 7)

    # alerts
    alert_threshold: int = _env_int("AIMES_ALERT_THRESHOLD", 60)
    desktop_alerts: bool = _env_bool("AIMES_DESKTOP_ALERTS", True)
    telegram_token: str = _env("AIMES_TELEGRAM_TOKEN", "")
    telegram_chat_id: str = _env("AIMES_TELEGRAM_CHAT_ID", "")

    # models
    semantic_model: str = _env("AIMES_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    transformer_model: str = _env("AIMES_TRANSFORMER_MODEL", "distilbert-base-uncased")
    max_body_chars: int = _env_int("AIMES_MAX_BODY_CHARS", 20000)

    # risk bands (configurable, calibrated in Phase 12 experiments)
    band_suspicious: int = _env_int("AIMES_BAND_SUSPICIOUS", 30)
    band_high: int = _env_int("AIMES_BAND_HIGH", 60)
    band_critical: int = _env_int("AIMES_BAND_CRITICAL", 80)

    # ensemble fusion: "stacking" falls back to "weighted" if no meta-model saved
    fusion: str = _env("AIMES_FUSION", "stacking")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.model_dir, self.log_dir, self.secrets_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()

# Default ensemble weights.  These are *priors* only: scripts/train_all.py
# re-fits them on validation data and writes the fitted values to the model
# registry.  Never present these numbers as experimental results.
DEFAULT_WEIGHTS: dict[str, float] = {
    "lexical": 0.26,
    "transformer": 0.16,
    "semantic": 0.12,
    "anomaly": 0.06,
    "url": 0.16,
    "header": 0.10,
    "attachment": 0.06,
    "impersonation": 0.05,
    "social_engineering": 0.03,
}

LAYER_NAMES: list[str] = list(DEFAULT_WEIGHTS.keys())

BASE_CLASSES: list[str] = ["HAM", "PROMOTIONAL_SPAM", "PHISHING"]

THREAT_TAGS: list[str] = [
    "SUSPICIOUS_URL",
    "SUSPICIOUS_ATTACHMENT",
    "IMPERSONATION",
    "SOCIAL_ENGINEERING",
    "FINANCIAL_SCAM",
    "ANOMALOUS",
]
