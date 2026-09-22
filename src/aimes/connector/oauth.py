"""Gmail OAuth 2.0.

Read-only by default.  The delete/trash scopes are never requested, so the
application is structurally incapable of removing your mail.  You are never
asked for your Gmail password.
"""
from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..logging_conf import get_logger
from .token_store import load_token, save_token

log = get_logger(__name__)

SCOPES_READONLY = ["https://www.googleapis.com/auth/gmail.readonly"]
SCOPES_MODIFY = ["https://www.googleapis.com/auth/gmail.modify"]  # labels only


def required_scopes() -> list[str]:
    return SCOPES_MODIFY if settings.gmail_allow_modify else SCOPES_READONLY


def get_credentials(interactive: bool = True):
    """Returns google.oauth2.credentials.Credentials or raises with guidance."""
    try:
        from google.auth.transport.requests import Request  # type: ignore
        from google.oauth2.credentials import Credentials  # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gmail support needs the Google client libraries:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    scopes = required_scopes()
    creds = None
    stored = load_token()
    if stored:
        try:
            creds = Credentials.from_authorized_user_info(stored, scopes)
        except Exception as exc:
            log.warning("stored token unusable (%s); re-authorising", type(exc).__name__)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_token(_to_dict(creds))
            return creds
        except Exception as exc:
            log.warning("token refresh failed (%s); starting a new consent flow", exc)

    if not interactive:
        raise RuntimeError("No valid Gmail token. Run: python scripts/connect_gmail.py")

    creds_file = Path(settings.gmail_credentials)
    if not creds_file.exists():
        raise RuntimeError(
            f"Missing {creds_file}.\n"
            "Google Cloud Console -> APIs & Services -> Credentials ->\n"
            "Create credentials -> OAuth client ID -> Desktop app -> Download JSON,\n"
            f"then save it as {creds_file}."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
    creds = flow.run_local_server(port=0, prompt="consent",
                                  authorization_prompt_message=
                                  "Opening your browser to authorise read access to Gmail...")
    save_token(_to_dict(creds))
    log.info("Gmail authorised with scopes: %s", scopes)
    return creds


def _to_dict(creds) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
