"""FastAPI application: dashboard, live monitor, analysis views, feedback API."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import database as db
from ..alerts.dispatcher import STREAM, dispatch
from ..config import BASE_CLASSES, LAYER_NAMES, settings
from ..connector.parser import parse_bytes
from ..explain.evidence import LAYER_LABEL, summarise
from ..explain.shap_explainer import explain_fusion, explain_lexical
from ..logging_conf import get_logger
from ..ml.registry import active_dir, load_registry
from ..monitor import MONITOR, connect_gmail
from ..pipeline import get_pipeline
from . import auth

log = get_logger(__name__)
WEB = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title=settings.app_name, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))

_RATE: dict[str, list[float]] = {}


def rate_limit(request: Request, bucket: str, limit: int = 30, window: float = 60.0) -> None:
    import time

    key = f"{bucket}:{request.client.host if request.client else 'local'}"
    now = time.time()
    hits = [t for t in _RATE.get(key, []) if now - t < window]
    if len(hits) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
    hits.append(now)
    _RATE[key] = hits


def current_user(request: Request) -> str:
    user = auth.verify(request.cookies.get(auth.COOKIE))
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    return user


def page(request: Request, name: str, **context):
    context.setdefault("user", auth.verify(request.cookies.get(auth.COOKIE)))
    context.setdefault("settings", settings)
    context.setdefault("monitor", MONITOR.status())
    context.setdefault("active_page", name.replace(".html", ""))
    return templates.TemplateResponse(request, name, context)


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    get_pipeline()
    db.purge_old_bodies()
    log.info("dashboard ready on http://%s:%d", settings.host, settings.port)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    return templates.TemplateResponse(request, "error.html",
                                      {"code": exc.status_code, "detail": exc.detail,
                                       "settings": settings, "user": None},
                                      status_code=exc.status_code)


# ----------------------------------------------------------------- auth
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return page(request, "login.html", error=None, needs_setup=db.user_count() == 0)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    rate_limit(request, "login", limit=8, window=300)
    if not db.verify_user(username, password):
        return page(request, "login.html", error="Those details did not match an account.",
                    needs_setup=db.user_count() == 0)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(auth.COOKIE, auth.issue(username), httponly=True,
                        samesite="lax", max_age=auth.MAX_AGE)
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE)
    return response


# ------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, user: str = Depends(current_user)):
    return page(request, "dashboard.html", stats=db.stats(),
                recent=db.recent_emails(12), alerts=db.recent_alerts(8))


@app.get("/live", response_class=HTMLResponse)
def live(request: Request, user: str = Depends(current_user)):
    return page(request, "live.html", recent=db.recent_emails(50))


@app.get("/stream")
async def stream(request: Request, user: str = Depends(current_user)):
    async def events():
        yield "retry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = STREAM.get_nowait()
                yield f"data: {event}\n\n"
            except Exception:
                await asyncio.sleep(1.0)
                yield ": keep-alive\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/email/{email_id}", response_class=HTMLResponse)
def email_detail(email_id: int, request: Request, user: str = Depends(current_user)):
    record = db.get_email(email_id)
    if not record:
        raise HTTPException(404, "No analysed email with that id")
    layers = []
    for row in record["layers"]:
        layers.append({**row,
                       "label": LAYER_LABEL.get(row["layer"], row["layer"]),
                       "features": json.loads(row["features"] or "{}"),
                       "evidence": json.loads(row["evidence"] or "[]")})
    scores = {row["layer"]: row["score"] for row in record["layers"]}
    model_dir = active_dir()
    contributions = explain_fusion(model_dir, scores) if model_dir else []
    terms = []
    if model_dir and record.get("raw_body"):
        terms = explain_lexical(model_dir, f"{record['subject']}\n{record['raw_body']}")
    return page(request, "email.html", email=record, layers=layers,
                tags=json.loads(record["tags"] or "[]"),
                reasons=json.loads(record["reasons"] or "[]"),
                contributions=contributions, terms=terms, classes=BASE_CLASSES)


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request, user: str = Depends(current_user)):
    return page(request, "analytics.html", stats=db.stats())


@app.get("/models", response_class=HTMLResponse)
def models(request: Request, user: str = Depends(current_user)):
    versions = db.model_versions()
    for version in versions:
        version["metrics"] = json.loads(version["metrics"] or "{}")
        version["meta"] = json.loads(version["meta"] or "{}")
    reports = {}
    for version in versions:
        path = Path(settings.model_dir) / version["version"] / "training_report.json"
        if path.exists():
            reports[version["version"]] = json.loads(path.read_text(encoding="utf-8"))
    pipeline = get_pipeline()
    layers = {name: {"loaded": detector.model_backed,
                     "label": LAYER_LABEL.get(name, name)}
              for name, detector in pipeline.detectors.items()}
    return page(request, "models.html", versions=versions, reports=reports,
                layers=layers, registry=load_registry(), layer_names=LAYER_NAMES)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: str = Depends(current_user)):
    return page(request, "settings.html", registry=load_registry())


# ------------------------------------------------------------------ api
@app.post("/api/feedback")
def api_feedback(request: Request, email_id: int = Form(...), verdict: str = Form(...),
                 corrected_label: str = Form(""), note: str = Form(""),
                 user: str = Depends(current_user)):
    rate_limit(request, "feedback", limit=60)
    if verdict not in ("correct", "incorrect"):
        raise HTTPException(400, "verdict must be 'correct' or 'incorrect'")
    if corrected_label and corrected_label not in BASE_CLASSES:
        raise HTTPException(400, f"corrected_label must be one of {BASE_CLASSES}")
    record = db.get_email(email_id)
    if not record:
        raise HTTPException(404, "No analysed email with that id")
    db.add_feedback(email_id, verdict, corrected_label or None,
                    record["base_class"], note[:500])
    return RedirectResponse(f"/email/{email_id}", status_code=303)


@app.post("/api/analyze")
async def api_analyze(request: Request, file: UploadFile = File(...),
                      user: str = Depends(current_user)):
    """Analyse an uploaded .eml file - the offline demo path."""
    rate_limit(request, "analyze", limit=20)
    raw = await file.read()
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 30 MB")
    email = parse_bytes(raw, message_id=f"upload:{file.filename}")
    result = get_pipeline().analyze(email)
    email_id = db.save_analysis(email, result)
    dispatch(email, result, email_id)
    return RedirectResponse(f"/email/{email_id}", status_code=303)


@app.get("/api/stats")
def api_stats(user: str = Depends(current_user)):
    return db.stats()


@app.get("/api/emails")
def api_emails(limit: int = 50, band: str = "", user: str = Depends(current_user)):
    return db.recent_emails(min(limit, 200), band or None)


@app.get("/api/email/{email_id}")
def api_email(email_id: int, user: str = Depends(current_user)):
    record = db.get_email(email_id)
    if not record:
        raise HTTPException(404, "No analysed email with that id")
    return record


@app.post("/api/monitor/start")
def api_monitor_start(user: str = Depends(current_user)):
    try:
        if MONITOR.provider is None:
            connect_gmail(interactive=False)
        MONITOR.start()
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return MONITOR.status()


@app.post("/api/monitor/stop")
def api_monitor_stop(user: str = Depends(current_user)):
    MONITOR.stop()
    return MONITOR.status()


@app.post("/api/monitor/poll")
def api_monitor_poll(user: str = Depends(current_user)):
    try:
        if MONITOR.provider is None:
            connect_gmail(interactive=False)
        return {"analysed": MONITOR.poll_once(), **MONITOR.status()}
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/monitor/status")
def api_monitor_status(user: str = Depends(current_user)):
    return MONITOR.status()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "mode": settings.mode,
            "model_version": get_pipeline().version or "untrained"}
