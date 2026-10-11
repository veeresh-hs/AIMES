"""Start the dashboard.  Equivalent to:
    uvicorn aimes.api.main:app --host 0.0.0.0 --port 8000
"""
import _bootstrap  # noqa: F401

from aimes.config import settings

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aimes.api.main:app", host="0.0.0.0", port=settings.port,
                reload=False, log_level="info")