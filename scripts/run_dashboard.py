"""Start the dashboard.  Equivalent to:
    uvicorn aimes.api.main:app --host 127.0.0.1 --port 8000
"""
import _bootstrap  # noqa: F401

from aimes.config import settings

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aimes.api.main:app", host=settings.host, port=settings.port,
                reload=False, log_level="info")
