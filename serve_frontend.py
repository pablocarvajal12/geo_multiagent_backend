"""
serve_frontend.py - Patch to serve the HTML frontend from FastAPI.
Import this in backend.py or run standalone.
"""
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent / "frontend"


def mount_frontend(app):
    """Mount the frontend HTML on the FastAPI app."""
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    # Mount static assets if needed
    if (FRONTEND_DIR / "static").exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
