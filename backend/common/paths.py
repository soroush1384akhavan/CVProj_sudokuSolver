from pathlib import Path
from app.config import settings


def public_url_for_run_file(run_id: str, filename: str) -> str:
    """Return a URL served by FastAPI StaticFiles."""
    return f"/media/{run_id}/{filename}"


def run_dir(run_id: str) -> Path:
    path = settings.runs_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
