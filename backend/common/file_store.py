from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.config import settings
from common.paths import run_dir


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def create_run_dir(run_id: str | None = None) -> tuple[str, Path]:
    rid = run_id or new_run_id()
    return rid, run_dir(rid)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
