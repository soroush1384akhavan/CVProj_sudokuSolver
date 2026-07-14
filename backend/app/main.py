from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import PredictResponse, SolveRequest, SolveResponse
from pipeline.sudoku_pipeline import run_prediction_pipeline, run_solve_pipeline

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(settings.runs_dir)), name="media")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    language: Literal["en", "fa"] = Form("en"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(image.filename or "upload.png").suffix or ".png"
    upload_path = settings.uploads_dir / f"upload_{Path(image.filename or 'image').stem}{suffix}"

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        return run_prediction_pipeline(upload_path, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    try:
        return run_solve_pipeline(request.board, request.run_id, request.original_board)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc