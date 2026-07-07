from pydantic import BaseModel, Field
from typing import Any


Board = list[list[int]]


class PhaseImage(BaseModel):
    key: str
    title: str
    description: str
    image_url: str | None = None


class PredictResponse(BaseModel):
    success: bool
    run_id: str
    board: Board
    confidence: list[list[float]]
    low_confidence_cells: list[dict[str, int]]
    phases: list[PhaseImage]
    message: str
    model_status: str


class SolveRequest(BaseModel):
    board: Board
    run_id: str | None = None
    original_board: Board | None = None


class SolveResponse(BaseModel):
    success: bool
    solved_board: Board | None = None
    phases: list[PhaseImage] = Field(default_factory=list)
    message: str
