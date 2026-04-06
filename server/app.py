"""
server/app.py — FastAPI server for Code Review OpenEnv Environment

Endpoints:
    POST /reset   → start a new episode (select task by index)
    POST /step    → submit a review and receive graded feedback
    GET  /state   → current episode metadata
    GET  /health  → liveness probe
    GET  /docs    → auto-generated Swagger UI (built-in)
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so `models` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    StepResult,
)
from server.environment import CodeReviewEnvironment


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response helpers
# ══════════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_index: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Task to load: 0=easy, 1=medium, 2=hard.",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "ok"
    environment: str = "code_review"
    version: str = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# App setup
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Code Review OpenEnv",
    description=(
        "An OpenEnv-compliant environment that evaluates AI agents on "
        "their ability to perform code reviews — detecting bugs, suggesting "
        "fixes, and explaining issues."
    ),
    version="1.0.0",
)

# Single shared environment instance
env = CodeReviewEnvironment()


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/reset", response_model=StepResult, summary="Reset the environment")
async def reset(request: ResetRequest) -> StepResult:
    """
    Initialize a new episode.

    Selects a task by index and returns the initial observation
    containing the code snippet and review instructions.
    """
    try:
        result = env.reset(task_index=request.task_index)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResult, summary="Submit a review")
async def step(action: CodeReviewAction) -> StepResult:
    """
    Submit the agent's code review for grading.

    Returns the reward (0.0–1.0), detailed feedback, and done=True
    (single-step episodes).
    """
    if not action.review.strip():
        raise HTTPException(status_code=422, detail="Review text cannot be empty.")
    try:
        result = env.step(action)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=CodeReviewState, summary="Get current state")
async def state() -> CodeReviewState:
    """Return the current episode metadata (difficulty, expected issues, etc.)."""
    return env.state


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse()


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint (for `python server/app.py`)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
