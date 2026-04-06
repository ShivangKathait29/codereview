"""
client.py — HTTP client for Code Review OpenEnv Environment

Provides a typed Python interface to communicate with the FastAPI server.
Usage:
    client = CodeReviewClient("http://localhost:7860")
    result = client.reset(task_index=0)
    result = client.step(review="The function has a division-by-zero bug.")
"""

from __future__ import annotations

import requests
from typing import Optional

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    StepResult,
)


class CodeReviewClient:
    """
    Typed HTTP client for the Code Review OpenEnv environment.

    All methods deserialize JSON responses into Pydantic models.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30) -> None:
        """
        Args:
            base_url: Root URL of the environment server (no trailing slash).
            timeout:  HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_index: int = 0) -> StepResult:
        """
        Start a new episode.

        Args:
            task_index: 0=easy, 1=medium, 2=hard.

        Returns:
            StepResult with the initial observation (code + instructions).
        """
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_index": task_index},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return StepResult(**response.json())

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, review: str) -> StepResult:
        """
        Submit a code review for grading.

        Args:
            review: The agent's code review text.

        Returns:
            StepResult with reward, feedback, and done flag.
        """
        action = CodeReviewAction(review=review)
        response = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return StepResult(**response.json())

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def state(self) -> CodeReviewState:
        """Retrieve current episode metadata."""
        response = requests.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return CodeReviewState(**response.json())

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Check if the server is alive."""
        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def __repr__(self) -> str:
        return f"CodeReviewClient(base_url={self.base_url!r})"
