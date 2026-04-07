"""
models.py — Data models for Code Review OpenEnv Environment

Defines the typed contracts between agent and environment:
  - CodeReviewAction: what the agent sends (a review string)
  - CodeReviewObservation: what the environment returns (code, instructions, feedback)
  - CodeReviewState: internal metadata (expected issues, difficulty, task index)
  - StepResult: transition tuple (observation, reward, done)
"""

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Action — the agent submits a code review
# ──────────────────────────────────────────────────────────────────────────────
class CodeReviewAction(BaseModel):
    """Action sent by the agent: a free-text code review."""
    review: str = Field(
        ...,
        description="The agent's code review text analyzing the provided code snippet.",
        min_length=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Observation — what the agent sees
# ──────────────────────────────────────────────────────────────────────────────
class CodeReviewObservation(BaseModel):
    """Observation returned to the agent after reset() or step()."""
    code: str = Field(
        default="",
        description="The code snippet to review.",
    )
    instructions: str = Field(
        default="",
        description="Instructions for the review task.",
    )
    feedback: str = Field(
        default="",
        description="Feedback from the grader after a step (empty on reset).",
    )


# ──────────────────────────────────────────────────────────────────────────────
# State — episode metadata (not directly visible to the agent)
# ──────────────────────────────────────────────────────────────────────────────
class CodeReviewState(BaseModel):
    """Internal state tracking expected issues and difficulty."""
    expected_issues: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases the grader looks for in the review.",
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty of the current task: easy | medium | hard.",
    )
    task_index: int = Field(
        default=0,
        description="Index of the current task (0–2).",
    )
    variant_id: str = Field(
        default="",
        description="Title/ID of the specific randomized task variant.",
    )
    done: bool = Field(
        default=False,
        description="Whether the current episode is complete.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# StepResult — returned by step() and reset()
# ──────────────────────────────────────────────────────────────────────────────
class StepResult(BaseModel):
    """Transition result from an environment step."""
    observation: CodeReviewObservation
    reward: float = Field(
        default=0.0,
        description="Scalar reward in [0.0, 1.0] (with possible penalties).",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    details: Optional[dict[str, float]] = Field(
        default=None,
        description="Detailed dictionary breakdown of sub-scores (issue, fix, explanation) provided natively by the grader.",
    )
