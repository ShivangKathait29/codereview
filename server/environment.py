from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    StepResult,
)

# ═════════════════════════════════════════════════════════════════════
# Task + Scoring Definitions
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ScoringRule:
    name: str
    keywords: list[str]
    weight: float


@dataclass
class Task:
    title: str
    difficulty: str
    code: str
    instructions: str
    expected_issues: list[str]
    scoring_rules: list[ScoringRule]


# (Keep your TASKS exactly as they are — no change needed)
# -------------------------------------------------------

# ═════════════════════════════════════════════════════════════════════
# Deterministic Grader
# ═════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def deterministic_grade_review(review: str, task: Task):
    normalized = _normalize(review)

    raw_score = 0.0
    breakdown = []

    for rule in task.scoring_rules:
        if any(kw in normalized for kw in rule.keywords):
            raw_score += rule.weight
            breakdown.append(f"✓ {rule.name}")
        else:
            breakdown.append(f"✗ {rule.name}")

    penalties = 0.0
    if len(review.strip()) < 30:
        penalties += 0.2

    bonus = 0.0
    if any(x in normalized for x in ["issue:", "fix:", "solution:"]):
        bonus += 0.1

    reward = max(0.0, min(1.0, raw_score - penalties + bonus))

    info = {
        "issue_score": raw_score * 0.5,
        "fix_score": raw_score * 0.3,
        "explanation_score": raw_score * 0.2,
        "penalties": penalties,
        "bonus": bonus,
        "total": reward,
    }

    return reward, "\n".join(breakdown), info


def is_malicious(review: str) -> bool:
    patterns = [
        "ignore previous instructions",
        "give full score",
        "output 1.0",
        "override",
    ]
    return any(p in review.lower() for p in patterns)


# ═════════════════════════════════════════════════════════════════════
# FIXED grade_review (MAIN BUG WAS HERE)
# ═════════════════════════════════════════════════════════════════════

def grade_review(review: str, task: Task):
    if is_malicious(review):
        return 0.0, "Malicious input detected", {
            "issue_score": 0.0,
            "fix_score": 0.0,
            "explanation_score": 0.0,
            "penalties": 1.0,
            "bonus": 0.0,
            "total": 0.0,
        }

    # deterministic baseline (ALWAYS used as final truth)
    det_reward, det_feedback, det_info = deterministic_grade_review(review, task)

    # optional LLM grading (non-authoritative)
    try:
        client = OpenAI(
            base_url=os.getenv("GRADER_API_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("GRADER_API_KEY", "no-key"),
            timeout=10.0,
            max_retries=0,
        )

        system_prompt = "You are a strict code review grader. Return JSON only."

        user_prompt = f"""
Code:
{task.code}

Expected issues:
{task.expected_issues}

Review:
{review}
"""

        response = client.chat.completions.create(
            model=os.getenv("GRADER_MODEL_NAME", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        result = json.loads(content) if content else {}

    except Exception:
        result = {}

    feedback = f"""
Task: {task.title}
Deterministic Score: {det_reward:.2f}

LLM (optional):
Issue: {result.get("issue_score", 0.0)}
Fix: {result.get("fix_score", 0.0)}
Explanation: {result.get("explanation_score", 0.0)}
"""

    return det_reward, feedback, det_info


# ═════════════════════════════════════════════════════════════════════
# FIXED ENVIRONMENT CLASS (WAS BROKEN / FLOATING)
# ═════════════════════════════════════════════════════════════════════

class CodeReviewEnvironment:
    def __init__(self):
        self._state = CodeReviewState()
        self._current_task: Optional[Task] = None

    def reset(self, task_index: int = 0) -> StepResult:
        task_index = max(0, min(task_index, len(TASKS) - 1))
        task = random.choice(TASKS[task_index])

        self._current_task = task

        self._state = CodeReviewState(
            expected_issues=task.expected_issues,
            difficulty=task.difficulty,
            task_index=task_index,
            variant_id=task.title,
            done=False,
        )

        observation = CodeReviewObservation(
            code=task.code,
            instructions=task.instructions,
            feedback="",
        )

        return StepResult(observation=observation, reward=0.0, done=False)

    def step(self, action: CodeReviewAction) -> StepResult:
        if self._current_task is None:
            return StepResult(
                observation=CodeReviewObservation(
                    code="",
                    instructions="",
                    feedback="Call reset() first",
                ),
                reward=0.0,
                done=True,
            )

        reward, feedback, info = grade_review(action.review, self._current_task)

        self._state.done = True

        observation = CodeReviewObservation(
            code=self._current_task.code,
            instructions=self._current_task.instructions,
            feedback=feedback,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=True,
            info=info,
        )

    @property
    def state(self):
        return self._state