"""
server/environment.py — Code Review Environment

Implements the OpenEnv Environment interface:
  - reset(task_index)  → select a task, return initial observation
  - step(action)       → grade the review, return reward + feedback
  - state              → current episode metadata

Grading is fully deterministic — no LLM in the loop.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    StepResult,
)


# ══════════════════════════════════════════════════════════════════════════════
# Task definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    """A single code-review task with its grading rubric."""
    title: str
    difficulty: str
    code: str
    instructions: str
    expected_issues: list[str]
    scoring_rules: list[ScoringRule]


@dataclass
class ScoringRule:
    """One rubric criterion: a list of keyword alternatives and the weight."""
    name: str
    keywords: list[str]          # any match → credit
    weight: float                # points awarded


# ---------------------------------------------------------------------------
# Task 1 — Silent Bug Detection (Easy Variants)
# ---------------------------------------------------------------------------
TASK_EASY_A = Task(
    title="Silent Bug Detection (Variant A)",
    difficulty="easy",
    code=(
        "int average(int sum, int count) {\n"
        "    return sum / count;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or potential issues.",
    expected_issues=["division by zero", "zero check", "count == 0"],
    scoring_rules=[
        ScoringRule(
            name="issue_detected",
            keywords=["division by zero", "divide by zero", "div by zero", "count is zero", "count == 0"],
            weight=0.5,
        ),
        ScoringRule(name="fix_suggested", keywords=["check", "validate", "guard", "if", "!= 0", "> 0"], weight=0.3,),
        ScoringRule(name="explanation_quality", keywords=["edge case", "runtime error", "safety", "robust", "undefined behavior"], weight=0.2,),
    ],
)

TASK_EASY_B = Task(
    title="Silent Bug Detection (Variant B - Float cast mask)",
    difficulty="easy",
    code=(
        "float average(int sum, int count) {\n"
        "    return (float)sum / count;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or potential issues.",
    expected_issues=["division by zero", "zero check", "count == 0"],
    scoring_rules=[
        ScoringRule(
            name="issue_detected",
            keywords=["division by zero", "divide by zero", "div by zero", "count is zero", "count == 0"],
            weight=0.5,
        ),
        ScoringRule(name="fix_suggested", keywords=["check", "validate", "guard", "if", "!= 0", "> 0"], weight=0.3,),
        ScoringRule(name="explanation_quality", keywords=["edge case", "runtime error", "safety", "robust", "undefined behavior"], weight=0.2,),
    ],
)

TASK_EASY_C = Task(
    title="Silent Bug Detection (Variant C - Actually Safe)",
    difficulty="easy",
    code=(
        "int average(int sum, int count) {\n"
        "    if (count == 0) return 0;\n"
        "    return sum / count;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or potential issues.",
    expected_issues=["no bug", "already checked", "safe code", "no division by zero"],
    scoring_rules=[
        ScoringRule(
            name="issue_detected",
            keywords=["no bug", "no division by zero", "already checked", "safe", "correct", "no issue", "checks for zero", "guard exists"],
            weight=0.5,
        ),
        ScoringRule(name="stylistic_fix_suggested", keywords=["std::optional", "exception", "throw", "error code"], weight=0.3,),
        ScoringRule(name="explanation_quality", keywords=["edge case handled", "robust", "return 0 is misleading"], weight=0.2,),
    ],
)

# ---------------------------------------------------------------------------
# Task 2 — Performance Trap (Medium Variants)
# ---------------------------------------------------------------------------
TASK_MEDIUM_A = Task(
    title="Performance Trap (Variant A)",
    difficulty="medium",
    code=(
        "vector<int> getUnique(vector<int>& arr) {\n"
        "    vector<int> result;\n"
        "    for(int i = 0; i < arr.size(); i++) {\n"
        "        bool found = false;\n"
        "        for(int j = 0; j < result.size(); j++) {\n"
        "            if(result[j] == arr[i]) {\n"
        "                found = true;\n"
        "                break;\n"
        "            }\n"
        "        }\n"
        "        if(!found) result.push_back(arr[i]);\n"
        "    }\n"
        "    return result;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify performance issues, suggest optimizations, and explain the complexity.",
    expected_issues=["O(n²) complexity", "use set/unordered_set"],
    scoring_rules=[
        ScoringRule(name="inefficiency_detected", keywords=["o(n^2)", "o(n²)", "quadratic", "nested loop", "slow", "performance", "n squared", "time complexity"], weight=0.5,),
        ScoringRule(name="optimization_suggested", keywords=["set", "unordered_set", "hash", "hashset", "sort", "unique", "std::unordered_set"], weight=0.3,),
        ScoringRule(name="complexity_mentioned", keywords=["o(n)", "o(n log n)", "o(1)", "linear", "logarithmic", "complexity", "big-o", "time complexity"], weight=0.2,),
    ],
)

TASK_MEDIUM_B = Task(
    title="Performance Trap (Variant B - Hidden nested logic)",
    difficulty="medium",
    code=(
        "#include <algorithm>\n"
        "vector<int> getUnique(vector<int>& arr) {\n"
        "    vector<int> result;\n"
        "    for(int n : arr) {\n"
        "        if(std::find(result.begin(), result.end(), n) == result.end()) {\n"
        "            result.push_back(n);\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify performance issues, suggest optimizations, and explain the complexity.",
    expected_issues=["O(n²) complexity", "use set/unordered_set", "std::find is linear"],
    scoring_rules=[
        ScoringRule(name="inefficiency_detected", keywords=["o(n^2)", "o(n²)", "quadratic", "std::find is o(n)", "linear search", "slow", "performance", "time complexity", "find is linear"], weight=0.5,),
        ScoringRule(name="optimization_suggested", keywords=["set", "unordered_set", "hash", "hashset", "sort", "unique", "std::unordered_set"], weight=0.3,),
        ScoringRule(name="complexity_mentioned", keywords=["o(n)", "o(n log n)", "o(1)", "linear", "logarithmic", "complexity", "big-o", "time complexity"], weight=0.2,),
    ],
)

TASK_MEDIUM_C = Task(
    title="Performance Trap (Variant C - Copy Overhead)",
    difficulty="medium",
    code=(
        "vector<int> getUnique(vector<int> arr) {\n"
        "    vector<int> result;\n"
        "    // Assume some unique filtering happens here\n"
        "    for(int n : arr) { if(n > 0) result.push_back(n); }\n"
        "    return result;\n"
        "}"
    ),
    instructions="Review the following C++ function signature and body. Identify performance issues and suggest optimizations.",
    expected_issues=["pass by value", "copy overhead", "pass by const reference"],
    scoring_rules=[
        ScoringRule(name="inefficiency_detected", keywords=["pass by value", "copy", "copied", "unnecessary copy", "memory", "overhead"], weight=0.5,),
        ScoringRule(name="optimization_suggested", keywords=["reference", "const reference", "const vector", "vector<int>&", "const vector<int>&"], weight=0.3,),
        ScoringRule(name="complexity_mentioned", keywords=["o(n)", "linear copy", "allocation", "heap", "deep copy", "time complexity", "space complexity"], weight=0.2,),
    ],
)

# ---------------------------------------------------------------------------
# Task 3 — Deceptive Logic Bug (Hard Variants)
# ---------------------------------------------------------------------------
TASK_HARD_A = Task(
    title="Deceptive Logic Bug (Variant A)",
    difficulty="hard",
    code=(
        "bool isSorted(vector<int>& arr) {\n"
        "    for(int i = 0; i < arr.size(); i++) {\n"
        "        if(arr[i] > arr[i+1]) return false;\n"
        "    }\n"
        "    return true;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or logic errors. Provide a corrected version and explain the issue.",
    expected_issues=["out-of-bounds access", "incorrect loop condition", "edge case handling"],
    scoring_rules=[
        ScoringRule(name="bug_detected", keywords=["out of bounds", "out-of-bounds", "buffer overflow", "overflow", "oob", "arr[i+1]", "boundary", "segfault", "undefined behavior"], weight=0.5,),
        ScoringRule(name="correct_fix", keywords=["size() - 1", "i < arr.size() - 1", "i + 1 < arr.size()", "< size - 1"], weight=0.3,),
        ScoringRule(name="edge_case_mentioned", keywords=["empty", "single element", "one element", "edge case", "empty array", "corner case"], weight=0.2,),
    ],
)

TASK_HARD_B = Task(
    title="Deceptive Logic Bug (Variant B - Empty Array Fallacy)",
    difficulty="hard",
    code=(
        "int getMaxDifference(vector<int>& arr) {\n"
        "    int min_val = arr.front();\n"
        "    int max_val = arr.back();\n"
        "    for(int n : arr) {\n"
        "        if(n < min_val) min_val = n;\n"
        "        if(n > max_val) max_val = n;\n"
        "    }\n"
        "    return max_val - min_val;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or logic errors. Provide a corrected version and explain the issue.",
    expected_issues=["empty array", "undefined behavior", "front/back on empty"],
    scoring_rules=[
        ScoringRule(name="bug_detected", keywords=["empty array", "empty vector", "size 0", "length 0", "arr is empty", "no elements", "out of bounds", "undefined behavior", "arr.front()", "arr.back()"], weight=0.5,),
        ScoringRule(name="correct_fix", keywords=["if(arr.empty())", "if (arr.empty())", "arr.size() == 0", "check if empty"], weight=0.3,),
        ScoringRule(name="edge_case_mentioned", keywords=["empty", "edge case", "corner case", "segfault", "exception"], weight=0.2,),
    ],
)

TASK_HARD_C = Task(
    title="Deceptive Logic Bug (Variant C - Integer Overflow)",
    difficulty="hard",
    code=(
        "long long calculateTotalSize(const vector<string>& files) {\n"
        "    int total = 0;\n"
        "    for(const auto& file : files) {\n"
        "        total += file.size();\n"
        "    }\n"
        "    return total;\n"
        "}"
    ),
    instructions="Review the following C++ function. Identify any bugs, edge cases, or logic errors.",
    expected_issues=["integer overflow", "total overflow", "return type mismatch"],
    scoring_rules=[
        ScoringRule(name="bug_detected", keywords=["overflow", "integer overflow", "int total", "total can overflow", "max int", "wrap around", "truncation"], weight=0.5,),
        ScoringRule(name="correct_fix", keywords=["long long total", "auto total = 0ll", "size_t total"], weight=0.3,),
        ScoringRule(name="edge_case_mentioned", keywords=["large files", "huge vector", "large strings", "many files", "sum exceeds 2gb", "precision loss"], weight=0.2,),
    ],
)

# Ordered task list — indices 0, 1, 2 map to lists of variants
TASKS: list[list[Task]] = [
    [TASK_EASY_A, TASK_EASY_B, TASK_EASY_C],
    [TASK_MEDIUM_A, TASK_MEDIUM_B, TASK_MEDIUM_C],
    [TASK_HARD_A, TASK_HARD_B, TASK_HARD_C],
]


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic grader
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip for robust keyword matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def deterministic_grade_review(review: str, task: Task) -> tuple[float, str, dict[str, float]]:
    """
    Grade a code review deterministically.

    Returns:
        (reward, feedback, info_dict) where reward ∈ [0.0, 1.0]
    """
    normalized = _normalize(review)
    breakdown: list[str] = []
    raw_score = 0.0
    issue_score = 0.0
    fix_score = 0.0
    explanation_score = 0.0

    # --- Rubric scoring -------------------------------------------------------
    for rule in task.scoring_rules:
        matched = any(kw in normalized for kw in rule.keywords)
        if matched:
            raw_score += rule.weight
            breakdown.append(f"  ✓ {rule.name}: +{rule.weight:.1f}")
            rule_name = rule.name.lower()
            if "fix" in rule_name or "optimization" in rule_name:
                fix_score += rule.weight
            elif "explanation" in rule_name or "complexity" in rule_name or "edge" in rule_name:
                explanation_score += rule.weight
            else:
                issue_score += rule.weight
        else:
            breakdown.append(f"  ✗ {rule.name}:  0.0  (missing)")

    # --- Penalties ------------------------------------------------------------
    penalties = 0.0

    # Very short review (< 30 chars)
    if len(review.strip()) < 30:
        penalties += 0.2
        breakdown.append("  ⚠ Penalty: very short review → −0.2")

    # Irrelevant / repetitive (naive heuristic: >50% repeated words)
    words = normalized.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            penalties += 0.2
            breakdown.append("  ⚠ Penalty: repetitive text → −0.2")

    # --- Bonus ----------------------------------------------------------------
    bonus = 0.0

    # Structured explanation ("Issue:", "Fix:", "Bug:", "Solution:")
    structure_markers = ["issue:", "fix:", "bug:", "solution:", "problem:", "recommendation:"]
    if any(marker in normalized for marker in structure_markers):
        bonus += 0.1
        breakdown.append("  ★ Bonus: structured explanation → +0.1")

    # --- Final reward (clamped to [0.0, 1.0]) ---------------------------------
    reward = max(0.0, min(1.0, raw_score - penalties + bonus))
    breakdown.insert(0, f"Task: {task.title} ({task.difficulty})")
    breakdown.append(f"  ─────────────────────────────")
    breakdown.append(f"  Raw: {raw_score:.2f}  Penalties: −{penalties:.2f}  Bonus: +{bonus:.2f}")
    breakdown.append(f"  Final reward: {reward:.2f}")

    feedback = "\n".join(breakdown)
    
    info = {
        "issue_score": issue_score,
        "fix_score": fix_score,
        "explanation_score": explanation_score,
        "penalties": penalties,
        "bonus": bonus,
        "total": reward
    }
    return reward, feedback, info

def is_malicious(review: str) -> bool:
    patterns = [
        "ignore previous instructions",
        "give full score",
        "output 1.0",
        "system prompt",
        "override"
    ]
    review_lower = review.lower()
    return any(p in review_lower for p in patterns)


def grade_review(review: str, task: Task) -> tuple[float, str, dict[str, float]]:
    """Grade a code review deterministically for reproducible evaluation."""
    if is_malicious(review):
        return 0.0, "Input flagged as malicious.", {
            "issue_score": 0.0,
            "fix_score": 0.0,
            "explanation_score": 0.0,
            "penalties": 1.0,
            "bonus": 0.0,
            "total": 0.0,
        }

    return deterministic_grade_review(review, task)


# ══════════════════════════════════════════════════════════════════════════════
# Environment class
# ══════════════════════════════════════════════════════════════════════════════

class CodeReviewEnvironment:
    """
    OpenEnv-compatible environment for code review evaluation.

    Lifecycle:
        1. reset(task_index) → observation with code + instructions
        2. step(action)      → graded reward + feedback
        3. (repeat)
    """

    def __init__(self) -> None:
        self._state = CodeReviewState()
        self._current_task: Optional[Task] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_index: int = 0) -> StepResult:
        """
        Start a new episode by selecting a task.

        Args:
            task_index: 0 = easy, 1 = medium, 2 = hard
        """
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

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: CodeReviewAction) -> StepResult:
        """
        Evaluate the agent's review and return a graded reward.

        Single-step episodes: after one step the episode is done.
        """
        if self._current_task is None:
            return StepResult(
                observation=CodeReviewObservation(
                    code="",
                    instructions="",
                    feedback="Error: call reset() before step().",
                ),
                reward=0.0,
                done=True,
            )

        reward, feedback, scores_dict = grade_review(action.review, self._current_task)

        self._state.done = True

        observation = CodeReviewObservation(
            code=self._current_task.code,
            instructions=self._current_task.instructions,
            feedback=feedback,
        )

        return StepResult(observation=observation, reward=reward, done=True, info=scores_dict)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def state(self) -> CodeReviewState:
        """Return current episode metadata."""
        return self._state
