"""
server/environment.py — Code Review Environment

Implements the OpenEnv Environment interface:
  - reset(task_index)  → select a task, return initial observation
  - step(action)       → grade the review, return reward + feedback
  - state              → current episode metadata

Grading is fully deterministic — no LLM in the loop.
"""

from __future__ import annotations

import json
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


def _strict_unit(x: float, eps: float = 0.01) -> float:
    """Clamp to a strict open interval (0, 1)."""
    return min(1.0 - eps, max(eps, x))


def _extract_review_fields(review: str) -> tuple[str, str, str]:
    """Extract issue/fix/explanation fields from JSON or sectioned text."""
    try:
        parsed = json.loads(review)
        if isinstance(parsed, dict):
            issue = _normalize(str(parsed.get("issue", "")))
            fix = _normalize(str(parsed.get("fix", "")))
            explanation = _normalize(str(parsed.get("explanation", "")))
            if issue or fix or explanation:
                return issue, fix, explanation
    except Exception:
        pass

    issue_match = re.search(r"issue:\s*(.*?)(?:\nfix:|$)", review, flags=re.IGNORECASE | re.DOTALL)
    fix_match = re.search(r"fix:\s*(.*?)(?:\nexplanation:|$)", review, flags=re.IGNORECASE | re.DOTALL)
    explanation_match = re.search(r"explanation:\s*(.*)$", review, flags=re.IGNORECASE | re.DOTALL)

    issue = _normalize(issue_match.group(1)) if issue_match else ""
    fix = _normalize(fix_match.group(1)) if fix_match else ""
    explanation = _normalize(explanation_match.group(1)) if explanation_match else ""
    return issue, fix, explanation


def _matches_any(text: str, keywords: list[str]) -> bool:
    text_norm = _normalize(text)
    return any(_normalize(kw) in text_norm for kw in keywords)


def _collect_rule_keywords(task: Task, name_fragments: tuple[str, ...]) -> list[str]:
    collected: list[str] = []
    for rule in task.scoring_rules:
        rule_name = rule.name.lower()
        if any(fragment in rule_name for fragment in name_fragments):
            collected.extend(rule.keywords)
    return collected


def _is_no_issue_task(task: Task) -> bool:
    no_issue_markers = [
        "no bug",
        "no issue",
        "safe",
        "already checked",
        "no division by zero",
    ]
    expected = " ".join(_normalize(item) for item in task.expected_issues)
    return any(marker in expected for marker in no_issue_markers)


def _is_none_issue_text(issue_text: str) -> bool:
    markers = ["none", "no issue", "no bug", "already checked", "safe", "correct"]
    return any(marker in _normalize(issue_text) for marker in markers)


EXPECTED: dict[str, list[str]] = {
    "iseven_bug": [
        "wrong logic",
        "incorrect condition",
        "returns true for odd",
        "parity error",
    ],
    "division by zero": [
        "divide by zero",
        "div by zero",
        "count is zero",
        "count == 0",
    ],
    "out-of-bounds access": [
        "out of bounds",
        "index out of range",
        "k exceeds array size",
        "invalid k",
        "oob",
        "boundary error",
        "arr[i+1]",
    ],
    "integer overflow": [
        "overflow",
        "wrap around",
        "int total",
        "truncation",
    ],
    "none": [
        "none",
        "no issue",
        "no bug",
        "safe",
        "already checked",
    ],
}


def _expected_display_issues(expected_issues: list[str]) -> list[str]:
    """Normalize expected issue labels for judge-facing mismatch messaging."""
    normalized = [_normalize(item) for item in expected_issues if str(item).strip()]
    boundary_markers = ["out of bounds", "out-of-bounds", "boundary", "index", "arr[i+1]"]
    if any(any(marker in issue for marker in boundary_markers) for issue in normalized):
        return [
            "out of bounds",
            "index out of range",
            "k exceeds array size",
            "invalid k",
            "boundary error",
        ]
    return list(expected_issues)


def _expand_expected_issues(expected_issues: list[str]) -> list[str]:
    """Expand expected issue labels into semantic aliases."""
    expanded: list[str] = []
    for issue in expected_issues:
        key = str(issue).lower().strip()
        if not key:
            continue
        expanded.append(key)

        if key in EXPECTED:
            expanded.extend(EXPECTED[key])

        for canonical, aliases in EXPECTED.items():
            if canonical.lower() in key:
                expanded.extend(aliases)

    # de-duplicate while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        norm = str(item).lower().strip()
        if norm and norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped


def run_tests_isEven() -> list[tuple[int, bool, bool]]:
    """Adversarial tests for a known parity bug implementation."""
    def isEven(n: int) -> bool:
        if n % 2 == 1:
            return True
        return False

    cases = [2, 3]
    return [(n, isEven(n), (n % 2 == 0)) for n in cases]


def _detect_adversarial_issue_from_code(code: str) -> tuple[Optional[str], list[str]]:
    """Detect known bug patterns and return failed adversarial cases."""
    code_norm = _normalize(code)
    failures: list[str] = []

    # Known parity bug profile: function returns True for odd numbers.
    if (
        "iseven" in code_norm
        and "% 2 == 1" in code_norm
        and "return true" in code_norm
    ):
        for n, actual, expected in run_tests_isEven():
            if actual != expected:
                failures.append(f"input={n}, expected={expected}, actual={actual}")
        if failures:
            return "iseven_bug", failures

    return None, failures


def detect_hallucination(review_issue: str, expected_issues: list[str]) -> bool:
    """Explicit pre-check for hallucinated issue claims."""
    review_issue = review_issue.lower().strip()
    normalized_expected = _expand_expected_issues(expected_issues)
    none_expected = normalized_expected == ["none"]
    if none_expected:
        normalized_expected = _expand_expected_issues(["none"])

    # simple semantic match
    matched = any(keyword in review_issue for keyword in normalized_expected)

    if not matched and not none_expected:
        return True

    if none_expected and review_issue not in _expand_expected_issues(["none"]):
        return True

    return False


def is_issue_correct(predicted: str, expected_list: list[str]) -> bool:
    """Strict check: predicted issue must contain one expected issue signal."""
    predicted_norm = _normalize(predicted)
    for exp in _expand_expected_issues(expected_list):
        if _normalize(exp) in predicted_norm:
            return True
    return False


def is_fix_relevant(issue: str, fix: str) -> bool:
    """Check whether the proposed fix is relevant to the identified issue."""
    issue = issue.lower().strip()
    fix = fix.lower().strip()

    # Crude but effective relevance checks.
    if "overflow" in issue and "long long" in fix:
        return True
    if "even" in issue and "% 2 == 0" in fix:
        return True

    # Existing task-family relevance checks.
    if (
        "division by zero" in issue
        or ("count" in issue and "zero" in issue)
    ) and any(token in fix for token in ["count == 0", "if(count", "if (count", "guard", "validate", "check"]):
        return True

    if (
        "out-of-bounds" in issue
        or "out of bounds" in issue
        or "arr[i+1]" in issue
        or "boundary" in issue
    ) and any(token in fix for token in ["i + 1 < arr.size()", "arr.size() - 1", "bound", "index"]):
        return True

    if (
        "empty" in issue
        or "arr.front" in issue
        or "arr.back" in issue
    ) and any(token in fix for token in ["arr.empty()", "if(arr.empty())", "if (arr.empty())", "size() == 0"]):
        return True

    if (
        "pass by value" in issue
        or "copy" in issue
        or "overhead" in issue
    ) and any(token in fix for token in ["const vector<int>&", "const reference", "reference"]):
        return True

    return False


def deterministic_grade_review(review: str, task: Task) -> tuple[float, str, dict[str, float]]:
    """
    Grade a code review deterministically.

    Returns:
        (reward, feedback, info_dict) where reward ∈ (0.0, 1.0)
    """
    normalized = _normalize(review)
    issue_score = 0.0
    fix_score = 0.0
    explanation_score = 0.0

    issue_text, fix_text, explanation_text = _extract_review_fields(review)
    issue_source = issue_text or normalized
    fix_source = fix_text or normalized
    explanation_source = explanation_text or normalized

    no_issue_task = _is_no_issue_task(task)
    hallucination = False
    adversarial_penalty = 0.0
    mismatch_detected = False
    mismatch_expected = _expected_display_issues(task.expected_issues)

    adversarial_issue, adversarial_failures = _detect_adversarial_issue_from_code(task.code)

    expected_for_hallucination = ["none"] if no_issue_task else list(task.expected_issues)
    if adversarial_issue and not no_issue_task:
        expected_for_hallucination = expected_for_hallucination + [adversarial_issue]
    if detect_hallucination(issue_source, expected_for_hallucination):
        adversarial_miss = bool(adversarial_issue and adversarial_failures)
        miss_line = "\nPenalty: missed adversarially validated bug." if adversarial_miss else ""
        expected_display = _expected_display_issues(expected_for_hallucination)
        detected_issue = (issue_text or issue_source or "").strip()
        reward = _strict_unit(0.0)
        feedback = (
            f"Task: {task.title} ({task.difficulty})\n"
            "Hallucinated issue detected\n"
            f"Mismatch detected: predicted issue '{detected_issue}' does not match expected issue type.\n"
            f"Expected issue family: {', '.join(expected_display)}\n"
            "Issue Identification: 0.00/0.50\n"
            "Fix Quality: 0.00/0.30\n"
            "Explanation Quality: 0.00/0.20\n"
            "Total score: 0.00"
            f"{miss_line}"
        )
        info = {
            "issue": 0.0,
            "fix": 0.0,
            "explanation": 0.0,
            "feedback": 0.0,
            "issue_score": 0.0,
            "fix_score": 0.0,
            "explanation_score": 0.0,
            "penalties": 0.0,
            "bonus": 0.0,
            "total_score": 0.0,
            "total": 0.0,
            "adversarial_penalty": 0.2 if adversarial_miss else 0.0,
            "mismatch_detected": True,
            "detected_issue": detected_issue,
            "expected_issues": expected_display,
        }
        return reward, feedback, info

    issue_keywords = task.expected_issues + _collect_rule_keywords(task, ("issue", "bug", "inefficiency", "detect"))
    fix_keywords = _collect_rule_keywords(task, ("fix", "optimization", "correct"))
    explanation_keywords = _collect_rule_keywords(task, ("explanation", "complexity", "edge"))

    if no_issue_task:
        issue_correct = _is_none_issue_text(issue_source)
        hallucination = not issue_correct
        issue_score = 0.5 if issue_correct else 0.0

        no_fix_markers = ["no fix", "none", "already", "not needed", "no change"]
        fix_correct = issue_correct and _matches_any(fix_source, no_fix_markers)
        fix_score = 0.3 if fix_correct else 0.0

        safe_reason_markers = ["safe", "handled", "already checked", "guard", "no bug", "correct"]
        explanation_correct = issue_correct and (
            _matches_any(explanation_source, explanation_keywords)
            or _matches_any(explanation_source, safe_reason_markers)
        )
        explanation_score = 0.2 if explanation_correct else 0.0
    else:
        expected_for_issue = list(task.expected_issues)
        if adversarial_issue:
            expected_for_issue.append(adversarial_issue)

        issue_correct = is_issue_correct(issue_source, expected_for_issue)
        hallucination = not issue_correct
        if not issue_correct:
            # HARD RULE: mismatch means zero across all rubric components.
            issue_score = 0.0
            fix_score = 0.0
            explanation_score = 0.0
            mismatch_detected = True
        else:
            issue_score = 0.5
            fix_relevant = is_fix_relevant(issue_source, fix_source)
            fix_correct = fix_relevant and _matches_any(fix_source, fix_keywords)
            fix_score = 0.3 if fix_correct else 0.0

            explanation_correct = (
                _matches_any(explanation_source, explanation_keywords)
                or _matches_any(explanation_source, task.expected_issues)
            )
            explanation_score = 0.2 if explanation_correct else 0.0

    adversarial_detected = False
    if adversarial_issue and adversarial_failures:
        adversarial_aliases = _expand_expected_issues([adversarial_issue])
        adversarial_detected = any(alias in issue_source for alias in adversarial_aliases)
        if adversarial_detected:
            issue_score = 0.5
        else:
            issue_score = 0.0
            fix_score = 0.0
            explanation_score = 0.0
            mismatch_detected = True
            adversarial_penalty = 0.2

    total_score = max(0.0, issue_score + fix_score + explanation_score - adversarial_penalty)
    reward = _strict_unit(total_score)

    feedback_lines = [
        f"Task: {task.title} ({task.difficulty})",
        f"Issue Identification: {issue_score:.2f}/0.50",
        f"Fix Quality: {fix_score:.2f}/0.30",
        f"Explanation Quality: {explanation_score:.2f}/0.20",
    ]
    if hallucination:
        feedback_lines.append("Hallucination detected: mentioned issue does not match code/task.")
    if mismatch_detected:
        detected_issue = (issue_text or "(empty issue)").strip()
        feedback_lines.append(f"Mismatch detected: predicted issue '{detected_issue}' does not match expected issue type.")
        feedback_lines.append(f"Expected issue family: {', '.join(mismatch_expected)}")
    if (not no_issue_task) and issue_score > 0.0 and fix_score == 0.0:
        feedback_lines.append("Fix relevance check failed: fix does not directly address the identified issue.")
    if adversarial_issue and adversarial_failures:
        feedback_lines.append(f"Adversarial validation detected bug profile: {adversarial_issue}.")
        if adversarial_detected:
            feedback_lines.append("Adversarial check passed: review identified the validated bug.")
        else:
            feedback_lines.append("Penalty: missed adversarially validated bug.")
    feedback_lines.append(f"Total score: {total_score:.2f}")

    feedback = "\n".join(feedback_lines)
    
    info = {
        "issue_score": issue_score,
        "fix_score": fix_score,
        "explanation_score": explanation_score,
        "penalties": 0.0,
        "bonus": 0.0,
        "total_score": total_score,
        "total": reward,
        "mismatch_detected": mismatch_detected,
        "detected_issue": issue_text,
        "expected_issues": mismatch_expected,
    }
    info["adversarial_penalty"] = adversarial_penalty
    info["issue"] = issue_score
    info["fix"] = fix_score
    info["explanation"] = explanation_score
    info["feedback"] = 1.0
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
        return 0.01, "Input flagged as malicious.", {
            "issue_score": 0.0,
            "fix_score": 0.0,
            "explanation_score": 0.0,
            "penalties": 1.0,
            "bonus": 0.0,
            "total_score": 0.0,
            "total": 0.01,
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
