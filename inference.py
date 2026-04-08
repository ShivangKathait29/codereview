"""
inference.py — Baseline inference script for Code Review OpenEnv

Runs all 3 tasks through an OpenAI-compatible LLM and submits the
generated reviews to the environment for grading.

Environment variables:
    API_BASE_URL  — OpenAI-compatible API base (default: http://localhost:8000/v1)
    MODEL_NAME    — Model identifier           (default: gpt-3.5-turbo)
    HF_TOKEN      — Hugging Face / API token   (optional; no default)
    ENV_URL       — Environment server URL      (default: http://localhost:7860)
"""

from __future__ import annotations

import datetime
import json
import os
import re
from openai import OpenAI

from client import CodeReviewClient


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
API_KEY: str | None = HF_TOKEN or OPENAI_API_KEY
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK_NAME: str = os.getenv("BENCHMARK_NAME", "code_review_env")

TASK_NAMES: list[str] = [
    "Task 1 — Silent Bug Detection (Easy)",
    "Task 2 — Performance Trap (Medium)",
    "Task 3 — Deceptive Logic Bug (Hard)",
]


# ══════════════════════════════════════════════════════════════════════════════
# LLM Review Generator
# ══════════════════════════════════════════════════════════════════════════════

def _one_line(text: str) -> str:
    """Collapse whitespace so logs always remain single-line."""
    return re.sub(r"\s+", " ", text).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(done).lower()
    error_val = _one_line(error) if error else "null"
    action_val = _one_line(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def generate_review(code: str, instructions: str) -> str:
    """
    Send the code + instructions to an OpenAI-compatible model in a two-step
    Chain-of-Thought pipeline to reduce hallucinations.
    """
    llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "no-key",
        timeout=60.0,
        max_retries=1
    )

    # Step 1: Reasoning Phase
    reasoning_prompt = f"""You are a senior software engineer.

Analyze the following code step by step.

Identify:
- potential bugs
- inefficiencies
- edge cases

Code:
```cpp
{code}
```
"""

    response_1 = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": reasoning_prompt},
        ],
        temperature=0.0
    )
    analysis = response_1.choices[0].message.content

    # Step 2: Final Review (with hallucination defense)
    final_prompt = f"""You are a strict code reviewer.

Use ONLY the validated insights below.
If unsure about an issue, DO NOT include it.
If the code is correct, explicitly state "No issues found."

Analysis:
{analysis}

Generate final review matching this JSON format exactly:
{{
"issue": "what is wrong",
"explanation": "why it is wrong",
"fix": "correct solution"
}}
"""

    response_2 = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    content = response_2.choices[0].message.content
    try:
        data = json.loads(content)
        issue = data.get("issue", "")
        explanation = data.get("explanation", "")
        fix = data.get("fix", "")
    except Exception as e:
        raise ValueError(f"LLM parse failed: {e}")

    final_review = json.dumps({
    "issue": issue,
    "fix": fix,
    "explanation": explanation
    })

    # Hallucination filter
    if "maybe" in final_review.lower():
        final_review = final_review.replace("maybe", "").replace("Maybe", "")
        
    return final_review


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run inference on all 3 tasks and report scores."""
    client = CodeReviewClient(base_url=ENV_URL)

    results = []

    for task_idx in range(3):
        task_label = TASK_NAMES[task_idx]
        rewards: list[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_label, env=BENCHMARK_NAME, model=MODEL_NAME)

        timestamp = datetime.datetime.now().isoformat()

        try:
            # 1. Reset environment with this task
            reset_result = client.reset(task_index=task_idx)
            code = reset_result.observation.code
            instructions = reset_result.observation.instructions

            # 2. Generate review via LLM
            review = generate_review(code, instructions)

            # 3. Submit review to environment
            step_result = client.step(review=review)
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)

            rewards.append(reward)
            steps_taken = 1
            score = max(0.0, min(1.0, reward))

            log_step(step=1, action=review, reward=reward, done=done, error=None)
            
            # Fetch current state to log the specific randomized variant
            state = client.state()
            success = done
            
            results.append({
                "task_id": task_label,
                "variant": state.variant_id,
                "review": review,
                "info": step_result.info,
                "reward": reward,
                "model": MODEL_NAME,
                "timestamp": timestamp
            })

        except Exception as exc:
            results.append({
                "task_id": task_label,
                "error": str(exc),
                "model": MODEL_NAME,
                "timestamp": timestamp
            })
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    # ── Summary ──────────────────────────────────────────────────────────
    with open("eval_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    valid_results = [r for r in results if "info" in r and r.get("info")]
    avg_score = sum(r["info"].get("total", 0.0) for r in valid_results) / len(valid_results) if valid_results else 0.0
    
    summary = {
        "num_tasks": len(results),
        "successful_tasks": len(valid_results),
        "avg_score": avg_score
    }
    
    with open("eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
