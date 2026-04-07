"""
inference.py — Baseline inference script for Code Review OpenEnv

Runs all 3 tasks through an OpenAI-compatible LLM and submits the
generated reviews to the environment for grading.

Environment variables:
    API_BASE_URL  — OpenAI-compatible API base (default: http://localhost:8000/v1)
    MODEL_NAME    — Model identifier           (default: gpt-3.5-turbo)
    HF_TOKEN      — Hugging Face / API token   (optional)
    ENV_URL       — Environment server URL      (default: http://localhost:7860)
"""

from __future__ import annotations

import datetime
import json
import os
from openai import OpenAI

from client import CodeReviewClient


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")

TASK_NAMES: list[str] = [
    "Task 1 — Silent Bug Detection (Easy)",
    "Task 2 — Performance Trap (Medium)",
    "Task 3 — Deceptive Logic Bug (Hard)",
]


# ══════════════════════════════════════════════════════════════════════════════
# LLM Review Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_review(code: str, instructions: str) -> str:
    """
    Send the code + instructions to an OpenAI-compatible model in a two-step
    Chain-of-Thought pipeline to reduce hallucinations.
    """
    llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "no-key",
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

    final_review = (
        f"Issue: {issue}\n"
        f"Fix: {fix}\n"
        f"Explanation: {explanation}"
    )

    # Hallucination filter
    if "maybe" in final_review.lower():
        final_review = final_review.replace("maybe", "").replace("Maybe", "")
        
    return final_review


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run inference on all 3 tasks and report scores."""
    print("=" * 60)
    print("  Code Review OpenEnv — Inference Run")
    print("=" * 60)
    print(f"  Model:  {MODEL_NAME}")
    print(f"  API:    {API_BASE_URL}")
    print(f"  Env:    {ENV_URL}")
    print("=" * 60)
    print()

    client = CodeReviewClient(base_url=ENV_URL)

    # Verify server is alive
    try:
        health = client.health()
        print(f"✓ Server healthy: {health}\n")
    except Exception as exc:
        print(f"✗ Cannot reach environment at {ENV_URL}: {exc}")
        sys.exit(1)

    results = []

    for task_idx in range(3):
        task_label = TASK_NAMES[task_idx]
        print(f"{'─' * 60}")
        print(f"  {task_label}")
        print(f"{'─' * 60}")

        # 1. Reset environment with this task
        reset_result = client.reset(task_index=task_idx)
        code = reset_result.observation.code
        instructions = reset_result.observation.instructions

        print(f"  Code snippet ({len(code)} chars):")
        for line in code.split("\n"):
            print(f"    │ {line}")
        print()

        # 2. Generate review via LLM
        print("  ⏳ Generating review via LLM...")
        timestamp = datetime.datetime.now().isoformat()
        
        try:
            review = generate_review(code, instructions)
            
            print(f"  Review ({len(review)} chars):")
            preview = review[:300] + ("..." if len(review) > 300 else "")
            for line in preview.split("\n"):
                print(f"    │ {line}")
            print()

            # 3. Submit review to environment
            step_result = client.step(review=review)
            reward = step_result.reward
            feedback = step_result.observation.feedback
            
            # Fetch current state to log the specific randomized variant
            state = client.state()

            print(f"  Reward: {reward:.2f}")
            print(f"  Feedback:\n{feedback}\n")
            
            results.append({
                "task_id": task_label,
                "variant": state.variant_id,
                "review": review,
                "info": step_result.info,
                "model": MODEL_NAME,
                "timestamp": timestamp
            })

        except Exception as exc:
            print(f"  ✗ LLM error: {exc}")
            results.append({
                "task_id": task_label,
                "error": str(exc),
                "model": MODEL_NAME,
                "timestamp": timestamp
            })

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

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for r in results:
        name = r["task_id"]
        if "error" in r:
            print(f"  ✗ {name}: ERROR ({r['error']})")
        else:
            score = r["info"].get("total", 0.0)
            status = "✓" if score >= 0.5 else "△" if score >= 0.3 else "✗"
            print(f"  {status} {name}: {score:.2f}")

    print(f"{'─' * 60}")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Saved detailed logs to eval_log.json & eval_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
