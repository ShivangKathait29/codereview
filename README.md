---
title: Codereview
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# 🔍 Code Review OpenEnv Environment

A production-ready [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment that evaluates AI agents on their ability to perform **real-world code reviews** — detecting bugs, suggesting fixes, and explaining issues.

---

## 🎯 What It Does

The environment presents C++ code snippets containing intentional bugs or performance issues. An AI agent submits a free-text code review, and the **deterministic grader** scores it based on:

| Criterion | Weight |
|---|---|
| Issue detection | +0.5 |
| Fix quality | +0.3 |
| Explanation depth | +0.2 |
| Structured format bonus | +0.1 |
| Short review penalty | −0.2 |
| Repetitive text penalty | −0.2 |

Final reward is clamped to **[0.0, 1.0]**.

For validator compatibility, task-level scores and total are emitted in the strict-open interval **(0.0, 1.0)**.

---

## 📐 Environment Spaces

### Action Space (`CodeReviewAction`)
- **`review`** (string): The agent's free-text code review analyzing the provided snippet. Supports markdown-style formatting and structured sections.

### Observation Space (`CodeReviewObservation`)
- **`code`** (string): The actual source code (C++) containing bugs or inefficiencies.
- **`instructions`** (string): Context-specific guidance for the current task.
- **`feedback`** (string): (Post-Step) Detailed breakdown from the grader, identifying hits on the rubric and applicable penalties.

### State Space (`CodeReviewState`)
Internal metadata used by the grader:
- **`expected_issues`**: Ground-truth keywords/patterns.
- **`difficulty`**: Task difficulty level (easy | medium | hard).
- **`variant_id`**: Randomly selected task variant title.

---

## 📈 Baseline Performance

Baseline scores achieved using **GPT-3.5-Turbo** as the review agent (see `inference.py`):

| Task | Difficulty | Baseline Score |
|---|---|---|
| Task 0 — Silent Bug Detection | 🟢 Easy | **0.85** |
| Task 1 — Performance Trap | 🟡 Medium | **0.62** |
| Task 2 — Deceptive Logic Bug | 🔴 Hard | **0.31** |
| **AVERAGE** | | **0.59** |

*Note: Scores represent the capability of the baseline agent to identify bugs semantically while providing a structured fix and explanation.*

---

## 🧪 Tasks

| # | Name | Difficulty | Bug Type |
|---|---|---|---|
| 0 | Silent Bug Detection | 🟢 Easy | Division by zero |
| 1 | Performance Trap | 🟡 Medium | O(n²) nested loop |
| 2 | Deceptive Logic Bug | 🔴 Hard | Off-by-one / OOB access |

---

## 🎬 Live Demo Features

The root route (`/`) serves an interactive demo UI that includes:

- Live evaluation pipeline (code input -> agent output -> grader output)
- Step-wise reasoning trace and score explainability
- Error Type Classification Dashboard
- Hallucination detection mode (false-positive penalty)
- Adversarial test generator with explicit reasons (`Why this test was generated`)
- Model comparison leaderboard
- Auto replay + downloadable replay report
- One-click **Judge Demo Mode** sequence
- Auto benchmark generation with summary table

---

## 📁 Project Structure

```
code_review_env/
├── models.py              # Pydantic data models (Action, Observation, State)
├── client.py              # Typed HTTP client for the environment
├── inference.py           # Baseline inference script (OpenAI-compatible)
├── Dockerfile             # Container configuration
├── openenv.yaml           # OpenEnv manifest
├── requirements.txt       # Python dependencies
├── README.md
└── server/
    ├── __init__.py
    ├── environment.py     # Core environment logic + deterministic grader
  └── app.py             # FastAPI server + demo UI + benchmark endpoints
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# From the project root
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

The app is now live at `http://localhost:7860`.

- Interactive demo UI: `http://localhost:7860/`
- Swagger docs: `http://localhost:7860/docs`
- API docs shortcut: `http://localhost:7860/api`

### 3. Test with cURL

```bash
# Health check
curl http://localhost:7860/health

# Reset (select task 0 = easy)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_index": 0}'

# Submit a review
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"review": "Issue: Division by zero when count is 0. Fix: Add a guard check if(count == 0) return 0. Explanation: This is an edge case that causes undefined behavior."}'
```

### 4. Run with Python Client

```python
from client import CodeReviewClient

client = CodeReviewClient("http://localhost:7860")
client.reset(task_index=0)
result = client.step(review="The function has a division-by-zero bug when count is 0.")
print(f"Reward: {result.reward}")
print(f"Feedback:\n{result.observation.feedback}")
```

---

## 🐳 Docker

```bash
# Build (from project root)
docker build -t code-review-env -f Dockerfile .

# Run
docker run -p 7860:7860 code-review-env
```

---

## 🤖 Inference Script

The `inference.py` script runs all 3 tasks through an OpenAI-compatible LLM:

```bash
# Set environment variables
export API_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-token-here"
export ENV_URL="http://localhost:7860"

# Run inference
python inference.py
```

**Output**: Per-task scores and an average score across all 3 tasks.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive live demo page |
| `/api` | GET | Redirect shortcut to Swagger docs |
| `/reset` | POST | Start a new episode (select task by index) |
| `/step` | POST | Submit a review for grading |
| `/state` | GET | Get current episode metadata |
| `/health` | GET | Liveness probe |
| `/metadata` | GET | OpenEnv metadata |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/mcp` | POST | Minimal MCP-compatible JSON-RPC response |
| `/demo/evaluate` | POST | End-to-end live evaluation pipeline |
| `/demo/compare` | POST | Multi-model leaderboard on same scenario |
| `/demo/adversarial` | POST | Adversarial test generation |
| `/demo/benchmark-report` | GET | Benchmark report JSON (summary + scenario rows) |
| `/demo/benchmark-table` | GET | Compact benchmark table payload |
| `/docs` | GET | Interactive Swagger UI |

### POST `/reset`

**Body**: `{"task_index": 0}` (0=easy, 1=medium, 2=hard)

**Response**: `StepResult` with initial observation.

### POST `/step`

**Body**: `{"review": "Your code review text..."}`

**Response**: `StepResult` with reward (0.0–1.0), feedback, and `done=true`.

### GET `/demo/benchmark-report`

Returns a deterministic benchmark report with:

- `summary_rows` (ranked model metrics)
- `scenario_rows` (per-scenario details)
- `table` (render-ready benchmark table text)

### GET `/demo/benchmark-table`

Returns a compact benchmark payload suitable for quick UI display/download.

---

## 🧭 Judge Demo Flow

Use the **Judge Demo Mode** button in the UI to run this scripted sequence automatically:

1. Standard bug-detection run
2. Hallucination trap run (false-positive simulation)
3. Adversarial test generation with rationale
4. Benchmark suite generation and leaderboard update

This gives a stable 60-second showcase with minimal manual steps.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `gpt-3.5-turbo` | Model identifier |
| `HF_TOKEN` | _(no default)_ | Hugging Face / API authentication token |
| `OPENAI_API_KEY` | _(no default)_ | Fallback API key if `HF_TOKEN` is not set |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

---

## 🏁 Acceptance Criteria

- [x] Environment runs locally and via Docker
- [x] HF Space responds to `/reset`
- [x] 3 tasks implemented (easy, medium, hard)
- [x] Grader returns 0.0–1.0 deterministically
- [x] `inference.py` runs successfully
- [x] No LLM in the grading loop
- [x] Runtime < 20 minutes on 2 vCPU / 8 GB RAM
- [x] Live demo UI with explainable scoring
- [x] Benchmark report endpoints + judge demo mode

---

## 📄 License

MIT
