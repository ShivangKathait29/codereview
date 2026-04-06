# 🚀 Code Review OpenEnv Environment (Hackathon Project)

---

## 🧠 Project Overview

### 📌 Summary
This project is a production-ready OpenEnv environment that simulates a **real-world code review workflow**. The environment evaluates AI agents on their ability to analyze code, detect issues, and provide meaningful feedback.

### 🎯 Objective
Enable training and evaluation of AI agents on:
- Bug detection
- Performance optimization
- Code correctness
- Explanation quality

### 👥 Target Users
- AI researchers
- LLM/RL engineers
- Hackathon judges
- Developers building agent systems

---

## 🧩 Tech Stack

- Python 3.11
- OpenEnv SDK
- FastAPI
- Uvicorn
- Pydantic
- Docker
- OpenAI-compatible API

---

## ⚙️ Core Features

### ✅ Must-Have
- Full OpenEnv compliance:
  - `reset()`, `step()`, `state()`
- Typed models:
  - Action, Observation, State
- 3 deterministic tasks:
  - Easy → Silent Bug Detection
  - Medium → Performance Optimization
  - Hard → Logical Bug Detection
- Reward function with partial scoring
- Deterministic grading system
- Baseline inference script (`inference.py`)
- Dockerfile
- Hugging Face Space deployment ready
- README

---

## 🧪 Task Definitions

---

### 🟢 Task 1 — Silent Bug Detection (Easy)

#### Code:
```cpp
int average(int sum, int count) {
    return sum / count;
}
Task 2 — Performance Trap (Medium)
Code:
vector<int> getUnique(vector<int>& arr) {
    vector<int> result;
    for(int i = 0; i < arr.size(); i++) {
        bool found = false;
        for(int j = 0; j < result.size(); j++) {
            if(result[j] == arr[i]) {
                found = true;
                break;
            }
        }
        if(!found) result.push_back(arr[i]);
    }
    return result;
}
Expected Issues:
O(n²) complexity
Should use set/unordered_set
Scoring:
inefficiency detected → +0.5
optimization suggestion → +0.3
complexity mention → +0.2
🔴 Task 3 — Deceptive Logic Bug (Hard)
Code:
bool isSorted(vector<int>& arr) {
    for(int i = 0; i < arr.size(); i++) {
        if(arr[i] > arr[i+1]) return false;
    }
    return true;
}
Expected Issues:
out-of-bounds access
incorrect loop condition
edge case handling
Scoring:
detects bug → +0.5
correct fix → +0.3
edge case mention → +0.2
🎯 Reward Design

Final reward:

reward = issue_score + fix_score + explanation_score - penalties
Penalties:
very short review → -0.2
irrelevant/repeated text → -0.2
Bonus:
structured explanation ("Issue:", "Fix:") → +0.1
📦 Data Models
Action
class CodeReviewAction(Action):
    review: str
Observation
class CodeReviewObservation(Observation):
    code: str
    instructions: str
    feedback: str
State
class CodeReviewState(State):
    expected_issues: list[str]
    difficulty: str
🔌 API Endpoints
/reset
/step
/state
/health
/docs
📁 Project Structure
code_review_env/
├── models.py
├── client.py
├── inference.py
├── openenv.yaml
├── README.md
├── server/
│   ├── environment.py
│   ├── app.py
│   └── Dockerfile
🧪 Inference Script Requirements
Uses OpenAI-compatible API
Reads:
API_BASE_URL
MODEL_NAME
HF_TOKEN
Runs all tasks
Outputs average score
⚠️ Constraints
Runtime < 20 minutes
Works on:
2 vCPU
8GB RAM
Deterministic grading
No LLM usage in grader
🏁 Acceptance Criteria
Environment runs locally and via Docker
HF Space responds to /reset
3 tasks implemented
Grader returns 0.0–1.0
inference.py runs successfully
🎉 End of Spec

---

# 🤖 2. FINAL AI AGENT PROMPT

Use this EXACT prompt:

---

```text
You are an expert Python backend engineer and AI systems developer.

Build a complete OpenEnv environment project based on the specification below.

IMPORTANT REQUIREMENTS:
- Follow OpenEnv architecture strictly
- Implement ALL required files:
  - models.py
  - client.py
  - server/environment.py
  - server/app.py
  - Dockerfile
  - inference.py
  - openenv.yaml
  - README.md
- Use clean, production-quality code
- Include type hints and comments
- Ensure Docker builds successfully
- Ensure environment runs with uvicorn
- Ensure inference.py produces reproducible scores
- DO NOT skip any part

PROJECT SPECIFICATION:

[PASTE FULL PROJECT_SPEC.md HERE]

IMPLEMENTATION DETAILS:

1. Environment:
- Single-step per task
- reset() selects a task
- step() evaluates review and returns reward

2. Grader:
- Must be deterministic
- Use keyword matching with flexibility
- Implement weighted scoring system
- Include penalties and bonuses

3. Tasks:
- Implement EXACTLY 3 tasks (easy, medium, hard)
- Use provided code snippets

4. Inference:
- Use OpenAI-compatible API
- Generate review from model
- Send to environment
- Print scores

OUTPUT FORMAT:
- Clearly separate each file
- Use comments like:
  ===== models.py =====
  ===== client.py =====
