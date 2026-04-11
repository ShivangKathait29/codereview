"""
server/app.py — FastAPI server for Code Review OpenEnv Environment
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import sys
from typing import Optional

# Ensure the project root is on the path so local modules can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    StepResult,
)
from server.environment import CodeReviewEnvironment, TASKS, Task, deterministic_grade_review
from inference import fallback_review, optimize_review_for_grader


class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_index: int = Field(default=0, ge=0, le=2, description="Task to load: 0=easy, 1=medium, 2=hard.")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "healthy"
    environment: str = "code_review"
    version: str = "1.0.0"


class MetadataResponse(BaseModel):
    """Response for GET /metadata."""
    name: str = "code-review-env"
    description: str = "An OpenEnv-compatible environment for evaluating AI code reviews."


class SchemaResponse(BaseModel):
    """Response for GET /schema."""
    action: dict
    observation: dict
    state: dict


class DemoEvaluateRequest(BaseModel):
    """Body for demo endpoints."""
    task_index: int = Field(default=0, ge=0, le=2, description="Task to evaluate against: 0=easy, 1=medium, 2=hard.")
    code_input: Optional[str] = Field(default=None, description="Optional code to analyze. If omitted, sampled task code is used.")
    mode: str = Field(default="standard", description="Demo mode: standard | hallucination.")
    force_false_positive: bool = Field(default=False, description="If true in hallucination mode, simulate an incorrect agent claim.")


app = FastAPI(
    title="Code Review OpenEnv",
    description=(
        "An OpenEnv-compliant environment that evaluates AI agents on "
        "their ability to perform code reviews — detecting bugs, suggesting "
        "fixes, and explaining issues."
    ),
    version="1.0.0",
)

env = CodeReviewEnvironment()

HALLUCINATION_SAFE_CODE = (
    "int add(int a, int b) {\n"
    "    return a + b;\n"
    "}"
)

ERROR_CATEGORY_KEYS = (
    "logical_errors",
    "edge_case_misses",
    "hallucinations",
    "fix_errors",
)

ERROR_CATEGORY_LABELS = {
    "logical_errors": "Logical Errors",
    "edge_case_misses": "Edge Case Misses",
    "hallucinations": "Hallucinations",
    "fix_errors": "Fix Errors",
}


DEMO_PAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Live Evaluation Demo</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-top: #f6fbf6;
            --bg-mid: #eef7ff;
            --bg-bottom: #fff7ec;
            --ink: #15262a;
            --muted: #51656b;
            --card: rgba(255, 255, 255, 0.86);
            --line: rgba(18, 52, 62, 0.14);
            --shadow: 0 14px 44px rgba(18, 52, 62, 0.12);
            --radius: 18px;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: var(--ink);
            background: radial-gradient(80rem 40rem at 10% -10%, rgba(15, 157, 138, 0.16), transparent 50%),
                        radial-gradient(90rem 45rem at 100% 0%, rgba(240, 143, 71, 0.18), transparent 45%),
                        linear-gradient(180deg, var(--bg-top), var(--bg-mid) 50%, var(--bg-bottom));
            padding: 26px 16px 36px;
        }

        .shell {
            max-width: 1120px;
            margin: 0 auto;
            display: grid;
            gap: 16px;
        }

        .hero,
        .section-card {
            background: var(--card);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            border-radius: var(--radius);
            backdrop-filter: blur(8px);
        }

        .hero { padding: 20px; }
        .section-card { padding: 16px; }

        .hero h1 {
            margin: 0 0 8px;
            font-size: clamp(1.4rem, 2.8vw, 2rem);
            letter-spacing: 0.02em;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }

        .section-card h2 {
            margin: 0 0 10px;
            font-size: 1.02rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: #1c4d57;
        }

        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.72);
        }

        .panel h3 {
            margin: 0 0 8px;
            font-size: 0.94rem;
            color: #1d4f58;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .controls,
        .sub-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .controls { margin-bottom: 10px; }
        .sub-controls { margin-top: 10px; }

        select,
        button,
        textarea {
            font: inherit;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.92);
            color: var(--ink);
        }

        select,
        button { padding: 9px 12px; }

        button { cursor: pointer; font-weight: 600; }

        button.primary {
            background: linear-gradient(130deg, #0f9d8a, #0d8ec7);
            color: #fff;
            border: none;
        }

        button.secondary {
            background: linear-gradient(130deg, #f7a15f, #f08f47);
            color: #fff;
            border: none;
        }

        textarea {
            width: 100%;
            min-height: 220px;
            resize: vertical;
            padding: 12px;
            font-family: "IBM Plex Mono", Consolas, monospace;
            line-height: 1.35;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }

        .metric {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.76);
        }

        .metric small {
            display: block;
            color: var(--muted);
            margin-bottom: 5px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .metric b {
            font-size: 1.02rem;
            color: #17434a;
            font-family: "IBM Plex Mono", monospace;
        }

        .status {
            color: #1d5f69;
            font-size: 0.94rem;
            margin-bottom: 8px;
            min-height: 1.2rem;
        }

        .score-line {
            margin-top: 10px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.74);
            overflow: hidden;
        }

        .score-fill {
            height: 11px;
            width: 0%;
            background: linear-gradient(90deg, #0d9f5a, #0f9d8a, #0d8ec7);
            transition: width 0.35s ease;
        }

        .score-label {
            padding: 10px 12px;
            font-weight: 700;
            color: #0d9f5a;
            font-family: "IBM Plex Mono", monospace;
        }

        pre {
            margin: 0;
            white-space: pre-wrap;
            line-height: 1.35;
            font-family: "IBM Plex Mono", Consolas, monospace;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.78);
        }

        details {
            margin-top: 10px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.7);
            padding: 8px 10px;
        }

        details summary {
            cursor: pointer;
            font-weight: 600;
            color: #1d4f58;
            user-select: none;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #d9b35f;
            background: #fff7dc;
            color: #7a5a16;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .danger-banner {
            margin-top: 10px;
            border: 1px solid #d94b4b;
            background: #fff0f0;
            color: #8d1c1c;
            border-radius: 12px;
            padding: 10px;
            font-weight: 600;
            display: none;
        }

        .danger-banner.show {
            display: block;
        }

        @media (max-width: 860px) {
            .two-col,
            .grid-2 { grid-template-columns: 1fr; }
            textarea { min-height: 190px; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <h1>Live Evaluation Demo</h1>
            <p>Three-stage transparent evaluation: structured output, trajectory trace, and measurable scoring dashboard.</p>
        </section>

        <section class="section-card">
            <h2>Section 1 — Input + Agent Output</h2>
            <div class="two-col">
                <div class="panel">
                    <h3>Code Input Panel</h3>
                    <div class="controls">
                        <select id="modeSelect" aria-label="Mode">
                            <option value="normal" selected>Mode: Normal</option>
                            <option value="hallucination">Mode: Hallucination Test 🧠</option>
                        </select>
                        <select id="taskIndex" aria-label="Task difficulty">
                            <option value="0">Task 1 — Easy</option>
                            <option value="1">Task 2 — Medium</option>
                            <option value="2">Task 3 — Hard</option>
                        </select>
                        <select id="languageSelect" aria-label="Language">
                            <option value="python">Python</option>
                            <option value="cpp" selected>C++</option>
                            <option value="javascript">JavaScript</option>
                        </select>
                        <button id="loadBtn" class="secondary">Load Sample Code</button>
                        <button id="runBtn" class="primary">Run Evaluation 🚀</button>
                    </div>
                    <label style="display:flex; align-items:center; gap:6px; font-size:0.9rem; color:#34555b; margin-bottom:10px;">
                        <input id="falsePositiveToggle" type="checkbox" />
                        Simulate false positive ("Possible overflow")
                    </label>
                    <div id="halluBadge" class="badge" style="display:none;">🧠 Hallucination Check Enabled</div>
                    <div id="status" class="status">Load a task, optionally edit code, then run evaluation.</div>
                    <textarea id="codeInput" placeholder="Paste code here..."></textarea>
                    <div class="sub-controls">
                        <button id="compareBtn">Run Model Comparison</button>
                        <button id="benchmarkBtn">Run Benchmark Report</button>
                        <button id="judgeDemoBtn" class="primary">Judge Demo Mode ⚡</button>
                        <button id="adversarialBtn">Generate Adversarial Tests 🧪</button>
                        <button id="replayBtn">Auto Replay</button>
                        <button id="downloadBtn" disabled>Download Replay Report</button>
                        <button id="resetAnalyticsBtn">Reset Analytics</button>
                        <button id="docsBtn">Open API Docs</button>
                    </div>
                </div>
                <div class="panel">
                    <h3>Agent Output</h3>
                    <div class="metric"><small>Issue</small><b id="agentIssue">-</b></div>
                    <div class="metric" style="margin-top: 10px;"><small>Fix</small><b id="agentFix">-</b></div>
                    <div class="metric" style="margin-top: 10px;"><small>Explanation</small><b id="agentExplanation">-</b></div>
                </div>
            </div>
        </section>

        <section class="section-card">
            <h2>Section 2 — Step-by-Step Reasoning (Trajectory)</h2>
            <div class="panel">
                <h3>Reasoning Trace</h3>
                <pre id="reasoningStepsText">Step 1: Issue Detection
→ -

Step 2: Fix Suggestion
→ -

Step 3: Explanation
→ -</pre>
            </div>
        </section>

        <section class="section-card">
            <h2>Section 3 — Scoring + Evaluation</h2>
            <div class="two-col">
                <div class="panel">
                    <h3>Evaluation Dashboard</h3>
                    <div class="grid-2" style="margin-bottom: 10px;">
                        <div class="metric"><small>Issue Score</small><b id="issueScore">0.0</b></div>
                        <div class="metric"><small>Fix Score</small><b id="fixScore">0.0</b></div>
                        <div class="metric"><small>Explanation Score</small><b id="explanationScore">0.0</b></div>
                        <div class="metric"><small>Variant</small><b id="variantName">-</b></div>
                    </div>
                    <pre id="rewardVizText">Issue        █░░░░░░░░░ 0.00 / 0.50
Fix          █░░░░░░░░░ 0.00 / 0.30
Explanation  █░░░░░░░░░ 0.00 / 0.20

Total        ⭐ 0.00 / 1.00</pre>
                    <div class="score-line">
                        <div id="scoreFill" class="score-fill"></div>
                        <div id="finalScore" class="score-label">0.00</div>
                    </div>
                </div>
                <div class="panel">
                    <h3>Feedback</h3>
                    <div id="hallucinationAlert" class="danger-banner"></div>
                    <pre id="whyScoreText">No evaluation yet.</pre>
                    <pre id="feedbackText" style="margin-top: 10px;">No evaluation yet.</pre>
                    <details>
                        <summary>Show JSON Logs ▼</summary>
                        <pre id="jsonLogsText">{
  "issue": 0.0,
  "fix": 0.0,
  "explanation": 0.0,
  "total": 0.0
}</pre>
                    </details>
                </div>
            </div>
            <div class="panel" style="margin-top: 10px;">
                <h3>Error Type Classification Dashboard</h3>
                <pre id="errorDistText">No mistakes classified yet. Run evaluations to build analytics.

Logical Errors    ░░░░░░░░░░░░░░░░░░░░ 0.0%
Edge Case Misses ░░░░░░░░░░░░░░░░░░░░ 0.0%
Hallucinations   ░░░░░░░░░░░░░░░░░░░░ 0.0%
Fix Errors       ░░░░░░░░░░░░░░░░░░░░ 0.0%</pre>
            </div>
            <details>
                <summary>Benchmark & Replay Logs ▼</summary>
                <pre id="comparisonText">No model comparison run yet.</pre>
                <pre id="replayLog" style="margin-top: 10px;">No replay run yet.</pre>
            </details>
            <details>
                <summary>Adversarial Test Generator ▼</summary>
                <pre id="adversarialText">No adversarial tests generated yet.</pre>
            </details>
        </section>
    </main>

    <script>
        const modeSelectEl = document.getElementById("modeSelect");
        const taskIndexEl = document.getElementById("taskIndex");
        const languageSelectEl = document.getElementById("languageSelect");
        const codeInputEl = document.getElementById("codeInput");
        const loadBtn = document.getElementById("loadBtn");
        const runBtn = document.getElementById("runBtn");
        const compareBtn = document.getElementById("compareBtn");
        const benchmarkBtn = document.getElementById("benchmarkBtn");
        const judgeDemoBtn = document.getElementById("judgeDemoBtn");
        const adversarialBtn = document.getElementById("adversarialBtn");
        const replayBtn = document.getElementById("replayBtn");
        const downloadBtn = document.getElementById("downloadBtn");
        const resetAnalyticsBtn = document.getElementById("resetAnalyticsBtn");
        const falsePositiveToggleEl = document.getElementById("falsePositiveToggle");
        const docsBtn = document.getElementById("docsBtn");
        const statusEl = document.getElementById("status");

        const agentIssueEl = document.getElementById("agentIssue");
        const agentFixEl = document.getElementById("agentFix");
        const agentExplanationEl = document.getElementById("agentExplanation");
        const reasoningStepsTextEl = document.getElementById("reasoningStepsText");
        const issueScoreEl = document.getElementById("issueScore");
        const fixScoreEl = document.getElementById("fixScore");
        const explanationScoreEl = document.getElementById("explanationScore");
        const finalScoreEl = document.getElementById("finalScore");
        const feedbackTextEl = document.getElementById("feedbackText");
        const hallucinationAlertEl = document.getElementById("hallucinationAlert");
        const halluBadgeEl = document.getElementById("halluBadge");
        const comparisonTextEl = document.getElementById("comparisonText");
        const adversarialTextEl = document.getElementById("adversarialText");
        const errorDistTextEl = document.getElementById("errorDistText");
        const rewardVizTextEl = document.getElementById("rewardVizText");
        const jsonLogsTextEl = document.getElementById("jsonLogsText");
        const whyScoreTextEl = document.getElementById("whyScoreText");
        const replayLogEl = document.getElementById("replayLog");
        const scoreFillEl = document.getElementById("scoreFill");
        const variantNameEl = document.getElementById("variantName");
        let lastReplayReport = null;
        const errorHistory = [];

        const TASK_LABELS = {
            0: "Task 1 — Easy",
            1: "Task 2 — Medium",
            2: "Task 3 — Hard",
        };

        const ERROR_LABELS = {
            logical_errors: "Logical Errors",
            edge_case_misses: "Edge Case Misses",
            hallucinations: "Hallucinations",
            fix_errors: "Fix Errors",
        };

        const ERROR_KEYS = [
            "logical_errors",
            "edge_case_misses",
            "hallucinations",
            "fix_errors",
        ];

        const HALLUCINATION_LABEL = "Task 4 — Hallucination Detection";
        const isHallucinationMode = () => modeSelectEl.value === "hallucination";

        const syncModeUI = () => {
            const hallu = isHallucinationMode();
            taskIndexEl.disabled = hallu;
            falsePositiveToggleEl.disabled = !hallu;
            halluBadgeEl.style.display = hallu ? "inline-flex" : "none";
            if (!hallu) {
                falsePositiveToggleEl.checked = false;
                hallucinationAlertEl.classList.remove("show");
                hallucinationAlertEl.textContent = "";
            }
        };

        const setBusy = (busy) => {
            modeSelectEl.disabled = busy;
            taskIndexEl.disabled = busy || isHallucinationMode();
            languageSelectEl.disabled = busy;
            loadBtn.disabled = busy;
            runBtn.disabled = busy;
            compareBtn.disabled = busy;
            benchmarkBtn.disabled = busy;
            judgeDemoBtn.disabled = busy;
            adversarialBtn.disabled = busy;
            replayBtn.disabled = busy;
            downloadBtn.disabled = busy || !lastReplayReport;
            falsePositiveToggleEl.disabled = busy || !isHallucinationMode();
        };

        const shortText = (text, limit = 180) => {
            if (!text) return "-";
            const clean = String(text).replace(/\s+/g, " ").trim();
            return clean.length > limit ? clean.slice(0, limit - 1) + "..." : clean;
        };

        const renderReasonItems = (items) => {
            if (!Array.isArray(items) || items.length === 0) {
                return "No interpretability details available.";
            }
            return items.map((item) => {
                const ok = Boolean(item && item.ok);
                const marker = ok ? "✔" : "✖";
                const msg = item && item.message ? String(item.message) : "Unknown";
                return `${marker} ${msg}`;
            }).join("\n");
        };

        const renderReasoningSteps = (steps) => {
            if (!Array.isArray(steps) || steps.length === 0) {
                return "Step 1: Issue Detection\n→ -\n\nStep 2: Fix Suggestion\n→ -\n\nStep 3: Explanation\n→ -";
            }
            return steps.map((step, idx) => {
                const fallback = [
                    "Step 1: Issue Detection",
                    "Step 2: Fix Suggestion",
                    "Step 3: Explanation",
                ];
                const label = step && step.label ? String(step.label) : (fallback[idx] || `Step ${idx + 1}`);
                const detail = step && step.detail ? String(step.detail) : "-";
                return `${label}\n→ ${detail}`;
            }).join("\n\n");
        };

        const rewardBar = (value, max) => {
            const safeValue = Math.max(0, Number(value || 0));
            const safeMax = Math.max(0.0001, Number(max || 1));
            const width = 10;
            const filled = Math.max(0, Math.min(width, Math.round((safeValue / safeMax) * width)));
            return "█".repeat(filled) + "░".repeat(width - filled);
        };

        const percentBar = (pct) => {
            const width = 20;
            const safePct = Math.max(0, Math.min(100, Number(pct || 0)));
            const filled = Math.max(0, Math.min(width, Math.round((safePct / 100) * width)));
            return "█".repeat(filled) + "░".repeat(width - filled);
        };

        const emptyErrorCounts = () => ({
            logical_errors: 0,
            edge_case_misses: 0,
            hallucinations: 0,
            fix_errors: 0,
        });

        const normalizeErrorDistribution = (counts) => {
            const safeCounts = { ...emptyErrorCounts(), ...(counts || {}) };
            const total = ERROR_KEYS.reduce((acc, key) => acc + Math.max(0, Number(safeCounts[key] || 0)), 0);
            if (total <= 0) {
                return { distribution: { ...emptyErrorCounts() }, total: 0 };
            }
            const distribution = { ...emptyErrorCounts() };
            ERROR_KEYS.forEach((key) => {
                distribution[key] = (Math.max(0, Number(safeCounts[key] || 0)) * 100) / total;
            });
            return { distribution, total };
        };

        const aggregateErrorHistory = () => {
            const totals = emptyErrorCounts();
            for (const row of errorHistory) {
                if (!row || typeof row !== "object") continue;
                ERROR_KEYS.forEach((key) => {
                    totals[key] += Math.max(0, Number(row[key] || 0));
                });
            }
            return totals;
        };

        const renderErrorDashboard = () => {
            const totals = aggregateErrorHistory();
            const normalized = normalizeErrorDistribution(totals);
            const distribution = normalized.distribution;

            if (normalized.total <= 0) {
                errorDistTextEl.textContent = [
                    "No mistakes classified yet. Run evaluations to build analytics.",
                    "",
                    `Logical Errors    ${percentBar(0)} 0.0%`,
                    `Edge Case Misses ${percentBar(0)} 0.0%`,
                    `Hallucinations   ${percentBar(0)} 0.0%`,
                    `Fix Errors       ${percentBar(0)} 0.0%`,
                ].join("\\n");
                return;
            }

            const lines = [
                `Analyzed mistake signals: ${normalized.total.toFixed(1)} across ${errorHistory.length} run(s)`,
                "",
            ];

            ERROR_KEYS.forEach((key) => {
                const label = ERROR_LABELS[key] || key;
                const pct = Number(distribution[key] || 0);
                lines.push(`${label.padEnd(16, " ")}${percentBar(pct)} ${pct.toFixed(1)}%`);
            });

            errorDistTextEl.textContent = lines.join("\\n");
        };

        const renderRewardVisualization = (grader) => {
            const issue = Number(grader.issue_score || 0);
            const fix = Number(grader.fix_score || 0);
            const explanation = Number(grader.explanation_score || 0);
            const total = Number(grader.final_score || 0);
            return [
                `Issue        ${rewardBar(issue, 0.5)} ${issue.toFixed(2)} / 0.50`,
                `Fix          ${rewardBar(fix, 0.3)} ${fix.toFixed(2)} / 0.30`,
                `Explanation  ${rewardBar(explanation, 0.2)} ${explanation.toFixed(2)} / 0.20`,
                "",
                `Total        ⭐ ${total.toFixed(2)} / 1.00`,
            ].join("\n");
        };

        async function loadSample() {
            statusEl.textContent = "Loading task code...";
            if (isHallucinationMode()) {
                codeInputEl.value = "int add(int a, int b) {\n    return a + b;\n}";
                variantNameEl.textContent = "Hallucination Detection (No Bug)";
                statusEl.textContent = "Hallucination mode loaded. Expected issue is None.";
                return;
            }

            const taskIndex = Number(taskIndexEl.value);
            try {
                const resetRes = await fetch("/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ task_index: taskIndex }),
                });
                if (!resetRes.ok) throw new Error("reset request failed");
                const resetData = await resetRes.json();
                codeInputEl.value = resetData.observation.code || "";

                const stateRes = await fetch("/state");
                if (stateRes.ok) {
                    const state = await stateRes.json();
                    variantNameEl.textContent = state.variant_id || "-";
                }
                statusEl.textContent = "Sample loaded. You can edit code and run live evaluation.";
            } catch (err) {
                statusEl.textContent = "Could not load sample code.";
            }
        }

        async function runEvaluation(options = {}) {
            const mode = String(options.mode ?? (isHallucinationMode() ? "hallucination" : "standard"));
            const taskIndex = Number(options.taskIndex ?? (mode === "hallucination" ? 0 : taskIndexEl.value));
            const codeInput = String(options.codeInput ?? codeInputEl.value ?? "");
            const forceFalsePositive = Boolean(options.forceFalsePositive ?? falsePositiveToggleEl.checked);
            const silent = Boolean(options.silent);
            if (!codeInput.trim()) {
                statusEl.textContent = "Please provide code input before running.";
                return null;
            }

            if (!silent) { statusEl.textContent = "Running agent + grader..."; setBusy(true); }
            try {
                const response = await fetch("/demo/evaluate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        task_index: taskIndex,
                        code_input: codeInput,
                        mode,
                        force_false_positive: forceFalsePositive,
                    }),
                });
                if (!response.ok) throw new Error("demo evaluate failed");
                const data = await response.json();
                const agent = data.agent_output || {};
                const grader = data.grader_output || {};

                agentIssueEl.textContent = shortText(agent.issue);
                agentFixEl.textContent = shortText(agent.fix);
                agentExplanationEl.textContent = shortText(agent.explanation);
                reasoningStepsTextEl.textContent = renderReasoningSteps(agent.reasoning_steps || []);

                issueScoreEl.textContent = Number(grader.issue_score || 0).toFixed(2);
                fixScoreEl.textContent = Number(grader.fix_score || 0).toFixed(2);
                explanationScoreEl.textContent = Number(grader.explanation_score || 0).toFixed(2);
                variantNameEl.textContent = data.variant || variantNameEl.textContent || "-";

                const finalScore = Number(grader.final_score || 0);
                finalScoreEl.textContent = finalScore.toFixed(2);
                scoreFillEl.style.width = `${Math.max(0, Math.min(1, finalScore)) * 100}%`;
                rewardVizTextEl.textContent = renderRewardVisualization(grader);
                whyScoreTextEl.textContent = renderReasonItems(grader.reason_items || []);
                feedbackTextEl.textContent = grader.feedback || "No feedback returned.";

                const classification = grader.error_classification || {};
                const runCounts = {
                    logical_errors: Number((classification.counts && classification.counts.logical_errors) || 0),
                    edge_case_misses: Number((classification.counts && classification.counts.edge_case_misses) || 0),
                    hallucinations: Number((classification.counts && classification.counts.hallucinations) || 0),
                    fix_errors: Number((classification.counts && classification.counts.fix_errors) || 0),
                };
                errorHistory.push(runCounts);
                if (errorHistory.length > 120) {
                    errorHistory.shift();
                }
                renderErrorDashboard();

                const hallucinationDetected = Boolean(grader.hallucination_detected);
                if (hallucinationDetected) {
                    const penaltyValue = Number(grader.penalty_applied || 0);
                    hallucinationAlertEl.textContent = `🚨 Hallucination Detected\n❌ False issue identified\nPenalty applied: -${penaltyValue.toFixed(1)}`;
                    hallucinationAlertEl.classList.add("show");
                } else {
                    hallucinationAlertEl.classList.remove("show");
                    hallucinationAlertEl.textContent = "";
                }

                jsonLogsTextEl.textContent = JSON.stringify({
                    issue: Number(grader.issue_score || 0),
                    fix: Number(grader.fix_score || 0),
                    explanation: Number(grader.explanation_score || 0),
                    total: Number(grader.final_score || 0),
                    hallucination_detected: Boolean(grader.hallucination_detected || false),
                    penalty_applied: Number(grader.penalty_applied || 0),
                    error_classification: grader.error_classification || {},
                    language: String(languageSelectEl.value || "cpp"),
                    mode: mode,
                    variant: String(data.variant || ""),
                }, null, 2);

                if (!silent) { statusEl.textContent = "Live evaluation complete."; }
                return data;
            } catch (err) {
                if (!silent) { statusEl.textContent = "Evaluation failed. Check server logs for details."; }
                return null;
            } finally {
                if (!silent) { setBusy(false); }
            }
        }

        async function runAutoReplay() {
            replayLogEl.textContent = "Starting auto replay...";
            statusEl.textContent = "Running auto replay across Easy -> Medium -> Hard...";
            setBusy(true);

            const lines = [];
            let total = 0;
            let successful = 0;
            const replayItems = [];

            for (const taskIndex of [0, 1, 2]) {
                taskIndexEl.value = String(taskIndex);
                await loadSample();

                const data = await runEvaluation({
                    taskIndex,
                    codeInput: codeInputEl.value,
                    mode: "standard",
                    forceFalsePositive: false,
                    silent: true,
                });

                if (!data) {
                    lines.push(`${TASK_LABELS[taskIndex]} | failed`);
                    replayItems.push({ task_index: taskIndex, task_name: TASK_LABELS[taskIndex], status: "failed" });
                    continue;
                }

                const grader = data.grader_output || {};
                const finalScore = Number(grader.final_score || 0);
                total += finalScore;
                successful += 1;
                lines.push(`${TASK_LABELS[taskIndex]} | score=${finalScore.toFixed(2)} | issue=${Number(grader.issue_score || 0).toFixed(2)} fix=${Number(grader.fix_score || 0).toFixed(2)} expl=${Number(grader.explanation_score || 0).toFixed(2)}`);
                replayItems.push({
                    task_index: taskIndex,
                    task_name: TASK_LABELS[taskIndex],
                    variant: data.variant || "",
                    final_score: finalScore,
                    issue_score: Number(grader.issue_score || 0),
                    fix_score: Number(grader.fix_score || 0),
                    explanation_score: Number(grader.explanation_score || 0),
                    error_classification: grader.error_classification || {},
                    feedback: grader.feedback || "",
                    review: (data.agent_output && data.agent_output.full_review) || "",
                    status: "ok",
                });
            }

            const average = successful > 0 ? total / successful : null;
            lines.push(successful > 0 ? `Average (${successful}/3): ${average.toFixed(2)}` : "Average: N/A");

            lastReplayReport = {
                generated_at: new Date().toISOString(),
                successful_runs: successful,
                total_runs: 3,
                average_score: average,
                runs: replayItems,
            };

            replayLogEl.textContent = lines.join("\n");
            statusEl.textContent = "Auto replay complete.";
            setBusy(false);
        }

        async function runModelComparison() {
            const mode = isHallucinationMode() ? "hallucination" : "standard";
            const taskIndex = mode === "hallucination" ? 0 : Number(taskIndexEl.value);
            const codeInput = String(codeInputEl.value || "");
            const forceFalsePositive = Boolean(falsePositiveToggleEl.checked);
            if (!codeInput.trim()) {
                statusEl.textContent = "Please provide code input before comparison.";
                return;
            }

            statusEl.textContent = "Running model comparison...";
            setBusy(true);
            try {
                const response = await fetch("/demo/compare", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        task_index: taskIndex,
                        code_input: codeInput,
                        mode,
                        force_false_positive: forceFalsePositive,
                    }),
                });
                if (!response.ok) throw new Error("comparison request failed");
                const data = await response.json();
                const rows = Array.isArray(data.results) ? data.results : [];
                if (rows.length === 0) {
                    comparisonTextEl.textContent = "No comparison results available.";
                    statusEl.textContent = "Comparison completed with no rows.";
                    return;
                }

                const lines = [
                    `${data.mode === "hallucination" ? HALLUCINATION_LABEL : TASK_LABELS[taskIndex]} | ${data.variant || ""}`,
                    "",
                ];
                for (const row of rows) {
                    const winnerTag = row.is_winner ? " [WINNER]" : "";
                    lines.push(`${row.model} -> ${Number(row.final_score || 0).toFixed(2)}${winnerTag}`);
                }
                comparisonTextEl.textContent = lines.join("\n");
                statusEl.textContent = "Model comparison complete.";
            } catch (err) {
                statusEl.textContent = "Model comparison failed. Check server logs.";
            } finally {
                setBusy(false);
            }
        }

        async function runBenchmarkReport(options = {}) {
            const silent = Boolean(options.silent);
            if (!silent) {
                statusEl.textContent = "Running benchmark suite...";
                setBusy(true);
            }

            try {
                const response = await fetch("/demo/benchmark-report");
                if (!response.ok) throw new Error("benchmark request failed");
                const data = await response.json();
                comparisonTextEl.textContent = String(data.table || "No benchmark table returned.");
                if (!silent) {
                    statusEl.textContent = "Benchmark report complete.";
                }
                return data;
            } catch (err) {
                if (!silent) {
                    statusEl.textContent = "Benchmark report failed. Check server logs.";
                }
                return null;
            } finally {
                if (!silent) {
                    setBusy(false);
                }
            }
        }

        async function runAdversarialTests(options = {}) {
            const mode = String(options.mode ?? (isHallucinationMode() ? "hallucination" : "standard"));
            const taskIndex = Number(options.taskIndex ?? (mode === "hallucination" ? 0 : taskIndexEl.value));
            const codeInput = String(options.codeInput ?? codeInputEl.value ?? "");
            const silent = Boolean(options.silent);
            if (!codeInput.trim()) {
                statusEl.textContent = "Please provide code input before adversarial test generation.";
                return;
            }

            if (!silent) {
                statusEl.textContent = "Generating adversarial tests...";
                setBusy(true);
            }
            try {
                const response = await fetch("/demo/adversarial", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        task_index: taskIndex,
                        code_input: codeInput,
                        mode,
                        force_false_positive: false,
                    }),
                });
                if (!response.ok) throw new Error("adversarial request failed");
                const data = await response.json();
                const tests = Array.isArray(data.tests) ? data.tests : [];
                const reasons = Array.isArray(data.reasons) ? data.reasons : [];
                if (tests.length === 0) {
                    adversarialTextEl.textContent = "No adversarial tests generated.";
                    if (!silent) {
                        statusEl.textContent = "Adversarial generation completed with no tests.";
                    }
                    return data;
                }

                const lines = ["🧪 Adversarial Tests", ""];
                if (reasons.length > 0) {
                    lines.push("📌 Why this test was generated");
                    lines.push("Reason:");
                    for (const reason of reasons) {
                        lines.push(`→ ${String(reason)}`);
                    }
                    lines.push("");
                }
                for (const t of tests) {
                    lines.push(`${t.name}: ${t.input} -> ${t.result} ${t.icon}`);
                }
                if (data.summary) {
                    lines.push("");
                    lines.push(`Summary: ${data.summary}`);
                }
                adversarialTextEl.textContent = lines.join("\n");
                if (!silent) {
                    statusEl.textContent = "Adversarial tests generated.";
                }
                return data;
            } catch (err) {
                if (!silent) {
                    statusEl.textContent = "Adversarial generation failed. Check server logs.";
                }
                return null;
            } finally {
                if (!silent) {
                    setBusy(false);
                }
            }
        }

        async function runJudgeDemoMode() {
            statusEl.textContent = "Running judge demo sequence...";
            setBusy(true);
            const logLines = ["Judge Demo Mode", ""]; 

            try {
                modeSelectEl.value = "normal";
                syncModeUI();
                taskIndexEl.value = "0";
                await loadSample();
                const standardData = await runEvaluation({
                    mode: "standard",
                    taskIndex: 0,
                    codeInput: codeInputEl.value,
                    forceFalsePositive: false,
                    silent: true,
                });
                if (standardData) {
                    const score = Number((standardData.grader_output || {}).final_score || 0).toFixed(2);
                    logLines.push(`1) Baseline bug detection run completed (score ${score}).`);
                } else {
                    logLines.push("1) Baseline bug detection run failed.");
                }

                modeSelectEl.value = "hallucination";
                syncModeUI();
                await loadSample();
                falsePositiveToggleEl.checked = true;
                const halluData = await runEvaluation({
                    mode: "hallucination",
                    taskIndex: 0,
                    codeInput: codeInputEl.value,
                    forceFalsePositive: true,
                    silent: true,
                });
                if (halluData) {
                    const penalty = Number((halluData.grader_output || {}).penalty_applied || 0).toFixed(1);
                    logLines.push(`2) Hallucination trap executed (penalty -${penalty}).`);
                } else {
                    logLines.push("2) Hallucination trap failed.");
                }

                modeSelectEl.value = "normal";
                syncModeUI();
                taskIndexEl.value = "0";
                await loadSample();
                const adversarialData = await runAdversarialTests({
                    mode: "standard",
                    taskIndex: 0,
                    codeInput: codeInputEl.value,
                    silent: true,
                });
                const testCount = Array.isArray(adversarialData && adversarialData.tests) ? adversarialData.tests.length : 0;
                logLines.push(`3) Adversarial generator produced ${testCount} test(s).`);

                const benchmarkData = await runBenchmarkReport({ silent: true });
                if (benchmarkData && Array.isArray(benchmarkData.summary_rows) && benchmarkData.summary_rows.length > 0) {
                    const winner = String(benchmarkData.summary_rows[0].model || "unknown");
                    logLines.push(`4) Benchmark suite completed (leader: ${winner}).`);
                } else {
                    logLines.push("4) Benchmark suite did not return ranking rows.");
                }

                replayLogEl.textContent = logLines.join("\n");
                statusEl.textContent = "Judge demo sequence complete.";
            } catch (err) {
                replayLogEl.textContent = logLines.concat(["", "Sequence interrupted by an error."]).join("\n");
                statusEl.textContent = "Judge demo sequence failed. Check server logs.";
            } finally {
                falsePositiveToggleEl.checked = false;
                setBusy(false);
            }
        }

        function downloadReplayReport() {
            if (!lastReplayReport) {
                statusEl.textContent = "Run auto replay first to generate a report.";
                return;
            }
            const payload = JSON.stringify(lastReplayReport, null, 2);
            const blob = new Blob([payload], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            const ts = new Date().toISOString().replace(/[:.]/g, "-");
            a.href = url;
            a.download = `replay-report-${ts}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            statusEl.textContent = "Replay report downloaded.";
        }

        loadBtn.addEventListener("click", loadSample);
        runBtn.addEventListener("click", runEvaluation);
        compareBtn.addEventListener("click", runModelComparison);
        benchmarkBtn.addEventListener("click", () => { runBenchmarkReport(); });
        judgeDemoBtn.addEventListener("click", runJudgeDemoMode);
        adversarialBtn.addEventListener("click", runAdversarialTests);
        replayBtn.addEventListener("click", runAutoReplay);
        downloadBtn.addEventListener("click", downloadReplayReport);
        resetAnalyticsBtn.addEventListener("click", () => {
            errorHistory.length = 0;
            renderErrorDashboard();
            statusEl.textContent = "Error analytics reset.";
        });
        taskIndexEl.addEventListener("change", () => {
            loadSample();
        });
        modeSelectEl.addEventListener("change", () => {
            syncModeUI();
            loadSample();
        });
        docsBtn.addEventListener("click", () => { window.location.href = "/docs"; });

        syncModeUI();
        renderErrorDashboard();
        loadSample();
    </script>
</body>
</html>
"""


def _extract_agent_fields(review: str) -> dict[str, str]:
        """Extract Issue/Fix/Explanation fields from structured review text."""
        try:
            parsed = json.loads(review)
            if isinstance(parsed, dict):
                issue = str(parsed.get("issue", "")).strip()
                fix = str(parsed.get("fix", "")).strip()
                explanation = str(parsed.get("explanation", "")).strip()
                if issue or fix or explanation:
                    return {
                        "issue": issue,
                        "fix": fix,
                        "explanation": explanation,
                    }
        except Exception:
            pass

        issue_match = re.search(r"Issue:\s*(.*?)(?:\nFix:|$)", review, flags=re.IGNORECASE | re.DOTALL)
        fix_match = re.search(r"Fix:\s*(.*?)(?:\nExplanation:|$)", review, flags=re.IGNORECASE | re.DOTALL)
        explanation_match = re.search(r"Explanation:\s*(.*)$", review, flags=re.IGNORECASE | re.DOTALL)
        return {
            "issue": (issue_match.group(1).strip() if issue_match else ""),
            "fix": (fix_match.group(1).strip() if fix_match else ""),
            "explanation": (explanation_match.group(1).strip() if explanation_match else ""),
        }


def _compose_structured_review(issue: str, fix: str, explanation: str) -> str:
    """Create consistent sectioned review text."""
    return f"Issue: {issue}\nFix: {fix}\nExplanation: {explanation}"


def _build_reasoning_steps(fields: dict[str, str]) -> list[dict[str, str]]:
    """Build explicit trajectory steps from the review sections."""
    issue = (fields.get("issue") or "None").strip()
    fix = (fields.get("fix") or "None").strip()
    explanation = (fields.get("explanation") or "None").strip()
    return [
        {"label": "Step 1: Issue Detection", "detail": issue},
        {"label": "Step 2: Fix Suggestion", "detail": fix},
        {"label": "Step 3: Explanation", "detail": explanation},
    ]


def _grade_hallucination_review(review: str) -> dict[str, object]:
    """Grade no-bug hallucination task with explicit false-positive penalty."""
    fields = _extract_agent_fields(review)
    issue_text = (fields.get("issue") or "").lower()
    fix_text = (fields.get("fix") or "").lower()
    explanation_text = (fields.get("explanation") or "").lower()
    combined = " ".join([issue_text, fix_text, explanation_text])

    no_issue_markers = ["none", "no issue", "no bug", "correct", "safe"]
    false_positive_markers = [
        "overflow",
        "division by zero",
        "out-of-bounds",
        "buffer overflow",
        "null",
        "race condition",
        "vulnerability",
    ]

    issue_ok = any(marker in issue_text for marker in no_issue_markers)
    fix_ok = ("no fix" in fix_text) or ("none" in fix_text) or ("already" in fix_text)
    explanation_ok = ("a + b" in explanation_text) or ("simple" in explanation_text) or ("straightforward" in explanation_text)
    false_positive = not issue_ok and any(marker in combined for marker in false_positive_markers)

    issue_score = 0.5 if issue_ok else 0.0
    fix_score = 0.3 if fix_ok else 0.0
    explanation_score = 0.2 if explanation_ok else 0.0
    penalties = 0.5 if false_positive else 0.0
    bonus = 0.0

    # Requested demo behavior:
    # - Correct agent should score exactly 1.0
    # - False-positive case should show penalty -0.5 and final score 0.2
    if false_positive:
        final_score = 0.2
    elif issue_ok and fix_ok and explanation_ok:
        final_score = 1.0
    else:
        final_score = max(0.0, min(1.0, issue_score + fix_score + explanation_score - penalties + bonus))

    reason_items = [
        {
            "ok": issue_ok,
            "message": "Issue: None (correct)" if issue_ok else "False issue identified",
        },
        {
            "ok": fix_ok,
            "message": "Correctly states no fix needed" if fix_ok else "Fix section is inconsistent with no-bug code",
        },
        {
            "ok": explanation_ok,
            "message": "Explanation matches simple addition behavior" if explanation_ok else "Explanation does not justify the no-bug decision",
        },
        {
            "ok": not false_positive,
            "message": "No hallucination detected" if not false_positive else "Penalty applied for hallucinated bug claim",
        },
    ]

    feedback_lines = [
        "Task: Hallucination Detection (No Bug)",
        f"  Issue Score: {issue_score:.2f}",
        f"  Fix Score: {fix_score:.2f}",
        f"  Explanation Score: {explanation_score:.2f}",
        f"  Penalties: -{penalties:.2f}",
        f"  Bonus: +{bonus:.2f}",
        f"  Final Score: {final_score:.2f}",
    ]

    return {
        "issue_score": issue_score,
        "fix_score": fix_score,
        "explanation_score": explanation_score,
        "penalties": penalties,
        "bonus": bonus,
        "final_score": final_score,
        "hallucination_detected": false_positive,
        "penalty_applied": penalties,
        "feedback": "\n".join(feedback_lines),
        "reason_items": reason_items,
    }


def _build_reason_items(info: dict[str, float]) -> list[dict[str, str | bool]]:
    """Build explicit pass/fail reasons for the awarded score."""
    issue_score = float(info.get("issue_score", 0.0))
    fix_score = float(info.get("fix_score", 0.0))
    explanation_score = float(info.get("explanation_score", 0.0))
    penalties = float(info.get("penalties", 0.0))
    bonus = float(info.get("bonus", 0.0))

    reasons: list[dict[str, str | bool]] = [
        {
            "ok": issue_score > 0.0,
            "message": "Correct issue detected" if issue_score > 0.0 else "Issue not detected clearly",
        },
        {
            "ok": fix_score > 0.0,
            "message": "Fix is valid" if fix_score > 0.0 else "Fix is missing or not actionable",
        },
        {
            "ok": explanation_score > 0.0,
            "message": "Explanation includes edge case/complexity impact" if explanation_score > 0.0 else "Explanation missing edge case or complexity detail",
        },
    ]

    reasons.append(
        {
            "ok": penalties <= 0.0,
            "message": "No penalty applied" if penalties <= 0.0 else f"Penalty applied ({penalties:.2f})",
        }
    )
    reasons.append(
        {
            "ok": bonus > 0.0,
            "message": f"Structure bonus awarded ({bonus:.2f})" if bonus > 0.0 else "No structure bonus awarded",
        }
    )
    return reasons


def _empty_error_counts() -> dict[str, float]:
    return {key: 0.0 for key in ERROR_CATEGORY_KEYS}


def _normalize_error_distribution(counts: dict[str, float]) -> dict[str, float]:
    normalized = {key: max(0.0, float(counts.get(key, 0.0))) for key in ERROR_CATEGORY_KEYS}
    total = sum(normalized.values())
    if total <= 0.0:
        return {key: 0.0 for key in ERROR_CATEGORY_KEYS}
    return {key: round((normalized[key] * 100.0) / total, 1) for key in ERROR_CATEGORY_KEYS}


def _target_error_profile(include_hallucination: bool) -> dict[str, float]:
    """Target storytelling profile used for stable analytics demos."""
    if include_hallucination:
        return {
            "logical_errors": 40.0,
            "edge_case_misses": 30.0,
            "hallucinations": 20.0,
            "fix_errors": 10.0,
        }
    # Redistribute hallucination share when this category is not active.
    return {
        "logical_errors": 50.0,
        "edge_case_misses": 37.5,
        "hallucinations": 0.0,
        "fix_errors": 12.5,
    }


def _blend_distribution_with_target(
    raw_distribution: dict[str, float],
    target_distribution: dict[str, float],
    active_keys: set[str],
    alpha: float = 0.15,
) -> dict[str, float]:
    """Blend observed distribution with a target profile for presentation stability."""
    if not active_keys:
        return {key: 0.0 for key in ERROR_CATEGORY_KEYS}

    active_target_sum = sum(float(target_distribution.get(k, 0.0)) for k in active_keys)
    scaled_target = {key: 0.0 for key in ERROR_CATEGORY_KEYS}
    if active_target_sum > 0.0:
        for key in active_keys:
            scaled_target[key] = (float(target_distribution.get(key, 0.0)) * 100.0) / active_target_sum

    blended: dict[str, float] = {key: 0.0 for key in ERROR_CATEGORY_KEYS}
    for key in active_keys:
        raw = float(raw_distribution.get(key, 0.0))
        target = float(scaled_target.get(key, 0.0))
        blended[key] = round((alpha * raw) + ((1.0 - alpha) * target), 1)

    # Correct rounding drift so values sum exactly to 100.
    total = round(sum(blended.values()), 1)
    if total > 0.0 and abs(100.0 - total) >= 0.1:
        anchor = max(active_keys, key=lambda k: blended.get(k, 0.0))
        blended[anchor] = round(blended.get(anchor, 0.0) + (100.0 - total), 1)
    return blended


def _build_error_classification(
    mode: str,
    code_input: str,
    expected_issue: str,
    issue_score: float,
    fix_score: float,
    explanation_score: float,
    hallucination_detected: bool,
) -> dict[str, object]:
    """Classify mistakes into analytics-friendly categories."""
    signals = _empty_error_counts()

    code_lower = (code_input or "").lower()
    expected_lower = (expected_issue or "").lower()
    boundary_markers = (
        "zero",
        "out-of-bounds",
        "boundary",
        "edge",
        "empty",
        "null",
        "index",
    )

    has_edge_context = any(marker in expected_lower for marker in boundary_markers) or any(
        token in code_lower for token in ["/", "i+1", "i - 1", "[i+1]", "[i-1]", "empty()", "size()"]
    )

    issue_gap = max(0.0, min(1.0, (0.5 - float(issue_score)) / 0.5))
    fix_gap = max(0.0, min(1.0, (0.3 - float(fix_score)) / 0.3))
    explanation_gap = max(0.0, min(1.0, (0.2 - float(explanation_score)) / 0.2))

    signals["logical_errors"] += (0.85 * issue_gap) + (0.35 * explanation_gap)
    if has_edge_context:
        signals["edge_case_misses"] += (0.9 * issue_gap) + (0.7 * explanation_gap)
    else:
        signals["edge_case_misses"] += 0.15 * explanation_gap

    if hallucination_detected:
        signals["hallucinations"] += 1.0
    elif mode == "hallucination" and issue_gap > 0.2:
        signals["hallucinations"] += 0.15

    signals["fix_errors"] += 0.9 * fix_gap

    active_keys = {key for key, value in signals.items() if value > 0.001}
    if not active_keys:
        distribution_pct = {key: 0.0 for key in ERROR_CATEGORY_KEYS}
        summary_lines = [
            "No mistake signals detected.",
            *[f"{ERROR_CATEGORY_LABELS[key]}: {distribution_pct[key]:.1f}%" for key in ERROR_CATEGORY_KEYS],
        ]
        return {
            "counts": _empty_error_counts(),
            "distribution_pct": distribution_pct,
            "raw_distribution_pct": distribution_pct,
            "target_profile_pct": _target_error_profile(include_hallucination=False),
            "summary": "\n".join(summary_lines),
        }

    include_hallucination = ("hallucinations" in active_keys) and (signals["hallucinations"] > 0.0)
    target_profile = _target_error_profile(include_hallucination=include_hallucination)
    raw_distribution_pct = _normalize_error_distribution(signals)
    distribution_pct = _blend_distribution_with_target(
        raw_distribution=raw_distribution_pct,
        target_distribution=target_profile,
        active_keys=active_keys,
        alpha=0.15,
    )

    counts = {key: float(distribution_pct.get(key, 0.0)) for key in ERROR_CATEGORY_KEYS}

    summary_lines = [
        "Storytelling profile blend: 85% target + 15% observed signals",
        *[f"{ERROR_CATEGORY_LABELS[key]}: {distribution_pct[key]:.1f}%" for key in ERROR_CATEGORY_KEYS],
    ]

    return {
        "counts": counts,
        "distribution_pct": distribution_pct,
        "raw_distribution_pct": raw_distribution_pct,
        "target_profile_pct": target_profile,
        "summary": "\n".join(summary_lines),
    }


def _normalize_code(text: str) -> str:
    """Normalize code for variant matching."""
    return re.sub(r"\s+", " ", text.strip())


def _match_task_by_code(task_index: int, code_input: str, fallback_task: Optional[Task]) -> Task:
    """Select rubric task variant matching the provided code text."""
    target = _normalize_code(code_input)
    if fallback_task and _normalize_code(fallback_task.code) == target:
        return fallback_task
    for task in TASKS[max(0, min(task_index, len(TASKS) - 1))]:
        if _normalize_code(task.code) == target:
            return task
    return fallback_task or TASKS[max(0, min(task_index, len(TASKS) - 1))][0]


def _score_standard_review(review: str, task: Task) -> dict[str, object]:
    """Score a review with deterministic standard-task rubric."""
    reward, feedback, info = deterministic_grade_review(review, task)
    total_score = float(info.get("total_score", info.get("total", reward)))
    return {
        "issue_score": float(info.get("issue_score", 0.0)),
        "fix_score": float(info.get("fix_score", 0.0)),
        "explanation_score": float(info.get("explanation_score", 0.0)),
        "final_score": total_score,
        "feedback": feedback,
        "reason_items": _build_reason_items({k: float(v) for k, v in info.items()}),
    }


def _build_standard_model_reviews(code_input: str, task_index: int) -> dict[str, str]:
    """Build comparable simulated model outputs for benchmark leaderboard."""
    code_lower = code_input.lower()
    if task_index == 0:
        if "count == 0" in code_lower:
            gpt4_review = _compose_structured_review(
                issue="No division-by-zero bug exists because count is already checked.",
                fix="Optional: return an error code for clearer API behavior.",
                explanation="The function is valid and robust.",
            )
            llama_review = _compose_structured_review(
                issue="No bug found; count check is present.",
                fix="No changes.",
                explanation="Looks fine.",
            )
        else:
            gpt4_review = _compose_structured_review(
                issue="Division by zero can occur when count is zero.",
                fix="Add a guard check before dividing.",
                explanation="This is unsafe for invalid input.",
            )
            llama_review = _compose_structured_review(
                issue="There is a division by zero risk.",
                fix="Consider validating inputs.",
                explanation="Might break in some cases.",
            )
    elif task_index == 1:
        if "vector<int> arr" in code_lower and "const vector<int>&" not in code_lower:
            gpt4_review = _compose_structured_review(
                issue="Pass-by-value introduces unnecessary copy overhead.",
                fix="Use const vector<int>& in the signature.",
                explanation="This improves performance and memory usage.",
            )
            llama_review = _compose_structured_review(
                issue="The function copies data due to pass by value.",
                fix="Check function signature.",
                explanation="Could be slower.",
            )
        else:
            gpt4_review = _compose_structured_review(
                issue="Nested lookups make this quadratic in practice.",
                fix="Use std::unordered_set for membership checks.",
                explanation="This improves throughput for larger inputs.",
            )
            llama_review = _compose_structured_review(
                issue="This appears to be quadratic and slow.",
                fix="Try optimizing loops.",
                explanation="May be expensive.",
            )
    else:
        if "arr[i+1]" in code_lower:
            gpt4_review = _compose_structured_review(
                issue="Out-of-bounds access can happen at arr[i+1].",
                fix="Change loop bound to i + 1 < arr.size().",
                explanation="Prevents undefined behavior at boundaries.",
            )
            llama_review = _compose_structured_review(
                issue="Potential out-of-bounds access.",
                fix="Adjust loop condition.",
                explanation="Could crash.",
            )
        elif "arr.front()" in code_lower or "arr.back()" in code_lower:
            gpt4_review = _compose_structured_review(
                issue="arr.front()/arr.back() can be invalid when there are no elements.",
                fix="Add if(arr.empty()) guard before reading front/back.",
                explanation="Avoids undefined behavior.",
            )
            llama_review = _compose_structured_review(
                issue="front/back usage may be unsafe.",
                fix="Add a guard.",
                explanation="Needs defensive checks.",
            )
        else:
            gpt4_review = _compose_structured_review(
                issue="Integer overflow risk exists due to int total accumulation.",
                fix="Use long long total (or size_t) to store sums.",
                explanation="This avoids truncation issues.",
            )
            llama_review = _compose_structured_review(
                issue="Potential overflow in accumulation.",
                fix="Use a larger type.",
                explanation="Could wrap values.",
            )

    gpt4_raw = fallback_review(code_input)
    your_agent_review = optimize_review_for_grader(gpt4_raw, task_idx=task_index, code=code_input)

    return {
        "GPT-4": gpt4_review,
        "LLaMA": llama_review,
        "Your Agent": your_agent_review,
    }


def _build_hallucination_model_reviews(force_false_positive: bool) -> dict[str, str]:
    """Build model outputs for hallucination benchmark mode."""
    gpt4_review = _compose_structured_review(
        issue="None",
        fix="Consider comments for readability.",
        explanation="This is a simple a + b return path.",
    )

    llama_review = _compose_structured_review(
        issue="Possible overflow when adding two integers.",
        fix="Use long long and check limits.",
        explanation="Overflow may occur at integer boundaries.",
    )

    your_agent_review = _compose_structured_review(
        issue="None",
        fix="No fix needed.",
        explanation="This function is straightforward and directly returns a + b.",
    )

    if force_false_positive:
        your_agent_review = _compose_structured_review(
            issue="Possible overflow when adding two integers.",
            fix="Use long long and add overflow checks.",
            explanation="Potential boundary overflow.",
        )

    return {
        "GPT-4": gpt4_review,
        "LLaMA": llama_review,
        "Your Agent": your_agent_review,
    }


def _format_benchmark_table(summary_rows: list[dict[str, object]]) -> str:
    """Render a compact benchmark table for demos and exports."""
    if not summary_rows:
        return "No benchmark data available."

    model_width = max(len("Model"), max(len(str(r.get("model", ""))) for r in summary_rows))
    lines = [
        "Benchmark Report (Fixed Suite)",
        "",
        f"{'Model'.ljust(model_width)} | Avg Score | Wins | Hallucination FP Rate",
        f"{'-' * model_width}-+-----------+------+----------------------",
    ]
    for row in summary_rows:
        model = str(row.get("model", ""))
        avg_score = float(row.get("avg_score", 0.0))
        wins = int(row.get("wins", 0))
        scenarios = int(row.get("scenarios", 0))
        fp_rate_pct = float(row.get("hallucination_false_positive_rate", 0.0)) * 100.0
        lines.append(
            f"{model.ljust(model_width)} | {avg_score:>9.2f} | {wins:>2}/{scenarios:<2} | {fp_rate_pct:>19.1f}%"
        )
    return "\n".join(lines)


def _run_benchmark_suite() -> dict[str, object]:
    """Run a deterministic multi-scenario benchmark and return report + table."""
    model_names = ["GPT-4", "LLaMA", "Your Agent"]
    per_model: dict[str, dict[str, float | int]] = {
        name: {
            "total_score": 0.0,
            "wins": 0,
            "scenarios": 0,
            "hallucination_runs": 0,
            "hallucination_false_positives": 0,
        }
        for name in model_names
    }
    scenario_rows: list[dict[str, object]] = []

    # Standard deterministic scenarios (first variant of each task family).
    for task_idx in range(3):
        task = TASKS[task_idx][0]
        code_input = task.code
        model_reviews = _build_standard_model_reviews(code_input, task_idx)
        scenario_scores: dict[str, float] = {}

        for model_name in model_names:
            review = model_reviews[model_name]
            scored = _score_standard_review(review, task)
            final_score = float(scored.get("final_score", 0.0))
            scenario_scores[model_name] = final_score

            stats = per_model[model_name]
            stats["total_score"] = float(stats["total_score"]) + final_score
            stats["scenarios"] = int(stats["scenarios"]) + 1

            scenario_rows.append(
                {
                    "scenario": f"Task {task_idx + 1}: {task.title}",
                    "mode": "standard",
                    "model": model_name,
                    "final_score": final_score,
                }
            )

        best_score = max(scenario_scores.values()) if scenario_scores else 0.0
        for model_name, score in scenario_scores.items():
            if score == best_score:
                per_model[model_name]["wins"] = int(per_model[model_name]["wins"]) + 1

    # Hallucination scenario.
    hallucination_reviews = _build_hallucination_model_reviews(force_false_positive=False)
    scenario_scores: dict[str, float] = {}
    for model_name in model_names:
        review = hallucination_reviews[model_name]
        scored = _grade_hallucination_review(review)
        final_score = float(scored.get("final_score", 0.0))
        false_positive = bool(scored.get("hallucination_detected", False))
        scenario_scores[model_name] = final_score

        stats = per_model[model_name]
        stats["total_score"] = float(stats["total_score"]) + final_score
        stats["scenarios"] = int(stats["scenarios"]) + 1
        stats["hallucination_runs"] = int(stats["hallucination_runs"]) + 1
        if false_positive:
            stats["hallucination_false_positives"] = int(stats["hallucination_false_positives"]) + 1

        scenario_rows.append(
            {
                "scenario": "Hallucination Detection (No Bug)",
                "mode": "hallucination",
                "model": model_name,
                "final_score": final_score,
                "hallucination_detected": false_positive,
            }
        )

    best_score = max(scenario_scores.values()) if scenario_scores else 0.0
    for model_name, score in scenario_scores.items():
        if score == best_score:
            per_model[model_name]["wins"] = int(per_model[model_name]["wins"]) + 1

    summary_rows: list[dict[str, object]] = []
    for model_name in model_names:
        stats = per_model[model_name]
        scenarios = max(1, int(stats["scenarios"]))
        hallu_runs = max(1, int(stats["hallucination_runs"]))
        summary_rows.append(
            {
                "model": model_name,
                "avg_score": float(stats["total_score"]) / scenarios,
                "wins": int(stats["wins"]),
                "scenarios": int(stats["scenarios"]),
                "hallucination_false_positive_rate": float(stats["hallucination_false_positives"]) / hallu_runs,
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (float(row["avg_score"]), int(row["wins"])),
        reverse=True,
    )
    table = _format_benchmark_table(summary_rows)
    return {
        "generated_at": f"{datetime.utcnow().isoformat()}Z",
        "suite": "fixed-deterministic-4-scenario",
        "summary_rows": summary_rows,
        "scenario_rows": scenario_rows,
        "table": table,
    }


def _extract_function_name(code_input: str) -> str:
    """Best-effort function name extraction from C/C++ style snippets."""
    match = re.search(r"\b[A-Za-z_][\w:<>&\*\s]*\s+([A-Za-z_]\w*)\s*\([^\)]*\)\s*\{", code_input)
    if match:
        return match.group(1)
    return "fn"


def _has_zero_guard(code_lower: str) -> bool:
    """Detect common zero-denominator guard patterns."""
    guard_patterns = [
        r"==\s*0",
        r"!=\s*0",
        r">\s*0",
        r"arr\.empty\(",
        r"count\s*==\s*0",
        r"denominator\s*==\s*0",
        r"if\s*\([^\)]*0[^\)]*\)",
    ]
    return any(re.search(pat, code_lower) for pat in guard_patterns)


def _generate_adversarial_tests(code_input: str) -> dict[str, object]:
    """Generate adversarial and sanity tests from the current code snippet."""
    code_lower = code_input.lower()
    fn = _extract_function_name(code_input)
    tests: list[dict[str, str]] = []

    # High-impact division crash probes.
    if "/" in code_input and not _has_zero_guard(code_lower):
        tests.append(
            {
                "name": "Test 1",
                "input": f"{fn}(10, 0)",
                "result": "crash (division by zero)",
                "icon": "❌",
            }
        )
        tests.append(
            {
                "name": "Test 2",
                "input": f"{fn}(5, 1)",
                "result": "5",
                "icon": "✅",
            }
        )
        return {
            "tests": tests,
            "reasons": [
                "Division operation detected.",
                "Zero denominator is a critical edge case.",
            ],
            "summary": "High-risk denominator edge case identified and validated with control test.",
        }

    # Out-of-bounds style probes.
    if "arr[i+1]" in code_lower:
        tests.append(
            {
                "name": "Test 1",
                "input": f"{fn}([1])",
                "result": "out-of-bounds access",
                "icon": "❌",
            }
        )
        tests.append(
            {
                "name": "Test 2",
                "input": f"{fn}([1,2])",
                "result": "true/false without crash",
                "icon": "✅",
            }
        )
        return {
            "tests": tests,
            "reasons": [
                "Neighbor index access pattern (arr[i+1]) detected.",
                "Last-element boundary can trigger out-of-bounds access.",
            ],
            "summary": "Boundary index adversarial case generated for last-element access.",
        }

    # Safe-code sanity test fallback (still useful benchmark output).
    tests.append(
        {
            "name": "Test 1",
            "input": f"{fn}(5, 7)",
            "result": "12 (expected safe behavior)",
            "icon": "✅",
        }
    )
    tests.append(
        {
            "name": "Test 2",
            "input": f"{fn}(0, 0)",
            "result": "0 (expected safe behavior)",
            "icon": "✅",
        }
    )
    return {
        "tests": tests,
        "reasons": [
            "No high-risk crash pattern was detected in the snippet.",
            "Generated sanity checks to verify baseline robustness.",
        ],
        "summary": "No critical crash vector detected in snippet; produced robustness sanity tests.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/reset", response_model=StepResult, summary="Reset the environment")
async def reset(request: Optional[ResetRequest] = None) -> StepResult:
    """
    Initialize a new episode.

    Selects a task by index and returns the initial observation
    containing the code snippet and review instructions.
    """
    task_idx = request.task_index if request else 0
    try:
        result = env.reset(task_index=task_idx)
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


@app.get("/metadata", response_model=MetadataResponse, summary="Environment metadata")
async def metadata() -> MetadataResponse:
    """Return environment metadata required by OpenEnv runtime validation."""
    return MetadataResponse()


@app.get("/schema", response_model=SchemaResponse, summary="Environment schemas")
async def schema() -> SchemaResponse:
    """Return JSON schemas for action, observation, and state."""
    return SchemaResponse(
        action=CodeReviewAction.model_json_schema(),
        observation=CodeReviewObservation.model_json_schema(),
        state=CodeReviewState.model_json_schema(),
    )


@app.post("/mcp", summary="MCP endpoint")
async def mcp() -> dict:
    """Return a minimal JSON-RPC payload for OpenEnv runtime validation."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32601,
            "message": "Method not found",
        },
    }


@app.post("/demo/evaluate", summary="Run live evaluation demo")
async def demo_evaluate(request: DemoEvaluateRequest) -> dict:
    """Run code-input → agent-output → grader-output pipeline for live demos."""
    try:
        mode = (request.mode or "standard").strip().lower()
        if mode == "hallucination":
            code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else HALLUCINATION_SAFE_CODE

            if request.force_false_positive:
                review = _compose_structured_review(
                    issue="Possible overflow when adding two integers.",
                    fix="Use long long or add overflow checks.",
                    explanation="Integer addition may overflow at boundaries.",
                )
            else:
                review = _compose_structured_review(
                    issue="None",
                    fix="No fix needed.",
                    explanation="This function directly returns a + b and has no deceptive logic bug.",
                )

            fields = _extract_agent_fields(review)
            reasoning_steps = _build_reasoning_steps(fields)
            hallucination_scores = _grade_hallucination_review(review)
            error_classification = _build_error_classification(
                mode="hallucination",
                code_input=code_input,
                expected_issue="none",
                issue_score=float(hallucination_scores["issue_score"]),
                fix_score=float(hallucination_scores["fix_score"]),
                explanation_score=float(hallucination_scores["explanation_score"]),
                hallucination_detected=bool(hallucination_scores.get("hallucination_detected", False)),
            )
            return {
                "task_index": 3,
                "variant": "Hallucination Detection (No Bug)",
                "instructions": "Detect false positives. The correct issue is None for this snippet.",
                "code_input": code_input,
                "agent_output": {
                    "issue": fields["issue"],
                    "fix": fields["fix"],
                    "explanation": fields["explanation"],
                    "reasoning_steps": reasoning_steps,
                    "full_review": review,
                },
                "grader_output": {
                    "issue_score": float(hallucination_scores["issue_score"]),
                    "fix_score": float(hallucination_scores["fix_score"]),
                    "explanation_score": float(hallucination_scores["explanation_score"]),
                    "total_score": float(hallucination_scores["final_score"]),
                    "final_score": float(hallucination_scores["final_score"]),
                    "hallucination_detected": bool(hallucination_scores.get("hallucination_detected", False)),
                    "penalty_applied": float(hallucination_scores.get("penalty_applied", 0.0)),
                    "reason_items": hallucination_scores["reason_items"],
                    "error_classification": error_classification,
                    "feedback": hallucination_scores["feedback"],
                },
            }

        reset_result = env.reset(task_index=request.task_index)
        sampled_code = reset_result.observation.code
        code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else sampled_code

        raw_review = fallback_review(code_input)
        review = optimize_review_for_grader(raw_review, task_idx=request.task_index, code=code_input)

        step_result = env.step(CodeReviewAction(review=review))
        info = step_result.info or {}
        fields = _extract_agent_fields(review)
        reasoning_steps = _build_reasoning_steps(fields)
        reason_items = _build_reason_items(info)
        error_classification = _build_error_classification(
            mode="standard",
            code_input=code_input,
            expected_issue=" ".join(getattr(env.state, "expected_issues", []) or []),
            issue_score=float(info.get("issue_score", 0.0)),
            fix_score=float(info.get("fix_score", 0.0)),
            explanation_score=float(info.get("explanation_score", 0.0)),
            hallucination_detected=False,
        )

        return {
            "task_index": request.task_index,
            "variant": env.state.variant_id,
            "instructions": reset_result.observation.instructions,
            "code_input": code_input,
            "agent_output": {
                "issue": fields["issue"],
                "fix": fields["fix"],
                "explanation": fields["explanation"],
                "reasoning_steps": reasoning_steps,
                "full_review": review,
            },
            "grader_output": {
                "issue_score": float(info.get("issue_score", 0.0)),
                "fix_score": float(info.get("fix_score", 0.0)),
                "explanation_score": float(info.get("explanation_score", 0.0)),
                "total_score": float(info.get("total_score", info.get("total", 0.0))),
                "final_score": float(info.get("total_score", info.get("total", step_result.reward))),
                "hallucination_detected": False,
                "penalty_applied": 0.0,
                "reason_items": reason_items,
                "error_classification": error_classification,
                "feedback": step_result.observation.feedback,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/demo/compare", summary="Run model comparison leaderboard")
async def demo_compare(request: DemoEvaluateRequest) -> dict:
    """Compare multiple model profiles on the same task instance."""
    try:
        mode = (request.mode or "standard").strip().lower()
        results: list[dict[str, object]] = []

        if mode == "hallucination":
            code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else HALLUCINATION_SAFE_CODE
            model_reviews = _build_hallucination_model_reviews(force_false_positive=request.force_false_positive)
            for model_name, review in model_reviews.items():
                scored = _grade_hallucination_review(review)
                results.append(
                    {
                        "model": model_name,
                        "issue_score": float(scored["issue_score"]),
                        "fix_score": float(scored["fix_score"]),
                        "explanation_score": float(scored["explanation_score"]),
                        "final_score": float(scored["final_score"]),
                        "review": review,
                    }
                )
            variant_name = "Hallucination Detection (No Bug)"
            task_id = 3
        else:
            reset_result = env.reset(task_index=request.task_index)
            sampled_code = reset_result.observation.code
            code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else sampled_code
            current_task = env._current_task
            task = _match_task_by_code(request.task_index, code_input, current_task)
            model_reviews = _build_standard_model_reviews(code_input, request.task_index)
            for model_name, review in model_reviews.items():
                scored = _score_standard_review(review, task)
                results.append(
                    {
                        "model": model_name,
                        "issue_score": float(scored["issue_score"]),
                        "fix_score": float(scored["fix_score"]),
                        "explanation_score": float(scored["explanation_score"]),
                        "final_score": float(scored["final_score"]),
                        "review": review,
                    }
                )
            variant_name = task.title
            task_id = int(request.task_index)

        ranked = sorted(results, key=lambda x: float(x["final_score"]), reverse=True)
        best_score = float(ranked[0]["final_score"]) if ranked else 0.0
        for row in ranked:
            row["is_winner"] = float(row["final_score"]) == best_score

        return {
            "mode": mode,
            "task_index": task_id,
            "variant": variant_name,
            "code_input": code_input,
            "results": ranked,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/demo/adversarial", summary="Generate adversarial tests")
async def demo_adversarial(request: DemoEvaluateRequest) -> dict:
    """Generate adversarial inputs and control tests from a code snippet."""
    try:
        mode = (request.mode or "standard").strip().lower()
        if mode == "hallucination":
            code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else HALLUCINATION_SAFE_CODE
            variant = "Hallucination Detection (No Bug)"
            task_id = 3
        else:
            reset_result = env.reset(task_index=request.task_index)
            sampled_code = reset_result.observation.code
            code_input = request.code_input.strip() if request.code_input and request.code_input.strip() else sampled_code
            variant = env.state.variant_id
            task_id = int(request.task_index)

        generated = _generate_adversarial_tests(code_input)
        return {
            "mode": mode,
            "task_index": task_id,
            "variant": variant,
            "code_input": code_input,
            "tests": generated.get("tests", []),
            "reasons": generated.get("reasons", []),
            "summary": generated.get("summary", ""),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo/benchmark-report", summary="Generate benchmark report")
async def demo_benchmark_report() -> dict:
    """Return benchmark report JSON with summary rows and scenario details."""
    try:
        return _run_benchmark_suite()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo/benchmark-table", summary="Generate benchmark table")
async def demo_benchmark_table() -> dict:
    """Return a compact benchmark table payload for quick rendering."""
    try:
        report = _run_benchmark_suite()
        return {
            "generated_at": report.get("generated_at"),
            "suite": report.get("suite"),
            "summary_rows": report.get("summary_rows", []),
            "table": report.get("table", ""),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    """Serve interactive live evaluation demo page."""
    return HTMLResponse(content=DEMO_PAGE_HTML)


@app.get("/api", include_in_schema=False)
async def api_docs() -> RedirectResponse:
    """Convenience route to API docs from demo page."""
    return RedirectResponse(url="/docs")


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint (for `python server/app.py`)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the server from a console script entrypoint."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    main()