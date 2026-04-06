"""Test script: runs the 3 difficulties multiple times to demonstrate adversarial variants."""
import requests
import json

BASE = "http://localhost:7860"

# Health check
try:
    r = requests.get(f"{BASE}/health")
    print(f"Health: {r.status_code} {r.json()}")
except requests.exceptions.ConnectionError:
    print(f"Failed to connect to {BASE}. Make sure the uvicorn server is running!")
    exit(1)

# Reviews for each task (These were written for Variant A usually)
# For the other variants, these generic reviews will score significantly lower,
# proving the agent must generalize and not hardcode.
reviews = [
    (
        "Issue: Division by zero when count is 0 causes undefined behavior. "
        "Fix: Add a guard check if(count == 0) return 0. "
        "Explanation: This is an edge case that causes a runtime error "
        "and should be handled with input validation for safety."
    ),
    (
        "Issue: This function has O(n^2) quadratic time complexity due to "
        "the nested loop. Fix: Use an unordered_set for O(n) lookups instead "
        "of a linear scan. Explanation: The time complexity can be reduced "
        "from O(n^2) to O(n) by using a hash set."
    ),
    (
        "Issue: Out-of-bounds access at arr[i+1] when i equals arr.size()-1. "
        "Fix: Change loop condition to i < arr.size() - 1. "
        "Explanation: Edge case when array is empty or has a single element "
        "causes undefined behavior."
    ),
]

task_names = [
    "Task 0 - Silent Bug Detection (Easy)",
    "Task 1 - Performance Trap (Medium)",
    "Task 2 - Deceptive Logic Bug (Hard)",
]

print("\n" + "=" * 55)
print("  ADVERSARIAL TESTING - 2 iterations per difficulty")
print("  (Hardcoded reviews will get varying rewards based on the variant!)")
print("=" * 55)

scores = []
for idx in range(3):
    print(f"\n>>>> {task_names[idx]}")
    
    # Run twice to try to hit different variants
    for iteration in range(2):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Reset
        r = requests.post(f"{BASE}/reset", json={"task_index": idx})
        obs = r.json()["observation"]
        code_lines = obs['code'].split('\n')
        # Print a short preview of the code to show the variant
        short_code = "\\n".join(code_lines[:min(2, len(code_lines))])
        print(f"  Code: {short_code}...")

        # Step
        r2 = requests.post(f"{BASE}/step", json={"review": reviews[idx]})
        result = r2.json()
        reward = result["reward"]
        scores.append(reward)
        print(f"  Reward: {reward}")
        print(f"  Done:   {result['done']}")
        print(f"  Feedback Preview:", result["observation"]["feedback"].split("\\n")[0])

# State
r3 = requests.get(f"{BASE}/state")
print(f"\nFinal State: {json.dumps(r3.json(), indent=2)}")

# Summary
avg = sum(scores) / len(scores) if scores else 0
print(f"\n{'=' * 55}")
print(f"  SUMMARY")
print(f"{'=' * 55}")
print(f"  Ran {len(scores)} episode tests total.")
print(f"  Average Reward: {avg:.2f} (If this is lower than 1.0, the adversarial check worked!)")
print(f"{'=' * 55}")
