import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CodeReviewEnvironment, grade_review

def main():
    env = CodeReviewEnvironment()
    
    # Let's get an easy task
    result = env.reset(task_index=0)
    task = env._current_task
    
    review = (
        "There is something wrong when parsing the variables. "
        "The bottom portion divides two integers and if the lower one is absolutely nothing, "
        "it will crash in a bad way. A guard like `if count == 0` is necessary."
    )
    
    print("Testing Grader with a highly semantic review (bypasses keyword matching mostly):")
    print("Review text:", review)
    print("=" * 60)
    
    # grade_review will try to hit localhost:8000/v1. If it fails, it prints a fallback message.
    os.environ["GRADER_API_URL"] = "http://localhost:8000/v1" # simulate local unavailable server
    
    reward, feedback = grade_review(review, task)
    print("\nResult Reward:", reward)
    print("\nFeedback:")
    print(feedback)

if __name__ == "__main__":
    main()
