import sys
import os

# Add local path so we can import models and environment easily
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CodeReviewEnvironment

def main():
    env = CodeReviewEnvironment()
    print("Testing Easy task variants randomness:")
    print("=" * 50)
    
    seen_variants = set()
    for i in range(10):
        # Reset requests a specific difficulty (task_index 0 = Easy)
        result = env.reset(task_index=0)
        # Extract title from the underlying task to see which variant was picked
        title = env._current_task.title
        seen_variants.add(title)
        print(f"Run {i+1}: Picked -> {title}")
    
    print("=" * 50)
    print(f"Total unique Easy variants seen: {len(seen_variants)} out of 3")
    if len(seen_variants) > 1:
        print("SUCCESS! The environment is dynamically picking different variants.")
    else:
        print("WARNING! Only saw 1 variant. Something might be wrong with random choice.")

if __name__ == "__main__":
    main()
