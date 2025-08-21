#!/usr/bin/env python3
"""
Neural Chess Engine Launcher
Run neural engine commands from the root directory
"""

import sys
import os

# Add the neural package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

def main():
    """Main launcher function"""
    if len(sys.argv) < 2:
        print("🧠 Neural Chess Engine Launcher")
        print("=" * 40)
        print("Usage: python run_neural.py <command>")
        print()
        print("Available commands:")
        print("  demo          - Basic neural learning demo")
        print("  train         - Full neural training")
        print("  grandmaster   - Grandmaster+ training (2-6 months)")
        print("  test          - Test grandmaster system")
        print()
        print("Examples:")
        print("  python run_neural.py demo")
        print("  python run_neural.py train")
        print("  python run_neural.py grandmaster")
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        from neural_demo import main as run_demo
        run_demo()
    elif command == "train":
        from train_neural_chess import main as run_training
        run_training()
    elif command == "grandmaster":
        from grandmaster_training import main as run_grandmaster
        run_grandmaster()
    elif command == "test":
        print("Testing grandmaster system...")
        os.system("python tests/test_grandmaster.py")
    else:
        print(f"Unknown command: {command}")
        print("Use: demo, train, grandmaster, or test")

if __name__ == "__main__":
    main()
