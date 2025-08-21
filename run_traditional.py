#!/usr/bin/env python3
"""
Traditional Chess Engine Launcher
Run traditional engine commands from the root directory
"""

import sys
import os

# Add the traditional package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'traditional'))

def main():
    """Main launcher function"""
    if len(sys.argv) < 2:
        print("♟️ Traditional Chess Engine Launcher")
        print("=" * 40)
        print("Usage: python run_traditional.py <command>")
        print()
        print("Available commands:")
        print("  interactive    - Interactive chess interface")
        print("  uci           - UCI protocol handler")
        print("  demo          - Engine capabilities demo")
        print("  test          - Run test suite")
        print()
        print("Examples:")
        print("  python run_traditional.py interactive")
        print("  python run_traditional.py uci")
        print("  python run_traditional.py demo")
        return
    
    command = sys.argv[1].lower()
    
    if command == "interactive":
        from interactive import main as run_interactive
        run_interactive()
    elif command == "uci":
        from uci_handler import main as run_uci
        run_uci()
    elif command == "demo":
        from demo import main as run_demo
        run_demo()
    elif command == "test":
        print("Running test suite...")
        os.system("python tests/test_engine.py")
    else:
        print(f"Unknown command: {command}")
        print("Use: interactive, uci, demo, or test")

if __name__ == "__main__":
    main()
