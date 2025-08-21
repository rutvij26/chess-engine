#!/usr/bin/env python3
"""
Visual Chess Training Launcher
Run visual training commands from the root directory
"""

import sys
import os

# Add the visual package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'visual'))

def main():
    """Main launcher function"""
    if len(sys.argv) < 2:
        print("🎨 Visual Chess Training Launcher")
        print("=" * 40)
        print("Usage: python run_visual.py <command>")
        print()
        print("Available commands:")
        print("  simple        - Quick 1-game visual demo")
        print("  training      - Full visual training menu")
        print("  quick         - Fast visual demo")
        print("  board         - Test visual chess board")
        print()
        print("Examples:")
        print("  python run_visual.py simple")
        print("  python run_visual.py training")
        print("  python run_visual.py board")
        return
    
    command = sys.argv[1].lower()
    
    if command == "simple":
        from simple_visual_training import simple_visual_training
        simple_visual_training()
    elif command == "training":
        from visual_training import main as run_training
        run_training()
    elif command == "quick":
        from quick_visual_demo import quick_visual_demo
        quick_visual_demo()
    elif command == "board":
        from visual_chess_board import VisualChessBoard
        board = VisualChessBoard()
        board.display_board()
    else:
        print(f"Unknown command: {command}")
        print("Use: simple, training, quick, or board")

if __name__ == "__main__":
    main()
