#!/usr/bin/env python3
"""
Quick Visual Demo
Shows a few moves with visual board display
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural'))
from neural_chess_engine import NeuralChessEngine

def quick_visual_demo():
    """Quick demo of visual chess board during neural training"""
    print("🎨 QUICK VISUAL NEURAL DEMO")
    print("=" * 40)
    print("Watch the neural network learn chess visually!")
    print()
    
    # Create engine with visual mode
    engine = NeuralChessEngine(visual_mode=True)
    
    if not engine.visual_mode:
        print("❌ Visual mode not available.")
        return
    
    print("✅ Visual mode enabled!")
    print("🎮 Playing 1 quick self-play game...")
    print("Watch the board progress!")
    print()
    
    # Play one short game with visual display
    game_data = engine.self_play_game(max_moves=8, show_progress=True)
    
    print(f"\n🎉 Game completed! {len(game_data)} positions collected.")
    print("The neural network learned from this game!")
    
    # Show final position
    engine.visual_board.display_board(
        engine.board,
        evaluation=engine.evaluate_position_neural(engine.board)
    )
    
    print("\n✅ Quick visual demo completed!")
    print("For full training, run: python visual_training.py")

if __name__ == "__main__":
    quick_visual_demo()
