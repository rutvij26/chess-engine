#!/usr/bin/env python3
"""
Simple Visual Neural Training Demo
Shows the neural network learning chess with a clean, scrollable board
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural'))
from neural_chess_engine import NeuralChessEngine

def simple_visual_training():
    """Simple demo of visual neural training"""
    print("🎨 SIMPLE VISUAL NEURAL TRAINING")
    print("=" * 50)
    print("Watch the neural network learn chess visually!")
    print("The board will show progress without clearing the screen.")
    print("You can scroll up to see the entire learning process!")
    print()
    
    # Create engine with visual mode
    engine = NeuralChessEngine(visual_mode=True)
    
    if not engine.visual_mode:
        print("❌ Visual mode not available.")
        return
    
    print("✅ Visual mode enabled!")
    print("🎮 Playing 1 self-play game to learn...")
    print("Watch the board progress in real-time!")
    print()
    
    # Play one game with visual display
    game_data = engine.self_play_game(max_moves=10, show_progress=True)
    
    print(f"\n🎉 Game completed! {len(game_data)} positions collected.")
    print("The neural network learned from this game!")
    
    # Show final position
    engine.visual_board.display_board(
        engine.board,
        evaluation=engine.evaluate_position_neural(engine.board)
    )
    
    print("\n✅ Simple visual training completed!")
    print("Scroll up to see the entire learning process!")
    print("\nFor more training, run: python visual_training.py")

if __name__ == "__main__":
    simple_visual_training()
