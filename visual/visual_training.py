#!/usr/bin/env python3
"""
Visual Neural Chess Training
Shows the chess board progress during neural network training
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural'))
from neural_chess_engine import NeuralChessEngine
import os

def visual_training_demo():
    """Demonstrate visual training with board display"""
    print("🎨 VISUAL NEURAL CHESS TRAINING")
    print("=" * 50)
    print("This will show the chess board progress during training!")
    print()
    
    # Create engine with visual mode
    engine = NeuralChessEngine(visual_mode=True)
    
    if not engine.visual_mode:
        print("❌ Visual mode not available. Make sure visual_chess_board.py is installed.")
        return
    
    print("✅ Visual mode enabled!")
    print("🎮 Starting visual training demo...")
    print()
    
    # Train on a few games with visual display
    num_games = 3
    epochs_per_game = 2
    
    print(f"Training on {num_games} games with {epochs_per_game} epochs each...")
    print("Watch the board progress in real-time!")
    print()
    
    # Start visual training
    engine.train_on_self_play(
        num_games=num_games,
        epochs_per_game=epochs_per_game,
        visual_training=True
    )
    
    print("\n🎉 Visual training demo completed!")
    print("You can now see how the neural network learns chess!")

def interactive_visual_play():
    """Play against the trained neural network with visual board"""
    print("\n🎮 INTERACTIVE VISUAL PLAY")
    print("=" * 40)
    
    # Load the best model if available
    model_files = [f for f in os.listdir('.') if f.startswith('chess_model_') and f.endswith('.pth')]
    
    if model_files:
        # Sort by game number and get the latest
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
        best_model = model_files[-1]
        print(f"Loading trained model: {best_model}")
        engine = NeuralChessEngine(model_path=best_model, visual_mode=True)
    else:
        print("No trained model found. Using untrained network.")
        engine = NeuralChessEngine(visual_mode=True)
    
    if not engine.visual_mode:
        print("❌ Visual mode not available.")
        return
    
    print("✅ Ready to play! You'll see the board after each move.")
    print("Commands: move (e.g., 'e2e4'), 'board', 'eval', 'quit'")
    print()
    
    while not engine.board.is_game_over():
        # Show current board
        engine.visual_board.display_board(
            engine.board,
            evaluation=engine.evaluate_position_neural(engine.board)
        )
        
        # Get user input
        command = input("\nEnter move (e.g., e2e4) or command: ").strip().lower()
        
        if command == 'quit':
            break
        elif command == 'board':
            continue  # Board already shown
        elif command == 'eval':
            eval_score = engine.evaluate_position_neural(engine.board)
            print(f"Position evaluation: {eval_score:+.2f}")
            continue
        elif command == 'help':
            print("Commands: move (e.g., 'e2e4'), 'board', 'eval', 'quit'")
            continue
        
        # Try to make the move
        if engine.make_move(command):
            print(f"✅ Move {command} made!")
            
            # Show position after move
            time.sleep(0.5)
            
            # Get engine's response
            if not engine.board.is_game_over():
                print("🤖 Engine is thinking...")
                best_move = engine.get_best_move(3, 2.0)
                
                if best_move:
                    engine.make_move(best_move)
                    print(f"🤖 Engine plays: {best_move}")
                    time.sleep(1.0)
                else:
                    print("❌ Engine couldn't find a move")
        else:
            print(f"❌ Invalid move: {command}")
            print("Valid moves:", [move.uci() for move in engine.board.legal_moves])
    
    # Show final position
    engine.visual_board.display_board(
        engine.board,
        evaluation=engine.evaluate_position_neural(engine.board)
    )
    
    # Game result
    if engine.board.is_checkmate():
        winner = "BLACK" if engine.board.turn else "WHITE"
        print(f"\n🎯 CHECKMATE! {winner} wins!")
    elif engine.board.is_stalemate():
        print("\n🤝 STALEMATE - Draw!")
    else:
        print("\n🏁 Game Over")

def main():
    """Main function for visual training and play"""
    print("🎨 VISUAL NEURAL CHESS ENGINE")
    print("=" * 40)
    print("1. Visual Training Demo")
    print("2. Interactive Visual Play")
    print("3. Exit")
    
    while True:
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == '1':
            visual_training_demo()
        elif choice == '2':
            interactive_visual_play()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
