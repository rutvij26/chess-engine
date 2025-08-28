#!/usr/bin/env python3
"""
Neural Chess Engine Runner
Run the neural chess engine with various options
"""

import sys
import os

# Add the neural directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

def main():
    while True:
        print("🧠 Neural Chess Engine Runner")
        print("=" * 40)
        print("Choose an option:")
        print("1. Neural Network Training (Self-Play)")
        print("2. PGN Move Training (Extend Existing Network)")
        print("3. Interactive Play")
        print("4. Model vs Model Battle")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Neural Network Training (Self-Play)...")
            from neural.train_neural_chess import main as train_main
            train_main()
            
        elif choice == "2":
            print("\n📚 Starting PGN Move Training...")
            try:
                from neural.pgn_move_trainer import PGNMoveTrainer
                import run_pgn_move_training
                run_pgn_move_training.main()
            except ImportError as e:
                print(f"❌ Error importing PGN move trainer: {e}")
                print("Please ensure all dependencies are installed")
            
        elif choice == "3":
            print("\n🎮 Starting Interactive Play...")
            from neural.neural_chess_engine import NeuralChessEngine
            engine = NeuralChessEngine()
            
            print("Interactive chess game started!")
            print("Type 'quit' to exit, or make moves in UCI format (e.g., 'e2e4')")
            
            while True:
                engine.print_board()
                print(f"\nTurn: {'White' if engine.board.turn else 'Black'}")
                
                if engine.board.is_game_over():
                    result = engine.get_game_result()
                    print(f"Game Over! Result: {result}")
                    break
                
                if engine.board.turn:  # White's turn (human)
                    move = input("Your move (UCI format): ").strip()
                    if move.lower() == 'quit':
                        break
                    if engine.make_move(move):
                        print(f"Move made: {move}")
                    else:
                        print("Invalid move, try again.")
                else:  # Black's turn (AI)
                    print("AI is thinking...")
                    best_move = engine.get_best_move(3, 2.0, verbose=False)
                    if best_move:
                        engine.make_move(best_move)
                        print(f"AI move: {best_move}")
                    else:
                        print("AI couldn't find a move.")
            
            print("Game ended.")
            
        elif choice == "4":
            print("\n⚔️  Starting Model vs Model Battle...")
            try:
                from neural.model_vs_model import ModelVsModelBattle
                import run_model_battle
                run_model_battle.main()
            except ImportError as e:
                print(f"❌ Error importing model battle: {e}")
                print("Please ensure all dependencies are installed")
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")
            continue
        
        # Ask if user wants to continue
        print("\n" + "=" * 40)
        continue_choice = input("Return to main menu? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
