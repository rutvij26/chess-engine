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
    print("🧠 Neural Chess Engine Runner")
    print("=" * 40)
    print("Choose an option:")
    print("1. Basic Training (Single-threaded)")
    print("2. Parallel Training (3 games simultaneously) - RECOMMENDED")
    print("3. Grandmaster Training (Advanced)")
    print("4. Interactive Play")
    print("5. Visual Training Demo")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Basic Training...")
        from neural.train_neural_chess import train_neural_chess_engine
        train_neural_chess_engine(
            num_games=20,
            epochs_per_game=3,
            learning_rate=0.001,
            save_interval=5,
            model_name="chess_neural_basic"
        )
        
    elif choice == "2":
        print("\n🚀 Starting Parallel Training...")
        from neural.train_neural_chess import train_neural_chess_engine_parallel
        train_neural_chess_engine_parallel(
            num_games=30,
            epochs_per_game=3,
            learning_rate=0.001,
            save_interval=10,
            model_name="chess_neural_parallel",
            num_parallel_games=3
        )
        
    elif choice == "3":
        print("\n🏆 Starting Grandmaster Training...")
        print("⚠️  WARNING: This will take a very long time!")
        confirm = input("Continue? (y/n): ").lower().startswith('y')
        if confirm:
            from neural.grandmaster_training import main as grandmaster_main
            grandmaster_main()
        else:
            print("Grandmaster training cancelled.")
            
    elif choice == "4":
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
                best_move = engine.get_best_move(3, 2.0)
                if best_move:
                    engine.make_move(best_move)
                    print(f"AI move: {best_move}")
                else:
                    print("AI couldn't find a move.")
        
        print("Game ended.")
        
    elif choice == "5":
        print("\n🎨 Starting Visual Training Demo...")
        from visual.visual_training import main as visual_main
        visual_main()
        
    elif choice == "6":
        print("Goodbye!")
        return
        
    else:
        print("Invalid choice. Please try again.")
        main()
        return
    
    print("\n🎉 Session completed!")
    print("📜 Check the generated PGN files:")
    print("   - game_histories.pgn (basic training)")
    print("   - grandmaster_game_histories.pgn (grandmaster training)")

if __name__ == "__main__":
    main()
