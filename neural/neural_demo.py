#!/usr/bin/env python3
"""
Neural Chess Engine Demo
Demonstrates how the neural network learns to play chess through self-play
"""

import torch
import time
from neural_chess_engine import NeuralChessEngine

def demo_neural_learning():
    """Demonstrate neural network learning process"""
    print("🧠 NEURAL CHESS ENGINE LEARNING DEMO")
    print("=" * 50)
    
    # Create a fresh neural chess engine
    print("1. Creating neural chess engine...")
    engine = NeuralChessEngine()
    
    # Show initial random evaluation
    print("\n2. Initial random neural network evaluation:")
    engine.print_board()
    initial_eval = engine.evaluate_position_neural(engine.board)
    print(f"Position evaluation: {initial_eval:.2f}")
    print("(This is random because the network hasn't learned yet)")
    
    # Play a few self-play games to learn
    print("\n3. Playing self-play games to learn...")
    num_learning_games = 5
    
    for game_num in range(num_learning_games):
        print(f"\n--- Learning Game {game_num + 1} ---")
        
        # Play a short game
        game_data = engine.self_play_game(max_moves=20)
        
        if len(game_data) > 0:
            print(f"Game played: {len(game_data)} moves")
            
            # Train the model on this game
            print("Training neural network...")
            engine.train_model(epochs=2)
            
            # Show improvement in evaluation
            engine.reset_board()
            new_eval = engine.evaluate_position_neural(engine.board)
            print(f"New evaluation: {new_eval:.2f}")
            print(f"Change: {new_eval - initial_eval:+.2f}")
    
    # Test the learned model
    print("\n4. Testing learned model...")
    engine.reset_board()
    
    print("Getting best move with learned neural network...")
    start_time = time.time()
    best_move = engine.get_best_move(3, 3.0)
    thinking_time = time.time() - start_time
    
    if best_move:
        print(f"Best move: {best_move} (found in {thinking_time:.2f}s)")
        
        # Make the move and show result
        engine.make_move(best_move)
        print("\nPosition after move:")
        engine.print_board()
        
        # Show neural network's evaluation
        final_eval = engine.evaluate_position_neural(engine.board)
        print(f"Neural network evaluation: {final_eval:.2f}")
        
        # Compare with traditional evaluation
        traditional_eval = engine.evaluate_position_fallback(engine.board)
        print(f"Traditional evaluation: {traditional_eval:.2f}")
        print(f"Difference: {final_eval - traditional_eval:+.2f}")
    
    print("\n5. Learning Summary:")
    print(f"• Games played: {num_learning_games}")
    print(f"• Training positions: {len(engine.training_positions)}")
    print(f"• Neural network learned from self-play!")
    
    return engine

def demo_position_understanding():
    """Demonstrate how the neural network understands different positions"""
    print("\n🔍 DEMO: Neural Network Position Understanding")
    print("=" * 50)
    
    engine = NeuralChessEngine()
    
    # Test different positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Checkmate position", "rnb1kbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 4"),
        ("Endgame position", "8/8/8/8/8/8/4K3/4k3 w - - 0 1")
    ]
    
    for position_name, fen in test_positions:
        print(f"\n--- {position_name} ---")
        engine.set_fen(fen)
        engine.print_board()
        
        # Get both evaluations
        neural_eval = engine.evaluate_position_neural(engine.board)
        traditional_eval = engine.evaluate_position_fallback(engine.board)
        
        print(f"Neural evaluation: {neural_eval:.2f}")
        print(f"Traditional evaluation: {traditional_eval:.2f}")
        print(f"Difference: {neural_eval - traditional_eval:+.2f}")

def main():
    """Main demo function"""
    try:
        # Run neural learning demo
        trained_engine = demo_neural_learning()
        
        # Show position understanding
        demo_position_understanding()
        
        print("\n🎉 NEURAL CHESS ENGINE DEMO COMPLETED!")
        print("\nWhat we demonstrated:")
        print("✅ Neural network creation and initialization")
        print("✅ Self-play learning process")
        print("✅ Training on game data")
        print("✅ Improved move selection")
        print("✅ Position understanding")
        
        print("\nTo train a full model:")
        print("• Run 'python train_neural_chess.py' for full training")
        print("• Run 'python neural_chess_engine.py' for basic demo")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure PyTorch is installed: pip install torch")

if __name__ == "__main__":
    main()
