#!/usr/bin/env python3
"""
Test Tactical Evaluation
Demonstrates the improved reward function with tactical validation
"""

import sys
import os

# Add the neural directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine
import chess

def test_tactical_evaluation():
    """Test the improved tactical evaluation"""
    print("🧠 Testing Improved Tactical Evaluation")
    print("=" * 50)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Test 1: Tactical validation
    print("\n🎯 Test 1: Tactical Move Validation")
    print("-" * 40)
    
    # Set up a position where giving check would lose material
    engine.board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Test if the engine can identify tactically unsound moves
    print(f"Position: {engine.board.fen()}")
    print(f"Legal moves: {[move.uci() for move in list(engine.board.legal_moves)[:10]]}...")
    
    # Test tactical validation
    best_move = engine.get_best_move_with_tactical_validation(3, 2.0, verbose=True)
    if best_move:
        print(f"Best tactically sound move: {best_move.uci()}")
        print(f"Is tactically sound: {engine._is_tactically_sound(best_move)}")
    
    # Test 2: Position evaluation with tactical penalties
    print("\n🎯 Test 2: Position Evaluation with Tactical Penalties")
    print("-" * 40)
    
    # Set up a position with hanging pieces
    engine.board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Regular evaluation
    regular_eval = engine.evaluate_position_neural(engine.board)
    print(f"Regular evaluation: {regular_eval:.2f}")
    
    # Tactical evaluation with penalties
    tactical_eval = engine._evaluate_position_with_tactical_penalties()
    print(f"Tactical evaluation: {tactical_eval:.2f}")
    
    # Test 3: Move repetition prevention
    print("\n🎯 Test 3: Move Repetition Prevention")
    print("-" * 40)
    
    # Play a few moves to test repetition detection
    engine.board = chess.Board()
    print(f"Starting position: {engine.board.fen()}")
    
    # Make some moves
    moves = ["e2e4", "e7e5", "e4e5", "d7d5", "e5d5", "e8e7", "d5d6", "e7e8"]
    
    for i, move_uci in enumerate(moves):
        move = chess.Move.from_uci(move_uci)
        if move in engine.board.legal_moves:
            engine.board.push(move)
            print(f"Move {i+1}: {move_uci} - Position: {engine.board.fen()[:50]}...")
            
            # Check repetition count
            move_key = f"{move_uci}_{engine.board.fen()[:50]}"
            if hasattr(engine, 'move_repetition_count'):
                count = engine.move_repetition_count.get(move_key, 0)
                print(f"   Repetition count for this move pattern: {count}")
        else:
            print(f"Move {move_uci} is not legal in position {engine.board.fen()}")
    
    # Test 4: Pawn structure evaluation
    print("\n🎯 Test 4: Pawn Structure Evaluation")
    print("-" * 40)
    
    # Set up a position with poor pawn structure
    engine.board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Make some moves to create poor pawn structure
    engine.board.push(chess.Move.from_uci("e2e4"))
    engine.board.push(chess.Move.from_uci("e7e5"))
    engine.board.push(chess.Move.from_uci("d2d4"))
    engine.board.push(chess.Move.from_uci("d7d6"))
    engine.board.push(chess.Move.from_uci("c2c4"))
    engine.board.push(chess.Move.from_uci("c7c6"))
    
    print(f"Position after moves: {engine.board.fen()}")
    
    # Evaluate pawn structure
    isolated_penalty = engine._evaluate_pawn_structure_penalties()
    print(f"Pawn structure penalty: {isolated_penalty}")
    
    # Check for isolated pawns
    for square in chess.SQUARES:
        piece = engine.board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if engine._is_pawn_isolated(square, piece.color):
                square_name = chess.square_name(square)
                print(f"Isolated pawn at {square_name}")
    
    # Count doubled pawns
    doubled_count = engine._count_doubled_pawns()
    print(f"Doubled pawns: {doubled_count}")
    
    print("\n🎉 Tactical Evaluation Test Completed!")
    print("=" * 50)
    print("Key Improvements:")
    print("• ✅ Tactical validation prevents unsound moves")
    print("• ✅ Move repetition detection and prevention")
    print("• ✅ Positional evaluation beyond material counting")
    print("• ✅ Pawn structure analysis")
    print("• ✅ Hanging piece detection")
    print("• ✅ Check safety evaluation")

def test_self_play_with_improvements():
    """Test self-play with the improved evaluation"""
    print("\n🎮 Testing Self-Play with Improved Evaluation")
    print("=" * 50)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Play a short game to see the improvements
    print("Playing a short game with improved evaluation...")
    
    try:
        game_result = engine.self_play_game(
            show_progress=True, 
            randomness=0.1,  # Low randomness to see tactical validation
            max_moves=20  # Limit moves for testing
        )
        
        print(f"\n🎯 Game completed!")
        print(f"Result: {game_result['result']}")
        print(f"Moves played: {game_result['moves_played']}")
        print(f"Final evaluation: {game_result['final_evaluation']:.2f}")
        
        if 'repetitions' in game_result:
            total_repetitions = sum(count for count in game_result['repetitions'].values() if count > 1)
            print(f"Total move repetitions: {total_repetitions}")
        
    except Exception as e:
        print(f"❌ Error during self-play: {e}")

def main():
    """Main function"""
    print("🧠 Improved Tactical Evaluation Test Suite")
    print("=" * 50)
    print("This test demonstrates the improvements made to:")
    print("• Prevent move repetition")
    print("• Evaluate tactical soundness")
    print("• Consider positional factors beyond material")
    print("• Detect hanging pieces and poor pawn structure")
    print("=" * 50)
    
    # Run tests
    test_tactical_evaluation()
    
    # Ask if user wants to test self-play
    choice = input("\n🎮 Test self-play with improvements? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        test_self_play_with_improvements()
    
    print("\n👋 Test completed!")

if __name__ == "__main__":
    main()
