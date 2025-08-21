#!/usr/bin/env python3
"""
Chess Engine Demo
Demonstrates the engine's capabilities with a sample game
"""

from chess_engine import ChessEngine
import time

def print_separator():
    """Print a separator line"""
    print("=" * 60)

def demo_basic_functionality():
    """Demonstrate basic engine functionality"""
    print("🎯 DEMO: Basic Engine Functionality")
    print_separator()
    
    engine = ChessEngine()
    
    print("Initial position:")
    engine.print_board()
    print(f"FEN: {engine.get_fen()}")
    
    # Show legal moves
    legal_moves = engine.get_legal_moves()
    print(f"\nLegal moves for White: {len(legal_moves)}")
    print("Sample moves:", " ".join(legal_moves[:10]))
    
    # Evaluate position
    score = engine.evaluate_position()
    print(f"Position evaluation: {score}")
    
    print("\n" + "="*60)

def demo_move_generation():
    """Demonstrate move generation and validation"""
    print("♟️ DEMO: Move Generation and Validation")
    print_separator()
    
    engine = ChessEngine()
    
    # Make some moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    
    for i, move in enumerate(moves):
        if i % 2 == 0:
            print(f"\nWhite's move {i//2 + 1}: {move}")
        else:
            print(f"Black's move {i//2 + 1}: {move}")
        
        if engine.make_move(move):
            engine.print_board()
            print(f"FEN: {engine.get_fen()}")
            
            # Show evaluation
            score = engine.evaluate_position()
            if score > 0:
                print(f"White is winning by {score} centipawns")
            elif score < 0:
                print(f"Black is winning by {abs(score)} centipawns")
            else:
                print("Position is equal")
        else:
            print(f"Invalid move: {move}")
            break
    
    print("\n" + "="*60)

def demo_engine_play():
    """Demonstrate engine vs engine play"""
    print("🤖 DEMO: Engine vs Engine Play")
    print_separator()
    
    engine = ChessEngine()
    engine.make_move("e2e4")  # Start with e4
    
    print("Starting position after 1. e4:")
    engine.print_board()
    
    # Engine plays a few moves
    for move_num in range(2, 6):
        print(f"\n--- Move {move_num} ---")
        
        # Black's move (engine)
        print("Black (engine) is thinking...")
        start_time = time.time()
        black_move = engine.get_best_move(4, 3.0)
        black_time = time.time() - start_time
        
        if black_move:
            print(f"Black plays: {black_move} (found in {black_time:.2f}s)")
            engine.make_move(black_move)
            engine.print_board()
            
            # Check if game is over
            if engine.is_game_over():
                result = engine.get_game_result()
                print(f"Game over! Result: {result}")
                break
        else:
            print("Black couldn't find a move")
            break
        
        # White's move (engine)
        print("White (engine) is thinking...")
        start_time = time.time()
        white_move = engine.get_best_move(4, 3.0)
        white_time = time.time() - start_time
        
        if white_move:
            print(f"White plays: {white_move} (found in {white_time:.2f}s)")
            engine.make_move(white_move)
            engine.print_board()
            
            # Check if game is over
            if engine.is_game_over():
                result = engine.get_game_result()
                print(f"Game over! Result: {result}")
                break
        else:
            print("White couldn't find a move")
            break
    
    print("\n" + "="*60)

def demo_position_analysis():
    """Demonstrate position analysis capabilities"""
    print("🔍 DEMO: Position Analysis")
    print_separator()
    
    engine = ChessEngine()
    
    # Set up an interesting position
    interesting_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4"
    engine.set_fen(interesting_fen)
    
    print("Analyzing position:")
    engine.print_board()
    print(f"FEN: {engine.get_fen()}")
    
    # Show legal moves
    legal_moves = engine.get_legal_moves()
    print(f"\nLegal moves for Black: {len(legal_moves)}")
    print("Available moves:", " ".join(legal_moves))
    
    # Evaluate position
    score = engine.evaluate_position()
    print(f"\nPosition evaluation: {score}")
    
    # Get engine's best move
    print("\nEngine analysis (depth 5):")
    start_time = time.time()
    best_move = engine.get_best_move(5, 5.0)
    analysis_time = time.time() - start_time
    
    if best_move:
        print(f"Best move: {best_move}")
        print(f"Analysis time: {analysis_time:.2f} seconds")
        
        # Show what happens after the move
        engine.make_move(best_move)
        print(f"\nPosition after {best_move}:")
        engine.print_board()
        
        new_score = engine.evaluate_position()
        print(f"New evaluation: {new_score}")
        print(f"Score change: {new_score - score}")
    
    print("\n" + "="*60)

def demo_special_positions():
    """Demonstrate handling of special chess positions"""
    print("🎭 DEMO: Special Chess Positions")
    print_separator()
    
    engine = ChessEngine()
    
    # Test checkmate position
    print("1. Checkmate Position:")
    checkmate_fen = "rnb1kbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 4"
    engine.set_fen(checkmate_fen)
    engine.print_board()
    
    if engine.is_game_over():
        result = engine.get_game_result()
        print(f"Game result: {result}")
        print("This is a checkmate position!")
    
    print("\n2. Stalemate Position:")
    stalemate_fen = "k7/1R6/1R6/8/8/8/8/7K b - - 0 1"
    engine.set_fen(stalemate_fen)
    engine.print_board()
    
    if engine.is_game_over():
        result = engine.get_game_result()
        print(f"Game result: {result}")
        print("This is a stalemate position!")
    
    print("\n" + "="*60)

def main():
    """Run the complete demo"""
    print("🚀 CHESS ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the capabilities of our chess engine")
    print("=" * 60)
    
    try:
        demo_basic_functionality()
        demo_move_generation()
        demo_engine_play()
        demo_position_analysis()
        demo_special_positions()
        
        print("🎉 DEMO COMPLETE!")
        print("\nThe chess engine successfully demonstrated:")
        print("✓ Basic functionality and move validation")
        print("✓ Move generation and board updates")
        print("✓ Engine vs engine play")
        print("✓ Position analysis and evaluation")
        print("✓ Special position handling")
        
        print("\nTo try the engine yourself:")
        print("• Run 'python interactive.py' for interactive play")
        print("• Run 'python uci_handler.py' for UCI compatibility")
        print("• Run 'python test_engine.py' to run tests")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please check that all dependencies are installed correctly.")

if __name__ == "__main__":
    main()
