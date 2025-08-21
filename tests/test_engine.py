#!/usr/bin/env python3
"""
Test suite for the Chess Engine
Tests various aspects of the engine functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'traditional'))
from chess_engine import ChessEngine
import time

def test_basic_functionality():
    """Test basic engine functionality"""
    print("Testing basic functionality...")
    engine = ChessEngine()
    
    # Test initial position
    assert engine.get_fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("✓ Initial position correct")
    
    # Test legal moves
    legal_moves = engine.get_legal_moves()
    assert len(legal_moves) == 20  # White has 20 legal moves in starting position
    print("✓ Legal moves count correct")
    
    # Test making a move
    assert engine.make_move("e2e4")
    print("✓ Move e2e4 successful")
    
    # Test board state after move
    assert "e4" in str(engine.board)
    print("✓ Board state updated correctly")
    
    print("Basic functionality tests passed!\n")

def test_evaluation():
    """Test position evaluation"""
    print("Testing position evaluation...")
    engine = ChessEngine()
    
    # Starting position should be roughly equal
    score = engine.evaluate_position()
    assert abs(score) < 100  # Should be close to 0
    print("✓ Starting position evaluation reasonable")
    
    # Test after a few moves
    engine.make_move("e2e4")
    engine.make_move("e7e5")
    engine.make_move("g1f3")
    
    score = engine.evaluate_position()
    print(f"✓ Position after 3 moves evaluated: {score}")
    
    print("Evaluation tests passed!\n")

def test_search():
    """Test search functionality"""
    print("Testing search functionality...")
    engine = ChessEngine()
    
    # Test shallow search
    start_time = time.time()
    best_move = engine.get_best_move(3, 2.0)
    elapsed = time.time() - start_time
    
    assert best_move in engine.get_legal_moves()
    print(f"✓ Search depth 3 completed in {elapsed:.2f}s")
    print(f"✓ Best move found: {best_move}")
    
    print("Search tests passed!\n")

def test_special_positions():
    """Test special chess positions"""
    print("Testing special positions...")
    engine = ChessEngine()
    
    # Test checkmate position
    checkmate_fen = "rnb1kbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 4"
    engine.set_fen(checkmate_fen)
    
    # This position should be checkmate for white
    assert engine.is_game_over()
    result = engine.get_game_result()
    print(f"✓ Checkmate position detected: {result}")
    
    # Test stalemate position
    stalemate_fen = "k7/1R6/1R6/8/8/8/8/7K b - - 0 1"
    engine.set_fen(stalemate_fen)
    
    assert engine.is_game_over()
    result = engine.get_game_result()
    print(f"✓ Stalemate position detected: {result}")
    
    print("Special position tests passed!\n")

def test_move_validation():
    """Test move validation"""
    print("Testing move validation...")
    engine = ChessEngine()
    
    # Valid moves
    assert engine.make_move("e2e4")
    assert engine.make_move("d7d5")
    print("✓ Valid moves accepted")
    
    # Invalid moves
    assert not engine.make_move("e2e5")  # Invalid pawn move
    assert not engine.make_move("a1a8")  # Invalid rook move
    print("✓ Invalid moves rejected")
    
    print("Move validation tests passed!\n")

def test_performance():
    """Test engine performance"""
    print("Testing engine performance...")
    engine = ChessEngine()
    
    # Test search at different depths
    depths = [2, 3, 4]
    for depth in depths:
        start_time = time.time()
        best_move = engine.get_best_move(depth, 5.0)
        elapsed = time.time() - start_time
        
        print(f"✓ Depth {depth}: {elapsed:.2f}s")
        assert best_move != ""
    
    print("Performance tests passed!\n")

def run_all_tests():
    """Run all tests"""
    print("Running Chess Engine Test Suite")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_evaluation()
        test_search()
        test_special_positions()
        test_move_validation()
        test_performance()
        
        print("🎉 All tests passed! The chess engine is working correctly.")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    run_all_tests()
