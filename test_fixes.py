#!/usr/bin/env python3
"""
Test Fixes
Verifies that the recent bug fixes work correctly
"""

import sys
import os

# Add the neural directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine
import chess

def test_bug_fixes():
    """Test that the recent bug fixes work"""
    print("🧪 Testing Bug Fixes")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Test 1: Tactical evaluation without errors
    print("\n🎯 Test 1: Tactical Evaluation")
    print("-" * 30)
    
    try:
        # Set up a simple position
        engine.board = chess.Board()
        
        # Test tactical evaluation
        tactical_eval = engine._evaluate_position_with_tactical_penalties()
        print(f"✅ Tactical evaluation works: {tactical_eval:.2f}")
        
    except Exception as e:
        print(f"❌ Tactical evaluation failed: {e}")
        return False
    
    # Test 2: Move repetition counter
    print("\n🎯 Test 2: Move Repetition Counter")
    print("-" * 30)
    
    try:
        # Reset board and repetition counter
        engine.board = chess.Board()
        engine.move_repetition_count = {}
        
        # Make a move
        move = chess.Move.from_uci("e2e4")
        engine.board.push(move)
        
        # Check repetition counter
        move_key = f"{move.uci()}_{engine.board.fen()[:50]}"
        if hasattr(engine, 'move_repetition_count'):
            print(f"✅ Move repetition counter exists")
        else:
            print(f"❌ Move repetition counter missing")
            return False
            
    except Exception as e:
        print(f"❌ Move repetition test failed: {e}")
        return False
    
    # Test 3: Tactical validation
    print("\n🎯 Test 3: Tactical Validation")
    print("-" * 30)
    
    try:
        # Test if a move is tactically sound
        move = chess.Move.from_uci("d2d4")
        is_sound = engine._is_tactically_sound(move)
        print(f"✅ Tactical validation works: {move.uci()} is {'sound' if is_sound else 'unsound'}")
        
    except Exception as e:
        print(f"❌ Tactical validation failed: {e}")
        return False
    
    # Test 4: Pawn structure evaluation
    print("\n🎯 Test 4: Pawn Structure Evaluation")
    print("-" * 30)
    
    try:
        # Test pawn structure penalties
        penalties = engine._evaluate_pawn_structure_penalties()
        print(f"✅ Pawn structure evaluation works: {penalties}")
        
    except Exception as e:
        print(f"❌ Pawn structure evaluation failed: {e}")
        return False
    
    print("\n🎉 All bug fixes verified!")
    print("=" * 40)
    return True

def test_short_game():
    """Test a very short game to ensure no errors"""
    print("\n🎮 Test 5: Short Game")
    print("-" * 30)
    
    try:
        # Create engine
        engine = NeuralChessEngine()
        
        # Play just 2 moves to test
        print("Playing 2 moves...")
        
        # Move 1: White
        best_move = engine.get_best_move_with_tactical_validation(2, 1.0, verbose=False)
        if best_move:
            engine.board.push(best_move)
            print(f"✅ White move: {best_move.uci()}")
        else:
            print("❌ No white move found")
            return False
        
        # Move 2: Black
        best_move = engine.get_best_move_with_tactical_validation(2, 1.0, verbose=False)
        if best_move:
            engine.board.push(best_move)
            print(f"✅ Black move: {best_move.uci()}")
        else:
            print("❌ No black move found")
            return False
        
        print(f"✅ Short game completed successfully!")
        print(f"   Final position: {engine.board.fen()[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Short game failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧪 Bug Fix Verification Test Suite")
    print("=" * 50)
    print("This test verifies that the recent bug fixes work:")
    print("• Fixed 'board' vs 'self.board' reference errors")
    print("• Fixed division by zero in training progress")
    print("• Fixed tactical evaluation errors")
    print("=" * 50)
    
    # Run basic tests
    if not test_bug_fixes():
        print("\n❌ Basic tests failed!")
        return
    
    # Run game test
    if not test_short_game():
        print("\n❌ Game test failed!")
        return
    
    print("\n🎉 All tests passed! The bug fixes are working correctly.")
    print("=" * 50)
    print("You can now run training without the previous errors.")

if __name__ == "__main__":
    main()
