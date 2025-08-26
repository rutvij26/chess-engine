#!/usr/bin/env python3
"""
Simple PGN Generation Test
Tests the PGN generation functionality without complex training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine
import time

def test_single_game():
    """Test PGN generation for a single game"""
    print("🧪 Testing PGN Generation")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine()
    
    print("🎮 Playing a single game...")
    
    # Play a game
    game_result = engine.self_play_game(show_progress=True)
    
    if game_result['move_history']:
        print(f"\n✅ Game completed!")
        print(f"   Moves played: {game_result['moves_played']}")
        print(f"   Final result: {game_result['result']}")
        print(f"   Final evaluation: {game_result['final_evaluation']:.3f}")
        
        # Generate PGN
        print("\n📜 Generating PGN notation...")
        pgn_game = engine.generate_pgn_game(
            game_result['move_history'],
            game_result['result']
        )
        
        print("\n📄 Generated PGN:")
        print("-" * 40)
        print(pgn_game)
        print("-" * 40)
        
        # Save to history
        print("\n💾 Saving to game history...")
        engine.save_game_to_history(pgn_game, 1, game_result)
        
        print("\n🎉 PGN generation test completed successfully!")
        print("Check 'games/game_histories.pgn' for the saved game.")
        
        return True
    else:
        print("❌ No game data generated")
        return False

def main():
    """Main test function"""
    print("🧠 Neural Chess Engine - PGN Generation Test")
    print("=" * 50)
    
    try:
        success = test_single_game()
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
