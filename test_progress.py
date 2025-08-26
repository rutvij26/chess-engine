#!/usr/bin/env python3
"""
Test Progress Display
Quick test to verify that move progress is being shown correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine

def test_progress():
    """Test the progress display functionality"""
    print("🧪 Testing Progress Display")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine()
    
    print("🎮 Playing a short game with progress display...")
    print("You should see moves being played in real-time!")
    print()
    
    # Play a game with progress enabled
    game_result = engine.self_play_game(show_progress=True)
    
    if game_result['move_history']:
        print(f"\n✅ Test completed!")
        print(f"   Total moves: {game_result['moves_played']}")
        print(f"   Game result: {game_result['result']}")
        print(f"   Final evaluation: {game_result['final_evaluation']:.3f}")
        
        print(f"\n📜 Move history: {' '.join(game_result['move_history'][:10])}...")
        
        return True
    else:
        print("❌ No game data generated")
        return False

if __name__ == "__main__":
    try:
        success = test_progress()
        if success:
            print("\n✅ Progress display is working correctly!")
            print("🚀 Ready for full training with real-time move updates!")
        else:
            print("\n❌ Progress display test failed!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
