#!/usr/bin/env python3
"""
Progress Features Demo
Demonstrates the new real-time progress indicators and enhanced training display
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine
import time

def demo_progress_features():
    """Demonstrate the new progress features"""
    print("🎯 Progress Features Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("• Real-time game progress during play")
    print("• Detailed training progress with batch updates")
    print("• Visual progress bars and performance metrics")
    print("• Enhanced PGN generation with progress")
    print("=" * 50)
    
    # Create engine
    engine = NeuralChessEngine()
    
    print("\n🎮 Playing a game with progress indicators...")
    print("Watch how the system shows:")
    print("  • Move-by-move progress")
    print("  • Position evaluations")
    print("  • Game status updates")
    print("  • Training progress")
    print()
    
    # Play a game with progress
    game_result = engine.self_play_game(show_progress=True)
    
    if game_result['move_history']:
        print(f"\n✅ Game completed!")
        print(f"   Moves played: {game_result['moves_played']}")
        print(f"   Final result: {game_result['result']}")
        print(f"   Final evaluation: {game_result['final_evaluation']:.3f}")
        
        # Generate PGN with progress
        print("\n📜 Generating PGN with progress...")
        pgn_game = engine.generate_pgn_game(
            game_result['move_history'],
            game_result['result']
        )
        
        # Save to history
        print("💾 Saving to game history...")
        engine.save_game_to_history(pgn_game, 1, game_result)
        
        print("\n🧠 Training model with progress indicators...")
        # Train the model with progress
        engine.train_model_with_progress(epochs=2)
        
        print("\n🎉 Progress features demo completed!")
        print("📜 Check 'games/game_histories.pgn' for the saved game")
        
        return True
    else:
        print("❌ No game data generated")
        return False

def main():
    """Main demo function"""
    print("🧠 Neural Chess Engine - Progress Features Demo")
    print("=" * 60)
    
    try:
        success = demo_progress_features()
        if success:
            print("\n✅ All progress features working correctly!")
            print("\n🚀 Ready for full training with enhanced progress display!")
        else:
            print("\n❌ Some features failed!")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
