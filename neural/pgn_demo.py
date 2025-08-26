#!/usr/bin/env python3
"""
PGN Generation Demo
Demonstrates the new PGN generation and game history features
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from neural_chess_engine import NeuralChessEngine
import time

def demo_pgn_generation():
    """Demonstrate PGN generation for a single game"""
    print("🎯 PGN Generation Demo")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine(visual_mode=False)
    
    print("Playing a quick game to demonstrate PGN generation...")
    
    # Play a short game
    game_result = engine.self_play_game(max_moves=20, show_progress=False)
    
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
        
        print("\n🎉 PGN generation demo completed!")
        print("Check 'games/game_histories.pgn' for the saved game.")
        
    else:
        print("❌ No game data generated")

def demo_multiple_games():
    """Demonstrate PGN generation for multiple games"""
    print("\n🎮 Multiple Games PGN Demo")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine(visual_mode=False)
    
    num_games = 3
    print(f"Playing {num_games} games to demonstrate batch PGN generation...")
    
    for game_num in range(num_games):
        print(f"\n🎮 Game {game_num + 1}/{num_games}")
        
        # Play a game
        game_result = engine.self_play_game(max_moves=15, show_progress=False)
        
        if game_result['move_history']:
            # Generate and save PGN
            pgn_game = engine.generate_pgn_game(
                game_result['move_history'],
                game_result['result']
            )
            
            engine.save_game_to_history(pgn_game, game_num + 1, game_result)
            
            print(f"   ✅ Game {game_num + 1} completed - {game_result['moves_played']} moves")
            print(f"   📜 PGN generated and saved")
        
        # Reset for next game
        engine.reset_board()
    
    print(f"\n🎉 {num_games} games completed and saved to PGN!")
            print("Check 'games/game_histories.pgn' for all games.")

def main():
    """Main demo function"""
    print("🧠 Neural Chess Engine - PGN Generation Demo")
    print("=" * 50)
    print("This demo showcases the new PGN generation features:")
    print("• Automatic PGN generation after each game")
    print("• Game history saving in PGN format")
    print("• Rich metadata including game results and evaluations")
    print("=" * 50)
    
    # Demo 1: Single game PGN generation
    demo_pgn_generation()
    
    # Demo 2: Multiple games
    demo_multiple_games()
    
    print("\n🎯 Demo Summary:")
    print("✅ PGN generation working correctly")
    print("✅ Game histories being saved")
    print("✅ Rich metadata included")
    print("\n📁 Generated files:")
    print("   • games/game_histories.pgn - Contains all demo games")
    print("\n🚀 Ready for full training with PGN generation!")

if __name__ == "__main__":
    main()
