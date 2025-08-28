#!/usr/bin/env python3
"""
Interactive Model vs Model Battle Runner
Allows users to select models and configure battle parameters
"""

import os
import sys
from neural.model_vs_model import ModelVsModelBattle


def main():
    """Main function for interactive model battles"""
    print("⚔️  Neural Chess Engine - Model vs Model Battle")
    print("🎯 Watch trained models fight it out!")
    print("=" * 50)
    
    # Create battle system
    battle_system = ModelVsModelBattle()
    
    # List available models
    available_models = battle_system.list_available_models()
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train some models first using the neural training option.")
        return
    
    print(f"📚 Available Models ({len(available_models)}):")
    for i, model in enumerate(available_models, 1):
        file_path = os.path.join("models", model)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   {i}. {model} ({file_size:.1f} MB)")
    
    # Get user selection for White
    print(f"\n🤍 Select White Model:")
    try:
        white_choice = int(input(f"Enter choice (1-{len(available_models)}): ").strip())
        if not (1 <= white_choice <= len(available_models)):
            print("❌ Invalid choice")
            return
        
        white_model = available_models[white_choice - 1]
        print(f"🤍 White: {white_model}")
        
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Get user selection for Black
    print(f"\n🖤 Select Black Model:")
    try:
        black_choice = int(input(f"Enter choice (1-{len(available_models)}): ").strip())
        if not (1 <= black_choice <= len(available_models)):
            print("❌ Invalid choice")
            return
        
        black_model = available_models[black_choice - 1]
        print(f"🖤 Black: {black_model}")
        
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Check if same model selected
    if white_model == black_model:
        print("⚠️  Same model selected for both sides - this will be interesting!")
    
    # Battle configuration
    print(f"\n⚙️  Battle Configuration:")
    print(f"   🤍 White: {white_model}")
    print(f"   🖤 Black: {black_model}")
    
    # Get number of games
    try:
        num_games = int(input("Enter number of games to play (default: 10): ").strip() or "10")
        if num_games <= 0:
            print("❌ Number of games must be positive")
            return
    except ValueError:
        print("❌ Invalid input, using 10 games")
        num_games = 10
    
    # Get max moves per game
    try:
        max_moves = int(input("Enter max moves per game (default: 200): ").strip() or "200")
        if max_moves <= 0:
            print("❌ Max moves must be positive")
            return
    except ValueError:
        print("❌ Invalid input, using 200 moves")
        max_moves = 200
    
    # Get display preference
    show_moves = input("Show moves in real-time? (y/n, default: n): ").strip().lower()
    show_moves = show_moves in ['y', 'yes']
    
    # Battle mode selection
    print(f"\n🎮 Battle Modes:")
    print("   1. Full Battle (detailed output)")
    print("   2. Quick Battle (minimal output)")
    print("   3. Tournament Mode (all vs all)")
    
    try:
        mode_choice = input("Select mode (1-3, default: 1): ").strip() or "1"
        if mode_choice not in ['1', '2', '3']:
            print("❌ Invalid choice, using Full Battle")
            mode_choice = "1"
    except ValueError:
        mode_choice = "1"
    
    # Confirm battle
    print(f"\n🚀 Battle Configuration:")
    print(f"   🤍 White: {white_model}")
    print(f"   🖤 Black: {black_model}")
    print(f"   🎮 Games: {num_games}")
    print(f"   📏 Max moves: {max_moves}")
    print(f"   📺 Show moves: {'Yes' if show_moves else 'No'}")
    print(f"   🎯 Mode: {['Full Battle', 'Quick Battle', 'Tournament'][int(mode_choice)-1]}")
    
    confirm = input("\nStart the battle? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Battle cancelled")
        return
    
    # Start battle based on mode
    print(f"\n⚔️  Starting Battle...")
    print("=" * 60)
    
    try:
        if mode_choice == "1":
            # Full Battle
            results = battle_system.battle_models(
                white_model, black_model, num_games, max_moves, show_moves
            )
        elif mode_choice == "2":
            # Quick Battle
            results = battle_system.quick_battle(
                white_model, black_model, num_games, show_moves
            )
        else:
            # Tournament Mode
            models = [white_model, black_model]
            results = battle_system.tournament_mode(models, num_games, not show_moves)
        
        if results:
            print(f"\n🎉 Battle completed successfully!")
            if 'pgn_file' in results:
                print(f"💾 Battle PGN saved: {results['pgn_file']}")
            print(f"📊 Check the battles/ directory for results!")
        
    except Exception as e:
        print(f"❌ Battle failed: {e}")
        import traceback
        traceback.print_exc()


def quick_battle_menu():
    """Quick battle menu for fast model selection"""
    print("⚡ Quick Battle Menu")
    print("=" * 30)
    
    battle_system = ModelVsModelBattle()
    available_models = battle_system.list_available_models()
    
    if not available_models:
        print("❌ No models available")
        return
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    try:
        white_idx = int(input("White model (number): ")) - 1
        black_idx = int(input("Black model (number): ")) - 1
        
        if 0 <= white_idx < len(available_models) and 0 <= black_idx < len(available_models):
            white_model = available_models[white_idx]
            black_model = available_models[black_idx]
            
            num_games = int(input("Number of games (default 5): ") or "5")
            
            print(f"\n⚡ Quick Battle: {white_model} vs {black_model}")
            results = battle_system.quick_battle(white_model, black_model, num_games)
            
            if results:
                print(f"🎉 Quick battle completed!")
        else:
            print("❌ Invalid model selection")
            
    except (ValueError, IndexError):
        print("❌ Invalid input")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_battle_menu()
    else:
        main()

