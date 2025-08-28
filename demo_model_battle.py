#!/usr/bin/env python3
"""
Demo script for Model vs Model Battle
Shows how to use the battle system programmatically
"""

import os
import sys
from neural.model_vs_model import ModelVsModelBattle


def demo_battle():
    """Demo the model vs model battle system"""
    print("⚔️  Model vs Model Battle Demo")
    print("=" * 40)
    
    # Create battle system
    battle_system = ModelVsModelBattle()
    
    # List available models
    available_models = battle_system.list_available_models()
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train some models first using the neural training option.")
        return
    
    print(f"📚 Found {len(available_models)} trained models:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    # Select models for demo
    if len(available_models) >= 2:
        white_model = available_models[0]
        black_model = available_models[1]
        print(f"\n🎯 Demo Battle: {white_model} vs {black_model}")
        
        # Quick battle demo
        print(f"⚡ Running quick battle (3 games)...")
        results = battle_system.quick_battle(
            white_model, black_model, num_games=3, show_moves=False
        )
        
        if results:
            print(f"\n✅ Demo battle completed!")
            print(f"📊 Results saved to battles/ directory")
        else:
            print(f"❌ Demo battle failed")
    
    elif len(available_models) == 1:
        # Only one model - battle against itself
        model = available_models[0]
        print(f"\n🎯 Demo Battle: {model} vs {model} (self-play)")
        
        results = battle_system.quick_battle(
            model, model, num_games=2, show_moves=False
        )
        
        if results:
            print(f"\n✅ Demo battle completed!")
            print(f"📊 Results saved to battles/ directory")
        else:
            print(f"❌ Demo battle failed")


def demo_tournament():
    """Demo tournament mode"""
    print("\n🏆 Tournament Mode Demo")
    print("=" * 30)
    
    battle_system = ModelVsModelBattle()
    available_models = battle_system.list_available_models()
    
    if len(available_models) >= 2:
        print(f"🎯 Running tournament with {len(available_models)} models")
        print(f"📊 Each model plays {len(available_models)-1} games against others")
        
        results = battle_system.tournament_mode(
            available_models, games_per_matchup=2, show_moves=False
        )
        
        if results:
            print(f"\n✅ Tournament completed!")
            print(f"📊 Results saved to battles/ directory")
        else:
            print(f"❌ Tournament failed")
    else:
        print("❌ Need at least 2 models for tournament")


if __name__ == "__main__":
    print("🧠 Neural Chess Engine - Model Battle Demo")
    print("=" * 50)
    
    # Run demos
    demo_battle()
    demo_tournament()
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 Use 'python run_model_battle.py' for interactive battles")
    print(f"💡 Use 'python run_neural.py' and select option 4 for battles")

