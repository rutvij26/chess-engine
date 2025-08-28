#!/usr/bin/env python3
"""
Randomness Training Demo
Demonstrates different randomness settings for self-play training
"""

import sys
import os

# Add the neural directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from neural.neural_chess_engine import NeuralChessEngine

def demo_randomness_training():
    """Demonstrate different randomness settings for training"""
    print("🎲 Randomness Training Demo")
    print("=" * 50)
    print("This demo shows how different randomness settings affect training:")
    print("• 0.0 = Always best move (deterministic)")
    print("• 0.2 = 20% random moves (balanced exploration) - RECOMMENDED")
    print("• 0.5 = 50% random moves (high exploration)")
    print("• 1.0 = Always random moves (pure exploration)")
    print("=" * 50)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Demo 1: Always best move (deterministic)
    print("\n🎯 Demo 1: Always Best Move (Deterministic)")
    print("-" * 40)
    print("Training with randomness=0.0 (always best move)")
    print("This will create very focused, consistent play")
    print("But may get stuck in local optima")
    
    try:
        # Train for just a few games to demonstrate
        engine.train_on_self_play(num_games=3, epochs_per_game=1, randomness=0.0)
    except KeyboardInterrupt:
        print("\n⏹️  Demo 1 interrupted by user")
    
    # Demo 2: Balanced exploration (20% randomness)
    print("\n🎲 Demo 2: Balanced Exploration (20% Randomness)")
    print("-" * 40)
    print("Training with randomness=0.2 (20% random moves)")
    print("This provides a good balance of focus and exploration")
    print("RECOMMENDED for most training scenarios")
    
    try:
        # Train for just a few games to demonstrate
        engine.train_on_self_play(num_games=3, epochs_per_game=1, randomness=0.2)
    except KeyboardInterrupt:
        print("\n⏹️  Demo 2 interrupted by user")
    
    # Demo 3: High exploration (50% randomness)
    print("\n🎲 Demo 3: High Exploration (50% Randomness)")
    print("-" * 40)
    print("Training with randomness=0.5 (50% random moves)")
    print("This encourages more diverse play patterns")
    print("Useful for breaking out of repetitive strategies")
    
    try:
        # Train for just a few games to demonstrate
        engine.train_on_self_play(num_games=3, epochs_per_game=1, randomness=0.5)
    except KeyboardInterrupt:
        print("\n⏹️  Demo 3 interrupted by user")
    
    # Demo 4: Pure exploration (100% randomness)
    print("\n🎲 Demo 4: Pure Exploration (100% Randomness)")
    print("-" * 40)
    print("Training with randomness=1.0 (always random moves)")
    print("This is purely for exploration and data collection")
    print("Not recommended for focused training")
    
    try:
        # Train for just a few games to demonstrate
        engine.train_on_self_play(num_games=3, epochs_per_game=1, randomness=1.0)
    except KeyboardInterrupt:
        print("\n⏹️  Demo 4 interrupted by user")
    
    print("\n🎉 Randomness Training Demo Completed!")
    print("=" * 50)
    print("Key Takeaways:")
    print("• randomness=0.0: Best for focused, consistent training")
    print("• randomness=0.2: Best for balanced training (RECOMMENDED)")
    print("• randomness=0.5: Best for breaking out of repetitive patterns")
    print("• randomness=1.0: Best for pure exploration and data collection")
    print("\nYou can now use these settings in your actual training!")

def interactive_randomness_training():
    """Interactive randomness training with user input"""
    print("\n🎲 Interactive Randomness Training")
    print("=" * 50)
    
    # Get training parameters from user
    try:
        num_games = int(input("Enter number of games to train (default: 10): ").strip() or "10")
        epochs_per_game = int(input("Enter epochs per game (default: 2): ").strip() or "2")
        
        print("\n🎲 Randomness Settings:")
        print("   0.0 = Always best move (deterministic)")
        print("   0.2 = 20% random moves (balanced exploration) - RECOMMENDED")
        print("   0.5 = 50% random moves (high exploration)")
        print("   1.0 = Always random moves (pure exploration)")
        
        randomness_input = input("Enter randomness (0.0-1.0, default: 0.2): ").strip() or "0.2"
        randomness = float(randomness_input)
        
        if not (0.0 <= randomness <= 1.0):
            raise ValueError("Randomness must be between 0.0 and 1.0")
            
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        print("Using default values: 10 games, 2 epochs, 20% randomness")
        num_games = 10
        epochs_per_game = 2
        randomness = 0.2
    
    # Create engine and start training
    print(f"\n🚀 Starting training with:")
    print(f"   🎮 Games: {num_games}")
    print(f"   🧠 Epochs per game: {epochs_per_game}")
    if randomness > 0:
        print(f"   🎲 Randomness: {randomness*100:.0f}% (exploration)")
    else:
        print("   🤖 Always best move (deterministic)")
    
    engine = NeuralChessEngine()
    
    try:
        engine.train_on_self_play(
            num_games=num_games, 
            epochs_per_game=epochs_per_game, 
            randomness=randomness
        )
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

def main():
    """Main function"""
    while True:
        print("\n🎲 Randomness Training Demo")
        print("=" * 40)
        print("Choose an option:")
        print("1. Run Full Demo (all randomness settings)")
        print("2. Interactive Training (custom settings)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            demo_randomness_training()
            
        elif choice == "2":
            interactive_randomness_training()
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-3.")
            continue
        
        # Ask if user wants to continue
        print("\n" + "=" * 40)
        continue_choice = input("Return to main menu? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
