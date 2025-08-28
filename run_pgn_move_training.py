#!/usr/bin/env python3
"""
PGN Move Training Runner
Extends existing neural network by training on PGN moves instead of self-play
"""

import os
import sys
from neural.pgn_move_trainer import PGNMoveTrainer


def main():
    """Main function for PGN move training"""
    print("🧠 Neural Chess Engine - PGN Move Training")
    print("📚 Extends existing network with real game moves")
    print("=" * 50)
    
    # Check if datasets directory exists
    datasets_dir = "datasets/pgn"
    if not os.path.exists(datasets_dir):
        print(f"❌ Datasets directory not found: {datasets_dir}")
        print("Please create the directory and add PGN files")
        return
    
    # List available PGN files
    pgn_files = [f for f in os.listdir(datasets_dir) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"❌ No PGN files found in {datasets_dir}")
        print("Please add PGN files to continue")
        return
    
    print("📁 Available PGN files:")
    for i, pgn_file in enumerate(pgn_files, 1):
        file_path = os.path.join(datasets_dir, pgn_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   {i}. {pgn_file} ({file_size:.1f} MB)")
    
    # Get user selection
    try:
        choice = int(input(f"\nSelect PGN file (1-{len(pgn_files)}): ").strip())
        if not (1 <= choice <= len(pgn_files)):
            print("❌ Invalid choice")
            return
        
        selected_pgn = pgn_files[choice - 1]
        pgn_path = os.path.join(datasets_dir, selected_pgn)
        
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Check for existing models to extend from
    models_dir = "models"
    existing_models = []
    if os.path.exists(models_dir):
        existing_models = [f for f in os.listdir(models_dir) 
                          if f.endswith('.pth') and 'final' in f]
    
    # Ask about model loading
    model_path = None
    if existing_models:
        print(f"\n📚 Found existing models to extend from:")
        for i, model in enumerate(existing_models, 1):
            print(f"   {i}. {model}")
        
        try:
            model_choice = input(f"\nExtend from existing model? (1-{len(existing_models)}, or 'n' for fresh start): ").strip()
            if model_choice.lower() != 'n':
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(existing_models):
                    model_path = os.path.join(models_dir, existing_models[model_idx])
                    print(f"📚 Will extend from: {existing_models[model_idx]}")
                else:
                    print("❌ Invalid choice, starting fresh")
            else:
                print("🆕 Starting fresh training")
        except ValueError:
            print("🆕 Starting fresh training")
    
    # Training parameters
    print(f"\n🔧 Training Parameters:")
    print(f"   📁 PGN file: {selected_pgn}")
    if model_path:
        print(f"   📚 Extend from: {os.path.basename(model_path)}")
    else:
        print(f"   🆕 Start fresh")
    
    # Get training parameters
    try:
        epochs = int(input("Enter number of epochs (default: 10): ").strip() or "10")
        batch_size = int(input("Enter batch size (default: 64): ").strip() or "64")
        learning_rate = float(input("Enter learning rate (default: 0.001): ").strip() or "0.001")
        max_games = input("Enter max games to load (default: 1000): ").strip()
        max_games = int(max_games) if max_games else 1000
        max_positions = input("Enter max positions to use (default: 5000): ").strip()
        max_positions = int(max_positions) if max_positions else 5000
        
        # Validate parameters
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            print("❌ Invalid parameters")
            return
            
    except ValueError:
        print("❌ Invalid input, using defaults")
        epochs = 10
        batch_size = 64
        learning_rate = 0.001
        max_games = 1000
        max_positions = 5000
    
    # Confirm training
    print(f"\n🚀 Training Configuration:")
    print(f"   📁 PGN file: {selected_pgn}")
    print(f"   📚 Model: {'Extend existing' if model_path else 'Fresh start'}")
    print(f"   🔄 Epochs: {epochs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📚 Learning rate: {learning_rate}")
    print(f"   🎮 Max games: {max_games}")
    print(f"   📊 Max positions: {max_positions}")
    
    confirm = input("\nStart PGN move training? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Training cancelled")
        return
    
    # Start training
    print(f"\n🚀 Starting PGN Move Training...")
    print("📚 This will extend your existing neural network with real game moves!")
    print("=" * 50)
    
    try:
        # Create trainer
        trainer = PGNMoveTrainer(model_path)
        
        # Train on PGN moves
        results = trainer.train_on_pgn_moves(
            pgn_path=pgn_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_games=max_games,
            max_positions=max_positions
        )
        
        # Training summary
        print(f"\n🎉 PGN Move Training completed successfully!")
        print(f"⏱️  Total time: {results['total_time']/60:.1f} minutes")
        print(f"📊 Positions processed: {results['positions_processed']:,}")
        print(f"🎮 Moves processed: {results['moves_processed']:,}")
        print(f"💾 Extended model saved: {results['final_model_path']}")
        
        # Ask about evaluation
        evaluate = input("\nEvaluate extended model on PGN moves? (y/n): ").strip().lower()
        if evaluate in ['y', 'yes']:
            print("\n" + "="*50)
            trainer.evaluate_model(pgn_path, max_games=100)
        
        print(f"\n🎯 Your neural network has been extended with real game moves!")
        print(f"💾 Extended model: {results['final_model_path']}")
        print(f"🚀 Ready to use for improved chess play!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
