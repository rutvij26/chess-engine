#!/usr/bin/env python3
"""
Demo script for PGN Dataset Training
Tests the PGN training functionality with a small dataset
"""

import os
import sys
from neural.pgn_dataset_trainer import PGNDatasetTrainer


def main():
    """Demo PGN dataset training"""
    print("🧠 PGN Dataset Training Demo")
    print("=" * 40)
    
    # Check for PGN files
    pgn_dir = "datasets/pgn"
    if not os.path.exists(pgn_dir):
        print(f"❌ PGN directory not found: {pgn_dir}")
        return
    
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"❌ No PGN files found in {pgn_dir}")
        return
    
    print(f"📁 Found {len(pgn_files)} PGN file(s)")
    pgn_path = os.path.join(pgn_dir, pgn_files[0])
    print(f"📖 Using: {pgn_files[0]}")
    
    # Create trainer
    print("\n🚀 Creating PGN trainer...")
    trainer = PGNDatasetTrainer()
    
    # Quick training demo (small number of games)
    print("\n🎯 Starting quick training demo...")
    print("   - Max games: 100")
    print("   - Epochs: 3")
    print("   - Batch size: 32")
    
    try:
        results = trainer.train_on_dataset(
            pgn_path=pgn_path,
            epochs=3,
            batch_size=32,
            learning_rate=0.001,
            max_games=100,
            max_positions_per_game=20
        )
        
        print(f"\n✅ Demo training completed!")
        print(f"💾 Model saved: {results['final_model_path']}")
        
    except Exception as e:
        print(f"❌ Demo training failed: {e}")


if __name__ == "__main__":
    main()
