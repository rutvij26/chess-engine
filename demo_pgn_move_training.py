#!/usr/bin/env python3
"""
Demo script for PGN Move Training
Tests the PGN move training functionality with a small dataset
"""

import os
import sys
from neural.pgn_move_trainer import PGNMoveTrainer


def main():
    """Demo PGN move training"""
    print("🧠 PGN Move Training Demo")
    print("📚 Extends existing neural network with real game moves")
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
    print("\n🚀 Creating PGN move trainer...")
    trainer = PGNMoveTrainer()
    
    # Quick training demo (small number of games and positions)
    print("\n🎯 Starting quick PGN move training demo...")
    print("   - Max games: 50")
    print("   - Max positions: 1000")
    print("   - Epochs: 3")
    print("   - Batch size: 32")
    
    try:
        results = trainer.train_on_pgn_moves(
            pgn_path=pgn_path,
            epochs=3,
            batch_size=32,
            learning_rate=0.001,
            max_games=50,
            max_positions=1000
        )
        
        print(f"\n✅ Demo PGN move training completed!")
        print(f"💾 Extended model saved: {results['final_model_path']}")
        
        # Quick evaluation
        print("\n🧪 Quick evaluation of extended model...")
        evaluation = trainer.evaluate_model(pgn_path, max_games=20)
        
        print(f"\n📊 Demo Results:")
        print(f"   🎯 MSE Loss: {evaluation['mse_loss']:.2f}")
        print(f"   🔗 Correlation: {evaluation['correlation']:.4f}")
        print(f"   ✅ Accuracy: {evaluation['accuracy_within_100']:.2%}")
        
    except Exception as e:
        print(f"❌ Demo training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
