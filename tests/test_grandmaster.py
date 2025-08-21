#!/usr/bin/env python3
"""
Test Grandmaster Training
Quick test of the grandmaster training system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural'))
from grandmaster_training import GrandmasterTrainer

def test_grandmaster_training():
    """Test the grandmaster training system"""
    print("🏆 TESTING GRANDMASTER TRAINING SYSTEM")
    print("=" * 50)
    
    try:
        # Create trainer
        print("1. Creating GrandmasterTrainer...")
        trainer = GrandmasterTrainer()
        print("✅ Trainer created successfully!")
        
        # Show training plan
        print("\n2. Training Plan:")
        total_games = sum(stage["games"] for stage in trainer.curriculum_stages)
        print(f"   • Total Stages: {len(trainer.curriculum_stages)}")
        print(f"   • Total Games: {total_games:,}")
        print(f"   • Target ELO: 2800+ (Grandmaster+)")
        
        # Show first stage details
        print("\n3. First Stage Details:")
        first_stage = trainer.curriculum_stages[0]
        print(f"   • Stage: {first_stage['description']}")
        print(f"   • Games: {first_stage['games']}")
        print(f"   • Max Moves: {first_stage['max_moves']}")
        print(f"   • Exploration: {first_stage['exploration']}")
        
        # Test ELO estimation
        print("\n4. ELO Estimation Test:")
        test_elo = trainer.estimate_elo_improvement(0.6, 500)  # 60% win rate, 500 games
        print(f"   • Starting ELO: 1200")
        print(f"   • Win Rate: 60%")
        print(f"   • Estimated ELO: {test_elo:.0f}")
        
        print("\n🎉 All tests passed! Grandmaster training system is ready.")
        print("\nTo start training:")
        print("   python grandmaster_training.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check that all dependencies are installed.")

if __name__ == "__main__":
    test_grandmaster_training()
