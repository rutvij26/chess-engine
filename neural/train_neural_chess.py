#!/usr/bin/env python3
"""
Neural Chess Engine Training Script
Trains a neural network to play chess through self-play learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from neural_chess_engine import NeuralChessEngine
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def plot_training_progress(losses, game_scores, save_path="training_progress.png"):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss over time
    ax1.plot(losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot game scores over time
    ax2.plot(game_scores)
    ax2.set_title('Game Scores Over Time')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Final Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def train_neural_chess_engine(
    num_games=100,
    epochs_per_game=3,
    learning_rate=0.001,
    save_interval=10,
    model_name="chess_neural_model"
):
    """Train the neural chess engine through self-play"""
    
    print("🧠 Neural Chess Engine Training")
    print("=" * 40)
    print(f"Training for {num_games} games")
    print(f"Epochs per game: {epochs_per_game}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 40)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Adjust learning rate
    for param_group in engine.optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Training tracking
    all_losses = []
    game_scores = []
    best_score = float('-inf')
    
    start_time = time.time()
    
    for game_num in range(num_games):
        print(f"\n🎮 Game {game_num + 1}/{num_games}")
        
        # Play a game and collect training data
        game_data = engine.self_play_game(max_moves=50)
        
        if len(game_data) > 0:
            # Get final game score
            final_position = game_data[-1][0]
            final_score = game_data[-1][1]
            game_scores.append(final_score)
            
            print(f"Game length: {len(game_data)} moves")
            print(f"Final score: {final_score:.2f}")
            
            # Train the model
            print("Training model...")
            engine.train_model(epochs_per_game)
            
            # Track loss (simplified - using last batch loss)
            if hasattr(engine, 'last_loss'):
                all_losses.append(engine.last_loss)
            
            # Save model periodically
            if (game_num + 1) % save_interval == 0:
                model_path = f"{model_name}_game_{game_num + 1}.pth"
                torch.save(engine.model.state_dict(), model_path)
                print(f"💾 Model saved: {model_path}")
                
                # Save training data
                engine.save_training_data(f"training_data_game_{game_num + 1}.pkl")
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time_per_game = elapsed / (game_num + 1)
        remaining_games = num_games - (game_num + 1)
        eta = remaining_games * avg_time_per_game
        
        print(f"⏱️  Progress: {game_num + 1}/{num_games} ({100 * (game_num + 1) / num_games:.1f}%)")
        print(f"⏰ ETA: {eta/60:.1f} minutes")
    
    # Final save
    final_model_path = f"{model_name}_final.pth"
    torch.save(engine.model.state_dict(), final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🎮 Games played: {num_games}")
    print(f"🧠 Model saved: {final_model_path}")
    
    # Plot progress if we have data
    if all_losses and game_scores:
        try:
            plot_training_progress(all_losses, game_scores)
        except Exception as e:
            print(f"Could not plot progress: {e}")
    
    return engine

def test_trained_model(model_path, num_test_games=10):
    """Test a trained neural chess model"""
    print(f"\n🧪 Testing trained model: {model_path}")
    print("=" * 40)
    
    # Load trained model
    engine = NeuralChessEngine(model_path)
    
    # Test performance
    wins = 0
    draws = 0
    losses = 0
    
    for game_num in range(num_test_games):
        print(f"\n🎮 Test game {game_num + 1}/{num_test_games}")
        
        engine.reset_board()
        moves_played = 0
        
        while not engine.board.is_game_over() and moves_played < 50:
            # Get best move from neural network
            best_move = engine.get_best_move(4, 2.0)
            
            if best_move:
                engine.make_move(best_move)
                moves_played += 1
                
                # Show position every 10 moves
                if moves_played % 10 == 0:
                    print(f"Move {moves_played}: {best_move}")
                    print(f"Position evaluation: {engine.evaluate_position_neural(engine.board):.2f}")
            else:
                break
        
        # Determine game result
        if engine.board.is_checkmate():
            if engine.board.turn:  # Black's turn = White won
                wins += 1
                print("✅ White won by checkmate")
            else:
                losses += 1
                print("❌ Black won by checkmate")
        elif engine.board.is_stalemate():
            draws += 1
            print("🤝 Draw by stalemate")
        else:
            draws += 1
            print("🤝 Draw by other means")
    
    # Test results
    print(f"\n📊 Test Results ({num_test_games} games):")
    print(f"✅ Wins: {wins}")
    print(f"🤝 Draws: {draws}")
    print(f"❌ Losses: {losses}")
    print(f"🏆 Win rate: {100 * wins / num_test_games:.1f}%")

def main():
    """Main training and testing function"""
    print("🧠 Neural Chess Engine - Training and Testing")
    print("=" * 50)
    
    # Training parameters
    NUM_GAMES = 50  # Start with fewer games for testing
    EPOCHS_PER_GAME = 3
    LEARNING_RATE = 0.001
    
    # Train the engine
    print("🚀 Starting training...")
    trained_engine = train_neural_chess_engine(
        num_games=NUM_GAMES,
        epochs_per_game=EPOCHS_PER_GAME,
        learning_rate=LEARNING_RATE,
        save_interval=10,
        model_name="chess_neural"
    )
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_trained_model("chess_neural_final.pth", num_test_games=5)
    
    print("\n🎉 All done! The neural chess engine has learned to play chess!")

if __name__ == "__main__":
    main()
