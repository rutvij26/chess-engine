#!/usr/bin/env python3
"""
Neural Chess Engine Training Script
Trains a neural network to play chess through self-play learning
Now with PGN generation and parallel game execution
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .neural_chess_engine import NeuralChessEngine
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import chess.pgn

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

def play_single_game(game_params):
    """Play a single game and return results - for parallel execution"""
    game_num, epochs_per_game, learning_rate, model_name = game_params
    
    # Create a new engine instance for this game
    engine = NeuralChessEngine()
    
    # Adjust learning rate
    for param_group in engine.optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print(f"🎮 Game {game_num} starting...")
    
    # Play the game
    game_result = engine.self_play_game(show_progress=False)
    
    # Generate PGN for this game
    if game_result['move_history']:
        pgn_game = engine.generate_pgn_game(
            game_result['move_history'], 
            game_result['result']
        )
        engine.save_game_to_history(pgn_game, game_num, game_result)
    
    # Train the model on this game's data
    if len(engine.training_positions) > 0:
        engine.train_model(epochs_per_game)
    
    # Get final game score
    final_score = game_result['final_evaluation']
    
    # Save model for this game
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}_game_{game_num}.pth"
    torch.save(engine.model.state_dict(), model_path)
    
    print(f"✅ Game {game_num} completed - Score: {final_score:.3f}, Moves: {game_result['moves_played']}")
    
    return {
        'game_num': game_num,
        'final_score': final_score,
        'moves_played': game_result['moves_played'],
        'result': game_result['result'],
        'model_path': model_path,
        'loss': getattr(engine, 'last_loss', 0.0) if hasattr(engine, 'last_loss') else 0.0
    }

def train_neural_chess_engine_parallel(
    num_games=100,
    epochs_per_game=3,
    learning_rate=0.001,
    save_interval=10,
    model_name="chess_neural_model",
    num_parallel_games=3
):
    """Train the neural chess engine through parallel self-play"""
    
    print("🧠 Neural Chess Engine Training - PARALLEL MODE")
    print("=" * 50)
    print(f"Training for {num_games} games")
    print(f"Parallel games: {num_parallel_games}")
    print(f"Epochs per game: {epochs_per_game}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 50)
    
    # Training tracking
    all_losses = []
    game_scores = []
    game_results = []
    
    start_time = time.time()
    
    # Process games in batches of parallel games
    for batch_start in range(0, num_games, num_parallel_games):
        batch_end = min(batch_start + num_parallel_games, num_games)
        batch_size = batch_end - batch_start
        
        print(f"\n🚀 Starting batch {batch_start//num_parallel_games + 1}: Games {batch_start + 1}-{batch_end}")
        
        # Prepare game parameters for this batch
        game_params = [
            (game_num + 1, epochs_per_game, learning_rate, model_name)
            for game_num in range(batch_start, batch_end)
        ]
        
        # Execute games in parallel
        with ProcessPoolExecutor(max_workers=num_parallel_games) as executor:
            # Submit all games in the batch
            future_to_game = {
                executor.submit(play_single_game, params): params[0] 
                for params in game_params
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_game):
                game_num = future_to_game[future]
                try:
                    result = future.result()
                    game_results.append(result)
                    game_scores.append(result['final_score'])
                    
                    if result['loss'] > 0:
                        all_losses.append(result['loss'])
                    
                    print(f"✅ Game {game_num} completed successfully")
                    
                except Exception as e:
                    print(f"❌ Game {game_num} failed: {e}")
        
        # Progress update
        completed_games = len(game_results)
        elapsed = time.time() - start_time
        avg_time_per_game = elapsed / completed_games
        remaining_games = num_games - completed_games
        eta = remaining_games * avg_time_per_game
        
        print(f"⏱️  Progress: {completed_games}/{num_games} ({100 * completed_games / num_games:.1f}%)")
        print(f"⏰ ETA: {eta/60:.1f} minutes")
        
        # Save models periodically
        if completed_games % save_interval == 0:
            print(f"💾 Saving models after {completed_games} games...")
    
    # Final save
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/{model_name}_final.pth"
    if game_results:
        # Use the last completed game's model as the final model
        last_game_result = game_results[-1]
        import shutil
        shutil.copy(last_game_result['model_path'], final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🎮 Games played: {len(game_results)}")
    print(f"🧠 Model saved: {final_model_path}")
    
    # Plot progress if we have data
    if all_losses and game_scores:
        try:
            plot_training_progress(all_losses, game_scores)
        except Exception as e:
            print(f"Could not plot progress: {e}")
    
    return game_results

def train_neural_chess_engine(
    num_games=100,
    epochs_per_game=3,
    learning_rate=0.001,
    save_interval=10,
    model_name="chess_neural_model"
):
    """Train the neural chess engine through self-play (legacy single-threaded version)"""
    
    print("🧠 Neural Chess Engine Training - SINGLE THREADED")
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
        game_result = engine.self_play_game()
        
        if game_result['game_data']:
            # Get final game score
            final_score = game_result['final_evaluation']
            game_scores.append(final_score)
            
            print(f"Game length: {game_result['moves_played']} moves")
            print(f"Final score: {final_score:.2f}")
            print(f"Game result: {game_result['result']}")
            
            # Generate and save PGN for this game
            if game_result['move_history']:
                pgn_game = engine.generate_pgn_game(
                    game_result['move_history'], 
                    game_result['result']
                )
                engine.save_game_to_history(pgn_game, game_num + 1, game_result)
            
            # Train the model
            print("Training model...")
            engine.train_model(epochs_per_game)
            
            # Track loss (simplified - using last batch loss)
            if hasattr(engine, 'last_loss'):
                all_losses.append(engine.last_loss)
            
            # Save model periodically
            if (game_num + 1) % save_interval == 0:
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{model_name}_game_{game_num + 1}.pth"
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
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/{model_name}_final.pth"
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
    """Test a trained neural chess model - PURE EVALUATION ONLY, NO LEARNING"""
    print(f"\n🧪 Testing trained model: {model_path}")
    print("=" * 40)
    print("⚠️  TESTING MODE: Model weights will NOT be updated during testing")
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
            best_move = engine.get_best_move(4, 2.0, verbose=False)
            
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
    print("Choose an option:")
    print("1. 🚀 Train new model")
    print("2. 🧪 Test existing model (evaluation only)")
    print("3. 🔄 Train and then test")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Training only
        run_training()
    elif choice == "2":
        # Testing only
        run_testing()
    elif choice == "3":
        # Train then test
        run_training()
        print("\n" + "="*50)
        run_testing()
    else:
        print("Invalid choice. Running training by default.")
        run_training()

def run_training():
    """Run the training process"""
    print("\n🚀 TRAINING MODE")
    print("=" * 30)
    
    # Training parameters
    NUM_GAMES = 30  # Start with fewer games for testing
    EPOCHS_PER_GAME = 3
    LEARNING_RATE = 0.001
    NUM_PARALLEL_GAMES = 3  # Run 3 games simultaneously
    
    # Get number of games from user
    try:
        num_games = int(input(f"Enter number of games to train (default: {NUM_GAMES}): ").strip() or NUM_GAMES)
    except ValueError:
        num_games = NUM_GAMES
        print(f"Invalid input, using default: {NUM_GAMES} games")
    
    print(f"\n🚀 Starting unified training for {num_games} games...")
    print(f"Running {NUM_PARALLEL_GAMES} games simultaneously")
    
    game_results = train_neural_chess_engine_parallel(
        num_games=num_games,
        epochs_per_game=EPOCHS_PER_GAME,
        learning_rate=LEARNING_RATE,
        save_interval=10,
        model_name="chess_neural",
        num_parallel_games=NUM_PARALLEL_GAMES
    )
    
    print("\n🎉 Training completed!")
    print("📜 All games have been saved to 'games/game_histories.pgn'")
    print("🧠 All models have been saved to 'models/' directory")

def run_testing():
    """Run the testing process"""
    print("\n🧪 TESTING MODE")
    print("=" * 30)
    
    # Check for available models
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found. Please train a model first.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print("❌ No model files found. Please train a model first.")
        return
    
    print("Available models:")
    for i, model in enumerate(sorted(model_files)):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input(f"\nSelect model to test (1-{len(model_files)}): ").strip())
        if 1 <= choice <= len(model_files):
            selected_model = sorted(model_files)[choice-1]
            model_path = f"{models_dir}/{selected_model}"
            
            num_test_games = int(input("Enter number of test games (default: 5): ").strip() or "5")
            
            print(f"\n🧪 Testing model: {selected_model}")
            test_trained_model(model_path, num_test_games)
        else:
            print("Invalid choice.")
    except (ValueError, IndexError):
        print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
