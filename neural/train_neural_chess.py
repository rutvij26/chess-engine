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
import threading
import datetime
import signal
import sys

def cleanup_on_exit():
    """Cleanup function to be called on exit"""
    print("\n🛑 Interrupt received. Cleaning up...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ CUDA memory cleared")
    except Exception as e:
        print(f"⚠️  CUDA cleanup warning: {e}")
    print("🔄 Exiting...")
    sys.exit(0)

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

def display_training_status(game_results, game_scores, start_time, num_games):
    """Display real-time training status"""
    if not game_results:
        return
    
    completed = len(game_results)
    elapsed = time.time() - start_time
    progress = (completed / num_games) * 100
    
    # Clear screen (works on most terminals)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🧠 NEURAL CHESS ENGINE - LIVE TRAINING STATUS")
    print("=" * 60)
    print(f"🕐 Started: {datetime.datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
    print(f"🕐 Current: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")
    print()
    
    # Progress bar
    bar_length = 40
    filled = int(bar_length * progress / 100)
    progress_bar = "█" * filled + "░" * (bar_length - filled)
    print(f"📊 Progress: {progress_bar} {progress:.1f}%")
    print(f"🎮 Games: {completed}/{num_games}")
    print()
    
    # Recent games
    print("🎯 RECENT GAMES:")
    for i, result in enumerate(game_results[-5:]):  # Show last 5 games
        game_num = result['game_num']
        score = result['final_score']
        moves = result['moves_played']
        result_str = result['result']
        print(f"   Game {game_num}: Score {score:.2f} | Moves {moves} | Result {result_str}")
    
    print()
    
    # Performance metrics
    if game_scores:
        avg_score = sum(game_scores) / len(game_scores)
        best_score = max(game_scores)
        worst_score = min(game_scores)
        print(f"📈 PERFORMANCE:")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Best Score: {best_score:.2f}")
        print(f"   Worst Score: {worst_score:.2f}")
        
        # ETA calculation
        if completed > 0:
            avg_time_per_game = elapsed / completed
            remaining_games = num_games - completed
            eta = remaining_games * avg_time_per_game
            print(f"   ⏰ ETA: {eta/60:.1f} minutes")
    
    print("=" * 60)

def play_single_game(game_params):
    """Play a single game and return results - for parallel execution"""
    game_num, epochs_per_game, learning_rate, model_name, existing_model_path, randomness = game_params
    
    # Create engine instance - load existing model if available
    if existing_model_path and os.path.exists(existing_model_path):
        print(f"      📚 Loading existing model: {existing_model_path}")
        engine = NeuralChessEngine(existing_model_path)
    else:
        print(f"      🆕 Creating fresh neural network")
        engine = NeuralChessEngine()
    
    # Adjust learning rate
    for param_group in engine.optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print(f"🎮 Game {game_num} starting...")
    
    # Play the game with progress enabled
    game_result = engine.self_play_game(show_progress=True, randomness=randomness)
    
    # Generate PGN for this game
    if game_result['move_history']:
        pgn_game = engine.generate_pgn_game(
            game_result['move_history'], 
            game_result['result']
        )
        engine.save_game_to_history(pgn_game, game_num, game_result)
    
    # Train the model on this game's data with progress
    if len(engine.training_positions) > 0:
        engine.train_model_with_progress(epochs_per_game)
    
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
    num_parallel_games=3,
    existing_model_path=None,
    randomness=0.2
):
    """Train the neural chess engine through parallel self-play"""
    
    print("🧠 Neural Chess Engine Training - PARALLEL MODE")
    print("=" * 50)
    print(f"Training for {num_games} games")
    print(f"Parallel games: {num_parallel_games}")
    print(f"Epochs per game: {epochs_per_game}")
    print(f"Learning rate: {learning_rate}")
    if randomness > 0:
        print(f"🎲 Randomness: {randomness*100:.0f}% (exploration)")
    else:
        print("🤖 Always best move (deterministic)")
    print("=" * 50)
    
    # Adjust parallel games if num_games is less than parallel games
    actual_parallel_games = min(num_parallel_games, num_games)
    if actual_parallel_games != num_parallel_games:
        print(f"⚠️  Adjusted parallel games from {num_parallel_games} to {actual_parallel_games} (num_games: {num_games})")
    
    # Training tracking
    all_losses = []
    game_scores = []
    game_results = []
    
    start_time = time.time()
    
    # Start real-time status display in a separate thread
    status_thread = threading.Thread(
        target=lambda: status_updater(game_results, game_scores, start_time, num_games),
        daemon=True
    )
    status_thread.start()
    
    try:
        # Process games in batches of parallel games
        for batch_start in range(0, num_games, actual_parallel_games):
            batch_end = min(batch_start + actual_parallel_games, num_games)
            batch_size = batch_end - batch_start
            
            print(f"\n🚀 Starting batch {batch_start//actual_parallel_games + 1}: Games {batch_start + 1}-{batch_end}")
            
            # Prepare game parameters for this batch
            game_params = [
                (game_num + 1, epochs_per_game, learning_rate, model_name, existing_model_path, randomness)
                for game_num in range(batch_start, batch_end)
            ]
            
            # Execute games in parallel
            print(f"      🚀 Submitting {len(game_params)} games to parallel executor...")
            
            with ProcessPoolExecutor(max_workers=actual_parallel_games) as executor:
                # Submit all games in the batch
                future_to_game = {
                    executor.submit(play_single_game, params): params[0] 
                    for params in game_params
                }
                
                print(f"      ⏳ Waiting for games to complete...")
                completed_in_batch = 0
                
                # Collect results as they complete with timeout
                for future in as_completed(future_to_game):
                    game_num = future_to_game[future]
                    try:
                        # Add timeout to prevent hanging
                        result = future.result(timeout=300)  # 5 minutes timeout per game
                        game_results.append(result)
                        game_scores.append(result['final_score'])
                        
                        if result['loss'] > 0:
                            all_losses.append(result['loss'])
                        
                        completed_in_batch += 1
                        print(f"      ✅ Game {game_num} completed ({completed_in_batch}/{len(game_params)})")
                        
                        # Show batch progress
                        if completed_in_batch < len(game_params):
                            remaining = len(game_params) - completed_in_batch
                            print(f"      ⏳ Waiting for {remaining} more game(s)...")
                        
                    except Exception as e:
                        print(f"      ❌ Game {game_num} failed: {e}")
                        # Cancel the future to free up resources
                        future.cancel()
            
            # Progress update with detailed statistics
            completed_games = len(game_results)
            elapsed = time.time() - start_time
            
            # Avoid division by zero
            if completed_games > 0:
                avg_time_per_game = elapsed / completed_games
                remaining_games = num_games - completed_games
                eta = remaining_games * avg_time_per_game
            else:
                avg_time_per_game = 0
                remaining_games = num_games
                eta = 0
            
            # Create a visual progress bar
            progress_bar_length = 30
            filled_length = int(progress_bar_length * completed_games / num_games)
            progress_bar = "█" * filled_length + "░" * (progress_bar_length - filled_length)
            
            print(f"\n📊 BATCH PROGRESS SUMMARY:")
            print(f"   {progress_bar} {completed_games}/{num_games} ({100 * completed_games / num_games:.1f}%)")
            print(f"   ⏱️  Elapsed: {elapsed/60:.1f} minutes")
            print(f"   ⏰ Avg time per game: {avg_time_per_game/60:.1f} minutes")
            print(f"   🎯 ETA: {eta/60:.1f} minutes")
            
            # Show performance metrics
            if game_scores:
                avg_score = sum(game_scores) / len(game_scores)
                best_score = max(game_scores)
                print(f"   📈 Avg score: {avg_score:.2f}")
                print(f"   🏆 Best score: {best_score:.2f}")
            
            # Save models periodically
            if completed_games % save_interval == 0:
                print(f"   💾 Saving models after {completed_games} games...")
        
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
        
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        # Cleanup CUDA memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 CUDA memory cleared")
        except Exception as e:
            print(f"⚠️  CUDA cleanup warning: {e}")
        
        # Stop status thread gracefully
        print("🔄 Stopping status updates...")
        try:
            # Set a flag to stop the status thread
            if 'status_thread' in locals() and status_thread.is_alive():
                status_thread.join(timeout=3)  # Wait up to 3 seconds
                if status_thread.is_alive():
                    print("⚠️  Status thread did not stop gracefully")
                else:
                    print("✅ Status thread stopped gracefully")
        except Exception as e:
            print(f"⚠️  Status thread cleanup warning: {e}")
        
        print("🧹 Cleanup completed")

def status_updater(game_results, game_scores, start_time, num_games):
    """Update status display every few seconds"""
    try:
        while len(game_results) < num_games:
            time.sleep(5)  # Update every 5 seconds
            display_training_status(game_results, game_scores, start_time, num_games)
    except Exception as e:
        print(f"⚠️  Status updater error: {e}")
    finally:
        print("🔄 Status updates stopped")

def train_neural_chess_engine(
    num_games=100,
    epochs_per_game=3,
    learning_rate=0.001,
    save_interval=10,
    model_name="chess_neural_model",
    randomness=0.2
):
    """Train the neural chess engine through self-play (legacy single-threaded version)"""
    
    print("🧠 Neural Chess Engine Training - SINGLE THREADED")
    print("=" * 40)
    print(f"Training for {num_games} games")
    print(f"Epochs per game: {epochs_per_game}")
    print(f"Learning rate: {learning_rate}")
    if randomness > 0:
        print(f"🎲 Randomness: {randomness*100:.0f}% (exploration)")
    else:
        print("🤖 Always best move (deterministic)")
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
        print("=" * 40)
        
        # Play a game and collect training data
        print("🎯 Playing game...")
        game_result = engine.self_play_game(show_progress=True, randomness=randomness)
        
        if game_result['game_data']:
            # Get final game score
            final_score = game_result['final_evaluation']
            game_scores.append(final_score)
            
            print(f"📊 Game Summary:")
            print(f"   Length: {game_result['moves_played']} moves")
            print(f"   Final score: {final_score:.2f}")
            print(f"   Result: {game_result['result']}")
            
            # Generate and save PGN for this game
            if game_result['move_history']:
                print("📜 Generating PGN...")
                pgn_game = engine.generate_pgn_game(
                    game_result['move_history'], 
                    game_result['result']
                )
                engine.save_game_to_history(pgn_game, game_num + 1, game_result)
            
            # Train the model
            print("🧠 Training model...")
            engine.train_model_with_progress(epochs_per_game)
            
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
        
        # Progress update with visual bar
        elapsed = time.time() - start_time
        
        # Avoid division by zero
        if game_num + 1 > 0:
            avg_time_per_game = elapsed / (game_num + 1)
            remaining_games = num_games - (game_num + 1)
            eta = remaining_games * avg_time_per_game
        else:
            avg_time_per_game = 0
            remaining_games = num_games
            eta = 0
        
        # Create progress bar
        progress_bar_length = 30
        filled_length = int(progress_bar_length * (game_num + 1) / num_games)
        progress_bar = "█" * filled_length + "░" * (progress_bar_length - filled_length)
        
        print(f"\n📊 OVERALL PROGRESS:")
        print(f"   {progress_bar} {game_num + 1}/{num_games} ({100 * (game_num + 1) / num_games:.1f}%)")
        print(f"   ⏱️  Elapsed: {elapsed/60:.1f} minutes")
        print(f"   ⏰ ETA: {eta/60:.1f} minutes")
        
        # Show performance so far
        if game_scores:
            avg_score = sum(game_scores) / len(game_scores)
            best_score = max(game_scores)
            print(f"   📈 Avg score: {avg_score:.2f}")
            print(f"   🏆 Best score: {best_score:.2f}")
        
        print("=" * 40)
    
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
                engine.make_move(best_move.uci())
                moves_played += 1
                
                # Show position every 10 moves
                if moves_played % 10 == 0:
                    print(f"Move {moves_played}: {best_move.uci()}")
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
    # Set up signal handlers for graceful cleanup
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_on_exit())
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_on_exit())
    
    print("🧠 Neural Chess Engine - Training and Testing")
    print("=" * 50)
    print("Choose an option:")
    print("1. 🚀 Train model (incremental - continues from existing version)")
    print("2. 🧪 Test existing model (evaluation only)")
    print("3. 🔄 Train and then test")
    print("4. 🚪 Quit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Training only
            run_training()
            break
        elif choice == "2":
            # Testing only
            run_testing()
            break
        elif choice == "3":
            # Train then test
            run_training()
            print("\n" + "="*50)
            run_testing()
            break
        elif choice == "4":
            # Quit
            print("\n👋 Thanks for using Neural Chess Engine!")
            print("🚪 Exiting...")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def run_training():
    """Run the training process"""
    print("\n🚀 INCREMENTAL TRAINING MODE")
    print("=" * 30)
    
    # Training parameters
    NUM_GAMES = 30  # Start with fewer games for testing
    EPOCHS_PER_GAME = 3
    LEARNING_RATE = 0.001
    NUM_PARALLEL_GAMES = 3  # Run 3 games simultaneously
    
    # Check for existing models and determine version
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Find the latest version
    existing_models = [f for f in os.listdir(models_dir) if f.startswith("chess_neural_v") and f.endswith("_final.pth")]
    existing_model_path = None
    
    if existing_models:
        # Extract version numbers and find the latest
        versions = []
        for model in existing_models:
            try:
                version = int(model.split("_v")[1].split("_")[0])
                versions.append(version)
            except:
                continue
        
        if versions:
            latest_version = max(versions)
            next_version = latest_version + 1
            existing_model_path = f"{models_dir}/chess_neural_v{latest_version}_final.pth"
            print(f"📚 Found existing model: Version {latest_version}")
            print(f"📁 Model path: {existing_model_path}")
            print(f"🔄 Will continue training to create: Version {next_version}")
            
            # Ask user if they want to continue from existing model
            continue_choice = input("Continue training from existing model? (y/n, default: y): ").strip().lower()
            if continue_choice in ['', 'y', 'yes']:
                base_model_name = f"chess_neural_v{next_version}"
                print(f"🚀 Continuing training to create {base_model_name}")
                print(f"📚 Will load existing model: {existing_model_path}")
            else:
                base_model_name = "chess_neural_v1"
                existing_model_path = None
                print(f"🆕 Starting fresh training with {base_model_name}")
        else:
            base_model_name = "chess_neural_v1"
            print(f"🆕 Starting fresh training with {base_model_name}")
    else:
        base_model_name = "chess_neural_v1"
        print(f"🆕 Starting fresh training with {base_model_name}")
    
    # Get number of games from user
    try:
        num_games = int(input(f"Enter number of games to train (default: {NUM_GAMES}): ").strip() or NUM_GAMES)
    except ValueError:
        num_games = NUM_GAMES
        print(f"Invalid input, using default: {NUM_GAMES} games")
    
    # Get randomness setting from user
    print("\n🎲 Randomness Settings:")
    print("   0.0 = Always best move (deterministic)")
    print("   0.2 = 20% random moves (balanced exploration) - RECOMMENDED")
    print("   0.5 = 50% random moves (high exploration)")
    print("   1.0 = Always random moves (pure exploration)")
    
    try:
        randomness_input = input(f"Enter randomness (0.0-1.0, default: 0.2): ").strip() or "0.2"
        randomness = float(randomness_input)
        if not (0.0 <= randomness <= 1.0):
            raise ValueError("Randomness must be between 0.0 and 1.0")
    except ValueError:
        randomness = 0.2
        print(f"Invalid input, using default: 0.2 (20% randomness)")
    
    print(f"\n🚀 Starting unified training for {num_games} games...")
    print(f"Running {NUM_PARALLEL_GAMES} games simultaneously")
    print(f"📚 Model version: {base_model_name}")
    if randomness > 0:
        print(f"🎲 Randomness: {randomness*100:.0f}% (exploration)")
    else:
        print("🤖 Always best move (deterministic)")
    
    try:
        game_results = train_neural_chess_engine_parallel(
            num_games=num_games,
            epochs_per_game=EPOCHS_PER_GAME,
            learning_rate=LEARNING_RATE,
            save_interval=10,
            model_name=base_model_name,
            num_parallel_games=NUM_PARALLEL_GAMES,
            existing_model_path=existing_model_path,
            randomness=randomness
        )
        
        print("\n🎉 Training completed!")
        print("📜 All games have been saved to 'games/game_histories.pgn'")
        if existing_model_path:
            print(f"🧠 Model version {base_model_name} created by continuing training from previous version")
        else:
            print(f"🧠 Fresh model version {base_model_name} created from scratch")
        print(f"💾 Model saved as: {base_model_name}_final.pth")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        print("\n🧹 Final cleanup...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ CUDA memory cleared")
        except Exception as e:
            print(f"⚠️  CUDA cleanup warning: {e}")
        
        print("🔄 Returning to main menu...")
        time.sleep(2)  # Give user time to see cleanup messages

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
