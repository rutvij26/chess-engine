#!/usr/bin/env python3
"""
PGN Dataset Trainer for Neural Chess Engine
Trains the model on existing PGN game files for faster learning
"""

import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import random

from .neural_chess_engine import NeuralChessEngine


class PGNDatasetTrainer:
    """Trainer for learning from PGN dataset files"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the PGN dataset trainer
        
        Args:
            model_path: Path to existing model to continue training from
        """
        if model_path and os.path.exists(model_path):
            print(f"📚 Loading existing model: {model_path}")
            self.engine = NeuralChessEngine(model_path)
        else:
            print("🆕 Creating fresh neural network")
            self.engine = NeuralChessEngine()
        
        self.device = self.engine.device
        print(f"🚀 Using device: {self.device}")
        
        # Training statistics
        self.training_stats = {
            'total_positions': 0,
            'total_games': 0,
            'training_losses': [],
            'validation_losses': [],
            'epochs_completed': 0
        }
    
    def load_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load games from a PGN file
        
        Args:
            pgn_path: Path to the PGN file
            max_games: Maximum number of games to load (None for all)
            
        Returns:
            List of game data dictionaries
        """
        print(f"📖 Loading PGN file: {pgn_path}")
        
        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        
        games_data = []
        game_count = 0
        
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    if max_games and game_count >= max_games:
                        break
                    
                    # Extract game information
                    game_data = self._extract_game_data(game)
                    if game_data:
                        games_data.append(game_data)
                        game_count += 1
                        
                        if game_count % 1000 == 0:
                            print(f"   📊 Loaded {game_count} games...")
                    
                except Exception as e:
                    print(f"⚠️  Error reading game {game_count}: {e}")
                    continue
        
        print(f"✅ Successfully loaded {len(games_data)} games from PGN file")
        return games_data
    
    def _extract_game_data(self, game: chess.pgn.Game) -> Optional[Dict[str, Any]]:
        """Extract training data from a single game
        
        Args:
            game: Chess game object from python-chess
            
        Returns:
            Dictionary containing game data or None if invalid
        """
        try:
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "*":
                return None  # Skip incomplete games
            
            # Convert result to numerical value
            if result == "1-0":
                result_value = 1.0  # White wins
            elif result == "0-1":
                result_value = -1.0  # Black wins
            elif result == "1/2-1/2":
                result_value = 0.0  # Draw
            else:
                return None
            
            # Extract positions and moves
            positions = []
            evaluations = []
            board = game.board()
            
            # Add starting position
            positions.append(board.copy())
            evaluations.append(result_value * 0.1)  # Slight bias toward result
            
            # Add positions after each move
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board.copy())
                
                # Evaluate position based on game result and current material
                material_score = self._calculate_material_score(board)
                position_eval = result_value * 0.3 + material_score * 0.7
                evaluations.append(position_eval)
            
            return {
                'positions': positions,
                'evaluations': evaluations,
                'result': result_value,
                'moves_count': len(game.mainline_moves()),
                'white_player': game.headers.get("White", "Unknown"),
                'black_player': game.headers.get("Black", "Unknown"),
                'event': game.headers.get("Event", "Unknown"),
                'date': game.headers.get("Date", "Unknown")
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting game data: {e}")
            return None
    
    def _calculate_material_score(self, board: chess.Board) -> float:
        """Calculate material-based position score
        
        Args:
            board: Chess board position
            
        Returns:
            Material score between -1 and 1
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King value not included in material calculation
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Normalize to [-1, 1] range
        total_material = white_material + black_material
        if total_material == 0:
            return 0.0
        
        material_score = (white_material - black_material) / total_material
        return max(-1.0, min(1.0, material_score))
    
    def prepare_training_data(self, games_data: List[Dict[str, Any]], 
                            max_positions_per_game: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from games
        
        Args:
            games_data: List of game data dictionaries
            max_positions_per_game: Maximum positions to use per game
            
        Returns:
            Tuple of (positions_tensor, evaluations_tensor)
        """
        print("🔧 Preparing training data...")
        
        all_positions = []
        all_evaluations = []
        
        for game_data in tqdm(games_data, desc="Processing games"):
            positions = game_data['positions']
            evaluations = game_data['evaluations']
            
            # Limit positions per game to avoid memory issues
            if len(positions) > max_positions_per_game:
                # Sample positions evenly across the game
                indices = np.linspace(0, len(positions) - 1, max_positions_per_game, dtype=int)
                positions = [positions[i] for i in indices]
                evaluations = [evaluations[i] for i in indices]
            
            # Convert positions to tensors
            for position, evaluation in zip(positions, evaluations):
                try:
                    position_tensor = self.engine.board_to_tensor(position)
                    all_positions.append(position_tensor)
                    all_evaluations.append(torch.tensor([evaluation], dtype=torch.float32))
                except Exception as e:
                    print(f"⚠️  Error processing position: {e}")
                    continue
        
        # Stack all tensors
        if not all_positions:
            raise ValueError("No valid positions found in games data")
        
        positions_tensor = torch.cat(all_positions, dim=0)
        evaluations_tensor = torch.cat(all_evaluations, dim=0)
        
        # Move to device
        positions_tensor = positions_tensor.to(self.device)
        evaluations_tensor = evaluations_tensor.to(self.device)
        
        print(f"✅ Prepared {len(positions_tensor)} positions for training")
        return positions_tensor, evaluations_tensor
    
    def train_on_dataset(self, pgn_path: str, 
                        epochs: int = 10,
                        batch_size: int = 64,
                        learning_rate: float = 0.001,
                        max_games: Optional[int] = None,
                        max_positions_per_game: int = 50,
                        validation_split: float = 0.1,
                        save_interval: int = 5) -> Dict[str, Any]:
        """Train the model on PGN dataset
        
        Args:
            pgn_path: Path to PGN file
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            max_games: Maximum games to load
            max_positions_per_game: Maximum positions per game
            validation_split: Fraction of data for validation
            save_interval: Save model every N epochs
            
        Returns:
            Training statistics dictionary
        """
        print(f"🚀 Starting PGN dataset training")
        print(f"📁 PGN file: {pgn_path}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📚 Learning rate: {learning_rate}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Load PGN data
        games_data = self.load_pgn_file(pgn_path, max_games)
        if not games_data:
            raise ValueError("No valid games found in PGN file")
        
        # Prepare training data
        positions_tensor, evaluations_tensor = self.prepare_training_data(
            games_data, max_positions_per_game
        )
        
        # Split into training and validation
        num_positions = len(positions_tensor)
        num_validation = int(num_positions * validation_split)
        num_training = num_positions - num_validation
        
        # Shuffle data
        indices = torch.randperm(num_positions)
        positions_tensor = positions_tensor[indices]
        evaluations_tensor = evaluations_tensor[indices]
        
        # Split data
        train_positions = positions_tensor[:num_training]
        train_evaluations = evaluations_tensor[:num_training]
        val_positions = positions_tensor[num_training:]
        val_evaluations = evaluations_tensor[num_training:]
        
        print(f"📊 Training positions: {num_training}")
        print(f"📊 Validation positions: {num_validation}")
        
        # Set up optimizer
        optimizer = optim.Adam(self.engine.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.engine.model.train()
            train_losses = []
            
            # Create batches
            num_batches = (num_training + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_training)
                
                batch_positions = train_positions[start_idx:end_idx]
                batch_evaluations = train_evaluations[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.engine.model(batch_positions).squeeze()
                loss = criterion(predictions, batch_evaluations.squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.engine.model.eval()
            with torch.no_grad():
                val_predictions = self.engine.model(val_positions).squeeze()
                val_loss = criterion(val_predictions, val_evaluations.squeeze())
            
            # Calculate statistics
            avg_train_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start_time
            
            # Store statistics
            self.training_stats['training_losses'].append(avg_train_loss)
            self.training_stats['validation_losses'].append(val_loss.item())
            self.training_stats['epochs_completed'] += 1
            
            # Print progress
            print(f"📊 Epoch {epoch + 1}/{epochs}")
            print(f"   🎯 Training Loss: {avg_train_loss:.6f}")
            print(f"   🧪 Validation Loss: {val_loss.item():.6f}")
            print(f"   ⏱️  Time: {epoch_time:.2f}s")
            print(f"   📈 Progress: {(epoch + 1) * 100 / epochs:.1f}%")
            print("-" * 30)
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_train_loss, val_loss.item())
        
        # Final save
        total_time = time.time() - start_time
        final_model_path = self._save_final_model()
        
        # Training summary
        print("🎉 PGN Dataset Training Completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Final training loss: {self.training_stats['training_losses'][-1]:.6f}")
        print(f"🧪 Final validation loss: {self.training_stats['validation_losses'][-1]:.6f}")
        print(f"💾 Model saved: {final_model_path}")
        
        return {
            'final_model_path': final_model_path,
            'training_stats': self.training_stats,
            'total_time': total_time,
            'positions_processed': num_positions,
            'games_processed': len(games_data)
        }
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint during training
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
        """
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/pgn_training_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.engine.model.state_dict(),
            'optimizer_state_dict': self.engine.optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': val_loss,
            'training_stats': self.training_stats
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self) -> str:
        """Save the final trained model
        
        Returns:
            Path to saved model
        """
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Find next version number
        existing_models = [f for f in os.listdir(models_dir) 
                          if f.startswith("chess_neural_pgn_v") and f.endswith("_final.pth")]
        
        if existing_models:
            versions = []
            for model in existing_models:
                try:
                    version = int(model.split("_v")[1].split("_")[0])
                    versions.append(version)
                except:
                    continue
            
            next_version = max(versions) + 1 if versions else 1
        else:
            next_version = 1
        
        model_name = f"chess_neural_pgn_v{next_version}_final.pth"
        model_path = f"{models_dir}/{model_name}"
        
        torch.save(self.engine.model.state_dict(), model_path)
        print(f"💾 Final model saved: {model_path}")
        
        return model_path
    
    def evaluate_model(self, pgn_path: str, max_games: int = 100) -> Dict[str, Any]:
        """Evaluate the trained model on a PGN dataset
        
        Args:
            pgn_path: Path to PGN file for evaluation
            max_games: Maximum games to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        print(f"🧪 Evaluating model on PGN dataset: {pgn_path}")
        
        # Load evaluation data
        games_data = self.load_pgn_file(pgn_path, max_games)
        if not games_data:
            raise ValueError("No valid games found for evaluation")
        
        # Prepare evaluation data
        positions_tensor, evaluations_tensor = self.prepare_training_data(
            games_data, max_positions_per_game=100
        )
        
        # Evaluate model
        self.engine.model.eval()
        with torch.no_grad():
            predictions = self.engine.model(positions_tensor).squeeze()
            
            # Calculate metrics
            mse_loss = nn.MSELoss()(predictions, evaluations_tensor.squeeze())
            mae_loss = nn.L1Loss()(predictions, evaluations_tensor.squeeze())
            
            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([predictions, evaluations_tensor.squeeze()]))[0, 1]
            
            # Calculate accuracy within tolerance
            tolerance = 0.1
            within_tolerance = torch.abs(predictions - evaluations_tensor.squeeze()) < tolerance
            accuracy = torch.mean(within_tolerance.float()).item()
        
        results = {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'correlation': correlation.item(),
            'accuracy_within_0.1': accuracy,
            'total_positions': len(positions_tensor),
            'total_games': len(games_data)
        }
        
        print("📊 Evaluation Results:")
        print(f"   🎯 MSE Loss: {results['mse_loss']:.6f}")
        print(f"   📏 MAE Loss: {results['mae_loss']:.6f}")
        print(f"   🔗 Correlation: {results['correlation']:.4f}")
        print(f"   ✅ Accuracy (within 0.1): {results['accuracy_within_0.1']:.2%}")
        print(f"   📊 Positions evaluated: {results['total_positions']}")
        print(f"   🎮 Games evaluated: {results['total_games']}")
        
        return results


def main():
    """Main function for PGN dataset training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train neural chess engine on PGN dataset")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--model_path", type=str, help="Path to existing model to continue from")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_games", type=int, help="Maximum games to load")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PGNDatasetTrainer(args.model_path)
    
    # Train on dataset
    try:
        results = trainer.train_on_dataset(
            pgn_path=args.pgn_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_games=args.max_games
        )
        
        # Evaluate if requested
        if args.evaluate:
            print("\n" + "="*50)
            trainer.evaluate_model(args.pgn_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"💾 Model saved to: {results['final_model_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
