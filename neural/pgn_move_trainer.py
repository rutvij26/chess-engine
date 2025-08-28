#!/usr/bin/env python3
"""
PGN Move Trainer for Neural Chess Engine
Extends existing neural network by training on PGN moves instead of self-play
Uses the existing ChessNeuralNetwork architecture
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

from .neural_chess_engine import NeuralChessEngine, ChessDataset, DataLoader


class PGNMoveTrainer:
    """Trainer that extends existing neural network using PGN move data"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the PGN move trainer
        
        Args:
            model_path: Path to existing model to extend training from
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
    
    def load_pgn_moves(self, pgn_path: str, max_games: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load moves from PGN file for training
        
        Args:
            pgn_path: Path to the PGN file
            max_games: Maximum number of games to load (None for all)
            
        Returns:
            List of move data dictionaries
        """
        print(f"📖 Loading PGN moves from: {pgn_path}")
        
        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        
        moves_data = []
        game_count = 0
        
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    if max_games and game_count >= max_games:
                        break
                    
                    # Extract moves and positions from this game
                    game_moves = self._extract_game_moves(game)
                    if game_moves:
                        moves_data.extend(game_moves)
                        game_count += 1
                        
                        if game_count % 100 == 0:
                            print(f"   📊 Processed {game_count} games...")
                    
                except Exception as e:
                    print(f"⚠️  Error reading game {game_count}: {e}")
                    continue
        
        print(f"✅ Successfully extracted {len(moves_data)} moves from {game_count} games")
        return moves_data
    
    def _extract_game_moves(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """Extract training data from a single game's moves
        
        Args:
            game: Chess game object from python-chess
            
        Returns:
            List of move data dictionaries
        """
        try:
            # Get game result
            result = game.headers.get("Result", "*")
            if result == "*":
                return []  # Skip incomplete games
            
            # Convert result to numerical value
            if result == "1-0":
                result_value = 1.0  # White wins
            elif result == "0-1":
                result_value = -1.0  # Black wins
            elif result == "1/2-1/2":
                result_value = 0.0  # Draw
            else:
                return []
            
            moves_data = []
            board = game.board()
            move_history = []  # Track move history to detect repetition
            
            # Process each move in the game
            for move_num, move in enumerate(game.mainline_moves()):
                # Get position before the move
                position_before = board.copy()
                
                # Make the move
                board.push(move)
                position_after = board.copy()
                
                # Detect repetitive moves and punish them
                repetition_penalty = self._detect_repetitive_moves(board, move_history, move_num)
                
                # Calculate target evaluation based on game result and position
                target_eval = self._calculate_target_evaluation(
                    position_after, result_value, move_num
                )
                
                # Apply repetition penalty
                target_eval += repetition_penalty
                
                # Store the move data
                move_data = {
                    'position': position_before,
                    'move': move,
                    'target_eval': target_eval,
                    'move_number': move_num,
                    'game_result': result_value,
                    'white_player': game.headers.get("White", "Unknown"),
                    'black_player': game.headers.get("Black", "Unknown"),
                    'event': game.headers.get("Event", "Unknown")
                }
                
                moves_data.append(move_data)
                
                # Update move history
                move_history.append(move.uci())
            
            return moves_data
            
        except Exception as e:
            print(f"⚠️  Error extracting moves from game: {e}")
            return []
    
    def _detect_repetitive_moves(self, board: chess.Board, move_history: List[str], move_num: int) -> float:
        """Detect and penalize repetitive moves
        
        Args:
            board: Current board position
            move_history: List of previous moves
            move_num: Current move number
            
        Returns:
            Penalty value (negative for repetitive moves)
        """
        if len(move_history) < 4:
            return 0.0
        
        # Check for immediate repetition (same move twice in a row)
        if len(move_history) >= 2 and move_history[-1] == move_history[-2]:
            return -200.0  # Heavy penalty for immediate repetition
        
        # Check for move repetition patterns
        if len(move_history) >= 6:
            # Check for 3-move repetition pattern
            if (move_history[-1] == move_history[-3] == move_history[-5] and
                move_history[-2] == move_history[-4] == move_history[-6]):
                return -300.0  # Very heavy penalty for 3-move repetition
        
        # Check for position repetition (same board state)
        if board.is_repetition(2):  # 2-fold repetition
            return -150.0
        
        # Check for excessive moves without progress
        if move_num > 50:  # After 50 moves, start penalizing long games
            return -50.0
        
        return 0.0  # No penalty
    
    def _detect_tactical_opportunities(self, board: chess.Board) -> float:
        """Detect tactical opportunities and reward them
        
        Args:
            board: Current board position
            
        Returns:
            Bonus value for tactical opportunities
        """
        bonus = 0.0
        
        # Check for checkmate opportunities
        if board.is_checkmate():
            if board.turn:  # Black's turn, White just mated
                bonus += 1000.0  # Maximum reward for checkmate
            else:  # White's turn, Black just mated
                bonus -= 1000.0  # Maximum penalty for being mated
        
        # Check for check opportunities
        elif board.is_check():
            if board.turn:  # Black's turn, White is checking
                bonus += 50.0  # Reward for checking opponent
            else:  # White's turn, Black is checking
                bonus -= 50.0  # Penalty for being checked
        
        # Check for capture opportunities
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Reward for having pieces that can capture
                if piece.color == board.turn:
                    # Check if this piece can capture something valuable
                    for target_square in chess.SQUARES:
                        target_piece = board.piece_at(target_square)
                        if target_piece and target_piece.color != piece.color:
                            # Calculate capture value
                            capture_value = self._get_piece_value(target_piece.piece_type)
                            bonus += capture_value * 0.1  # Small reward for capture potential
        
        # Check for development (pieces in center)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                bonus += 20.0  # Reward for controlling center
        
        return bonus
    
    def _get_piece_value(self, piece_type: chess.PieceType) -> float:
        """Get piece value for tactical evaluation
        
        Args:
            piece_type: Type of chess piece
            
        Returns:
            Piece value
        """
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.5,
            chess.KING: 0.0
        }
        return piece_values.get(piece_type, 0.0)
    
    def _calculate_target_evaluation(self, position: chess.Board, game_result: float, move_number: int) -> float:
        """Calculate target evaluation for a position
        
        Args:
            position: Chess board position
            game_result: Game result (-1, 0, 1)
            move_number: Move number in the game
            
        Returns:
            Target evaluation value
        """
        # Strong game result weighting - winning/losing should be heavily emphasized
        if game_result == 1.0:  # White wins
            base_eval = 800.0  # Strong positive bias for winning
        elif game_result == -1.0:  # Black wins
            base_eval = -800.0  # Strong negative bias for losing
        else:  # Draw
            base_eval = 0.0  # Neutral for draws
        
        # Material evaluation with stronger weighting
        material_score = self._calculate_material_score(position) * 600.0  # Scale material importance
        
        # Position evaluation (considering move number)
        # Early game: strong material focus, late game: strong result focus
        early_game_weight = max(0.3, 1.0 - move_number / 30.0)  # Material weight decreases slower
        late_game_weight = 1.0 - early_game_weight
        
        # Combine evaluations with stronger weighting
        final_eval = (
            base_eval * late_game_weight + 
            material_score * early_game_weight
        )
        
        # Add tactical awareness - punish material loss more aggressively
        if abs(material_score) > 200:  # Significant material advantage/disadvantage
            final_eval += material_score * 0.5  # Additional material weighting
        
        # Detect and reward tactical opportunities
        tactical_bonus = self._detect_tactical_opportunities(position)
        final_eval += tactical_bonus
        
        # Ensure result is within bounds and has strong signal
        final_eval = max(-1000.0, min(1000.0, final_eval))
        
        # Add small random noise to prevent overfitting to exact values
        noise = np.random.uniform(-10, 10)
        final_eval += noise
        
        return final_eval
    
    def _calculate_material_score(self, board: chess.Board) -> float:
        """Calculate material-based position score with enhanced piece values
        
        Args:
            board: Chess board position
            
        Returns:
            Material score between -1 and 1
        """
        # Enhanced piece values that better reflect chess theory
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,  # Slightly higher than bishop
            chess.BISHOP: 3.3,  # Slightly higher than knight
            chess.ROOK: 5.0,
            chess.QUEEN: 9.5,   # Slightly higher than 9
            chess.KING: 0       # King value not included in material calculation
        }
        
        white_material = 0
        black_material = 0
        
        # Count material for each side
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Calculate material difference
        material_diff = white_material - black_material
        
        # Normalize to [-1, 1] range with stronger signal
        total_material = white_material + black_material
        if total_material == 0:
            return 0.0
        
        # Use stronger normalization for better signal
        material_score = material_diff / (total_material * 0.8)  # Stronger signal
        
        # Add bonus for piece development and position
        material_score = max(-1.0, min(1.0, material_score))
        
        return material_score
    
    def prepare_training_data(self, moves_data: List[Dict[str, Any]], 
                            max_positions: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from moves
        
        Args:
            moves_data: List of move data dictionaries
            max_positions: Maximum positions to use (None for all)
            
        Returns:
            Tuple of (positions_tensor, evaluations_tensor)
        """
        print("🔧 Preparing training data from moves...")
        
        # Limit positions if specified
        if max_positions and len(moves_data) > max_positions:
            # Sample positions evenly across the dataset
            indices = np.linspace(0, len(moves_data) - 1, max_positions, dtype=int)
            moves_data = [moves_data[i] for i in indices]
            print(f"   📊 Sampled {max_positions} positions from {len(moves_data)} total moves")
        
        all_positions = []
        all_evaluations = []
        
        for move_data in tqdm(moves_data, desc="Processing moves"):
            try:
                # Convert position to tensor
                position_tensor = self.engine.board_to_tensor(move_data['position'])
                target_eval = torch.tensor([move_data['target_eval']], dtype=torch.float32)
                
                all_positions.append(position_tensor)
                all_evaluations.append(target_eval)
                
            except Exception as e:
                print(f"⚠️  Error processing position: {e}")
                continue
        
        # Stack all tensors
        if not all_positions:
            raise ValueError("No valid positions found in moves data")
        
        positions_tensor = torch.cat(all_positions, dim=0)
        evaluations_tensor = torch.cat(all_evaluations, dim=0)
        
        # Move to device
        positions_tensor = positions_tensor.to(self.device)
        evaluations_tensor = evaluations_tensor.to(self.device)
        
        print(f"✅ Prepared {len(positions_tensor)} positions for training")
        return positions_tensor, evaluations_tensor
    
    def train_on_pgn_moves(self, pgn_path: str, 
                          epochs: int = 10,
                          batch_size: int = 64,
                          learning_rate: float = 0.001,
                          max_games: Optional[int] = None,
                          max_positions: Optional[int] = None,
                          validation_split: float = 0.1,
                          save_interval: int = 5) -> Dict[str, Any]:
        """Train the existing neural network on PGN moves
        
        Args:
            pgn_path: Path to PGN file
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            max_games: Maximum games to load
            max_positions: Maximum positions to use
            validation_split: Fraction of data for validation
            save_interval: Save model every N epochs
            
        Returns:
            Training statistics dictionary
        """
        print(f"🚀 Starting PGN Move Training (Extending Existing Network)")
        print(f"📁 PGN file: {pgn_path}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📚 Learning rate: {learning_rate}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Load PGN moves data
        moves_data = self.load_pgn_moves(pgn_path, max_games)
        if not moves_data:
            raise ValueError("No valid moves found in PGN file")
        
        # Prepare training data
        positions_tensor, evaluations_tensor = self.prepare_training_data(
            moves_data, max_positions
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
        
        # Create dataset and dataloader
        train_dataset = ChessDataset(train_positions, train_evaluations)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer with existing model
        optimizer = optim.Adam(self.engine.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.engine.model.train()
            train_losses = []
            
            for batch_positions, batch_evaluations in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                # Move batch to device
                batch_positions = batch_positions.to(self.device)
                batch_evaluations = batch_evaluations.to(self.device)
                
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
        print("🎉 PGN Move Training Completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Final training loss: {self.training_stats['training_losses'][-1]:.6f}")
        print(f"🧪 Final validation loss: {self.training_stats['validation_losses'][-1]:.6f}")
        print(f"💾 Extended model saved: {final_model_path}")
        
        return {
            'final_model_path': final_model_path,
            'training_stats': self.training_stats,
            'total_time': total_time,
            'positions_processed': num_positions,
            'moves_processed': len(moves_data)
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
        
        checkpoint_path = f"{checkpoint_dir}/pgn_moves_epoch_{epoch}.pth"
        
        torch.save(self.engine.model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self) -> str:
        """Save the final extended model
        
        Returns:
            Path to saved model
        """
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Find next version number for extended model
        existing_models = [f for f in os.listdir(models_dir) 
                          if f.startswith("chess_neural_extended_v") and f.endswith("_final.pth")]
        
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
        
        model_name = f"chess_neural_extended_v{next_version}_final.pth"
        model_path = f"{models_dir}/{model_name}"
        
        torch.save(self.engine.model.state_dict(), model_path)
        print(f"💾 Extended model saved: {model_path}")
        
        return model_path
    
    def evaluate_model(self, pgn_path: str, max_games: int = 100) -> Dict[str, Any]:
        """Evaluate the extended model on PGN moves
        
        Args:
            pgn_path: Path to PGN file for evaluation
            max_games: Maximum games to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        print(f"🧪 Evaluating extended model on PGN moves: {pgn_path}")
        
        # Load evaluation data
        moves_data = self.load_pgn_moves(pgn_path, max_games)
        if not moves_data:
            raise ValueError("No valid moves found for evaluation")
        
        # Prepare evaluation data
        positions_tensor, evaluations_tensor = self.prepare_training_data(
            moves_data, max_positions=1000
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
            tolerance = 100  # Within 100 centipawns
            within_tolerance = torch.abs(predictions - evaluations_tensor.squeeze()) < tolerance
            accuracy = torch.mean(within_tolerance.float()).item()
        
        results = {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'correlation': correlation.item(),
            'accuracy_within_100': accuracy,
            'total_positions': len(positions_tensor),
            'total_moves': len(moves_data)
        }
        
        print("📊 Evaluation Results:")
        print(f"   🎯 MSE Loss: {results['mse_loss']:.2f}")
        print(f"   📏 MAE Loss: {results['mae_loss']:.2f}")
        print(f"   🔗 Correlation: {results['correlation']:.4f}")
        print(f"   ✅ Accuracy (within 100): {results['accuracy_within_100']:.2%}")
        print(f"   📊 Positions evaluated: {results['total_positions']}")
        print(f"   🎮 Moves evaluated: {results['total_moves']}")
        
        return results


def main():
    """Main function for PGN move training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extend neural chess engine with PGN move training")
    parser.add_argument("--pgn_path", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--model_path", type=str, help="Path to existing model to extend from")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_games", type=int, help="Maximum games to load")
    parser.add_argument("--max_positions", type=int, help="Maximum positions to use")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PGNMoveTrainer(args.model_path)
    
    # Train on PGN moves
    try:
        results = trainer.train_on_pgn_moves(
            pgn_path=args.pgn_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_games=args.max_games,
            max_positions=args.max_positions
        )
        
        # Evaluate if requested
        if args.evaluate:
            print("\n" + "="*50)
            trainer.evaluate_model(args.pgn_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"💾 Extended model saved to: {results['final_model_path']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
