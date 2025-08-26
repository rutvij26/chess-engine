#!/usr/bin/env python3
"""
Grandmaster+ Chess Training Script
Advanced training to reach superhuman chess level
"""

import torch
import torch.nn as nn
import torch.optim as optim
import chess
from .neural_chess_engine import NeuralChessEngine
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import random

class GrandmasterTrainer:
    """Advanced trainer for reaching grandmaster+ level"""
    
    def __init__(self, model_path: str = None):
        self.engine = NeuralChessEngine(model_path=model_path, visual_mode=False)
        self.training_history = []
        self.performance_metrics = []
        
        # Advanced training parameters
        self.curriculum_stages = [
            {"games": 1000, "max_moves": 20, "exploration": 0.3, "description": "Opening Mastery"},
            {"games": 2000, "max_moves": 40, "exploration": 0.2, "description": "Middlegame Strategy"},
            {"games": 3000, "max_moves": 60, "exploration": 0.15, "description": "Complex Tactics"},
            {"games": 5000, "max_moves": 80, "exploration": 0.1, "description": "Endgame Excellence"},
            {"games": 10000, "max_moves": 100, "exploration": 0.05, "description": "Full Game Mastery"},
            {"games": 20000, "max_moves": 150, "exploration": 0.02, "description": "Superhuman Play"},
            {"games": 50000, "max_moves": 200, "exploration": 0.01, "description": "Grandmaster+ Level"}
        ]
        
        # Performance tracking
        self.elo_estimate = 1200  # Starting rating
        self.games_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        
    def estimate_elo_improvement(self, win_rate: float, games_played: int) -> float:
        """Estimate ELO rating based on performance"""
        if games_played < 100:
            return 1200  # Not enough data
        
        # Simplified ELO calculation
        expected_score = 0.5
        actual_score = win_rate
        k_factor = 32 if games_played < 1000 else 16
        
        elo_change = k_factor * (actual_score - expected_score)
        return max(1200, min(3000, self.elo_estimate + elo_change))
    
    def advanced_self_play_game(self, max_moves: int, exploration_rate: float) -> dict:
        """Advanced self-play with sophisticated move selection"""
        self.engine.reset_board()
        game_data = []
        move_history = []
        
        # Game state tracking
        game_phases = {
            "opening": 0,
            "middlegame": 0,
            "endgame": 0
        }
        
        for move_num in range(max_moves):
            if self.engine.board.is_game_over():
                break
            
            # Determine game phase
            if move_num < 10:
                game_phases["opening"] += 1
            elif move_num < 30:
                game_phases["middlegame"] += 1
            else:
                game_phases["endgame"] += 1
            
            # Advanced move selection
            if random.random() < exploration_rate:
                # Exploration: try different strategies
                move = self._exploratory_move()
            else:
                # Exploitation: use best known move
                move = self._best_move_with_depth_scaling(move_num)
            
            if move:
                move_history.append(move.uci())
                self.engine.board.push(move)
                
                # Collect training data
                position_tensor = self.engine.board_to_tensor(self.engine.board)
                evaluation = self.engine.evaluate_position_neural(self.engine.board)
                
                game_data.append({
                    'position': position_tensor,
                    'evaluation': evaluation,
                    'move': move.uci(),
                    'phase': self._get_game_phase(move_num),
                    'move_number': move_num + 1
                })
        
        # Game result analysis
        result = self._analyze_game_result()
        
        return {
            'game_data': game_data,
            'move_history': move_history,
            'result': result,
            'phases': game_phases,
            'length': len(move_history)
        }
    
    def _exploratory_move(self) -> chess.Move:
        """Make an exploratory move to discover new strategies"""
        legal_moves = list(self.engine.board.legal_moves)
        if not legal_moves:
            return None
        
        # Different exploration strategies
        strategy = random.choice(['random', 'tactical', 'positional', 'endgame'])
        
        if strategy == 'random':
            return random.choice(legal_moves)
        elif strategy == 'tactical':
            # Prefer captures and checks
            tactical_moves = [m for m in legal_moves if self.engine.board.is_capture(m) or self.engine.board.gives_check(m)]
            return random.choice(tactical_moves) if tactical_moves else random.choice(legal_moves)
        elif strategy == 'positional':
            # Prefer center control and development
            positional_moves = [m for m in legal_moves if self._is_positional_move(m)]
            return random.choice(positional_moves) if positional_moves else random.choice(legal_moves)
        else:  # endgame
            # Prefer king activity and pawn advancement
            endgame_moves = [m for m in legal_moves if self._is_endgame_move(m)]
            return random.choice(endgame_moves) if endgame_moves else random.choice(legal_moves)
    
    def _best_move_with_depth_scaling(self, move_number: int) -> chess.Move:
        """Get best move with depth scaling based on game phase"""
        if move_number < 10:  # Opening
            depth = 4
        elif move_number < 30:  # Middlegame
            depth = 5
        else:  # Endgame
            depth = 6
        
        best_move_uci = self.engine.get_best_move(depth, time_limit=2.0)
        if best_move_uci:
            return chess.Move.from_uci(best_move_uci)
        return None
    
    def _is_positional_move(self, move: chess.Move) -> bool:
        """Check if move is positional (center control, development)"""
        # Center squares
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        return move.to_square in center_squares
    
    def _is_endgame_move(self, move: chess.Move) -> bool:
        """Check if move is good for endgame"""
        # King activity, pawn advancement
        piece = self.engine.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            return True
        if piece and piece.piece_type == chess.PAWN:
            # Prefer pawn advancement
            rank = chess.square_rank(move.to_square)
            return rank > 3  # Advancing pawns
        return False
    
    def _get_game_phase(self, move_number: int) -> str:
        """Determine current game phase"""
        if move_number < 10:
            return "opening"
        elif move_number < 30:
            return "middlegame"
        else:
            return "endgame"
    
    def _analyze_game_result(self) -> dict:
        """Analyze the result of a completed game"""
        board = self.engine.board
        
        if board.is_checkmate():
            winner = "black" if board.turn else "white"
            return {"type": "checkmate", "winner": winner, "score": 1.0 if winner == "white" else -1.0}
        elif board.is_stalemate():
            return {"type": "stalemate", "winner": "draw", "score": 0.0}
        elif board.is_insufficient_material():
            return {"type": "insufficient_material", "winner": "draw", "score": 0.0}
        else:
            # Evaluate final position
            final_eval = self.engine.evaluate_position_neural(board)
            return {"type": "time_limit", "winner": "white" if final_eval > 0 else "black", "score": final_eval}
    
    def curriculum_training(self) -> None:
        """Progressive training through different skill levels"""
        print("🏆 GRANDMASTER+ TRAINING INITIATED")
        print("=" * 60)
        print("This will train the engine through 7 stages to reach superhuman level")
        print("Total estimated time: 2-6 months depending on hardware")
        print("=" * 60)
        
        total_games = sum(stage["games"] for stage in self.curriculum_stages)
        current_game = 0
        
        for stage_num, stage in enumerate(self.curriculum_stages):
            print(f"\n🎯 STAGE {stage_num + 1}: {stage['description']}")
            print(f"Games: {stage['games']}, Max Moves: {stage['max_moves']}, Exploration: {stage['exploration']}")
            print("-" * 50)
            
            stage_start_time = time.time()
            stage_wins = 0
            stage_draws = 0
            stage_losses = 0
            
            for game_num in range(stage['games']):
                current_game += 1
                
                # Progress update
                if game_num % 100 == 0:
                    progress = (current_game / total_games) * 100
                    elapsed = time.time() - stage_start_time
                    eta = (elapsed / (game_num + 1)) * (stage['games'] - game_num)
                    print(f"Progress: {progress:.1f}% | Game {game_num + 1}/{stage['games']} | ETA: {eta/60:.1f} min")
                
                # Play advanced game
                game_result = self.advanced_self_play_game(
                    max_moves=stage['max_moves'],
                    exploration_rate=stage['exploration']
                )
                
                # Update statistics
                if game_result['result']['winner'] == 'white':
                    stage_wins += 1
                    self.wins += 1
                elif game_result['result']['winner'] == 'draw':
                    stage_draws += 1
                    self.draws += 1
                else:
                    stage_losses += 1
                    self.losses += 1
                
                self.games_played += 1
                
                # Generate and save PGN for this game
                if game_result['move_history']:
                    pgn_game = self.generate_pgn_game(
                        game_result['move_history'],
                        game_result['result']['type'],
                        stage['description']
                    )
                    
                    game_stats = {
                        'result': game_result['result']['type'],
                        'moves_played': game_result['length'],
                        'final_evaluation': game_result['result']['score'],
                        'elo_estimate': self.elo_estimate
                    }
                    
                    self.save_game_to_history(pgn_game, current_game, stage['description'], game_stats)
                
                # Add to training data
                for data_point in game_result['game_data']:
                    self.engine.training_positions.append(data_point['position'])
                    self.engine.training_evaluations.append(data_point['evaluation'])
                
                # Train periodically
                if len(self.engine.training_positions) >= 1000:
                    self.engine.train_model(epochs=3)
                    self.engine.training_positions = []
                    self.engine.training_evaluations = []
                
                # Save model periodically
                if current_game % 1000 == 0:
                    self._save_grandmaster_model(current_game)
                    self._save_training_stats()
            
            # Stage completion
            stage_time = time.time() - stage_start_time
            win_rate = stage_wins / stage['games']
            self.elo_estimate = self.estimate_elo_improvement(win_rate, stage['games'])
            
            print(f"\n✅ STAGE {stage_num + 1} COMPLETED!")
            print(f"Time: {stage_time/3600:.1f} hours")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Estimated ELO: {self.elo_estimate:.0f}")
            
            # Save stage model
            self._save_grandmaster_model(current_game, stage_name=stage['description'])
            
            # Performance analysis
            self._analyze_stage_performance(stage_num, stage, win_rate, stage_time)
        
        print(f"\n🎉 GRANDMASTER+ TRAINING COMPLETED!")
        print(f"Total Games: {self.games_played}")
        print(f"Final ELO Estimate: {self.elo_estimate:.0f}")
        print(f"Training Time: {(time.time() - self.start_time)/3600:.1f} hours")
    
    def generate_pgn_game(self, move_history: list, game_result: str = None, stage_info: str = None) -> str:
        """Generate PGN notation for a completed grandmaster training game"""
        # Create a new game
        game = chess.pgn.Game()
        
        # Set game metadata
        game.headers["Event"] = "Grandmaster+ Chess Training"
        game.headers["Site"] = "AI Training Session"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = f"{self.games_played}"
        game.headers["White"] = "Grandmaster AI"
        game.headers["Black"] = "Grandmaster AI"
        
        if stage_info:
            game.headers["Stage"] = stage_info
        
        # Set result if provided
        if game_result:
            game.headers["Result"] = game_result
        
        # Replay the moves to build the game
        board = chess.Board()
        node = game
        
        for move_uci in move_history:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    node = node.add_variation(move)
                    board.push(move)
                else:
                    print(f"Warning: Illegal move {move_uci} in move history")
                    break
            except ValueError:
                print(f"Warning: Invalid move format {move_uci}")
                break
        
        # Set the final position
        game.end().board = board
        
        return str(game)
    
    def save_game_to_history(self, pgn_game: str, game_number: int, stage_info: str = None, game_stats: dict = None):
        """Save a completed grandmaster game to the game histories file"""
        history_file = "grandmaster_game_histories.pgn"
        
        # Create game header with metadata
        header = f"\n[Event \"Grandmaster Training Game {game_number}\"]\n"
        header += f"[Site \"AI Training Session\"]\n"
        header += f"[Date \"{datetime.now().strftime('%Y.%m.%d')}\"]\n"
        header += f"[Round \"{game_number}\"]\n"
        header += f"[White \"Grandmaster AI\"]\n"
        header += f"[Black \"Grandmaster AI\"]\n"
        
        if stage_info:
            header += f"[Stage \"{stage_info}\"]\n"
        
        if game_stats:
            if 'result' in game_stats:
                header += f"[Result \"{game_stats['result']}\"]\n"
            if 'moves_played' in game_stats:
                header += f"[Moves \"{game_stats['moves_played']}\"]\n"
            if 'final_evaluation' in game_stats:
                header += f"[Evaluation \"{game_stats['final_evaluation']:.3f}\"]\n"
            if 'elo_estimate' in game_stats:
                header += f"[ELO \"{game_stats['elo_estimate']:.0f}\"]\n"
        
        # Append to history file
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(header)
            f.write(pgn_game)
            f.write("\n\n")
        
        print(f"💾 Grandmaster Game {game_number} saved to {history_file}")
    
    def _save_grandmaster_model(self, game_number: int, stage_name: str = "") -> None:
        """Save the model with grandmaster naming convention"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_suffix = f"_{stage_name.replace(' ', '_')}" if stage_name else ""
        filename = f"grandmaster_model_{game_number:06d}{stage_suffix}_{timestamp}.pth"
        
        torch.save(self.engine.model.state_dict(), filename)
        print(f"💾 Model saved: {filename}")
    
    def _save_training_stats(self) -> None:
        """Save training statistics and progress"""
        stats = {
            'games_played': self.games_played,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'elo_estimate': self.elo_estimate,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('grandmaster_training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _analyze_stage_performance(self, stage_num: int, stage: dict, win_rate: float, stage_time: float) -> None:
        """Analyze performance at each training stage"""
        performance = {
            'stage': stage_num + 1,
            'description': stage['description'],
            'games': stage['games'],
            'win_rate': win_rate,
            'time_hours': stage_time / 3600,
            'elo_estimate': self.elo_estimate
        }
        
        self.performance_metrics.append(performance)
        
        # Save performance data
        with open('grandmaster_performance.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)

def main():
    """Main function for grandmaster training"""
    print("🏆 GRANDMASTER+ CHESS TRAINING")
    print("=" * 50)
    print("This will train a neural network to reach superhuman chess level")
    print("WARNING: This will take 2-6 months and significant computational resources!")
    print()
    
    # Check for existing model
    model_files = [f for f in os.listdir('.') if f.startswith('grandmaster_model_')]
    
    if model_files:
        print("📁 Found existing grandmaster models:")
        for model in sorted(model_files)[-3:]:  # Show last 3
            print(f"  • {model}")
        
        use_existing = input("\nUse existing model? (y/n): ").lower().startswith('y')
        if use_existing:
            best_model = sorted(model_files)[-1]
            print(f"🔄 Continuing training from: {best_model}")
            trainer = GrandmasterTrainer(model_path=best_model)
        else:
            trainer = GrandmasterTrainer()
    else:
        print("🆕 Starting fresh grandmaster training")
        trainer = GrandmasterTrainer()
    
    # Confirm training
    print(f"\n📊 Training Plan:")
    total_games = sum(stage["games"] for stage in trainer.curriculum_stages)
    print(f"  • Total Games: {total_games:,}")
    print(f"  • Stages: {len(trainer.curriculum_stages)}")
    print(f"  • Estimated Time: 2-6 months")
    print(f"  • Target ELO: 2800+ (Grandmaster+)")
    
    confirm = input("\n🚀 Start Grandmaster+ Training? (type 'YES' to confirm): ")
    if confirm != 'YES':
        print("❌ Training cancelled")
        return
    
    # Start training
    trainer.start_time = time.time()
    trainer.curriculum_training()

if __name__ == "__main__":
    main()
