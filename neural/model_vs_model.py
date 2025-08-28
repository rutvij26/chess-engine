#!/usr/bin/env python3
"""
Model vs Model Self-Play Battle
Allows two trained neural chess models to play against each other
"""

import chess
import torch
import numpy as np
import os
import time
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

from .neural_chess_engine import NeuralChessEngine


class ModelVsModelBattle:
    """Battle system for two neural chess models"""
    
    def __init__(self):
        """Initialize the battle system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Battle statistics
        self.battle_stats = {
            'total_games': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'average_game_length': 0,
            'total_moves': 0
        }
    
    def list_available_models(self) -> list:
        """List all available trained models
        
        Returns:
            List of model filenames
        """
        models_dir = "models"
        if not os.path.exists(models_dir):
            return []
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.pth') and 'final' in file:
                models.append(file)
        
        return sorted(models)
    
    def load_model(self, model_filename: str) -> NeuralChessEngine:
        """Load a trained model
        
        Args:
            model_filename: Name of the model file
            
        Returns:
            Loaded neural chess engine
        """
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📚 Loading model: {model_filename}")
        engine = NeuralChessEngine(model_path)
        return engine
    
    def play_single_game(self, white_model: NeuralChessEngine, black_model: NeuralChessEngine, 
                         max_moves: int = 200, show_moves: bool = True) -> Dict[str, Any]:
        """Play a single game between two models
        
        Args:
            white_model: Model playing as White
            black_model: Model playing as Black
            max_moves: Maximum moves before declaring draw
            show_moves: Whether to display moves in real-time
            
        Returns:
            Game result dictionary
        """
        board = chess.Board()
        move_history = []
        game_start_time = time.time()
        
        if show_moves:
            print(f"\n🎮 Starting game: {os.path.basename(white_model.model_path or 'White')} vs {os.path.basename(black_model.model_path or 'Black')}")
            print("=" * 60)
            print(f"🔍 Initial board state:")
            print(f"   FEN: {board.fen()}")
            print(f"   Turn: {'White' if board.turn else 'Black'}")
            print(f"   Legal moves: {[move.uci() for move in list(board.legal_moves)[:5]]}...")
        
        # Game loop
        for move_num in range(max_moves):
            # Check game state
            if board.is_checkmate():
                winner = "Black" if board.turn else "White"
                result = "1-0" if winner == "White" else "0-1"
                if show_moves:
                    print(f"🏆 Checkmate! {winner} wins!")
                break
            elif board.is_stalemate():
                result = "1/2-1/2"
                if show_moves:
                    print("🤝 Stalemate - Draw!")
                break
            elif board.is_insufficient_material():
                result = "1/2-1/2"
                if show_moves:
                    print("🤝 Insufficient material - Draw!")
                break
            elif board.is_repetition(3):
                result = "1/2-1/2"
                if show_moves:
                    print("🤝 Threefold repetition - Draw!")
                break
            elif board.is_fifty_moves():
                result = "1/2-1/2"
                if show_moves:
                    print("🤝 Fifty moves without capture - Draw!")
                break
            
            # Select current player's model
            current_model = white_model if board.turn == chess.WHITE else black_model
            
            # Get best move from current model for the current board position
            try:
                if show_moves and move_num < 5:  # Debug first few moves
                    print(f"🔍 Current position (move {move_num + 1}):")
                    print(f"   FEN: {board.fen()}")
                    print(f"   Turn: {'White' if board.turn else 'Black'}")
                    print(f"   Legal moves: {[move.uci() for move in list(board.legal_moves)[:5]]}...")
                
                # Validate board state before getting move
                if not board.is_valid():
                    print(f"⚠️  Invalid board state detected!")
                    print(f"   FEN: {board.fen()}")
                    break
                
                # Debug: show board state before getting move
                if show_moves and move_num < 5:
                    print(f"   🔍 Board validation: {board.is_valid()}")
                    print(f"   📍 Board hash: {board._transposition_key()}")
                
                # Create a copy of the board for the model to work with
                board_copy = board.copy()
                
                # Debug: validate the copy
                if show_moves and move_num < 5:
                    print(f"   🔍 Board copy validation: {board_copy.is_valid()}")
                    print(f"   📍 Board copy hash: {board_copy._transposition_key()}")
                
                best_move = current_model.get_best_move_for_position(board_copy, depth=3, time_limit=2.0, verbose=False)
                
                # Debug: validate the original board after the model call
                if show_moves and move_num < 5:
                    print(f"   🔍 Original board validation after model call: {board.is_valid()}")
                    print(f"   📍 Original board hash after model call: {board._transposition_key()}")
                
                # Additional validation: check if the board state is still valid
                if not board.is_valid():
                    print(f"⚠️  Board state became invalid after model call!")
                    print(f"   FEN: {board.fen()}")
                    print(f"   Hash: {board._transposition_key()}")
                    break
                
                if best_move is None:
                    # No legal moves found
                    result = "1-0" if board.turn else "0-1"
                    if show_moves:
                        print(f"🏆 No legal moves! {'Black' if board.turn else 'White'} wins!")
                    break
                
                # Validate that the move is legal
                if best_move not in board.legal_moves:
                    print(f"⚠️  Model returned illegal move: {best_move.uci()}")
                    print(f"   Current board FEN: {board.fen()}")
                    print(f"   Legal moves: {[move.uci() for move in list(board.legal_moves)]}")
                    print(f"   Board validation: {board.is_valid()}")
                    # Fall back to random legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        best_move = np.random.choice(legal_moves)
                        print(f"🔄 Using random legal move: {best_move.uci()}")
                        # Validate the random move is legal
                        if best_move not in board.legal_moves:
                            print(f"⚠️  Random move is also illegal! This suggests board corruption.")
                            print(f"   Board FEN: {board.fen()}")
                            print(f"   Board validation: {board.is_valid()}")
                            # Try to find any legal move
                            all_moves = [chess.Move.from_uci(f"{sq1}{sq2}") for sq1 in range(64) for sq2 in range(64)]
                            legal_moves = [move for move in all_moves if move in board.legal_moves]
                            if legal_moves:
                                best_move = legal_moves[0]
                                print(f"🔄 Found legal move: {best_move.uci()}")
                            else:
                                print(f"❌ No legal moves found at all!")
                                result = "1-0" if board.turn else "0-1"
                                if show_moves:
                                    print(f"🏆 No legal moves! {'Black' if board.turn else 'White'} wins!")
                                break
                    else:
                        # No legal moves - game over
                        result = "1-0" if board.turn else "0-1"
                        if show_moves:
                            print(f"🏆 No legal moves! {'Black' if board.turn else 'White'} wins!")
                        break
                
                # Make the move
                board.push(best_move)
                move_history.append(best_move.uci())
                
                # Validate board state after move
                if not board.is_valid():
                    print(f"⚠️  Board state became invalid after move {best_move.uci()}!")
                    print(f"   FEN: {board.fen()}")
                    break
                
                # Display move
                if show_moves:
                    player = "White" if board.turn == chess.BLACK else "Black"
                    try:
                        move_san = board.san(best_move)
                        print(f"{move_num + 1:2d}. {move_san}")
                    except Exception as e:
                        print(f"⚠️  Error getting SAN for move {best_move.uci()}: {e}")
                        print(f"   Board FEN: {board.fen()}")
                        print(f"   Board validation: {board.is_valid()}")
                        # Try to get the move in UCI format instead
                        print(f"{move_num + 1:2d}. {best_move.uci()}")
                    
                    # Debug: show board state after move
                    if move_num < 5:  # Debug first few moves
                        print(f"   📍 Position after move: {board.fen()}")
                        print(f"   🔄 Next turn: {'White' if board.turn else 'Black'}")
                        print(f"   ⚖️  Legal moves: {[move.uci() for move in list(board.legal_moves)[:5]]}...")
                        print(f"   🔍 Board validation: {board.is_valid()}")
                        print(f"   📍 Board hash: {board._transposition_key()}")
                
                # Check for immediate checkmate after move
                if board.is_checkmate():
                    winner = "Black" if board.turn else "White"
                    result = "1-0" if winner == "White" else "0-1"
                    if show_moves:
                        print(f"🏆 Checkmate! {winner} wins!")
                    break
                
            except Exception as e:
                print(f"⚠️  Error getting move from model: {e}")
                # Default to random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    random_move = np.random.choice(legal_moves)
                    print(f"🔄 Using random legal move: {random_move.uci()}")
                    
                    # Validate board state before random move
                    if not board.is_valid():
                        print(f"⚠️  Board state invalid before random move!")
                        print(f"   FEN: {board.fen()}")
                        break
                    
                    board.push(random_move)
                    move_history.append(random_move.uci())
                    
                    # Validate board state after random move
                    if not board.is_valid():
                        print(f"⚠️  Board state became invalid after random move {random_move.uci()}!")
                        print(f"   FEN: {board.fen()}")
                        break
                    
                    if show_moves:
                        try:
                            move_san = board.san(random_move)
                            print(f"{move_num + 1:2d}. {move_san} (random)")
                        except Exception as e:
                            print(f"⚠️  Error getting SAN for random move {random_move.uci()}: {e}")
                            print(f"   Board FEN: {board.fen()}")
                            print(f"   Board validation: {board.is_valid()}")
                            # Try to get the move in UCI format instead
                            print(f"{move_num + 1:2d}. {random_move.uci()} (random)")
                else:
                    result = "1-0" if board.turn else "0-1"
                    break
        else:
            # Max moves reached
            result = "1/2-1/2"
            if show_moves:
                print("🤝 Maximum moves reached - Draw!")
        
        # Calculate game statistics
        game_time = time.time() - game_start_time
        game_length = len(move_history)
        
        # Update battle statistics
        self.battle_stats['total_games'] += 1
        self.battle_stats['total_moves'] += game_length
        
        if result == "1-0":
            self.battle_stats['white_wins'] += 1
        elif result == "0-1":
            self.battle_stats['black_wins'] += 1
        else:
            self.battle_stats['draws'] += 1
        
        # Update average game length
        total_games = self.battle_stats['total_games']
        self.battle_stats['average_game_length'] = self.battle_stats['total_moves'] / total_games
        
        game_result = {
            'result': result,
            'moves': move_history,
            'game_length': game_length,
            'game_time': game_time,
            'final_fen': board.fen(),
            'move_history_san': move_history  # Use UCI moves instead of SAN to avoid board state issues
        }
        
        if show_moves:
            print(f"📊 Game completed in {game_time:.1f}s")
            print(f"📈 Total moves: {game_length}")
            print(f"🏁 Result: {result}")
            print("-" * 60)
        
        return game_result
    
    def battle_models(self, white_model_path: str, black_model_path: str, 
                     num_games: int = 10, max_moves: int = 200, 
                     show_moves: bool = True, save_pgn: bool = True) -> Dict[str, Any]:
        """Battle two models over multiple games
        
        Args:
            white_model_path: Path to White's model
            black_model_path: Path to Black's model
            num_games: Number of games to play
            max_moves: Maximum moves per game
            show_moves: Whether to show moves in real-time
            save_pgn: Whether to save games to PGN file
            
        Returns:
            Battle results dictionary
        """
        print(f"⚔️  Starting Model Battle!")
        print(f"🤍 White: {os.path.basename(white_model_path)}")
        print(f"🖤 Black: {os.path.basename(black_model_path)}")
        print(f"🎮 Games: {num_games}")
        print(f"📏 Max moves per game: {max_moves}")
        print("=" * 60)
        
        # Load models
        try:
            white_model = self.load_model(white_model_path)
            black_model = self.load_model(black_model_path)
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return {}
        
        # Reset battle statistics
        self.battle_stats = {
            'total_games': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'average_game_length': 0,
            'total_moves': 0
        }
        
        # Play games
        game_results = []
        start_time = time.time()
        
        for game_num in range(num_games):
            if show_moves:
                print(f"\n🎮 Game {game_num + 1}/{num_games}")
            else:
                print(f"🎮 Playing game {game_num + 1}/{num_games}...", end="")
            
            game_result = self.play_single_game(
                white_model, black_model, max_moves, show_moves
            )
            game_results.append(game_result)
            
            if not show_moves:
                print(f" ✅ ({game_result['result']})")
        
        # Battle summary
        total_time = time.time() - start_time
        battle_summary = self._generate_battle_summary(game_results, total_time)
        
        # Save PGN if requested
        if save_pgn:
            pgn_path = self._save_battle_pgn(game_results, white_model_path, black_model_path)
            battle_summary['pgn_file'] = pgn_path
        
        return battle_summary
    
    def _generate_battle_summary(self, game_results: list, total_time: float) -> Dict[str, Any]:
        """Generate summary of the battle
        
        Args:
            game_results: List of game results
            total_time: Total time taken for all games
            
        Returns:
            Battle summary dictionary
        """
        print("\n" + "=" * 60)
        print("🏆 BATTLE SUMMARY")
        print("=" * 60)
        
        # Calculate statistics
        white_wins = sum(1 for r in game_results if r['result'] == "1-0")
        black_wins = sum(1 for r in game_results if r['result'] == "0-1")
        draws = sum(1 for r in game_results if r['result'] == "1/2-1/2")
        
        total_games = len(game_results)
        total_moves = sum(r['game_length'] for r in game_results)
        avg_game_length = total_moves / total_games if total_games > 0 else 0
        
        # Display results
        print(f"📊 Total Games: {total_games}")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"📈 Average Game Length: {avg_game_length:.1f} moves")
        print(f"🎯 Total Moves: {total_moves}")
        print()
        print(f"🏆 Results:")
        print(f"   🤍 White Wins: {white_wins} ({white_wins/total_games*100:.1f}%)")
        print(f"   🖤 Black Wins: {black_wins} ({black_wins/total_games*100:.1f}%)")
        print(f"   🤝 Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        # Determine winner
        if white_wins > black_wins:
            winner = "White"
            margin = white_wins - black_wins
        elif black_wins > white_wins:
            winner = "Black"
            margin = black_wins - white_wins
        else:
            winner = "Tie"
            margin = 0
        
        print(f"\n🏅 Winner: {winner}")
        if winner != "Tie":
            print(f"📊 Margin: {margin} games")
        
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   🎯 White Win Rate: {white_wins/total_games*100:.1f}%")
        print(f"   🎯 Black Win Rate: {black_wins/total_games*100:.1f}%")
        print(f"   🤝 Draw Rate: {draws/total_games*100:.1f}%")
        
        return {
            'total_games': total_games,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'winner': winner,
            'margin': margin,
            'total_time': total_time,
            'total_moves': total_moves,
            'average_game_length': avg_game_length,
            'game_results': game_results
        }
    
    def _save_battle_pgn(self, game_results: list, white_model: str, black_model: str) -> str:
        """Save battle games to PGN file
        
        Args:
            game_results: List of game results
            white_model: White model filename
            black_model: Black model filename
            
        Returns:
            Path to saved PGN file
        """
        # Create battles directory
        battles_dir = "battles"
        os.makedirs(battles_dir, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        white_name = os.path.splitext(os.path.basename(white_model))[0]
        black_name = os.path.splitext(os.path.basename(black_model))[0]
        pgn_filename = f"battle_{white_name}_vs_{black_name}_{timestamp}.pgn"
        pgn_path = os.path.join(battles_dir, pgn_filename)
        
        # Write PGN file
        with open(pgn_path, 'w', encoding='utf-8') as pgn_file:
            for i, game_result in enumerate(game_results):
                # Game header
                pgn_file.write(f'[Event "Model Battle"]\n')
                pgn_file.write(f'[Site "Neural Chess Engine"]\n')
                pgn_file.write(f'[Date "{time.strftime("%Y.%m.%d")}"]\n')
                pgn_file.write(f'[Round "{i+1}"]\n')
                pgn_file.write(f'[White "{white_name}"]\n')
                pgn_file.write(f'[Black "{black_name}"]\n')
                pgn_file.write(f'[Result "{game_result["result"]}"]\n')
                pgn_file.write(f'[GameLength "{game_result["game_length"]}"]\n')
                pgn_file.write(f'[GameTime "{game_result["game_time"]:.1f}s"]\n')
                pgn_file.write('\n')
                
                # Game moves - convert UCI to SAN for PGN
                moves = game_result['move_history_san']
                if moves:
                    # Create a temporary board to convert UCI to SAN
                    temp_board = chess.Board()
                    san_moves = []
                    
                    for uci_move in moves:
                        try:
                            move = chess.Move.from_uci(uci_move)
                            san_move = temp_board.san(move)
                            san_moves.append(san_move)
                            temp_board.push(move)
                        except:
                            # Fallback to UCI if conversion fails
                            san_moves.append(uci_move)
                    
                    # Write moves in PGN format
                    for j, move in enumerate(san_moves):
                        if j % 2 == 0:
                            pgn_file.write(f'{(j//2)+1}. {move}')
                        else:
                            pgn_file.write(f' {move}\n')
                    
                    # Add result if game didn't end with a move
                    if len(san_moves) % 2 == 1:
                        pgn_file.write(f' {game_result["result"]}\n')
                    else:
                        pgn_file.write(f'\n{game_result["result"]}\n')
                else:
                    # No moves - just write result
                    pgn_file.write(f'{game_result["result"]}\n')
                
                pgn_file.write('\n')
        
        print(f"💾 Battle PGN saved: {pgn_path}")
        return pgn_path
    
    def quick_battle(self, white_model: str, black_model: str, 
                    num_games: int = 5, show_moves: bool = False) -> Dict[str, Any]:
        """Quick battle with minimal output
        
        Args:
            white_model: White model filename
            black_model: Black model filename
            num_games: Number of games to play
            show_moves: Whether to show individual moves
            
        Returns:
            Battle results
        """
        print(f"⚡ Quick Battle: {white_model} vs {black_model}")
        print(f"🎮 Playing {num_games} games...")
        
        return self.battle_models(
            white_model, black_model, num_games, 
            max_moves=150, show_moves=show_moves, save_pgn=True
        )
    
    def tournament_mode(self, models: list, games_per_matchup: int = 3, 
                       show_moves: bool = False) -> Dict[str, Any]:
        """Tournament mode - all models play against each other
        
        Args:
            models: List of model filenames
            games_per_matchup: Games per model matchup
            show_moves: Whether to show moves
            
        Returns:
            Tournament results
        """
        print(f"🏆 TOURNAMENT MODE")
        print(f"🎯 Models: {len(models)}")
        print(f"🎮 Games per matchup: {games_per_matchup}")
        print(f"📊 Total matchups: {len(models) * (len(models) - 1) // 2}")
        print("=" * 60)
        
        tournament_results = {}
        total_games = 0
        
        # Play each model against every other model
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                print(f"\n⚔️  Matchup {total_games//games_per_matchup + 1}: {model1} vs {model2}")
                
                # Play games
                battle_result = self.battle_models(
                    model1, model2, games_per_matchup, 
                    max_moves=150, show_moves=show_moves, save_pgn=True
                )
                
                # Store results
                matchup_key = f"{model1}_vs_{model2}"
                tournament_results[matchup_key] = battle_result
                total_games += games_per_matchup
        
        # Generate tournament summary
        tournament_summary = self._generate_tournament_summary(tournament_results)
        
        print(f"\n🎉 Tournament completed! {total_games} total games played.")
        return tournament_summary
    
    def _generate_tournament_summary(self, tournament_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tournament summary
        
        Args:
            tournament_results: Results from all matchups
            
        Returns:
            Tournament summary
        """
        print("\n" + "=" * 60)
        print("🏆 TOURNAMENT SUMMARY")
        print("=" * 60)
        
        # Calculate overall statistics
        total_games = 0
        model_wins = {}
        model_losses = {}
        model_draws = {}
        
        for matchup, results in tournament_results.items():
            total_games += results['total_games']
            
            # Parse model names from matchup
            models = matchup.split('_vs_')
            if len(models) == 2:
                white_model, black_model = models
                
                # Count wins/losses/draws for each model
                for model in [white_model, black_model]:
                    if model not in model_wins:
                        model_wins[model] = 0
                        model_losses[model] = 0
                        model_draws[model] = 0
                
                model_wins[white_model] += results['white_wins']
                model_wins[black_model] += results['black_wins']
                model_draws[white_model] += results['draws']
                model_draws[black_model] += results['draws']
        
        # Display model rankings
        print(f"📊 Total Games: {total_games}")
        print(f"\n🏅 Model Rankings:")
        
        rankings = []
        for model in model_wins:
            wins = model_wins[model]
            losses = model_losses[model]
            draws = model_draws[model]
            total = wins + losses + draws
            win_rate = (wins + draws * 0.5) / total if total > 0 else 0
            
            rankings.append((model, wins, losses, draws, win_rate))
        
        # Sort by win rate
        rankings.sort(key=lambda x: x[4], reverse=True)
        
        for i, (model, wins, losses, draws, win_rate) in enumerate(rankings, 1):
            print(f"   {i}. {model}: {wins}W {losses}L {draws}D ({win_rate*100:.1f}%)")
        
        return {
            'total_games': total_games,
            'model_rankings': rankings,
            'matchup_results': tournament_results
        }


def main():
    """Main function for model vs model battles"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Battle two neural chess models")
    parser.add_argument("--white", type=str, required=True, help="White model filename")
    parser.add_argument("--black", type=str, required=True, help="Black model filename")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--max_moves", type=int, default=200, help="Maximum moves per game")
    parser.add_argument("--show_moves", action="store_true", help="Show moves in real-time")
    parser.add_argument("--quick", action="store_true", help="Quick battle mode")
    parser.add_argument("--tournament", action="store_true", help="Tournament mode (all vs all)")
    
    args = parser.parse_args()
    
    # Create battle system
    battle_system = ModelVsModelBattle()
    
    if args.tournament:
        # Tournament mode
        models = [args.white, args.black]
        results = battle_system.tournament_mode(models, args.games, not args.show_moves)
    elif args.quick:
        # Quick battle
        results = battle_system.quick_battle(args.white, args.black, args.games, args.show_moves)
    else:
        # Full battle
        results = battle_system.battle_models(
            args.white, args.black, args.games, args.max_moves, args.show_moves
        )
    
    if results:
        print(f"\n🎉 Battle completed successfully!")
        print(f"📊 Results saved to battles/ directory")


if __name__ == "__main__":
    main()
