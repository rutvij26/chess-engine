import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os

# Import visual board
try:
    from visual.visual_chess_board import VisualChessBoard
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False
    print("Visual board not available - install visual/visual_chess_board.py for better display")

class ChessNeuralNetwork(nn.Module):
    """Neural network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNeuralNetwork, self).__init__()
        
        # Input: 8x8x12 (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Output: position evaluation
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        
        x = self.dropout(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return torch.tanh(x) * 1000  # Output between -1000 and 1000

class ChessDataset(Dataset):
    """Dataset for chess positions and their evaluations"""
    
    def __init__(self, positions, evaluations):
        self.positions = positions
        self.evaluations = evaluations
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

class NeuralChessEngine:
    """Chess engine that learns from self-play using neural networks"""
    
    def __init__(self, model_path: str = None, visual_mode: bool = True):
        self.board = chess.Board()
        self.model = ChessNeuralNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        # Piece values for fallback evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Training data storage
        self.training_positions = []
        self.training_evaluations = []
        
        # Visual board setup
        self.visual_mode = visual_mode and VISUAL_AVAILABLE
        if self.visual_mode:
            self.visual_board = VisualChessBoard()
        else:
            self.visual_board = None
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to neural network input tensor"""
        # Create 8x8x12 tensor (6 piece types x 2 colors)
        tensor = torch.zeros(12, 8, 8)
        
        piece_channels = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # White pieces: channels 0-5, Black pieces: channels 6-11
                if piece.color == chess.WHITE:
                    channel = piece_channels[piece.piece_type]
                else:
                    channel = piece_channels[piece.piece_type] + 6
                
                tensor[channel, rank, file] = 1.0
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def evaluate_position_neural(self, board: chess.Board) -> float:
        """Evaluate position using neural network"""
        try:
            with torch.no_grad():
                tensor = self.board_to_tensor(board)
                evaluation = self.model(tensor)
                return evaluation.item()
        except Exception as e:
            print(f"Neural evaluation failed: {e}, using fallback")
            return self.evaluate_position_fallback(board)
    
    def evaluate_position_fallback(self, board: chess.Board) -> float:
        """Fallback evaluation using traditional methods"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def get_best_move(self, depth: int, time_limit: float = 5.0) -> str:
        """Get best move using minimax with neural evaluation"""
        start_time = time.time()
        best_move = None
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            try:
                score, move = self.search(current_depth)
                if move:
                    best_move = move
                    print(f"Depth {current_depth}: {move.uci()} (score: {score:.1f})")
            except Exception as e:
                print(f"Error at depth {current_depth}: {e}")
                break
        
        return best_move.uci() if best_move else ""
    
    def search(self, depth: int, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[float, Optional[chess.Move]]:
        """Search using minimax with alpha-beta pruning"""
        if depth == 0:
            return self.evaluate_position_neural(self.board), None
        
        moves = list(self.board.legal_moves)
        if not moves:
            if self.board.is_checkmate():
                return -20000 if self.board.turn else 20000, None
            return 0, None
        
        best_move = moves[0]
        best_score = float('-inf') if self.board.turn else float('inf')
        
        for move in moves:
            self.board.push(move)
            score, _ = self.search(depth - 1, alpha, beta)
            self.board.pop()
            
            if self.board.turn:  # White's turn (maximizing)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:  # Black's turn (minimizing)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
            
            if alpha >= beta:
                break
        
        return best_score, best_move
    
    def self_play_game(self, max_moves: int = 100, show_progress: bool = True) -> Dict[str, Any]:
        """Play a game against itself and return training data with game information"""
        self.board = chess.Board()
        game_data = []
        move_history = []
        
        if show_progress and self.visual_mode:
            self.visual_board.display_board(self.board, evaluation=0.0, move_number=1)
            time.sleep(0.5)
        
        for move_num in range(max_moves):
            if self.board.is_game_over():
                break
            
            # Get evaluation before move
            pre_eval = self.evaluate_position_neural(self.board)
            
            # Make a move (either best move or random for exploration)
            if random.random() < 0.8:  # 80% best move, 20% random
                best_move = self.get_best_move(3, 1.0)
                if best_move:
                    move = chess.Move.from_uci(best_move)
                    move_history.append(move.uci())
                    self.board.push(move)
            else:
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    move_history.append(move.uci())
                    self.board.push(move)
            
            # Get evaluation after move
            post_eval = self.evaluate_position_neural(self.board)
            
            # Store position and evaluation
            position_tensor = self.board_to_tensor(self.board)
            game_data.append((position_tensor, post_eval))
            
            # Add to training data
            self.training_positions.append(position_tensor)
            self.training_evaluations.append(post_eval)
            
            # Show visual progress
            if show_progress and self.visual_mode:
                self.visual_board.display_game_progress(
                    self.board, 
                    move_history, 
                    post_eval
                )
                time.sleep(0.2)  # Show each position briefly
        
        # Get final game result
        final_result = self.get_game_result()
        final_evaluation = self.evaluate_position_neural(self.board)
        
        return {
            'game_data': game_data,
            'move_history': move_history,
            'result': final_result,
            'moves_played': len(move_history),
            'final_evaluation': final_evaluation,
            'game_over': self.board.is_game_over()
        }
    
    def train_on_self_play(self, num_games: int = 100, epochs_per_game: int = 5, visual_training: bool = True):
        """Train the neural network on self-play games"""
        print(f"Starting self-play training with {num_games} games...")
        
        for game_num in range(num_games):
            print(f"Playing game {game_num + 1}/{num_games}")
            
            # Show training progress if visual mode
            if visual_training and self.visual_mode:
                self.visual_board.display_training_progress(
                    game_num + 1, 
                    num_games, 
                    0.0,  # Will be updated after training
                    len(self.training_positions)
                )
                time.sleep(1.0)
            
            # Play a game
            game_result = self.self_play_game(show_progress=visual_training)
            
            # Generate and save PGN for this game
            if game_result['move_history']:
                pgn_game = self.generate_pgn_game(
                    game_result['move_history'], 
                    game_result['result']
                )
                self.save_game_to_history(pgn_game, game_num + 1, game_result)
            
            # Train on this game's data
            if len(self.training_positions) > 0:
                self.train_model(epochs_per_game)
            
            # Save model periodically
            if (game_num + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"chess_model_game_{game_num + 1}.pth")
                print(f"Model saved after {game_num + 1} games")
        
        print("Self-play training completed!")
    
    def train_model(self, epochs: int = 1):
        """Train the neural network on collected data"""
        if len(self.training_positions) < 10:
            return
        
        # Convert to tensors
        positions = torch.cat(self.training_positions, dim=0)
        evaluations = torch.tensor(self.training_evaluations, dtype=torch.float32).unsqueeze(1)
        
        # Create dataset and dataloader
        dataset = ChessDataset(positions, evaluations)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_positions, batch_evaluations in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_positions)
                loss = self.criterion(outputs, batch_evaluations)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        
        self.model.eval()
    
    def save_training_data(self, filename: str = "chess_training_data.pkl"):
        """Save training data to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'positions': self.training_positions,
                'evaluations': self.training_evaluations
            }, f)
        print(f"Training data saved to {filename}")
    
    def load_training_data(self, filename: str = "chess_training_data.pkl"):
        """Load training data from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.training_positions = data['positions']
                self.training_evaluations = data['evaluations']
            print(f"Training data loaded from {filename}")
    
    def make_move(self, move_uci: str) -> bool:
        """Make a move on the board"""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except ValueError:
            return False
    
    def print_board(self):
        """Print the current board state"""
        print(self.board)
    
    def get_fen(self) -> str:
        """Get current position in FEN notation"""
        return self.board.fen()
    
    def reset_board(self):
        """Reset board to starting position"""
        self.board = chess.Board()
    
    def generate_pgn_game(self, move_history: List[str], game_result: str = None) -> str:
        """Generate PGN notation for a completed game"""
        # Create a new game
        game = chess.pgn.Game()
        
        # Set game metadata
        game.headers["Event"] = "Neural Chess Self-Play"
        game.headers["Site"] = "AI Training"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Neural Engine"
        game.headers["Black"] = "Neural Engine"
        
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
    
    def save_game_to_history(self, pgn_game: str, game_number: int, game_stats: Dict[str, Any] = None):
        """Save a completed game to the game histories file"""
        history_file = "game_histories.pgn"
        
        # Create game header with metadata
        header = f"\n[Event \"Neural Chess Training Game {game_number}\"]\n"
        header += f"[Site \"AI Training Session\"]\n"
        header += f"[Date \"{time.strftime('%Y.%m.%d')}\"]\n"
        header += f"[Round \"{game_number}\"]\n"
        header += f"[White \"Neural Engine\"]\n"
        header += f"[Black \"Neural Engine\"]\n"
        
        if game_stats:
            if 'result' in game_stats:
                header += f"[Result \"{game_stats['result']}\"]\n"
            if 'moves_played' in game_stats:
                header += f"[Moves \"{game_stats['moves_played']}\"]\n"
            if 'final_evaluation' in game_stats:
                header += f"[Evaluation \"{game_stats['final_evaluation']:.3f}\"]\n"
        
        # Append to history file
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(header)
            f.write(pgn_game)
            f.write("\n\n")
        
        print(f"💾 Game {game_number} saved to {history_file}")
    
    def get_game_result(self) -> str:
        """Get the result of the current game"""
        if self.board.is_checkmate():
            return "1-0" if not self.board.turn else "0-1"
        elif self.board.is_stalemate():
            return "1/2-1/2"
        elif self.board.is_insufficient_material():
            return "1/2-1/2"
        elif self.board.is_fifty_moves():
            return "1/2-1/2"
        elif self.board.is_repetition():
            return "1/2-1/2"
        else:
            return "*"  # Game not finished

def main():
    """Main function to demonstrate neural chess engine"""
    print("🧠 Neural Chess Engine - Learning from Self-Play")
    print("=" * 50)
    
    # Create engine
    engine = NeuralChessEngine()
    
    # Train on self-play games
    print("\n1. Training on self-play games...")
    engine.train_on_self_play(num_games=50, epochs_per_game=3)
    
    # Save training data
    engine.save_training_data()
    
    # Test the trained model
    print("\n2. Testing trained model...")
    engine.reset_board()
    engine.print_board()
    
    print("\nGetting best move with trained neural network...")
    best_move = engine.get_best_move(4, 5.0)
    print(f"Best move: {best_move}")
    
    if best_move:
        engine.make_move(best_move)
        print("\nPosition after move:")
        engine.print_board()
    
    print("\n🎉 Neural chess engine demonstration completed!")

if __name__ == "__main__":
    main()
