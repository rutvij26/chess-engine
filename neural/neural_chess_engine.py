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
    
    def __init__(self, model_path: str = None):
        self.board = chess.Board()
        self.model_path = model_path  # Store the model path for reference
        
        # GPU acceleration setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.model = ChessNeuralNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            # Load to CPU first, then move to device
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
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
        
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
    
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
        
        # Positional bonuses/penalties
        score += self._evaluate_positional_factors(board)
        
        return score
    
    def _evaluate_positional_factors(self, board: chess.Board) -> float:
        """Evaluate positional factors beyond material"""
        score = 0
        
        # Center control bonus
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                if piece.color == chess.WHITE:
                    score += 10
                else:
                    score -= 10
        
        # Development bonus (pieces moved from starting position)
        score += self._evaluate_development(board)
        
        # King safety
        score += self._evaluate_king_safety(board)
        
        # Pawn structure
        score += self._evaluate_pawn_structure(board)
        
        return score
    
    def _evaluate_development(self, board: chess.Board) -> float:
        """Evaluate piece development"""
        score = 0
        
        # Bonus for knights and bishops developed
        for square in [chess.B1, chess.G1, chess.C1, chess.F1]:  # White starting squares
            if not board.piece_at(square) or board.piece_at(square).piece_type not in [chess.KNIGHT, chess.BISHOP]:
                score += 15  # Bonus for developing these pieces
        
        for square in [chess.B8, chess.G8, chess.C8, chess.F8]:  # Black starting squares
            if not board.piece_at(square) or board.piece_at(square).piece_type not in [chess.KNIGHT, chess.BISHOP]:
                score -= 15  # Bonus for developing these pieces
        
        return score
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety"""
        score = 0
        
        # Penalty for exposed kings
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square:
            # Penalty for king in center during middlegame
            if chess.square_rank(white_king_square) > 1:
                score -= 20
        
        if black_king_square:
            # Penalty for king in center during middlegame
            if chess.square_rank(black_king_square) < 6:
                score += 20
        
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure"""
        score = 0
        
        # Bonus for connected pawns
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                # Check for connected pawns
                for adjacent in [square + 1, square - 1, square + 8, square - 8]:
                    if 0 <= adjacent < 64:
                        adjacent_piece = board.piece_at(adjacent)
                        if adjacent_piece and adjacent_piece.piece_type == chess.PAWN and adjacent_piece.color == piece.color:
                            if piece.color == chess.WHITE:
                                score += 5
                            else:
                                score -= 5
        
        return score
    
    def get_best_move(self, depth: int, time_limit: float = 5.0, verbose: bool = True) -> Optional[chess.Move]:
        """Get best move using minimax with neural evaluation"""
        start_time = time.time()
        best_move = None
        best_score = float('-inf') if self.board.turn else float('inf')
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            try:
                score, move = self.search(current_depth)
                if move:
                    best_move = move
                    best_score = score
                    # Only print the final result for this depth if verbose
                    if verbose:
                        print(f"Depth {current_depth}: {move.uci()} (score: {score:.1f})")
            except Exception as e:
                if verbose:
                    print(f"Error at depth {current_depth}: {e}")
                break
        
        return best_move
    
    def get_best_move_for_position(self, board: chess.Board, depth: int, time_limit: float = 5.0, verbose: bool = True) -> Optional[chess.Move]:
        """Get best move for a specific board position without changing internal state"""
        start_time = time.time()
        best_move = None
        best_score = float('-inf') if board.turn else float('inf')
        
        # Debug: check if the board position is valid
        if verbose:
            print(f"🔍 Evaluating position: {board.fen()}")
            print(f"   Turn: {'White' if board.turn else 'Black'}")
            print(f"   Legal moves: {[move.uci() for move in list(board.legal_moves)[:5]]}...")
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            try:
                score, move = self.search_position(board, current_depth)
                if move:
                    best_move = move
                    best_score = score
                    # Only print the final result for this depth if verbose
                    if verbose:
                        print(f"Depth {current_depth}: {move.uci()} (score: {score:.1f})")
            except Exception as e:
                if verbose:
                    print(f"Error at depth {current_depth}: {e}")
                break
        
        if verbose and best_move:
            print(f"🎯 Best move found: {best_move.uci()}")
            print(f"   Is legal: {best_move in board.legal_moves}")
        
        # Final validation: ensure the move is legal
        if best_move and best_move not in board.legal_moves:
            print(f"⚠️  Warning: Best move {best_move.uci()} is not legal in position {board.fen()}")
            # Try to find a legal move as fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]  # Use first legal move as fallback
                print(f"🔄 Using fallback legal move: {best_move.uci()}")
            else:
                best_move = None
        
        return best_move
    
    def search_position(self, board: chess.Board, depth: int, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[float, Optional[chess.Move]]:
        """Search using minimax with alpha-beta pruning for a specific board position"""
        if depth == 0:
            return self.evaluate_position_neural(board), None
        
        moves = list(board.legal_moves)
        if not moves:
            if board.is_checkmate():
                return -20000 if board.turn else 20000, None
            return 0, None
        
        best_move = moves[0]
        best_score = float('-inf') if board.turn else float('inf')
        
        for move in moves:
            # Work on a copy to avoid modifying the original board
            board_copy = board.copy()
            board_copy.push(move)
            score, _ = self.search_position(board_copy, depth - 1, alpha, beta)
            
            if board.turn:  # White's turn (maximizing)
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
    
    def get_best_move_with_tactical_validation(self, depth: int, time_limit: float = 5.0, verbose: bool = True) -> Optional[chess.Move]:
        """Get best move with tactical validation to avoid unsound moves"""
        start_time = time.time()
        best_move = None
        best_score = float('-inf') if self.board.turn else float('inf')
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            try:
                score, move = self.search(current_depth)
                if move:
                    # Tactical validation: check if move is sound
                    if self._is_tactically_sound(move):
                        best_move = move
                        best_score = score
                        if verbose:
                            print(f"Depth {current_depth}: {move.uci()} (score: {score:.1f}) - Tactically sound")
                    else:
                        if verbose:
                            print(f"Depth {current_depth}: {move.uci()} (score: {score:.1f}) - Tactically unsound, skipping")
                        continue
            except Exception as e:
                if verbose:
                    print(f"Error at depth {current_depth}: {e}")
                break
        
        # If no tactically sound move found, fall back to regular best move
        if best_move is None:
            if verbose:
                print("No tactically sound move found, using regular best move")
            best_move = self.get_best_move(depth, time_limit, verbose)
        
        return best_move
    
    def _is_tactically_sound(self, move: chess.Move) -> bool:
        """Check if a move is tactically sound (doesn't lose material)"""
        # Make the move on a copy
        board_copy = self.board.copy()
        board_copy.push(move)
        
        # Check if the move gives check
        if board_copy.is_check():
            # If it's a check, verify it's not a losing move
            # Look for opponent's responses
            opponent_moves = list(board_copy.legal_moves)
            if opponent_moves:
                # Check if any opponent move can capture the checking piece
                checking_piece_square = move.to_square
                for opp_move in opponent_moves:
                    if opp_move.to_square == checking_piece_square:
                        # Opponent can capture the checking piece
                        # Check if this is good for us
                        board_copy2 = board_copy.copy()
                        board_copy2.push(opp_move)
                        
                        # Compare material before and after
                        material_before = self._get_material_value(self.board)
                        material_after = self._get_material_value(board_copy2)
                        
                        if self.board.turn:  # White's turn
                            if material_after < material_before:
                                return False  # We lose material
                        else:  # Black's turn
                            if material_after > material_before:
                                return False  # We lose material
        
        # Check if the move hangs a piece
        piece_value_before = self._get_piece_value_at_square(self.board, move.from_square)
        piece_value_after = self._get_piece_value_at_square(board_copy, move.to_square)
        
        # If we moved to a square where our piece can be captured for free
        if piece_value_after > 0:
            # Check if opponent can capture it
            for opp_move in board_copy.legal_moves:
                if opp_move.to_square == move.to_square:
                    # Opponent can capture our piece
                    board_copy3 = board_copy.copy()
                    board_copy3.push(opp_move)
                    
                    # Check if we can recapture
                    can_recapture = False
                    for our_move in board_copy3.legal_moves:
                        if our_move.to_square == move.to_square:
                            can_recapture = True
                            break
                    
                    if not can_recapture:
                        # We can't recapture, this move hangs a piece
                        return False
        
        return True
    
    def _is_tactically_unsound(self, move: chess.Move) -> bool:
        """Check if a move is tactically unsound"""
        return not self._is_tactically_sound(move)
    
    def _get_material_value(self, board: chess.Board) -> int:
        """Get total material value on the board"""
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    total += value
                else:
                    total -= value
        return total
    
    def _get_piece_value_at_square(self, board: chess.Board, square: int) -> int:
        """Get the value of a piece at a specific square"""
        piece = board.piece_at(square)
        if piece:
            return self.piece_values[piece.piece_type]
        return 0
    
    def _evaluate_position_with_tactical_penalties(self) -> float:
        """Evaluate position with penalties for tactical errors"""
        base_eval = self.evaluate_position_neural(self.board)
        
        # Penalties for tactical weaknesses
        tactical_penalty = 0
        
        # Penalty for hanging pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                if self._is_piece_hanging(square, piece.color):
                    tactical_penalty += self.piece_values[piece.piece_type] * 0.1  # 10% penalty
        
        # Penalty for exposed king
        if self.board.is_check():
            tactical_penalty += 50  # Penalty for being in check
        
        # Penalty for poor pawn structure
        tactical_penalty += self._evaluate_pawn_structure_penalties()
        
        # Apply penalty
        if self.board.turn:  # White's turn
            final_eval = base_eval - tactical_penalty
        else:  # Black's turn
            final_eval = base_eval + tactical_penalty
        
        return final_eval
    
    def _is_piece_hanging(self, square: int, color: bool) -> bool:
        """Check if a piece is hanging (can be captured for free or with advantage)"""
        piece = self.board.piece_at(square)
        if not piece or piece.color != color:
            return False
        
        piece_value = self.piece_values[piece.piece_type]
        
        # Check if opponent can capture this piece
        for opp_move in self.board.legal_moves:
            if opp_move.to_square == square:
                # Opponent can capture our piece
                # Check if we can recapture
                self.board.push(opp_move)
                can_recapture = False
                best_recapture_value = 0
                
                for our_move in self.board.legal_moves:
                    if our_move.to_square == square:
                        can_recapture = True
                        recapture_piece = self.board.piece_at(our_move.from_square)
                        if recapture_piece:
                            best_recapture_value = max(best_recapture_value, self.piece_values[recapture_piece.piece_type])
                
                self.board.pop()
                
                if not can_recapture:
                    return True  # Piece is hanging
                elif best_recapture_value < piece_value:
                    return True  # We lose material in the exchange
        
        return False
    
    def _evaluate_pawn_structure_penalties(self) -> float:
        """Evaluate pawn structure and return penalties"""
        penalty = 0
        
        # Penalty for isolated pawns
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if self._is_pawn_isolated(square, piece.color):
                    penalty += 20
        
        # Penalty for doubled pawns
        penalty += self._count_doubled_pawns() * 15
        
        return penalty
    
    def _is_pawn_isolated(self, square: int, color: bool) -> bool:
        """Check if a pawn is isolated (no friendly pawns on adjacent files)"""
        file = chess.square_file(square)
        
        # Check adjacent files
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file <= 7:
                # Check if there are any friendly pawns on adjacent files
                has_friendly_pawn = False
                for rank in range(8):
                    adj_square = chess.square(adj_file, rank)
                    adj_piece = self.board.piece_at(adj_square)
                    if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == color:
                        has_friendly_pawn = True
                        break
                
                if has_friendly_pawn:
                    return False  # Not isolated
        
        return True  # Isolated
    
    def _count_doubled_pawns(self) -> int:
        """Count the number of doubled pawns"""
        doubled_count = 0
        
        for file in range(8):
            pawns_in_file = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    pawns_in_file += 1
            
            if pawns_in_file > 1:
                doubled_count += pawns_in_file - 1
        
        return doubled_count
    
    def self_play_game(self, show_progress: bool = True, randomness: float = 0.2) -> Dict[str, Any]:
        """Play a game against itself and return training data with game information
        
        Args:
            show_progress: Whether to show detailed progress
            randomness: Probability of making a random move (0.0 = always best, 1.0 = always random)
        """
        self.board = chess.Board()
        game_data = []
        move_history = []
        move_num = 0
        
        # Reset move repetition counter for new game
        self.move_repetition_count = {}
        
        print("      🎯 Starting new game...")
        if randomness > 0:
            print(f"      🎲 Randomness: {randomness*100:.0f}%")
        else:
            print("      🤖 Always best move")
        
        while not self.board.is_game_over():
            move_num += 1
            
            # Get evaluation before move
            pre_eval = self.evaluate_position_neural(self.board)
            
            # Show current position info
            if show_progress and move_num % 5 == 0:
                turn_color = "White" if self.board.turn else "Black"
                print(f"      📍 Move {move_num}: {turn_color}'s turn - Eval: {pre_eval:.2f}")
            
            # Show move counter every 3 moves for better feedback
            if show_progress and move_num % 3 == 0:
                print(f"      🔄 Processing move {move_num}...")
            
            # Make a move (either best move or random based on randomness parameter)
            if random.random() > randomness:  # (1-randomness)% best move, randomness% random
                if show_progress and move_num % 5 == 0:
                    print(f"      🤖 Thinking... (best move)")
                
                # Use tactical validation to avoid unsound moves
                best_move = self.get_best_move_with_tactical_validation(3, 1.0, verbose=False)
                
                if best_move:
                    # Check for move repetition
                    move_key = f"{best_move.uci()}_{self.board.fen()[:50]}"  # Move + position hash
                    if hasattr(self, 'move_repetition_count'):
                        if move_key in self.move_repetition_count:
                            self.move_repetition_count[move_key] += 1
                            if self.move_repetition_count[move_key] > 2:  # Penalize excessive repetition
                                if show_progress:
                                    print(f"      ⚠️  Move repetition detected: {best_move.uci()}")
                        else:
                            self.move_repetition_count[move_key] = 1
                    else:
                        self.move_repetition_count = {move_key: 1}
                    
                    move_history.append(best_move.uci())
                    self.board.push(best_move)
                    
                    # Show move immediately for better feedback
                    if show_progress:
                        print(f"      ♟️  Move {move_num}: {best_move.uci()}")
            else:
                if show_progress and move_num % 5 == 0:
                    print(f"      🎲 Exploring... (random move)")
                
                # Get random move that's not excessively repetitive
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    # Filter out moves that have been repeated too many times
                    non_repetitive_moves = []
                    for move in legal_moves:
                        move_key = f"{move.uci()}_{self.board.fen()[:50]}"
                        if not hasattr(self, 'move_repetition_count') or move_key not in self.move_repetition_count or self.move_repetition_count[move_key] < 2:
                            non_repetitive_moves.append(move)
                    
                    # Use non-repetitive moves if available, otherwise use all legal moves
                    if non_repetitive_moves:
                        move = random.choice(non_repetitive_moves)
                    else:
                        move = random.choice(legal_moves)
                    
                    # Update repetition count
                    move_key = f"{move.uci()}_{self.board.fen()[:50]}"
                    if not hasattr(self, 'move_repetition_count'):
                        self.move_repetition_count = {}
                    self.move_repetition_count[move_key] = self.move_repetition_count.get(move_key, 0) + 1
                    
                    move_history.append(move.uci())
                    self.board.push(move)
                    
                    # Show move immediately for better feedback
                    if show_progress:
                        print(f"      🎲 Move {move_num}: {move.uci()} (random)")
            
            # Get evaluation after move with tactical penalties
            post_eval = self._evaluate_position_with_tactical_penalties()
            
            # Store position and evaluation
            position_tensor = self.board_to_tensor(self.board)
            game_data.append((position_tensor, post_eval))
            
            # Add to training data
            self.training_positions.append(position_tensor)
            self.training_evaluations.append(post_eval)
            
            # Show detailed progress every 10 moves
            if show_progress and move_num % 10 == 0:
                eval_change = post_eval - pre_eval
                eval_arrow = "↗️" if eval_change > 0 else "↘️" if eval_change < 0 else "→"
                print(f"      📊 Move {move_num}: {move_history[-1]} | Eval: {pre_eval:.2f} → {post_eval:.2f} {eval_arrow}")
                
                # Show game status
                if self.board.is_check():
                    print(f"      ⚠️  CHECK!")
                elif self.board.is_checkmate():
                    print(f"      🎯 CHECKMATE!")
                    break
                
                # Show move count progress
                print(f"      📈 Progress: {move_num} moves played")
        
        # Get final game result
        final_result = self.get_game_result()
        final_evaluation = self._evaluate_position_with_tactical_penalties()
        
        if show_progress:
            print(f"      🏁 Game completed in {move_num} moves")
            print(f"      📊 Final evaluation: {final_evaluation:.2f}")
            print(f"      🎯 Result: {final_result}")
            if hasattr(self, 'move_repetition_count'):
                total_repetitions = sum(count for count in self.move_repetition_count.values() if count > 1)
                print(f"      🔄 Total move repetitions: {total_repetitions}")
        
        return {
            'game_data': game_data,
            'move_history': move_history,
            'result': final_result,
            'moves_played': len(move_history),
            'final_evaluation': final_evaluation,
            'game_over': self.board.is_game_over(),
            'repetitions': getattr(self, 'move_repetition_count', {})
        }
    
    def train_on_self_play(self, num_games: int = 100, epochs_per_game: int = 5, randomness: float = 0.2):
        """Train the neural network on self-play games
        
        Args:
            num_games: Number of games to play
            epochs_per_game: Number of training epochs per game
            randomness: Probability of making a random move (0.0 = always best, 1.0 = always random)
        """
        print(f"Starting self-play training with {num_games} games...")
        if randomness > 0:
            print(f"🎲 Randomness: {randomness*100:.0f}% (exploration)")
        else:
            print("🤖 Always best move (deterministic)")
        
        for game_num in range(num_games):
            print(f"Playing game {game_num + 1}/{num_games}")
            
            # Play a game with specified randomness
            game_result = self.self_play_game(show_progress=True, randomness=randomness)
            
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
                # Ensure models directory exists
                os.makedirs("models", exist_ok=True)
                model_path = f"models/chess_model_game_{game_num + 1}.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"💾 Model saved after {game_num + 1} games: {model_path}")
        
        print("Self-play training completed!")
    
    def train_model(self, epochs: int = 1):
        """Train the neural network on collected data"""
        if len(self.training_positions) < 10:
            return
        
        # Convert to tensors and move to device
        positions = torch.cat(self.training_positions, dim=0).to(self.device)
        evaluations = torch.tensor(self.training_evaluations, dtype=torch.float32).unsqueeze(1).to(self.device)
        
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
    
    def train_model_with_progress(self, epochs: int = 1):
        """Train the neural network with detailed progress indicators"""
        if len(self.training_positions) < 10:
            print("   ⚠️  Not enough training data (< 10 positions)")
            return
        
        print(f"   📊 Training on {len(self.training_positions)} positions...")
        
        # Convert to tensors and move to device
        positions = torch.cat(self.training_positions, dim=0).to(self.device)
        evaluations = torch.tensor(self.training_evaluations, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Create dataset and dataloader
        dataset = ChessDataset(positions, evaluations)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        
        total_batches = len(dataloader)
        print(f"   🔄 {total_batches} batches per epoch")
        
        for epoch in range(epochs):
            print(f"   🧠 Epoch {epoch + 1}/{epochs}:")
            total_loss = 0
            batch_count = 0
            
            for batch_positions, batch_evaluations in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_positions)
                loss = self.criterion(outputs, batch_evaluations)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Show progress every few batches
                if batch_count % max(1, total_batches // 10) == 0 or batch_count == total_batches:
                    progress = (batch_count / total_batches) * 100
                    current_loss = loss.item()
                    print(f"      📈 Batch {batch_count}/{total_batches} ({progress:.1f}%) - Loss: {current_loss:.4f}")
            
            avg_loss = total_loss / total_batches
            print(f"   ✅ Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}")
            
            # Store loss for tracking
            self.last_loss = avg_loss
        
        self.model.eval()
        print(f"   🎯 Training completed - Final Loss: {self.last_loss:.4f}")
    
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
    
    def set_fen(self, fen: str):
        """Set board to a specific FEN position"""
        self.board = chess.Board(fen)
    
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
        # Ensure games directory exists
        os.makedirs("games", exist_ok=True)
        history_file = "games/game_histories.pgn"
        
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
    
    # Train on self-play games with different randomness settings
    print("\n1. Training on self-play games...")
    
    # Option 1: Always best move (deterministic)
    print("\n🎯 Training with always best move (deterministic):")
    engine.train_on_self_play(num_games=10, epochs_per_game=2, randomness=0.0)
    
    # Option 2: 20% randomness (balanced exploration)
    print("\n🎲 Training with 20% randomness (balanced exploration):")
    engine.train_on_self_play(num_games=10, epochs_per_game=2, randomness=0.2)
    
    # Option 3: 50% randomness (high exploration)
    print("\n🎲 Training with 50% randomness (high exploration):")
    engine.train_on_self_play(num_games=10, epochs_per_game=2, randomness=0.5)
    
    # Save training data
    engine.save_training_data()
    
    # Test the trained model
    print("\n2. Testing trained model...")
    engine.reset_board()
    engine.print_board()
    
    print("\nGetting best move with trained neural network...")
    best_move = engine.get_best_move(4, 5.0)
    print(f"Best move: {best_move.uci()}")
    
    if best_move:
        engine.make_move(best_move.uci())
        print("\nPosition after move:")
        engine.print_board()
    
    print("\n🎉 Neural chess engine demonstration completed!")

if __name__ == "__main__":
    main()
