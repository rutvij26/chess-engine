import chess
import chess.engine
import numpy as np
from typing import List, Tuple, Optional
import time

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
        self.transposition_table = {}
        self.move_order_table = {}
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional evaluation
        self.pawn_table = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ])
        
        self.knight_table = np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ])
        
        self.bishop_table = np.array([
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ])
        
        self.rook_table = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ])
        
        self.queen_table = np.array([
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ])
        
        self.king_middle_table = np.array([
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ])
        
        self.king_end_table = np.array([
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ])

    def reset_board(self):
        """Reset the board to starting position"""
        self.board = chess.Board()
        self.transposition_table.clear()
        self.move_order_table.clear()

    def make_move(self, move_uci: str) -> bool:
        """Make a move on the board using UCI notation"""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except ValueError:
            return False

    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in UCI notation"""
        return [move.uci() for move in self.board.legal_moves]

    def evaluate_position(self) -> int:
        """Evaluate the current board position"""
        if self.board.is_checkmate():
            return -20000 if self.board.turn else 20000
        
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and positional evaluation
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                value = self.piece_values[piece.piece_type]
                position_value = self.get_piece_position_value(piece, square)
                
                if piece.color == chess.WHITE:
                    score += value + position_value
                else:
                    score -= value + position_value
        
        # Mobility evaluation
        white_mobility = len(list(self.board.legal_moves))
        self.board.push(chess.Move.null())
        black_mobility = len(list(self.board.legal_moves))
        self.board.pop()
        
        score += (white_mobility - black_mobility) * 10
        
        # Pawn structure evaluation
        score += self.evaluate_pawn_structure()
        
        # King safety evaluation
        score += self.evaluate_king_safety()
        
        return score

    def get_piece_position_value(self, piece: chess.Piece, square: chess.Square) -> int:
        """Get the positional value of a piece on a given square"""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if piece.color == chess.BLACK:
            rank = 7 - rank
        
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[rank][file]
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[rank][file]
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[rank][file]
        elif piece.piece_type == chess.ROOK:
            return self.rook_table[rank][file]
        elif piece.piece_type == chess.QUEEN:
            return self.queen_table[rank][file]
        elif piece.piece_type == chess.KING:
            # Use endgame table if few pieces remain
            if self.count_pieces() <= 12:
                return self.king_end_table[rank][file]
            else:
                return self.king_middle_table[rank][file]
        
        return 0

    def count_pieces(self) -> int:
        """Count total number of pieces on the board"""
        return len(self.board.piece_map())

    def evaluate_pawn_structure(self) -> int:
        """Evaluate pawn structure"""
        score = 0
        
        # Doubled pawns penalty
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            if white_pawns > 1:
                score -= 20 * (white_pawns - 1)
            if black_pawns > 1:
                score += 20 * (black_pawns - 1)
        
        return score

    def evaluate_king_safety(self) -> int:
        """Evaluate king safety"""
        score = 0
        
        # Penalize king being in center during middlegame
        if self.count_pieces() > 12:
            white_king_square = self.board.king(chess.WHITE)
            black_king_square = self.board.king(chess.BLACK)
            
            if white_king_square:
                file = chess.square_file(white_king_square)
                if 2 <= file <= 5:
                    score -= 30
            
            if black_king_square:
                file = chess.square_file(black_king_square)
                if 2 <= file <= 5:
                    score += 30
        
        return score

    def order_moves(self, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Captures
            if self.board.is_capture(move):
                score += 1000
                # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
                victim_piece = self.board.piece_at(move.to_square)
                attacker_piece = self.board.piece_at(move.from_square)
                if victim_piece and attacker_piece:
                    victim_value = self.piece_values[victim_piece.piece_type]
                    attacker_value = self.piece_values[attacker_piece.piece_type]
                    score += victim_value - attacker_value
            
            # Promotions
            if move.promotion:
                score += 900
            
            # Checks
            self.board.push(move)
            if self.board.is_check():
                score += 100
            self.board.pop()
            
            # Move history heuristic
            if move in self.move_order_table:
                score += self.move_order_table[move]
            
            move_scores.append((move, score))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]

    def search(self, depth: int, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[int, Optional[chess.Move]]:
        """Search for the best move using minimax with alpha-beta pruning"""
        if depth == 0:
            return self.evaluate_position(), None
        
        # Transposition table lookup
        board_hash = hash(self.board.fen())
        if board_hash in self.transposition_table:
            stored_depth, stored_score, stored_move = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_score, stored_move
        
        moves = list(self.board.legal_moves)
        if not moves:
            if self.board.is_checkmate():
                return -20000 if self.board.turn else 20000, None
            return 0, None
        
        # Order moves for better pruning
        moves = self.order_moves(moves)
        
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
        
        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_score, best_move)
        
        # Update move history
        if best_move in self.move_order_table:
            self.move_order_table[best_move] += depth
        else:
            self.move_order_table[best_move] = depth
        
        return best_score, best_move

    def get_best_move(self, depth: int, time_limit: float = 5.0) -> str:
        """Get the best move within a time limit"""
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
                    print(f"Depth {current_depth}: {move.uci()} (score: {score})")
            except Exception as e:
                print(f"Error at depth {current_depth}: {e}")
                break
        
        return best_move.uci() if best_move else ""

    def print_board(self):
        """Print the current board state"""
        print(self.board)

    def get_fen(self) -> str:
        """Get the current position in FEN notation"""
        return self.board.fen()

    def set_fen(self, fen: str):
        """Set the board position from FEN notation"""
        try:
            self.board = chess.Board(fen)
            self.transposition_table.clear()
            self.move_order_table.clear()
        except ValueError as e:
            print(f"Invalid FEN: {e}")

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.board.is_game_over()

    def get_game_result(self) -> Optional[str]:
        """Get the game result"""
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
        return None
