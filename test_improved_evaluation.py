#!/usr/bin/env python3
"""
Test script for improved PGN move training evaluation
Verifies that the model learns proper chess concepts
"""

import chess
import sys
import os

# Add neural directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

from pgn_move_trainer import PGNMoveTrainer


def test_evaluation_improvements():
    """Test the improved evaluation logic"""
    print("🧠 Testing Improved PGN Move Training Evaluation")
    print("=" * 50)
    
    # Create trainer
    trainer = PGNMoveTrainer()
    
    # Test different board positions
    test_positions = [
        ("Starting position", chess.Board()),
        ("White winning position", create_winning_position()),
        ("Black winning position", create_losing_position()),
        ("Drawish position", create_drawish_position()),
        ("Tactical position", create_tactical_position()),
    ]
    
    for name, board in test_positions:
        print(f"\n📊 Testing: {name}")
        print(f"   Board: {board.fen()}")
        
        # Test different scenarios
        test_scenarios = [
            ("White wins", 1.0, 10),
            ("Black wins", -1.0, 10),
            ("Draw", 0.0, 10),
            ("Late game white wins", 1.0, 40),
            ("Late game black wins", -1.0, 40),
        ]
        
        for scenario_name, result, move_num in test_scenarios:
            eval_score = trainer._calculate_target_evaluation(board, result, move_num)
            print(f"   {scenario_name} (move {move_num}): {eval_score:+.1f}")
        
        # Test repetition detection
        if name == "Tactical position":
            repetition_penalty = trainer._detect_repetitive_moves(
                board, ["e2e4", "e7e5", "e4e5"], 3
            )
            print(f"   Repetition penalty: {repetition_penalty:+.1f}")
        
        # Test tactical opportunities
        tactical_bonus = trainer._detect_tactical_opportunities(board)
        print(f"   Tactical bonus: {tactical_bonus:+.1f}")


def create_winning_position():
    """Create a position where White is clearly winning"""
    board = chess.Board()
    # White has extra queen
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))
    return board


def create_losing_position():
    """Create a position where Black is clearly winning"""
    board = chess.Board()
    # Black has extra queen
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))
    return board


def create_drawish_position():
    """Create a position that's roughly equal"""
    board = chess.Board()
    # Remove most pieces, leave kings and few pawns
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type not in [chess.KING, chess.PAWN]:
            board.remove_piece_at(square)
    return board


def create_tactical_position():
    """Create a position with tactical opportunities"""
    board = chess.Board()
    # White queen attacking black king
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E5, chess.Piece(chess.KING, chess.BLACK))
    return board


if __name__ == "__main__":
    test_evaluation_improvements()
