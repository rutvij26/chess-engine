#!/usr/bin/env python3
"""
Visual Chess Board Display
Shows chess board graphically with clean piece symbols - SCROLLABLE!
"""

import chess
import time
import os

class VisualChessBoard:
    """Clean visual representation of chess board that's scrollable"""
    
    def __init__(self):
        # Clean Unicode chess piece symbols
        self.piece_symbols = {
            chess.PAWN: {
                chess.WHITE: '♙',
                chess.BLACK: '♟'
            },
            chess.KNIGHT: {
                chess.WHITE: '♘',
                chess.BLACK: '♞'
            },
            chess.BISHOP: {
                chess.WHITE: '♗',
                chess.BLACK: '♝'
            },
            chess.ROOK: {
                chess.WHITE: '♖',
                chess.BLACK: '♜'
            },
            chess.QUEEN: {
                chess.WHITE: '♕',
                chess.BLACK: '♛'
            },
            chess.KING: {
                chess.WHITE: '♔',
                chess.BLACK: '♚'
            }
        }
        
        # Clean board design - no blocks!
        self.light_square = ' '
        self.dark_square = '·'  # Just a subtle dot
        self.highlight = '○'    # Clean circle for highlighting
        
    def display_board(self, board, last_move=None, evaluation=None, move_number=None, clear_screen=False):
        """Display the chess board with clean formatting - optionally clear screen"""
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        print("♟️  CHESS BOARD ♟️")
        print("=" * 50)
        
        if move_number:
            print(f"Move: {move_number}")
        if evaluation is not None:
            print(f"Evaluation: {evaluation:+.1f}")
        if last_move:
            print(f"Last move: {last_move}")
        print()
        
        # File labels (a-h)
        print("    a   b   c   d   e   f   g   h")
        print("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
        
        # Board squares
        for rank in range(7, -1, -1):  # 8 to 1
            row = f"{rank + 1} │"
            
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                
                # Determine square color
                is_light = (rank + file) % 2 == 0
                
                # Highlight last move
                if last_move and square in [last_move.from_square, last_move.to_square]:
                    square_char = self.highlight
                elif is_light:
                    square_char = self.light_square
                else:
                    square_char = self.dark_square
                
                # Add piece or empty square
                if piece:
                    piece_symbol = self.piece_symbols[piece.piece_type][piece.color]
                    row += f" {piece_symbol} "
                else:
                    row += f" {square_char} "
                
                if file < 7:
                    row += "│"
            
            row += "│"
            print(row)
            
            # Add rank separators
            if rank > 0:
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
        
        print("  └───┴───┴───┴───┴───┴───┴───┴───┘")
        print("    a   b   c   d   e   f   g   h")
        
        # Turn indicator
        turn_color = "WHITE" if board.turn else "BLACK"
        print(f"\nTurn: {turn_color}")
        
        # Game status
        if board.is_checkmate():
            winner = "BLACK" if board.turn else "WHITE"
            print(f"🎯 CHECKMATE! {winner} wins!")
        elif board.is_stalemate():
            print("🤝 STALEMATE - Draw!")
        elif board.is_check():
            print("⚠️  CHECK!")
        elif board.is_game_over():
            print("🏁 Game Over")
        
        print("\n" + "─" * 50)  # Separator line
    
    def display_move_animation(self, board, move, delay=0.5):
        """Show move animation by highlighting the move"""
        print(f"\n🎯 Making move: {move.uci()}")
        time.sleep(delay)
        
        # Show position before move
        self.display_board(board, evaluation=None, move_number=None)
        time.sleep(delay)
        
        # Make the move
        board.push(move)
        
        # Show position after move
        self.display_board(board, last_move=move, evaluation=None, move_number=None)
        time.sleep(delay)
    
    def display_game_progress(self, board, move_history, current_eval, game_number=None):
        """Display current game state with move history - SCROLLABLE!"""
        print(f"\n🎮 GAME PROGRESS - Game {game_number if game_number else 'Current'}")
        print("=" * 60)
        
        # Show current board
        self.display_board(board, evaluation=current_eval, clear_screen=False)
        
        # Show move history
        if move_history:
            print(f"\n📜 Move History ({len(move_history)} moves):")
            moves_per_line = 5
            for i in range(0, len(move_history), moves_per_line):
                line_moves = move_history[i:i + moves_per_line]
                move_numbers = [f"{j+1}.{move_history[j]}" for j in range(i, min(i + moves_per_line, len(move_history)))]
                print(" ".join(move_numbers))
        
        print("\n" + "═" * 60)  # Thick separator
    
    def display_training_progress(self, game_number, total_games, current_loss, positions_collected):
        """Display training progress information"""
        print(f"\n🧠 TRAINING PROGRESS")
        print("=" * 40)
        print(f"Game: {game_number}/{total_games}")
        print(f"Progress: {100 * game_number / total_games:.1f}%")
        print(f"Current Loss: {current_loss:.4f}")
        print(f"Positions Collected: {positions_collected}")
        
        # Progress bar
        bar_length = 20
        filled_length = int(bar_length * game_number / total_games)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"[{bar}] {game_number}/{total_games}")
        print("─" * 40)

def demo_visual_board():
    """Demonstrate the clean visual chess board"""
    print("🎨 Clean Visual Chess Board Demo - SCROLLABLE!")
    print("=" * 50)
    print("This board will show progress without clearing the screen!")
    print("You can scroll up to see the game history!")
    print()
    
    board = chess.Board()
    visual = VisualChessBoard()
    
    # Show starting position
    visual.display_board(board, evaluation=0.0, move_number=1)
    time.sleep(1)
    
    # Make some moves
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b8c6")
    ]
    
    for i, move in enumerate(moves):
        print(f"\n🎯 Move {i+1}: {move.uci()}")
        time.sleep(0.5)
        
        # Show position before move
        visual.display_board(board, evaluation=None, move_number=i+1)
        time.sleep(0.5)
        
        # Make the move
        board.push(move)
        
        # Show position after move
        visual.display_board(board, last_move=move, evaluation=None, move_number=i+1)
        time.sleep(0.5)
    
    print("\n✅ Clean visual board demo completed!")
    print("Scroll up to see the entire game progress!")

if __name__ == "__main__":
    demo_visual_board()
