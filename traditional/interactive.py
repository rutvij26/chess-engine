#!/usr/bin/env python3
"""
Interactive Chess Engine Interface
A simple command-line interface for testing the chess engine
"""

from chess_engine import ChessEngine
import time

def print_help():
    """Print help information"""
    print("\nChess Engine Commands:")
    print("  help                    - Show this help")
    print("  board                   - Display current board")
    print("  moves                   - Show legal moves")
    print("  move <move>             - Make a move (e.g., e2e4)")
    print("  engine <depth>          - Get engine move (e.g., engine 4)")
    print("  evaluate                - Evaluate current position")
    print("  fen                     - Show current FEN")
    print("  setfen <fen>            - Set position from FEN")
    print("  reset                   - Reset to starting position")
    print("  quit                    - Exit program")
    print("\nExample moves: e2e4, d7d5, g1f3, e7e6")

def main():
    """Main interactive loop"""
    engine = ChessEngine()
    print("Chess Engine v1.0")
    print("Type 'help' for commands")
    
    while True:
        try:
            command = input("\nchess> ").strip().lower()
            if not command:
                continue
                
            parts = command.split()
            cmd = parts[0]
            
            if cmd == "help":
                print_help()
                
            elif cmd == "board":
                engine.print_board()
                
            elif cmd == "moves":
                moves = engine.get_legal_moves()
                if moves:
                    print("Legal moves:", " ".join(moves))
                else:
                    print("No legal moves available")
                    
            elif cmd == "move" and len(parts) > 1:
                move = parts[1]
                if engine.make_move(move):
                    print(f"Move {move} played")
                    engine.print_board()
                    
                    # Check game status
                    if engine.is_game_over():
                        result = engine.get_game_result()
                        print(f"Game over! Result: {result}")
                else:
                    print(f"Invalid move: {move}")
                    
            elif cmd == "engine" and len(parts) > 1:
                try:
                    depth = int(parts[1])
                    print(f"Engine thinking at depth {depth}...")
                    start_time = time.time()
                    best_move = engine.get_best_move(depth, 10.0)
                    elapsed = time.time() - start_time
                    
                    if best_move:
                        print(f"Best move: {best_move} (found in {elapsed:.2f}s)")
                        engine.make_move(best_move)
                        engine.print_board()
                        
                        # Check game status
                        if engine.is_game_over():
                            result = engine.get_game_result()
                            print(f"Game over! Result: {result}")
                    else:
                        print("No move found")
                        
                except ValueError:
                    print("Invalid depth. Use: engine <depth>")
                    
            elif cmd == "evaluate":
                score = engine.evaluate_position()
                print(f"Position evaluation: {score}")
                if score > 0:
                    print("White is winning")
                elif score < 0:
                    print("Black is winning")
                else:
                    print("Position is equal")
                    
            elif cmd == "fen":
                print(engine.get_fen())
                
            elif cmd == "setfen" and len(parts) > 1:
                fen = " ".join(parts[1:])
                try:
                    engine.set_fen(fen)
                    print("Position set from FEN")
                    engine.print_board()
                except Exception as e:
                    print(f"Invalid FEN: {e}")
                    
            elif cmd == "reset":
                engine.reset_board()
                print("Board reset to starting position")
                engine.print_board()
                
            elif cmd == "quit":
                print("Goodbye!")
                break
                
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
