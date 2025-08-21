import sys
import time
from chess_engine import ChessEngine

class UCIHandler:
    def __init__(self):
        self.engine = ChessEngine()
        self.thinking = False
        self.search_depth = 20
        self.search_time = 5.0
        
    def run(self):
        """Main UCI protocol loop"""
        while True:
            try:
                command = input().strip()
                if not command:
                    continue
                    
                self.handle_command(command)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def handle_command(self, command: str):
        """Handle UCI commands"""
        parts = command.split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        
        if cmd == "uci":
            self.send_uci_info()
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            self.engine.reset_board()
        elif cmd == "position":
            self.handle_position(parts[1:])
        elif cmd == "go":
            self.handle_go(parts[1:])
        elif cmd == "stop":
            self.thinking = False
        elif cmd == "quit":
            sys.exit(0)
        elif cmd == "setoption":
            self.handle_setoption(parts[1:])
        elif cmd == "debug":
            pass  # Debug mode not implemented
        else:
            print(f"Unknown command: {command}")
    
    def send_uci_info(self):
        """Send UCI engine information"""
        print("id name ChessEngine v1.0")
        print("id author AI Assistant")
        print("option name SearchDepth type spin default 20 min 1 max 50")
        print("option name SearchTime type spin default 5 min 1 max 60")
        print("uciok")
    
    def handle_position(self, args):
        """Handle position command"""
        if not args:
            return
            
        if args[0] == "startpos":
            self.engine.reset_board()
            if len(args) > 1 and args[1] == "moves":
                for move in args[2:]:
                    self.engine.make_move(move)
        elif args[0] == "fen":
            fen_parts = []
            i = 1
            while i < len(args) and args[i] != "moves":
                fen_parts.append(args[i])
                i += 1
            
            fen = " ".join(fen_parts)
            self.engine.set_fen(fen)
            
            if i < len(args) and args[i] == "moves":
                for move in args[i+1:]:
                    self.engine.make_move(move)
    
    def handle_go(self, args):
        """Handle go command"""
        depth = self.search_depth
        time_limit = self.search_time
        
        i = 0
        while i < len(args):
            if args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                time_limit = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "wtime" and i + 1 < len(args):
                # Time control handling (simplified)
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                # Time control handling (simplified)
                i += 2
            elif args[i] == "infinite":
                depth = 50
                time_limit = 3600.0
                i += 1
            else:
                i += 1
        
        self.thinking = True
        self.start_search(depth, time_limit)
    
    def handle_setoption(self, args):
        """Handle setoption command"""
        if len(args) >= 3 and args[0] == "name" and args[2] == "value":
            option_name = args[1]
            option_value = args[3]
            
            if option_name == "SearchDepth":
                try:
                    self.search_depth = int(option_value)
                except ValueError:
                    pass
            elif option_name == "SearchTime":
                try:
                    self.search_time = float(option_value)
                except ValueError:
                    pass
    
    def start_search(self, depth: int, time_limit: float):
        """Start the search process"""
        if not self.thinking:
            return
            
        try:
            start_time = time.time()
            best_move = self.engine.get_best_move(depth, time_limit)
            
            if self.thinking and best_move:
                elapsed = (time.time() - start_time) * 1000
                print(f"bestmove {best_move}")
            else:
                print("bestmove 0000")
                
        except Exception as e:
            print(f"Error during search: {e}")
            print("bestmove 0000")

def main():
    """Main entry point"""
    handler = UCIHandler()
    handler.run()

if __name__ == "__main__":
    main()
