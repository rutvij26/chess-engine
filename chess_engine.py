#!/usr/bin/env python3
"""
Chess Engine Project - Main Launcher
Comprehensive chess engine with traditional and neural approaches
"""

import os
import sys

def show_menu():
    """Display the main menu"""
    print("♟️ CHESS ENGINE PROJECT")
    print("=" * 50)
    print("Choose your chess engine experience:")
    print()
    print("🧮 TRADITIONAL ENGINE (Handcrafted Rules)")
    print("  1. Interactive Play")
    print("  2. UCI Protocol (for chess GUIs)")
    print("  3. Engine Demo")
    print("  4. Run Tests")
    print()
    print("🧠 NEURAL ENGINE (Self-Learning AI)")
    print("  5. Basic Learning Demo")
    print("  6. Full Training")
    print("  7. Test Neural System")
    print()
    print("🎨 VISUAL TRAINING (Recommended!)")
    print("  9. Quick Visual Demo (1 game)")
    print("  10. Full Visual Training")
    print("  11. Test Visual Board")
    print()
    print("📚 DOCUMENTATION")
    print("  12. View Project Structure")
    print("  13. Open Documentation")
    print()
    print("0. Exit")
    print()

def run_traditional_engine():
    """Run traditional engine commands"""
    print("\n🧮 TRADITIONAL ENGINE")
    print("-" * 30)
    print("1. Interactive Play")
    print("2. UCI Protocol")
    print("3. Engine Demo")
    print("4. Back to Main Menu")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        os.system("python run_traditional.py interactive")
    elif choice == "2":
        os.system("python run_traditional.py uci")
    elif choice == "3":
        os.system("python run_traditional.py demo")
    elif choice == "4":
        return
    else:
        print("Invalid choice. Returning to main menu.")

def run_neural_engine():
    """Run neural engine commands"""
    print("\n🧠 NEURAL ENGINE")
    print("-" * 20)
    print("1. Basic Learning Demo")
    print("2. Full Training")
    print("3. Test Neural System")
    print("5. Back to Main Menu")
    
    choice = input("\nChoose option (1-5): ").strip()
    
    if choice == "1":
        os.system("python run_neural.py demo")
    elif choice == "2":
        os.system("python run_neural.py train")
    elif choice == "3":
        print("\n🧠 Starting Neural Network Training...")
        os.system("python run_neural.py")
        print("Training cancelled.")
    elif choice == "4":
        os.system("python run_neural.py test")
    elif choice == "5":
        return
    else:
        print("Invalid choice. Returning to main menu.")

def run_visual_training():
    """Run visual training commands"""
    print("\n🎨 VISUAL TRAINING")
    print("-" * 20)
    print("1. Quick Visual Demo (1 game)")
    print("2. Full Visual Training")
    print("3. Test Visual Board")
    print("4. Back to Main Menu")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        os.system("python run_visual.py simple")
    elif choice == "2":
        os.system("python run_visual.py training")
    elif choice == "3":
        os.system("python run_visual.py board")
    elif choice == "4":
        return
    else:
        print("Invalid choice. Returning to main menu.")

def show_project_structure():
    """Display the project structure"""
    print("\n📁 PROJECT STRUCTURE")
    print("=" * 50)
    print("chess-engine/")
    print("├── 📁 traditional/          # Traditional chess engine")
    print("│   ├── chess_engine.py      # Core engine")
    print("│   ├── uci_handler.py       # UCI protocol")
    print("│   ├── interactive.py       # Command interface")
    print("│   └── demo.py              # Capabilities demo")
    print()
    print("├── 📁 neural/               # Neural network engine")
    print("│   ├── neural_chess_engine.py # Neural engine")
    print("│   ├── train_neural_chess.py  # Training script")
    print("│   ├── neural_demo.py         # Learning demo")
    print("│   └── pgn_demo.py            # PGN generation demo")
    print()
    print("├── 📁 visual/               # Visual components")
    print("│   ├── visual_chess_board.py  # Clean chess board")
    print("│   ├── visual_training.py     # Visual training menu")
    print("│   ├── simple_visual_training.py # Quick demo")
    print("│   └── quick_visual_demo.py   # Fast demo")
    print()
    print("├── 📁 docs/                  # Documentation")
    print("│   ├── NEURAL_README.md       # Neural engine guide")
    print("│   ├── PROJECT_SUMMARY.md     # Complete overview")
    print("│   └── PROJECT_SUMMARY.md     # Complete overview")
    print()
    print("├── 📁 tests/                  # Test files")
    print("│   └── test_engine.py          # Traditional tests")
    print()
    print("├── 📁 scripts/                # Utility scripts")
    print("│   └── run_engine.bat          # Windows launcher")
    print()
    print("├── run_traditional.py         # Traditional launcher")
    print("├── run_neural.py              # Neural launcher")
    print("├── run_visual.py              # Visual launcher")
    print("└── chess_engine.py            # This main launcher")
    print()
    input("Press Enter to continue...")

def open_documentation():
    """Open documentation files"""
    print("\n📚 DOCUMENTATION")
    print("-" * 20)
    print("1. View NEURAL_README.md")
    print("2. View PROJECT_SUMMARY.md")
    print("3. View NEURAL_README.md")
    print("4. Back to Main Menu")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        os.system("start docs/NEURAL_README.md")
    elif choice == "2":
        os.system("start docs/PROJECT_SUMMARY.md")
    elif choice == "3":
            os.system("start docs/NEURAL_README.md")
    elif choice == "4":
        return
    else:
        print("Invalid choice. Returning to main menu.")

def main():
    """Main function"""
    while True:
        show_menu()
        choice = input("Enter your choice (0-13): ").strip()
        
        if choice == "0":
            print("\n👋 Thanks for using the Chess Engine Project!")
            print("Goodbye! ♟️")
            break
        elif choice == "1":
            run_traditional_engine()
        elif choice == "2":
            os.system("python run_traditional.py uci")
        elif choice == "3":
            os.system("python run_traditional.py demo")
        elif choice == "4":
            os.system("python run_traditional.py test")
        elif choice == "5":
            os.system("python run_neural.py demo")
        elif choice == "6":
            os.system("python run_neural.py train")
        elif choice == "7":
            print("\n🧠 Starting Neural Network Training...")
            os.system("python run_neural.py")
            print("Training cancelled.")
        elif choice == "8":
            os.system("python run_neural.py test")
        elif choice == "9":
            os.system("python run_visual.py simple")
        elif choice == "10":
            os.system("python run_visual.py training")
        elif choice == "11":
            os.system("python run_visual.py board")
        elif choice == "12":
            show_project_structure()
        elif choice == "13":
            open_documentation()
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
