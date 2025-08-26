#!/usr/bin/env python3
"""
Neural Chess Engine Launcher
Main entry point for the neural chess engine
"""

import os
import sys

def show_project_structure():
    """Display the simplified project structure"""
    print("\n📁 Project Structure:")
    print("=" * 40)
    print("chess-engine/")
    print("├── 🧠 neural/                    # Neural chess engine")
    print("│   ├── neural_chess_engine.py    # Core neural engine")
    print("│   ├── train_neural_chess.py     # Training script")
    print("│   ├── neural_demo.py            # Neural engine demo")
    print("│   └── pgn_demo.py               # PGN generation demo")
    print("├── 📁 models/                     # Trained neural models")
    print("├── 📁 games/                      # PGN game files")
    print("├── 📁 docs/                       # Documentation")
    print("├── 🚀 run_neural.py              # Neural engine runner")
    print("├── 🧪 test_gpu.py                # GPU acceleration test")
    print("└── 📖 README.md                  # Project documentation")

def run_neural_engine():
    """Run the neural chess engine"""
    print("\n🧠 Neural Chess Engine")
    print("=" * 30)
    print("Choose an option:")
    print("1. 🚀 Neural Network Training")
    print("2. 🧪 Test GPU Acceleration")
    print("3. 📁 View Project Structure")
    print("4. 📖 Open Documentation")
    print("5. 🔙 Back to Main Menu")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Neural Network Training...")
        os.system("python run_neural.py")
    elif choice == "2":
        print("\n🧪 Testing GPU Acceleration...")
        os.system("python test_gpu.py")
    elif choice == "3":
        show_project_structure()
        input("\nPress Enter to continue...")
        run_neural_engine()
    elif choice == "4":
        open_documentation()
    elif choice == "5":
        return
    else:
        print("Invalid choice. Please try again.")
        run_neural_engine()

def open_documentation():
    """Open documentation files"""
    print("\n📖 Documentation")
    print("=" * 20)
    print("Choose a document to view:")
    print("1. 📖 README.md (Main documentation)")
    print("2. 🚀 GPU_ACCELERATION_README.md (GPU setup)")
    print("3. 🔄 RESTRUCTURE_SUMMARY.md (Project changes)")
    print("4. 🔙 Back to Neural Engine Menu")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        os.system("notepad README.md")
    elif choice == "2":
        os.system("notepad GPU_ACCELERATION_README.md")
    elif choice == "3":
        os.system("notepad RESTRUCTURE_SUMMARY.md")
    elif choice == "4":
        run_neural_engine()
        return
    else:
        print("Invalid choice. Please try again.")
        open_documentation()
    
    open_documentation()

def main():
    """Main launcher function"""
    print("🧠 Neural Chess Engine")
    print("=" * 40)
    print("Welcome to the Neural Chess Engine!")
    print("This engine learns to play chess through self-play using neural networks.")
    print()
    print("Features:")
    print("• 🚀 GPU acceleration (CUDA support)")
    print("• 🧠 Neural network training")
    print("• 📊 Self-play learning")
    print("• 💾 Model checkpointing")
    print("• 📜 PGN game generation")
    print()
    
    while True:
        print("\nMain Menu:")
        print("1. 🚀 Launch Neural Engine")
        print("2. 📁 View Project Structure")
        print("3. 📖 Open Documentation")
        print("4. 🧪 Test GPU Acceleration")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_neural_engine()
        elif choice == "2":
            show_project_structure()
            input("\nPress Enter to continue...")
        elif choice == "3":
            open_documentation()
        elif choice == "4":
            print("\n🧪 Testing GPU Acceleration...")
            os.system("python test_gpu.py")
            input("\nPress Enter to continue...")
        elif choice == "5":
            print("\n👋 Goodbye! Thanks for using the Neural Chess Engine!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
