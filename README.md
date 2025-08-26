# ♟️ Chess Engine Project

A comprehensive chess engine project featuring both traditional and neural network approaches to chess AI.

## 🏗️ **What's Included**

### **🧮 Traditional Chess Engine**

- **Knowledge Source**: Human chess expertise (handcrafted rules)
- **Evaluation**: Mathematical formulas + piece-square tables
- **Strength**: Immediate, consistent, beginner-intermediate level
- **Learning**: None - knowledge is static

### **🧠 Neural Chess Engine**

- **Knowledge Source**: Self-play learning (no human input)
- **Evaluation**: Neural network predictions
- **Strength**: Improves over time, can reach advanced level
- **Learning**: Continuous improvement through training

## 🚀 **Quick Start**

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

## 🎮 **All Available Commands**

### **🎯 Main Launcher (Recommended!)**

```bash
python chess_engine.py
```

- Interactive menu for all chess engine options
- Easy navigation between traditional, neural, and visual components

### **🧮 Traditional Engine Commands**

#### **Using Main Launcher**

```bash
python chess_engine.py
# Choose option 1-4 for traditional engine
```

#### **Direct Commands**

```bash
python run_traditional.py interactive    # Interactive chess interface
python run_traditional.py uci            # UCI protocol handler
python run_traditional.py demo           # Engine capabilities demo
python run_traditional.py test           # Run test suite
```

#### **Interactive Play**

```bash
python run_traditional.py interactive
```

- `help` - Show available commands
- `board` - Display current board
- `moves` - Show legal moves
- `move <move>` - Make a move (e.g., `move e2e4`)
- `engine <depth>` - Get engine's best move
- `evaluate` - Show position evaluation
- `fen` - Show current FEN notation
- `setfen <fen>` - Set position from FEN
- `reset` - Reset to starting position
- `quit` - Exit

#### **UCI Protocol (for chess GUIs)**

```bash
python run_traditional.py uci
```

- Compatible with Chess.com, Lichess, Arena, etc.

#### **Demo & Testing**

```bash
python run_traditional.py demo           # See engine capabilities
python run_traditional.py test           # Run test suite
```

### **🧠 Neural Engine Commands**

#### **Using Main Launcher**

```bash
python chess_engine.py
# Choose option 5-8 for neural engine
```

#### **Direct Commands**

```bash
python run_neural.py demo          # Basic neural learning demo
python run_neural.py train         # Full neural training
python run_neural.py                # 🧠 Neural network training
python run_neural.py test          # Test neural system
```

#### **Visual Training (Recommended!)**

```bash
python run_visual.py simple        # Quick 1-game demo with visual board
python run_visual.py training      # Full visual training menu
python run_visual.py quick         # Fast visual demo
```

#### **Traditional Training (Text-based)**

```bash
python run_neural.py demo          # Learning demonstration
python run_neural.py train         # Full training script
```

#### **Visual Training Menu Options**

When you run `python visual_training.py`, you get:

1. **Visual Training Demo** - Watch neural network learn with clean board
2. **Interactive Visual Play** - Play against trained model with visual board
3. **Exit**

### **🎨 Visual Board Commands**

#### **Using Main Launcher**

```bash
python chess_engine.py
# Choose option 9-11 for visual training
```

#### **Direct Commands**

```bash
python run_visual.py simple        # Quick 1-game visual demo
python run_visual.py training      # Full visual training menu
python run_visual.py quick         # Fast visual demo
python run_visual.py board         # Test the clean visual board
```

## 📁 **Project Structure**

```
chess-engine/
├── 📁 traditional/              # Traditional chess engine
│   ├── chess_engine.py          # Core engine with handcrafted evaluation
│   ├── uci_handler.py           # UCI protocol support
│   ├── interactive.py            # Command-line interface
│   ├── demo.py                   # Capability demonstration
│   └── __init__.py              # Package initialization
│
├── 📁 neural/                   # Neural network engine
│   ├── neural_chess_engine.py   # Neural network-based engine
│   ├── train_neural_chess.py    # Training script
│   ├── neural_demo.py           # Learning demonstration
│   ├── train_neural_chess.py    # 🧠 Neural network training
│   └── __init__.py              # Package initialization
│
├── 📁 visual/                   # Visual components
│   ├── visual_chess_board.py    # Clean, scrollable chess board
│   ├── visual_training.py       # Visual training menu
│   ├── simple_visual_training.py # Quick visual demo
│   ├── quick_visual_demo.py     # Fast visual demo
│   └── __init__.py              # Package initialization
│
├── 📁 docs/                     # Documentation
│   ├── NEURAL_README.md          # Neural engine guide
│   ├── PROJECT_SUMMARY.md        # Complete project overview
│   └── NEURAL_README.md          # 🧠 Neural training guide
│
├── 📁 tests/                    # Test files
│   ├── test_engine.py            # Traditional engine tests
│   └── test_engine.py           # Traditional engine tests
│   └── __init__.py              # Package initialization
│
├── 📁 scripts/                  # Utility scripts
│   └── run_engine.bat           # Windows launcher
│
├── chess_engine.py              # 🎯 Main launcher (recommended!)
├── run_traditional.py           # Traditional engine launcher
├── run_neural.py                # Neural engine launcher
├── run_visual.py                # Visual training launcher
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🎯 **Recommended Learning Path**

### **🎯 Option 1: Use Main Launcher (Easiest!)**

```bash
python chess_engine.py
```

- Interactive menu for all options
- No need to remember specific commands
- Easy navigation between components

### **Option 2: Step-by-Step Commands**

#### **1. Start with Traditional Engine (5 minutes)**

```bash
python run_traditional.py interactive
```

- Learn basic chess engine concepts
- See how handcrafted evaluation works

#### **2. Try Visual Neural Training (10 minutes)**

```bash
python run_visual.py simple
```

- Watch neural network learn chess visually
- Clean, scrollable board display
- See the learning process in action

#### **3. Full Neural Training (30+ minutes)**

```bash
python run_visual.py training
```

- Choose option 1 for visual training demo
- Watch multiple games with progress tracking
- See how the network improves over time

#### **4. Advanced Usage**

```bash
python run_neural.py train          # Custom training parameters
python run_traditional.py uci       # Use with chess GUIs
```

#### **5. 🧠 Neural Network Training (Advanced!)**

```bash
python run_neural.py                # Start neural network training
```

**What it does**:

- **Target**: Configurable number of training games
- **Duration**: Depends on number of games specified
- **Games**: User-specified number of self-play games
- **Result**: Trained neural chess AI

## 🔧 **Training Parameters**

### **Quick Demo (Recommended for first time)**

```bash
python run_visual.py simple        # 1 game, 10 moves max
```

### **Full Training**

```bash
python run_visual.py training      # Menu-driven training
```

### **Custom Training**

Edit `neural/train_neural_chess.py`:

```python
NUM_GAMES = 100          # More games = stronger play
EPOCHS_PER_GAME = 3      # More training per game
LEARNING_RATE = 0.001    # How fast it learns
```

## 🎨 **Visual Features**

### **Clean Chess Board**

- ✅ **No block characters** - Clean Unicode pieces
- ✅ **Scrollable** - Screen doesn't clear, see full history
- ✅ **Move highlighting** - Last moves shown with circles (○)
- ✅ **Real-time progress** - Watch learning happen
- ✅ **Game history** - Scroll up to see entire process

### **What You'll See**

1. **Starting Position**: Clean chess board
2. **Move by Move**: Each move with highlighting
3. **Position Evaluation**: Neural network's assessment
4. **Game Progress**: Complete history you can scroll through
5. **Learning Outcome**: Positions collected and learned from

## 🚨 **Common Issues & Solutions**

### **Visual Board Not Working**

```bash
# Make sure you have the visual board file
ls visual_chess_board.py

# If missing, the neural engine will fall back to text mode
```

### **Training Takes Too Long**

```bash
# Use quick demo instead
python simple_visual_training.py  # Just 1 game, 10 moves
```

### **Want to See More Games**

```bash
python visual_training.py         # Choose option 1 for multiple games
```

## 🏆 **Achievement Levels**

| Level              | Command                            | Time         | What You'll See                   |
| ------------------ | ---------------------------------- | ------------ | --------------------------------- |
| **Beginner**       | `python interactive.py`            | 5 min        | Traditional chess engine          |
| **Explorer**       | `python simple_visual_training.py` | 10 min       | Neural learning with visual board |
| **Learner**        | `python visual_training.py`        | 30 min       | Multiple games, progress tracking |
| **Master**         | `python train_neural_chess.py`     | 2+ hours     | Full training, save models        |
| **🏆 Grandmaster** | `python run_neural.py`             | Configurable | Trained neural chess AI           |

## 🎉 **Why This Project is Special**

1. **Dual Approach**: Shows both traditional and modern AI methods
2. **Visual Learning**: Clean, scrollable chess board during training
3. **Educational**: Perfect for learning AI and chess programming
4. **Scalable**: Can be improved with more training/computation
5. **Realistic**: Demonstrates actual AI learning process

## 🚀 **Get Started Now!**

### **🎯 Option 1: Main Launcher (Easiest!)**

```bash
# Install everything
pip install -r requirements.txt

# Run the main launcher
python chess_engine.py
```

### **Option 2: Direct Commands**

```bash
# Install everything
pip install -r requirements.txt

# Quick visual demo (recommended first)
python run_visual.py simple

# Full visual training
python run_visual.py training

# Traditional engine
python run_traditional.py interactive
```

**Watch your neural network transform from knowing nothing about chess to becoming a decent player!** 🧠♟️

---

_From handcrafted rules to self-learning AI with beautiful visual boards - you've got the full spectrum of chess engine technology!_
