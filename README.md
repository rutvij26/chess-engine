# 🧠 Neural Chess Engine

A self-learning chess engine that uses neural networks to play chess through self-play learning. The engine automatically improves its play by analyzing games and updating its neural network weights.

## 🚀 Features

- **🧠 Neural Network Learning**: Self-improving AI through self-play
- **🚀 GPU Acceleration**: CUDA support for faster training (RTX 4070 recommended)
- **📊 Self-Play Training**: Automatic game generation and learning
- **💾 Model Checkpointing**: Save and load trained models
- **📜 PGN Generation**: Export games in standard chess notation
- **🎮 Interactive Play**: Play against the trained neural network
- **📁 Organized Storage**: Models and games stored in dedicated directories

## 🎯 Quick Start

### 1. **Test GPU Acceleration**

```bash
python test_gpu.py
```

This will verify your GPU setup and show performance information.

### 2. **Launch the Engine**

```bash
python chess_engine.py
```

Choose from the main menu to train, test, or play.

### 3. **Direct Neural Training**

```bash
python run_neural.py
```

Start neural network training directly.

## 🏗️ Project Structure

```
chess-engine/
├── 🧠 neural/                    # Neural chess engine core
│   ├── neural_chess_engine.py    # Main neural engine
│   ├── train_neural_chess.py     # Training orchestration
│   ├── neural_demo.py            # Demo functionality
│   └── pgn_demo.py               # PGN generation demo
├── 📁 models/                     # Trained neural models (.pth files)
├── 📁 games/                      # PGN game files
├── 📁 docs/                       # Documentation
├── 🚀 run_neural.py              # Neural engine runner
├── 🧪 test_gpu.py                # GPU acceleration test
├── 📖 README.md                  # This file
├── 🚀 GPU_ACCELERATION_README.md # GPU setup guide
└── 🔄 RESTRUCTURE_SUMMARY.md     # Project evolution
```

## 🎮 How It Works

### **Neural Network Architecture**

- **Input**: 8x8x12 tensor representing chess board (6 piece types × 2 colors)
- **Convolutional Layers**: 3 layers with batch normalization and dropout
- **Fully Connected**: 4 layers for position evaluation
- **Output**: Position score between -1000 and +1000

### **Learning Process**

1. **Self-Play**: Engine plays games against itself
2. **Position Collection**: Stores board positions and evaluations
3. **Neural Training**: Updates network weights using collected data
4. **Iterative Improvement**: Repeats process to improve play quality

### **Move Selection**

- **Minimax Search**: Traditional chess search algorithm
- **Neural Evaluation**: Uses trained network for position scoring
- **Alpha-Beta Pruning**: Optimized search with pruning
- **Iterative Deepening**: Progressive depth increase

## 🚀 GPU Acceleration

The engine automatically detects and uses CUDA-capable GPUs:

- **Automatic Detection**: Falls back to CPU if GPU unavailable
- **Memory Management**: Efficient GPU memory usage
- **Performance Boost**: 5-50x faster training and inference
- **RTX 4070 Optimized**: Tested and optimized for your GPU

## 📊 Training Parameters

### **Default Settings**

- **Games per training**: 30 (configurable)
- **Epochs per game**: 3
- **Learning rate**: 0.001
- **Parallel games**: 3 (simultaneous)
- **Save interval**: Every 10 games

### **Customization**

All parameters can be adjusted during training:

- Number of games
- Training epochs
- Learning rate
- Parallel execution

## 🎯 Achievement Levels

| Level            | Description            | Training Time |
| ---------------- | ---------------------- | ------------- |
| **Beginner**     | Basic piece movement   | 0 games       |
| **Novice**       | Simple tactics         | 10 games      |
| **Intermediate** | Position understanding | 30 games      |
| **Advanced**     | Strategic play         | 100 games     |
| **Expert**       | Complex combinations   | 500+ games    |

## 📁 File Organization

### **Models Directory**

- `models/chess_neural_game_X.pth` - Individual game checkpoints
- `models/chess_neural_final.pth` - Final trained model

### **Games Directory**

- `games/game_histories.pgn` - All training games in PGN format

## 🔧 Requirements

### **Python Packages**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install chess matplotlib numpy
```

### **System Requirements**

- **GPU**: NVIDIA GPU with CUDA support (RTX 4070 recommended)
- **Memory**: 8GB+ GPU memory
- **Storage**: 1GB+ free space for models and games
- **OS**: Windows 10/11, Linux, or macOS

## 🚨 Troubleshooting

### **GPU Not Detected**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, install PyTorch with CUDA support.

### **Out of Memory**

- Reduce batch size in training
- Use fewer parallel games
- Monitor GPU memory usage

### **Training Issues**

- Ensure sufficient training data
- Check learning rate settings
- Verify model save paths

## 📚 Documentation

- **📖 README.md** - This comprehensive guide
- **🚀 GPU_ACCELERATION_README.md** - Detailed GPU setup
- **🔄 RESTRUCTURE_SUMMARY.md** - Project evolution history

## 🎉 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Test GPU**: `python test_gpu.py`
4. **Launch engine**: `python chess_engine.py`
5. **Start training**: Choose option 1 from the menu

## 🤝 Contributing

This is a focused neural chess engine project. Contributions should focus on:

- Neural network improvements
- Training optimizations
- GPU acceleration enhancements
- Documentation improvements

## 📄 License

This project is open source and available under the MIT License.

---

**🎮 Ready to train your neural chess engine? Start with `python chess_engine.py` and watch it learn!**
