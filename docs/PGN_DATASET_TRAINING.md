# 📚 PGN Dataset Training Guide

This guide explains how to use the new PGN Dataset Training feature to train your neural chess engine on existing game databases for much faster learning.

## 🚀 Why PGN Dataset Training?

**Traditional Self-Play Training:**

- Engine plays games against itself
- Generates training data slowly
- Takes many games to learn basic concepts
- Limited by engine's current knowledge

**PGN Dataset Training:**

- Learns from high-quality human/engine games
- Access to millions of positions instantly
- Learns advanced strategies and tactics
- Much faster training progress
- Better final model quality

## 📁 Dataset Structure

Place your PGN files in the `datasets/pgn/` directory:

```
chess-engine/
├── datasets/
│   └── pgn/
│       ├── lichess_db_standard_rated_2014-11.pgn
│       ├── master_games.pgn
│       └── your_custom_games.pgn
```

## 🎯 How to Use

### Option 1: Main Menu

1. Run `python run_neural.py`
2. Choose option `2. PGN Dataset Training (Fast Learning)`
3. Follow the interactive prompts

### Option 2: Direct Script

```bash
python run_pgn_training.py
```

### Option 3: Command Line

```bash
python -m neural.pgn_dataset_trainer --pgn_path datasets/pgn/your_file.pgn --epochs 20 --batch_size 128
```

### Option 4: Demo Mode

```bash
python demo_pgn_training.py
```

## ⚙️ Training Parameters

| Parameter              | Default | Description                                        |
| ---------------------- | ------- | -------------------------------------------------- |
| **Epochs**             | 10      | Number of training passes through the dataset      |
| **Batch Size**         | 64      | Training batch size (larger = faster, more memory) |
| **Learning Rate**      | 0.001   | How fast the model learns                          |
| **Max Games**          | All     | Limit games to load (useful for large datasets)    |
| **Max Positions/Game** | 50      | Limit positions per game to prevent memory issues  |

## 🔧 Advanced Usage

### Continue from Existing Model

```python
from neural.pgn_dataset_trainer import PGNDatasetTrainer

# Continue training from existing model
trainer = PGNDatasetTrainer("models/chess_neural_v1_final.pth")

# Train on dataset
results = trainer.train_on_dataset(
    pgn_path="datasets/pgn/master_games.pgn",
    epochs=15,
    batch_size=128,
    learning_rate=0.0005
)
```

### Custom Training Configuration

```python
# Custom training with validation split
results = trainer.train_on_dataset(
    pgn_path="datasets/pgn/your_data.pgn",
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    max_games=1000,
    max_positions_per_game=100,
    validation_split=0.2,  # 20% for validation
    save_interval=5  # Save every 5 epochs
)
```

### Model Evaluation

```python
# Evaluate trained model
evaluation = trainer.evaluate_model(
    pgn_path="datasets/pgn/test_games.pgn",
    max_games=100
)

print(f"MSE Loss: {evaluation['mse_loss']:.6f}")
print(f"Correlation: {evaluation['correlation']:.4f}")
print(f"Accuracy: {evaluation['accuracy_within_0.1']:.2%}")
```

## 📊 Training Process

### 1. **Data Loading**

- Reads PGN file game by game
- Extracts board positions and game results
- Converts to neural network input format
- Shows progress with loading indicators

### 2. **Data Preparation**

- Converts chess positions to tensors
- Creates training/validation splits
- Applies data augmentation and sampling
- Optimizes memory usage

### 3. **Training Loop**

- Batch processing for efficiency
- Real-time loss monitoring
- Validation on held-out data
- Automatic checkpoint saving

### 4. **Model Saving**

- Periodic checkpoints during training
- Final model with versioning
- Training statistics and metrics
- Ready for immediate use

## 🎮 Expected Results

### Training Speed

- **Self-Play**: 30 games = ~2-3 hours
- **PGN Dataset**: 1000 games = ~15-30 minutes
- **Speed Improvement**: 5-10x faster

### Model Quality

- **Self-Play**: Basic tactical awareness
- **PGN Dataset**: Advanced positional understanding
- **Quality Improvement**: 2-3x better evaluation

### Learning Progress

- **Epoch 1**: Basic piece values and simple tactics
- **Epoch 5**: Position evaluation and basic strategy
- **Epoch 10**: Advanced tactics and endgame knowledge
- **Epoch 20+**: Master-level understanding

## 🚨 Troubleshooting

### Memory Issues

```bash
# Reduce batch size and positions per game
python run_pgn_training.py
# Choose smaller batch size (32 instead of 64)
# Choose fewer positions per game (25 instead of 50)
```

### Slow Training

```bash
# Increase batch size for GPU efficiency
# Use smaller validation split (0.05 instead of 0.1)
# Reduce max positions per game
```

### Poor Results

```bash
# Check PGN file quality
# Increase number of epochs
# Adjust learning rate (try 0.0005 or 0.002)
# Use more games from dataset
```

## 📈 Performance Tips

### For Large Datasets

1. **Start Small**: Begin with 100-1000 games
2. **Gradual Increase**: Add more games as model improves
3. **Checkpoint Often**: Save progress every 5-10 epochs
4. **Monitor Memory**: Watch GPU/CPU memory usage

### For Best Results

1. **Quality Data**: Use high-rated games (2000+ ELO)
2. **Balanced Results**: Mix wins, losses, and draws
3. **Recent Games**: Prefer modern opening theory
4. **Clean PGN**: Remove corrupted or incomplete games

### For Speed

1. **GPU Usage**: Ensure CUDA is available
2. **Batch Size**: Use largest batch size your memory allows
3. **Data Sampling**: Limit positions per game
4. **Validation**: Use smaller validation split

## 🔄 Integration with Self-Play

### Hybrid Training Approach

1. **Phase 1**: Train on PGN dataset (fast learning)
2. **Phase 2**: Continue with self-play (refinement)
3. **Phase 3**: Alternate between both methods

### Example Workflow

```python
# Start with PGN dataset training
trainer = PGNDatasetTrainer()
results = trainer.train_on_dataset("datasets/pgn/master_games.pgn", epochs=20)

# Continue with self-play training
from neural.train_neural_chess import train_neural_chess_engine_parallel
train_neural_chess_engine_parallel(
    num_games=50,
    epochs_per_game=5,
    existing_model_path=results['final_model_path']
)
```

## 📚 Example Datasets

### Free PGN Sources

- **Lichess Database**: High-quality online games
- **Chess.com**: Professional and amateur games
- **FIDE Database**: International tournament games
- **ChessBase**: Professional game collections

### Recommended Starting Datasets

1. **Lichess Rated Games**: Good for general learning
2. **Master Games**: High-quality strategic play
3. **Opening Databases**: Specific opening theory
4. **Endgame Studies**: Endgame technique

## 🎯 Next Steps

After PGN dataset training:

1. **Test the Model**: Play against it to see improvement
2. **Self-Play Refinement**: Continue training with self-play
3. **Evaluate Performance**: Compare with previous models
4. **Iterate**: Use new model as starting point for next training

## 🆘 Getting Help

If you encounter issues:

1. **Check Dependencies**: Ensure all packages are installed
2. **Verify PGN Files**: Check file format and content
3. **Monitor Resources**: Watch memory and GPU usage
4. **Start Small**: Begin with small datasets and parameters
5. **Check Logs**: Look for error messages and warnings

---

**🎮 Ready to accelerate your chess engine training? Start with PGN dataset training today!**
