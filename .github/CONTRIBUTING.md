# 🤝 Contributing to Chess Engine

Thank you for your interest in contributing to the Chess Engine project! This document provides guidelines and information for contributors.

## 🎯 Project Overview

The Chess Engine is a sophisticated neural network-based chess engine that learns through self-play, featuring:

- **Incremental Learning**: Model versioning and continuation
- **Parallel Training**: Multi-game simultaneous execution
- **PGN Generation**: Standard chess notation export
- **Progress Tracking**: Real-time training monitoring
- **Robust Cleanup**: Graceful shutdown and resource management

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8 or higher
- Git
- Basic understanding of chess and neural networks
- Familiarity with PyTorch (for neural engine contributions)

### **Setup Development Environment**

1. **Fork the repository**

   ```bash
   git clone https://github.com/your-username/chess-engine.git
   cd chess-engine
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

### **1. Create a Feature Branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### **2. Make Your Changes**

- Follow the coding standards (see below)
- Write tests for new functionality
- Update documentation as needed

### **3. Test Your Changes**

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_neural_engine.py

# Run with coverage
pytest tests/ --cov=neural --cov=traditional --cov=visual

# Run linting
flake8 neural/ traditional/ visual/ tests/
black --check neural/ traditional/ visual/ tests/
isort --check-only neural/ traditional/ visual/ tests/
```

### **4. Commit Your Changes**

```bash
git add .
git commit -m "feat: add new training visualization feature

- Add real-time training progress charts
- Implement performance metrics dashboard
- Update documentation with usage examples

Closes #123"
```

### **5. Push and Create Pull Request**

```bash
git push origin feature/your-feature-name
```

## 📝 Coding Standards

### **Python Style Guide**

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use descriptive variable and function names

### **Code Organization**

```
neural/
├── __init__.py
├── neural_chess_engine.py      # Core neural engine
├── train_neural_chess.py       # Training orchestration
└── neural_demo.py              # Demo scripts

traditional/
├── __init__.py
├── chess_engine.py             # Traditional engine
├── demo.py                     # Demo scripts
└── uci_handler.py              # UCI protocol

visual/
├── __init__.py
├── visual_chess_board.py       # Visual board
├── visual_training.py          # Visual training
└── demo scripts
```

### **Documentation Standards**

- All public functions must have docstrings
- Use Google-style docstrings
- Include examples for complex functions
- Update README.md for user-facing changes

### **Example Function Documentation**

```python
def train_neural_chess_engine_parallel(
    num_games: int = 100,
    epochs_per_game: int = 3,
    learning_rate: float = 0.001,
    model_name: str = "chess_neural_model",
    num_parallel_games: int = 3,
    existing_model_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Train the neural chess engine through parallel self-play.

    This function orchestrates parallel training of the neural chess engine
    by running multiple games simultaneously and training the model on the
    collected data.

    Args:
        num_games: Total number of games to play for training
        epochs_per_game: Number of training epochs per game
        learning_rate: Learning rate for the neural network optimizer
        model_name: Base name for saving trained models
        num_parallel_games: Number of games to run in parallel
        existing_model_path: Path to existing model for incremental training

    Returns:
        List of dictionaries containing training results for each game

    Raises:
        RuntimeError: If training fails due to insufficient resources
        ValueError: If training parameters are invalid

    Example:
        >>> results = train_neural_chess_engine_parallel(
        ...     num_games=30,
        ...     epochs_per_game=5,
        ...     learning_rate=0.001,
        ...     num_parallel_games=3
        ... )
        >>> print(f"Training completed with {len(results)} games")
        Training completed with 30 games
    """
```

## 🧪 Testing Guidelines

### **Test Structure**

```
tests/
├── __init__.py
├── test_neural_engine.py       # Neural engine tests
├── test_traditional_engine.py  # Traditional engine tests
├── test_visual_interface.py    # Visual interface tests
├── test_training.py            # Training pipeline tests
└── conftest.py                 # Test configuration and fixtures
```

### **Test Requirements**

- **Coverage**: Aim for 90%+ code coverage
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test training speed and memory usage
- **Edge Cases**: Test error conditions and boundary cases

### **Example Test**

```python
import pytest
from neural.neural_chess_engine import NeuralChessEngine

class TestNeuralChessEngine:
    """Test suite for NeuralChessEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine instance for each test."""
        return NeuralChessEngine()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.board is not None
        assert engine.model is not None
        assert engine.optimizer is not None

    def test_board_to_tensor_conversion(self, engine):
        """Test board to tensor conversion."""
        tensor = engine.board_to_tensor(engine.board)
        assert tensor.shape == (1, 12, 8, 8)  # Expected shape
        assert tensor.dtype == torch.float32

    @pytest.mark.parametrize("depth,expected_moves", [
        (1, 20),  # Starting position has 20 legal moves
        (2, 400), # Depth 2 has ~400 positions
    ])
    def test_search_depth(self, engine, depth, expected_moves):
        """Test search at different depths."""
        best_move = engine.get_best_move(depth, 2.0)
        assert best_move is not None
        assert engine.board.is_legal(best_move)
```

## 🔍 Code Review Process

### **Pull Request Requirements**

- **Description**: Clear description of changes
- **Testing**: Evidence that changes work correctly
- **Documentation**: Updated documentation and examples
- **Code Quality**: Follows project coding standards
- **Performance**: No significant performance regressions

### **Review Checklist**

- [ ] Code follows project style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

## 🚨 Common Issues and Solutions

### **Training Pipeline Issues**

- **Problem**: Training hangs or gets stuck
- **Solution**: Check CUDA memory, reduce parallel games, add timeouts

### **Model Loading Issues**

- **Problem**: Can't load existing models
- **Solution**: Verify model file integrity, check PyTorch version compatibility

### **Performance Issues**

- **Problem**: Training is too slow
- **Solution**: Enable GPU acceleration, optimize batch sizes, reduce model complexity

## 📚 Learning Resources

### **Chess Engine Development**

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [Stockfish Documentation](https://stockfishchess.org/documentation/)
- [UCI Protocol Specification](https://www.chessprogramming.org/UCI)

### **Neural Networks and PyTorch**

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning for Chess](https://arxiv.org/abs/1711.09667)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

### **Python Development**

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Testing with pytest](https://pytest.org/)

## 🤝 Community Guidelines

### **Communication**

- Be respectful and constructive
- Use clear, descriptive language
- Provide context for questions and suggestions
- Help other contributors when possible

### **Issue Reporting**

- Use appropriate issue templates
- Provide detailed reproduction steps
- Include system information and error messages
- Search existing issues before creating new ones

### **Feature Requests**

- Explain the problem you're solving
- Provide use cases and examples
- Consider implementation complexity
- Be open to alternative solutions

## 🏆 Recognition

### **Contributor Levels**

- **Contributor**: First successful contribution
- **Regular Contributor**: Multiple contributions over time
- **Core Contributor**: Significant contributions and project knowledge
- **Maintainer**: Project leadership and decision-making

### **Contributions We Value**

- Bug fixes and improvements
- New features and enhancements
- Documentation and examples
- Testing and quality assurance
- Community support and mentoring

## 📞 Getting Help

### **Support Channels**

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions
- **Email**: For sensitive or private matters

### **Before Asking for Help**

- Check existing documentation
- Search GitHub issues and discussions
- Try to reproduce the problem
- Gather relevant system information
- Prepare a clear, detailed description

## 🎉 Thank You!

Thank you for contributing to the Chess Engine project! Your contributions help make this project better for everyone in the chess and AI communities.

---

_This document is a living guide. If you have suggestions for improvements, please submit a pull request or open an issue._
