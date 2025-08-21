"""
Neural Chess Engine Package
Contains the neural network-based chess engine with self-play learning
"""

from .neural_chess_engine import NeuralChessEngine, ChessNeuralNetwork
from .train_neural_chess import train_neural_chess_engine
from .grandmaster_training import GrandmasterTrainer

__all__ = ['NeuralChessEngine', 'ChessNeuralNetwork', 'train_neural_chess_engine', 'GrandmasterTrainer']
