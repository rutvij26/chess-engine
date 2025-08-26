#!/usr/bin/env python3
"""
GPU Acceleration Test Script
Tests if CUDA/GPU acceleration is working for the neural chess engine
"""

import torch
import sys
import os

# Add the neural directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural'))

def test_gpu():
    """Test GPU acceleration"""
    print("🚀 GPU Acceleration Test")
    print("=" * 40)
    
    # Test PyTorch CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test tensor operations on GPU
        print("\n🧪 Testing GPU tensor operations...")
        device = torch.device("cuda")
        
        # Create test tensors
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # Test matrix multiplication on GPU
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.mm(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time)
        
        print(f"GPU matrix multiplication time: {gpu_time:.2f} ms")
        print(f"Result tensor shape: {z.shape}")
        print(f"Result tensor device: {z.device}")
        
        # Test CPU for comparison
        print("\n🖥️  Testing CPU tensor operations...")
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        import time
        start_time_cpu = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        end_time_cpu = time.time()
        
        cpu_time = (end_time_cpu - start_time_cpu) * 1000
        print(f"CPU matrix multiplication time: {cpu_time:.2f} ms")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"🚀 GPU is {speedup:.1f}x faster than CPU!")
        
    else:
        print("❌ CUDA not available. Using CPU only.")
        print("Make sure you have:")
        print("1. NVIDIA GPU with CUDA support")
        print("2. CUDA toolkit installed")
        print("3. PyTorch with CUDA support")
    
    # Test neural chess engine
    print("\n🧠 Testing Neural Chess Engine...")
    try:
        from neural.neural_chess_engine import NeuralChessEngine
        
        # Create engine (should show GPU info)
        engine = NeuralChessEngine()
        
        # Test a simple evaluation
        import chess
        board = chess.Board()
        evaluation = engine.evaluate_position_neural(board)
        print(f"Position evaluation: {evaluation:.2f}")
        
        print("✅ Neural chess engine GPU test successful!")
        
    except Exception as e:
        print(f"❌ Neural chess engine test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu()
