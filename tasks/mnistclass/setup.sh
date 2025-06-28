#!/bin/bash

# MNIST Recognition Project Setup Script

echo "🚀 Setting up MNIST Recognition Project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Initialize project
echo "🏗️ Initializing project..."
uv sync

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p ray_results

# Check Python version
echo "🐍 Python version:"
uv run python --version

# Verify installations
echo "✅ Verifying installations..."

echo "Checking PyTorch..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "Checking CUDA availability..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Checking other dependencies..."
uv run python -c "
import wandb
import hydra
import ray
from torchinfo import summary
print('✅ All dependencies installed successfully!')
"

# Test model creation
echo "🧪 Testing model creation..."
uv run python -c "
from src.model import LightweightCNN
model = LightweightCNN()
params = model.count_parameters()
print(f'Model parameters: {params:,}')
if params < 10000:
    print('✅ Parameter constraint satisfied!')
else:
    print('❌ Too many parameters!')
"

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up wandb: uv run wandb login"
echo "2. Train model: uv run python train.py"
echo "3. Evaluate: uv run python evaluate.py"
echo "4. Tune hyperparameters: uv run python tune.py"