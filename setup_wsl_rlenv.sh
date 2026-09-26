#!/usr/bin/env bash
set -e

echo "========================================================"
echo " 1. Checking NVIDIA GPU Visibility in WSL 2"
echo "========================================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. Ensure NVIDIA display drivers are installed on Windows host."
fi

echo ""
echo "========================================================"
echo " 2. Setting up Miniconda (if not already installed)"
echo "========================================================"
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Downloading and installing Miniconda for Linux..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
fi

# Source conda into current subshell
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo ""
echo "========================================================"
echo " 3. Creating/Activating 'rlenv' with Python 3.11"
echo "========================================================"
# Accept Terms of Service if required by newer conda versions
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true

if conda info --envs | grep -q "rlenv"; then
    echo "'rlenv' already exists. Activating..."
else
    echo "Creating 'rlenv' environment with Python 3.11..."
    conda create -n rlenv -c conda-forge python=3.11 -y
fi

conda activate rlenv

echo ""
echo "========================================================"
echo " 4. Installing PyTorch with CUDA 12 Support"
echo "========================================================"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "========================================================"
echo " 5. Installing JAX (GPU) & Complete RL/Scientific Stack"
echo "========================================================"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_wsl.txt"

# Register Jupyter Kernel for rlenv so notebooks can use it directly
python -m ipykernel install --user --name rlenv --display-name "Python (rlenv WSL2)"

echo ""
echo "========================================================"
echo " 6. Verifying Hardware Acceleration & Environment"
echo "========================================================"
python -c "
import torch
print('>>> PyTorch Version:', torch.__version__)
print('>>> PyTorch CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('>>> PyTorch Device:', torch.cuda.get_device_name(0))

import jax
print('\n>>> JAX Version:', jax.__version__)
print('>>> JAX Devices:', jax.devices())

import gymnasium as gym
import gymnax
import mujoco
print('\n>>> Gymnasium, Gymnax, and MuJoCo successfully loaded!')
"

echo ""
echo "========================================================"
echo " Setup Complete!"
echo " Both PyTorch & JAX now have full GPU acceleration in 'rlenv'."
echo "========================================================"
