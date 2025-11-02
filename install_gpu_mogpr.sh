#!/bin/bash
# GPU MOGPR Installation Script for FuseTS
# Run with: bash install_gpu_mogpr.sh

echo "🚀 Installing GPU-accelerated MOGPR for FuseTS"
echo "================================================"

# Detect system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "✅ Apple Silicon (M1/M2/M3) detected"
        echo "Installing PyTorch with MPS support..."
        pip install torch torchvision torchaudio
        pip install gpytorch
    else:
        echo "⚠️  Intel Mac - GPU acceleration not available"
        echo "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio
        pip install gpytorch
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name --format=csv,noheader
        
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install gpytorch
    else
        echo "⚠️  No NVIDIA GPU detected"
        echo "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio
        pip install gpytorch
    fi
else
    echo "❓ Unknown OS: $OSTYPE"
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
    pip install gpytorch
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 << EOF
import torch
import gpytorch

print("✅ PyTorch installed:", torch.__version__)
print("✅ GPyTorch installed:", gpytorch.__version__)

if torch.cuda.is_available():
    print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    print("✅ Apple Silicon GPU (MPS) detected")
else:
    print("ℹ️  No GPU detected - CPU mode only")
    
print("\n🎉 Installation complete!")
print("Set USE_GPU = True in your notebook to enable GPU acceleration")
EOF
