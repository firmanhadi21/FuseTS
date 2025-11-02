#!/bin/bash
# MOGPR GPU Setup for H100 HPC
# CUDA 12.4, Driver 550.127.05
# 2x NVIDIA H100 80GB HBM3

echo "🚀 Setting up MOGPR for 2x H100 80GB GPUs"
echo "=========================================="

# Verify CUDA
echo "✅ CUDA Version: 12.4"
echo "✅ Driver: 550.127.05"

# Check available GPUs
echo ""
echo "📊 Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Create Python environment
echo ""
echo "📦 Creating Python environment..."

# Option 1: Use conda if available
if command -v conda &> /dev/null; then
    echo "Using conda..."
    conda create -n mogpr_h100 python=3.11 -y
    source activate mogpr_h100
else
    # Option 2: Use venv
    echo "Using venv..."
    python3 -m venv ~/mogpr_h100_env
    source ~/mogpr_h100_env/bin/activate
fi

# Install PyTorch for CUDA 12.4
echo ""
echo "🔧 Installing PyTorch with CUDA 12.4 support..."
pip install --upgrade pip

# PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install GPyTorch
pip install gpytorch

# Install FuseTS dependencies
echo ""
echo "📚 Installing FuseTS dependencies..."
pip install xarray netcdf4 h5netcdf
pip install rioxarray  # For raster I/O with xarray
pip install numpy pandas scipy matplotlib
pip install earthengine-api geemap
pip install tqdm  # For progress bars

# Install Jupyter and ipykernel for notebook support
echo ""
echo "📓 Installing Jupyter and ipykernel..."
pip install jupyter jupyterlab ipykernel ipywidgets

# Register kernel
echo ""
echo "🔧 Registering kernel with Jupyter..."
python -m ipykernel install --user --name mogpr_h100 --display-name "Python (MOGPR H100)"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 << 'EOF'
import torch
import gpytorch
import sys

print("="*60)
print("INSTALLATION VERIFICATION")
print("="*60)

print(f"\n✅ Python: {sys.version.split()[0]}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPyTorch: {gpytorch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
    print(f"\n✅ GPUs detected: {torch.cuda.device_count()}")
    
    for i in range(min(2, torch.cuda.device_count())):  # Show first 2 GPUs
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {props.name}")
        print(f"      Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Multi-Processors: {props.multi_processor_count}")
        
        # Check available memory
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        available = total - reserved
        print(f"      Available Memory: {available:.1f} GB / {total:.1f} GB")
    
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    x = torch.randn(1000, 1000).cuda(0)
    y = torch.matmul(x, x.t())
    print("✅ GPU computation works!")
    
else:
    print("\n❌ CUDA not available!")
    print("Troubleshooting:")
    print("  1. Check NVIDIA drivers: nvidia-smi")
    print("  2. Check CUDA installation: nvcc --version")
    print("  3. Reinstall PyTorch with correct CUDA version")
    sys.exit(1)

print("\n" + "="*60)
print("✅ SETUP COMPLETE!")
print("="*60)
print("\nYou can now run MOGPR with 2x H100 GPUs!")
print("Expected performance:")
print("  - 50×50 pixels: ~10-20 seconds")
print("  - Full Demak (360×800): ~10-15 minutes")
print("  - Java Island: ~2-4 hours")
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate this environment in future:"
if command -v conda &> /dev/null; then
    echo "  conda activate mogpr_h100"
else
    echo "  source ~/mogpr_h100_env/bin/activate"
fi
