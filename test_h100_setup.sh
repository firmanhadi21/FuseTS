#!/bin/bash
# Quick test script for H100 GPU setup
# Run this interactively to verify everything works

echo "🧪 H100 MOGPR Quick Test"
echo "========================"

# Test 1: Check Python and packages
echo ""
echo "Test 1: Python packages"
python3 << 'EOF'
import sys
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

try:
    import gpytorch
    print(f"✅ GPyTorch: {gpytorch.__version__}")
except ImportError:
    print("❌ GPyTorch not installed")
    sys.exit(1)

try:
    import xarray
    print(f"✅ xarray: {xarray.__version__}")
except ImportError:
    print("❌ xarray not installed")
    sys.exit(1)
EOF

# Test 2: GPU Detection
echo ""
echo "Test 2: GPU Detection"
python3 << 'EOF'
import torch

if not torch.cuda.is_available():
    print("❌ No CUDA GPUs detected!")
    sys.exit(1)

print(f"✅ {torch.cuda.device_count()} GPU(s) detected")

for i in range(min(2, torch.cuda.device_count())):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Check available memory
    torch.cuda.set_device(i)
    free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1e9
    print(f"  Available: {free:.1f} GB")
EOF

# Test 3: GPU Computation
echo ""
echo "Test 3: GPU Computation"
python3 << 'EOF'
import torch
import time

device = torch.device("cuda:0")
print(f"Testing computation on {torch.cuda.get_device_name(0)}...")

# Small matrix multiplication test
sizes = [1000, 5000, 10000]
for size in sizes:
    x = torch.randn(size, size, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    
    y = torch.matmul(x, x.t())
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gflops = (2 * size**3) / elapsed / 1e9
    print(f"  {size}×{size}: {elapsed*1000:.1f}ms ({gflops:.0f} GFLOPS)")

print("✅ GPU computation working!")
EOF

# Test 4: MOGPR Import
echo ""
echo "Test 4: MOGPR Import"
python3 << 'EOF'
import sys
sys.path.insert(0, './src')

try:
    from fusets.mogpr_gpu import MOGPRTransformerGPU
    print("✅ MOGPRTransformerGPU imported successfully")
    
    import torch
    device = torch.device("cuda:0")
    mogpr = MOGPRTransformerGPU(device=device, batch_size=64)
    print("✅ MOGPRTransformerGPU initialized")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure mogpr_gpu.py is in src/fusets/")
    sys.exit(1)
EOF

# Test 5: Small dummy dataset test
echo ""
echo "Test 5: Dummy Dataset Test (30 seconds)"
python3 << 'EOF'
import sys
sys.path.insert(0, './src')

import torch
import xarray as xr
import numpy as np
from fusets.mogpr_gpu import MOGPRTransformerGPU
import time

print("Creating small test dataset (10×10 pixels, 10 timesteps)...")

# Create dummy data
n_time, n_y, n_x = 10, 10, 10
times = np.arange(n_time)

dummy_data = xr.Dataset({
    'VV': (['t', 'y', 'x'], np.random.randn(n_time, n_y, n_x) * 5 - 10),
    'VH': (['t', 'y', 'x'], np.random.randn(n_time, n_y, n_x) * 5 - 15),
    'S2ndvi': (['t', 'y', 'x'], np.random.rand(n_time, n_y, n_x) * 0.8 + 0.1),
}, coords={
    't': times,
    'y': np.arange(n_y),
    'x': np.arange(n_x),
})

# Add some NaNs to simulate gaps
dummy_data['S2ndvi'][3:7, :, :] = np.nan

print(f"Test dataset: {dummy_data.dims}")

# Initialize MOGPR
device = torch.device("cuda:0")
mogpr = MOGPRTransformerGPU(device=device, batch_size=32)

print("Running MOGPR fusion on test data...")
start = time.time()

try:
    fused_result = mogpr.fit_transform(dummy_data)
    elapsed = time.time() - start
    
    print(f"✅ Test passed in {elapsed:.1f} seconds!")
    print(f"   Result shape: {fused_result.dims}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "=================================="
echo "✅ ALL TESTS PASSED!"
echo "=================================="
echo ""
echo "Your H100 GPU setup is ready for MOGPR!"
echo ""
echo "Next steps:"
echo "  1. Upload your dataset (combined_dataset.nc) to HPC"
echo "  2. Run: sbatch run_mogpr_h100.slurm"
echo "  3. Monitor: watch -n 1 nvidia-smi"
