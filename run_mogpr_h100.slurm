#!/bin/bash
#SBATCH --job-name=mogpr_h100
#SBATCH --gres=gpu:2              # Request 2 H100 GPUs
#SBATCH --cpus-per-task=32        # 16 CPUs per GPU
#SBATCH --mem=128G                # 128 GB RAM
#SBATCH --time=02:00:00           # 2 hour time limit
#SBATCH --output=mogpr_%j.out
#SBATCH --error=mogpr_%j.err

# Job configuration for 2x NVIDIA H100 80GB

echo "=================================="
echo "MOGPR GPU Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================="

# Activate environment
# Adjust path based on your setup
if command -v conda &> /dev/null; then
    source activate mogpr_h100
else
    source ~/mogpr_h100_env/bin/activate
fi

# Show GPU allocation
echo ""
echo "📊 Allocated GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run MOGPR processing
echo ""
echo "🚀 Starting MOGPR processing..."
echo ""

python3 << 'PYTHON_SCRIPT'
import torch
import xarray as xr
import numpy as np
import time
from datetime import datetime

# Import MOGPR (adjust import based on your setup)
import sys
sys.path.insert(0, './src')
from fusets.mogpr_gpu import MOGPRTransformerGPU

print("="*70)
print(f"MOGPR GPU Processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# GPU Information
print(f"\n✅ GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {props.name}")
    print(f"      Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Current memory usage
    torch.cuda.set_device(i)
    allocated = torch.cuda.memory_allocated(i) / 1e9
    cached = torch.cuda.memory_reserved(i) / 1e9
    free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1e9
    print(f"      Free: {free:.1f} GB")

# Configuration
USE_GPU = 0  # Use first H100
BATCH_SIZE = 512  # Large batch for H100's 80GB memory

print(f"\n⚙️  Configuration:")
print(f"   GPU Device: {USE_GPU}")
print(f"   Batch Size: {BATCH_SIZE}")

# Load dataset
print(f"\n📂 Loading dataset...")
dataset_path = 'combined_dataset.nc'  # Adjust to your data path

try:
    combined_dataset = xr.open_dataset(dataset_path)
    print(f"✅ Dataset loaded: {combined_dataset.dims}")
    
    # Show data variables
    print(f"   Variables: {list(combined_dataset.data_vars)}")
    
    # Calculate dataset size
    total_pixels = combined_dataset.dims['y'] * combined_dataset.dims['x']
    total_timesteps = combined_dataset.dims['t']
    data_size_gb = (total_pixels * total_timesteps * len(combined_dataset.data_vars) * 4) / 1e9
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Timesteps: {total_timesteps}")
    print(f"   Data size: ~{data_size_gb:.2f} GB")
    
except FileNotFoundError:
    print(f"❌ Dataset not found: {dataset_path}")
    print("Please ensure your data is uploaded to the HPC")
    sys.exit(1)

# Initialize MOGPR
print(f"\n🚀 Initializing MOGPR on H100...")
device = torch.device(f"cuda:{USE_GPU}")
mogpr = MOGPRTransformerGPU(device=device, batch_size=BATCH_SIZE)

# Process
print(f"\n⚡ Running MOGPR fusion...")
print(f"Expected time: ~{total_pixels / 10000:.0f}-{total_pixels / 5000:.0f} minutes")
print("-" * 70)

start_time = time.time()

try:
    fused_result = mogpr.fit_transform(combined_dataset)
    
    elapsed = time.time() - start_time
    pixels_per_sec = total_pixels / elapsed
    
    print("-" * 70)
    print(f"\n✅ MOGPR fusion completed!")
    print(f"   Processing time: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
    print(f"   Speed: {pixels_per_sec:.0f} pixels/second")
    print(f"   Speed: {pixels_per_sec * 60:.0f} pixels/minute")
    
    # GPU memory stats
    for i in [USE_GPU]:
        torch.cuda.set_device(i)
        peak_memory = torch.cuda.max_memory_allocated(i) / 1e9
        print(f"   Peak GPU {i} memory: {peak_memory:.2f} GB")
    
    # Save results
    output_file = f'fused_result_h100_{datetime.now().strftime("%Y%m%d_%H%M%S")}.nc'
    print(f"\n💾 Saving results to: {output_file}")
    
    fused_result.to_netcdf(output_file)
    
    import os
    file_size = os.path.getsize(output_file) / 1e9
    print(f"✅ Saved! File size: {file_size:.2f} GB")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n❌ Error after {elapsed/60:.1f} minutes:")
    print(f"   {type(e).__name__}: {e}")
    
    import traceback
    print("\n📋 Full traceback:")
    traceback.print_exc()
    
    sys.exit(1)

finally:
    # Cleanup
    torch.cuda.empty_cache()
    print("\n🧹 GPU cache cleared")

PYTHON_SCRIPT

echo ""
echo "=================================="
echo "Job completed: $(date)"
echo "=================================="

# Show final GPU state
nvidia-smi
