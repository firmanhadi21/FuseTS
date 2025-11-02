#!/bin/bash
# Quick Jupyter launcher for HPC H100

echo "🚀 Jupyter Notebook Launcher for H100 GPUs"
echo "=========================================="
echo ""

# Check if in compute node or login node
if [ -z "$SLURM_JOB_ID" ]; then
    echo "⚠️  You're on a login node (no GPU access)"
    echo ""
    echo "Please request a GPU compute node first:"
    echo ""
    echo "  srun --gres=gpu:1 --mem=64G --time=04:00:00 --cpus-per-task=16 --pty bash"
    echo ""
    read -p "Do you want me to request a GPU node now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Requesting GPU node..."
        srun --gres=gpu:1 --mem=64G --time=04:00:00 --cpus-per-task=16 --pty bash -c "$(realpath $0)"
        exit 0
    else
        echo "Cancelled. Please request GPU node manually."
        exit 1
    fi
fi

# We're on compute node
COMPUTE_NODE=$(hostname)
PORT=8888

# Activate environment
echo "📦 Activating environment..."
if command -v conda &> /dev/null; then
    source activate mogpr_h100
else
    source ~/mogpr_h100_env/bin/activate
fi

# Verify GPU
echo ""
echo "🔍 Checking GPU access..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU detected!")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU not available on this node"
    exit 1
fi

# Get HPC hostname for SSH tunnel instructions
HPC_HOST=$(hostname -f | awk -F. '{print $2"."$3}' | sed 's/^.//' || echo "your_hpc")

echo ""
echo "=========================================="
echo "✅ Starting Jupyter Notebook on GPU Node"
echo "=========================================="
echo ""
echo "📍 Compute Node: $COMPUTE_NODE"
echo "📍 Port: $PORT"
echo ""
echo "To access from your local computer:"
echo ""
echo "1️⃣  Open a NEW terminal on your LOCAL machine"
echo ""
echo "2️⃣  Run this command:"
echo "    ssh -L $PORT:$COMPUTE_NODE:$PORT $USER@$HPC_HOST"
echo ""
echo "3️⃣  Keep that terminal open"
echo ""
echo "4️⃣  Open browser to: http://localhost:$PORT"
echo ""
echo "=========================================="
echo ""

# Start Jupyter
cd ~/FuseTS || cd ~
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
