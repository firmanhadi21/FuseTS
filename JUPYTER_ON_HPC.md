# Running Jupyter Notebook on HPC with H100 GPUs

## 🎯 Two Methods to Run Jupyter on HPC

### **Method 1: SSH Tunnel (Recommended - Most Secure)**

This method runs Jupyter on HPC and accesses it from your local browser.

#### Step 1: Start Jupyter on HPC with GPU

```bash
# SSH to HPC
ssh username@your_hpc

# Request GPU node (interactive)
srun --gres=gpu:1 --mem=64G --time=04:00:00 --cpus-per-task=16 --pty bash

# Activate environment
source ~/mogpr_h100_env/bin/activate
# OR: conda activate mogpr_h100

# Start Jupyter (note the port and hostname)
jupyter notebook --no-browser --port=8888
```

You'll see output like:
```
[I 10:30:45.123 NotebookApp] Jupyter Notebook 6.5.4 is running at:
[I 10:30:45.123 NotebookApp] http://compute-node-123:8888/?token=abc123def456...
```

**Note the hostname (`compute-node-123`) and token!**

#### Step 2: Create SSH Tunnel (From Your Local Machine)

Open a **NEW terminal on your local Mac**:

```bash
# General format:
ssh -L 8888:compute-node-hostname:8888 username@your_hpc

# Example:
ssh -L 8888:compute-node-123:8888 username@hpc.university.edu
```

Keep this terminal open!

#### Step 3: Open in Browser

On your local Mac, open browser and go to:
```
http://localhost:8888
```

Paste the token when prompted, or use the full URL with token from Step 1.

**You now have Jupyter running on H100 GPU, accessed from your local browser!** 🎉

---

### **Method 2: JupyterLab on HPC Login Node (Quick Testing)**

For quick tests (not for heavy computation):

```bash
# SSH to HPC
ssh username@your_hpc

# Activate environment
source ~/mogpr_h100_env/bin/activate

# Start JupyterLab
jupyter lab --port=8889 --no-browser
```

Then SSH tunnel from local Mac:
```bash
ssh -L 8889:localhost:8889 username@your_hpc
```

Open browser: `http://localhost:8889`

⚠️ **Warning**: This runs on login node (no GPU). Only for light work!

---

## 🚀 Method 3: SLURM Batch Job with Jupyter

For long-running notebook sessions:

Create `jupyter_h100.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=jupyter_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=jupyter_%j.out

# Get the hostname and port
HOSTNAME=$(hostname)
PORT=8888

# Activate environment
if command -v conda &> /dev/null; then
    source activate mogpr_h100
else
    source ~/mogpr_h100_env/bin/activate
fi

echo "=========================================="
echo "Jupyter Server Starting"
echo "=========================================="
echo "Hostname: $HOSTNAME"
echo "Port: $PORT"
echo ""
echo "To connect from your local machine, run:"
echo "  ssh -L $PORT:$HOSTNAME:$PORT $USER@$(hostname -f | cut -d. -f2-)"
echo ""
echo "Then open browser: http://localhost:$PORT"
echo "=========================================="

# Start Jupyter
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
```

Submit:
```bash
sbatch jupyter_h100.slurm
```

Check output for connection instructions:
```bash
cat jupyter_*.out
```

---

## 📓 Using Your MOGPR Notebook on HPC

### Step 1: Upload Notebook

```bash
# From local Mac
scp S1_S2_MOGPR_Fusion_Tutorial.ipynb username@hpc:/path/to/FuseTS/
```

### Step 2: Select Kernel in Jupyter

1. Open the notebook in Jupyter
2. Click **Kernel** → **Change kernel** → **Python (MOGPR H100)**
3. You should see the kernel name at top-right

### Step 3: Verify GPU Access

Add a test cell at the top:

```python
# Verify GPU access
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ No GPU detected!")
```

### Step 4: Update GPU Settings

Find the cell with `USE_GPU = False` and change to:

```python
USE_GPU = True  # Enable GPU on HPC!

if USE_GPU:
    import torch
    import gpytorch
    from fusets.mogpr_gpu import MOGPRTransformerGPU
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU requested but not available, falling back to CPU")
        USE_GPU = False
```

### Step 5: Run!

Run your notebook cells normally. MOGPR will use the H100 GPU!

---

## 🔧 Complete Setup Script

Update your `setup_h100_mogpr.sh` to include Jupyter:

```bash
#!/bin/bash
# Already updated in the setup script!

# The setup script now installs:
# - jupyter
# - jupyterlab  
# - ipykernel
# - ipywidgets

# And registers the kernel automatically
```

After running setup, verify kernel installation:

```bash
jupyter kernelspec list
```

Should show:
```
Available kernels:
  mogpr_h100    /home/username/.local/share/jupyter/kernels/mogpr_h100
  python3       /usr/local/share/jupyter/kernels/python3
```

---

## 🎨 VSCode Remote SSH (Alternative - Easiest!)

If you use **VSCode**, this is the **easiest** method:

### Step 1: Install VSCode Extensions (Local Mac)

1. Install "Remote - SSH" extension
2. Install "Jupyter" extension

### Step 2: Connect to HPC

1. Press `Cmd+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Enter: `username@your_hpc`
4. Wait for connection...

### Step 3: Open Notebook

1. Open folder: `/path/to/FuseTS`
2. Open `S1_S2_MOGPR_Fusion_Tutorial.ipynb`
3. Select kernel: **Python (MOGPR H100)**
4. Run cells!

VSCode will:
- ✅ Handle SSH tunneling automatically
- ✅ Show GPU usage in terminal
- ✅ Provide autocomplete and debugging
- ✅ Let you edit files directly on HPC

**This is my recommended approach!** 🌟

---

## 📊 Monitoring GPU Usage

### In Jupyter Notebook

Add this cell to monitor GPU:

```python
# GPU monitoring cell
import torch
import subprocess

def show_gpu_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Total: {total:.2f} GB")
            print(f"  Free: {total - reserved:.2f} GB")
    else:
        print("No GPU available")

# Call it
show_gpu_usage()
```

### In Terminal

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or get specific info
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

---

## 🐛 Troubleshooting

### Issue: Kernel dies when running MOGPR

**Solution**: Increase memory allocation:

```bash
srun --gres=gpu:1 --mem=128G --time=04:00:00 --cpus-per-task=32 --pty bash
```

### Issue: "Kernel not found"

**Solution**: Re-register kernel:

```bash
python -m ipykernel install --user --name mogpr_h100 --display-name "Python (MOGPR H100)"
```

### Issue: SSH tunnel disconnects

**Solution**: Use `tmux` or `screen`:

```bash
# Start tmux session
tmux new -s jupyter

# Start Jupyter
jupyter notebook --no-browser --port=8888

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t jupyter
```

### Issue: Port 8888 already in use

**Solution**: Use different port:

```bash
jupyter notebook --no-browser --port=8889
# Then SSH tunnel: ssh -L 8889:compute-node:8889 ...
```

---

## 📋 Quick Reference

### Start Jupyter on GPU Node:
```bash
srun --gres=gpu:1 --mem=64G --time=04:00:00 --pty bash
source ~/mogpr_h100_env/bin/activate
jupyter notebook --no-browser --port=8888
```

### SSH Tunnel from Local Mac:
```bash
ssh -L 8888:compute-node-XXX:8888 username@hpc
```

### Open in Browser:
```
http://localhost:8888
```

### Check Kernel:
```bash
jupyter kernelspec list
```

### Monitor GPU:
```bash
watch -n 1 nvidia-smi
```

---

## ✅ Recommended Workflow

1. **VSCode Remote SSH** (easiest for development)
   - Best for: Interactive development, debugging
   - Run on: GPU compute node via `srun`

2. **Jupyter via SSH Tunnel** (good for long sessions)
   - Best for: Long-running analyses
   - Run on: GPU compute node via SLURM batch job

3. **SLURM Script** (best for production)
   - Best for: Final production runs
   - Run on: Batch queue
   - Use: `run_mogpr_h100.slurm`

---

**Happy GPU Computing!** 🚀

Your H100 GPUs are 80-120x faster than CPU - enjoy the speedup!
