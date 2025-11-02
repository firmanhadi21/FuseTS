# VSCode Remote SSH Setup for HPC

## 🎯 Easiest Way to Run Jupyter on HPC

VSCode's Remote SSH is the **easiest and most powerful** way to work with Jupyter on HPC!

## 📦 Prerequisites (One-Time Setup)

### On Your Local Mac:

1. **Install VSCode**: https://code.visualstudio.com/

2. **Install Extensions** (in VSCode):
   - Press `Cmd+Shift+X`
   - Install:
     - "Remote - SSH" by Microsoft
     - "Python" by Microsoft  
     - "Jupyter" by Microsoft

3. **Configure SSH** (optional but recommended):

Edit `~/.ssh/config` on your Mac:

```ssh-config
Host hpc
    HostName your_hpc_address.edu
    User your_username
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Now you can connect with just: `ssh hpc`

## 🚀 Step-by-Step Guide

### 1. Connect to HPC

In VSCode:
1. Press `Cmd+Shift+P`
2. Type: "Remote-SSH: Connect to Host"
3. Select your HPC (or enter manually)
4. New VSCode window opens connected to HPC!

### 2. Request GPU Node

In VSCode's integrated terminal:

```bash
# Request interactive GPU node
srun --gres=gpu:1 --mem=64G --time=04:00:00 --cpus-per-task=16 --pty bash

# Activate environment
source ~/mogpr_h100_env/bin/activate

# Stay in this terminal!
```

### 3. Open Notebook

1. **File** → **Open Folder** → `/path/to/FuseTS`
2. Click on `S1_S2_MOGPR_Fusion_Tutorial.ipynb`
3. Top-right: Click kernel selector
4. Choose: **Python (MOGPR H100)**
5. Run cells! 🎉

## ✨ Why VSCode Remote is Amazing

✅ **No SSH tunnel needed** - works automatically  
✅ **GPU access** - runs on compute node  
✅ **File editing** - edit files directly on HPC  
✅ **Integrated terminal** - run commands alongside notebook  
✅ **Autocomplete** - full IntelliSense on HPC code  
✅ **Debugging** - set breakpoints, inspect variables  
✅ **Git integration** - commit changes on HPC  
✅ **Extensions** - all your VSCode extensions work!  

## 📊 Monitoring GPU in VSCode

### Method 1: Terminal Command

Open terminal in VSCode (`` Ctrl+` ``):

```bash
watch -n 1 nvidia-smi
```

### Method 2: Notebook Cell

Add monitoring cell:

```python
import torch
import IPython

def monitor_gpu():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {allocated:.1f}/{total:.1f} GB ({allocated/total*100:.1f}%)")
    
# Auto-refresh every 2 seconds
from IPython.display import clear_output
import time

for _ in range(10):  # Monitor for 20 seconds
    clear_output(wait=True)
    monitor_gpu()
    time.sleep(2)
```

## 🎨 Recommended VSCode Layout

```
┌─────────────────────────────────────────────┐
│ File Explorer    │ Notebook                 │
│                  │                          │
│ FuseTS/          │ Cell 1: GPU Check       │
│ ├── src/         │ Cell 2: Load Data       │
│ ├── notebooks/   │ Cell 3: MOGPR Fusion    │
│ └── data/        │ Cell 4: Visualization   │
│                  │                          │
├──────────────────┴──────────────────────────┤
│ Terminal: watch nvidia-smi                  │
└─────────────────────────────────────────────┘
```

To arrange:
1. Open notebook
2. `` Ctrl+` `` to open terminal
3. **View** → **Appearance** → **Toggle Side Panel**

## 🔧 Advanced: Port Forwarding

If you need to access other services (TensorBoard, web apps):

In VSCode:
1. Press `Cmd+Shift+P`
2. "Forward a Port"
3. Enter port number (e.g., 6006 for TensorBoard)
4. Access at `http://localhost:6006`

## 🐛 Troubleshooting

### Issue: "Kernel not found"

In VSCode terminal:
```bash
source ~/mogpr_h100_env/bin/activate
python -m ipykernel install --user --name mogpr_h100 --display-name "Python (MOGPR H100)"
```

Then restart VSCode.

### Issue: "No GPU detected"

Make sure you're on a **compute node**, not login node:

```bash
# Check if on compute node
echo $SLURM_JOB_ID

# If empty, you're on login node - request GPU:
srun --gres=gpu:1 --mem=64G --time=04:00:00 --pty bash
```

### Issue: Connection keeps dropping

Add to `~/.ssh/config` on your Mac:

```ssh-config
Host hpc
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
```

### Issue: Slow to connect

Enable SSH multiplexing in `~/.ssh/config`:

```ssh-config
Host hpc
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
```

## 🎯 Complete Workflow Example

### Day 1: Setup (5 minutes)

```bash
# On HPC
cd ~
git clone https://github.com/firmanhadi21/FuseTS.git
cd FuseTS
./setup_h100_mogpr.sh
```

### Day 2: Development

1. Open VSCode on Mac
2. Connect to HPC via Remote SSH
3. Request GPU node in terminal
4. Open notebook
5. Select kernel: Python (MOGPR H100)
6. Run cells with GPU!

### Day 3: Production

Use SLURM batch job for large-scale runs:

```bash
sbatch run_mogpr_h100.slurm
```

Monitor progress:
```bash
tail -f mogpr_*.out
```

## 📚 Keyboard Shortcuts (in Jupyter)

| Action | Shortcut |
|--------|----------|
| Run cell | `Shift+Enter` |
| Run cell, insert below | `Alt+Enter` |
| Insert cell above | `A` (command mode) |
| Insert cell below | `B` (command mode) |
| Delete cell | `DD` (command mode) |
| Toggle line numbers | `L` (command mode) |
| Show autocomplete | `Tab` |
| Show docs | `Shift+Tab` |

## 🔥 Pro Tips

1. **Save frequently** - HPC jobs can timeout
2. **Use checkpoints** - Save intermediate results
3. **Monitor memory** - H100 has 80GB, but don't max it out
4. **Use tmux** - Keep sessions alive if connection drops
5. **Git commits** - Commit working code before big changes

## 🎉 Summary

**Best setup for HPC Jupyter:**

1. ✅ **VSCode Remote SSH** - Easiest, most powerful
2. ✅ **Interactive GPU node** - `srun --gres=gpu:1`
3. ✅ **MOGPR H100 kernel** - GPU-accelerated
4. ✅ **Monitor with nvidia-smi** - Watch GPU usage

**You get:**
- 80-120x speedup over CPU
- Full Jupyter notebook experience  
- Direct HPC file access
- Integrated debugging
- All in your familiar VSCode!

**Enjoy your H100 GPUs!** 🚀
