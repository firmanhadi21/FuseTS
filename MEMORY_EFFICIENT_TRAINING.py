# ============================================================================
# MEMORY-EFFICIENT FULL TRAINING
# ============================================================================
#
# Optimized for shared GPU with limited free memory
# Works with as little as 2-4 GB free GPU memory
#
# ============================================================================

import torch
import gc
import time

print("="*80)
print("💾 MEMORY-EFFICIENT FULL TRAINING")
print("="*80)

# Step 1: Clear all GPU memory from previous runs
print("\n🧹 Clearing GPU memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

    # Check available memory
    free_memory = torch.cuda.mem_get_info()[0] / 1e9
    total_memory = torch.cuda.mem_get_info()[1] / 1e9
    print(f"   GPU memory: {free_memory:.2f} GB free / {total_memory:.2f} GB total")

    if free_memory < 2.0:
        print(f"\n   ⚠️  WARNING: Only {free_memory:.2f} GB free!")
        print(f"   You may need to:")
        print(f"   1. Restart your Jupyter kernel")
        print(f"   2. Kill other GPU processes")
        print(f"   3. Use even smaller batch size")

# Step 2: Delete any large variables from memory
print("\n🗑️  Deleting old variables...")
vars_to_delete = ['model', 'pred_train', 'pred_val', 'X_train_tensor',
                  'y_train_tensor', 'X_val_tensor', 'y_val_tensor']
for var_name in vars_to_delete:
    if var_name in globals():
        del globals()[var_name]
        print(f"   Deleted: {var_name}")

gc.collect()
torch.cuda.empty_cache()

# Step 3: Reload module
print("\n🔄 Reloading module...")
import sys, importlib
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
print("   ✅ Module reloaded")

# Step 4: Calculate optimal batch size based on free memory
free_memory = torch.cuda.mem_get_info()[0] / 1e9
if free_memory >= 10:
    batch_size = 2_000_000
    print(f"\n   Using large batch: 2M samples")
elif free_memory >= 5:
    batch_size = 1_000_000
    print(f"\n   Using medium batch: 1M samples")
elif free_memory >= 3:
    batch_size = 500_000
    print(f"\n   Using small batch: 500K samples")
else:
    batch_size = 250_000
    print(f"\n   Using tiny batch: 250K samples (slow but works)")

# Step 5: Run training with memory-efficient settings
print("\n🚀 Starting memory-efficient training...")
print(f"   Batch size: {batch_size:,}")
print(f"   Epochs: 50")
print(f"   Dataset: Full (54M samples)")
print()

start_time = time.time()

try:
    model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(
        combined_dataset,        # Full dataset
        batch_size=batch_size,   # Adjusted to available memory
        learning_rate=0.001,
        epochs=50,
        warmup_epochs=5,
        val_split=0.2,
        verbose=True
    )

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("✅ MEMORY-EFFICIENT TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
    print(f"\n🎯 Final Performance:")
    print(f"   Training R²:   {metrics_train['r2']:.4f}")
    print(f"   Validation R²: {metrics_val['r2']:.4f}")
    print(f"   MAE:           {metrics_val['mae']:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'metrics_train': metrics_train,
        'metrics_val': metrics_val,
        'batch_size': batch_size
    }, 's1_ndvi_model_full_memory_efficient.pth')
    print(f"\n💾 Model saved: s1_ndvi_model_full_memory_efficient.pth")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("\n" + "="*80)
        print("❌ STILL OUT OF MEMORY!")
        print("="*80)
        print(f"\nTried batch size: {batch_size:,}")
        print(f"\nYou need to:")
        print(f"1. Restart Jupyter kernel: Kernel → Restart")
        print(f"2. Re-run data loading cells")
        print(f"3. Run this training cell immediately")
        print(f"\nOR wait for other GPU processes to finish:")
        print(f"   Check with: nvidia-smi")
    else:
        raise

print("="*80)
