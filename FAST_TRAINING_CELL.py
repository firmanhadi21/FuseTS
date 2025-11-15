# ============================================================================
# FAST TRAINING CELL - Run this for 10-20 minute training
# ============================================================================

import time
import torch

print("="*80)
print("⚡ FAST TRAINING MODE")
print("="*80)

# Step 1: Subsample dataset spatially (75% reduction)
print("\n📊 Subsampling dataset...")
combined_dataset_fast = combined_dataset.isel(
    y=slice(None, None, 2),  # Every 2nd pixel in Y
    x=slice(None, None, 2)   # Every 2nd pixel in X
)

print(f"   Original: {combined_dataset['VV'].shape} = {combined_dataset['VV'].size:,} pixels")
print(f"   Fast:     {combined_dataset_fast['VV'].shape} = {combined_dataset_fast['VV'].size:,} pixels")
print(f"   Reduction: {(1 - combined_dataset_fast['VV'].size / combined_dataset['VV'].size) * 100:.0f}%")

# Step 2: Reload module with latest fixes
print("\n🔄 Reloading module...")
import sys, importlib
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
print("   ✅ Module reloaded")

# Step 3: Run fast training
print("\n🚀 Starting fast training...")
print("   Target: 10-20 minutes total")
print("   Epochs: 30 (with early stopping)")
print("   Batch size: 1M samples")
print()

start_time = time.time()

model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(
    combined_dataset_fast,    # ← Using subsampled dataset
    batch_size=1_024_000,     # ← 4x larger batch
    learning_rate=0.001,
    epochs=30,                # ← 5x fewer epochs
    warmup_epochs=3,          # ← Faster warmup
    val_split=0.2,
    verbose=True
)

total_time = time.time() - start_time

print("\n" + "="*80)
print("✅ FAST TRAINING COMPLETE")
print("="*80)
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"\n🎯 Performance:")
print(f"   Training R²:   {metrics_train['r2']:.4f}")
print(f"   Validation R²: {metrics_val['r2']:.4f}")
print(f"   MAE:           {metrics_val['mae']:.4f}")
print(f"   RMSE:          {metrics_val['rmse']:.4f}")

if metrics_val['r2'] >= 0.55:
    print(f"\n   🎉 SUCCESS! Validation R² ≥ 0.55 - TARGET ACHIEVED!")
elif metrics_val['r2'] >= 0.45:
    print(f"\n   ✓ GOOD! Validation R² ≥ 0.45 - Close to target")
else:
    print(f"\n   ⚠️ Below target but model is working")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'history': history,
    'metrics_train': metrics_train,
    'metrics_val': metrics_val
}, 's1_ndvi_model_fast.pth')
print(f"\n💾 Model saved: s1_ndvi_model_fast.pth")
print("="*80)
