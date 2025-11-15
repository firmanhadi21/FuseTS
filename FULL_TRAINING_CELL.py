# ============================================================================
# FULL DATASET TRAINING - Optimized for 4-6 hours (vs 21 hours original)
# ============================================================================
#
# Use this ONLY if fast training R² < 0.55
# Expected improvement: +0.05 to +0.10 R² over fast training
#
# ============================================================================

import time
import torch

print("="*80)
print("🔥 FULL DATASET TRAINING - OPTIMIZED")
print("="*80)

# Reload module
print("\n🔄 Reloading module...")
import sys, importlib
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
print("   ✅ Module reloaded")

# Configuration
print("\n⚙️  Configuration:")
print("   Dataset: FULL (all 54M samples)")
print("   Batch size: 2M (optimized for H100)")
print("   Epochs: 50 (with early stopping)")
print("   Expected time: 4-6 hours")
print("   Expected R²: 0.50-0.65")
print()

# Ask for confirmation
print("⚠️  This will take 4-6 hours. Make sure:")
print("   1. Your session won't timeout")
print("   2. GPU will remain available")
print("   3. You have time to wait")
print()

start_time = time.time()

# Run full training
model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(
    combined_dataset,        # ← FULL dataset (no subsampling)
    batch_size=2_000_000,    # ← 2M batch size (optimized for H100)
    learning_rate=0.001,     # ← Standard learning rate
    epochs=50,               # ← Reduced from 150 (still plenty)
    warmup_epochs=5,         # ← Standard warmup
    val_split=0.2,           # ← 80/20 split
    verbose=True
)

total_time = time.time() - start_time

print("\n" + "="*80)
print("✅ FULL DATASET TRAINING COMPLETE")
print("="*80)
print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
print(f"\n🎯 Final Performance:")
print(f"   Training R²:   {metrics_train['r2']:.4f}")
print(f"   Validation R²: {metrics_val['r2']:.4f}")
print(f"   MAE:           {metrics_val['mae']:.4f}")
print(f"   RMSE:          {metrics_val['rmse']:.4f}")

# Compare to target
if metrics_val['r2'] >= 0.70:
    print(f"\n   🎉 EXCELLENT! R² ≥ 0.70 - Publication quality!")
elif metrics_val['r2'] >= 0.55:
    print(f"\n   ✅ SUCCESS! R² ≥ 0.55 - TARGET ACHIEVED!")
elif metrics_val['r2'] >= 0.50:
    print(f"\n   ✓ GOOD! R² ≥ 0.50 - Usable but below target")
else:
    print(f"\n   ⚠️ Below 0.50 - Consider MOGPR alternative")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'history': history,
    'metrics_train': metrics_train,
    'metrics_val': metrics_val,
    'training_time': total_time,
    'dataset_size': 'full_54M'
}, 's1_ndvi_model_full.pth')
print(f"\n💾 Model saved: s1_ndvi_model_full.pth")

# Compare to fast training (if available)
try:
    import os
    if os.path.exists('s1_ndvi_model_fast.pth'):
        fast_model = torch.load('s1_ndvi_model_fast.pth', weights_only=False)
        fast_r2 = fast_model['metrics_val']['r2']
        improvement = metrics_val['r2'] - fast_r2

        print(f"\n📊 Comparison to Fast Training:")
        print(f"   Fast training R²:  {fast_r2:.4f}")
        print(f"   Full training R²:  {metrics_val['r2']:.4f}")
        print(f"   Improvement:       {improvement:+.4f} ({improvement/fast_r2*100:+.1f}%)")

        if improvement > 0.05:
            print(f"   ✅ Significant improvement! Worth the extra time.")
        elif improvement > 0.02:
            print(f"   ✓ Moderate improvement.")
        else:
            print(f"   ⚠️ Small improvement - fast training was nearly as good.")
except:
    pass

print("="*80)
