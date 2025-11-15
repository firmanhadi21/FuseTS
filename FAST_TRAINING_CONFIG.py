# ============================================================================
# FAST TRAINING CONFIGURATION
# ============================================================================
#
# Use this for much faster training (10-20 minutes vs 21 hours)
#
# Optimizations:
# 1. Larger batch size (1M vs 256k) → fewer iterations
# 2. Fewer epochs (30 vs 150) with early stopping
# 3. Spatial subsampling → reduce dataset size
#
# ============================================================================

import numpy as np
import time

print("="*80)
print("⚡ FAST TRAINING CONFIGURATION")
print("="*80)

# ============================================================================
# OPTION 1: Subsample spatially (fastest - recommended)
# ============================================================================
# Take every Nth pixel to reduce dataset size by ~75%
# This gives you 13M samples instead of 54M

print("\n📊 Current dataset:")
print(f"   Shape: {combined_dataset['VV'].shape}")
print(f"   Total pixels: {combined_dataset['VV'].size:,}")

# Spatial stride (take every 2nd pixel in each dimension)
SPATIAL_STRIDE = 2

print(f"\n⚡ Applying spatial subsampling (stride={SPATIAL_STRIDE})...")

combined_dataset_fast = combined_dataset.isel(
    y=slice(None, None, SPATIAL_STRIDE),
    x=slice(None, None, SPATIAL_STRIDE)
)

print(f"   New shape: {combined_dataset_fast['VV'].shape}")
print(f"   New total pixels: {combined_dataset_fast['VV'].size:,}")
print(f"   Reduction: {(1 - combined_dataset_fast['VV'].size / combined_dataset['VV'].size):.0%}")

# ============================================================================
# OPTION 2: Random sampling (alternative)
# ============================================================================
# Randomly sample 10M pixels instead of using all 54M
# Uncomment this if you prefer random sampling over spatial subsampling

# print("\n⚡ Alternative: Random sampling to 10M pixels...")
#
# n_times, n_y, n_x = combined_dataset['VV'].shape
# total_pixels = n_times * n_y * n_x
# target_pixels = 10_000_000
#
# # Calculate how many pixels to keep
# keep_ratio = target_pixels / total_pixels
#
# # Create random mask
# np.random.seed(42)
# mask = np.random.random((n_times, n_y, n_x)) < keep_ratio
#
# # Apply mask
# for var in ['VV', 'VH', 'S2ndvi']:
#     combined_dataset_fast[var] = combined_dataset[var].where(mask)
#
# print(f"   Kept ~{target_pixels:,} pixels")

# ============================================================================
# Fast training parameters
# ============================================================================

FAST_CONFIG = {
    'batch_size': 1_024_000,      # 1M batch (4x larger than before)
    'learning_rate': 0.001,
    'epochs': 30,                  # Reduced from 150
    'warmup_epochs': 3,            # Reduced from 5
    'val_split': 0.2,
    'verbose': True
}

print(f"\n⚙️  Training configuration:")
print(f"   Batch size: {FAST_CONFIG['batch_size']:,}")
print(f"   Epochs: {FAST_CONFIG['epochs']}")
print(f"   Warmup epochs: {FAST_CONFIG['warmup_epochs']}")

# Estimate training time
n_samples_approx = combined_dataset_fast['VV'].size * 0.839  # 83.9% valid from diagnostic
n_train = n_samples_approx * 0.8  # 80% for training
batches_per_epoch = n_train / FAST_CONFIG['batch_size']
seconds_per_batch = 2.5  # Measured from previous run
seconds_per_epoch = batches_per_epoch * seconds_per_batch
total_time = seconds_per_epoch * FAST_CONFIG['epochs'] / 60  # minutes

print(f"\n⏱️  Estimated training time:")
print(f"   Batches per epoch: ~{batches_per_epoch:.0f}")
print(f"   Time per epoch: ~{seconds_per_epoch/60:.1f} minutes")
print(f"   Total time (30 epochs): ~{total_time:.0f} minutes")
print(f"   With early stopping: likely 10-20 minutes")

print("\n✅ Fast configuration ready!")
print("="*80)
