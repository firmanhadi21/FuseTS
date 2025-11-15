# ============================================================================
# CPU-BASED TRAINING - Minimal GPU Memory Usage
# ============================================================================
#
# Strategy: Keep training data on CPU, only move batches to GPU
# Works with as little as 500MB free GPU memory
# Slower but will complete successfully
#
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import gc

print("="*80)
print("🖥️  CPU-BASED TRAINING (Minimal GPU Memory)")
print("="*80)

# Clear GPU
torch.cuda.empty_cache()
gc.collect()

free_gb = torch.cuda.mem_get_info()[0] / 1e9
print(f"\nGPU memory: {free_gb:.2f} GB free")

# Reload module
import sys, importlib
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])

from improved_s1_ndvi_fusion_v2 import (
    prepare_enhanced_features_v2,
    ImprovedS1NDVIModelV2,
    WarmupCosineSchedule
)

print("\n" + "="*80)
print("🔧 FEATURE ENGINEERING")
print("="*80)

# Extract features (CPU-based)
X_all, y_all, mask_valid, feature_names = prepare_enhanced_features_v2(
    combined_dataset,
    verbose=True
)

X_filtered = X_all[mask_valid]
y_filtered = y_all[mask_valid]

print(f"\nFiltered data: {len(X_filtered):,} samples")

# Train/val split
n_samples = len(X_filtered)
n_val = int(n_samples * 0.2)
n_train = n_samples - n_val

indices = np.random.permutation(n_samples)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train = X_filtered[train_idx]
y_train = y_filtered[train_idx]
X_val = X_filtered[val_idx]
y_val = y_filtered[val_idx]

print(f"Training: {len(X_train):,} samples")
print(f"Validation: {len(X_val):,} samples")

# Normalize
print("\nNormalizing features...")
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

y_mean = y_train.mean()
y_std = y_train.std()
y_train_norm = (y_train - y_mean) / y_std
y_val_norm = (y_val - y_mean) / y_std

# CRITICAL: Keep data on CPU, use pin_memory for fast transfer
print("\nCreating CPU tensors with pin_memory...")
X_train_cpu = torch.FloatTensor(X_train_norm)  # CPU
y_train_cpu = torch.FloatTensor(y_train_norm.reshape(-1, 1))  # CPU
X_val_cpu = torch.FloatTensor(X_val_norm)  # CPU
y_val_cpu = torch.FloatTensor(y_val_norm.reshape(-1, 1))  # CPU

train_dataset = TensorDataset(X_train_cpu, y_train_cpu)
train_loader = DataLoader(
    train_dataset,
    batch_size=500_000,  # Can be larger since data is on CPU
    shuffle=True,
    num_workers=0,       # No workers to avoid shared memory issues
    pin_memory=True      # Fast CPU→GPU transfer
)

print(f"Batch size: 500,000")
print(f"Batches per epoch: {len(train_loader)}")

# Initialize model on GPU
device = torch.device('cuda')
model = ImprovedS1NDVIModelV2(input_dim=X_train.shape[1]).to(device)

print(f"\n🏗️  Model on GPU: {next(model.parameters()).device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = WarmupCosineSchedule(optimizer, warmup_epochs=5, total_epochs=50,
                                 base_lr=0.001, min_lr=1e-6)

print("\n" + "="*80)
print("🏃 TRAINING (CPU→GPU Batch Transfer)")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
patience = 25
history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}

start_time = time.time()

for epoch in range(50):
    # Training
    model.train()
    epoch_start = time.time()
    epoch_loss = 0
    n_batches = 0

    for batch_X_cpu, batch_y_cpu in train_loader:
        # Move batch to GPU (only this batch, not all data!)
        batch_X = batch_X_cpu.to(device, non_blocking=True)
        batch_y = batch_y_cpu.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        # Free GPU memory immediately
        del batch_X, batch_y, outputs, loss
        if n_batches % 10 == 0:
            torch.cuda.empty_cache()

    epoch_loss /= n_batches

    # Validation (process in chunks to save memory)
    model.eval()
    val_loss = 0
    chunk_size = 500_000
    n_val_batches = 0

    with torch.no_grad():
        for i in range(0, len(X_val_cpu), chunk_size):
            chunk_X = X_val_cpu[i:i+chunk_size].to(device)
            chunk_y = y_val_cpu[i:i+chunk_size].to(device)

            val_outputs = model(chunk_X)
            val_loss += criterion(val_outputs, chunk_y).item()
            n_val_batches += 1

            del chunk_X, chunk_y, val_outputs

    val_loss /= n_val_batches

    # Update learning rate
    current_lr = scheduler.step()

    epoch_time = time.time() - epoch_start
    history['train_loss'].append(epoch_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(current_lr)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'y_mean': y_mean,
            'y_std': y_std
        }, 's1_ndvi_model_cpu_trained.pth')
    else:
        patience_counter += 1

    # Progress
    elapsed = time.time() - start_time
    eta = (elapsed / (epoch + 1)) * (50 - epoch - 1)
    val_indicator = "↓" if val_loss < best_val_loss else "↑"

    print(f"Epoch {epoch+1:3d}/50: Loss={epoch_loss:.4f}, Val={val_loss:.4f}{val_indicator}, "
          f"LR={current_lr:.6f} [Time: {epoch_time:.1f}s, ETA: {eta/60:.1f}m]")

    if patience_counter >= patience:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

total_time = time.time() - start_time

# Load best model and evaluate
checkpoint = torch.load('s1_ndvi_model_cpu_trained.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print("\n" + "="*80)
print("📊 EVALUATION")
print("="*80)

# Evaluate in chunks
model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(X_val_cpu), 500_000):
        chunk = X_val_cpu[i:i+500_000].to(device)
        pred = model(chunk)
        predictions.append(pred.cpu().numpy())
        del chunk, pred

predictions = np.concatenate(predictions).flatten()
predictions_denorm = predictions * y_std + y_mean
predictions_denorm = np.clip(predictions_denorm, -1, 1)

r2 = r2_score(y_val, predictions_denorm)
mae = mean_absolute_error(y_val, predictions_denorm)
rmse = np.sqrt(mean_squared_error(y_val, predictions_denorm))

print(f"\n✅ TRAINING COMPLETE")
print(f"   Total time: {total_time/3600:.1f} hours")
print(f"   Validation R²: {r2:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")

if r2 >= 0.55:
    print(f"   🎉 TARGET ACHIEVED!")
elif r2 >= 0.50:
    print(f"   ✓ Close to target")
else:
    print(f"   ⚠️ Below target")

print("="*80)
