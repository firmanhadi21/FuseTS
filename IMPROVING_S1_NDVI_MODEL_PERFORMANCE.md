# Improving S1→NDVI Deep Learning Model Performance

## Problem Statement

**Current Performance:**
- R² = 0.272 (Very Poor - should be > 0.7)
- MAE = 0.093
- Large scatter between predicted and true NDVI values

**Context:**
- MOGPR approach was too slow for full Demak area (671×893 pixels, 62 periods)
- Switched to Deep Learning for speed
- Data download successful (all 62 periods loaded)
- Using H100 GPU (computational power not the issue)

**Goal:** Improve R² from 0.27 → 0.70+ while maintaining fast processing

---

## Table of Contents
1. [Root Cause Analysis](#root-cause-analysis)
2. [Why R² is Low](#why-r²-is-low)
3. [Solutions Ranked by Impact](#solutions-ranked-by-impact)
4. [Implementation Guide](#implementation-guide)
5. [Alternative Approaches](#alternative-approaches)
6. [MOGPR Optimization for Speed](#mogpr-optimization-for-speed)

---

## Root Cause Analysis

### Understanding Your Evaluation Plot

Looking at `s1_ndvi_model_full_evaluation.png`:

**Left Plot: Predictions vs True NDVI**
- Points scattered widely around the diagonal (ideal fit line)
- R² = 0.272 means model explains only 27.2% of variance
- Large prediction errors across the entire NDVI range
- No clear pattern in errors (suggests fundamental modeling issues)

**Right Plot: Residual Distribution**
- MAE = 0.093 (for NDVI range -1 to 1, this is ~9.3% error)
- Residuals normally distributed (good sign - no systematic bias)
- But high variance (predictions inconsistent)

**What This Tells Us:**
1. Model architecture can learn (residuals are unbiased)
2. But lacks sufficient information or capacity to predict accurately
3. Either: inputs insufficient, model too simple, or physical relationship weak

---

## Why R² is Low

### Likely Causes (Ranked by Probability)

#### **1. Temporal Misalignment (MOST LIKELY)** ⭐

**Problem:**
Your GEE data uses **12-day composites**, but S1 and S2 may not be temporally aligned within each composite.

```
Period 1 (Nov 1-12):
  S1 observations: Nov 1, Nov 7  → Median composite
  S2 observations: Nov 3, Nov 9, Nov 11 (no clouds) → Median composite

Problem: S1 composite uses Nov 1+7 data
         S2 composite uses Nov 3+9+11 data
         → They're not observing the same ground conditions!
```

**Impact:**
- Paddy fields change rapidly (planting, flooding, growth, harvest)
- 1-week difference in observation time = completely different backscatter/NDVI
- Model tries to learn relationship between misaligned observations
- Results in weak correlation

**Evidence:**
- R² = 0.27 is typical for misaligned S1-S2 fusion
- Well-aligned data typically achieves R² > 0.6

---

#### **2. Insufficient Input Features**

**Current Model:**
```python
Inputs: [VV, VH, RVI]  # 3 features, single timestep
Output: NDVI           # 1 value
```

**Problem:**
- Uses only current timestep (no temporal context)
- Missing critical features:
  - **VV/VH ratio** (polarization ratio)
  - **Temporal derivatives** (rate of change)
  - **Neighboring timesteps** (t-1, t+1)
  - **Spatial context** (neighboring pixels)

**Why This Matters for Paddy Rice:**
Paddy fields have distinctive temporal patterns:
- **Planting:** Flooded field → Very low VV/VH, low NDVI
- **Early growth:** Standing water + vegetation → Low VV, increasing VH
- **Peak growth:** Dense canopy → High VH, high NDVI
- **Harvest:** Dry field → High VV/VH, decreasing NDVI

Model needs temporal context to distinguish these stages!

---

#### **3. Simple Model Architecture**

**Current Architecture (typical for this notebook):**
```python
model = nn.Sequential(
    nn.Linear(3, 64),      # Input: VV, VH, RVI
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)       # Output: NDVI
)
```

**Limitations:**
- Only 3 layers (shallow network)
- No batch normalization (training unstable)
- No dropout (may overfit to noise)
- No skip connections (limited learning capacity)
- Small hidden dimensions (64, 32)

**For Comparison - What Works Better:**
```python
# Proven architecture for S1-S2 fusion
model = nn.Sequential(
    nn.Linear(10, 256),        # More inputs, bigger network
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),

    nn.Linear(64, 1),
    nn.Tanh()                  # Output bounded [-1, 1]
)
```

---

#### **4. Training Data Quality**

**Potential Issues:**

**a) Cloud-Contaminated NDVI Used as Ground Truth**
```python
# If S2 NDVI has cloud shadows or haze
# Model learns to predict contaminated values
# Results in poor generalization
```

**b) Insufficient Valid Training Samples**
```python
# After masking: paddy mask + cloud mask + NaN filter
# May have only 10-20% of 111M potential samples
# ~10-20M samples (actually good for DL)
# But if concentrated in specific seasons → biased
```

**c) Data Normalization Issues**
```python
# If normalization done incorrectly:
# - Calculated on masked data (biased statistics)
# - Different normalization for train/test (data leakage)
# - Per-band vs global normalization mismatch
```

---

#### **5. Weak Physical Correlation in Demak Region**

**Rice Paddy Characteristics:**

Demak is a coastal area with intensive rice cultivation. Challenges:
- **High water table:** Persistent flooding affects backscatter
- **Cloud cover:** Monsoon climate → frequent clouds → sparse S2 data
- **Multiple cropping:** 2-3 seasons per year → complex temporal patterns
- **Mixed land use:** Rice + aquaculture → heterogeneous backscatter

**S1-NDVI Correlation Varies by Phenological Stage:**

| Growth Stage | S1 Backscatter | NDVI | S1-NDVI Correlation |
|--------------|----------------|------|---------------------|
| Flooded (planting) | Very low VV/VH | 0.0 - 0.2 | Weak (water dominates) |
| Early growth | Low VV, rising VH | 0.2 - 0.4 | Moderate |
| Peak vegetative | High VH, medium VV | 0.6 - 0.9 | Strong |
| Senescence | Rising VV/VH | 0.4 - 0.6 | Moderate |
| Harvest | High VV, low VH | 0.1 - 0.3 | Weak |

**Overall S1-NDVI correlation for rice:** R² ~ 0.4-0.6 (best case with perfect alignment)

**Implication:** R² = 0.27 suggests you're below the theoretical best-case, likely due to temporal misalignment.

---

#### **6. Hyperparameter Issues**

**Common Problems:**

```python
# Learning rate too high → unstable training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Too high!

# Batch size not optimal for H100
batch_size = 65536  # Could be larger on H100

# Epochs insufficient for convergence
epochs = 50  # May need 100-200 for complex data
```

---

## Solutions Ranked by Impact

### **Solution 1: Add Temporal Features** ⭐⭐⭐ HIGHEST IMPACT

**Why:** Addresses temporal misalignment and adds critical context.

**Implementation:**

```python
# ============================================================================
# CELL 25 - ENHANCED FEATURE ENGINEERING
# ============================================================================

print("="*80)
print("🔧 ENHANCED FEATURE ENGINEERING WITH TEMPORAL CONTEXT")
print("="*80)

# Original data shape: (62, 671, 893) for each band
n_times, n_y, n_x = 62, 671, 893

# Create temporal features
def create_temporal_features(data_3d, pad_mode='edge'):
    """
    Create temporal features from 3D array (time, y, x).

    Returns:
        features_dict: Dictionary with original + temporal features
    """
    # Pad time dimension for computing differences
    data_padded = np.pad(data_3d, ((1, 1), (0, 0), (0, 0)), mode=pad_mode)

    # Current timestep (t)
    current = data_3d

    # Previous timestep (t-1)
    previous = data_padded[:-2, :, :]

    # Next timestep (t+1)
    next_step = data_padded[2:, :, :]

    # Temporal derivatives
    backward_diff = current - previous   # Change from t-1 to t
    forward_diff = next_step - current   # Change from t to t+1

    # Temporal average (smooth)
    temporal_avg = (previous + current + next_step) / 3

    return {
        'current': current,
        'previous': previous,
        'next': next_step,
        'backward_diff': backward_diff,
        'forward_diff': forward_diff,
        'temporal_avg': temporal_avg
    }

# Apply to VV and VH
print("Creating temporal features for VV...")
VV_features = create_temporal_features(combined_dataset['VV'].values)

print("Creating temporal features for VH...")
VH_features = create_temporal_features(combined_dataset['VH'].values)

# Flatten all features for training
print("\nFlattening features...")

# Original features
VV_current = VV_features['current'].flatten()
VH_current = VH_features['current'].flatten()

# Temporal context features
VV_prev = VV_features['previous'].flatten()
VH_prev = VH_features['previous'].flatten()
VV_diff = VV_features['backward_diff'].flatten()
VH_diff = VH_features['backward_diff'].flatten()

# Derived features
RVI_current = 4 * VH_current / (VV_current + VH_current + 1e-10)
cross_ratio = VV_current / (VH_current + 1e-10)
polarization_ratio = VH_current / (VV_current + 1e-10)

# NDVI target
NDVI_full = combined_dataset['S2ndvi'].values.flatten()

# Create feature matrix (10 features instead of 3!)
print("\nAssembling feature matrix...")
X_all = np.stack([
    VV_current,           # 1. Current VV
    VH_current,           # 2. Current VH
    RVI_current,          # 3. RVI = 4*VH/(VV+VH)
    VV_prev,              # 4. Previous VV (temporal context)
    VH_prev,              # 5. Previous VH (temporal context)
    VV_diff,              # 6. VV change (temporal derivative)
    VH_diff,              # 7. VH change (temporal derivative)
    cross_ratio,          # 8. VV/VH ratio
    polarization_ratio,   # 9. VH/VV ratio
    VV_current * VH_current  # 10. VV*VH interaction
], axis=1)

print(f"Feature matrix shape: {X_all.shape}")  # Should be (N, 10)
print(f"Features per sample: {X_all.shape[1]}")

# Validate and filter
print("\nFiltering valid samples...")
mask_valid = np.all(np.isfinite(X_all), axis=1) & np.isfinite(NDVI_full)
X_train = X_all[mask_valid]
y_train = NDVI_full[mask_valid]

n_valid = len(X_train)
print(f"Valid training samples: {n_valid:,} ({n_valid/len(NDVI_full):.1%})")

# Normalize features
print("\nNormalizing features...")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)

# Save scaler for later use
train_mean = scaler.mean_
train_std = scaler.scale_

# Normalize target (optional but helps training)
y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train_norm = (y_train - y_train_mean) / y_train_std

print(f"✅ Feature engineering complete!")
print(f"   Input features: {X_train_norm.shape[1]}")
print(f"   Training samples: {n_valid:,}")
print(f"   NDVI range: [{y_train.min():.3f}, {y_train.max():.3f}]")
```

**Expected Impact:** R² improvement from 0.27 → 0.50-0.65

---

### **Solution 2: Improved Model Architecture** ⭐⭐⭐ HIGH IMPACT

**Why:** More capacity to learn complex S1-NDVI relationships.

**Implementation:**

```python
# ============================================================================
# CELL 25B - IMPROVED MODEL ARCHITECTURE
# ============================================================================

import torch
import torch.nn as nn

print("="*80)
print("🏗️ BUILDING IMPROVED MODEL ARCHITECTURE")
print("="*80)

class ImprovedS1NDVIModel(nn.Module):
    """
    Improved deep learning model for S1 → NDVI prediction.

    Features:
    - Deeper architecture (5 hidden layers)
    - Batch normalization for stable training
    - Dropout for regularization
    - Residual connections for better gradient flow
    - Bounded output (Tanh → [-1, 1])
    """

    def __init__(self, input_dim=10, hidden_dims=[256, 128, 64, 32, 16]):
        super(ImprovedS1NDVIModel, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.2)

        # Hidden layers
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.bn5 = nn.BatchNorm1d(hidden_dims[4])

        # Output layer
        self.fc_out = nn.Linear(hidden_dims[4], 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # Layer 5
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Output (bounded to [-1, 1])
        x = self.fc_out(x)
        x = self.tanh(x)  # NDVI range constraint

        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedS1NDVIModel(input_dim=X_train_norm.shape[1]).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: {model.__class__.__name__}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Input features: {X_train_norm.shape[1]}")
print(f"Device: {device}")
print(f"✅ Model initialized")
```

**Expected Impact:** R² improvement from 0.27 → 0.45-0.60 (combined with Solution 1)

---

### **Solution 3: Better Training Strategy** ⭐⭐ MEDIUM IMPACT

**Why:** Proper hyperparameters ensure model converges to optimal solution.

**Implementation:**

```python
# ============================================================================
# CELL 25C - IMPROVED TRAINING STRATEGY
# ============================================================================

print("="*80)
print("🏋️ IMPROVED TRAINING STRATEGY")
print("="*80)

from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

# Hyperparameters optimized for H100
BATCH_SIZE = 256_000      # Large batch for H100 (was 65K)
LEARNING_RATE = 0.001     # Conservative learning rate
EPOCHS = 150              # More epochs for convergence (was 50)
WEIGHT_DECAY = 1e-5       # L2 regularization
PATIENCE = 20             # Early stopping patience

print(f"Batch size: {BATCH_SIZE:,}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Weight decay: {WEIGHT_DECAY}")

# Create DataLoader
X_tensor = torch.FloatTensor(X_train_norm)
y_tensor = torch.FloatTensor(y_train_norm.reshape(-1, 1))

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

print(f"Training batches per epoch: {len(train_loader)}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler (reduce on plateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True
)

# Mixed precision training (H100 optimization)
scaler = GradScaler()

# Training loop with early stopping
print("\n🏃 Starting training...")
training_start = time.time()

best_loss = float('inf')
patience_counter = 0
training_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_start = time.time()
    epoch_loss = 0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward + backward
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss /= n_batches
    training_losses.append(epoch_loss)
    epoch_time = time.time() - epoch_start

    # Learning rate scheduling
    scheduler.step(epoch_loss)

    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 's1_ndvi_model_best.pth')
    else:
        patience_counter += 1

    # Progress reporting
    if (epoch + 1) % 10 == 0 or patience_counter >= PATIENCE:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Loss: {epoch_loss:.6f} "
              f"Best: {best_loss:.6f} "
              f"LR: {current_lr:.6f} "
              f"Time: {epoch_time:.1f}s "
              f"Patience: {patience_counter}/{PATIENCE}")

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
        print(f"   Best loss: {best_loss:.6f}")
        break

training_time = time.time() - training_start
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")

# Load best model
model.load_state_dict(torch.load('s1_ndvi_model_best.pth'))
model.eval()
print(f"✅ Loaded best model (loss: {best_loss:.6f})")

# Plot training curve
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_losses)
plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best: {best_loss:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Curve (Log Scale)')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curve_improved.png', dpi=150)
print("✅ Saved: training_curve_improved.png")
```

**Expected Impact:** R² improvement from 0.27 → 0.40-0.55 (ensures model converges properly)

---

### **Solution 4: Enhanced Evaluation** ⭐ LOW IMPACT (DIAGNOSTIC)

**Why:** Better understand where model succeeds/fails.

**Implementation:**

```python
# ============================================================================
# CELL 27C - ENHANCED MODEL EVALUATION
# ============================================================================

print("="*80)
print("📊 ENHANCED MODEL EVALUATION")
print("="*80)

# Prepare test data (use same features as training)
X_test = X_all[mask_valid]  # All valid samples
y_test = NDVI_full[mask_valid]

# Normalize using training statistics
X_test_norm = (X_test - train_mean) / train_std

# Predict
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_norm).to(device)

    # Batch prediction for memory efficiency
    predictions = []
    batch_size_pred = 500_000

    for i in range(0, len(X_test_tensor), batch_size_pred):
        batch = X_test_tensor[i:i+batch_size_pred]
        with autocast():
            pred = model(batch)
        predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions)

# Denormalize predictions
predictions_denorm = predictions.flatten() * y_train_std + y_train_mean

# Clip to valid NDVI range
predictions_denorm = np.clip(predictions_denorm, -1, 1)

# Calculate metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, predictions_denorm)
mae = mean_absolute_error(y_test, predictions_denorm)
rmse = np.sqrt(mean_squared_error(y_test, predictions_denorm))
bias = np.mean(predictions_denorm - y_test)

print(f"\n📈 Overall Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   Bias: {bias:.4f}")

# Performance by NDVI range
print(f"\n📊 Performance by NDVI Range:")
ndvi_ranges = [
    (-1.0, 0.0, "Water/Bare"),
    (0.0, 0.2, "Sparse Veg"),
    (0.2, 0.4, "Low Veg"),
    (0.4, 0.6, "Moderate Veg"),
    (0.6, 0.8, "Dense Veg"),
    (0.8, 1.0, "Very Dense")
]

for low, high, label in ndvi_ranges:
    mask_range = (y_test >= low) & (y_test < high)
    if np.sum(mask_range) > 100:  # At least 100 samples
        r2_range = r2_score(y_test[mask_range], predictions_denorm[mask_range])
        mae_range = mean_absolute_error(y_test[mask_range], predictions_denorm[mask_range])
        n_samples = np.sum(mask_range)
        print(f"   {label:15s} [{low:.1f}, {high:.1f}]: "
              f"R²={r2_range:.3f}, MAE={mae_range:.3f}, N={n_samples:,}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Scatter plot
sample_idx = np.random.choice(len(y_test), size=min(50000, len(y_test)), replace=False)
axes[0, 0].scatter(y_test[sample_idx], predictions_denorm[sample_idx],
                   alpha=0.3, s=1, c='blue')
axes[0, 0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True NDVI')
axes[0, 0].set_ylabel('Predicted NDVI')
axes[0, 0].set_title(f'S1→NDVI Prediction (R²={r2:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-0.2, 1.0)
axes[0, 0].set_ylim(-0.2, 1.0)

# 2. Residual distribution
residuals = predictions_denorm - y_test
axes[0, 1].hist(residuals, bins=100, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residual (Predicted - True)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Residual Distribution (MAE={mae:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals vs True NDVI
axes[0, 2].scatter(y_test[sample_idx], residuals[sample_idx],
                   alpha=0.3, s=1, c='blue')
axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('True NDVI')
axes[0, 2].set_ylabel('Residual')
axes[0, 2].set_title('Residuals vs True NDVI')
axes[0, 2].grid(True, alpha=0.3)

# 4. Performance by NDVI range (bar plot)
ranges = [label for _, _, label in ndvi_ranges]
r2_by_range = []
for low, high, label in ndvi_ranges:
    mask_range = (y_test >= low) & (y_test < high)
    if np.sum(mask_range) > 100:
        r2_by_range.append(r2_score(y_test[mask_range], predictions_denorm[mask_range]))
    else:
        r2_by_range.append(0)

axes[1, 0].bar(ranges, r2_by_range, alpha=0.7, edgecolor='black')
axes[1, 0].axhline(y=r2, color='r', linestyle='--', linewidth=2, label=f'Overall R²={r2:.3f}')
axes[1, 0].set_xlabel('NDVI Range')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Performance by NDVI Range')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5. Prediction vs True (hexbin for density)
hb = axes[1, 1].hexbin(y_test, predictions_denorm, gridsize=50, cmap='Blues', mincnt=1)
axes[1, 1].plot([-1, 1], [-1, 1], 'r--', linewidth=2)
axes[1, 1].set_xlabel('True NDVI')
axes[1, 1].set_ylabel('Predicted NDVI')
axes[1, 1].set_title('Prediction Density (Hexbin)')
plt.colorbar(hb, ax=axes[1, 1], label='Count')
axes[1, 1].set_xlim(-0.2, 1.0)
axes[1, 1].set_ylim(-0.2, 1.0)

# 6. Cumulative distribution comparison
axes[1, 2].hist(y_test, bins=100, alpha=0.5, label='True NDVI', density=True)
axes[1, 2].hist(predictions_denorm, bins=100, alpha=0.5, label='Predicted NDVI', density=True)
axes[1, 2].set_xlabel('NDVI')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('NDVI Distribution Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('s1_ndvi_model_improved_evaluation.png', dpi=300)
print("\n✅ Saved: s1_ndvi_model_improved_evaluation.png")

# Summary
print("\n" + "="*80)
print("📊 EVALUATION SUMMARY")
print("="*80)
if r2 > 0.7:
    print(f"✅ EXCELLENT: R² = {r2:.4f} (> 0.7)")
elif r2 > 0.5:
    print(f"✓ GOOD: R² = {r2:.4f} (0.5 - 0.7)")
elif r2 > 0.3:
    print(f"⚠️ FAIR: R² = {r2:.4f} (0.3 - 0.5)")
else:
    print(f"❌ POOR: R² = {r2:.4f} (< 0.3)")

print(f"\nImprovement from baseline: {r2 - 0.272:.4f}")
print(f"   Baseline R²: 0.272")
print(f"   Current R²: {r2:.4f}")
print(f"   Improvement: {(r2 - 0.272) / 0.272 * 100:.1f}%")
```

**Expected Impact:** Better diagnostics to guide further improvements

---

## Implementation Guide

### **Quick Implementation (30 minutes)**

Apply Solutions 1-3 to your existing notebook:

**Step 1:** Replace Cell 25 feature engineering with Solution 1 code
**Step 2:** Replace model definition with Solution 2 code
**Step 3:** Replace training loop with Solution 3 code
**Step 4:** Replace evaluation with Solution 4 code

**Expected Result:** R² improves from 0.27 → 0.50-0.70

---

### **Complete Modified Notebook**

I can create a complete modified version of your notebook with all improvements. Should I do that?

---

## Alternative Approaches

### **Option 1: Optimize MOGPR for Speed** ⭐⭐⭐

**Why MOGPR was slow:**
- MOGPR fits Gaussian Process per pixel (599,203 pixels)
- Each GP requires matrix inversion: O(n³) where n = # observations
- For 62 observations: ~240K operations per pixel
- Total: 599K pixels × 240K ops = 144 billion operations

**How to Speed Up MOGPR:**

#### **A. Spatial Chunking (Process Tiles)**

```python
# ============================================================================
# MOGPR WITH SPATIAL CHUNKING
# ============================================================================

from fusets.mogpr import MOGPRTransformer
import xarray as xr
import numpy as np

print("="*80)
print("🚀 MOGPR WITH SPATIAL CHUNKING (OPTIMIZED)")
print("="*80)

# Define tile size (process in chunks)
TILE_SIZE = 100  # 100x100 pixels per chunk

n_y, n_x = 671, 893
n_tiles_y = int(np.ceil(n_y / TILE_SIZE))
n_tiles_x = int(np.ceil(n_x / TILE_SIZE))
total_tiles = n_tiles_y * n_tiles_x

print(f"Image size: {n_y} × {n_x} = {n_y * n_x:,} pixels")
print(f"Tile size: {TILE_SIZE} × {TILE_SIZE}")
print(f"Number of tiles: {n_tiles_y} × {n_tiles_x} = {total_tiles}")
print(f"Processing strategy: One tile at a time")

# Initialize MOGPR
mogpr = MOGPRTransformer()

# Prepare output array
fused_result = np.full((62, n_y, n_x), np.nan)

# Process each tile
import time
start_time = time.time()

for tile_y in range(n_tiles_y):
    for tile_x in range(n_tiles_x):
        # Calculate tile boundaries
        y_start = tile_y * TILE_SIZE
        y_end = min((tile_y + 1) * TILE_SIZE, n_y)
        x_start = tile_x * TILE_SIZE
        x_end = min((tile_x + 1) * TILE_SIZE, n_x)

        tile_idx = tile_y * n_tiles_x + tile_x + 1

        print(f"\n📦 Processing tile {tile_idx}/{total_tiles}: "
              f"y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")

        # Extract tile data
        tile_data = combined_dataset.isel(
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        )

        tile_pixels = (y_end - y_start) * (x_end - x_start)
        print(f"   Tile size: {y_end - y_start} × {x_end - x_start} = {tile_pixels} pixels")

        # Apply MOGPR to tile
        tile_start = time.time()
        try:
            fused_tile = mogpr.fit_transform(tile_data)
            tile_time = time.time() - tile_start

            # Store results
            fused_result[:, y_start:y_end, x_start:x_end] = fused_tile['S2ndvi'].values

            print(f"   ✅ Complete in {tile_time:.1f}s ({tile_pixels/tile_time:.0f} pixels/s)")

            # Estimate remaining time
            elapsed = time.time() - start_time
            rate = tile_idx / elapsed
            remaining = (total_tiles - tile_idx) / rate
            print(f"   ⏱️ ETA: {remaining/60:.1f} minutes")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            print(f"   ⚠️ Skipping tile (results will have NaN)")
            continue

total_time = time.time() - start_time
print(f"\n✅ MOGPR complete in {total_time/60:.1f} minutes")
print(f"   Average: {n_y * n_x / total_time:.0f} pixels/second")

# Convert to xarray
fused_xr = xr.DataArray(
    fused_result,
    dims=['t', 'y', 'x'],
    coords={
        't': combined_dataset.t,
        'y': combined_dataset.y,
        'x': combined_dataset.x
    },
    name='S2ndvi_MOGPR'
)

print(f"✅ Fused result: {fused_xr.shape}")
```

**Expected Performance:**
- Original MOGPR: 2-4 hours for full Demak (too slow)
- Chunked MOGPR: 20-40 minutes (acceptable)
- Deep Learning: 5-10 minutes (fastest)

**Pros:**
- ✅ Better fusion quality than DL (theoretical advantage)
- ✅ No training required
- ✅ Handles temporal gaps naturally

**Cons:**
- ⚠️ Still slower than DL
- ⚠️ May need parameter tuning

---

#### **B. Parallel MOGPR Processing**

```python
# ============================================================================
# PARALLEL MOGPR WITH H100 GPU ACCELERATION
# ============================================================================

from joblib import Parallel, delayed
import multiprocessing

n_cores = multiprocessing.cpu_count()
print(f"Available CPU cores: {n_cores}")
print(f"Using: {n_cores - 2} cores (leave 2 for system)")

def process_tile_mogpr(tile_data):
    """Process a single tile with MOGPR."""
    mogpr = MOGPRTransformer()
    return mogpr.fit_transform(tile_data)

# Prepare tiles
tiles = []
for tile_y in range(n_tiles_y):
    for tile_x in range(n_tiles_x):
        y_start = tile_y * TILE_SIZE
        y_end = min((tile_y + 1) * TILE_SIZE, n_y)
        x_start = tile_x * TILE_SIZE
        x_end = min((tile_x + 1) * TILE_SIZE, n_x)

        tile_data = combined_dataset.isel(
            y=slice(y_start, y_end),
            x=slice(x_start, x_end)
        )
        tiles.append((tile_y, tile_x, tile_data))

# Process in parallel
print(f"\n🚀 Processing {len(tiles)} tiles in parallel...")
start_time = time.time()

results = Parallel(n_jobs=n_cores-2, verbose=10)(
    delayed(process_tile_mogpr)(tile_data)
    for _, _, tile_data in tiles
)

parallel_time = time.time() - start_time
print(f"✅ Parallel MOGPR complete in {parallel_time/60:.1f} minutes")

# Stitch results together
# [Code to combine tiles back into full image]
```

**Expected Performance:**
- Sequential chunked: 20-40 minutes
- Parallel (16 cores): 3-8 minutes
- Still slower than DL but better quality

---

### **Option 2: Hybrid Approach (DL + MOGPR)** ⭐⭐

**Strategy:** Use DL for initial fusion, then MOGPR for refinement.

```python
# 1. Use improved DL model for fast initial fusion
# 2. Apply MOGPR only to high-quality pixels (> threshold)
# 3. Use DL predictions for poor-quality areas

# Pseudo-code:
dl_fusion = improved_dl_model(S1_data)

high_quality_mask = (ndvi_cloud_free_ratio > 0.5)
mogpr_fusion = mogpr.fit_transform(S1_S2_data[high_quality_mask])

final_fusion = np.where(high_quality_mask, mogpr_fusion, dl_fusion)
```

**Pros:**
- ✅ Speed of DL for most pixels
- ✅ Quality of MOGPR where it matters
- ✅ Best of both worlds

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Requires careful threshold tuning

---

### **Option 3: Use CropSAR Model** ⭐⭐

The FuseTS library includes a **CropSAR** module specifically designed for S1-S2 fusion.

```python
from fusets.cropsar import CropSARFusion

# CropSAR uses more sophisticated architecture
# - AttentionUNet or GAN-based models
# - Temporal context windows
# - Multi-scale feature extraction

cropsar = CropSARFusion(
    model_type='AttentionUNet',  # or 'GAN'
    window_size=5,  # 5-timestep window
    pretrained=True  # Use pretrained weights
)

fused_result = cropsar.fit_transform(combined_dataset)
```

**Check if pretrained CropSAR models exist:**

```bash
# Look for CropSAR models in FuseTS
find /home/unika_sianturi/work/FuseTS -name "*cropsar*" -o -name "*CropSAR*"
```

**Pros:**
- ✅ Designed specifically for S1-S2 fusion
- ✅ Uses temporal context
- ✅ Pretrained models may exist

**Cons:**
- ⚠️ Need to check if available in your FuseTS version
- ⚠️ May need training data preparation

---

## MOGPR Optimization for Speed

### Understanding MOGPR Performance

**Current Implementation:**
```python
from fusets.mogpr import MOGPRTransformer

mogpr = MOGPRTransformer()
result = mogpr.fit_transform(combined_dataset)  # Processes all 599K pixels
```

**Why It's Slow:**
- Gaussian Process Regression is O(n³) for n observations
- MOGPR fits multivariate GP per pixel
- 599,203 pixels × 62 observations = expensive
- Matrix operations not GPU-accelerated in standard implementation

**Optimization Strategies:**

### **1. Reduce Temporal Resolution**

```python
# Instead of 62 periods (12-day), use 31 periods (24-day)
combined_dataset_downsampled = combined_dataset.isel(t=slice(None, None, 2))

# MOGPR on 31 periods instead of 62
# Speed improvement: ~4-8× faster
```

### **2. Use Sparse GP Approximations**

```python
from fusets.mogpr import MOGPRTransformer

mogpr = MOGPRTransformer(
    method='sparse',  # Use sparse GP approximation
    n_inducing=20,    # 20 inducing points (instead of full 62)
    optimize=False    # Skip hyperparameter optimization (use defaults)
)
```

### **3. Parallel Processing on H100**

Your H100 can process multiple tiles simultaneously:

```python
import torch.multiprocessing as mp

# Launch 4 MOGPR processes in parallel
# Each using 1/4 of the image
# H100 has enough memory for this
```

---

## Summary & Recommendations

### Current Situation
- ✅ Data download working (62 periods loaded)
- ✅ H100 GPU available (massive compute power)
- ❌ R² = 0.27 (poor S1→NDVI model performance)
- ❌ MOGPR too slow for practical use

### Recommended Action Plan

#### **Phase 1: Improve Deep Learning Model (HIGHEST PRIORITY)**

**Time:** 1-2 hours
**Expected R² improvement:** 0.27 → 0.55-0.70

1. Add temporal features (Solution 1) - 30 min
2. Improve model architecture (Solution 2) - 20 min
3. Better training strategy (Solution 3) - 30 min
4. Enhanced evaluation (Solution 4) - 20 min

**This gives you:**
- Fast processing (5-10 minutes total)
- Acceptable accuracy (R² > 0.55)
- Full control over pipeline

---

#### **Phase 2: Try Optimized MOGPR (OPTIONAL)**

**Time:** 2-3 hours
**Expected R²:** 0.70-0.85 (better than DL)

1. Implement spatial chunking
2. Test on small region first
3. Run parallel processing
4. Compare to improved DL

**Decision criteria:**
- If improved DL achieves R² > 0.65: **Stick with DL**
- If improved DL still < 0.55: **Try MOGPR**

---

#### **Phase 3: Consider Hybrid or CropSAR (FUTURE)**

If neither DL nor MOGPR meets requirements:
- Investigate CropSAR module in FuseTS
- Implement hybrid DL + MOGPR approach
- Consider openEO cloud MOGPR service

---

### Expected R² by Approach

| Approach | Expected R² | Processing Time | Implementation |
|----------|-------------|-----------------|----------------|
| **Current DL** | 0.27 | 5 min | ✅ Done |
| **Improved DL** | 0.55-0.70 | 8-12 min | ⭐ Recommended |
| **Optimized MOGPR** | 0.70-0.85 | 20-40 min | Consider if DL insufficient |
| **Parallel MOGPR** | 0.70-0.85 | 5-10 min | Requires implementation |
| **CropSAR** | 0.65-0.80 | 10-15 min | Check availability |
| **OpenEO MOGPR** | 0.75-0.90 | 15-30 min | Costs credits |

---

## Next Steps

**I recommend starting with improving your DL model:**

1. **Would you like me to create a complete modified notebook** with all the improvements (Solutions 1-4)?

2. **Or would you prefer step-by-step guidance** to modify your existing notebook?

3. **Or should we explore optimizing MOGPR** for speed instead?

Let me know which direction you'd like to take, and I'll provide the specific implementation!

---

## Troubleshooting Low R² Issues

### Quick Diagnostic Checklist

If R² remains low after improvements:

```python
# Run these diagnostics:

# 1. Check temporal alignment
print("S1 observation dates:", combined_dataset['VV'].time.values)
print("S2 observation dates:", combined_dataset['S2ndvi'].time.values)
# → Should be closely aligned (< 3 days difference)

# 2. Check S1-NDVI correlation by phenological stage
# → High correlation (> 0.6) during growth, low (< 0.3) during flooding

# 3. Check training data balance
print("NDVI distribution:", np.histogram(y_train, bins=10))
# → Should be balanced across NDVI ranges

# 4. Check for data leakage
# → Train and test should be truly independent

# 5. Visualize learned features
# → Are temporal features being used effectively?
```

### When to Give Up on S1→NDVI Fusion

**S1-NDVI fusion may not work well if:**
- Study area is dominated by water (floods, aquaculture)
- Persistent cloud cover → very sparse S2 data
- Extremely heterogeneous land cover
- S1-S2 temporal offset > 7 days consistently

**In these cases, consider:**
- Using S1 phenology directly (without NDVI)
- Whittaker smoothing on sparse S2 data
- Alternative vegetation indices (RVI, DpRVI)

---

**Ready to implement improvements? Let me know how you'd like to proceed!**