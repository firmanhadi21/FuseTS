# Integration Guide: Improved S1→NDVI Model

## Quick Start

You have two options to integrate the improved model into your workflow:

---

## Option 1: Direct Integration (Recommended)

Add this code to your `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb` notebook:

### **Step 1: Import the improved fusion module**

Add a new cell after your data loading cells (after Cell 11):

```python
# Import improved fusion module
import sys
sys.path.insert(0, '/home/unika_sianturi/work/FuseTS')

from improved_s1_ndvi_fusion import run_improved_fusion

print("✅ Improved fusion module loaded!")
```

### **Step 2: Run the improved fusion pipeline**

Replace your existing training cells (Cells 24-33) with this single cell:

```python
# ============================================================================
# RUN IMPROVED S1→NDVI FUSION
# ============================================================================

# Your combined_dataset should already be loaded and masked
# This will:
# 1. Create enhanced features (10 features instead of 3)
# 2. Train improved model (5 layers with batch norm + dropout)
# 3. Evaluate with comprehensive metrics
# 4. Generate visualization

model, predictions, metrics, scaler, history = run_improved_fusion(
    combined_dataset,
    batch_size=256_000,      # Large batch for H100
    learning_rate=0.001,     # Conservative learning rate
    epochs=150,              # More epochs (with early stopping)
    verbose=True
)

# Display results
print(f"\n🎯 Performance Improvement:")
print(f"   Baseline R²: 0.272")
print(f"   Improved R²: {metrics['r2']:.4f}")
print(f"   Improvement: {(metrics['r2'] - 0.272) / 0.272 * 100:.1f}%")
```

### **Step 3: Use predictions for gap-filling**

Continue with your existing phenology workflow:

```python
# Reshape predictions back to spatial format
n_times, n_y, n_x = combined_dataset['VV'].shape
mask_valid = ~np.isnan(combined_dataset['S2ndvi'].values.flatten())

# Create gap-filled NDVI array
ndvi_dl = np.full((n_times * n_y * n_x,), np.nan)
ndvi_dl[mask_valid] = predictions

# Reshape to 3D
ndvi_dl_3d = ndvi_dl.reshape(n_times, n_y, n_x)

# Add to dataset
combined_dataset['S2ndvi_DL'] = (['t', 'y', 'x'], ndvi_dl_3d)

print("✅ Gap-filled NDVI added to dataset!")

# Continue with phenology extraction (your existing cells)
```

---

## Option 2: Step-by-Step Integration

If you prefer more control, integrate each component separately:

### **A. Replace Feature Engineering (Cell 25)**

```python
# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

from improved_s1_ndvi_fusion import prepare_enhanced_features

print("Creating enhanced features with temporal context...")

X_all, y_all, mask_valid, feature_names = prepare_enhanced_features(
    combined_dataset,
    verbose=True
)

# Extract valid samples
X_train = X_all[mask_valid]
y_train = y_all[mask_valid]

print(f"\n✅ Features ready:")
print(f"   Input features: {len(feature_names)}")
print(f"   Training samples: {len(X_train):,}")
print(f"   Feature names: {feature_names}")
```

### **B. Replace Model Definition (Cell 25)**

```python
# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

from improved_s1_ndvi_fusion import ImprovedS1NDVIModel

# Initialize improved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedS1NDVIModel(input_dim=len(feature_names)).to(device)

# Check model
total_params, trainable_params = model.count_parameters()
print(f"Model: ImprovedS1NDVIModel")
print(f"Total parameters: {total_params:,}")
print(f"Device: {device}")
```

### **C. Replace Training Loop (Cell 25)**

```python
# ============================================================================
# IMPROVED TRAINING STRATEGY
# ============================================================================

from improved_s1_ndvi_fusion import train_improved_model

print("Training improved model...")

model, scaler, history = train_improved_model(
    X_train, y_train,
    batch_size=256_000,
    learning_rate=0.001,
    epochs=150,
    patience=20,
    device=device,
    verbose=True
)

print("✅ Training complete!")
print(f"   Best loss: {min(history['train_loss']):.6f}")
print(f"   Epochs trained: {len(history['train_loss'])}")
```

### **D. Replace Evaluation (Cell 27)**

```python
# ============================================================================
# ENHANCED EVALUATION
# ============================================================================

from improved_s1_ndvi_fusion import evaluate_model, plot_evaluation

print("Evaluating model...")

predictions, metrics = evaluate_model(
    model, X_train, y_train, scaler,
    history['y_mean'], history['y_std'],
    device=device,
    verbose=True
)

# Create comprehensive plots
plot_evaluation(y_train, predictions, metrics, history)

print(f"\n✅ Evaluation complete!")
print(f"   R² Score: {metrics['r2']:.4f}")
print(f"   Improvement: {metrics['r2'] - 0.272:.4f} ({(metrics['r2'] - 0.272) / 0.272 * 100:.1f}%)")
```

---

## Expected Timeline

### With Your H100 GPU:

| Step | Time | Description |
|------|------|-------------|
| **Import module** | < 5 seconds | Load improved_s1_ndvi_fusion.py |
| **Feature engineering** | 30-60 seconds | Create 10 features with temporal context |
| **Model training** | 5-8 minutes | 150 epochs with early stopping |
| **Evaluation** | 30-60 seconds | Generate predictions and metrics |
| **Visualization** | 10-20 seconds | Create comprehensive plots |
| **TOTAL** | **7-10 minutes** | Complete improved pipeline |

---

## Expected Results

### Performance Improvement:

```
Baseline (your current model):
  R² = 0.272
  MAE = 0.093

Expected with improvements:
  R² = 0.55 - 0.70  (100-150% improvement)
  MAE = 0.06 - 0.08  (15-35% improvement)

Best case:
  R² > 0.70  (publication quality)
```

---

## Troubleshooting

### Issue 1: Import Error

```python
# If you get: ModuleNotFoundError: No module named 'improved_s1_ndvi_fusion'

# Solution: Add the path explicitly
import sys
sys.path.insert(0, '/home/unika_sianturi/work/FuseTS')
from improved_s1_ndvi_fusion import run_improved_fusion
```

### Issue 2: CUDA Out of Memory (unlikely with H100)

```python
# If you get CUDA OOM error, reduce batch size:

model, predictions, metrics, scaler, history = run_improved_fusion(
    combined_dataset,
    batch_size=128_000,  # Reduced from 256K
    epochs=150,
    verbose=True
)
```

### Issue 3: Training Too Slow

```python
# Check GPU is being used:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Expected on H100:
# CUDA available: True
# GPU: NVIDIA H100
```

### Issue 4: R² Still Low (< 0.4)

This suggests fundamental issues with S1-NDVI relationship in your data:

```python
# Diagnostic: Check temporal alignment
print("Temporal alignment check:")
for t in range(5):
    print(f"Period {t+1}:")
    print(f"  VV valid: {np.sum(~np.isnan(combined_dataset['VV'][t]))}")
    print(f"  VH valid: {np.sum(~np.isnan(combined_dataset['VH'][t]))}")
    print(f"  NDVI valid: {np.sum(~np.isnan(combined_dataset['S2ndvi'][t]))}")

# If NDVI has very few valid pixels (< 10%), temporal misalignment likely
```

---

## What to Do After Integration

### 1. Check Results

After running the improved model, check the evaluation plot:

- **R² > 0.7:** Excellent! Proceed with phenology extraction
- **R² = 0.5-0.7:** Good! Usable for most applications
- **R² = 0.3-0.5:** Fair. Consider optimizing MOGPR instead
- **R² < 0.3:** Poor. Review diagnostic section below

### 2. Compare with Baseline

```python
# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline (your original model - if you saved it)
axes[0].scatter(y_test_baseline, pred_baseline, alpha=0.3, s=1)
axes[0].plot([-1, 1], [-1, 1], 'r--')
axes[0].set_title(f'Baseline (R²=0.272)')
axes[0].set_xlabel('True NDVI')
axes[0].set_ylabel('Predicted NDVI')

# Improved model
axes[1].scatter(y_test, predictions, alpha=0.3, s=1)
axes[1].plot([-1, 1], [-1, 1], 'r--')
axes[1].set_title(f'Improved (R²={metrics["r2"]:.3f})')
axes[1].set_xlabel('True NDVI')
axes[1].set_ylabel('Predicted NDVI')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
```

### 3. Proceed with Phenology Extraction

If R² > 0.5, continue with your existing phenology workflow:

```python
# Your existing phenology code (Cell 35+)
from fusets.analytics import phenology

# Use the improved gap-filled NDVI
ndvi_for_phenology = combined_dataset['S2ndvi_DL'].rename({'t': 'time'})

# Extract phenology metrics
phenology_metrics = phenology(ndvi_for_phenology)

# Visualize and export results
# (your existing code)
```

---

## Advanced: Predict on New Data

Once trained, you can use the model to predict NDVI for new S1 observations:

```python
# ============================================================================
# PREDICT NDVI FOR NEW S1 DATA
# ============================================================================

def predict_ndvi(new_s1_data, model, scaler, y_mean, y_std):
    """
    Predict NDVI from new Sentinel-1 data.

    Args:
        new_s1_data: xarray Dataset with VV, VH (same format as training)
        model: Trained model
        scaler: Feature scaler from training
        y_mean, y_std: Target normalization parameters

    Returns:
        predicted_ndvi: xarray DataArray with predicted NDVI
    """
    # Prepare features
    X_new, _, mask_valid, _ = prepare_enhanced_features(
        new_s1_data,
        verbose=False
    )

    # Normalize
    X_new_norm = scaler.transform(X_new[mask_valid])

    # Predict
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_new_norm).to(device)
        pred_norm = model(X_tensor).cpu().numpy().flatten()

    # Denormalize
    pred = pred_norm * y_std + y_mean
    pred = np.clip(pred, -1, 1)

    # Reshape back to original dimensions
    n_times, n_y, n_x = new_s1_data['VV'].shape
    pred_full = np.full((n_times * n_y * n_x,), np.nan)
    pred_full[mask_valid] = pred

    # Convert to xarray
    predicted_ndvi = xr.DataArray(
        pred_full.reshape(n_times, n_y, n_x),
        dims=['t', 'y', 'x'],
        coords={
            't': new_s1_data.t,
            'y': new_s1_data.y,
            'x': new_s1_data.x
        },
        name='NDVI_predicted'
    )

    return predicted_ndvi


# Example usage:
# ndvi_2024 = predict_ndvi(s1_data_2024, model, scaler,
#                          history['y_mean'], history['y_std'])
```

---

## Summary

### What You've Gained:

✅ **Better features:** 3 → 10 features (temporal context included)
✅ **Better architecture:** 3 → 5 layers with batch norm and dropout
✅ **Better training:** Learning rate scheduling, early stopping, mixed precision
✅ **Better evaluation:** Comprehensive metrics by NDVI range
✅ **Fast processing:** 7-10 minutes on H100 (vs hours for MOGPR)

### Expected R² Improvement:

- **Current:** R² = 0.272 (poor)
- **Expected:** R² = 0.55-0.70 (good to excellent)
- **Best case:** R² > 0.70 (publication quality)

### Next Steps:

1. **Integrate into your notebook** (Option 1 recommended)
2. **Run the improved pipeline** (7-10 minutes)
3. **Check R² score:**
   - If > 0.5: Proceed with phenology extraction
   - If < 0.5: Consider optimized MOGPR or investigate data quality
4. **Compare results** with your baseline
5. **Use gap-filled NDVI** for phenology analysis

---

## Need Help?

If you encounter issues or R² is still low:

1. Check the diagnostic sections in `IMPROVING_S1_NDVI_MODEL_PERFORMANCE.md`
2. Run temporal alignment checks
3. Examine per-NDVI-range performance
4. Consider trying optimized MOGPR instead

**Ready to integrate? Let me know if you want me to create a complete modified notebook or if you'll integrate step-by-step!**
