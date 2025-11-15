# IndexError Fix V2: Empty Array in Feature Filtering

## New Error Location

After fixing the `download_times` issue, a new `IndexError` appeared at:

```
File ~/work/FuseTS/improved_s1_ndvi_fusion_v2.py:304, in prepare_enhanced_features_v2
    304 volatility_99 = np.percentile(VV_volatility[mask_finite], 99)
IndexError: index -1 is out of bounds for axis 0 with size 0
```

## Root Cause

The `mask_finite` filter is producing an **empty array**, meaning:
- Either `X_all` contains no finite values
- Or `NDVI_full` contains no finite values
- Or the intersection is empty

This happens when `combined_dataset` is empty, corrupted, or all data has been masked out.

## Fix Applied

Updated `improved_s1_ndvi_fusion_v2.py` lines 290-326:

### 1. Added Early Validation (lines 297-311)

```python
if verbose:
    print(f"   After finite check: {np.sum(mask_finite):,} / {len(mask_finite):,} samples")

if np.sum(mask_finite) == 0:
    raise ValueError(
        "❌ CRITICAL: No finite values found in data!\n"
        "   This usually means:\n"
        "   1. combined_dataset contains only NaN/Inf values\n"
        "   2. Data loading failed silently\n"
        "   3. All data was masked out during preprocessing\n"
        f"   X_all shape: {X_all.shape}\n"
        f"   Finite X samples: {np.sum(np.all(np.isfinite(X_all), axis=1)):,}\n"
        f"   Finite NDVI samples: {np.sum(np.isfinite(NDVI_full)):,}"
    )
```

**Purpose**: Detect empty data early with informative error message

### 2. Added Safety Check for Percentile (lines 319-326)

```python
# Remove samples with extreme volatility (likely bad data)
# Only compute percentile if we have valid data
if np.sum(mask_finite) > 0:
    volatility_99 = np.percentile(VV_volatility[mask_finite], 99)
    mask_volatility = (VV_volatility < volatility_99) & (VH_volatility < volatility_99)
else:
    # Fallback: no volatility filtering if no finite data
    mask_volatility = np.ones(len(VV_volatility), dtype=bool)
```

**Purpose**: Prevent IndexError when computing percentile on empty array

## Diagnostic Tools Created

### 1. DIAGNOSTIC_CELL.py

**Usage**: Run this in a notebook cell **before** the fusion training:

```python
exec(open('DIAGNOSTIC_CELL.py').read())
```

**What it checks**:
- ✅ `combined_dataset` exists and has correct structure
- ✅ All required variables (VV, VH, S2ndvi) are present
- ✅ Each variable has finite values
- ✅ Sufficient overlap between S1 and S2 data
- ✅ NDVI values are in valid range [-1, 1]
- ✅ Enough samples for training (recommends 100k+)

**Output example**:
```
🔍 COMBINED_DATASET DIAGNOSTIC
================================================================================
✅ combined_dataset exists
📊 Structure:
   Dimensions: {'t': 62, 'y': 671, 'x': 893}
   Variables: ['VV', 'VH', 'S2ndvi']

📈 Data Quality Check:
   VV:
      Shape: (62, 671, 893)
      Total elements: 37,155,166
      Finite: 8,500,000 (22.9%)
      Range: [-25.4321, -5.1234] dB

🔗 Overlap Analysis:
   Total pixels: 37,155,166
   ALL THREE finite: 6,200,000 (16.7%)

📋 Diagnosis:
   ✅ GOOD: 6,200,000 samples available
   This should be sufficient for training
```

### 2. diagnose_dataset.py

**Usage**: More detailed diagnostics module:

```python
from diagnose_dataset import diagnose_dataset
diagnose_dataset(combined_dataset)
```

**Additional checks**:
- Per-variable statistics (mean, std, range)
- Per-time-slice coverage analysis
- Identification of empty time periods
- Spatial coverage patterns

## How to Use These Fixes

### Step 1: Check Your Data

Add this cell **RIGHT AFTER** loading data:

```python
# DIAGNOSTIC CELL - Check data quality
exec(open('DIAGNOSTIC_CELL.py').read())
```

### Step 2: Interpret Results

**If you see:**

```
❌ CRITICAL ERROR: No pixels have all three variables!
```

**Actions:**
1. Re-run `LOAD_LOCAL_TIFS_CELL.py` and check for errors
2. Verify GeoTIFF files exist: `ls -lh gee_assets_download/*.tif`
3. Check file contents: `gdalinfo gee_assets_download/period_01.tif`
4. Temporarily comment out masking cell to see if mask is too restrictive

**If you see:**

```
⚠️ WARNING: Very few valid samples (<10,000)
```

**Actions:**
1. Expand study area (reduce masking)
2. Check paddy mask overlap with data
3. Use more time periods
4. Relax quality filters

**If you see:**

```
❌ ERROR: NDVI outside valid range [-1, 1]!
```

**Actions:**
1. Check band order in LOAD_LOCAL_TIFS_CELL.py (lines 45-47)
2. Verify GeoTIFF export from GEE had correct band order
3. Use `gdalinfo -stats period_01.tif` to check band statistics

### Step 3: Run Training

Only after diagnostics pass:

```python
model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(
    combined_dataset,
    batch_size=256_000,
    learning_rate=0.001,
    epochs=150,
    warmup_epochs=5,
    val_split=0.2,
    verbose=True
)
```

## Common Issues and Solutions

### Issue 1: All Data is NaN

**Symptoms**:
```
Finite: 0 (0.0%)
❌ NO FINITE VALUES!
```

**Solution**:
- Check if GeoTIFF files actually contain data
- Verify rioxarray can open files correctly
- Check if masking removed all pixels

### Issue 2: Wrong NDVI Range

**Symptoms**:
```
NDVI Range: [-25.43, -5.12]
❌ ERROR: NDVI outside valid range!
```

**Solution**:
Band order is wrong. In `LOAD_LOCAL_TIFS_CELL.py`, swap bands:

```python
# If S2ndvi looks like backscatter, try:
VV = data.sel(band=3).drop('band')      # was 1
VH = data.sel(band=2).drop('band')      # was 2
S2ndvi = data.sel(band=1).drop('band')  # was 3
```

### Issue 3: Very Few Valid Samples

**Symptoms**:
```
ALL THREE finite: 5,000 (0.1%)
⚠️ WARNING: Very few valid samples
```

**Solutions**:

1. **Check temporal overlap**:
   ```python
   # Count valid pixels per variable
   print(f"VV valid: {np.sum(np.isfinite(combined_dataset['VV'].values))}")
   print(f"VH valid: {np.sum(np.isfinite(combined_dataset['VH'].values))}")
   print(f"NDVI valid: {np.sum(np.isfinite(combined_dataset['S2ndvi'].values))}")
   ```

2. **Relax paddy mask**:
   ```python
   # Temporarily skip masking to test
   # combined_dataset = combined_dataset.where(paddy_mask)  # Comment out
   ```

3. **Check CRS alignment** in masking cell - shapefile and data must have same CRS

## Files Modified

1. **`improved_s1_ndvi_fusion_v2.py`** (lines 290-326)
   - Added early validation for empty data
   - Added safety check for percentile calculation
   - Improved error messages

2. **`LOAD_LOCAL_TIFS_CELL.py`** (from previous fix)
   - Added `download_times` tracking

## Files Created

1. **`DIAGNOSTIC_CELL.py`** - Quick data quality check for notebooks
2. **`diagnose_dataset.py`** - Detailed diagnostic module
3. **`INDEXERROR_FIX_V2.md`** - This document

## Quick Reference

```python
# 1. Load data
exec(open('LOAD_LOCAL_TIFS_CELL.py').read())

# 2. Run diagnostics (NEW!)
exec(open('DIAGNOSTIC_CELL.py').read())

# 3. If diagnostics pass, run training
model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(...)
```

## Status

✅ **Fixed**: IndexError in percentile calculation
✅ **Added**: Early data validation with clear error messages
✅ **Created**: Diagnostic tools to identify data issues
📋 **Action Required**: User must run diagnostics to identify specific data problem

---

**Date**: 2025-11-13
**Version**: 2.0
