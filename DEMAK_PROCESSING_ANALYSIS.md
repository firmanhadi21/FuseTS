# Demak Full Area Processing - Analysis & Solutions

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Hardware Context](#hardware-context)
3. [Data Dimensions & Memory Requirements](#data-dimensions--memory-requirements)
4. [Why Processing Fails - Root Causes](#why-processing-fails---root-causes)
5. [Detailed Failure Point Analysis](#detailed-failure-point-analysis)
6. [Solutions Ranked by Priority](#solutions-ranked-by-priority)
7. [OpenEO vs Local Processing Comparison](#openeo-vs-local-processing-comparison)
8. [Implementation Guides](#implementation-guides)
9. [Troubleshooting Checklist](#troubleshooting-checklist)

---

## Executive Summary

**Problem:** User experiences consistent failures when processing the full Demak area (671×893 pixels, 62 temporal periods) in `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb`.

**Key Finding:** With an **NVIDIA H100 GPU** (80GB VRAM), hardware is NOT the bottleneck. The failures are likely due to:
1. **Data download issues** from Google Earth Engine (62 sequential asset requests)
2. **Over-aggressive masking** removing too many valid pixels
3. **Silent failures** in error handling (notebook continues despite errors)
4. **Missing data validation** before critical processing steps

**Quick Answer:** The H100 has more than enough power (80GB VRAM vs 2GB required). Issues are in the data pipeline, not computation.

---

## Hardware Context

### Your Setup: NVIDIA H100

**Specifications:**
- **GPU VRAM:** 80GB
- **CUDA Cores:** 14,592 (Hopper architecture)
- **Tensor Cores:** 456 (4th generation)
- **Memory Bandwidth:** 3TB/s
- **FP32 Performance:** ~51 TFLOPS

**Required for Demak Processing:**
- **GPU VRAM:** 1-2GB (your H100 has 40-80× more than needed)
- **CPU RAM:** 8GB minimum (trivial for H100 systems)
- **Training Time:** 2-5 minutes on H100 (vs 15-30 min on consumer GPUs)

### Implications

**GPU is NOT your bottleneck.** With an H100, you could process:
- 40× larger areas than Demak
- Multiple regions in parallel
- Full Java Island if properly chunked

**Therefore, failures must be due to:**
1. Data acquisition/network issues
2. Software bugs or silent errors
3. Data quality problems (missing/corrupted data)
4. Logical errors in processing pipeline

---

## Data Dimensions & Memory Requirements

### Full Demak Area

```
Spatial Resolution: 50 meters
Spatial Extent: 671 × 893 pixels = 599,203 pixels
Geographic Size: ~33.5 km × 44.7 km
Temporal Coverage: 62 periods (12-day composites)
Time Range: November 1, 2023 - October 31, 2025 (2 years)
Bands: 3 (VV, VH, S2ndvi)
Total Datapoints: 62 × 671 × 893 × 3 = 111,451,758
```

### Memory Footprint

| Component | Size (GB) | H100 Capacity | Headroom |
|-----------|-----------|---------------|----------|
| Raw Data (float32) | 0.42 | 80 GB | 190× |
| Training Arrays | 0.78 | 80 GB | 102× |
| Peak RAM Usage | 2.71 | ~256-512 GB | 94-188× |
| GPU Training | 1.02 | 80 GB | 78× |
| Model Parameters | 0.006 | 80 GB | 13,333× |

**Conclusion:** H100 has 78-190× more capacity than required. Memory is NOT the issue.

---

## Why Processing Fails - Root Causes

### Given your H100 hardware, failures are NOT due to:
- ❌ Insufficient GPU memory
- ❌ Slow training speed
- ❌ Computational bottlenecks

### Actual Root Causes (in order of likelihood):

#### **1. Google Earth Engine Data Download Failures** (Most Likely)

**Problem:**
- Notebook downloads 62 sequential GEE assets
- Each asset: VV, VH, S2ndvi bands at 50m resolution
- Total download: ~400-500 MB
- Any network interruption = incomplete data

**Evidence from notebook:**
```python
# Cell 9 - GEE Asset Loading
for period in range(1, 63):
    try:
        img_data = geemap.ee_to_numpy(asset_img, ...)
    except Exception as e:
        print(f"⚠️ Warning: Could not load period {period}: {e}")
        continue  # ⚠️ SKIPS FAILED PERIOD AND CONTINUES!
```

**Critical Issue:** Notebook continues even if 50% of periods fail to download!

**Symptoms you might see:**
```
⚠️ Warning: Could not load period 15: Connection timeout
⚠️ Warning: Could not load period 23: API rate limit exceeded
⚠️ Warning: Could not load period 41: Authentication expired
✅ Successfully loaded 45 out of 62 periods
```

Then later:
```
❌ ERROR: Not enough valid training data!
Valid training samples: 452 / 111,451,758 (0.0004%)
```

---

#### **2. Over-Aggressive Data Masking**

**Problem:**
Paddy field mask + cloud masking + NaN filtering may remove 90-99% of pixels.

**Masking Pipeline:**
```
Original pixels:       599,203 (100%)
After paddy mask:      ~120,000 (20%)  ← Agricultural areas only
After cloud mask:      ~24,000 (4%)    ← S2 NDVI has clouds
After NaN filtering:   ~1,200 (0.2%)   ← May be below training threshold
```

**Evidence from notebook:**
```python
# Cell 11 - Paddy field mask application
paddy_mask = geometry_mask([feature['geometry'] for feature in gdf.geometry], ...)
combined_dataset = combined_dataset.where(~paddy_mask)

# Cell 25 - Training data filtering
mask_valid = ~np.isnan(VV_full) & ~np.isnan(VH_full) & ~np.isnan(NDVI_full)
mask_valid &= ~np.isinf(VV_full) & ~np.isinf(VH_full) & ~np.isinf(NDVI_full)
n_valid = np.sum(mask_valid)

if n_valid < 1000:
    print("❌ ERROR: Not enough valid training data!")
    # ⚠️ BUT NOTEBOOK CONTINUES ANYWAY!
```

**Critical Issue:** Validation checks exist but don't stop execution!

---

#### **3. Silent Failures in Error Handling**

**Problem:** Notebook has validation checks but doesn't halt on critical errors.

**Example 1: Data Loading**
```python
# Loads 45 out of 62 periods successfully
# ⚠️ Continues with incomplete time series
# Results in poor temporal coverage and gaps
```

**Example 2: Training Data Validation**
```python
if n_valid < 1000:
    print("❌ ERROR: Not enough valid training data!")
    # ⚠️ No sys.exit() or raise Exception
    # Continues to train on garbage data
```

**Example 3: Phenology Calculation**
```python
# No validation that fused_full['S2ndvi_DL'] has valid data
phenology_metrics = phenology(ndvi_for_phenology)
# ⚠️ May produce all-NaN results silently
```

---

#### **4. Missing Intermediate Validation**

**Problem:** Notebook doesn't validate data quality between steps.

**What's Missing:**

```python
# After GEE download - should check:
assert loaded_periods == 62, f"Only loaded {loaded_periods}/62 periods!"

# After masking - should check:
valid_ratio = np.sum(~combined_dataset.isnull()) / combined_dataset.size
assert valid_ratio > 0.10, f"Only {valid_ratio:.1%} valid data after masking!"

# After training - should check:
assert model_loss < 0.1, f"Model failed to converge: loss={model_loss}"

# Before phenology - should check:
coverage = np.mean(~np.isnan(ndvi_for_phenology))
assert coverage > 0.5, f"NDVI coverage too low: {coverage:.1%}"
```

**Current State:** These checks don't exist, so failures propagate silently.

---

## Detailed Failure Point Analysis

### Failure Mode Matrix

| Failure Point | Cell | Symptom | Stops Execution? | Current Handling |
|---------------|------|---------|------------------|------------------|
| **GEE Download Timeout** | 9 | "Could not load period X" | ❌ No | Skips period, continues |
| **GEE API Rate Limit** | 9 | "Too many requests" | ❌ No | Skips period, continues |
| **Empty Dataset After Mask** | 18, 20, 21 | "Dataset is EMPTY (all NaN)" | ❌ No | Prints warning, continues |
| **Insufficient Training Data** | 25 | "Not enough valid training data" | ❌ No | Prints error, continues |
| **Model Training Divergence** | 25 | Loss increases or NaN | ❌ No | No validation |
| **Prediction Produces NaN** | 27, 32 | All predictions = NaN | ❌ No | No validation |
| **Phenology Extraction Fails** | 35 | All SOS/EOS = NaN | ❌ No | No validation |

**Key Pattern:** All failures allow notebook to continue, resulting in garbage outputs.

---

### Detailed Analysis by Processing Stage

#### **Stage 1: GEE Asset Download (Cell 9)**

**Expected Behavior:**
```
Loading period 1/62... ✓
Loading period 2/62... ✓
...
Loading period 62/62... ✓
✅ Successfully loaded all 62 periods
```

**Actual Behavior (when failing):**
```
Loading period 1/62... ✓
Loading period 2/62... ✓
...
Loading period 15/62... ⚠️ Warning: Could not load period 15: Connection timeout
Loading period 16/62... ✓
...
Loading period 41/62... ⚠️ Warning: Could not load period 41: API rate limit
...
✅ Successfully loaded 60 out of 62 periods
```

**Impact:**
- Time series has gaps at periods 15 and 41
- MOGPR/smoothing algorithms struggle with irregular sampling
- Phenology detection may miss seasonal transitions

**Why This Happens:**
1. **Network instability:** GEE servers or user connection drops
2. **API rate limiting:** 62 sequential requests may hit rate limits
3. **Authentication expiry:** Long downloads may exceed token lifetime
4. **Asset permissions:** Some periods may have restricted access

**Detection:**
```python
# Check how many periods actually loaded
print(f"Expected: 62 periods")
print(f"Loaded: {len(combined_dataset.t)} periods")
print(f"Missing: {62 - len(combined_dataset.t)} periods")
```

---

#### **Stage 2: Data Masking (Cell 11)**

**Expected Behavior:**
```
Original pixels: 599,203
After paddy mask: 119,841 (20%)
Valid agricultural area preserved ✓
```

**Actual Behavior (when failing):**
```
Original pixels: 599,203
After paddy mask: 0 (0%)
❌ CRITICAL ERROR: Dataset is EMPTY (all NaN)!
```

**Why This Happens:**

1. **Inverted Mask:**
```python
# WRONG: This inverts the mask
combined_dataset = combined_dataset.where(paddy_mask)

# CORRECT:
combined_dataset = combined_dataset.where(~paddy_mask)
```

2. **CRS Mismatch:**
```python
# Shapefile in EPSG:4326 (lat/lon)
# Raster in EPSG:32749 (UTM Zone 49S)
# Mask doesn't align with data
```

3. **Geometry Outside Raster Bounds:**
```python
# Paddy shapefile covers different region than GEE assets
# Mask excludes all pixels
```

**Detection:**
```python
# After masking, check coverage
valid_pixels = np.sum(~combined_dataset['VV'].isnull())
total_pixels = combined_dataset['VV'].size
coverage = valid_pixels / total_pixels

print(f"Valid pixels after masking: {valid_pixels:,} ({coverage:.1%})")
assert coverage > 0.05, "Mask removed too many pixels!"
```

---

#### **Stage 3: Training Data Preparation (Cell 25)**

**Expected Behavior:**
```
Total pixels: 37,151,386
Valid training samples: 26,005,810 (70%)
✅ Sufficient data for training
```

**Actual Behavior (when failing):**
```
Total pixels: 37,151,386
Valid training samples: 452 (0.001%)
❌ ERROR: Not enough valid training data!
[Continues training anyway on 452 samples]
[Model produces garbage predictions]
```

**Why This Happens:**

1. **Cascading Mask Effects:**
   - Paddy mask removes 80% of pixels
   - Cloud mask removes 75% of remaining pixels
   - NaN filter removes 50% of remaining pixels
   - Result: 80% × 75% × 50% = 97% removed

2. **Incorrect NaN Handling:**
```python
# S2 NDVI has many NaN values (clouds)
# Filtering removes all pixels with ANY NaN in 62 timesteps
mask_valid = ~np.isnan(NDVI_full)  # ← Very strict!

# This means a pixel with 61 valid observations and 1 cloud is removed
```

3. **Inf Values Not Properly Handled:**
```python
# VV/VH in dB scale can have -inf values
# log(0) = -inf during RVI calculation
VV_full[np.isinf(VV_full)] = np.nan  # ← This is done, but...
# If ANY timestep has inf, entire pixel removed
```

**Detection:**
```python
# Detailed breakdown of data loss
print("\n📊 Data Quality Report:")
print(f"Total pixels: {VV_full.size:,}")
print(f"VV valid: {np.sum(~np.isnan(VV_full)):,} ({np.mean(~np.isnan(VV_full)):.1%})")
print(f"VH valid: {np.sum(~np.isnan(VH_full)):,} ({np.mean(~np.isnan(VH_full)):.1%})")
print(f"NDVI valid: {np.sum(~np.isnan(NDVI_full)):,} ({np.mean(~np.isnan(NDVI_full)):.1%})")
print(f"All three valid: {n_valid:,} ({n_valid/VV_full.size:.1%})")

# Per-period validation
for t in range(62):
    valid_t = np.sum(mask_valid.reshape(62, -1)[t, :])
    print(f"Period {t+1}: {valid_t:,} valid pixels")
```

---

#### **Stage 4: Model Training (Cell 25)**

**With H100, training should take 2-5 minutes even on full dataset.**

**Expected GPU Utilization:**
```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA H100        On   | 00000000:00:05.0 Off |                    0 |
| N/A   45C    P0    150W / 350W |   1024MiB / 81920MiB |     12%      Default |
+-------------------------------+----------------------+----------------------+

GPU utilization: 5-15% (underutilized on H100!)
Memory usage: 1GB / 80GB (1.25%)
```

**Why H100 is Underutilized:**
- Small model (1.5M parameters)
- Batch size too small (65,536 samples)
- Single GPU, no parallelization
- Data loading bottleneck (CPU → GPU transfer)

**Potential Improvements for H100:**
```python
# Increase batch size to utilize H100's memory
batch_size = 1_000_000  # 1M samples per batch (H100 can handle it)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Enable TF32 (H100's tensor cores)
torch.set_float32_matmul_precision('high')

# Multi-stream data loading
train_loader = DataLoader(..., num_workers=8, pin_memory=True)
```

**Expected Speedup with Optimizations:**
- Current: 2-5 minutes
- Optimized: 30-60 seconds

---

#### **Stage 5: Prediction/Gap-filling (Cell 27, 32)**

**Expected Behavior:**
```
Predicting NDVI from S1 for all pixels...
Batch 1/6: 100,000 pixels ✓
Batch 2/6: 100,000 pixels ✓
...
Batch 6/6: 99,203 pixels ✓
✅ Gap-filling complete: 98.5% coverage improvement
```

**Actual Behavior (when failing):**
```
Predicting NDVI from S1 for all pixels...
Batch 1/6: 100,000 pixels ✓ (all NaN)
Batch 2/6: 100,000 pixels ✓ (all NaN)
...
⚠️ Warning: All predictions are NaN
```

**Why This Happens:**

1. **Model Trained on Garbage Data:**
   - If training had insufficient valid samples (< 1,000)
   - Model didn't learn meaningful relationships
   - Produces NaN or random predictions

2. **Input Data Normalization Issues:**
```python
# Training used normalized data
X_train_norm = (X_train - X_train.mean()) / X_train.std()

# Prediction MUST use same normalization
# If mean/std calculated on bad data → wrong normalization → NaN outputs
```

3. **Model in Wrong Mode:**
```python
# If model.eval() not called
# Dropout layers active during inference
# May produce inconsistent/NaN results
```

**Detection:**
```python
# After prediction, validate
ndvi_dl = fused_full['S2ndvi_DL'].values
print(f"NDVI_DL range: [{np.nanmin(ndvi_dl):.3f}, {np.nanmax(ndvi_dl):.3f}]")
print(f"NDVI_DL valid: {np.sum(~np.isnan(ndvi_dl)):,} ({np.mean(~np.isnan(ndvi_dl)):.1%})")
print(f"NDVI_DL mean: {np.nanmean(ndvi_dl):.3f}")

assert np.sum(~np.isnan(ndvi_dl)) > 0.5 * ndvi_dl.size, "Prediction failed!"
assert -1 <= np.nanmin(ndvi_dl) <= 1, "NDVI values out of range!"
```

---

#### **Stage 6: Phenology Extraction (Cell 35)**

**Expected Behavior:**
```
Extracting phenological metrics...
✅ SOS detected for 95% of pixels
✅ EOS detected for 93% of pixels
✅ 2-3 seasons identified per pixel
```

**Actual Behavior (when failing):**
```
Extracting phenological metrics...
⚠️ Warning: SOS = NaN for 100% of pixels
⚠️ Warning: EOS = NaN for 100% of pixels
❌ No valid phenology metrics extracted
```

**Why This Happens:**

1. **Insufficient Temporal Coverage:**
```python
# Phenology detection requires:
# - At least 12 observations per season
# - Clear seasonal pattern (peak + trough)
# - Temporal continuity (no large gaps)

# If input NDVI has:
# - < 30 valid observations out of 62 periods
# - Missing critical periods (planting/harvest)
# - All values = NaN from failed prediction
# → Phenology extraction fails
```

2. **NDVI Range Issues:**
```python
# Phenology algorithms expect:
# - NDVI range: 0.2 - 0.9 for vegetation
# - Clear amplitude: peak - trough > 0.2
# - Smooth time series (no noise spikes)

# If NDVI has:
# - All values < 0.1 (water/bare soil)
# - No seasonal variation (flat line)
# - Extreme noise or outliers
# → No peaks/troughs detected
```

3. **Algorithm Parameters:**
```python
# FuseTS phenology() uses TIMESAT methodology
# Default parameters may not suit Indonesian paddy:
# - Season detection threshold
# - Minimum season length
# - Amplitude threshold

# Indonesian paddy characteristics:
# - 3 seasons per year (vs 1-2 in temperate regions)
# - Shorter growing cycles (90-120 days)
# - Lower NDVI amplitude due to persistent cloud cover
```

**Detection:**
```python
# Before phenology extraction, validate input
print("\n📊 Pre-Phenology Validation:")
print(f"Time dimension: {len(ndvi_for_phenology.time)}")
print(f"Valid observations per pixel: {np.sum(~np.isnan(ndvi_for_phenology), axis=0).mean():.1f}")
print(f"NDVI range: [{np.nanmin(ndvi_for_phenology):.3f}, {np.nanmax(ndvi_for_phenology):.3f}]")
print(f"Mean amplitude: {np.nanstd(ndvi_for_phenology, axis=0).mean():.3f}")

# Check if data suitable for phenology
assert len(ndvi_for_phenology.time) >= 30, "Insufficient temporal coverage!"
assert np.nanmean(ndvi_for_phenology) > 0.2, "NDVI too low (non-vegetated)!"
```

---

## Solutions Ranked by Priority

### Given Your H100 Hardware

**Hardware is NOT the constraint.** Focus on data pipeline reliability.

---

### **Solution 1: Add Robust Error Handling & Validation** ⭐ HIGHEST PRIORITY

**Why:** This addresses the root cause - silent failures.

**Implementation:**

```python
# ============================================================================
# CELL 9A - Add after GEE download loop
# ============================================================================

print("\n" + "="*80)
print("📋 DATA DOWNLOAD VALIDATION")
print("="*80)

# Check how many periods loaded
expected_periods = 62
actual_periods = len(combined_dataset.t)
missing_periods = expected_periods - actual_periods

print(f"Expected periods: {expected_periods}")
print(f"Loaded periods: {actual_periods}")
print(f"Missing periods: {missing_periods}")

if missing_periods > 0:
    print(f"\n⚠️ WARNING: {missing_periods} periods failed to download")
    missing_idx = set(range(1, 63)) - set(combined_dataset.t.values)
    print(f"Missing period numbers: {sorted(missing_idx)}")

    if missing_periods > 5:
        raise ValueError(f"❌ CRITICAL: Too many missing periods ({missing_periods}/62). "
                         f"Check network connection and GEE authentication.")
    else:
        print(f"✓ Acceptable: Continuing with {actual_periods} periods")
else:
    print("✅ All periods loaded successfully!")

# ============================================================================
# CELL 11A - Add after paddy mask application
# ============================================================================

print("\n" + "="*80)
print("📋 MASKING VALIDATION")
print("="*80)

# Calculate coverage before and after
total_pixels = combined_dataset['VV'].size
valid_before = np.sum(~combined_dataset['VV'].isnull())
coverage_before = valid_before / total_pixels

print(f"Total pixels: {total_pixels:,}")
print(f"Valid pixels before mask: {valid_before:,} ({coverage_before:.1%})")

# After masking
valid_after = np.sum(~combined_dataset['VV'].isnull())
coverage_after = valid_after / total_pixels
removed = valid_before - valid_after
removal_rate = removed / valid_before

print(f"Valid pixels after mask: {valid_after:,} ({coverage_after:.1%})")
print(f"Removed by mask: {removed:,} ({removal_rate:.1%})")

# Validation thresholds
if coverage_after < 0.01:  # Less than 1% coverage
    raise ValueError(f"❌ CRITICAL: Mask removed too many pixels! "
                     f"Only {coverage_after:.2%} coverage remaining. "
                     f"Check mask alignment and CRS.")
elif coverage_after < 0.05:  # Less than 5% coverage
    print(f"⚠️ WARNING: Low coverage after masking ({coverage_after:.1%}). "
          f"Consider relaxing mask constraints.")
else:
    print(f"✅ Masking successful: {coverage_after:.1%} coverage retained")

# ============================================================================
# CELL 25A - Replace existing validation in training cell
# ============================================================================

print("\n" + "="*80)
print("📋 TRAINING DATA VALIDATION")
print("="*80)

# Detailed data quality report
total_samples = VV_full.size
print(f"Total samples: {total_samples:,}")

# Per-band validation
vv_valid = np.sum(~np.isnan(VV_full) & ~np.isinf(VV_full))
vh_valid = np.sum(~np.isnan(VH_full) & ~np.isinf(VH_full))
ndvi_valid = np.sum(~np.isnan(NDVI_full) & ~np.isinf(NDVI_full))

print(f"VV valid: {vv_valid:,} ({vv_valid/total_samples:.1%})")
print(f"VH valid: {vh_valid:,} ({vh_valid/total_samples:.1%})")
print(f"NDVI valid: {ndvi_valid:,} ({ndvi_valid/total_samples:.1%})")

# Combined validation
mask_valid = (~np.isnan(VV_full) & ~np.isnan(VH_full) & ~np.isnan(NDVI_full) &
              ~np.isinf(VV_full) & ~np.isinf(VH_full) & ~np.isinf(NDVI_full))
n_valid = np.sum(mask_valid)
valid_ratio = n_valid / total_samples

print(f"All bands valid: {n_valid:,} ({valid_ratio:.1%})")

# Validation thresholds
MIN_SAMPLES_ABSOLUTE = 100_000  # Minimum for H100 training
MIN_SAMPLES_RATIO = 0.01  # Minimum 1% valid data

print(f"\nValidation thresholds:")
print(f"  Minimum absolute samples: {MIN_SAMPLES_ABSOLUTE:,}")
print(f"  Minimum valid ratio: {MIN_SAMPLES_RATIO:.1%}")

if n_valid < MIN_SAMPLES_ABSOLUTE:
    raise ValueError(f"❌ CRITICAL: Insufficient training data! "
                     f"Only {n_valid:,} valid samples (need {MIN_SAMPLES_ABSOLUTE:,}). "
                     f"Check data quality and masking.")
elif valid_ratio < MIN_SAMPLES_RATIO:
    print(f"⚠️ WARNING: Low valid data ratio ({valid_ratio:.2%}). "
          f"Model may not train well.")
else:
    print(f"✅ Sufficient training data: {n_valid:,} samples ({valid_ratio:.1%})")

# Per-period validation
print(f"\n📊 Per-Period Data Quality:")
mask_reshaped = mask_valid.reshape(62, -1)
for t in range(62):
    valid_t = np.sum(mask_reshaped[t, :])
    valid_ratio_t = valid_t / mask_reshaped.shape[1]
    status = "✓" if valid_ratio_t > 0.1 else "⚠️"
    print(f"  Period {t+1:2d}: {valid_t:,} valid pixels ({valid_ratio_t:.1%}) {status}")

# ============================================================================
# CELL 27A - Add after prediction
# ============================================================================

print("\n" + "="*80)
print("📋 PREDICTION VALIDATION")
print("="*80)

ndvi_dl = fused_full['S2ndvi_DL'].values
total_predictions = ndvi_dl.size
valid_predictions = np.sum(~np.isnan(ndvi_dl))
valid_ratio = valid_predictions / total_predictions

print(f"Total predictions: {total_predictions:,}")
print(f"Valid predictions: {valid_predictions:,} ({valid_ratio:.1%})")
print(f"NDVI_DL range: [{np.nanmin(ndvi_dl):.3f}, {np.nanmax(ndvi_dl):.3f}]")
print(f"NDVI_DL mean: {np.nanmean(ndvi_dl):.3f} ± {np.nanstd(ndvi_dl):.3f}")

# Validation checks
if valid_ratio < 0.5:
    raise ValueError(f"❌ CRITICAL: Prediction failed! "
                     f"Only {valid_ratio:.1%} valid predictions. "
                     f"Model may not have trained properly.")

ndvi_min = np.nanmin(ndvi_dl)
ndvi_max = np.nanmax(ndvi_dl)
if ndvi_min < -1 or ndvi_max > 1:
    print(f"⚠️ WARNING: NDVI values out of expected range [-1, 1]")
    print(f"  Observed range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")
    print(f"  Clipping to valid range...")
    fused_full['S2ndvi_DL'] = fused_full['S2ndvi_DL'].clip(-1, 1)
else:
    print(f"✅ NDVI values within valid range")

print(f"✅ Prediction successful: {valid_ratio:.1%} coverage")

# ============================================================================
# CELL 35A - Add before phenology extraction
# ============================================================================

print("\n" + "="*80)
print("📋 PRE-PHENOLOGY VALIDATION")
print("="*80)

# Validate input time series
n_timesteps = len(ndvi_for_phenology.time)
print(f"Time dimension: {n_timesteps} observations")

# Per-pixel validation
valid_obs_per_pixel = np.sum(~np.isnan(ndvi_for_phenology), axis=0)
mean_valid_obs = valid_obs_per_pixel.mean()
min_valid_obs = valid_obs_per_pixel.min()
max_valid_obs = valid_obs_per_pixel.max()

print(f"Valid observations per pixel:")
print(f"  Mean: {mean_valid_obs:.1f}")
print(f"  Min: {min_valid_obs}")
print(f"  Max: {max_valid_obs}")

# NDVI statistics
ndvi_mean = np.nanmean(ndvi_for_phenology)
ndvi_std = np.nanstd(ndvi_for_phenology)
ndvi_min = np.nanmin(ndvi_for_phenology)
ndvi_max = np.nanmax(ndvi_for_phenology)

print(f"\nNDVI statistics:")
print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")
print(f"  Mean: {ndvi_mean:.3f} ± {ndvi_std:.3f}")

# Amplitude check (per-pixel temporal variability)
amplitude_per_pixel = np.nanstd(ndvi_for_phenology, axis=0)
mean_amplitude = np.nanmean(amplitude_per_pixel)
print(f"  Mean temporal amplitude: {mean_amplitude:.3f}")

# Validation thresholds
MIN_TIMESTEPS = 30
MIN_VALID_OBS_PER_PIXEL = 20
MIN_MEAN_NDVI = 0.2
MIN_AMPLITUDE = 0.05

print(f"\nValidation thresholds:")
print(f"  Minimum timesteps: {MIN_TIMESTEPS}")
print(f"  Minimum valid obs/pixel: {MIN_VALID_OBS_PER_PIXEL}")
print(f"  Minimum mean NDVI: {MIN_MEAN_NDVI}")
print(f"  Minimum amplitude: {MIN_AMPLITUDE}")

# Check thresholds
issues = []
if n_timesteps < MIN_TIMESTEPS:
    issues.append(f"Insufficient timesteps: {n_timesteps} < {MIN_TIMESTEPS}")

if mean_valid_obs < MIN_VALID_OBS_PER_PIXEL:
    issues.append(f"Insufficient valid obs/pixel: {mean_valid_obs:.1f} < {MIN_VALID_OBS_PER_PIXEL}")

if ndvi_mean < MIN_MEAN_NDVI:
    issues.append(f"NDVI too low (non-vegetated?): {ndvi_mean:.3f} < {MIN_MEAN_NDVI}")

if mean_amplitude < MIN_AMPLITUDE:
    issues.append(f"Amplitude too low (no seasonality?): {mean_amplitude:.3f} < {MIN_AMPLITUDE}")

if issues:
    print("\n⚠️ WARNINGS:")
    for issue in issues:
        print(f"  - {issue}")
    if len(issues) >= 3:
        raise ValueError("❌ CRITICAL: Input data unsuitable for phenology extraction. "
                         "Too many quality issues detected.")
    else:
        print("\n⚠️ Proceeding with caution...")
else:
    print("\n✅ Input data suitable for phenology extraction")
```

**Expected Impact:**
- Catches failures immediately when they occur
- Provides detailed diagnostics for debugging
- Prevents garbage outputs from propagating
- Clear error messages for troubleshooting

---

### **Solution 2: Implement Robust GEE Download with Retry Logic** ⭐ HIGH PRIORITY

**Why:** Network issues are likely the #1 cause of failures.

**Implementation:**

```python
# ============================================================================
# CELL 9B - Replace existing GEE download loop
# ============================================================================

import time
from typing import List, Tuple

def download_gee_asset_with_retry(
    asset_path: str,
    region: ee.Geometry,
    scale: int = 50,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[np.ndarray, bool]:
    """
    Download GEE asset with automatic retry on failure.

    Returns:
        (data, success): Numpy array and success flag
    """
    for attempt in range(max_retries):
        try:
            # Load asset
            asset_img = ee.Image(asset_path)

            # Download to numpy
            img_data = geemap.ee_to_numpy(
                asset_img,
                region=region,
                scale=scale,
                bands=['VV', 'VH', 'S2ndvi']
            )

            # Validate downloaded data
            if img_data is None or img_data.size == 0:
                raise ValueError("Downloaded data is empty")

            # Check for all-NaN bands
            for band_idx in range(img_data.shape[0]):
                if np.all(np.isnan(img_data[band_idx, :, :])):
                    raise ValueError(f"Band {band_idx} is all NaN")

            return img_data, True

        except Exception as e:
            print(f"  ⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")

            if attempt < max_retries - 1:
                print(f"  ⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  ❌ All retry attempts failed")
                return None, False

    return None, False


# Main download loop with progress tracking
print("="*80)
print("📥 DOWNLOADING GEE ASSETS WITH RETRY LOGIC")
print("="*80)

num_periods = 62
base_path = "projects/ee-geodeticengineeringundip/assets/FuseTS"
dataset_list = []
failed_periods = []
download_times = []

for period in range(1, num_periods + 1):
    period_str = f"Period_{period:02d}"
    asset_path = f"{base_path}/S1_S2_Nov2023_Oct2025_{period_str}"

    print(f"\n📦 Downloading period {period}/{num_periods}: {period_str}")

    start_time = time.time()
    img_data, success = download_gee_asset_with_retry(
        asset_path,
        region=REGION if USE_SMALL_REGION else None,
        scale=50,
        max_retries=3,
        retry_delay=5
    )
    elapsed = time.time() - start_time
    download_times.append(elapsed)

    if success:
        # Convert to xarray Dataset
        ds = xr.Dataset({
            'VV': (['y', 'x'], img_data[0, :, :]),
            'VH': (['y', 'x'], img_data[1, :, :]),
            'S2ndvi': (['y', 'x'], img_data[2, :, :])
        })
        ds = ds.assign_coords({'t': period})
        dataset_list.append(ds)
        print(f"  ✅ Success in {elapsed:.1f}s")
    else:
        failed_periods.append(period)
        print(f"  ❌ Failed after {elapsed:.1f}s")

# Summary
print("\n" + "="*80)
print("📊 DOWNLOAD SUMMARY")
print("="*80)
print(f"Total periods: {num_periods}")
print(f"Successfully downloaded: {len(dataset_list)}")
print(f"Failed: {len(failed_periods)}")
print(f"Success rate: {len(dataset_list)/num_periods:.1%}")
print(f"Total download time: {sum(download_times)/60:.1f} minutes")
print(f"Average time per period: {np.mean(download_times):.1f}s")

if failed_periods:
    print(f"\n❌ Failed periods: {failed_periods}")

    if len(failed_periods) > 10:  # More than 10 failures
        raise ValueError(f"Too many failed downloads ({len(failed_periods)}/{num_periods}). "
                         f"Check network connection and GEE authentication.")
    else:
        print(f"⚠️ WARNING: Continuing with {len(dataset_list)} periods")
        print(f"⚠️ Time series will have gaps at periods: {failed_periods}")
else:
    print("\n✅ All periods downloaded successfully!")

# Combine into single dataset
if len(dataset_list) == 0:
    raise ValueError("❌ CRITICAL: No data downloaded! Cannot proceed.")

combined_dataset = xr.concat(dataset_list, dim='t')
print(f"\n✅ Combined dataset shape: {combined_dataset.dims}")
```

**Expected Impact:**
- Automatic retry on network failures
- Detailed progress tracking
- Clear identification of failed periods
- Stops execution if too many failures
- Estimated completion time

---

### **Solution 3: Optimize for H100 Performance** (OPTIONAL)

**Why:** Your H100 is massively underutilized (1% GPU memory, <15% compute).

**Benefits:**
- 2-5 minutes → 30-60 seconds training time
- Process multiple regions in parallel
- Train larger/better models

**Implementation:**

```python
# ============================================================================
# CELL 16A - H100 Optimization
# ============================================================================

print("="*80)
print("🚀 H100 GPU OPTIMIZATION")
print("="*80)

import torch

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available! H100 not detected.")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {gpu_name}")
print(f"GPU Memory: {gpu_memory_gb:.1f} GB")

if "H100" not in gpu_name:
    print(f"⚠️ WARNING: Expected H100, found {gpu_name}")
else:
    print("✅ H100 detected!")

# Enable H100-specific optimizations
print("\n🔧 Enabling H100 optimizations:")

# 1. Enable TF32 for tensor cores (2-8× faster on H100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("  ✓ TF32 tensor cores enabled")

# 2. Set high precision for matrix multiplication
torch.set_float32_matmul_precision('high')
print("  ✓ High precision matmul enabled")

# 3. Enable CUDNN auto-tuner
torch.backends.cudnn.benchmark = True
print("  ✓ CUDNN auto-tuner enabled")

# 4. Enable asynchronous GPU operations
torch.cuda.set_device(0)
print("  ✓ GPU device set")

print("\n✅ H100 optimization complete")

# ============================================================================
# CELL 25B - Optimized Training for H100
# ============================================================================

print("="*80)
print("🏋️ TRAINING OPTIMIZED FOR H100")
print("="*80)

# H100-optimized hyperparameters
BATCH_SIZE_H100 = 500_000  # 500K samples (vs 65K default)
NUM_WORKERS = 8  # Parallel data loading
PIN_MEMORY = True  # Faster CPU→GPU transfer
EPOCHS = 50

print(f"Batch size: {BATCH_SIZE_H100:,} (H100 optimized)")
print(f"Epochs: {EPOCHS}")
print(f"Data workers: {NUM_WORKERS}")

# Prepare DataLoader
from torch.utils.data import TensorDataset, DataLoader

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_norm)
y_train_tensor = torch.FloatTensor(y_train_norm.reshape(-1, 1))

# Create dataset and loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_H100,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True  # Keep workers alive
)

print(f"✅ DataLoader created: {len(train_loader)} batches")

# Mixed precision training for H100
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

print("\n🏃 Starting training...")
training_start = time.time()

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss /= n_batches
    epoch_time = time.time() - epoch_start

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Loss: {epoch_loss:.6f} "
              f"Time: {epoch_time:.2f}s")

training_time = time.time() - training_start
print(f"\n✅ Training complete in {training_time:.1f}s ({training_time/60:.2f} min)")
print(f"⚡ H100 utilization: ~{BATCH_SIZE_H100 * 3 * 4 / 1e9:.2f}GB GPU memory used")

# ============================================================================
# CELL 27B - Optimized Prediction for H100
# ============================================================================

print("="*80)
print("🔮 PREDICTION OPTIMIZED FOR H100")
print("="*80)

BATCH_SIZE_PRED = 1_000_000  # 1M pixels per batch (H100 can handle it)

print(f"Prediction batch size: {BATCH_SIZE_PRED:,}")
print(f"Total pixels to predict: {n_pixels:,}")
print(f"Number of batches: {int(np.ceil(n_pixels / BATCH_SIZE_PRED))}")

model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, n_pixels, BATCH_SIZE_PRED):
        batch_end = min(i + BATCH_SIZE_PRED, n_pixels)
        batch_size_actual = batch_end - i

        # Prepare batch
        X_batch = np.stack([
            VV_all[i:batch_end],
            VH_all[i:batch_end],
            RVI_all[i:batch_end]
        ], axis=1)

        # Normalize
        X_batch_norm = (X_batch - train_mean) / train_std

        # Predict with mixed precision
        X_batch_tensor = torch.FloatTensor(X_batch_norm).to(device, non_blocking=True)

        with autocast():
            pred = model(X_batch_tensor)

        # Denormalize
        pred_denorm = pred.cpu().numpy() * train_std_ndvi + train_mean_ndvi
        predictions.append(pred_denorm.flatten())

        if (i // BATCH_SIZE_PRED) % 10 == 0:
            progress = (batch_end / n_pixels) * 100
            print(f"  Progress: {progress:.1f}%")

predictions = np.concatenate(predictions)
print(f"\n✅ Prediction complete: {len(predictions):,} pixels")
```

**Expected Performance on H100:**
- Training: 30-60 seconds (vs 2-5 minutes)
- Prediction: 5-10 seconds (vs 30-60 seconds)
- Total speedup: 3-5×

---

### **Solution 4: Use openEO for Full Production Workflow** (PAID ALTERNATIVE)

**Why:** Offload all processing to cloud, avoid local issues entirely.

**When to Use:**
- After free trial (1000 credits)
- For operational/production use
- Processing multiple regions regularly
- Want professional MOGPR implementation

**Cost Estimate for Demak:**
- Full area: ~200-500 credits per run
- Your 1000 free credits: 2-5 full Demak runs

**Advantages over local processing:**
- ✅ No GPU needed
- ✅ No data download (uses cloud collections)
- ✅ Professional MOGPR implementation
- ✅ Handles any area size
- ✅ Reproducible results
- ✅ Progress tracking

**Disadvantages:**
- ❌ Costs money after free trial
- ❌ Less control over fusion parameters
- ❌ Dependent on openEO service availability

**Implementation:** See "OpenEO vs Local Processing Comparison" section below.

---

### **Solution 5: Add Spatial Chunking** (FOR LARGER AREAS)

**Why:** Your Demak area (671×893) fits in memory with H100. But for larger regions (e.g., full Java Island), chunking is essential.

**When to Use:**
- Processing areas > 2000×2000 pixels
- System RAM < 32GB
- Multiple regions in parallel

**Current Status:** NOT needed for Demak with H100, but useful for scaling.

**Implementation:** Available if you want to process larger areas later.

---

## OpenEO vs Local Processing Comparison

### **Your Current Workflow (Local + GEE)**

```
Google Earth Engine
  ├─ Export 62 periods of S1+S2 data
  ├─ Download to GEE assets
  └─ Load via geemap.ee_to_numpy()
        ↓
Local Processing (H100)
  ├─ Apply paddy field mask
  ├─ Train S1→NDVI deep learning model
  ├─ Predict NDVI for cloudy pixels
  └─ Extract phenology metrics
        ↓
Outputs: NetCDF files, PNG visualizations
```

**Pros:**
- ✅ 100% FREE (GEE + local compute)
- ✅ Full control over all parameters
- ✅ Custom models (AttentionUNet, CropSAR, etc.)
- ✅ No dependency on external services
- ✅ H100 provides excellent performance

**Cons:**
- ❌ Network-dependent (GEE downloads)
- ❌ Requires local GPU + software setup
- ❌ Manual error handling
- ❌ No built-in MOGPR (uses custom DL model)

---

### **OpenEO Workflow (Cloud)**

```
openEO Platform
  ├─ Load RVI_ASC collection (pre-computed)
  ├─ Load NDVI collection (pre-computed)
  ├─ Apply MOGPR fusion (built-in service)
  ├─ Extract phenology (built-in service)
  └─ Download results as NetCDF
```

**Pros:**
- ✅ No local GPU needed
- ✅ No data download step
- ✅ Professional MOGPR implementation
- ✅ Handles any area size
- ✅ Progress tracking
- ✅ Reproducible results

**Cons:**
- ❌ COSTS MONEY (after 1000 free credits)
- ❌ Less control over fusion parameters
- ❌ Limited to openEO-supported algorithms
- ❌ Dependent on service availability
- ❌ May not support Indonesian-specific models

---

### **Hybrid Approach (RECOMMENDED)**

**For Development/Research:**
- Use local processing (your current workflow)
- Free, flexible, full control
- Leverage your H100

**For Production/Operations:**
- Use openEO for routine processing
- Reliable, scalable, professional
- Worth the cost for operational use

**For Large-Scale:**
- Use openEO for initial fusion
- Download fused results
- Apply custom analytics locally

---

## Implementation Guides

### **Guide 1: Adapting Your Notebook for Demak with openEO**

If you want to try openEO for Demak processing:

```python
# ============================================================================
# NEW NOTEBOOK: Demak_OpenEO_MOGPR_Phenology.ipynb
# ============================================================================

import openeo
import numpy as np
import matplotlib.pyplot as plt

# 1. Connect to openEO
print("Connecting to openEO...")
connection = openeo.connect("openeo.vito.be").authenticate_oidc()

# 2. Define Demak area
# Use your actual study area coordinates
demak_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [110.40, -6.95],  # SW corner
        [110.50, -6.95],  # SE corner
        [110.50, -6.85],  # NE corner
        [110.40, -6.85],  # NW corner
        [110.40, -6.95]   # Close polygon
    ]]
}

# Or load from your shapefile
import geopandas as gpd
demak_gdf = gpd.read_file('/path/to/your/demak_boundary.shp')
demak_polygon = demak_gdf.geometry.iloc[0].__geo_interface__

# 3. Define temporal extent (match your current workflow)
temporal_extent = ["2023-11-01", "2025-10-31"]  # 2 years

print(f"Study area: {demak_polygon}")
print(f"Temporal extent: {temporal_extent}")

# 4. Create MOGPR-fused datacube
print("\nCreating MOGPR fusion workflow...")
mogpr = connection.datacube_from_process(
    'mogpr_s1_s2',
    namespace='https://openeo.vito.be/openeo/1.1/processes/u:fusets/mogpr_s1_s2',
    polygon=demak_polygon,
    date=temporal_extent,
    s1_collection='RVI ASC',  # Pre-computed Radar Vegetation Index
    s2_collection='NDVI'      # Pre-computed NDVI from Sentinel-2
)

# Set metadata for band selection
from openeo.metadata import CollectionMetadata, BandDimension, Band
mogpr.metadata = CollectionMetadata(
    metadata={},
    dimensions=[
        BandDimension(
            name='bands',
            bands=[Band('RVI ASC'), Band('NDVI')]
        )
    ]
)

print("✅ MOGPR datacube created")

# 5. Extract NDVI band for phenology
mogpr_ndvi = mogpr.band('NDVI')

# 6. Apply phenology extraction
print("Adding phenology service...")
phenology = connection.datacube_from_process(
    'phenology',
    namespace='https://openeo.vito.be/openeo/1.1/processes/u:fusets/phenology',
    data=mogpr_ndvi
)

print("✅ Phenology workflow created")

# 7. Submit batch job
print("\nSubmitting batch job to openEO...")
output_file = './demak_openeo_phenology.nc'

job = phenology.execute_batch(
    output_file,
    out_format="netcdf",
    title='Demak - MOGPR + Phenology (OpenEO)',
    job_options={
        'executor-memory': '8g',
        'udf-dependency-archives': [
            'https://artifactory.vgt.vito.be:443/artifactory/auxdata-public/ai4food/fusets_venv.zip#tmp/venv',
            'https://artifactory.vgt.vito.be:443/artifactory/auxdata-public/ai4food/fusets.zip#tmp/venv_static'
        ]
    }
)

print(f"✅ Job submitted: {job.job_id}")
print(f"📊 Monitor at: https://openeo.vito.be/")

# 8. Results will be downloaded to './demak_openeo_phenology.nc'
print(f"\n✅ Results saved to: {output_file}")

# 9. Visualize results
print("\nVisualizing results...")
import netCDF4 as nc

nc_file = nc.Dataset(output_file)
keys = [x for x in nc_file.variables.keys()
        if x not in ['phenology', 'x', 'y', 'crs', 't', 'time']]

print(f"Available phenology metrics: {keys}")

# Plot SOS/EOS
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Start of Season
sos = nc_file.variables['sos_times'][0]
im1 = axes[0].imshow(sos, cmap='RdYlGn')
axes[0].set_title('Start of Season (Day of Year)')
fig.colorbar(im1, ax=axes[0])

# End of Season
eos = nc_file.variables['eos_times'][0]
im2 = axes[1].imshow(eos, cmap='RdYlGn')
axes[1].set_title('End of Season (Day of Year)')
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('demak_openeo_phenology_sos_eos.png', dpi=300)
print("✅ Visualization saved to: demak_openeo_phenology_sos_eos.png")
```

**Expected Runtime:**
- Job submission: < 1 minute
- Cloud processing: 10-30 minutes (depends on queue)
- Download: 1-5 minutes

**Expected Cost:**
- ~200-500 credits (depends on Demak area size)
- Your 1000 free credits: 2-5 runs

---

### **Guide 2: Quick Fixes for Your Current Notebook**

If you want to stick with local processing, apply these minimal fixes:

```python
# ============================================================================
# QUICK FIX 1: Stop on GEE download failures
# ============================================================================

# In Cell 9, after the download loop, add:

if len(dataset_list) < 50:  # Less than 50 out of 62 periods
    raise ValueError(f"❌ CRITICAL: Only {len(dataset_list)}/62 periods downloaded. "
                     f"Check network and retry.")

# ============================================================================
# QUICK FIX 2: Stop on insufficient training data
# ============================================================================

# In Cell 25, replace the validation check with:

if n_valid < 100_000:  # Need at least 100K samples for H100
    raise ValueError(f"❌ CRITICAL: Only {n_valid:,} valid training samples. "
                     f"Need at least 100,000. Check masking and data quality.")

# ============================================================================
# QUICK FIX 3: Validate predictions
# ============================================================================

# In Cell 27, after prediction, add:

ndvi_dl = fused_full['S2ndvi_DL'].values
valid_ratio = np.sum(~np.isnan(ndvi_dl)) / ndvi_dl.size

if valid_ratio < 0.5:
    raise ValueError(f"❌ CRITICAL: Prediction failed ({valid_ratio:.1%} valid). "
                     f"Model did not train properly.")

if np.nanmin(ndvi_dl) < -1.5 or np.nanmax(ndvi_dl) > 1.5:
    print("⚠️ WARNING: NDVI values out of range. Clipping...")
    fused_full['S2ndvi_DL'] = fused_full['S2ndvi_DL'].clip(-1, 1)

# ============================================================================
# QUICK FIX 4: Pre-validate phenology input
# ============================================================================

# In Cell 35, before phenology extraction, add:

mean_valid_obs = np.sum(~np.isnan(ndvi_for_phenology), axis=0).mean()

if mean_valid_obs < 20:
    raise ValueError(f"❌ CRITICAL: Insufficient temporal coverage "
                     f"({mean_valid_obs:.1f} valid obs/pixel). "
                     f"Need at least 20 observations for phenology.")

if np.nanmean(ndvi_for_phenology) < 0.15:
    print("⚠️ WARNING: Low NDVI values (non-vegetated area?)")
```

**These 4 quick fixes will:**
- Stop execution on critical errors
- Provide clear error messages
- Save time by failing fast
- Make debugging much easier

---

### **Guide 3: Debugging Your Current Failures**

**Step-by-step debugging process:**

#### **Step 1: Check GEE Download**

Run this diagnostic cell after Cell 9:

```python
print("\n" + "="*80)
print("🔍 GEE DOWNLOAD DIAGNOSTIC")
print("="*80)

print(f"Expected periods: 62")
print(f"Loaded periods: {len(combined_dataset.t)}")
print(f"Period range: {combined_dataset.t.min().values} - {combined_dataset.t.max().values}")

# Check for gaps
expected_periods = set(range(1, 63))
actual_periods = set(combined_dataset.t.values)
missing_periods = expected_periods - actual_periods

if missing_periods:
    print(f"❌ Missing periods: {sorted(missing_periods)}")
else:
    print("✅ All periods present")

# Check data quality per period
print("\nPer-period data quality:")
for t in combined_dataset.t.values:
    period_data = combined_dataset.sel(t=t)

    vv_valid = np.sum(~np.isnan(period_data['VV'].values))
    vh_valid = np.sum(~np.isnan(period_data['VH'].values))
    ndvi_valid = np.sum(~np.isnan(period_data['S2ndvi'].values))
    total_pixels = period_data['VV'].size

    print(f"  Period {t:2d}: VV={vv_valid/total_pixels:.1%}, "
          f"VH={vh_valid/total_pixels:.1%}, "
          f"NDVI={ndvi_valid/total_pixels:.1%}")
```

**What to look for:**
- Missing periods → Network failures during download
- Periods with 0% valid data → GEE asset corruption or wrong path
- NDVI < 10% valid → Heavy cloud cover

---

#### **Step 2: Check Masking**

Run this diagnostic cell after Cell 11:

```python
print("\n" + "="*80)
print("🔍 MASKING DIAGNOSTIC")
print("="*80)

# Visualize mask
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data (before mask)
axes[0].imshow(combined_dataset['VV'].isel(t=0).values, cmap='gray')
axes[0].set_title('VV Before Mask (Period 1)')

# After mask
axes[1].imshow(combined_dataset['VV'].isel(t=0).values, cmap='gray')
axes[1].set_title('VV After Mask (Period 1)')

# Mask coverage
mask_vis = ~np.isnan(combined_dataset['VV'].isel(t=0).values)
axes[2].imshow(mask_vis, cmap='RdYlGn')
axes[2].set_title('Valid Pixels (Green) vs Masked (Red)')

plt.tight_layout()
plt.savefig('masking_diagnostic.png', dpi=150)
print("✅ Saved: masking_diagnostic.png")

# Statistics
total_pixels = combined_dataset['VV'].size
valid_pixels = np.sum(~np.isnan(combined_dataset['VV'].values))
masked_pixels = total_pixels - valid_pixels

print(f"\nTotal pixels: {total_pixels:,}")
print(f"Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
print(f"Masked pixels: {masked_pixels:,} ({masked_pixels/total_pixels:.1%})")
```

**What to look for:**
- Valid pixels < 1% → Mask inverted or misaligned
- Valid pixels < 10% → Mask too aggressive
- Visual check: Does the mask match expected agricultural areas?

---

#### **Step 3: Check Training Data**

Run this diagnostic cell in Cell 25:

```python
print("\n" + "="*80)
print("🔍 TRAINING DATA DIAGNOSTIC")
print("="*80)

# Visualize data distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# VV distribution
axes[0, 0].hist(VV_full[mask_valid], bins=50, alpha=0.7)
axes[0, 0].set_title('VV Distribution (dB)')
axes[0, 0].set_xlabel('Backscatter (dB)')

# VH distribution
axes[0, 1].hist(VH_full[mask_valid], bins=50, alpha=0.7)
axes[0, 1].set_title('VH Distribution (dB)')
axes[0, 1].set_xlabel('Backscatter (dB)')

# NDVI distribution
axes[0, 2].hist(NDVI_full[mask_valid], bins=50, alpha=0.7)
axes[0, 2].set_title('NDVI Distribution')
axes[0, 2].set_xlabel('NDVI')

# RVI distribution
RVI_valid = RVI_full[mask_valid]
axes[1, 0].hist(RVI_valid, bins=50, alpha=0.7)
axes[1, 0].set_title('RVI Distribution')
axes[1, 0].set_xlabel('RVI')

# VH vs NDVI scatter
sample_idx = np.random.choice(n_valid, size=min(10000, n_valid), replace=False)
axes[1, 1].scatter(VH_train[sample_idx], y_train[sample_idx], alpha=0.1, s=1)
axes[1, 1].set_title('VH vs NDVI')
axes[1, 1].set_xlabel('VH (dB)')
axes[1, 1].set_ylabel('NDVI')

# Temporal coverage per pixel
valid_per_pixel = mask_valid.reshape(62, -1).sum(axis=0)
axes[1, 2].hist(valid_per_pixel, bins=62, alpha=0.7)
axes[1, 2].set_title('Valid Observations per Pixel')
axes[1, 2].set_xlabel('# Valid Timesteps')
axes[1, 2].set_ylabel('# Pixels')

plt.tight_layout()
plt.savefig('training_data_diagnostic.png', dpi=150)
print("✅ Saved: training_data_diagnostic.png")

# Statistics
print(f"\nData ranges:")
print(f"  VV: [{np.min(VV_train):.2f}, {np.max(VV_train):.2f}] dB")
print(f"  VH: [{np.min(VH_train):.2f}, {np.max(VH_train):.2f}] dB")
print(f"  RVI: [{np.min(RVI_train):.2f}, {np.max(RVI_train):.2f}]")
print(f"  NDVI: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")

print(f"\nMean valid timesteps per pixel: {valid_per_pixel.mean():.1f} / 62")
```

**What to look for:**
- VV/VH distributions centered around -15 to -10 dB → Good
- NDVI distribution: 0.2 - 0.8 for vegetation → Good
- VH vs NDVI scatter shows positive correlation → Good for model training
- Valid timesteps < 20 per pixel → Insufficient temporal coverage

---

#### **Step 4: Monitor Training**

Add this to Cell 25 during training:

```python
# Enhanced training monitoring
import time

print("\n" + "="*80)
print("🏋️ TRAINING PROGRESS")
print("="*80)

model.train()
training_losses = []
training_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0
    n_batches = 0

    for i in range(0, n_train, batch_size):
        batch = torch.FloatTensor(X_train_norm[i:i+batch_size]).to(device)
        target = torch.FloatTensor(y_train_norm[i:i+batch_size].reshape(-1, 1)).to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss /= n_batches
    training_losses.append(epoch_loss)
    epoch_time = time.time() - epoch_start

    # Progress report every 10 epochs
    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - training_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        print(f"Epoch [{epoch+1:3d}/{epochs}] "
              f"Loss: {epoch_loss:.6f} "
              f"Time: {epoch_time:.1f}s "
              f"ETA: {eta:.0f}s")

    # Early stopping check
    if epoch > 10 and epoch_loss > training_losses[epoch-10]:
        print(f"⚠️ WARNING: Loss not decreasing (may be diverging)")

total_time = time.time() - training_start
print(f"\n✅ Training complete in {total_time:.1f}s")

# Plot training curve
plt.figure(figsize=(10, 5))
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('training_curve.png', dpi=150)
print("✅ Saved: training_curve.png")

# Check final loss
final_loss = training_losses[-1]
if final_loss > 0.1:
    print(f"⚠️ WARNING: High final loss ({final_loss:.4f}). Model may not have converged.")
elif final_loss < 0.001:
    print(f"⚠️ WARNING: Very low loss ({final_loss:.4f}). Possible overfitting.")
else:
    print(f"✅ Good final loss: {final_loss:.4f}")
```

**What to look for:**
- Loss decreasing steadily → Good
- Loss stagnant or increasing → Model not learning (bad data or wrong architecture)
- Loss < 0.01 → Model converged well
- Loss > 0.1 → Model struggling (insufficient data or poor quality)

---

## Troubleshooting Checklist

### Pre-Flight Checklist (Before Running Notebook)

```
Hardware:
[ ] GPU detected: nvidia-smi shows H100
[ ] GPU memory free: > 2GB available
[ ] System RAM free: > 16GB available
[ ] Disk space free: > 20GB available

Authentication:
[ ] GEE authenticated: ee.Initialize() succeeds
[ ] GEE assets accessible: Can list projects/ee-geodeticengineeringundip/assets/FuseTS
[ ] Network stable: Ping google.com succeeds

Data:
[ ] Study area defined: demak_polygon or REGION
[ ] Temporal extent set: temp_ext = ["2023-11-01", "2025-10-31"]
[ ] Paddy mask file exists: java_island_paddy_mask.shp
[ ] Output directory exists: mkdir -p outputs/
```

---

### Runtime Monitoring Checklist

```
Cell 9 (GEE Download):
[ ] All 62 periods downloaded
[ ] No "Could not load period X" warnings
[ ] combined_dataset.dims shows (t: 62, y: 671, x: 893)

Cell 11 (Masking):
[ ] Valid pixels after mask > 5%
[ ] Mask visualization shows agricultural areas
[ ] No "Dataset is EMPTY" error

Cell 25 (Training):
[ ] Valid training samples > 100,000
[ ] Training loss decreasing
[ ] Final loss < 0.1
[ ] No NaN in training data

Cell 27 (Prediction):
[ ] Predictions within [-1, 1] range
[ ] Valid predictions > 50%
[ ] NDVI_DL mean ~ 0.3-0.7

Cell 35 (Phenology):
[ ] Input NDVI has > 30 valid obs/pixel
[ ] SOS/EOS not all NaN
[ ] Phenology visualization shows patterns
```

---

### Common Error Messages & Solutions

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `"Could not load period X"` | GEE network timeout | Retry download, check auth |
| `"Dataset is EMPTY (all NaN)"` | Mask inverted or misaligned | Check mask CRS and geometry |
| `"Not enough valid training data"` | Over-aggressive masking | Relax cloud/paddy mask |
| `"CUDA out of memory"` | Batch size too large | Reduce batch_size (shouldn't happen on H100) |
| `"Loss = NaN"` | Bad data or wrong normalization | Check for inf/NaN in training data |
| `"All predictions are NaN"` | Model didn't train | Check training loss, increase epochs |
| `"No SOS/EOS detected"` | Insufficient temporal coverage | Check NDVI time series quality |
| `RuntimeError: CUDA error` | GPU driver issue | Restart kernel, check nvidia-smi |
| `ConnectionError` | Network issue | Check internet, retry |
| `PermissionError` | GEE auth expired | Re-authenticate: ee.Authenticate() |

---

### Performance Benchmarks (H100)

**Expected timings for full Demak area (671×893):**

| Stage | Expected Time | Warning If > |
|-------|---------------|--------------|
| GEE Download (62 periods) | 1-3 minutes | 10 minutes |
| Data Loading & Masking | < 30 seconds | 2 minutes |
| Training Prep | < 1 minute | 5 minutes |
| Model Training (50 epochs) | 2-5 minutes | 15 minutes |
| Prediction | 30-60 seconds | 5 minutes |
| Phenology Extraction | 30-60 seconds | 5 minutes |
| **TOTAL** | **5-15 minutes** | **45 minutes** |

**If your processing takes > 45 minutes total, something is wrong.**

---

### H100 GPU Health Check

```bash
# Check GPU status
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.XX       Driver Version: 535.XX       CUDA Version: 12.2   |
# |-------------------------------+----------------------+----------------------+
# |   0  NVIDIA H100        On   | 00000000:00:05.0 Off |                    0 |
# | N/A   40C    P0    150W / 350W |      0MiB / 81920MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Check CUDA available in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA H100

# Monitor GPU during training (run in separate terminal)
watch -n 1 nvidia-smi
```

---

## Summary & Recommendations

### Key Findings

1. **Hardware is NOT the problem**
   - Your H100 has 40-80× more capacity than needed
   - GPU memory: 80GB vs 2GB required
   - Training should take 2-5 minutes (vs hours on consumer GPUs)

2. **Root causes of failure (in order of likelihood)**
   - GEE download failures (network timeouts, API limits)
   - Over-aggressive masking (removes 90%+ of pixels)
   - Silent error handling (notebook continues despite failures)
   - Missing data validation (no checks between stages)

3. **Solutions prioritized by impact**
   - **HIGH PRIORITY:** Add robust error handling & validation
   - **HIGH PRIORITY:** Implement retry logic for GEE downloads
   - **MEDIUM PRIORITY:** Optimize for H100 (3-5× speedup)
   - **LOW PRIORITY:** Add spatial chunking (not needed for Demak)
   - **ALTERNATIVE:** Use openEO cloud processing (costs money)

### Recommended Action Plan

#### **Phase 1: Quick Fixes (30 minutes)**
1. Add 4 validation checks (see Guide 2)
2. Test on small region first (USE_SMALL_REGION=True)
3. Monitor GPU usage with nvidia-smi

#### **Phase 2: Robust Pipeline (2-3 hours)**
1. Implement retry logic for GEE downloads
2. Add comprehensive validation at each stage
3. Create diagnostic visualizations
4. Test on full Demak area

#### **Phase 3: Optimization (Optional, 1-2 hours)**
1. Optimize for H100 (larger batches, TF32, mixed precision)
2. Add progress tracking and ETA
3. Parallelize preprocessing

#### **Phase 4: Production (Optional)**
1. Consider openEO for operational use
2. Add spatial chunking for larger areas
3. Implement automated quality reports

### Next Steps

**I recommend:**

1. **Start with Quick Fixes** (Guide 2)
   - Takes 30 minutes
   - Will immediately reveal failure point
   - Provides clear error messages

2. **Run diagnostic cells** (Guide 3)
   - Step through notebook cell by cell
   - Check output after each stage
   - Identify exact failure point

3. **Share specific error message**
   - Run notebook with quick fixes
   - Copy the error message you get
   - I'll provide targeted solution

4. **Consider openEO for comparison**
   - Use 100-200 of your 1000 free credits
   - See if cloud processing works better
   - Compare results quality

### Final Thoughts

**With your H100 GPU, you have one of the most powerful systems available for this work.** The processing should be fast and reliable. If it's not, the problem is in the data pipeline (GEE downloads, masking, data quality), not computation.

The solutions provided will:
- ✅ Catch failures immediately (fail fast)
- ✅ Provide clear diagnostics
- ✅ Prevent garbage outputs
- ✅ Make debugging easy
- ✅ Leverage your H100's power

**Let me know:**
1. What specific error you're getting
2. How far the notebook gets before failing
3. Whether you want to try quick fixes or full robust pipeline

I'm here to help debug and get your workflow running smoothly!