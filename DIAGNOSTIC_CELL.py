# ============================================================================
# DIAGNOSTIC CELL - Run this to check your combined_dataset
# ============================================================================
#
# Copy-paste this into a notebook cell BEFORE running the fusion model
# This will identify why you're getting IndexError
#
# ============================================================================

import numpy as np

print("="*80)
print("🔍 COMBINED_DATASET DIAGNOSTIC")
print("="*80)

# Check if combined_dataset exists
try:
    print(f"\n✅ combined_dataset exists")
    print(f"   Type: {type(combined_dataset)}")
except NameError:
    print(f"\n❌ ERROR: combined_dataset is not defined!")
    print(f"   You need to run the data loading cell first")
    raise

# Check structure
print(f"\n📊 Structure:")
print(f"   Dimensions: {dict(combined_dataset.dims)}")
print(f"   Variables: {list(combined_dataset.data_vars)}")

# Required variables
required_vars = ['VV', 'VH', 'S2ndvi']
missing_vars = [v for v in required_vars if v not in combined_dataset.data_vars]

if missing_vars:
    print(f"\n❌ ERROR: Missing required variables: {missing_vars}")
    raise ValueError(f"combined_dataset must have {required_vars}")

# Check each variable
print(f"\n📈 Data Quality Check:")

for var in required_vars:
    data = combined_dataset[var].values

    total = data.size
    finite = np.sum(np.isfinite(data))
    nan = np.sum(np.isnan(data))

    print(f"\n   {var}:")
    print(f"      Shape: {data.shape}")
    print(f"      Total elements: {total:,}")
    print(f"      Finite: {finite:,} ({finite/total:.1%})")
    print(f"      NaN: {nan:,} ({nan/total:.1%})")

    if finite > 0:
        finite_data = data[np.isfinite(data)]
        print(f"      Range: [{np.min(finite_data):.4f}, {np.max(finite_data):.4f}]")
        print(f"      Mean: {np.mean(finite_data):.4f}")
    else:
        print(f"      ❌ NO FINITE VALUES!")

# Check overlap (critical for training)
print(f"\n🔗 Overlap Analysis:")

VV_flat = combined_dataset['VV'].values.flatten()
VH_flat = combined_dataset['VH'].values.flatten()
NDVI_flat = combined_dataset['S2ndvi'].values.flatten()

mask_VV = np.isfinite(VV_flat)
mask_VH = np.isfinite(VH_flat)
mask_NDVI = np.isfinite(NDVI_flat)
mask_all = mask_VV & mask_VH & mask_NDVI

n_total = len(VV_flat)
n_VV = np.sum(mask_VV)
n_VH = np.sum(mask_VH)
n_NDVI = np.sum(mask_NDVI)
n_overlap = np.sum(mask_all)

print(f"   Total pixels: {n_total:,}")
print(f"   VV finite: {n_VV:,} ({n_VV/n_total:.1%})")
print(f"   VH finite: {n_VH:,} ({n_VH/n_total:.1%})")
print(f"   S2ndvi finite: {n_NDVI:,} ({n_NDVI/n_total:.1%})")
print(f"   ALL THREE finite: {n_overlap:,} ({n_overlap/n_total:.1%})")

# Final verdict
print(f"\n📋 Diagnosis:")

if n_overlap == 0:
    print(f"   ❌ CRITICAL ERROR: No pixels have all three variables!")
    print(f"\n   Possible causes:")
    print(f"   1. Data loading failed - check LOAD_LOCAL_TIFS_CELL.py output")
    print(f"   2. Paddy mask removed all data - check masking cell")
    print(f"   3. Files are corrupted or in wrong format")
    print(f"\n   Action needed:")
    print(f"   - Re-run LOAD_LOCAL_TIFS_CELL.py and check for errors")
    print(f"   - Comment out masking cell temporarily")
    print(f"   - Check GeoTIFF files with gdalinfo")

elif n_overlap < 10000:
    print(f"   ⚠️  WARNING: Very few valid samples ({n_overlap:,})")
    print(f"   Training may fail or perform poorly")
    print(f"   Need at least 10,000 samples, preferably 100,000+")
    print(f"\n   Consider:")
    print(f"   - Using a larger study area")
    print(f"   - Relaxing the paddy mask")
    print(f"   - Checking for excessive cloud/gap filtering")

elif n_overlap < 100000:
    print(f"   ⚠️  Marginal: {n_overlap:,} samples available")
    print(f"   Training should work but performance may be limited")
    print(f"   Recommended: 100,000+ samples for best results")

else:
    print(f"   ✅ GOOD: {n_overlap:,} samples available")
    print(f"   This should be sufficient for training")

# Check NDVI range (common issue)
if n_NDVI > 0:
    ndvi_values = NDVI_flat[mask_NDVI]
    ndvi_min = np.min(ndvi_values)
    ndvi_max = np.max(ndvi_values)

    print(f"\n🌿 NDVI Range Check:")
    print(f"   Min: {ndvi_min:.4f}")
    print(f"   Max: {ndvi_max:.4f}")

    if ndvi_min < -1 or ndvi_max > 1:
        print(f"   ❌ ERROR: NDVI outside valid range [-1, 1]!")
        print(f"   This suggests wrong band order in GeoTIFF files")
        print(f"   Check LOAD_LOCAL_TIFS_CELL.py band assignments")
    elif ndvi_min < -0.5:
        print(f"   ⚠️  Warning: NDVI min is very low ({ndvi_min:.4f})")
        print(f"   Check for water pixels or wrong band")
    else:
        print(f"   ✅ NDVI range looks valid")

print(f"\n" + "="*80)

if n_overlap > 0:
    print(f"✅ Dataset looks usable - you can proceed with training")
else:
    print(f"❌ Dataset has critical issues - fix data loading first!")

print("="*80)
