# ============================================================================
# LOAD DATA FROM LOCAL GEOTIFF FILES (Replace Cell 8 in improved notebook)
# ============================================================================
#
# USE THIS CELL INSTEAD OF CELL 8 if you already have GeoTIFF files downloaded
# in gee_assets_download/period_01.tif, period_02.tif, etc.
#
# ============================================================================

import rioxarray
import xarray as xr
import numpy as np
from pathlib import Path
import time

print("="*80)
print("📂 LOADING DATA FROM LOCAL GEOTIFF FILES")
print("="*80)

# Initialize download_times list for compatibility with summary cell
download_times = []

# Configuration
GEOTIFF_DIR = Path('gee_assets_download')
NUM_PERIODS = 62  # Number of periods you have

print(f"\nLooking for TIF files in: {GEOTIFF_DIR.absolute()}")

# Load all periods
dataset_list = []
successful_periods = []
overall_start = time.time()

for period in range(1, NUM_PERIODS + 1):
    tif_path = GEOTIFF_DIR / f"period_{period:02d}.tif"

    if not tif_path.exists():
        print(f"⚠️  Period {period:02d}: File not found - {tif_path}")
        continue

    start_time = time.time()
    try:
        # Open GeoTIFF with rioxarray
        # This creates an xarray DataArray with dimension 'band'
        data = rioxarray.open_rasterio(tif_path, masked=True)

        # ✅ CRITICAL: Verify band order and assign correct variable names
        # GEE exports as: Band 1 = VV, Band 2 = VH, Band 3 = S2ndvi

        # Extract each band with correct name
        VV = data.sel(band=1).drop('band')
        VH = data.sel(band=2).drop('band')
        S2ndvi = data.sel(band=3).drop('band')

        # ✅ VALIDATION: Check S2ndvi range (should be -1 to 1, not backscatter)
        ndvi_min = float(S2ndvi.min())
        ndvi_max = float(S2ndvi.max())

        # Create xarray Dataset with proper variable names
        period_ds = xr.Dataset({
            'VV': VV,
            'VH': VH,
            'S2ndvi': S2ndvi
        })

        # Add time coordinate (period number)
        period_ds = period_ds.expand_dims(t=[period])

        dataset_list.append(period_ds)
        successful_periods.append(period)

        elapsed = time.time() - start_time
        download_times.append(elapsed)

        # Validation output
        if -1 <= ndvi_min <= 1 and -1 <= ndvi_max <= 1:
            status = "✅"
        else:
            status = f"❌ WARNING: S2ndvi range [{ndvi_min:.2f}, {ndvi_max:.2f}] outside [-1,1]!"

        print(f"Period {period:02d}: VV=[{float(VV.min()):.1f}, {float(VV.max()):.1f}] dB, "
              f"VH=[{float(VH.min()):.1f}, {float(VH.max()):.1f}] dB, "
              f"S2ndvi=[{ndvi_min:.3f}, {ndvi_max:.3f}] {status}")

    except Exception as e:
        print(f"❌ Period {period:02d}: Error loading - {e}")
        elapsed = time.time() - start_time
        download_times.append(elapsed)
        continue

total_load_time = time.time() - overall_start
print(f"\n✅ Successfully loaded {len(dataset_list)} out of {NUM_PERIODS} periods")
print(f"   Total loading time: {total_load_time:.1f}s ({total_load_time/60:.1f} min)")

# Concatenate all periods along time dimension
if dataset_list:
    combined_dataset = xr.concat(dataset_list, dim='t')

    print(f"\n📊 Combined dataset shape: {dict(combined_dataset.dims)}")
    print(f"   Variables: {list(combined_dataset.data_vars)}")
    print(f"   Time periods: {len(combined_dataset.t)}")
    print(f"   Spatial dimensions: {combined_dataset.dims['y']} × {combined_dataset.dims['x']}")

    # Final validation
    print(f"\n🔍 FINAL VALIDATION:")
    print(f"   VV range:     [{float(combined_dataset['VV'].min()):.2f}, {float(combined_dataset['VV'].max()):.2f}] dB")
    print(f"   VH range:     [{float(combined_dataset['VH'].min()):.2f}, {float(combined_dataset['VH'].max()):.2f}] dB")
    print(f"   S2ndvi range: [{float(combined_dataset['S2ndvi'].min()):.4f}, {float(combined_dataset['S2ndvi'].max()):.4f}]")

    # Check if S2ndvi is in correct range
    ndvi_min = float(combined_dataset['S2ndvi'].min())
    ndvi_max = float(combined_dataset['S2ndvi'].max())

    if -1 <= ndvi_min <= 1 and -1 <= ndvi_max <= 1:
        print(f"\n   ✅ S2ndvi values are CORRECT (in [-1, 1] range)")
        print(f"   ✅ Ready for training!")
    else:
        print(f"\n   ❌ ERROR: S2ndvi values are WRONG!")
        print(f"   ❌ S2ndvi range [{ndvi_min:.2f}, {ndvi_max:.2f}] suggests backscatter, not NDVI")
        print(f"   ❌ This will cause R² = -0.8 failure!")
        print(f"\n   Possible causes:")
        print(f"   1. GeoTIFF bands in wrong order (check with gdalinfo)")
        print(f"   2. S2ndvi band contains VV or VH data (GEE export bug)")
        print(f"   3. Band indices off by one (band 3 vs band 2)")
else:
    print("\n❌ No datasets loaded! Check file paths and permissions.")
    raise FileNotFoundError(f"No GeoTIFF files found in {GEOTIFF_DIR}")

print("="*80)
