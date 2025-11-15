# MOGPR Quick Reference Card

## Essential Data Format Checklist

```python
# ✅ CORRECT MOGPR Input Format
import xarray as xr
import pandas as pd

ds = xr.Dataset({
    'VV': (['t', 'y', 'x'], vv_data),      # ← Dimension: 't' (not 'time'!)
    'VH': (['t', 'y', 'x'], vh_data),      # ← Variable: 'VH' (exact name)
    'S2ndvi': (['t', 'y', 'x'], ndvi_data) # ← Variable: 'S2ndvi' (not 'NDVI')
})

# Add datetime coordinates
ds = ds.assign_coords({'t': pd.date_range('2024-01-01', periods=30, freq='12D')})

# Verify
assert 't' in ds.dims                    # ✓ Time dimension named 't'
assert 'S2ndvi' in ds.data_vars          # ✓ NDVI variable named 'S2ndvi'
assert pd.api.types.is_datetime64_any_dtype(ds['t'])  # ✓ Datetime coords
```

## Point-Based Workflow (One-Pager)

### R: Extract and Export
```r
library(sits)
source("scripts/sits_to_fusets.R")

# 1. Extract time series
s1_samples <- sits_get_data(s1_cube, samples = points)
s2_samples <- sits_get_data(s2_cube, samples = points)
s2_samples <- sits_apply(s2_samples, NDVI = (B08 - B04) / (B08 + B04))

# 2. Export to CSV
sits_to_fusets_csv(s1_samples, "s1.csv", bands = c("VV", "VH"))
sits_to_fusets_csv(s2_samples, "s2.csv", bands = c("NDVI"))
```

### Python: Load, Fuse, Analyze
```python
from fusets.io.sits_bridge import load_sits_csv, prepare_mogpr_format
from fusets.mogpr import mogpr
from fusets.analytics import phenology

# 1. Load
s1 = load_sits_csv("s1.csv", band_cols=['VV', 'VH'])
s2 = load_sits_csv("s2.csv", band_cols=['NDVI'])

# 2. Prepare
data = prepare_mogpr_format(s1['VV'], s1['VH'], s2['NDVI'], s2['t'])

# 3. Fuse
fused = mogpr(data, include_uncertainties=True, include_raw_inputs=True)

# 4. Extract
ndvi_gapfilled = fused['S2ndvi_FUSED']  # Gap-filled
ndvi_uncertainty = fused['S2ndvi_STD']   # Uncertainty
ndvi_original = fused['S2ndvi_RAW']      # Original (with gaps)

# 5. Phenology
pheno = phenology(ndvi_gapfilled, detection_method='seasonal_amplitude')
planting_dates = pheno.da_sos_times  # Day of year
harvest_dates = pheno.da_eos_times
```

## Raster-Based Workflow (One-Pager)

### R: Export Spatial Cube
```r
library(sits)
source("scripts/sits_to_fusets.R")

# Export as GeoTIFF stacks
sits_cube_to_fusets_geotiff(s1_cube, "output/s1", bands = c("VV", "VH"))
sits_cube_to_fusets_geotiff(s2_cube, "output/s2", bands = c("NDVI"))
```

### Python: Load, Fuse, Export Maps
```python
from fusets.io.sits_bridge import load_sits_geotiff
from fusets.mogpr import mogpr
from fusets.analytics import phenology
import xarray as xr

# 1. Load with chunking (for large files)
s1_vv = load_sits_geotiff("output/s1/VV_stack.tif")
s1_vh = load_sits_geotiff("output/s1/VH_stack.tif")
s2_ndvi = load_sits_geotiff("output/s2/NDVI_stack.tif")

# 2. Create Dataset (rename dimensions!)
data = xr.Dataset({
    'VV': s1_vv.rename({'band': 't'}),
    'VH': s1_vh.rename({'band': 't'}),
    'S2ndvi': s2_ndvi.rename({'band': 't'})
})

# 3. Apply MOGPR
fused = mogpr(data, prediction_period='P5D')  # 5-day composites

# 4. Extract and export
fused['S2ndvi_FUSED'].rio.to_raster('NDVI_gapfilled_stack.tif', compress='lzw')

# 5. Phenology mapping
pheno = phenology(fused['S2ndvi_FUSED'])
pheno.da_sos_times.rio.to_raster('SOS_map.tif')  # Planting date map
pheno.da_eos_times.rio.to_raster('EOS_map.tif')  # Harvest date map
```

## Common Transformations

### Fix Dimension Names
```python
# Before: {'time': 30, 'x': 100, 'y': 100}
# After:  {'t': 30, 'x': 100, 'y': 100}
ds = ds.rename({'time': 't'})
ds = ds.rename({'band': 't'})
ds = ds.rename({'Index': 't'})
```

### Fix Variable Names
```python
# Before: {'NDVI', 'sentinel1_vv', 'sentinel1_vh'}
# After:  {'S2ndvi', 'VV', 'VH'}
ds = ds.rename({
    'NDVI': 'S2ndvi',
    'sentinel1_vv': 'VV',
    'sentinel1_vh': 'VH'
})
```

### Align Time Coordinates
```python
import pandas as pd

# Option A: Interpolate to common grid
common_times = pd.date_range('2024-01-01', '2024-12-31', freq='12D')
s1_aligned = s1.interp(t=common_times)
s2_aligned = s2.interp(t=common_times)

# Option B: Use one as reference
data = prepare_mogpr_format(s1['VV'], s1['VH'], s2['NDVI'],
                            time_coords=s2['t'])  # Use S2 times
```

## MOGPR Parameters

```python
fused = mogpr(
    array,                          # xr.Dataset with VV, VH, S2ndvi
    variables=['VV', 'VH', 'S2ndvi'],  # Variables to fuse (default: all)
    time_dimension='t',             # Time dim name (default: 't')
    prediction_period='P5D',        # Output temporal resolution ('P5D'=5-day)
    include_uncertainties=True,     # Add _STD variables
    include_raw_inputs=True         # Add _RAW variables
)
```

### Output Variables
| Input Variable | Fused Output | Uncertainty | Raw Input |
|----------------|--------------|-------------|-----------|
| `VV` | `VV_FUSED` | `VV_STD` | `VV_RAW` |
| `VH` | `VH_FUSED` | `VH_STD` | `VH_RAW` |
| `S2ndvi` | `S2ndvi_FUSED` | `S2ndvi_STD` | `S2ndvi_RAW` |

## Phenology Parameters

```python
pheno = phenology(
    ndvi_timeseries,                    # Gap-filled NDVI
    detection_method='seasonal_amplitude',  # or 'first_of_slope', 'median_of_slope'
    amplitude_threshold=0.2,            # Min amplitude to detect season
    absolute_threshold=0.3,             # For 'absolute_value' method
    slope_threshold=0.01                # For slope-based methods
)
```

### Available Metrics
```python
pheno.da_sos_times          # Start of Season (day of year)
pheno.da_eos_times          # End of Season (day of year)
pheno.da_los                # Length of Season (days)
pheno.da_sos_values         # NDVI value at SOS
pheno.da_eos_values         # NDVI value at EOS
pheno.da_peak_time          # Day of year of peak
pheno.da_peak_value         # Peak NDVI value
pheno.da_season_amplitude   # Peak - Base
pheno.da_seasonal_integral  # Time-integrated NDVI
```

## Troubleshooting Commands

```python
# Check data structure
print(ds.dims)              # Should show 't' dimension
print(list(ds.data_vars))   # Should show 'VV', 'VH', 'S2ndvi'
print(ds['t'].values)       # Should be datetime objects

# Check for NaN values
print(ds['S2ndvi'].isnull().sum())  # Count missing values

# Visualize single location
import matplotlib.pyplot as plt
loc_id = 0
plt.plot(ds['t'], ds['S2ndvi'].isel(location=loc_id), 'o-')
plt.show()

# Check memory usage
print(f"Dataset size: {ds.nbytes / 1e9:.2f} GB")

# Test on subset
ds_test = ds.isel(x=slice(0, 10), y=slice(0, 10))  # Small spatial subset
fused_test = mogpr(ds_test)  # Should be fast
```

## Memory Management for Large Rasters

### Option 1: Dask Chunking
```python
import rioxarray as rxr

# Load with chunks
data = rxr.open_rasterio('large_file.tif',
                         chunks={'band': 10, 'y': 512, 'x': 512})
```

### Option 2: Spatial Tiling
```python
# Process in 1000x1000 pixel tiles
for i in range(0, height, 1000):
    for j in range(0, width, 1000):
        tile = ds.isel(y=slice(i, i+1000), x=slice(j, j+1000))
        tile_fused = mogpr(tile)
        # Save tile
```

### Option 3: Temporal Chunking
```python
# Process 10 time steps at a time
for t_start in range(0, len(ds['t']), 10):
    chunk = ds.isel(t=slice(t_start, t_start+10))
    chunk_fused = mogpr(chunk)
    # Save chunk
```

## Validation Workflow

```python
# 1. Extract gap-filled and original
ndvi_fused = fused['S2ndvi_FUSED']
ndvi_raw = fused['S2ndvi_RAW']
ndvi_std = fused['S2ndvi_STD']

# 2. Compare at cloud-free locations
import numpy as np

clear_pixels = ~np.isnan(ndvi_raw)
rmse = np.sqrt(np.mean((ndvi_fused[clear_pixels] - ndvi_raw[clear_pixels])**2))
bias = np.mean(ndvi_fused[clear_pixels] - ndvi_raw[clear_pixels])

print(f"RMSE (clear pixels): {rmse:.4f}")
print(f"Bias: {bias:.4f}")

# 3. Uncertainty analysis
avg_uncertainty = ndvi_std.mean().values
print(f"Average uncertainty: {avg_uncertainty:.4f}")

# 4. Visual comparison
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(ndvi_raw.isel(t=15), vmin=0, vmax=1, cmap='RdYlGn')
axes[0].set_title('Original (with gaps)')
axes[1].imshow(ndvi_fused.isel(t=15), vmin=0, vmax=1, cmap='RdYlGn')
axes[1].set_title('Gap-filled')
axes[2].imshow(ndvi_std.isel(t=15), vmin=0, vmax=0.2, cmap='Reds')
axes[2].set_title('Uncertainty')
plt.tight_layout()
plt.show()
```

## Common Error Messages

| Error | Fix |
|-------|-----|
| `KeyError: 't'` | Rename dimension: `ds.rename({'time': 't'})` |
| `KeyError: 'S2ndvi'` | Rename variable: `ds.rename({'NDVI': 'S2ndvi'})` |
| `ValueError: conflicting sizes` | Align time coordinates between S1 and S2 |
| `MemoryError` | Use chunking or process in tiles |
| `ImportError: GPy` | Install: `pip install GPy` |
| `All NaN slice encountered` | Check for completely missing time series |

## Performance Benchmarks (Approximate)

| Dataset Size | Point-Based | Raster-Based | Notes |
|--------------|-------------|--------------|-------|
| 100 points, 30 dates | ~10 seconds | N/A | Fast, ideal for validation |
| 1000 points, 30 dates | ~1 minute | N/A | Still manageable |
| 100x100 pixels, 30 dates | N/A | ~5 minutes | Small test area |
| 1000x1000 pixels, 30 dates | N/A | ~3 hours | Use tiling |
| 10000x10000 pixels, 30 dates | N/A | Days | Requires HPC or cloud |

*Note: Benchmarks on typical laptop (8GB RAM, i5 CPU). GPU acceleration not currently supported.*

---

**Full Documentation**: [SITS_TO_MOGPR_GUIDE.md](SITS_TO_MOGPR_GUIDE.md)
**Quick Start**: [docs/SITS_MOGPR_README.md](docs/SITS_MOGPR_README.md)
