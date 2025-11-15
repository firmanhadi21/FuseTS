# SITS Datacube to MOGPR: Complete Guide

This guide provides comprehensive instructions for saving datacubes collected using the R `sits` package and processing them with FuseTS MOGPR (Multi-Output Gaussian Process Regression) for sensor fusion and gap-filling.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Workflow 1: Point-Based MOGPR](#workflow-1-point-based-mogpr-recommended-for-validation)
4. [Workflow 2: Raster-Based MOGPR](#workflow-2-raster-based-mogpr-for-spatial-mapping)
5. [Understanding MOGPR Data Requirements](#understanding-mogpr-data-requirements)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Overview

### What is MOGPR?

MOGPR (Multi-Output Gaussian Process Regression) leverages correlations between multiple time series (e.g., Sentinel-1 VV, VH, and Sentinel-2 NDVI) to:
- **Fill gaps** in cloudy/missing optical data using radar observations
- **Improve time series quality** by fusing complementary sensors
- **Provide uncertainty estimates** for gap-filled values

### sits + MOGPR Workflow

```
┌─────────────────┐
│   R sits        │  1. Extract multi-sensor data from MPC/BDC
│   Data Cube     │  2. Calculate indices (NDVI, RVI, etc.)
└────────┬────────┘  3. Export to CSV or GeoTIFF
         │
         ▼
┌─────────────────┐
│  FuseTS Bridge  │  4. Load with sits_bridge module
│  Module         │  5. Format for MOGPR compatibility
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MOGPR Fusion   │  6. Apply multi-sensor fusion
│  & Gap-filling  │  7. Extract fused time series
└────────┬────────┘  8. Optional: phenology extraction
         │
         ▼
┌─────────────────┐
│  Analysis &     │  9. Export results (CSV/GeoTIFF)
│  Visualization  │ 10. Statistical analysis
└─────────────────┘
```

---

## Prerequisites

### R Environment
```r
install.packages(c("sits", "sf", "terra", "dplyr", "tidyr"))
```

### Python Environment
```bash
cd /home/unika_sianturi/work/FuseTS
pip install -e .
pip install rioxarray GPy  # For GeoTIFF support and MOGPR
```

### Helper Scripts
```bash
# Load sits-to-FuseTS helper functions in R
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")
```

---

## Workflow 1: Point-Based MOGPR (Recommended for Validation)

### Overview
Point-based MOGPR processes time series at specific locations (field samples, validation points, etc.). This is **faster** and ideal for:
- Field validation studies
- Training data generation
- Quick exploratory analysis
- Agricultural monitoring at specific sites

### Step 1.1: Extract Multi-Sensor Time Series with sits (R)

```r
library(sits)
library(sf)
library(dplyr)

# Source FuseTS helper functions
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")

# Define study area or load sample points
study_area <- st_read("study_area.shp")
sample_points <- st_sample(study_area, size = 100)

# Or load existing points
# sample_points <- st_read("field_validation_points.shp")

# ============================================
# Get Sentinel-1 data (VV and VH polarizations)
# ============================================
s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("VV", "VH")
)

# Extract S1 time series at sample points
s1_samples <- sits_get_data(s1_cube, samples = sample_points)

# ============================================
# Get Sentinel-2 data
# ============================================
s2_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("B04", "B08")  # Red, NIR for NDVI
)

# Extract S2 time series
s2_samples <- sits_get_data(s2_cube, samples = sample_points)

# Calculate NDVI
s2_samples <- sits_apply(s2_samples,
  NDVI = (B08 - B04) / (B08 + B04)
)

# ============================================
# Export to FuseTS-compatible CSV files
# ============================================
sits_to_fusets_csv(s1_samples,
                   "data/s1_timeseries.csv",
                   bands = c("VV", "VH"))

sits_to_fusets_csv(s2_samples,
                   "data/s2_timeseries.csv",
                   bands = c("NDVI"))

print("✓ Time series exported to CSV files")
```

### Step 1.2: Load and Prepare Data for MOGPR (Python)

```python
import sys
sys.path.insert(0, '/home/unika_sianturi/work/FuseTS/src')

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from fusets.io.sits_bridge import load_sits_csv, prepare_mogpr_format
from fusets.mogpr import mogpr
from fusets.analytics import phenology

# ============================================
# Load S1 and S2 time series
# ============================================
s1_data = load_sits_csv(
    "data/s1_timeseries.csv",
    time_col='Index',
    band_cols=['VV', 'VH'],
    location_cols=['longitude', 'latitude', 'label']  # Optional location identifiers
)

s2_data = load_sits_csv(
    "data/s2_timeseries.csv",
    time_col='Index',
    band_cols=['NDVI'],
    location_cols=['longitude', 'latitude', 'label']
)

print(f"S1 data dimensions: {s1_data.dims}")
print(f"S2 data dimensions: {s2_data.dims}")
print(f"S1 variables: {list(s1_data.data_vars)}")
print(f"S2 variables: {list(s2_data.data_vars)}")

# ============================================
# Prepare combined dataset for MOGPR
# ============================================
# IMPORTANT: MOGPR requires consistent time coordinates across all bands
# and specific naming conventions: 'VV', 'VH', 'S2ndvi'

# Option A: Using helper function (recommended for S1+S2 fusion)
mogpr_data = prepare_mogpr_format(
    s1_vv=s1_data['VV'],
    s1_vh=s1_data['VH'],
    s2_ndvi=s2_data['NDVI'],
    time_coords=s2_data['t']
)

print("\n✓ Data prepared for MOGPR")
print(f"MOGPR input shape: {mogpr_data.dims}")
print(f"MOGPR variables: {list(mogpr_data.data_vars)}")
```

### Step 1.3: Apply MOGPR Fusion (Python)

```python
# ============================================
# Apply MOGPR for gap-filling and fusion
# ============================================
print("Starting MOGPR fusion (this may take a few minutes)...")

fused_data = mogpr(
    mogpr_data,
    variables=['VV', 'VH', 'S2ndvi'],  # All variables to fuse
    time_dimension='t',
    prediction_period=None,  # Use original time steps; or 'P5D' for 5-day outputs
    include_uncertainties=True,  # Include std dev estimates
    include_raw_inputs=True      # Keep original data for comparison
)

print("✓ MOGPR fusion complete")
print(f"Fused data variables: {list(fused_data.data_vars)}")

# ============================================
# Extract gap-filled NDVI
# ============================================
ndvi_fused = fused_data['S2ndvi_FUSED']  # Gap-filled NDVI
ndvi_std = fused_data['S2ndvi_STD']      # Uncertainty estimates
ndvi_raw = fused_data['S2ndvi_RAW']      # Original (with gaps)

# Also extract fused S1 data
vv_fused = fused_data['VV_FUSED']
vh_fused = fused_data['VH_FUSED']
```

### Step 1.4: Visualize Results (Python)

```python
# ============================================
# Visualize fusion results for a sample location
# ============================================
sample_location = 0  # First point

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Get location dimension name (depends on your CSV structure)
location_dim = [dim for dim in ndvi_raw.dims if dim != 't'][0]

# Plot 1: Gap-filled NDVI comparison
axes[0].plot(ndvi_raw['t'],
             ndvi_raw.isel({location_dim: sample_location}),
             'o', alpha=0.5, color='gray', markersize=6,
             label='Original NDVI (with cloud gaps)')
axes[0].plot(ndvi_fused['t'],
             ndvi_fused.isel({location_dim: sample_location}),
             '-', linewidth=2, color='green',
             label='MOGPR Gap-filled NDVI')
axes[0].fill_between(ndvi_fused['t'].values,
                      (ndvi_fused.isel({location_dim: sample_location}) -
                       ndvi_std.isel({location_dim: sample_location})).values,
                      (ndvi_fused.isel({location_dim: sample_location}) +
                       ndvi_std.isel({location_dim: sample_location})).values,
                      alpha=0.3, color='green', label='Uncertainty (±1 std)')
axes[0].set_ylabel('NDVI', fontsize=12)
axes[0].set_title('MOGPR Gap-Filling Results', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Sentinel-1 contribution
axes[1].plot(s1_data['t'],
             s1_data['VV'].isel({location_dim: sample_location}),
             '-o', alpha=0.7, label='VV (original)', color='blue')
axes[1].plot(vv_fused['t'],
             vv_fused.isel({location_dim: sample_location}),
             '--', label='VV (fused)', color='darkblue')
axes[1].set_ylabel('VV Backscatter (dB)', fontsize=12)
axes[1].set_title('Sentinel-1 VV: Original vs Fused', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Plot 3: Uncertainty over time
uncertainty_percent = (ndvi_std / ndvi_fused * 100).isel({location_dim: sample_location})
axes[2].plot(ndvi_fused['t'], uncertainty_percent, '-', color='red', linewidth=2)
axes[2].set_ylabel('Uncertainty (%)', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_title('MOGPR Uncertainty Over Time', fontsize=13)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mogpr_fusion_results_point.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved: mogpr_fusion_results_point.png")
```

### Step 1.5: Extract Phenology from Fused Data (Python)

```python
# ============================================
# Extract phenological metrics from gap-filled NDVI
# ============================================
pheno_results = phenology(
    ndvi_fused,
    detection_method='seasonal_amplitude',
    amplitude_threshold=0.2
)

# Extract metrics
sos_doy = pheno_results.da_sos_times.values      # Start of Season (day of year)
eos_doy = pheno_results.da_eos_times.values      # End of Season
los_days = pheno_results.da_los.values           # Length of Season
peak_ndvi = pheno_results.da_peak_value.values   # Peak NDVI
amplitude = pheno_results.da_season_amplitude.values

# Create results DataFrame
results_df = pd.DataFrame({
    'location_id': range(len(sos_doy)),
    'SOS_doy': sos_doy,
    'EOS_doy': eos_doy,
    'LOS_days': los_days,
    'peak_NDVI': peak_ndvi,
    'amplitude': amplitude
})

# Export
results_df.to_csv('phenology_from_mogpr_fusion.csv', index=False)
print("\n✓ Phenology metrics exported: phenology_from_mogpr_fusion.csv")
print(results_df.describe())
```

---

## Workflow 2: Raster-Based MOGPR (For Spatial Mapping)

### Overview
Raster-based MOGPR processes entire spatial grids pixel-by-pixel. This is ideal for:
- Wall-to-wall mapping
- Spatial phenology analysis
- Large-scale agricultural monitoring
- Creating gap-filled NDVI maps

**⚠️ Warning**: Raster MOGPR is computationally intensive. For large areas, consider:
- Processing in tiles
- Using chunking with Dask
- Starting with a small test area

### Step 2.1: Extract Spatial Datacube with sits (R)

```r
library(sits)
library(sf)

# Source helper functions
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")

# Define study area
study_area <- st_read("study_area.shp")

# ============================================
# Get Sentinel-1 spatial cube
# ============================================
s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("VV", "VH")
)

# ============================================
# Get Sentinel-2 spatial cube
# ============================================
s2_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("B04", "B08")
)

# Calculate NDVI
s2_ndvi_cube <- sits_apply(s2_cube,
  NDVI = (B08 - B04) / (B08 + B04)
)

# ============================================
# Export as GeoTIFF stacks
# ============================================
# Export S1 bands
sits_cube_to_fusets_geotiff(
  s1_cube,
  output_dir = "data/raster_mogpr/s1",
  bands = c("VV", "VH")
)

# Export S2 NDVI
sits_cube_to_fusets_geotiff(
  s2_ndvi_cube,
  output_dir = "data/raster_mogpr/s2",
  bands = c("NDVI")
)

print("✓ Spatial datacubes exported as GeoTIFF stacks")
```

### Step 2.2: Load Raster Stacks (Python)

```python
import sys
sys.path.insert(0, '/home/unika_sianturi/work/FuseTS/src')

import xarray as xr
import rioxarray as rxr
import pandas as pd
import numpy as np
from pathlib import Path

from fusets.io.sits_bridge import load_sits_geotiff
from fusets.mogpr import mogpr
from fusets.analytics import phenology

# ============================================
# Load GeoTIFF stacks
# ============================================
print("Loading raster stacks...")

# Load S1 bands
s1_vv_stack = load_sits_geotiff("data/raster_mogpr/s1/VV_stack.tif")
s1_vh_stack = load_sits_geotiff("data/raster_mogpr/s1/VH_stack.tif")

# Load S2 NDVI
s2_ndvi_stack = load_sits_geotiff("data/raster_mogpr/s2/NDVI_stack.tif")

print(f"✓ S1 VV shape: {s1_vv_stack.shape}")
print(f"✓ S1 VH shape: {s1_vh_stack.shape}")
print(f"✓ S2 NDVI shape: {s2_ndvi_stack.shape}")
print(f"✓ CRS: {s2_ndvi_stack.rio.crs}")

# ============================================
# Prepare for MOGPR
# ============================================
# IMPORTANT: MOGPR requires xarray.Dataset with specific structure:
# - Dimensions: (t, x, y) or (t, y, x)
# - Variable names: 'VV', 'VH', 'S2ndvi'
# - Consistent time coordinates

# Create xarray Dataset
mogpr_raster_data = xr.Dataset({
    'VV': s1_vv_stack.rename({'band': 't'}),
    'VH': s1_vh_stack.rename({'band': 't'}),
    'S2ndvi': s2_ndvi_stack.rename({'band': 't'})
})

# Ensure all bands have the same time coordinates
# Use S2 dates as reference (more reliable due to cloud masking)
# In practice, you may need to align/interpolate S1 and S2 dates

print(f"\n✓ MOGPR raster data prepared")
print(f"  Dimensions: {mogpr_raster_data.dims}")
print(f"  Variables: {list(mogpr_raster_data.data_vars)}")
```

### Step 2.3: Apply MOGPR to Raster (With Chunking)

```python
import dask.array as da

# ============================================
# Load data with Dask chunking for memory efficiency
# ============================================
print("Loading raster with Dask chunking...")

# Reload with chunks
s1_vv_stack = rxr.open_rasterio(
    "data/raster_mogpr/s1/VV_stack.tif",
    chunks={'band': 10, 'y': 512, 'x': 512}  # Adjust based on available memory
)
s1_vh_stack = rxr.open_rasterio(
    "data/raster_mogpr/s1/VH_stack.tif",
    chunks={'band': 10, 'y': 512, 'x': 512}
)
s2_ndvi_stack = rxr.open_rasterio(
    "data/raster_mogpr/s2/NDVI_stack.tif",
    chunks={'band': 10, 'y': 512, 'x': 512}
)

# Create Dataset
mogpr_raster_data = xr.Dataset({
    'VV': s1_vv_stack.rename({'band': 't'}),
    'VH': s1_vh_stack.rename({'band': 't'}),
    'S2ndvi': s2_ndvi_stack.rename({'band': 't'})
})

# ============================================
# Process in spatial tiles (for very large areas)
# ============================================
# Option A: Process entire area (if memory allows)
print("\nApplying MOGPR fusion to raster...")
print("⚠️  This may take significant time for large areas...")

fused_raster = mogpr(
    mogpr_raster_data,
    variables=['VV', 'VH', 'S2ndvi'],
    time_dimension='t',
    prediction_period='P5D',  # 5-day composites
    include_uncertainties=True,
    include_raw_inputs=False  # Save memory
)

# Compute the result (triggers Dask computation)
print("Computing fused raster (this will take time)...")
fused_raster = fused_raster.compute()

print("✓ MOGPR raster fusion complete")

# Option B: Process in spatial tiles (for large areas)
# See "Performance Optimization" section below
```

### Step 2.4: Export Fused Rasters (Python)

```python
# ============================================
# Export gap-filled NDVI as GeoTIFF
# ============================================
ndvi_fused_raster = fused_raster['S2ndvi_FUSED']
ndvi_uncertainty = fused_raster['S2ndvi_STD']

# Export time series stack
ndvi_fused_raster.rio.to_raster(
    'outputs/NDVI_gapfilled_stack.tif',
    compress='lzw',
    tiled=True
)

ndvi_uncertainty.rio.to_raster(
    'outputs/NDVI_uncertainty_stack.tif',
    compress='lzw',
    tiled=True
)

print("✓ Fused NDVI raster exported")

# ============================================
# Export individual time steps
# ============================================
output_dir = Path('outputs/ndvi_timeseries')
output_dir.mkdir(parents=True, exist_ok=True)

for i, time_step in enumerate(ndvi_fused_raster['t'].values):
    filename = f"NDVI_fused_{pd.Timestamp(time_step).strftime('%Y%m%d')}.tif"
    ndvi_fused_raster.isel(t=i).rio.to_raster(
        output_dir / filename,
        compress='lzw'
    )

print(f"✓ Exported {len(ndvi_fused_raster['t'])} individual GeoTIFFs")
```

### Step 2.5: Spatial Phenology Mapping (Python)

```python
# ============================================
# Extract phenology metrics from fused raster
# ============================================
print("Extracting spatial phenology metrics...")

pheno_raster = phenology(
    ndvi_fused_raster,
    detection_method='seasonal_amplitude',
    amplitude_threshold=0.2
)

print("✓ Phenology extraction complete")

# ============================================
# Export phenology maps
# ============================================
pheno_outputs = {
    'SOS_map.tif': pheno_raster.da_sos_times,
    'EOS_map.tif': pheno_raster.da_eos_times,
    'LOS_map.tif': pheno_raster.da_los,
    'peak_NDVI_map.tif': pheno_raster.da_peak_value,
    'amplitude_map.tif': pheno_raster.da_season_amplitude,
    'productivity_map.tif': pheno_raster.da_seasonal_integral
}

for filename, data_array in pheno_outputs.items():
    data_array.rio.to_raster(
        f'outputs/{filename}',
        compress='lzw',
        tiled=True
    )

print(f"✓ Exported {len(pheno_outputs)} phenology maps")
```

### Step 2.6: Visualize Phenology Maps (Python)

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================
# Create phenology map visualization
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# SOS map
im1 = axes[0, 0].imshow(pheno_raster.da_sos_times,
                        cmap='RdYlGn', vmin=1, vmax=365)
axes[0, 0].set_title('Start of Season (Day of Year)', fontsize=13, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# EOS map
im2 = axes[0, 1].imshow(pheno_raster.da_eos_times,
                        cmap='RdYlGn_r', vmin=1, vmax=365)
axes[0, 1].set_title('End of Season (Day of Year)', fontsize=13, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# LOS map
im3 = axes[0, 2].imshow(pheno_raster.da_los,
                        cmap='viridis', vmin=0, vmax=200)
axes[0, 2].set_title('Length of Season (Days)', fontsize=13, fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

# Peak NDVI
im4 = axes[1, 0].imshow(pheno_raster.da_peak_value,
                        cmap='YlGn', vmin=0, vmax=1)
axes[1, 0].set_title('Peak NDVI Value', fontsize=13, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Amplitude
im5 = axes[1, 1].imshow(pheno_raster.da_season_amplitude,
                        cmap='plasma', vmin=0, vmax=1)
axes[1, 1].set_title('Season Amplitude', fontsize=13, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

# Productivity
im6 = axes[1, 2].imshow(pheno_raster.da_seasonal_integral,
                        cmap='copper_r')
axes[1, 2].set_title('Seasonal Integral (Productivity)', fontsize=13, fontweight='bold')
axes[1, 2].axis('off')
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.suptitle('MOGPR-Fused Phenology Maps', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('outputs/phenology_maps_from_mogpr.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Phenology maps visualization saved")
```

---

## Understanding MOGPR Data Requirements

### Critical Data Format Requirements

MOGPR expects data in a very specific format. Understanding this is key to successful fusion.

#### 1. Dimension Names
```python
# ✓ CORRECT - dimension must be named 't' (not 'time', 'band', or 'date')
ds = xr.Dataset({
    'VV': (['t', 'y', 'x'], data_vv),
    'VH': (['t', 'y', 'x'], data_vh),
    'S2ndvi': (['t', 'y', 'x'], data_ndvi)
})

# ✗ INCORRECT - will fail
ds = xr.Dataset({
    'VV': (['time', 'y', 'x'], data_vv),  # Wrong: 'time' instead of 't'
})
```

#### 2. Variable Naming Conventions
```python
# For S1+S2 fusion, use these exact names:
# - 'VV': Sentinel-1 VV polarization
# - 'VH': Sentinel-1 VH polarization
# - 'S2ndvi': Sentinel-2 NDVI (note: 'S2ndvi' not 'NDVI')

# After fusion, variables are renamed:
# - 'VV_FUSED': Gap-filled VV
# - 'VH_FUSED': Gap-filled VH
# - 'S2ndvi_FUSED': Gap-filled NDVI
# - 'VV_STD': Uncertainty estimate for VV
# - 'S2ndvi_RAW': Original NDVI (if include_raw_inputs=True)
```

#### 3. Time Coordinate Format
```python
# Time coordinates should be datetime objects
import pandas as pd

# ✓ CORRECT
time_coords = pd.date_range('2024-01-01', periods=30, freq='12D')
ds = ds.assign_coords({'t': time_coords})

# ✓ ALSO CORRECT (from sits CSV)
time_coords = pd.to_datetime(df['Index'])
ds = ds.assign_coords({'t': time_coords})
```

#### 4. Handling Missing Data
```python
# MOGPR handles NaN values for gap-filling
# Mark cloudy/missing pixels as NaN

# Example: mask cloudy S2 pixels
s2_ndvi_clean = s2_ndvi.where(cloud_mask == 0, np.nan)

# MOGPR will use S1 data to fill these gaps
```

### Data Structure Comparison

| Source | sits CSV (point) | sits GeoTIFF (raster) | Required for MOGPR |
|--------|------------------|----------------------|-------------------|
| **Dimensions** | `(t, location_id)` | `(band, y, x)` | `(t, y, x)` or `(t, location_id)` |
| **Time dim name** | `Index` or `time` | `band` | `t` (must rename!) |
| **Format** | pandas → xarray | rasterio → xarray | xarray.Dataset |
| **Variables** | Multiple columns | Separate files/bands | Named DataArrays in Dataset |

---

## Common Use Cases

### Use Case 1: Agricultural Monitoring with S1+S2 Fusion

**Goal**: Fill gaps in Sentinel-2 NDVI during cloudy season using Sentinel-1 radar

```python
# Full workflow example
from fusets.io.sits_bridge import load_sits_csv, prepare_mogpr_format
from fusets.mogpr import mogpr
from fusets.analytics import phenology

# Load data
s1 = load_sits_csv("s1_data.csv", band_cols=['VV', 'VH'])
s2 = load_sits_csv("s2_data.csv", band_cols=['NDVI'])

# Prepare
mogpr_input = prepare_mogpr_format(
    s1_vv=s1['VV'],
    s1_vh=s1['VH'],
    s2_ndvi=s2['NDVI'],
    time_coords=s2['t']
)

# Fuse
fused = mogpr(mogpr_input, include_uncertainties=True)

# Extract phenology from gap-filled NDVI
pheno = phenology(fused['S2ndvi_FUSED'])

# Export planting dates
planting_dates = pheno.da_sos_times
```

### Use Case 2: Multi-Year Phenology Trend Analysis

```python
# Process multiple years and analyze trends
years = [2022, 2023, 2024]
phenology_results = {}

for year in years:
    # Load year-specific data
    s1 = load_sits_csv(f"data/s1_{year}.csv", band_cols=['VV', 'VH'])
    s2 = load_sits_csv(f"data/s2_{year}.csv", band_cols=['NDVI'])

    # MOGPR fusion
    mogpr_input = prepare_mogpr_format(s1['VV'], s1['VH'], s2['NDVI'], s2['t'])
    fused = mogpr(mogpr_input)

    # Extract phenology
    pheno = phenology(fused['S2ndvi_FUSED'])
    phenology_results[year] = pheno

# Analyze SOS trends
import pandas as pd
trend_df = pd.DataFrame({
    year: results.da_sos_times.values
    for year, results in phenology_results.items()
})

# Calculate trend (e.g., earlier/later planting over time)
trend_df.mean(axis=0).plot(title='Start of Season Trend')
```

### Use Case 3: Yield Prediction Features

```python
# Extract comprehensive phenology features for ML
from fusets.mogpr import mogpr
from fusets.analytics import phenology
import pandas as pd

# MOGPR fusion
fused = mogpr(mogpr_input)

# Extract phenology
pheno = phenology(fused['S2ndvi_FUSED'])

# Create feature matrix
features = pd.DataFrame({
    'sos_doy': pheno.da_sos_times.values,
    'eos_doy': pheno.da_eos_times.values,
    'los_days': pheno.da_los.values,
    'peak_ndvi': pheno.da_peak_value.values,
    'amplitude': pheno.da_season_amplitude.values,
    'integral': pheno.da_seasonal_integral.values,
    'sos_value': pheno.da_sos_values.values,
    'eos_value': pheno.da_eos_values.values
})

# Train yield model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(features, observed_yields)
```

---

## Troubleshooting

### Issue 1: "Dimension 't' not found"

**Cause**: Time dimension is not named 't'

**Solution**:
```python
# Check current dimension names
print(ds.dims)  # e.g., {'time': 30, 'x': 100, 'y': 100}

# Rename to 't'
ds = ds.rename({'time': 't'})  # or 'band', 'Index', etc.
```

### Issue 2: "Variables VV, VH, S2ndvi not found"

**Cause**: Incorrect variable naming

**Solution**:
```python
# Check current variables
print(list(ds.data_vars))  # e.g., ['NDVI', 'sentinel1_vv']

# Rename variables
ds = ds.rename({
    'sentinel1_vv': 'VV',
    'sentinel1_vh': 'VH',
    'NDVI': 'S2ndvi'
})
```

### Issue 3: "Time coordinates mismatch"

**Cause**: S1 and S2 have different acquisition dates

**Solution**:
```python
# Option A: Interpolate to common time grid
import numpy as np
import pandas as pd

common_times = pd.date_range('2024-01-01', '2024-12-31', freq='12D')

s1_interp = s1.interp(t=common_times, method='linear')
s2_interp = s2.interp(t=common_times, method='linear')

# Option B: Use S2 times as reference (recommended)
# MOGPR will use all available S1 data to fill S2 gaps
mogpr_input = prepare_mogpr_format(
    s1_vv=s1['VV'],
    s1_vh=s1['VH'],
    s2_ndvi=s2['NDVI'],
    time_coords=s2['t']  # Use S2 times
)
```

### Issue 4: Memory errors with large rasters

**Solution**:
```python
# Use Dask chunking
import dask.array as da

# Load with chunks
ds = xr.open_dataset('large_file.nc', chunks={'t': 10, 'y': 512, 'x': 512})

# Or process in spatial tiles (see Performance Optimization)
```

### Issue 5: "GPy import error"

**Cause**: GPy not installed

**Solution**:
```bash
pip install GPy
# If issues persist:
conda install -c conda-forge gpy
```

---

## Performance Optimization

### Strategy 1: Spatial Tiling for Large Rasters

```python
import numpy as np
from pathlib import Path

def process_raster_in_tiles(input_ds, tile_size=1000, output_dir='outputs/tiles'):
    """
    Process large raster in spatial tiles to manage memory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = input_ds.dims['y'], input_ds.dims['x']

    # Calculate number of tiles
    n_tiles_y = int(np.ceil(height / tile_size))
    n_tiles_x = int(np.ceil(width / tile_size))

    print(f"Processing {n_tiles_y} × {n_tiles_x} = {n_tiles_y * n_tiles_x} tiles")

    tile_results = []

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y_start = i * tile_size
            y_end = min((i + 1) * tile_size, height)
            x_start = j * tile_size
            x_end = min((j + 1) * tile_size, width)

            print(f"Processing tile ({i}, {j}): y={y_start}:{y_end}, x={x_start}:{x_end}")

            # Extract tile
            tile = input_ds.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))

            # Process tile with MOGPR
            tile_fused = mogpr(tile, include_uncertainties=False)

            # Save tile
            tile_filename = output_dir / f"tile_{i:03d}_{j:03d}.nc"
            tile_fused.to_netcdf(tile_filename)

            tile_results.append((i, j, tile_fused))

            # Clean up memory
            del tile, tile_fused

    print("✓ All tiles processed")
    return tile_results

# Usage
fused_tiles = process_raster_in_tiles(mogpr_raster_data, tile_size=500)
```

### Strategy 2: Temporal Chunking

```python
def process_temporal_chunks(input_ds, time_chunk_size=10):
    """
    Process long time series in temporal chunks
    """
    n_times = len(input_ds['t'])
    n_chunks = int(np.ceil(n_times / time_chunk_size))

    results = []

    for i in range(n_chunks):
        t_start = i * time_chunk_size
        t_end = min((i + 1) * time_chunk_size, n_times)

        print(f"Processing temporal chunk {i+1}/{n_chunks}: t={t_start}:{t_end}")

        chunk = input_ds.isel(t=slice(t_start, t_end))
        chunk_fused = mogpr(chunk)
        results.append(chunk_fused)

    # Concatenate results
    full_result = xr.concat(results, dim='t')
    return full_result
```

### Strategy 3: Parallel Processing

```python
from multiprocessing import Pool
import functools

def process_single_location(loc_id, data):
    """Process single location time series"""
    mogpr_input = prepare_mogpr_format(
        s1_vv=data['VV'].isel(location=loc_id),
        s1_vh=data['VH'].isel(location=loc_id),
        s2_ndvi=data['NDVI'].isel(location=loc_id),
        time_coords=data['t']
    )
    return mogpr(mogpr_input)

# Parallel processing
with Pool(processes=4) as pool:
    process_func = functools.partial(process_single_location, data=combined_data)
    results = pool.map(process_func, range(n_locations))
```

---

## Summary and Best Practices

### ✓ Best Practices

1. **Start small**: Test with a few points or small tile before processing entire region
2. **Use consistent time grids**: Align S1 and S2 to common time steps
3. **Check data quality**: Visualize raw data before fusion
4. **Monitor memory**: Use chunking for large rasters
5. **Validate results**: Compare fused NDVI with ground truth when available
6. **Export uncertainties**: Use `include_uncertainties=True` to assess fusion quality
7. **Document parameters**: Record lambda values, thresholds, detection methods

### Common Workflow Checklist

- [ ] Data extracted from sits (CSV or GeoTIFF)
- [ ] Time dimension renamed to 't'
- [ ] Variables renamed to FuseTS conventions ('VV', 'VH', 'S2ndvi')
- [ ] Time coordinates are datetime objects
- [ ] Missing/cloudy pixels marked as NaN
- [ ] Test run on small subset
- [ ] Visualize results before full processing
- [ ] Export fused data and uncertainties
- [ ] Extract phenology metrics
- [ ] Validate against ground truth (if available)

### Quick Reference

| Task | Function | Module |
|------|----------|--------|
| Load sits CSV | `load_sits_csv()` | `fusets.io.sits_bridge` |
| Load sits GeoTIFF | `load_sits_geotiff()` | `fusets.io.sits_bridge` |
| Prepare S1+S2 data | `prepare_mogpr_format()` | `fusets.io.sits_bridge` |
| Apply MOGPR fusion | `mogpr()` | `fusets.mogpr` |
| Extract phenology | `phenology()` | `fusets.analytics` |
| Whittaker smoothing | `whittaker()` | `fusets` |

---

## Additional Resources

- **FuseTS Main Documentation**: `/home/unika_sianturi/work/FuseTS/README.md`
- **sits + FuseTS Workflow**: `/home/unika_sianturi/work/FuseTS/SITS_FUSETS_WORKFLOW.md`
- **Example Notebook**: `/home/unika_sianturi/work/FuseTS/notebooks/Paddyfield_Phenology_sits_FuseTS.ipynb`
- **sits Documentation**: https://e-sensing.github.io/sitsbook/
- **MOGPR Paper**: Pipia et al. (2019) - https://doi.org/10.1016/j.rse.2019.111452

---

**Last Updated**: 2025-11-15
**FuseTS Version**: Compatible with v0.2.0+
