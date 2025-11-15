# sits + FuseTS Integration Workflow

This guide demonstrates how to use the R `sits` package for data acquisition and FuseTS for advanced time series analytics, sensor fusion, and phenology extraction.

## Why Use sits + FuseTS Together?

### sits Strengths (Data Acquisition)
- **Multiple data sources**: Microsoft Planetary Computer, Brazil Data Cube, AWS, Sentinel Hub
- **No authentication complexity**: Often easier than GEE authentication
- **Built-in preprocessing**: Cloud masking, harmonization, quality filtering
- **Efficient time series extraction**: Optimized for large-scale extraction
- **R ecosystem**: Integrates with `sf`, `terra`, `stars` for geospatial operations

### FuseTS Strengths (Advanced Analytics)
- **MOGPR sensor fusion**: Leverage correlations between S1/S2 for gap-filling
- **Whittaker smoothing**: Fast, robust temporal smoothing
- **Phenological metrics**: TIMESAT-based SOS/EOS detection with multiple methods
- **Deep learning fusion**: CropSAR models for advanced S1+S2 integration
- **OpenEO integration**: Scale to cloud processing when needed

## Installation

### R Setup
```r
install.packages("sits")
install.packages("sf")
install.packages("dplyr")
install.packages("tidyr")

# Source the helper functions
source("scripts/sits_to_fusets.R")
```

### Python Setup
```bash
cd /home/unika_sianturi/work/FuseTS
pip install -e .
pip install rioxarray  # For GeoTIFF support
```

## Workflow 1: Point-Based Time Series Analysis

### Step 1: Extract Time Series with sits (R)

```r
library(sits)
library(sf)

# Define your study area
roi <- st_read("study_area.geojson")

# Option A: From Microsoft Planetary Computer
cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = roi,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("B04", "B08", "B11")  # Red, NIR, SWIR
)

# Get S2 time series
s2_samples <- sits_get_data(cube, samples = field_points)

# Option B: Get Sentinel-1 data
s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = roi,
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("VV", "VH")
)

s1_samples <- sits_get_data(s1_cube, samples = field_points)

# Calculate vegetation indices
s2_samples <- sits_apply(s2_samples,
  NDVI = (B08 - B04) / (B08 + B04),
  EVI = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1)
)

# Export to FuseTS-compatible CSV
sits_to_fusets_csv(s2_samples, "s2_timeseries.csv",
                   bands = c("NDVI", "EVI"))

sits_to_fusets_csv(s1_samples, "s1_timeseries.csv",
                   bands = c("VV", "VH"))
```

### Step 2: Process with FuseTS (Python)

```python
import pandas as pd
from fusets.io.sits_bridge import load_sits_csv, prepare_mogpr_format
from fusets.mogpr import MOGPRTransformer
from fusets import whittaker
from fusets.analytics import phenology
import matplotlib.pyplot as plt

# Load S1 and S2 data
s1_data = load_sits_csv("s1_timeseries.csv",
                        time_col='Index',
                        band_cols=['VV', 'VH'])

s2_data = load_sits_csv("s2_timeseries.csv",
                        time_col='Index',
                        band_cols=['NDVI'])

# Option 1: MOGPR Fusion for gap-filling
# Combine S1 and S2 into single dataset
combined = prepare_mogpr_format(
    s1_vv=s1_data['VV'],
    s1_vh=s1_data['VH'],
    s2_ndvi=s2_data['NDVI'],
    time_coords=s2_data['t']
)

# Apply MOGPR fusion
mogpr = MOGPRTransformer()
fused = mogpr.fit_transform(combined)

# Extract gap-filled NDVI
ndvi_gapfilled = fused['S2ndvi']

# Option 2: Whittaker smoothing only (faster)
ndvi_smoothed = whittaker(s2_data['NDVI'], lmbd=10000)

# Extract phenological metrics
pheno_results = phenology(ndvi_gapfilled,
                         detection_method='seasonal_amplitude',
                         amplitude_threshold=0.2)

# Access results
sos_doy = pheno_results.da_sos_times      # Start of Season (day of year)
eos_doy = pheno_results.da_eos_times      # End of Season (day of year)
los = pheno_results.da_los                # Length of Season (days)
season_amplitude = pheno_results.da_season_amplitude
peak_value = pheno_results.da_peak_value

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Time series with phenological markers
axes[0].plot(s2_data['t'], s2_data['NDVI'], 'o', alpha=0.5, label='Original NDVI')
axes[0].plot(s2_data['t'], ndvi_gapfilled, '-', label='MOGPR Gap-filled')
axes[0].axvline(sos_doy, color='green', linestyle='--', label='SOS')
axes[0].axvline(eos_doy, color='red', linestyle='--', label='EOS')
axes[0].legend()
axes[0].set_ylabel('NDVI')
axes[0].set_title('Time Series with Phenological Markers')

# Plot 2: S1 contribution
axes[1].plot(s1_data['t'], s1_data['VV'], label='VV')
axes[1].plot(s1_data['t'], s1_data['VH'], label='VH')
axes[1].legend()
axes[1].set_ylabel('Backscatter (dB)')
axes[1].set_xlabel('Date')
axes[1].set_title('Sentinel-1 Time Series')

plt.tight_layout()
plt.savefig('phenology_analysis.png', dpi=300)

# Export results
results_df = pd.DataFrame({
    'location': s2_data.coords.get('location', range(len(sos_doy))),
    'SOS_doy': sos_doy.values,
    'EOS_doy': eos_doy.values,
    'LOS_days': los.values,
    'peak_NDVI': peak_value.values,
    'amplitude': season_amplitude.values
})
results_df.to_csv('phenology_metrics.csv', index=False)
```

## Workflow 2: Spatial Raster Processing

### Step 1: Create Raster Time Series Cube (R)

```r
library(sits)

# Get spatial cube for tile
cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  tiles = "my_tile_id",
  start_date = "2024-01-01",
  end_date = "2024-12-31",
  bands = c("B04", "B08")
)

# Calculate NDVI across all images
ndvi_cube <- sits_apply(cube,
  NDVI = (B08 - B04) / (B08 + B04)
)

# Export as multi-band GeoTIFF stack
sits_cube_to_fusets_geotiff(ndvi_cube,
                             output_dir = "fusets_input",
                             bands = "NDVI")

# This creates: fusets_input/NDVI_stack.tif
```

### Step 2: Spatial Phenology Mapping (Python)

```python
import rioxarray as rxr
import xarray as xr
import numpy as np
from fusets.io.sits_bridge import load_sits_geotiff
from fusets import whittaker
from fusets.analytics import phenology
import matplotlib.pyplot as plt

# Load GeoTIFF stack
ndvi_stack = load_sits_geotiff("fusets_input/NDVI_stack.tif")

# Smooth time series spatially
ndvi_smoothed = whittaker(ndvi_stack, lmbd=10000)

# Extract phenological metrics for entire raster
pheno_map = phenology(ndvi_smoothed,
                      detection_method='seasonal_amplitude',
                      amplitude_threshold=0.2)

# Export phenology maps as GeoTIFF
pheno_map.da_sos_times.rio.to_raster("SOS_map.tif")
pheno_map.da_eos_times.rio.to_raster("EOS_map.tif")
pheno_map.da_los.rio.to_raster("LOS_map.tif")
pheno_map.da_peak_value.rio.to_raster("peak_NDVI_map.tif")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SOS map
im1 = axes[0, 0].imshow(pheno_map.da_sos_times, cmap='RdYlGn', vmin=1, vmax=365)
axes[0, 0].set_title('Start of Season (Day of Year)')
plt.colorbar(im1, ax=axes[0, 0])

# EOS map
im2 = axes[0, 1].imshow(pheno_map.da_eos_times, cmap='RdYlGn_r', vmin=1, vmax=365)
axes[0, 1].set_title('End of Season (Day of Year)')
plt.colorbar(im2, ax=axes[0, 1])

# Length of Season
im3 = axes[1, 0].imshow(pheno_map.da_los, cmap='viridis', vmin=0, vmax=365)
axes[1, 0].set_title('Length of Season (Days)')
plt.colorbar(im3, ax=axes[1, 0])

# Peak NDVI
im4 = axes[1, 1].imshow(pheno_map.da_peak_value, cmap='YlGn', vmin=0, vmax=1)
axes[1, 1].set_title('Peak NDVI Value')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('phenology_maps.png', dpi=300)
```

## Workflow 3: Multi-Sensor Fusion with Deep Learning

For advanced S1+S2 fusion using CropSAR models:

```python
from fusets.cropsar import CropSARFusion
from fusets.io.sits_bridge import load_sits_geotiff

# Load S1 VV, VH and S2 NDVI stacks
s1_vv = load_sits_geotiff("fusets_input/S1_VV_stack.tif")
s1_vh = load_sits_geotiff("fusets_input/S1_VH_stack.tif")
s2_ndvi = load_sits_geotiff("fusets_input/S2_NDVI_stack.tif")

# Apply CropSAR fusion
cropsar = CropSARFusion(model_type='AttentionUNet')
fused_ndvi = cropsar.predict(s1_vv, s1_vh, s2_ndvi)

# Extract phenology from fused result
pheno = phenology(fused_ndvi, detection_method='seasonal_amplitude')
```

## Workflow 4: Seamless R-Python Integration with Reticulate

For users comfortable with both languages:

```r
library(sits)
library(reticulate)

# Set up Python environment
use_condaenv("fusets_env")

# Import FuseTS
fusets <- import("fusets")
sits_bridge <- import("fusets.io.sits_bridge")

# Get data with sits
cube <- sits_cube(source = "MPC", ...)
samples <- sits_get_data(cube, samples = points)

# Export and immediately process
sits_to_fusets_csv(samples, "temp_data.csv")

# Load in Python
py_data <- sits_bridge$load_sits_csv("temp_data.csv")

# Apply MOGPR
mogpr <- fusets$mogpr$MOGPRTransformer()
fused <- mogpr$fit_transform(py_data)

# Extract phenology
pheno <- fusets$analytics$phenology(fused)

# Access results back in R
sos_values <- py_to_r(pheno$da_sos_times)
```

## Practical Examples by Use Case

### Agricultural Monitoring (Paddy Rice)
```r
# R: Extract S1+S2 time series for paddy fields
paddy_points <- st_read("paddy_training_points.shp")
s1_s2_data <- sits_get_data(cube, samples = paddy_points)
sits_to_fusets_csv(s1_s2_data, "paddy_timeseries.csv")
```

```python
# Python: Gap-fill and extract planting dates
from fusets.io.sits_bridge import load_sits_csv
from fusets import whittaker
from fusets.analytics import phenology

data = load_sits_csv("paddy_timeseries.csv")
smoothed = whittaker(data['NDVI'], lmbd=5000)
pheno = phenology(smoothed, detection_method='first_of_slope')

planting_dates = pheno.da_sos_times  # Planting ~ SOS
harvest_dates = pheno.da_eos_times   # Harvest ~ EOS
```

### Forest Phenology
```r
# R: Extract forest NDVI time series
forest_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = forest_roi,
  bands = c("B04", "B08")
)
sits_cube_to_fusets_geotiff(forest_cube, "forest_ndvi")
```

```python
# Python: Detect greening/senescence timing
from fusets.io.sits_bridge import load_sits_geotiff
from fusets.analytics import phenology

ndvi = load_sits_geotiff("forest_ndvi/NDVI_stack.tif")
pheno = phenology(ndvi, detection_method='median_of_slope')

green_up = pheno.da_sos_times    # Leaf-on timing
senescence = pheno.da_eos_times  # Leaf-off timing
```

### Crop Yield Prediction
```python
# Use phenology metrics as ML features
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Extract features
features_df = pd.DataFrame({
    'sos': pheno.da_sos_times.values.flatten(),
    'eos': pheno.da_eos_times.values.flatten(),
    'los': pheno.da_los.values.flatten(),
    'peak_ndvi': pheno.da_peak_value.values.flatten(),
    'amplitude': pheno.da_season_amplitude.values.flatten(),
    'integral': pheno.da_seasonal_integral.values.flatten()
})

# Train yield model
rf = RandomForestRegressor()
rf.fit(features_df, observed_yields)
```

## Comparison: sits vs GEE for FuseTS Workflows

| Aspect | sits (MPC/BDC) | Google Earth Engine |
|--------|----------------|---------------------|
| **Authentication** | Simple (often no auth needed) | More complex (service account/OAuth) |
| **Data sources** | MPC, BDC, AWS, Sentinel Hub | Extensive catalog |
| **Processing location** | Local or cloud | Always cloud |
| **Time series extraction** | Very efficient | Can be slow for large extractions |
| **Export format** | Direct to CSV/GeoTIFF | Must export to Drive, then download |
| **R integration** | Native | Via rgee package |
| **Learning curve** | Moderate | Steeper |
| **Best for** | Local/regional analysis, R users | Large-scale, global analysis |

## Troubleshooting

### Issue: "rioxarray not available"
```bash
pip install rioxarray
```

### Issue: sits authentication fails
```r
# For MPC (usually no auth needed)
Sys.setenv("MPC_TOKEN" = "your_token")

# For AWS
Sys.setenv("AWS_ACCESS_KEY_ID" = "your_key")
Sys.setenv("AWS_SECRET_ACCESS_KEY" = "your_secret")
```

### Issue: Memory errors with large rasters
```python
# Process in chunks
import dask.array as da

# Load with dask
ndvi = load_sits_geotiff("large_stack.tif", chunks={'t': 10, 'y': 1000, 'x': 1000})
smoothed = whittaker(ndvi, lmbd=10000)
smoothed.compute()  # Process with dask
```

### Issue: MOGPR dimension mismatch
```python
# Ensure all inputs have same time dimension
print(s1_data.dims)  # Should be: ('t', ...)
print(s2_data.dims)  # Should be: ('t', ...)

# Rename if needed
s1_data = s1_data.rename({'time': 't'})
```

## Performance Tips

1. **Use sits for data acquisition**: Much faster than manual API calls
2. **Process point data in Python**: MOGPR works best on point time series
3. **Use chunking for large rasters**: Essential for memory management
4. **Cache intermediate results**: Save smoothed data before phenology extraction
5. **Use Whittaker for speed, MOGPR for accuracy**: Choose based on requirements

## References

- **sits documentation**: https://e-sensing.github.io/sitsbook/
- **FuseTS documentation**: See `FuseTS/README.md`
- **Microsoft Planetary Computer**: https://planetarycomputer.microsoft.com/
- **TIMESAT phenology methods**: Jönsson & Eklundh (2004)

## Next Steps

1. **Install sits and FuseTS** following installation instructions above
2. **Run the R helper script**: `source("scripts/sits_to_fusets.R")`
3. **Try the point-based workflow** with a small sample of field locations
4. **Explore spatial processing** once comfortable with point analysis
5. **Experiment with different phenology detection methods** for your use case

For questions or issues, see the main FuseTS documentation or open an issue on GitHub.
