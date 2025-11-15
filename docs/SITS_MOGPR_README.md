# SITS to MOGPR Integration - Quick Start

This directory contains documentation for integrating R `sits` package data with FuseTS MOGPR workflows.

## 📚 Documentation Files

### Main Guides

1. **[SITS_TO_MOGPR_GUIDE.md](../SITS_TO_MOGPR_GUIDE.md)** ⭐ **START HERE**
   - Complete guide for saving sits datacubes and using them with MOGPR
   - Covers both **point-based** and **raster-based** workflows
   - Includes code examples, troubleshooting, and performance optimization
   - **Best for**: New users, comprehensive workflows, production use

2. **[SITS_FUSETS_WORKFLOW.md](../SITS_FUSETS_WORKFLOW.md)**
   - General sits + FuseTS integration (Whittaker, phenology, CropSAR)
   - Multiple workflow options including reticulate integration
   - Use case examples and performance tips
   - **Best for**: Understanding sits advantages, general FuseTS workflows

### Code Resources

3. **[src/fusets/io/sits_bridge.py](../src/fusets/io/sits_bridge.py)**
   - Python bridge module for loading sits data
   - Functions: `load_sits_csv()`, `load_sits_geotiff()`, `prepare_mogpr_format()`
   - **Best for**: API reference, understanding data structures

4. **[scripts/sits_to_fusets.R](../scripts/sits_to_fusets.R)**
   - R helper functions for exporting sits data
   - Functions: `sits_to_fusets_csv()`, `sits_cube_to_fusets_geotiff()`
   - **Best for**: R users, custom export workflows

### Example Notebooks

5. **[notebooks/Paddyfield_Phenology_sits_FuseTS.ipynb](../notebooks/Paddyfield_Phenology_sits_FuseTS.ipynb)**
   - End-to-end paddyfield phenology analysis
   - Demonstrates sits extraction → FuseTS smoothing → phenology
   - Includes visualization and filtering for agricultural patterns
   - **Best for**: Learning by example, paddyfield applications

## 🚀 Quick Start

### Point-Based MOGPR (5-Minute Workflow)

**In R:**
```r
library(sits)
source("scripts/sits_to_fusets.R")

# Extract S1+S2 time series
s1_samples <- sits_get_data(s1_cube, samples = points)
s2_samples <- sits_get_data(s2_cube, samples = points)

# Export
sits_to_fusets_csv(s1_samples, "s1.csv", bands = c("VV", "VH"))
sits_to_fusets_csv(s2_samples, "s2.csv", bands = c("NDVI"))
```

**In Python:**
```python
from fusets.io.sits_bridge import load_sits_csv, prepare_mogpr_format
from fusets.mogpr import mogpr

# Load
s1 = load_sits_csv("s1.csv", band_cols=['VV', 'VH'])
s2 = load_sits_csv("s2.csv", band_cols=['NDVI'])

# Prepare and fuse
data = prepare_mogpr_format(s1['VV'], s1['VH'], s2['NDVI'], s2['t'])
fused = mogpr(data, include_uncertainties=True)

# Gap-filled NDVI
ndvi_gapfilled = fused['S2ndvi_FUSED']
```

### Raster-Based MOGPR

**In R:**
```r
# Export spatial cube as GeoTIFF
sits_cube_to_fusets_geotiff(s1_cube, "output/s1", bands = c("VV", "VH"))
sits_cube_to_fusets_geotiff(s2_cube, "output/s2", bands = c("NDVI"))
```

**In Python:**
```python
from fusets.io.sits_bridge import load_sits_geotiff
from fusets.mogpr import mogpr
import xarray as xr

# Load rasters
s1_vv = load_sits_geotiff("output/s1/VV_stack.tif")
s1_vh = load_sits_geotiff("output/s1/VH_stack.tif")
s2_ndvi = load_sits_geotiff("output/s2/NDVI_stack.tif")

# Prepare dataset
data = xr.Dataset({
    'VV': s1_vv.rename({'band': 't'}),
    'VH': s1_vh.rename({'band': 't'}),
    'S2ndvi': s2_ndvi.rename({'band': 't'})
})

# Apply MOGPR
fused = mogpr(data)

# Export
fused['S2ndvi_FUSED'].rio.to_raster('NDVI_gapfilled.tif')
```

## 🎯 Which Workflow Should I Use?

| Use Case | Workflow | Documentation |
|----------|----------|---------------|
| Field validation, sample-based analysis | **Point-based MOGPR** | [SITS_TO_MOGPR_GUIDE.md](../SITS_TO_MOGPR_GUIDE.md#workflow-1-point-based-mogpr-recommended-for-validation) |
| Wall-to-wall mapping, spatial analysis | **Raster-based MOGPR** | [SITS_TO_MOGPR_GUIDE.md](../SITS_TO_MOGPR_GUIDE.md#workflow-2-raster-based-mogpr-for-spatial-mapping) |
| General sits integration (Whittaker, phenology) | **sits + FuseTS** | [SITS_FUSETS_WORKFLOW.md](../SITS_FUSETS_WORKFLOW.md) |
| Just smoothing, no fusion | **Whittaker only** | [SITS_FUSETS_WORKFLOW.md](../SITS_FUSETS_WORKFLOW.md#workflow-1-point-based-time-series-analysis) |

## 🔧 Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| "Dimension 't' not found" | Wrong dimension name | Rename: `ds.rename({'time': 't'})` |
| "Variable 'S2ndvi' not found" | Wrong variable name | Use exact names: 'VV', 'VH', 'S2ndvi' |
| "GPy import error" | Missing dependency | `pip install GPy` |
| Memory error | Large raster | Use chunking (see guide) |

**Full troubleshooting section**: [SITS_TO_MOGPR_GUIDE.md#troubleshooting](../SITS_TO_MOGPR_GUIDE.md#troubleshooting)

## 📖 Key Concepts

### MOGPR Data Requirements

✅ **Required**:
- Dimension named **'t'** (not 'time', 'band', 'date')
- Variables: **'VV', 'VH', 'S2ndvi'** (exact naming)
- Datetime coordinates for time dimension
- xarray.Dataset format (not DataArray)

✅ **Optional but recommended**:
- `include_uncertainties=True` to get std dev estimates
- `include_raw_inputs=True` to compare before/after
- `prediction_period='P5D'` for 5-day output composites

### Output Variable Naming

After MOGPR fusion:
- `S2ndvi_FUSED` → Gap-filled NDVI
- `S2ndvi_STD` → Uncertainty estimates
- `S2ndvi_RAW` → Original input (if `include_raw_inputs=True`)
- Same pattern for `VV_FUSED`, `VH_FUSED`

## 📊 Example Results

### Before MOGPR (Cloudy Season)
```
Time:  Jan  Feb  Mar  Apr  May
NDVI:  0.7  NaN  NaN  0.6  0.8   ← 50% missing due to clouds
VV:   -12  -13  -11  -12  -13   ← Complete coverage
```

### After MOGPR Fusion
```
Time:  Jan  Feb  Mar  Apr  May
NDVI:  0.7  0.65 0.62 0.6  0.8   ← Gaps filled using VV correlation
Std:   0.02 0.08 0.09 0.03 0.02  ← Higher uncertainty for filled values
```

## 🔗 Related Resources

- **FuseTS Main Docs**: [README.md](../README.md)
- **MOGPR Paper**: [Pipia et al. (2019)](https://doi.org/10.1016/j.rse.2019.111452)
- **sits Documentation**: [e-sensing.github.io/sitsbook](https://e-sensing.github.io/sitsbook/)
- **Microsoft Planetary Computer**: [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com/)

## 💡 Tips

1. **Start small**: Test with a few points before processing full area
2. **Check data quality**: Visualize raw time series before fusion
3. **Use uncertainties**: High std dev indicates poor fusion quality
4. **Validate**: Compare fused NDVI with ground truth when available
5. **Memory management**: Use chunking for rasters > 1GB

---

**Need help?** See the complete guide: [SITS_TO_MOGPR_GUIDE.md](../SITS_TO_MOGPR_GUIDE.md)

**Last Updated**: 2025-11-15
