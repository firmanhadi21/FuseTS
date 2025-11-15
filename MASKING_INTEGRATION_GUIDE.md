# Paddy Field Masking Integration Guide

## Overview

This document describes the paddy field masking functionality added to `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb`. The masking ensures that all analysis is restricted to paddy field areas only.

---

## Changes Summary

### 1. Added Imports (Cell 3)

**Location:** After "## 1. Import Required Libraries"

```python
# Additional imports for shapefile-based masking
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
```

### 2. New Section: "3.5. Load and Apply Paddy Field Mask" (Cells 10-11)

**Location:** Between "## 3. LOAD GEE ASSET" and "## 4. Visualize Input Data"

#### Markdown Header (Cell 10)
```markdown
## 3.5. Load and Apply Paddy Field Mask

This section loads the paddy field shapefile and applies it as a mask to the dataset,
ensuring that only paddy field pixels are analyzed in subsequent steps.
```

#### Implementation (Cell 11)

**Key Operations:**

1. **Load Shapefile**
   - Path: `/home/unika_sianturi/work/FuseTS/data/klambu-glapan.shp`
   - Features: 1043 polygons
   - CRS: EPSG:4326 (automatically reprojected if needed)

2. **Create Raster Mask**
   - Match `combined_dataset` spatial dimensions
   - Use `rasterio.features.geometry_mask()` for efficient rasterization
   - Create proper transform from dataset coordinates

3. **Apply Mask**
   - Apply to all variables: VV, VH, S2ndvi
   - Use `xarray.where()` to preserve NaN handling
   - Non-paddy pixels become NaN

4. **Statistics & Validation**
   - Print masking statistics
   - Generate 3-panel validation plot
   - Save mask as `paddy_mask` variable in dataset

### 3. Updated Training Section (Cell 25)

**Location:** In "TRAIN S1→NDVI MODEL ON FULL DATASET"

**Added comment before `valid_mask` creation:**
```python
# NOTE: Paddy field mask has already been applied in Section 3.5
# This valid_mask only filters out NaN/inf values from the remaining data
valid_mask = (
    ~np.isnan(VV_full) & ~np.isinf(VV_full) &
    ~np.isnan(VH_full) & ~np.isinf(VH_full) &
    ~np.isnan(RVI_full) & ~np.isinf(RVI_full) &
    ~np.isnan(NDVI_full) & ~np.isinf(NDVI_full)
)
```

---

## Workflow Integration

```
┌─────────────────────────────────────────────────────────────┐
│  NOTEBOOK WORKFLOW                                          │
└─────────────────────────────────────────────────────────────┘

Cell 1-2:   Import Libraries
              ↓
Cell 3:     Import Shapefile Processing Libraries ⭐ NEW
              ↓
Cell 4-8:   Load GEE Assets
              ↓ (creates combined_dataset)
              ↓
Cell 10-11: Load & Apply Paddy Field Mask ⭐ NEW
              ↓ (updates combined_dataset with mask)
              ↓
Cell 12:    Visualize Input Data
              ↓ (now shows masked data only)
              ↓
Cell 13-24: GPU Setup & Data Cleaning
              ↓
Cell 25+:   Training ⭐ UPDATED COMMENT
              ↓ (trains on paddy pixels only)
              ↓
Cell 26+:   Prediction & Analysis
              ↓ (all outputs are paddy-only)
              ↓
Cell 30+:   Phenology Extraction
              ↓
Cell 40+:   Multi-Season Detection & Export
```

---

## Technical Details

### Masking Method

```python
# 1. Load shapefile and reproject to EPSG:4326
paddy_gdf = gpd.read_file(shapefile_path)
if paddy_gdf.crs != 'EPSG:4326':
    paddy_gdf = paddy_gdf.to_crs('EPSG:4326')

# 2. Create rasterization transform
transform = from_bounds(
    west=x_coords[0] - abs(x_res)/2,
    south=y_coords[-1] - abs(y_res)/2,
    east=x_coords[-1] + abs(x_res)/2,
    north=y_coords[0] + abs(y_res)/2,
    width=nx, height=ny
)

# 3. Rasterize polygons to mask
from rasterio.features import geometry_mask
paddy_mask_array = ~geometry_mask(
    paddy_gdf.geometry,
    out_shape=(ny, nx),
    transform=transform,
    invert=False
)

# 4. Create xarray DataArray
paddy_mask = xr.DataArray(
    paddy_mask_array,
    dims=['y', 'x'],
    coords={'y': y_coords, 'x': x_coords},
    name='paddy_mask'
)

# 5. Apply to dataset
for var in ['VV', 'VH', 'S2ndvi']:
    combined_dataset[var] = combined_dataset[var].where(paddy_mask)
```

### CRS Handling

- **Shapefile CRS:** Automatically detected and reprojected to EPSG:4326
- **Dataset CRS:** EPSG:4326 (Google Earth Engine standard)
- **Resolution:** Calculated from dataset x/y coordinates (~50m)

### Data Dimensions

```
combined_dataset:
  Dimensions:
    t: 62 (time periods, 12-day composites)
    y: 671 (rows)
    x: 893 (columns)

  Total pixels per time step: 671 × 893 = 599,003
  Paddy pixels: ~X (depends on shapefile coverage)
  Non-paddy pixels: Set to NaN
```

---

## Outputs

### Statistics Printed

```
📊 Mask Statistics:
   Total pixels:      599,003
   Paddy pixels:      XXX,XXX (XX.XX%)
   Non-paddy pixels:  XXX,XXX (XX.XX%)

🎯 Applying mask to dataset variables...
   VV: XXX,XXX → XXX,XXX valid pixels (XX.X% retained)
   VH: XXX,XXX → XXX,XXX valid pixels (XX.X% retained)
   S2ndvi: XXX,XXX → XXX,XXX valid pixels (XX.X% retained)
```

### Generated Files

**paddy_mask_applied.png**
- 3-panel validation plot:
  1. Paddy field mask (binary)
  2. Mean VV backscatter (masked to paddy areas)
  3. Mean NDVI (masked to paddy areas)
- Resolution: 150 DPI
- Format: PNG with tight bounding box

---

## Usage Notes

### Running the Notebook

1. **Execute cells in order**
   - Cell 3 must run before cell 11
   - Cell 8 (Load GEE Asset) must complete before cell 11
   - Cell 11 must complete before any analysis

2. **Check shapefile path**
   - Default: `/home/unika_sianturi/work/FuseTS/data/klambu-glapan.shp`
   - Update path in Cell 11 if different location

3. **Verify masking**
   - Check the validation plot `paddy_mask_applied.png`
   - Verify statistics match expectations
   - Ensure paddy pixels align with expected regions

### Error Handling

The code includes comprehensive error handling:

```python
# Checks dataset existence
if 'combined_dataset' in locals() or 'gee_dataset' in locals():
    # ... masking code ...
else:
    print("⚠️  Warning: No dataset loaded yet.")
    print("   Please run the 'LOAD GEE ASSET' section first.")

# Handles missing shapefile
try:
    paddy_gdf = gpd.read_file(shapefile_path)
    # ... processing ...
except FileNotFoundError:
    print(f"❌ Error: Shapefile not found at {shapefile_path}")
except Exception as e:
    print(f"❌ Error applying mask: {e}")
    import traceback
    traceback.print_exc()
```

### Accessing the Mask Later

The mask is saved in the dataset for later use:

```python
# Access the mask
paddy_mask = combined_dataset['paddy_mask']

# Use in custom analysis
masked_data = my_variable.where(paddy_mask)

# Count paddy pixels
n_paddy = paddy_mask.sum().values
```

---

## Benefits

1. **Focused Analysis**
   - All computations restricted to relevant areas
   - Reduces processing time
   - Eliminates noise from non-agricultural areas

2. **Accurate Statistics**
   - Phenology metrics calculated only for paddy fields
   - Training data from relevant pixels only
   - Improved model accuracy

3. **Clean Visualizations**
   - Plots show only paddy field areas
   - Easier interpretation of results
   - Professional presentation

4. **Reproducibility**
   - Consistent spatial extent across analyses
   - Documented mask application
   - Traceable methodology

---

## Troubleshooting

### Issue: "Shapefile not found"
**Solution:** Verify path in Cell 11 matches actual location

### Issue: "Mask covers wrong areas"
**Solution:**
- Check shapefile CRS matches data
- Verify shapefile bounds overlap with dataset
- Inspect validation plot for alignment issues

### Issue: "No paddy pixels detected"
**Solution:**
- Check if shapefile and dataset overlap spatially
- Verify coordinate systems are compatible
- Ensure shapefile has valid geometries

### Issue: "Mask has too few/many pixels"
**Solution:**
- Check rasterization resolution matches dataset
- Verify transform calculation is correct
- Compare with expected coverage area

---

## Files Modified

- **Notebook:** `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb`
  - Added: 3 new cells (1 import, 1 markdown, 1 code)
  - Modified: 1 cell (training section comment)
  - Total cells: 64 (was 61)

- **Documentation:**
  - `MASKING_CHANGES_SUMMARY.txt` - Detailed change log
  - `MASKING_INTEGRATION_GUIDE.md` - This guide

---

## Contact & Support

For questions or issues with the masking functionality:
1. Check the validation plot (`paddy_mask_applied.png`)
2. Review printed statistics in notebook output
3. Verify shapefile path and CRS
4. Examine error messages and traceback

---

**Last Updated:** 2025-11-08
**Notebook Version:** 64 cells (with paddy masking)
