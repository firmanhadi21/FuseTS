# Separate R + Python Workflow for sits + FuseTS

This guide shows the **recommended workflow** for using sits (R) and FuseTS (Python) **without** needing R-Python integration tools like `reticulate` or `rpy2`.

## Why Separate Workflows?

**Advantages**:
- ✅ No complex R-Python integration setup
- ✅ Use native tools (RStudio for R, Jupyter for Python)
- ✅ Easier debugging
- ✅ No conda/rpy2 version conflicts
- ✅ More stable and reproducible

**Simple principle**: Use **files** (CSV, GeoTIFF) to bridge R and Python

---

## Setup

### 1. Install R Separately

```bash
# Option A: System R (recommended)
sudo apt-get update
sudo apt-get install r-base r-base-dev

# Option B: Conda R (in separate environment)
conda create -n r_sits r-base
conda activate r_sits
```

### 2. Install sits in R

Open R or RStudio:

```r
# Install sits and dependencies
install.packages("sits")
install.packages(c("sf", "terra", "dplyr", "tidyr"))

# Test installation
library(sits)
packageVersion("sits")
```

### 3. Install FuseTS in Python

```bash
# In your Python environment
cd /home/unika_sianturi/work/FuseTS
pip install -e .
pip install rioxarray matplotlib pandas geopandas
```

**Note**: You do **NOT** need `pysits` or `rpy2`!

---

## Workflow Step-by-Step

### PART 1: Data Extraction in R

#### Option 1: Use RStudio (Easiest)

1. Open RStudio
2. Open script: `/home/unika_sianturi/work/FuseTS/scripts/extract_sits_data.R`
3. Modify study area and dates
4. Run the script
5. Data exported to `/home/unika_sianturi/work/FuseTS/data/sits_exports/`

#### Option 2: Run R Script from Terminal

```bash
# Make script executable
chmod +x /home/unika_sianturi/work/FuseTS/scripts/extract_sits_data.R

# Run in R
R --vanilla < /home/unika_sianturi/work/FuseTS/scripts/extract_sits_data.R
```

#### Option 3: Interactive R Session

```bash
R

# In R:
source("/home/unika_sianturi/work/FuseTS/scripts/extract_sits_data.R")
```

**Output**:
- Point-based: `s1_timeseries.csv`, `s2_ndvi_timeseries.csv`
- Raster-based: `s1_rasters/VV_stack.tif`, `s2_rasters/NDVI_stack.tif`

---

### PART 2: Processing in Python

#### Open Jupyter Notebook

```bash
cd /home/unika_sianturi/work/FuseTS/notebooks
jupyter notebook Paddyfield_Phenology_S1_S2_Fusion.ipynb
```

#### Update File Paths

In the notebook, update these paths to match your R exports:

```python
# For point-based workflow
s1_data = load_sits_csv(
    "/home/unika_sianturi/work/FuseTS/data/sits_exports/s1_timeseries.csv",
    time_col='Index',
    band_cols=['VV', 'VH']
)

s2_data = load_sits_csv(
    "/home/unika_sianturi/work/FuseTS/data/sits_exports/s2_ndvi_timeseries.csv",
    time_col='Index',
    band_cols=['NDVI']
)
```

Or for raster-based:

```python
# For raster-based workflow
s1_vv_stack = load_sits_geotiff("/home/unika_sianturi/work/FuseTS/data/sits_exports/s1_rasters/VV_stack.tif")
s1_vh_stack = load_sits_geotiff("/home/unika_sianturi/work/FuseTS/data/sits_exports/s1_rasters/VH_stack.tif")
s2_ndvi_stack = load_sits_geotiff("/home/unika_sianturi/work/FuseTS/data/sits_exports/s2_rasters/NDVI_stack.tif")
```

#### Run Notebook Cells

Execute cells in order:
1. **Load data** → Check data quality
2. **Apply MOGPR fusion** → Gap-filled NDVI
3. **Extract phenology** → SOS, EOS, LOS
4. **Visualize results** → Plots and maps
5. **Export outputs** → CSV and GeoTIFF

**Output**:
- `phenology_mogpr_vs_s2only_comparison.csv`
- `paddyfield_phenology_mogpr_fused.csv`
- Phenology maps (if raster workflow)

---

## Complete Example

### Example 1: Quick Test with Point Data

**Step 1: R Code** (save as `test_extract.R`)

```r
library(sits)
library(sf)
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")

# Small test area
bbox <- st_bbox(c(xmin = 110.6, ymin = -6.9,
                  xmax = 110.7, ymax = -6.8), crs = 4326)
study_area <- st_as_sfc(bbox) %>% st_as_sf()

# Get S2 data
s2_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-06-30",
  bands = c("B04", "B08")
)

s2_ndvi <- sits_apply(s2_cube, NDVI = (B08 - B04) / (B08 + B04))

# Get S1 data
s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = study_area,
  start_date = "2024-01-01",
  end_date = "2024-06-30",
  bands = c("VV", "VH")
)

# Sample 50 points
points <- st_sample(study_area, size = 50)
s1_ts <- sits_get_data(s1_cube, samples = points)
s2_ts <- sits_get_data(s2_ndvi, samples = points)

# Export
sits_to_fusets_csv(s1_ts, "s1_test.csv", bands = c("VV", "VH"))
sits_to_fusets_csv(s2_ts, "s2_test.csv", bands = c("NDVI"))

cat("✓ Test data exported!\n")
```

Run in R:
```r
source("test_extract.R")
```

**Step 2: Python Code** (in Jupyter)

```python
from fusets.io.sits_bridge import load_sits_csv
from fusets.mogpr import MOGPRTransformer
from fusets.analytics import phenology
import xarray as xr

# Load data
s1_data = load_sits_csv("s1_test.csv", time_col='Index', band_cols=['VV', 'VH'])
s2_data = load_sits_csv("s2_test.csv", time_col='Index', band_cols=['NDVI'])

# Combine for MOGPR
combined = xr.Dataset({
    'VV': s1_data['VV'],
    'VH': s1_data['VH'],
    'S2ndvi': s2_data['NDVI']
})

# Apply MOGPR fusion
print("Applying MOGPR fusion...")
mogpr = MOGPRTransformer()
fused = mogpr.fit_transform(combined)

# Extract phenology
print("Extracting phenology...")
pheno = phenology(fused['S2ndvi'], detection_method='seasonal_amplitude')

print("✓ Done!")
print(f"SOS (day of year): {pheno.da_sos_times.values}")
print(f"EOS (day of year): {pheno.da_eos_times.values}")
```

---

## Troubleshooting

### Issue: "sits package not found" in R

**Solution**:
```r
install.packages("sits")
library(sits)
```

### Issue: "Cannot import fusets" in Python

**Solution**:
```bash
cd /home/unika_sianturi/work/FuseTS
pip install -e .
```

### Issue: "File not found" when loading in Python

**Solution**: Check that R export paths match Python load paths
```python
import os
os.path.exists("/home/unika_sianturi/work/FuseTS/data/sits_exports/s1_timeseries.csv")
```

### Issue: sits_to_fusets_csv function not found

**Solution**: Source the helper script in R
```r
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")
```

---

## File Transfer Formats

### CSV (Point Time Series)

**Pros**:
- Small file size
- Human-readable
- Easy to inspect

**Cons**:
- No spatial reference
- Point data only

**Use for**: Sample-based analysis, validation

### GeoTIFF (Raster Stacks)

**Pros**:
- Preserves spatial reference
- Full spatial coverage
- Standard GIS format

**Cons**:
- Large file size
- Slower processing

**Use for**: Spatial phenology mapping, area statistics

---

## Best Practices

1. **Keep R and Python separate** - Don't try to mix them in one session
2. **Use descriptive filenames** - Include dates, sensors, processing level
3. **Document parameters** - Save sits extraction parameters for reproducibility
4. **Validate exports** - Check CSV/GeoTIFF files before moving to Python
5. **Use version control** - Track both R scripts and Python notebooks

---

## Summary

**R Session** (RStudio or terminal):
```r
library(sits)
# Extract S1 + S2 data
# Export to CSV or GeoTIFF
```

**File Transfer**:
```
CSV or GeoTIFF files bridge R → Python
```

**Python Session** (Jupyter):
```python
from fusets.io.sits_bridge import load_sits_csv
# Load exported data
# Apply MOGPR fusion
# Extract phenology
```

**No `pysits`, `reticulate`, or `rpy2` needed!**

---

## Quick Reference Commands

### In R:
```r
# Install sits
install.packages("sits")

# Run extraction script
source("/home/unika_sianturi/work/FuseTS/scripts/extract_sits_data.R")
```

### In Python:
```bash
# Open notebook
jupyter notebook /home/unika_sianturi/work/FuseTS/notebooks/Paddyfield_Phenology_S1_S2_Fusion.ipynb
```

---

For questions or issues, see:
- sits documentation: https://e-sensing.github.io/sitsbook/
- FuseTS documentation: `/home/unika_sianturi/work/FuseTS/README.md`
