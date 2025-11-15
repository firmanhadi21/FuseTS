# Quick Reference: Converting Fusion Notebook from GEE to MPC

## TL;DR - What You Need to Do

1. **Run MPC_Data_Prep_Fixed.ipynb first** to generate NetCDF files
2. **Replace Cell 8** in S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb with MPC loading code
3. **Change variable names**: `gee_dataset` → `mpc_dataset` (rest of notebook)
4. **Everything else stays the same** (MOGPR, deep learning sections unchanged)

---

## Side-by-Side Comparison

### Current (GEE) Workflow
```
Cell 3-6:  GEE Setup & Auth
   ↓
Cell 8:    Download 62 individual GeoTIFFs from GEE Assets
   ↓
Cell 8:    Parse & stack into xarray
   ↓
Cell 9+:   MOGPR & Deep Learning
```

**Time**: ~1-2 hours (download + processing)
**Dependencies**: Google Earth Engine account, geemap, ee library
**Data**: Individual GeoTIFFs for each period

### New (MPC) Workflow
```
Run MPC_Data_Prep_Fixed.ipynb in separate notebook
   ↓
Generates: mpc_data/*.nc files (pre-composited, properly formatted)
   ↓
Cell 8:    Load single NetCDF file (seconds)
   ↓
Cell 9+:   MOGPR & Deep Learning (identical code)
```

**Time**: ~30 seconds to load
**Dependencies**: xarray, pathlib (already have these)
**Data**: Single NetCDF file (already formatted for FuseTS)

---

## File Locations & Naming

### MPC Data Prep Output
```
mpc_data/
├── test_timeseries.nc                                    # Quick test (3 periods)
├── klambu_glapan_2024-11-01_2025-11-07_final.nc         # Full data (31 periods)
└── test_preview.png                                      # Preview image
```

### What's Inside the NetCDF
```python
ds = xr.open_dataset('mpc_data/test_timeseries.nc')

# Dimensions: (t: 3, y: 836, x: 424)
# Data Variables: VV, VH, S2ndvi (float32)
# Coordinates: t (datetime64), y (float), x (float)
# Attributes: crs, spatial_resolution, temporal_resolution, etc.
```

---

## Exact Changes to Make

### Change 1: Update Cell 3 (Markdown)
**OLD**:
```markdown
## 2. Data Acquisition and Loading from GEE

For this tutorial, we'll create synthetic S1 and S2 time series data. 
In practice, you would load your actual GeoTIFF stacks or data from other sources.
```

**NEW**:
```markdown
## 2. Data Acquisition and Loading from MPC

We'll load pre-processed S1/S2 data from Microsoft Planetary Computer.
First, ensure you've run **MPC_Data_Prep_Fixed.ipynb** to generate the NetCDF file.
```

### Change 2: Update Cell 4 (Markdown)
**OLD**:
```markdown
### 📋 Prerequisites: Export Data to GEE Assets First

Before running this notebook, you must:
1. Open `GEE_Data_Preparation_for_FuseTS_Assets.ipynb`
...
```

**NEW**:
```markdown
### 📋 Prerequisites: Generate MPC Data First

Before running this notebook:
1. Open **MPC_Data_Prep_Fixed.ipynb** in a new tab
2. Run all cells to download & process Sentinel-1/2 data from Microsoft Planetary Computer
3. This generates: `mpc_data/test_timeseries.nc` or `mpc_data/klambu_glapan_*.nc`
4. Return to this notebook once data is ready
```

### Change 3: Delete Cells 5 & 6
These contain GEE-specific asset verification code. Remove entirely.

### Change 4: Replace Cell 7 (Markdown)
**OLD**:
```markdown
## 3. LOAD GEE ASSET
```

**NEW**:
```markdown
## 3. LOAD MPC DATA FROM NETCDF
```

### Change 5: COMPLETELY REPLACE Cell 8
Delete all GEE loading code. Replace with:

```python
import xarray as xr
from pathlib import Path

# Configuration
USE_MPC_DATA = True
MPC_DATA_DIR = Path('mpc_data')
INPUT_FILENAME = 'test_timeseries.nc'  # or 'klambu_glapan_2024-11-01_2025-11-07_final.nc'

def load_mpc_data_from_netcdf(mpc_dir, filename):
    """Load pre-processed MPC S1/S2 NetCDF file"""
    filepath = Path(mpc_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}\nRun MPC_Data_Prep_Fixed.ipynb first!")
    
    print(f"Loading MPC data from: {filepath}")
    ds = xr.open_dataset(filepath)
    
    # Ensure time dimension named 't'
    if 'time' in ds.dims:
        ds = ds.rename({'time': 't'})
    
    # Verify variables
    required = ['VV', 'VH', 'S2ndvi']
    if not all(v in ds.data_vars for v in required):
        raise ValueError(f"Missing variables: expected {required}, got {list(ds.data_vars)}")
    
    print(f"✅ Loaded: {ds['VV'].shape} (time, y, x)")
    return ds

if USE_MPC_DATA:
    try:
        mpc_dataset = load_mpc_data_from_netcdf(MPC_DATA_DIR, INPUT_FILENAME)
        print("\nDataset ready for MOGPR:")
        print(mpc_dataset)
    except Exception as e:
        print(f"Error: {e}")
        mpc_dataset = None
```

### Change 6: Update Cell 10 onwards (Search & Replace)
Find all instances of `gee_dataset` and replace with `mpc_dataset`:

```bash
# In Jupyter: Ctrl+H (Find & Replace)
# Find: gee_dataset
# Replace: mpc_dataset
# Replace All
```

Also remove:
- `initialize_gee()` calls
- Any `ee.` references
- Any `geemap.` references

### Change 7: Update Cell 14 (Data validation)
**OLD**:
```python
# ⚠️ URGENT: Check if combined_dataset exists and has data
if gee_dataset is None:
    print("⚠️ GEE Assets not loaded. Using synthetic data...")
    combined_dataset = create_synthetic_data()
else:
    combined_dataset = gee_dataset
```

**NEW**:
```python
# Check if MPC data exists and has data
if mpc_dataset is None:
    print("⚠️ MPC data not loaded!")
    print("Please run MPC_Data_Prep_Fixed.ipynb first")
else:
    combined_dataset = mpc_dataset
    print(f"Using MPC data: {combined_dataset['VV'].shape}")
```

---

## Data Validation Checklist

After loading, verify the data:

```python
# Run this to validate
assert mpc_dataset is not None, "Dataset not loaded"
assert mpc_dataset['VV'].dims == ('t', 'y', 'x'), f"Wrong dims: {mpc_dataset['VV'].dims}"
assert mpc_dataset['VV'].shape[0] > 0, "No time steps"
assert 'S2ndvi' in mpc_dataset, "Missing NDVI"

print("✅ All checks passed!")
print(f"Dataset shape: {mpc_dataset['VV'].shape}")
print(f"Time range: {mpc_dataset['t'].values[0]} to {mpc_dataset['t'].values[-1]}")
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| FileNotFoundError: no such file | MPC prep notebook not run | Run MPC_Data_Prep_Fixed.ipynb first |
| ValueError: Missing variables | Wrong file or incomplete export | Check file exists: `ls mpc_data/*.nc` |
| Wrong dimensions (time, y, x) | Forgot rename | Add: `ds = ds.rename({'time': 't'})` |
| Data all NaN | Download failed | Re-run MPC prep, check internet |
| Shape mismatch with MOGPR | CRS/extent issue | Check CRS in attributes: `ds.attrs['crs']` |

---

## Testing Workflow

### Step 1: Generate test data
```python
# In MPC_Data_Prep_Fixed.ipynb
periods[:3]  # Process only first 3 periods
# Output: mpc_data/test_timeseries.nc (~small file)
```

### Step 2: Load and test
```python
# In fusion notebook
INPUT_FILENAME = 'test_timeseries.nc'
mpc_dataset = load_mpc_data_from_netcdf(MPC_DATA_DIR, INPUT_FILENAME)
print(mpc_dataset)  # Should show 3 time periods
```

### Step 3: Test MOGPR
```python
# In fusion notebook, MOGPR section
from fusets.mogpr import MOGPRTransformer
mogpr = MOGPRTransformer()
result = mogpr.fit_transform(mpc_dataset)
print(result)
```

### Step 4: Full data (after test works)
```python
# In MPC_Data_Prep_Fixed.ipynb
# Remove [:3] slicing, process all periods
# Output: mpc_data/klambu_glapan_2024-11-01_2025-11-07_final.nc

# In fusion notebook
INPUT_FILENAME = 'klambu_glapan_2024-11-01_2025-11-07_final.nc'
mpc_dataset = load_mpc_data_from_netcdf(MPC_DATA_DIR, INPUT_FILENAME)
# Continue with full analysis
```

---

## Performance Comparison

| Operation | GEE | MPC |
|-----------|-----|-----|
| Download 3 periods | ~5-10 min | Done (pre-downloaded) |
| Parse & stack | ~2-3 min | <1 sec (already in NetCDF) |
| Load into memory | ~2 min | ~0.5 sec |
| Total time | ~10-15 min | ~0.5 sec |
| Dependencies | 4 external packages | Already have them |
| API calls | 62+ | 0 |

---

## Key Takeaway

The MPC approach is **much simpler** because:
- No cloud API (faster, more reliable)
- No authentication hassles
- Data is pre-formatted for time series
- Single file instead of 62 downloads
- Everything else in notebook stays exactly the same
