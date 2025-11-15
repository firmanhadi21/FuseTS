# Fusion Notebook Analysis: GEE to MPC Conversion Guide

## Overview
This document summarizes the key differences between the GEE-based fusion notebook and the required changes for MPC data loading, based on analysis of three notebooks:
- `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb` (original GEE version)
- `S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb` (current MPC copy)
- `MPC_Data_Prep_Fixed.ipynb` (MPC data preparation)

---

## 1. CURRENT DATA LOADING APPROACH (GEE)

### Configuration (Cell 8)
```python
USE_GEE_ASSETS = True
ASSET_BASE_PATH = 'projects/ee-geodeticengineeringundip/assets/FuseTS'
ASSET_NAME_PREFIX = 'S1_S2_Nov2023_Oct2025_Period_'
NUM_PERIODS = 62
OUTPUT_LOCAL_DIR = 'gee_assets_download'
SCALE = 50  # meters
```

### Data Loading Process
1. **Authentication**: Initialize Google Earth Engine with project credentials
2. **Asset Discovery**: Verify individual asset images exist (Period_01, Period_02, etc.)
3. **Download**: Download each period as individual GeoTIFF files via geemap
   - Uses `geemap.download_ee_image()` for each period
   - Downloads to `gee_assets_download/period_XX.tif`
4. **Data Parsing**: 
   - Open each GeoTIFF with rioxarray
   - Extract 3 bands: VV (band 0), VH (band 1), S2ndvi (band 2)
   - Stack into numpy arrays along time axis

### Dataset Creation
```python
ds = xr.Dataset({
    'VV': (['t', 'y', 'x'], vv_stack),
    'VH': (['t', 'y', 'x'], vh_stack),
    'S2ndvi': (['t', 'y', 'x'], ndvi_stack)
}, coords={
    't': time_coords,      # datetime objects
    'y': y_coords,         # lat/northing values
    'x': x_coords          # lon/easting values
})
```

### Time Coordinate Generation
```python
start_date = datetime(2024, 11, 1)
period_center = start_date + timedelta(days=(period-1)*12 + 6)  # Center of each 12-day period
```

---

## 2. MPC DATA FORMAT AND STRUCTURE

### Data Source
Microsoft Planetary Computer (MPC) provides Sentinel-1 and Sentinel-2 data via STAC API
- **Collection**: `sentinel-2-l2a` and `sentinel-1-rtc`
- **Access**: Via pystac_client + rioxarray
- **Format**: Individual scenes, combined into 12-day composites

### MPC Data Preparation Output (from MPC_Data_Prep_Fixed.ipynb)

**Output Location**: `mpc_data/` directory
**File Format**: NetCDF (.nc)
**File Naming**: `klambu_glapan_2024-11-01_2025-11-07_final.nc` (or test version)

**Data Structure**:
```python
xr.Dataset with:
  Dimensions: (t: N_periods, y: N_pixels_y, x: N_pixels_x)
  
  Data Variables:
    - VV     (t, y, x): float32 - Sentinel-1 VV band
    - VH     (t, y, x): float32 - Sentinel-1 VH band
    - S2ndvi (t, y, x): float32 - Sentinel-2 NDVI
  
  Coordinates:
    - t (time): datetime64[ns] - Time stamps (center of 12-day periods)
    - y: float64 - Northing/latitude (UTM or WGS84)
    - x: float64 - Easting/longitude (UTM or WGS84)
  
  Attributes:
    - temporal_resolution: '12-day composites'
    - spatial_resolution: '50m'
    - crs: 'EPSG:32749' (UTM Zone 49S) or 'EPSG:4326'
    - date_range: 'YYYY-MM-DD to YYYY-MM-DD'
    - region: 'Kabupaten Demak'
    - fusets_ready: True
```

### Key Differences in MPC Data
1. **CRS Handling**: MPC data comes in UTM projection (EPSG:32749), not WGS84
   - Geometry must be reprojected to match data CRS before clipping
   - Use `rio.clip(geometry)` instead of `rio.clip_box(bbox)`
2. **Data Source**: Already processed and composited (not raw individual scenes)
3. **File Format**: NetCDF instead of GeoTIFF (already organized time series)
4. **No GEE Required**: Pure open-source workflow via planetary-computer + STAC

---

## 3. REQUIRED CHANGES FOR MPC DATA LOADING

### 3.1 Imports Change
**GEE Version**:
```python
import ee
import geemap
```

**MPC Version** (already exists in prep notebook):
```python
import planetary_computer
import pystac_client
import rioxarray
import xarray as xr
import geopandas as gpd
from pathlib import Path
```

### 3.2 Configuration Section Changes
**Current (Cell 8)**:
```python
USE_GEE_ASSETS = True
ASSET_BASE_PATH = 'projects/ee-geodeticengineeringundip/assets/FuseTS'
ASSET_NAME_PREFIX = 'S1_S2_Nov2023_Oct2025_Period_'
NUM_PERIODS = 62
OUTPUT_LOCAL_DIR = 'gee_assets_download'
```

**New (for MPC)**:
```python
# Point to MPC data output from MPC_Data_Prep_Fixed.ipynb
USE_MPC_DATA = True
MPC_DATA_DIR = Path('mpc_data')
INPUT_FILENAME = 'klambu_glapan_2024-11-01_2025-11-07_final.nc'
# OR test version:
# INPUT_FILENAME = 'test_timeseries.nc'
```

### 3.3 Data Loading Function Replacement
**Old Function** (Cell 8):
```python
def load_gee_assets_to_xarray(asset_base_path, name_prefix, num_periods=62, region=None, scale=50):
    # Complex GEE authentication
    # Download loop for each period
    # Parse individual GeoTIFFs
    # Stack into xarray
```

**New Function**:
```python
def load_mpc_data_from_netcdf(mpc_dir, filename):
    """
    Load pre-processed MPC S1/S2 data from NetCDF file
    
    Parameters:
    -----------
    mpc_dir : Path
        Directory containing NetCDF file (default: 'mpc_data/')
    filename : str
        NetCDF filename (e.g., 'klambu_glapan_2024-11-01_2025-11-07_final.nc')
    
    Returns:
    --------
    xr.Dataset : FuseTS-ready dataset with (t, y, x) structure
    """
    import xarray as xr
    from pathlib import Path
    
    filepath = Path(mpc_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")
    
    print(f"📖 Loading MPC data from: {filepath}")
    
    # Load NetCDF
    ds = xr.open_dataset(filepath)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {ds['VV'].shape}")
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Time range: {ds.coords['t'].values[0]} to {ds.coords['t'].values[-1]}")
    print(f"   CRS: {ds.attrs.get('crs', 'Unknown')}")
    print(f"   Region: {ds.attrs.get('region', 'Unknown')}")
    
    # Verify required variables exist
    required_vars = ['VV', 'VH', 'S2ndvi']
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")
    
    # Ensure time dimension is named 't'
    if 'time' in ds.dims and 't' not in ds.dims:
        ds = ds.rename({'time': 't'})
    
    # Add metadata if not present
    if 'fusets_ready' not in ds.attrs:
        ds.attrs['fusets_ready'] = True
    
    return ds
```

### 3.4 Initialization and Execution Change
**Current (Cell 8, bottom)**:
```python
if USE_GEE_ASSETS:
    print("🌍 LOADING DATA FROM GEE ASSETS")
    initialize_gee()
    
    gee_dataset = load_gee_assets_to_xarray(
        ASSET_BASE_PATH,
        ASSET_NAME_PREFIX,
        num_periods=NUM_PERIODS,
        region=REGION,
        scale=SCALE
    )
```

**New**:
```python
if USE_MPC_DATA:
    print("🌍 LOADING DATA FROM MPC (NetCDF)")
    
    try:
        mpc_dataset = load_mpc_data_from_netcdf(
            MPC_DATA_DIR,
            INPUT_FILENAME
        )
        
        if mpc_dataset is not None:
            print("\n" + "="*70)
            print("📊 DATASET SUMMARY")
            print("="*70)
            print(mpc_dataset)
            print("\n✅ Data ready for MOGPR processing!")
            print("   Use 'mpc_dataset' for fusion analysis")
        else:
            print("❌ Failed to load MPC data")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  MPC data file not found!")
        print("   Please run MPC_Data_Prep_Fixed.ipynb first to generate the NetCDF file")
        mpc_dataset = None
```

### 3.5 Variable Naming Changes Throughout Notebook
The rest of the notebook refers to loaded data as `gee_dataset` - this needs updating:

**Search & Replace**:
- `gee_dataset` → `mpc_dataset`
- `combined_dataset = gee_dataset` → `combined_dataset = mpc_dataset`
- Remove all GEE-specific code blocks (e.g., GEE Asset verification cells)

---

## 4. SPECIFIC CELL CHANGES SUMMARY

| Cell | Current Purpose | MPC Changes |
|------|-----------------|------------|
| Cell 3 | GEE data acquisition intro | Update to MPC intro |
| Cell 4 | Prerequisites for GEE exports | Replace with "Run MPC_Data_Prep_Fixed.ipynb first" |
| Cell 5 | GEE asset check dialog | Remove (not needed for NetCDF) |
| Cell 6 | GEE authentication check | Remove entirely |
| Cell 7 | Header for data loading | Keep, update text |
| Cell 8 | Load GEE assets | **MAJOR CHANGE**: Replace with NetCDF loading function |
| Cell 9+ | All downstream cells | Change `gee_dataset` → `mpc_dataset` |

---

## 5. DATA FLOW COMPARISON

### GEE Workflow
```
GEE Earth Engine (Cloud)
       ↓
Individual Period Assets (S1/S2)
       ↓
Download via geemap (62 GeoTIFFs)
       ↓
Parse each GeoTIFF
       ↓
Stack into xarray (this notebook)
       ↓
xarray Dataset for MOGPR
```

### MPC Workflow
```
Microsoft Planetary Computer (STAC)
       ↓
Sentinel-1 & Sentinel-2 scenes
       ↓
MPC_Data_Prep_Fixed.ipynb (separate workflow)
       ↓
Create 12-day composites
       ↓
Save to NetCDF
       ↓
Load NetCDF (this notebook) ← SINGLE FUNCTION CALL
       ↓
xarray Dataset for MOGPR
```

---

## 6. BACKWARD COMPATIBILITY NOTE

The MPC approach is actually **simpler** because:
1. No GEE authentication required
2. Data is pre-composited (faster)
3. NetCDF is already structured for time series
4. Single file load vs. 62 individual downloads
5. No CRS confusion (handled in prep notebook)

The downstream MOGPR and deep learning processing remains **identical** because both approaches produce the same xarray Dataset structure.

---

## 7. ERROR HANDLING

### Common MPC Issues
```python
# File not found
FileNotFoundError: NetCDF file not found: mpc_data/klambu_glapan_2024-11-01_2025-11-07_final.nc
→ Run MPC_Data_Prep_Fixed.ipynb first

# Missing variables
ValueError: Missing required variables: ['S2ndvi']
→ Check that prep notebook completed all periods successfully

# Dimension mismatch
→ Ensure dataset has dimensions (t, y, x) not (time, y, x)
   Use: ds = ds.rename({'time': 't'}) if needed
```

---

## 8. TEST PLAN FOR CONVERSION

1. **Verify MPC data exists**:
   ```bash
   ls -lh mpc_data/*.nc
   ```

2. **Test loading function**:
   ```python
   test_ds = load_mpc_data_from_netcdf(Path('mpc_data'), 'test_timeseries.nc')
   print(test_ds)
   ```

3. **Verify dataset structure**:
   ```python
   assert 'VV' in test_ds.data_vars
   assert 'VH' in test_ds.data_vars
   assert 'S2ndvi' in test_ds.data_vars
   assert test_ds['VV'].dims == ('t', 'y', 'x')
   ```

4. **Run MOGPR section** (should work with no other changes):
   ```python
   from fusets.mogpr import MOGPRTransformer
   mogpr = MOGPRTransformer()
   result = mogpr.fit_transform(test_ds)
   ```

