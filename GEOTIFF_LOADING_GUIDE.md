# 📁 Loading Individual GeoTIFF Files for MOGPR Fusion

## Overview

If you exported your Sentinel-1/2 data from GEE as **individual GeoTIFF files** (e.g., `period_01.tif`, `period_02.tif`, etc.) instead of GEE Image Assets, you need to use the simpler loading approach.

## Two Loading Options

### Option 1: Load from Local GeoTIFF Directory (Recommended)

**Use when:** You have downloaded GeoTIFF files locally (on HPC, laptop, or Colab)

**Function:** `load_geotiff_periods_to_xarray()`

**Usage:**
```python
# Specify directory containing your GeoTIFF files
GEOTIFF_DIR = '/path/to/your/geotiff/files'

# Load the data
combined_dataset = load_geotiff_periods_to_xarray(
    geotiff_dir=GEOTIFF_DIR,
    num_periods=31,
    file_pattern='period_{:02d}.tif'
)
```

**Expected file structure:**
```
/path/to/your/geotiff/files/
├── period_01.tif
├── period_02.tif
├── period_03.tif
...
└── period_31.tif
```

**Platform-specific paths:**
- **HPC:** `/home/username/data/demak_s1_s2/`
- **Local Mac:** `/Users/username/Downloads/demak_geotiffs/`
- **Local Windows:** `C:/Users/username/Downloads/demak_geotiffs/`
- **Google Colab:** `/content/drive/MyDrive/FuseTS_Data/demak/`

---

### Option 2: Load from GEE Image Assets

**Use when:** Your data is stored as GEE Image Assets (not GeoTIFF files)

**Function:** `load_gee_assets_to_xarray()`

**Usage:**
```python
# Initialize GEE
initialize_gee()

# Load from GEE Assets
combined_dataset = load_gee_assets_to_xarray(
    asset_base_path='projects/ee-geodeticengineeringundip/assets/FuseTS',
    name_prefix='S1_S2_Nov2024_Oct2025_Period_',
    num_periods=31,
    region=None,
    scale=50
)
```

---

## How to Check Which Method You Need

### If you exported to **Google Drive** from GEE:
1. Check your Google Drive folder
2. Do you see individual `.tif` files (period_01.tif, etc.)? → **Use Option 1**
3. Did you import these into GEE Assets? → **Use Option 2**

### If you exported to **GEE Assets** directly:
1. Go to GEE Code Editor → Assets tab
2. Do you see Image assets (not GeoTIFF files)? → **Use Option 2**

---

## Workflow for GeoTIFF Files (Option 1)

### Step 1: Download GeoTIFFs from Google Drive
If your files are in Google Drive, download them to your working environment:

**On HPC:**
```bash
# Create directory
mkdir -p ~/data/demak_s1_s2

# Use rclone, gdown, or manual download
# Then upload to HPC
```

**On Google Colab:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Files will be accessible at:
# /content/drive/MyDrive/your_folder_name/
```

**On Local Computer:**
- Download ZIP from Google Drive
- Extract to a folder
- Note the absolute path

### Step 2: Update the Path in Notebook

Open `S1_S2_MOGPR_Fusion_Tutorial.ipynb` and find the cell:

```python
# OPTION 1: Load from local GeoTIFF directory
GEOTIFF_DIR = '/path/to/your/geotiff/files'  # ← CHANGE THIS
```

Replace with your actual path:
```python
GEOTIFF_DIR = '/home/yourusername/data/demak_s1_s2'
```

### Step 3: Run the Loading Cell

Execute the cell that calls `load_geotiff_periods_to_xarray()`:

```python
combined_dataset = load_geotiff_periods_to_xarray(
    geotiff_dir=GEOTIFF_DIR,
    num_periods=31,
    file_pattern='period_{:02d}.tif'
)
```

### Step 4: Verify the Data

You should see output like:
```
📁 Loading GeoTIFF files from: /home/username/data/demak_s1_s2
   Pattern: period_{:02d}.tif
   Expected periods: 31

📥 Loading 31 GeoTIFF files...
   Loaded 5/31 periods...
   Loaded 10/31 periods...
   ...
   Loaded 31/31 periods...

✅ Successfully loaded 31/31 periods

📊 Dataset Summary:
   Shape: (31, 360, 800) (time, y, x)
   Time range: 2024-11-07 to 2025-10-29
   Spatial extent: 360 x 800 pixels
   Bands: VV, VH, S2ndvi
   Total size: 214.3 MB
```

---

## Expected GeoTIFF Format

Each GeoTIFF file should have **3 bands** in this order:
1. **Band 1:** VV (Sentinel-1 VV polarization)
2. **Band 2:** VH (Sentinel-1 VH polarization)
3. **Band 3:** S2ndvi (Sentinel-2 NDVI)

**If your GeoTIFFs have different band order:**
- Modify the `load_geotiff_periods_to_xarray()` function
- Adjust the band indexing: `period_data.isel(band=X)`

**If you only have NDVI (1 band):**
- The function will automatically handle it
- VV and VH will be filled with NaN values

---

## Troubleshooting

### Error: "Directory not found"
```python
❌ Directory not found: /path/to/your/geotiff/files
```

**Solution:**
1. Check the path is correct (absolute path, not relative)
2. Verify the directory exists: `ls /path/to/your/geotiff/files`
3. On HPC: Use full path starting with `/home/`
4. On Colab: Make sure Drive is mounted

### Error: "No data loaded"
```python
❌ No data loaded!
💡 Make sure GeoTIFF files are in: /path/to/dir
   Expected files: period_01.tif, period_02.tif, ...
```

**Solution:**
1. Check file naming: Must match `period_01.tif`, `period_02.tif`, etc.
2. If different naming, update `file_pattern` parameter:
   ```python
   file_pattern='S1_S2_Period_{:02d}.tif'  # Example for different naming
   ```
3. Verify files exist: `ls /path/to/your/geotiff/files/`

### Warning: "Only 1 band found"
```python
⚠️  Period 01: Only 1 band found, treating as NDVI
```

**Impact:**
- NDVI will be loaded
- VV and VH will be NaN
- MOGPR fusion **won't work** (needs all 3 bands)

**Solution:**
- Re-export from GEE with all 3 bands (VV, VH, NDVI)
- Or export separate files and combine them

### File not found for some periods
```python
⚠️  Period 15: File not found: period_15.tif
```

**Impact:**
- Missing periods will be skipped
- Time series will have gaps
- MOGPR can still work but with reduced temporal coverage

**Solution:**
- Check which periods are missing
- Re-export missing periods from GEE
- Or adjust `num_periods` to match available files

---

## Performance Tips

### On HPC:
- Store GeoTIFFs on fast storage (SSD, not network drive)
- Use scratch space for temporary processing
- Typical load time: 10-30 seconds for 31 periods

### On Google Colab:
- Upload to Colab session storage (faster) instead of Drive (slower)
- Or use Drive for permanent storage and accept slower load times
- Typical load time: 1-3 minutes from Drive

### Memory Usage:
- Full Demak dataset (31 × 360 × 800 × 3 bands): ~214 MB
- Safe for HPC (160GB available) and Colab (12GB RAM)
- If out of memory: reduce spatial extent or use subset

---

## Next Steps After Loading

Once you've successfully loaded the data:

1. **Visualize the data** (Cell 23):
   ```python
   visualize_timeseries(combined_dataset)
   ```

2. **Check GPU availability** (Cell 24-25):
   ```python
   USE_GPU = True  # Set to True on HPC with H100
   ```

3. **Create a subset for testing** (Cell 26):
   ```python
   subset_data = combined_dataset.isel(x=slice(0, 50), y=slice(0, 50))
   ```

4. **Run MOGPR fusion** (Cell 27):
   ```python
   fused_result = mogpr_fusion_with_uncertainty(subset_data)
   ```

5. **Scale to full dataset** (after testing):
   ```python
   fused_result = mogpr_fusion_with_uncertainty(combined_dataset)
   ```

---

## Contact & Support

If you encounter issues:
1. Check this guide first
2. Verify file paths and naming
3. Test with a small subset (1-2 periods) first
4. Check the notebook cells are run in order
5. Ensure all required packages are installed (rioxarray, xarray, numpy)

**Required packages:**
```bash
pip install rioxarray xarray numpy scipy
```

On HPC with mogpr_h100 environment:
```bash
conda activate mogpr_h100
# All packages should already be installed
```
