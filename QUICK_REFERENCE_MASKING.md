# Quick Reference: Paddy Field Masking

## What Changed?

**Notebook:** `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb`

### 3 New Cells Added:

1. **Cell 3** - Shapefile processing imports
2. **Cell 10** - Section header (markdown)
3. **Cell 11** - Masking implementation (code)

### 1 Cell Updated:

- **Cell 25** - Added comment in training section

---

## Quick Start

### 1. Run the notebook normally until Cell 11

```python
# Cell 8: Load GEE Assets (as before)
# Creates: combined_dataset

# Cell 11: NEW - Apply Masking (automatic)
# Updates: combined_dataset with paddy-only pixels
# Generates: paddy_mask_applied.png
```

### 2. Check the output

Look for these messages:
```
✅ Shapefile loaded successfully
   Features: 1043
   CRS: EPSG:4326

✅ Paddy mask created successfully!
📊 Mask Statistics:
   Total pixels:      599,003
   Paddy pixels:      XXX,XXX (XX.XX%)
   Non-paddy pixels:  XXX,XXX (XX.XX%)
```

### 3. Verify with the plot

Open `paddy_mask_applied.png` to see:
- Panel 1: Paddy field mask
- Panel 2: Masked VV backscatter
- Panel 3: Masked NDVI

---

## Key Features

### Automatic Integration
- ✅ No manual intervention needed
- ✅ Works with existing workflow
- ✅ All downstream analysis uses masked data automatically

### Robust Error Handling
- ✅ Checks if dataset exists
- ✅ Handles missing shapefile
- ✅ Reports detailed errors

### Comprehensive Statistics
- ✅ Pixel counts and percentages
- ✅ Per-variable masking stats
- ✅ Validation visualization

---

## Important Notes

### File Paths
```python
Shapefile: /home/unika_sianturi/work/FuseTS/data/klambu-glapan.shp
Output:    ./paddy_mask_applied.png
```

### Cell Execution Order
```
MUST run in order:
  Cell 2  → Import libraries
  Cell 3  → Import shapefile libraries (NEW)
  Cell 8  → Load GEE assets
  Cell 11 → Apply masking (NEW)
  Cell 12+ → Continue with analysis
```

### Accessing the Mask
```python
# The mask is stored in the dataset
paddy_mask = combined_dataset['paddy_mask']

# Use it in custom analysis
my_masked_data = my_data.where(paddy_mask)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Shapefile not found" | Check path in Cell 11 |
| "No dataset loaded" | Run Cell 8 first |
| Wrong mask coverage | Check validation plot, verify CRS |
| Import errors | Run Cell 3 to import required libraries |

---

## Files Created

1. **Modified notebook:** `S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb` (64 cells)
2. **Validation plot:** `paddy_mask_applied.png` (generated when Cell 11 runs)
3. **Documentation:**
   - `MASKING_CHANGES_SUMMARY.txt` - Detailed changes
   - `MASKING_INTEGRATION_GUIDE.md` - Complete guide
   - `QUICK_REFERENCE_MASKING.md` - This file

---

## Cell 11 Code Summary

```python
# Load shapefile
shapefile_path = '/home/unika_sianturi/work/FuseTS/data/klambu-glapan.shp'
paddy_gdf = gpd.read_file(shapefile_path)

# Reproject to match dataset
paddy_gdf = paddy_gdf.to_crs('EPSG:4326')

# Rasterize to create mask
paddy_mask_array = ~geometry_mask(
    paddy_gdf.geometry,
    out_shape=(ny, nx),
    transform=transform
)

# Apply to dataset
for var in ['VV', 'VH', 'S2ndvi']:
    combined_dataset[var] = combined_dataset[var].where(paddy_mask)

# Validate and visualize
plt.savefig('paddy_mask_applied.png')
```

---

**Ready to use!** Just run the notebook cells in order.
