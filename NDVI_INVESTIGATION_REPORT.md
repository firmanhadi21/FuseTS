# NDVI Data Loading Investigation Report
## S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb Analysis

### Executive Summary

The investigation reveals **NO SCALING ISSUE** with NDVI data. The S2ndvi values are correctly loaded and in the proper range [-1, 1]. The blank NDVI map in `paddy_mask_applied.png` is likely due to **visualization issues** rather than data loading problems.

---

## 1. How S2ndvi is Loaded from GEE Assets

### Source: GEE Export (GEE_Data_Preparation_for_FuseTS_Assets.ipynb)

**NDVI Calculation in GEE:**
```python
def calculate_ndvi_toa(image):
    # B8 = NIR, B4 = Red (same as Level-2A)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Load Level-1C TOA data WITHOUT cloud masking
s2_collection = (ee.ImageCollection('COPERNICUS/S2')
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                .map(calculate_ndvi_toa)
                .select(['NDVI']))
```

**Key Points:**
- NDVI calculated directly in GEE: `(NIR - Red) / (NIR + Red)`
- Exported as 'S2ndvi' band in combined product
- No explicit scaling applied in GEE
- Data type when exported: Float32 (default for GEE calculations)
- Standard NDVI range: [-1.0, 1.0]

---

### Loading in FuseTS Notebook: `load_gee_assets_to_xarray()`

**Function Location:** S1_S2_GEE_DL_Fusion_Demak_FULL.ipynb, Cell 9

```python
def load_gee_assets_to_xarray(asset_base_path, name_prefix, num_periods=62, 
                               region=None, scale=50):
    """
    Load individual GEE Asset images and convert to xarray Dataset 
    compatible with FuseTS
    """
    
    # Download each period individually
    for period in range(1, num_periods + 1):
        asset_id = f'{asset_base_path}/{name_prefix}{period:02d}'
        
        # Download GeoTIFF using geemap
        geemap.download_ee_image(
            img,
            filename=output_file,
            region=region,
            scale=scale,
            crs='EPSG:4326'
        )
        
        # Load the GeoTIFF
        period_data = rioxarray.open_rasterio(output_file)
        
        # Extract bands (assuming order: VV, VH, S2ndvi or NDVI)
        if len(period_data.band) >= 3:
            vv = period_data.isel(band=0).values      # First band = VV
            vh = period_data.isel(band=1).values      # Second band = VH
            ndvi = period_data.isel(band=2).values    # Third band = NDVI/S2ndvi
        
        all_vv.append(vv)
        all_vh.append(vh)
        all_ndvi.append(ndvi)
    
    # Stack into numpy arrays
    vv_stack = np.stack(all_vv, axis=0)    # Shape: (time, y, x)
    vh_stack = np.stack(all_vh, axis=0)
    ndvi_stack = np.stack(all_ndvi, axis=0)
    
    # Create FuseTS-compatible xarray Dataset
    ds = xr.Dataset({
        'VV': (['t', 'y', 'x'], vv_stack),
        'VH': (['t', 'y', 'x'], vh_stack),
        'S2ndvi': (['t', 'y', 'x'], ndvi_stack)
    }, coords={
        't': time_coords,
        'y': y_coords,
        'x': x_coords
    })
```

**Critical Finding: NO SCALING OR TYPE CONVERSION**
- NDVI loaded directly from GeoTIFF with `rioxarray.open_rasterio()`
- No division by 10000 or scaling factor
- No explicit type conversion (stays as whatever rioxarray loads)
- Data is stacked as-is into numpy arrays
- Values should be in standard NDVI range [-1.0, 1.0]

---

## 2. Data Type and Scaling Transformations

### Investigation of Data Types

**GEE Export:**
- NDVI calculated as normalized difference (always in [-1, 1])
- Exported as GeoTIFF using standard float32 encoding

**Loading Process:**
- `rioxarray.open_rasterio()` preserves the GeoTIFF data type
- No explicit scaling factors found in loading code
- No division operations detected

---

## 3. Actual S2ndvi Data Statistics from Execution

### From Cell 18: Data Verification Output

```
Dataset structure:
<xarray.Dataset> Size: 2GB
Dimensions:     (t: 62, y: 892, x: 1170)

📈 Data variables:
S2ndvi:
  Shape: (62, 892, 1170)
  Valid: 12,038,189 / 64,705,680 (18.6%)
  Range: [-inf, 0.6129]
```

**Critical Finding: -inf Values Detected!**
- 2,041,019 -inf/-inf values out of 64,705,680 total pixels (3.2%)
- This is UNUSUAL for standard NDVI values
- Suggests a division by zero issue in GEE (when NIR = Red)

### From Cell 20: After Cleaning

```
Processing S2ndvi...
  -inf/+inf values: 2,041,019 (3.2%)
  Zero values: 227,084 (0.4%)
  ✅ Valid values after cleaning: 12,038,189 (18.6%)

Final Data Quality:
S2ndvi:
  Valid: 12,038,189 / 64,705,680 (18.6%)
  Range: [-0.2693, 0.6129]
  Mean: 0.1791
```

**After cleaning:**
- -inf values replaced with NaN
- Valid range: [-0.2693, 0.6129] (proper NDVI range!)
- Mean: 0.1791 (reasonable for tropical vegetation)
- 18.6% valid data (due to paddy masking)

### From Cell 21: Final Data Quality Check

```
S2ndvi:
  Shape: (62, 892, 1170)
  Total elements: 64,705,680
  Valid values: 12,038,189 (18.6%)
  NaN values: 52,667,491 (81.4%)
  Range: [-0.2693, 0.6129]
  Mean: 0.1791
```

**Conclusion:** S2ndvi data is correctly loaded and in proper range!

---

## 4. Masking Operations Applied to S2ndvi

### From Cell 11: Paddy Field Mask Application

```
🗺️  LOADING AND APPLYING PADDY FIELD MASK

Dataset to mask: gee_dataset
Dataset shape: FrozenMappingWarningOnValuesAccess({'t': 62, 'y': 892, 'x': 1170})

✅ Shapefile loaded successfully
   Features: 1043
   CRS: EPSG:4326
   Bounds: [110.517654  -7.108495 111.033707  -6.717609]

📊 Mask Statistics:
   Total pixels:      1,043,640
   Paddy pixels:      227,084 (21.76%)
   Non-paddy pixels:  816,556 (78.24%)

🎯 Applying mask to dataset variables...
   VV: 64,705,680 → 14,079,208 valid pixels (21.8% retained)
   VH: 64,705,680 → 14,079,208 valid pixels (21.8% retained)
   S2ndvi: 64,705,680 → 14,079,208 valid pixels (21.8% retained)

✅ Masking complete! Dataset updated with paddy-only pixels.
```

**Masking Details:**
1. Shapefile (Klambu-Glapan paddy fields) loaded: 1043 features
2. Rasterized to match dataset grid (892 × 1170)
3. 21.76% of pixels are paddy areas
4. All variables (VV, VH, S2ndvi) masked identically
5. Non-paddy areas set to NaN

**Key Issue:** Masking is correct, but also limits visibility!
- Only 21.8% of data is paddy area
- The visualization may show 78.2% as NaN/white

---

## 5. Root Cause Analysis: Why NDVI Map Appears Blank

### Finding 1: Data is NOT Blank
- Valid S2ndvi values: 12,038,189 / 64,705,680 (18.6%)
- Range: [-0.2693, 0.6129] (proper NDVI)
- Mean: 0.1791 (reasonable for tropical vegetation)
- **Data quality verified: PASS**

### Finding 2: Masking is Very Aggressive
```
After paddy masking:
- 78.2% of pixels become NaN (outside paddy areas)
- Only 21.8% retained as valid
- But 3.2% of those have -inf (set to NaN in cleaning)
- Final valid: 18.6% only
```

### Root Cause: Combination of Two Factors

1. **Aggressive Spatial Masking:**
   - Paddy shapefile covers only 21.76% of study area
   - Rest of area becomes NaN in visualization
   - This makes 78.2% of the map appear "blank"

2. **Data Gaps (Cloud Cover, Missing S2):**
   - Only 18.6% of paddy pixels have valid NDVI
   - 81.4% are NaN (due to cloud cover, S2 availability gaps)
   - Some periods have 0 S2 images available

3. **Visualization Issue (Likely Primary):**
   - The `paddy_mask_applied.png` shows the entire grid
   - Non-paddy areas are explicitly masked (NaN → white/transparent)
   - Visualization color scale may not be optimized for sparse data
   - Low contrast between NaN areas and valid data areas

---

## 6. Data Export Process from GEE

### Export Configuration (Cell 19 of GEE_Data_Preparation_for_FuseTS_Assets.ipynb)

```python
task = ee.batch.Export.image.toAsset(
    image=image_with_metadata,
    description=f'Asset_Period_{period_num:02d}',
    assetId=period_asset_id,
    scale=scale,            # scale=10m for S2 native resolution
    region=geometry,
    maxPixels=1e13,
    crs='EPSG:4326',
    pyramidingPolicy={'.default': 'mean'}
)
```

**No Scaling Applied in Export:**
- Export resolution: 10m (native S2 resolution)
- Data format: GeoTIFF (Float32 by default)
- No explicit scaling factor
- No integer encoding (would scale for int16)
- NDVI remains in [-1, 1] range

---

## 7. Verification: Data Type Confirmation

From the execution outputs:

```
Dataset structure:
<xarray.Dataset> Size: 2GB
Data variables:
    VV          (t, y, x) float64 518MB ...
    VH          (t, y, x) float64 518MB ...
    S2ndvi      (t, y, x) float64 518MB ...
```

**Data Type:** float64 (64-bit float)
- GeoTIFF float32 → rioxarray promotes to float64
- Correct for scientific computing
- No truncation or integer conversion

---

## 8. Actual Values Sample

### Valid NDVI Pixels (from Cell 21):
```
S2ndvi Range: [-0.2693, 0.6129]
S2ndvi Mean: 0.1791
```

**Interpretation:**
- Minimum (-0.2693): Water bodies or bare soil
- Maximum (0.6129): Healthy, dense vegetation (typical for rice during growth)
- Mean (0.1791): Mixed paddy vegetation states over time

**This is CORRECT NDVI data!**
- No scaling by 10000
- No data type issues
- Proper vegetation dynamics captured

---

## Summary Table

| Aspect | Finding | Status |
|--------|---------|--------|
| **NDVI Calculation** | (NIR - Red)/(NIR + Red) in GEE | Correct |
| **Export Scaling** | None applied | Correct |
| **Data Type** | Float32 (float64 after loading) | Correct |
| **Value Range** | [-1, 1] stored, [-0.2693, 0.6129] actual | Correct |
| **No /10000 Division** | Confirmed - none found | Correct |
| **Loading Process** | Direct GeoTIFF → numpy (no conversion) | Correct |
| **Masking Operations** | Applied to all bands equally | Correct |
| **Data Validity** | 18.6% valid after masking | Expected |
| **Blank Map Cause** | 78.2% masked + visualization scale | Root Cause |

---

## Recommendations

### Why the NDVI Map Appears Blank

1. **78.2% of pixels are outside paddy areas** (masked to NaN)
2. **18.6% remaining pixels have valid NDVI** (should be visible)
3. **Visualization color scale** may need adjustment for sparse data

### To Fix the Visualization

**Option 1: Better Masking Visualization**
```python
# Show only paddy areas with proper scaling
masked_ndvi = combined_dataset['S2ndvi'].where(combined_dataset['paddy_mask'])
masked_ndvi.mean(dim='t').plot(cmap='RdYlGn', vmin=-0.3, vmax=0.7)
```

**Option 2: Check Individual Time Steps**
```python
# Visualize a single period with high data availability
combined_dataset['S2ndvi'].sel(t=combined_dataset.t[20]).plot(
    cmap='RdYlGn', robust=True
)
```

**Option 3: Verify Data Exists**
```python
# Count valid pixels per time step
valid_counts = combined_dataset['S2ndvi'].notnull().sum(['y', 'x'])
print(valid_counts)  # Should show non-zero for each period
```

---

## Conclusion

**The NDVI data is correctly loaded with NO SCALING ISSUES.**

- NDVI values range from -0.2693 to 0.6129 (proper range)
- Mean NDVI: 0.1791 (reasonable for tropical paddy rice)
- No division by 10000 or other scaling
- Data type is correct (float64)
- The blank NDVI map is likely a **visualization/masking display issue**, not a data quality problem

The data is ready for MOGPR fusion analysis.

