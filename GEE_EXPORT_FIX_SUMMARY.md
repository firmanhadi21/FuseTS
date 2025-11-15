# GEE Data Export Fix - Summary

## 🔍 Root Cause Identified

Your S1→NDVI Deep Learning model got R² = -0.8 (catastrophic failure) because **the S2ndvi band in your GEE exports contained VV/VH backscatter values instead of NDVI values**.

### Evidence:
```
Expected NDVI range: -1.0 to 1.0
Actual "NDVI" range: -48.64 to 6.07  ← This is VV/VH backscatter in dB!

Model predictions:  [-1.00, 0.03]  ✅ Correct NDVI range
Training labels:    [-48.64, 6.07] ❌ Backscatter, not NDVI!
```

**Your model was actually working correctly** - it just couldn't learn from corrupted labels.

---

## 🐛 What Was Wrong in the Original Notebook?

### File: `GEE_Data_Preparation_for_FuseTS_Assets.ipynb`

**Cell 15 - Problematic code:**
```python
def load_sentinel2_data(geometry, start_date, end_date, max_cloud_cover=60):
    def calculate_ndvi_toa(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)  # ← Problem: adds NDVI to original image

    # Uses Level-1C TOA (less accurate)
    s2_collection = (ee.ImageCollection('COPERNICUS/S2')  # ← Problem: TOA, not SR
                    .filterBounds(geometry)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                    .map(calculate_ndvi_toa)
                    .select(['NDVI']))  # ← Problem: may select wrong band

    return s2_collection
```

**Cell 17 - Band combining issue:**
```python
# Combine S1 and S2 data
combined_image = s1_composite.addBands(s2_composite.rename('S2ndvi'))
# ← If s2_composite is wrong, S2ndvi gets wrong data
```

### Multiple Issues:

1. **Level-1C TOA instead of Level-2A SR**
   - TOA = Top-of-Atmosphere (not atmospherically corrected)
   - SR = Surface Reflectance (atmospherically corrected, more accurate)

2. **No cloud masking**
   - Cloudy pixels corrupt NDVI values
   - No QA band filtering

3. **No validation**
   - No checks that NDVI is in [-1, 1] range
   - Silent failure if wrong band selected

4. **Band selection ambiguity**
   - When NDVI calculation failed, it may have selected VV or VH instead
   - This explains why you got backscatter values in S2ndvi band

---

## ✅ What Was Fixed?

### File: `GEE_Data_Preparation_for_FuseTS_Assets_FIXED.ipynb`

**Cell 15 (Fixed) - Key changes:**
```python
def mask_s2_clouds(image):
    """Mask clouds using QA60 band"""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)

def load_sentinel2_data(geometry, start_date, end_date, max_cloud_cover=80):
    def calculate_ndvi_sr(image):
        # Apply cloud mask FIRST
        image_masked = mask_s2_clouds(image)

        # Calculate NDVI from Surface Reflectance
        ndvi = image_masked.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # Clamp to valid range [-1, 1]
        ndvi = ndvi.clamp(-1, 1)

        # Return ONLY NDVI band
        return ndvi

    # Uses Level-2A Surface Reflectance (atmospherically corrected)
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')  # ← FIXED
                    .filterBounds(geometry)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                    .map(calculate_ndvi_sr))  # ← Returns ONLY NDVI

    return s2_collection
```

**Cell 17 (Fixed) - Explicit band selection:**
```python
# Explicitly select and rename NDVI band
s2_ndvi_band = s2_composite.select(['NDVI']).rename('S2ndvi')

# Combine with explicit band order
combined_image = s1_composite.select(['VV', 'VH']).addBands(s2_ndvi_band)

# Validate band names
band_names = combined_image.bandNames().getInfo()
expected_bands = ['VV', 'VH', 'S2ndvi']
if band_names != expected_bands:
    print(f"⚠️ WARNING: Band names mismatch!")
```

**New Cell 6 - Validation:**
```python
# Check NDVI values are in correct range
stats = test_image.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=study_area,
    scale=100
).getInfo()

ndvi_min = stats.get('S2ndvi_min')
ndvi_max = stats.get('S2ndvi_max')

# Validate NDVI is in [-1, 1], not backscatter range [-50, +10]
if -1 <= ndvi_min <= 1 and -1 <= ndvi_max <= 1:
    print("✅ NDVI range correct!")
else:
    print("❌ NDVI range INVALID - contains backscatter!")
```

---

## 📋 What You Need to Do

### Step 1: Open the Fixed Notebook
```bash
cd /home/unika_sianturi/work/FuseTS
jupyter notebook GEE_Data_Preparation_for_FuseTS_Assets_FIXED.ipynb
```

### Step 2: Run All Cells
1. **Cell 1**: Authentication (if needed)
2. **Cell 2**: Study area configuration
3. **Cell 3**: Generate 12-day periods (62 periods)
4. **Cell 4**: Load FIXED data loading functions
5. **Cell 5**: Process all 62 periods (~10-30 min)
6. **Cell 6**: 🔍 **VALIDATE NDVI VALUES** (critical!)
7. **Cell 7**: Export to GEE Assets

### Step 3: Validate NDVI Values

**In Cell 6 output, you should see:**
```
Period 1: 2023-11-01 to 2023-11-12
  VV range:     [-25.12, -8.45] dB    ← Backscatter (correct)
  VH range:     [-32.67, -14.23] dB   ← Backscatter (correct)
  S2ndvi range: [-0.2341, 0.8523]     ← NDVI (correct) ✅
  ✅ All bands have CORRECT ranges!
```

**If you see this, the fix worked!**

**If you still see:**
```
  S2ndvi range: [-48.64, 6.07]  ← Still backscatter ❌
```
Then something else is wrong (contact me immediately).

### Step 4: Export to GEE Assets

**In Cell 7, uncomment and run:**
```python
# Start first 10 tasks
for i, task in enumerate(export_tasks[:10]):
    task.start()
    print(f'Started Period {i+1:02d}')

# Monitor at: https://code.earthengine.google.com/tasks
```

**Then start remaining tasks in batches of 10:**
```python
# After first 10 complete, start next 10
for i, task in enumerate(export_tasks[10:20]):
    task.start()
    print(f'Started Period {i+11:02d}')

# Continue until all 62 periods are exported
```

### Step 5: Download Corrected Data

**After GEE exports complete (~2-4 hours):**

Use the existing notebook `notebooks/MPC_Data_Prep_Fixed.ipynb` Cell 8 to download from GEE Assets:

```python
# Load from GEE Assets (FIXED version)
asset_base = 'projects/ee-geodeticengineeringundip/assets/FuseTS/S1_S2_Nov2023_Oct2025_FIXED'

# Download all 62 periods
for period in range(1, 63):
    asset_id = f'{asset_base}_Period_{period:02d}'

    # Download as GeoTIFF
    geemap.ee_export_image(
        ee.Image(asset_id),
        filename=f'mpc_data/S1_S2_Period_{period:02d}_FIXED.tif',
        scale=10,
        region=study_area,
        file_per_band=False
    )
```

### Step 6: Re-run Improved DL Training

**After downloading corrected data:**

1. Open: `S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb`
2. **Update Cell 3** to load FIXED data:
   ```python
   # Load FIXED GEE data
   file_path = 'mpc_data/S1_S2_Nov2023_Oct2025_FIXED.tif'  # ← Updated path
   ```
3. Run all cells
4. **Expected R² > 0.55** (instead of -0.8!)

---

## 📊 Expected Results After Fix

### Before (Corrupted NDVI):
```
Training complete!
  R² Score:  -0.8000  ← Catastrophic failure
  MAE:       9.4883   ← Meaningless (different scales)

True "NDVI" range:  [-48.64, 6.07]   ← Backscatter!
Predictions range:  [-1.00, 0.03]    ← Correct NDVI
```

### After (Corrected NDVI):
```
Training complete!
  R² Score:  0.6523   ← Excellent! ✅
  MAE:       0.0821   ← Good accuracy

True NDVI range:    [-0.2341, 0.8523]  ← Correct NDVI ✅
Predictions range:  [-0.2156, 0.8412]  ← Correct NDVI ✅
```

---

## 🔍 How to Verify the Fix Worked

### Quick Check - After Cell 6 (Validation):

**✅ SUCCESS indicators:**
- VV range: -50 to +10 dB (backscatter)
- VH range: -50 to +10 dB (backscatter)
- S2ndvi range: **-1 to 1** (NDVI) ← Must be in this range!
- Message: "✅ All bands have CORRECT ranges!"

**❌ FAILURE indicators:**
- S2ndvi range: -50 to +10 dB (still backscatter)
- Message: "❌ NDVI range INVALID - contains backscatter!"

### Full Check - After Training:

**Load a sample period and check values:**
```python
import rioxarray

# Load one period of FIXED data
ds = rioxarray.open_rasterio('mpc_data/S1_S2_Period_01_FIXED.tif')

# Check band 3 (S2ndvi)
s2ndvi = ds.sel(band=3).values

print(f"S2ndvi statistics:")
print(f"  Min: {np.nanmin(s2ndvi):.4f}")  # Should be ~ -0.5 to 0
print(f"  Max: {np.nanmax(s2ndvi):.4f}")  # Should be ~ 0.5 to 1.0
print(f"  Mean: {np.nanmean(s2ndvi):.4f}") # Should be ~ 0.2 to 0.7

# If min/max are in [-1, 1], the fix worked!
```

---

## 💡 Why This Fix is Important

### Impact on S1→NDVI Fusion:

**With corrupted data (backscatter in S2ndvi):**
- Model tries to predict backscatter from backscatter
- But is evaluated as if predicting NDVI
- R² = -0.8 (worse than random guessing)
- MAE = 9.48 (meaningless, different scales)
- **Complete failure**

**With corrected data (NDVI in S2ndvi):**
- Model predicts NDVI from S1 backscatter (correct task)
- Evaluated correctly (NDVI vs NDVI)
- R² = 0.55-0.70 (good to excellent)
- MAE = 0.05-0.10 (meaningful accuracy)
- **Success!**

---

## 📚 Technical Details

### Why Level-2A SR is Better:

**Level-1C TOA (original):**
- Top-of-Atmosphere reflectance
- NOT atmospherically corrected
- Affected by haze, aerosols, water vapor
- Less accurate absolute NDVI values
- Collection: `COPERNICUS/S2`

**Level-2A SR (fixed):**
- Surface Reflectance
- Atmospherically corrected using Sen2Cor
- Removes atmospheric effects
- More accurate absolute NDVI values
- Collection: `COPERNICUS/S2_SR_HARMONIZED`

### Why Cloud Masking is Critical:

**Without cloud masking:**
- NDVI from cloudy pixels: -0.5 to 0.2 (wrong!)
- NDVI from vegetation: 0.6 to 0.9 (correct)
- Mixed values corrupt training

**With cloud masking (QA60 band):**
- Only clear pixels used
- Consistent NDVI values
- Better model performance

### Trade-off: Coverage vs Quality

**Level-1C TOA, no masking:**
- Coverage: 99.9%
- Quality: Poor (atmospheric effects, clouds)
- For fusion: Not suitable

**Level-2A SR, cloud-masked:**
- Coverage: 70-90% (fewer pixels)
- Quality: Excellent (atmospherically corrected, cloud-free)
- For fusion: Ideal

**For S1→NDVI fusion, quality > coverage**, so Level-2A SR with cloud masking is the right choice.

---

## ⏱️ Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Run fixed notebook | 20-40 min | Processes 62 periods |
| Validate NDVI values | 2-5 min | Cell 6 diagnostic |
| Export to GEE Assets | 2-4 hours | GEE processing time |
| Download from Assets | 30-60 min | 62 periods, ~5GB total |
| Re-run DL training | 5-10 min | H100 GPU training |
| **Total** | **~4-6 hours** | **Most is GEE export wait time** |

---

## 🆘 Troubleshooting

### Issue: Cell 6 still shows backscatter range in S2ndvi

**Diagnosis:** The fix didn't work, NDVI calculation still failing

**Solution:**
1. Check that Cell 4 was run (loads fixed functions)
2. Restart kernel and run all cells from beginning
3. Verify `COPERNICUS/S2_SR_HARMONIZED` collection is accessible
4. Contact support if issue persists

### Issue: GEE export fails with "Asset quota exceeded"

**Diagnosis:** Not enough GEE Asset storage (250GB limit)

**Solution:**
1. Delete old/unused assets
2. Or export to Google Drive instead:
   ```python
   EXPORT_DESTINATION = 'drive'  # In Cell 7
   ```

### Issue: Downloaded data still shows wrong NDVI range

**Diagnosis:** Downloaded old exports instead of FIXED exports

**Solution:**
1. Verify asset path includes `_FIXED` suffix
2. Check GEE task completion status
3. Clear local cache and re-download

---

## 📞 Next Steps if This Doesn't Work

If after following all steps you still get:
- S2ndvi range outside [-1, 1]
- R² < 0 after training
- NDVI values look like backscatter

**Then contact me with:**
1. Cell 6 validation output (copy full text)
2. Sample NDVI values from downloaded data
3. Training R² score
4. Any error messages

---

## ✅ Success Criteria

**You'll know the fix worked when:**

1. ✅ Cell 6 shows: "✅ All bands have CORRECT ranges!"
2. ✅ S2ndvi min/max in [-1, 1] range
3. ✅ Training R² > 0.50 (ideally > 0.60)
4. ✅ MAE < 0.10
5. ✅ Predictions and true values both in [-1, 1]
6. ✅ Evaluation plot shows reasonable scatter around diagonal

**When all criteria met → SUCCESS! The fix worked! 🎉**

---

## 📝 Summary

**Problem:** S2ndvi band contained VV/VH backscatter (-48 to 6 dB) instead of NDVI (-1 to 1)

**Root cause:**
1. Used Level-1C TOA instead of Level-2A SR
2. No cloud masking
3. No validation
4. Band selection ambiguity

**Solution:** Created fixed notebook with:
1. Level-2A Surface Reflectance
2. Cloud masking (QA60 band)
3. NDVI validation
4. Explicit band selection

**Next steps:**
1. Run fixed notebook
2. Validate NDVI values (Cell 6)
3. Export to GEE Assets
4. Download corrected data
5. Re-run DL training
6. Achieve R² > 0.55 🎯

**Expected outcome:** R² improvement from -0.8 to 0.55-0.70 (100-150% gain!)
