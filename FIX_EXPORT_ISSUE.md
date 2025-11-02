# 🔧 FIX: Missing Data in GEE Assets Export

## Problem Identified

The `export_timeseries_to_asset()` function in `GEE_Data_Preparation_for_FuseTS_Assets.ipynb` exports images **WITHOUT** applying the Java Island mask, resulting in assets with bounds but no actual pixel data for most of Java.

## Root Cause

**File:** `GEE_Data_Preparation_for_FuseTS_Assets.ipynb`  
**Cell:** 13 (Export functions)  
**Line:** ~490

```python
# CURRENT CODE (BROKEN):
task = ee.batch.Export.image.toAsset(
    image=image_with_metadata,  # ← No mask applied!
    description=f'Asset_Period_{period_num:02d}',
    assetId=period_asset_id,
    scale=scale,
    region=geometry,  # ← This is Java+buffer geometry, but image has no data
    ...
)
```

The `geometry` parameter defines the export bounds, but the `image` itself doesn't have the mask applied, so it exports empty pixels.

## Solution

### Option 1: Don't Use Mask (Simplest - Recommended for Testing)

Export the full composite without mask restriction:

```python
# In export_timeseries_to_asset() function, around line 490:
task = ee.batch.Export.image.toAsset(
    image=image_with_metadata,
    description=f'Asset_Period_{period_num:02d}',
    assetId=period_asset_id,
    scale=scale,
    region=geometry.bounds(),  # Use bounds() to get bbox
    maxPixels=1e13,
    crs='EPSG:4326',
    pyramidingPolicy={'.default': 'mean'}
)
```

This will export all data within the bounding box (including ocean), but it will work.

### Option 2: Apply Mask to Image (Correct Long-term Solution)

Modify the export function to apply the Java Island mask:

```python
def export_timeseries_to_asset(collection, geometry, scale, asset_id, mask=None):
    """
    Export the time series collection to GEE Assets as ImageCollection
    
    NEW PARAMETER:
    mask : ee.Image, optional
        Binary mask to apply (1=export, 0=no-data)
    """
    tasks = []
    image_list = collection.toList(collection.size())
    
    for i in range(len(successful_periods)):
        image = ee.Image(image_list.get(i))
        period_num = successful_periods[i]['period']
        period_info = successful_periods[i]
        
        # Add comprehensive metadata
        image_with_metadata = image.set({
            'period': period_num,
            'start_date': period_info['start_str'],
            'end_date': period_info['end_str'],
            'center_date': period_info['center_date'].strftime('%Y-%m-%d'),
            'doy_center': period_info['doy_center'],
            'year': period_info['year'],
            'month': period_info['month'],
            'system:time_start': ee.Date(period_info['start_str']).millis(),
            'system:time_end': ee.Date(period_info['end_str']).millis()
        })
        
        # APPLY MASK IF PROVIDED
        if mask is not None:
            image_with_metadata = image_with_metadata.updateMask(mask.gt(0))
        
        # Create asset ID for this period
        period_asset_id = f'{asset_id}_Period_{period_num:02d}'
        
        task = ee.batch.Export.image.toAsset(
            image=image_with_metadata,
            description=f'Asset_Period_{period_num:02d}',
            assetId=period_asset_id,
            scale=scale,
            region=geometry,
            maxPixels=1e13,
            crs='EPSG:4326',
            pyramidingPolicy={'.default': 'mean'}
        )
        
        tasks.append(task)
    
    return tasks
```

Then in the export call (around line 603), pass the mask:

```python
# Need to recreate Java mask as ee.Image from the geometry
# Or load the original mask file
java_mask = ee.Image(1).clip(study_area)  # Simple mask from geometry

export_tasks = export_timeseries_to_asset(
    time_series_collection,
    study_area,
    SCALE,
    asset_id,
    mask=java_mask  # ← Add mask parameter
)
```

## Quick Fix Steps

1. **Open** `GEE_Data_Preparation_for_FuseTS_Assets.ipynb`

2. **Find Cell 13** (export functions section)

3. **Locate** the `export_timeseries_to_asset()` function (~line 445-500)

4. **Choose** Option 1 or Option 2 above

5. **Delete old assets:**
   - Go to https://code.earthengine.google.com
   - Navigate to Assets tab
   - Delete folder: `projects/ee-geodeticengineeringundip/assets/FuseTS`
   - Or delete individual `Period_XX` assets

6. **Re-run export** with fixed code

7. **Verify** using the diagnostic cells in `S1_S2_MOGPR_Fusion_Tutorial.ipynb`

## Testing

After re-export, run this in the MOGPR tutorial notebook:

```python
# Should now show data for ALL regions
✅ HAS DATA    Western Java (Banten)      (105.50°E, -6.50°N)
✅ HAS DATA    West Java                  (107.00°E, -6.80°N)
✅ HAS DATA    Central Java               (109.50°E, -7.20°N)
✅ HAS DATA    Central Java (Coast)       (110.50°E, -6.90°N)
✅ HAS DATA    Eastern Java               (112.50°E, -7.50°N)
✅ HAS DATA    East Java (Tip)            (114.00°E, -7.80°N)
```

## Alternative: Use Google Drive Export Instead

If assets continue to have issues, use Drive export as fallback:

In `GEE_Data_Preparation_for_FuseTS_Assets.ipynb`:
```python
EXPORT_DESTINATION = 'drive'  # Instead of 'asset'
```

Drive exports are more reliable but have size limits.
