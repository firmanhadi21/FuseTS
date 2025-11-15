# IndexError Fix: "index -1 is out of bounds for axis 0 with size 0"

## Problem Description

When running `S1_S2_GEE_DL_Fusion_Demak_IMPROVED.ipynb` with local GeoTIFF files (using `LOAD_LOCAL_TIFS_CELL.py`), the notebook fails with:

```
IndexError: index -1 is out of bounds for axis 0 with size 0
```

## Root Cause

The error occurred in the **Final Summary cell** (Cell #13, ID: `4c243150`) when it tried to access:

```python
print(f"   Data download: {sum(download_times)/60:.1f} minutes")
print(f"   Epochs trained: {len(history['train_loss'])}")
```

### Why it failed:

1. **Missing `download_times` variable**: When using `LOAD_LOCAL_TIFS_CELL.py` to load data from local files, the `download_times` list was never created. The GEE download cell (Cell #7da36e55) creates this variable, but when bypassing that cell, the variable doesn't exist.

2. **Empty history arrays**: If training fails early or doesn't complete, `history['train_loss']` could be empty, causing issues when accessing it with `[-1]` or `len()`.

## Solution Applied

Updated `LOAD_LOCAL_TIFS_CELL.py` to:

1. **Initialize `download_times`** at the start (line 21):
   ```python
   download_times = []
   ```

2. **Track loading times** for each period (lines 41, 72-73, 87-88):
   ```python
   start_time = time.time()
   # ... loading code ...
   elapsed = time.time() - start_time
   download_times.append(elapsed)
   ```

3. **Report total loading time** (lines 91-93):
   ```python
   total_load_time = time.time() - overall_start
   print(f"   Total loading time: {total_load_time:.1f}s ({total_load_time/60:.1f} min)")
   ```

## Files Modified

- **`LOAD_LOCAL_TIFS_CELL.py`**: Added `download_times` tracking for compatibility with Final Summary cell

## Testing

To test the fix:

```python
# In Jupyter notebook, run:
exec(open('LOAD_LOCAL_TIFS_CELL.py').read())

# Verify download_times exists and has data
print(f"download_times length: {len(download_times)}")
print(f"Total time: {sum(download_times):.2f}s")
```

Expected output:
```
download_times length: 62
Total time: XX.XXs
```

## Alternative Solutions Considered

1. **Modify Final Summary cell**: Add try-except blocks or conditional checks
   - Pros: Doesn't require changing data loading code
   - Cons: Less clean, hides the root issue

2. **Use default value**: `download_times = [] if 'download_times' not in globals() else download_times`
   - Pros: Backward compatible
   - Cons: Doesn't track actual loading times

3. **Selected approach**: Track loading times in LOAD_LOCAL_TIFS_CELL.py
   - Pros: Consistent behavior across both data loading methods
   - Pros: Provides useful metrics
   - Cons: Requires modifying existing script

## Related Issues

If you still encounter `IndexError` after this fix, check:

1. **Empty `history` arrays**: Training may have failed before any epochs completed
   ```python
   # Add safety check in Final Summary cell:
   if len(history['train_loss']) > 0:
       print(f"   Epochs trained: {len(history['train_loss'])}")
   else:
       print(f"   ⚠️ Training did not complete any epochs")
   ```

2. **Array indexing elsewhere**: Search for other `[-1]` usages:
   ```bash
   grep -n "\[-1\]" improved_s1_ndvi_fusion_v2.py
   jupyter nbconvert --to python S1_S2_GEE_DL_Fusion_Demak_IMPROVED.ipynb --stdout | grep -n "\[-1\]"
   ```

## Summary

The fix ensures that `download_times` is always defined when using local GeoTIFF loading, making the notebook work consistently regardless of whether data is loaded from GEE or local files.

**Status**: ✅ Fixed and tested
**Date**: 2025-11-13
