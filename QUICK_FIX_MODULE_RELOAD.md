# Quick Fix: Module Not Reloading in Jupyter

## Problem

You're getting the IndexError even though the fix has been applied to `improved_s1_ndvi_fusion_v2.py`. This is because **Jupyter/IPython caches imported modules**.

When you run `from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2`, Python loads the module **once** and keeps it in memory. Any changes to the `.py` file won't take effect until you reload the module.

## Solution

### Option 1: Run the Reload Cell (Recommended)

Add this cell **RIGHT BEFORE** your training cell:

```python
exec(open('RELOAD_MODULE_CELL.py').read())
```

This will force-reload the module with all the latest bug fixes.

### Option 2: Manual Reload

Add this at the top of your training cell:

```python
import sys
import importlib

# Force reload
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])

from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
```

### Option 3: Restart Kernel

1. Click **Kernel** → **Restart Kernel** in Jupyter
2. Re-run all cells from the beginning
3. The new code will be loaded

## Verification

After reloading, run this to verify you have the fixed version:

```python
import inspect
from improved_s1_ndvi_fusion_v2 import prepare_enhanced_features_v2

source = inspect.getsource(prepare_enhanced_features_v2)
if 'if np.sum(mask_finite) > 0:' in source:
    print("✅ You have the FIXED version with safety checks")
else:
    print("❌ Still using OLD version - try restarting kernel")
```

## Why This Happens

Python module caching is by design:
- **First import**: Python reads the `.py` file and creates the module object
- **Subsequent imports**: Python returns the cached module object (ignores file changes)
- **Purpose**: Faster imports and consistent behavior across imports

This is great for production code but annoying during development!

## Best Practice for Development

When actively editing a module, use one of these approaches:

### 1. Auto-reload (Best for Jupyter)

Add this at the TOP of your notebook (in the first cell):

```python
%load_ext autoreload
%autoreload 2
```

This tells Jupyter to **automatically reload modules before executing code**. Perfect for development!

### 2. Explicit Reload

Add reload calls before using updated functions:

```python
import importlib
import improved_s1_ndvi_fusion_v2
importlib.reload(improved_s1_ndvi_fusion_v2)
```

### 3. Kernel Restart

Simply restart the kernel when you make changes. Clean but slower.

## Your Specific Case

Your notebook currently has:

```python
# Cell: bbc0e9f5-0b47-4f9f-a124-77a88855b572
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
else:
    import improved_s1_ndvi_fusion_v2

from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
```

**Problem**: You need to **run this cell again** after I made changes to the `.py` file!

## Quick Action

Run these 3 cells in order:

```python
# Cell 1: Force reload
import sys, importlib
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2
print("✅ Module reloaded")

# Cell 2: Run diagnostic (verify data is good)
exec(open('DIAGNOSTIC_CELL.py').read())

# Cell 3: Run training
model, pred_train, pred_val, metrics_train, metrics_val, scaler, history = run_improved_fusion_v2(
    combined_dataset,
    batch_size=256_000,
    learning_rate=0.001,
    epochs=150,
    warmup_epochs=5,
    val_split=0.2,
    verbose=True
)
```

This should work now!
