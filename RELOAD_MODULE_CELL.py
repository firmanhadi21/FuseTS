# ============================================================================
# FORCE RELOAD improved_s1_ndvi_fusion_v2 MODULE
# ============================================================================
#
# Run this cell BEFORE running the training cell to ensure you have the
# latest version with all bug fixes
#
# ============================================================================

import sys
import importlib

# Force reload the module to get the latest changes
if 'improved_s1_ndvi_fusion_v2' in sys.modules:
    print("🔄 Reloading improved_s1_ndvi_fusion_v2 module...")
    importlib.reload(sys.modules['improved_s1_ndvi_fusion_v2'])
    print("✅ Module reloaded successfully!")
else:
    print("📦 Loading improved_s1_ndvi_fusion_v2 module for first time...")
    import improved_s1_ndvi_fusion_v2
    print("✅ Module loaded!")

# Import the function
from improved_s1_ndvi_fusion_v2 import run_improved_fusion_v2

print("\n🔍 Verification:")
print(f"   Module file: {sys.modules['improved_s1_ndvi_fusion_v2'].__file__}")
print(f"   run_improved_fusion_v2 available: {callable(run_improved_fusion_v2)}")
print("\n✅ Ready to run training!")
