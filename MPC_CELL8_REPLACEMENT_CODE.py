"""
MPC Data Loading Code for S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb

This code replaces Cell 8 in the fusion notebook to load MPC data
from pre-processed NetCDF files instead of downloading from GEE.
"""

import xarray as xr
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

USE_MPC_DATA = True  # Set to True to load from NetCDF, False to use synthetic
MPC_DATA_DIR = Path('mpc_data')

# Choose which file to load:
# - 'test_timeseries.nc' for quick testing (3 periods, smaller)
# - 'klambu_glapan_2024-11-01_2025-11-07_final.nc' for full data (31 periods)
INPUT_FILENAME = 'test_timeseries.nc'

# ============================================================================
# Data Loading Function
# ============================================================================

def load_mpc_data_from_netcdf(mpc_dir, filename):
    """
    Load pre-processed MPC S1/S2 data from NetCDF file
    
    This function loads the output from MPC_Data_Prep_Fixed.ipynb and
    returns it in xarray format ready for MOGPR processing.
    
    Parameters:
    -----------
    mpc_dir : Path or str
        Directory containing NetCDF file (default: 'mpc_data/')
    filename : str
        NetCDF filename
        Options:
        - 'test_timeseries.nc' (quick test, 3 periods)
        - 'klambu_glapan_2024-11-01_2025-11-07_final.nc' (full, 31 periods)
    
    Returns:
    --------
    xr.Dataset
        FuseTS-ready dataset with structure:
        - Dimensions: (t: time_periods, y: pixels_north, x: pixels_east)
        - Variables: VV, VH, S2ndvi (all float32)
        - Coordinates: t (datetime), y (northing), x (easting)
    
    Raises:
    -------
    FileNotFoundError
        If the NetCDF file does not exist
    ValueError
        If required variables are missing
    """
    
    filepath = Path(mpc_dir) / filename
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"\nNetCDF file not found: {filepath}\n"
            f"Please ensure:\n"
            f"  1. MPC_Data_Prep_Fixed.ipynb has been run\n"
            f"  2. The output file exists in: {mpc_dir}/\n"
            f"  3. File name matches: {filename}"
        )
    
    print(f"📖 Loading MPC data from: {filepath}")
    
    # Load the NetCDF file
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        raise ValueError(f"Error reading NetCDF file: {e}")
    
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {ds['VV'].shape}")
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Time range: {ds.coords['t'].values[0]} to {ds.coords['t'].values[-1]}")
    if 'crs' in ds.attrs:
        print(f"   CRS: {ds.attrs['crs']}")
    if 'spatial_resolution' in ds.attrs:
        print(f"   Spatial resolution: {ds.attrs['spatial_resolution']}")
    
    # Verify required variables exist
    required_vars = ['VV', 'VH', 'S2ndvi']
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Missing required variables: {missing}\n"
            f"Dataset has: {list(ds.data_vars)}"
        )
    
    # Ensure time dimension is named 't' (FuseTS requirement)
    if 'time' in ds.dims and 't' not in ds.dims:
        print("   Renaming 'time' dimension to 't'...")
        ds = ds.rename({'time': 't'})
    
    # Add metadata if not present
    if 'fusets_ready' not in ds.attrs:
        ds.attrs['fusets_ready'] = True
    
    return ds


# ============================================================================
# Execution
# ============================================================================

if USE_MPC_DATA:
    print("="*70)
    print("🌍 LOADING DATA FROM MPC (Microsoft Planetary Computer)")
    print("="*70)
    
    try:
        # Load the MPC data
        mpc_dataset = load_mpc_data_from_netcdf(MPC_DATA_DIR, INPUT_FILENAME)
        
        if mpc_dataset is not None:
            print("\n" + "="*70)
            print("📊 DATASET SUMMARY")
            print("="*70)
            print(mpc_dataset)
            print("\n✅ Data ready for MOGPR processing!")
            print("   Variable name to use: 'mpc_dataset'")
            print("   Continue with: combined_dataset = mpc_dataset")
        else:
            print("❌ Failed to load MPC data")
            mpc_dataset = None
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Next steps:")
        print("   1. Open MPC_Data_Prep_Fixed.ipynb in a new tab")
        print("   2. Run all cells (or at least through cell 7 for test data)")
        print("   3. Come back and re-run this cell")
        mpc_dataset = None
    
    except ValueError as e:
        print(f"\n❌ Data validation error: {e}")
        mpc_dataset = None

else:
    print("ℹ️  Using synthetic data (USE_MPC_DATA=False)")
    print("   To use MPC data, set USE_MPC_DATA=True above")
    mpc_dataset = None


# ============================================================================
# Next Step: Use the data
# ============================================================================

# After loading, use the data in subsequent cells:
# 
# if mpc_dataset is not None:
#     # Prepare for MOGPR
#     combined_dataset = mpc_dataset
#     
#     # Extract coordinates for visualization
#     s1_vv = mpc_dataset['VV'].values
#     s1_vh = mpc_dataset['VH'].values
#     s2_ndvi = mpc_dataset['S2ndvi'].values
#     time_coords = mpc_dataset['t'].values
#     y_coords = mpc_dataset['y'].values
#     x_coords = mpc_dataset['x'].values
#     
#     print(f"\nData shapes:")
#     print(f"  VV: {s1_vv.shape}")
#     print(f"  VH: {s1_vh.shape}")
#     print(f"  NDVI: {s2_ndvi.shape}")

