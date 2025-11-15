#!/usr/bin/env python3
"""
Diagnostic script to check combined_dataset before running fusion
Run this in your notebook to identify data issues
"""

import numpy as np

def diagnose_dataset(combined_dataset):
    """
    Diagnose issues with combined_dataset that might cause IndexError

    Args:
        combined_dataset: xarray Dataset with VV, VH, S2ndvi
    """
    print("="*80)
    print("🔍 DATASET DIAGNOSTIC")
    print("="*80)

    # Check structure
    print("\n📊 Structure:")
    print(f"   Dimensions: {dict(combined_dataset.dims)}")
    print(f"   Variables: {list(combined_dataset.data_vars)}")
    print(f"   Coordinates: {list(combined_dataset.coords)}")

    # Check if data exists
    for var in ['VV', 'VH', 'S2ndvi']:
        if var not in combined_dataset.data_vars:
            print(f"\n❌ ERROR: Variable '{var}' not found!")
            continue

        data = combined_dataset[var].values

        print(f"\n📈 {var}:")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")

        # Check for data
        total_elements = data.size
        finite_elements = np.sum(np.isfinite(data))
        nan_elements = np.sum(np.isnan(data))
        inf_elements = np.sum(np.isinf(data))

        print(f"   Total elements: {total_elements:,}")
        print(f"   Finite values:  {finite_elements:,} ({finite_elements/total_elements:.1%})")
        print(f"   NaN values:     {nan_elements:,} ({nan_elements/total_elements:.1%})")
        print(f"   Inf values:     {inf_elements:,} ({inf_elements/total_elements:.1%})")

        # Value range (only for finite values)
        if finite_elements > 0:
            finite_data = data[np.isfinite(data)]
            print(f"   Range: [{np.min(finite_data):.4f}, {np.max(finite_data):.4f}]")
            print(f"   Mean ± std: {np.mean(finite_data):.4f} ± {np.std(finite_data):.4f}")
        else:
            print(f"   ❌ NO FINITE VALUES!")

    # Check spatial patterns
    print(f"\n🗺️  Spatial Coverage:")
    for var in ['VV', 'VH', 'S2ndvi']:
        if var not in combined_dataset.data_vars:
            continue

        data = combined_dataset[var].values

        # Check each time slice
        if len(data.shape) == 3:
            n_times = data.shape[0]
            valid_per_time = [np.sum(np.isfinite(data[t, :, :])) for t in range(n_times)]

            if n_times > 0:
                print(f"   {var}:")
                print(f"      Time slices with data: {np.sum(np.array(valid_per_time) > 0)} / {n_times}")
                print(f"      Avg valid pixels per time: {np.mean(valid_per_time):.0f}")
                print(f"      Min valid pixels: {np.min(valid_per_time)}")
                print(f"      Max valid pixels: {np.max(valid_per_time)}")

                # Check if any time slice is completely empty
                empty_slices = [t for t, v in enumerate(valid_per_time) if v == 0]
                if empty_slices:
                    print(f"      ⚠️  Empty time slices: {len(empty_slices)} (periods: {empty_slices[:10]}...)")

    # Overall assessment
    print(f"\n📋 Assessment:")

    all_finite = True
    for var in ['VV', 'VH', 'S2ndvi']:
        if var not in combined_dataset.data_vars:
            print(f"   ❌ Missing variable: {var}")
            all_finite = False
        else:
            data = combined_dataset[var].values
            if np.sum(np.isfinite(data)) == 0:
                print(f"   ❌ {var} has NO finite values!")
                all_finite = False
            elif np.sum(np.isfinite(data)) < 1000:
                print(f"   ⚠️  {var} has very few finite values (<1000)")

    if all_finite:
        print(f"   ✅ Dataset structure looks OK")
        print(f"   ✅ All variables have finite values")

        # Check if enough overlap
        VV_finite = np.isfinite(combined_dataset['VV'].values.flatten())
        VH_finite = np.isfinite(combined_dataset['VH'].values.flatten())
        NDVI_finite = np.isfinite(combined_dataset['S2ndvi'].values.flatten())

        all_finite_mask = VV_finite & VH_finite & NDVI_finite
        overlap_count = np.sum(all_finite_mask)

        print(f"\n   Samples with ALL three variables finite: {overlap_count:,}")

        if overlap_count < 10000:
            print(f"   ⚠️  WARNING: Very few samples ({overlap_count:,}) have all three variables!")
            print(f"   This may indicate:")
            print(f"   - Poor temporal overlap between S1 and S2")
            print(f"   - Excessive masking")
            print(f"   - Data loading issues")
        else:
            print(f"   ✅ Sufficient overlap for training ({overlap_count:,} samples)")
    else:
        print(f"   ❌ CRITICAL: Dataset has missing or empty variables!")
        print(f"   Cannot proceed with training until data issues are resolved")

    print("="*80)

    return all_finite


if __name__ == "__main__":
    print("Import this module and call diagnose_dataset(combined_dataset)")
    print("Example:")
    print("  from diagnose_dataset import diagnose_dataset")
    print("  diagnose_dataset(combined_dataset)")
