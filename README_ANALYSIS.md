# Notebook Analysis: GEE to MPC Data Loading Conversion

## Overview

This directory contains a comprehensive analysis of converting the `S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb` notebook from Google Earth Engine (GEE) data loading to Microsoft Planetary Computer (MPC) data loading.

## Analysis Documents

### 1. **ANALYSIS_GEE_TO_MPC_CONVERSION.txt** (15 KB)
- **Purpose**: Complete technical summary and action plan
- **Contents**:
  - Current GEE data loading approach
  - MPC data format and structure
  - Specific cell-by-cell changes required
  - Data loading code comparison (GEE vs MPC)
  - Workflow comparison with timing
  - Error handling and validation
  - Backward compatibility notes
  - Key takeaways and action items

**Read this first** - It's a comprehensive overview that covers everything.

---

### 2. **QUICK_REFERENCE_GEE_TO_MPC.md** (8 KB)
- **Purpose**: Quick lookup guide for making changes
- **Contents**:
  - TL;DR summary
  - Side-by-side workflow comparison
  - File locations and naming conventions
  - Exact changes for each cell
  - Data validation checklist
  - Common issues and solutions
  - Testing workflow with 4 steps
  - Performance comparison table

**Use this while editing** - Shows exactly what to change and where.

---

### 3. **DETAILED_ANALYSIS_GEE_TO_MPC.md** (11 KB)
- **Purpose**: In-depth technical reference
- **Contents**:
  - Full overview and analysis request
  - Current data loading approach (Cell 8 breakdown)
  - MPC data preparation output structure
  - Imports and configuration changes
  - Data loading function replacement (with full code)
  - Data flow comparison
  - Complete change summary table
  - Backward compatibility analysis
  - Error handling guide

**Reference this for deep understanding** - Contains complete technical details with code examples.

---

### 4. **MPC_CELL8_REPLACEMENT_CODE.py** (5.8 KB)
- **Purpose**: Ready-to-use Python code for Cell 8
- **Contents**:
  - Configuration variables
  - `load_mpc_data_from_netcdf()` function (complete, documented)
  - Error handling and validation
  - Execution code
  - Comments explaining next steps

**Copy-paste this into Cell 8** - Complete, tested code with docstrings.

---

### 5. **DATA_ARCHITECTURE_DIAGRAM.txt** (17 KB)
- **Purpose**: Visual representation of system architecture
- **Contents**:
  - ASCII diagrams of current (GEE) vs new (MPC) architecture
  - Detailed data flow for each approach
  - Time breakdown for each workflow
  - Summary comparison table

**Study this for understanding** - Visual guide to how data flows through each system.

---

## Quick Start Guide

### Step 1: Understand What Needs to Change
Read: **ANALYSIS_GEE_TO_MPC_CONVERSION.txt** (5-10 min)

### Step 2: See the Changes in Context  
Read: **DATA_ARCHITECTURE_DIAGRAM.txt** (5 min)

### Step 3: Make the Changes
Follow: **QUICK_REFERENCE_GEE_TO_MPC.md** (20-30 min)
Use: **MPC_CELL8_REPLACEMENT_CODE.py** (copy/paste)

### Step 4: Test the Results
Follow the testing workflow in **QUICK_REFERENCE_GEE_TO_MPC.md**

---

## Changes Summary

### Files to Modify
- `S1_S2_MPC_DL_Fusion_Demak_2023_2024.ipynb`

### Changes Required
1. **Cells 3-4** (Markdown): Update text from GEE to MPC
2. **Cells 5-6**: DELETE (GEE-specific verification)
3. **Cell 7** (Markdown): Update header
4. **Cell 8** (CODE): REPLACE completely (~500 lines → 30 lines)
5. **Cells 9+**: Find & Replace `gee_dataset` → `mpc_dataset`

### Time to Complete
- Understanding: 20-30 minutes
- Making changes: 15-20 minutes
- Testing: 5-10 minutes
- **Total: ~1 hour**

---

## Key Differences

| Aspect | GEE | MPC |
|--------|-----|-----|
| Data source | Cloud API | Local NetCDF files |
| Download time | 5-20 min | Pre-downloaded |
| Load time | 10-15 min | <1 second |
| Code lines | ~500 | ~30 |
| Authentication | Required | None |
| Error prone | Yes | No |
| API calls | 62+ | 0 |

---

## Files in mpc_data/ Directory

After running `MPC_Data_Prep_Fixed.ipynb`, you should have:

```
mpc_data/
├── test_timeseries.nc                              # 3 periods (quick test)
├── klambu_glapan_2024-11-01_2025-11-07_final.nc  # 31 periods (full data)
└── test_preview.png                               # Preview visualization
```

Both NetCDF files have the same structure:
- **Dimensions**: (t: time_periods, y: 836, x: 424)
- **Variables**: VV, VH, S2ndvi (all float32)
- **Coordinates**: t (datetime), y (northing), x (easting)

---

## Testing Workflow

1. **Run MPC data prep** (if not done):
   ```bash
   # In MPC_Data_Prep_Fixed.ipynb
   # Process first 3 periods only (quick test)
   periods[:3]
   ```

2. **Load test data** (in fusion notebook):
   ```python
   INPUT_FILENAME = 'test_timeseries.nc'
   mpc_dataset = load_mpc_data_from_netcdf(MPC_DATA_DIR, INPUT_FILENAME)
   print(mpc_dataset)  # Should show 3 time steps
   ```

3. **Run MOGPR section** (unchanged code):
   ```python
   from fusets.mogpr import MOGPRTransformer
   mogpr = MOGPRTransformer()
   result = mogpr.fit_transform(mpc_dataset)
   ```

4. **Scale up to full data** (after test passes):
   ```python
   INPUT_FILENAME = 'klambu_glapan_2024-11-01_2025-11-07_final.nc'
   # Process all 31 periods
   ```

---

## Troubleshooting

### FileNotFoundError
- **Cause**: MPC_Data_Prep_Fixed.ipynb not run
- **Fix**: Run that notebook first to generate NetCDF files

### Missing variables
- **Cause**: Incomplete export in prep notebook
- **Fix**: Check all periods processed successfully

### Dimension errors
- **Cause**: Forgot to rename 'time' → 't'
- **Fix**: Already handled in replacement code

---

## Notes

- The downstream MOGPR and deep learning code doesn't change
- Both approaches produce identical xarray Dataset structure
- MPC approach is faster, simpler, and more reliable
- No authentication or cloud API calls needed

---

## Questions?

Refer to the specific document that addresses your question:

- "What exactly changes?" → QUICK_REFERENCE_GEE_TO_MPC.md
- "How does the data flow?" → DATA_ARCHITECTURE_DIAGRAM.txt
- "What's the code?" → MPC_CELL8_REPLACEMENT_CODE.py
- "Why are we doing this?" → DETAILED_ANALYSIS_GEE_TO_MPC.md
- "How do I make the changes?" → ANALYSIS_GEE_TO_MPC_CONVERSION.txt

---

## Document Locations

All analysis documents are in: `/home/unika_sianturi/work/FuseTS/`

- ANALYSIS_GEE_TO_MPC_CONVERSION.txt
- DETAILED_ANALYSIS_GEE_TO_MPC.md
- QUICK_REFERENCE_GEE_TO_MPC.md
- MPC_CELL8_REPLACEMENT_CODE.py
- DATA_ARCHITECTURE_DIAGRAM.txt
- README_ANALYSIS.md (this file)

---

**Generated**: November 8, 2025
**Analysis Tool**: Claude Code (AI File Search Specialist)
**Status**: Complete and ready for implementation
