# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FuseTS is a Time Series & Data Fusion toolbox integrated with openEO for multi-temporal, multi-sensor earth observation data integration and analysis. It's part of the AI4FOOD project and provides advanced time series analytics and machine learning capabilities for land environment monitoring.

## Common Development Commands

### Testing
```bash
pytest                                    # Run all tests
pytest tests/test_specific_module.py      # Run specific test file
pytest tests/ -v                         # Run tests with verbose output
pytest --norecursedirs=tests/fusets_openeo_tests  # Exclude openEO integration tests
```

### Code Quality
```bash
black .                      # Format code with Black (line length: 120)
ruff check .                 # Run Ruff linter (fixes imports automatically)
ruff check --fix .           # Auto-fix linting issues
pre-commit run --all-files   # Run all pre-commit hooks
```

### Build and Installation
```bash
pip install -e .             # Install in development mode
pip install -e .[dev]        # Install with development dependencies
python setup.py build        # Build the package
```

## Core Architecture

### Main Components

The package is structured around three primary time series processing algorithms:

1. **MOGPR (Multi-Output Gaussian Process Regression)** (`src/fusets/mogpr.py`)
   - Integrates multiple time series and reconstructs gaps using correlations between indicators
   - Based on Luca Pipia's methodology for sensor fusion
   - Implements `MOGPRTransformer` class inheriting from `BaseEstimator`

2. **Whittaker Smoothing** (`src/fusets/whittaker.py`)
   - Computationally efficient method for smoothing and gap-filling time series
   - Handles missing/cloud-masked values with weight vectors
   - Implements `WhittakerTransformer` class with configurable lambda parameter

3. **Peak Valley Detection** (`src/fusets/peakvalley.py`)
   - Time series analysis for detecting peaks and valleys in temporal data

4. **Phenology Analysis** (`src/fusets/_phenolopy.py`, `src/fusets/analytics.py`)
   - Comprehensive phenological metrics calculation based on TIMESAT 3.3 methodology
   - Calculates Start of Season (SOS), End of Season (EOS), Length of Season, and other phenometrics
   - Multiple detection methods: seasonal_amplitude, first_of_slope, median_of_slope, absolute_value
   - Accessed via `fusets.analytics.phenology()` function

5. **CropSAR Integration** (`src/fusets/cropsar/__init__.py`)
   - Advanced Sentinel-1 + Sentinel-2 fusion using deep learning models
   - Handles VV/VH SAR data with multitemporal speckle filtering
   - Supports GAN and AttentionUNet models for gap-filling and reconstruction
   - Processes power/dB conversions and spatial windowing automatically

### Base Classes

- **BaseEstimator** (`src/fusets/base.py`): Base class for all transformers, providing sklearn-style interface
- **Analytics utilities** (`src/fusets/analytics.py`): Common analytical functions including phenology
- **XArray utilities** (`src/fusets/_xarray_utils.py`): Helper functions for working with xarray data structures

### OpenEO Integration

The `src/fusets/openeo/` directory contains User Defined Functions (UDFs) for cloud processing:
- `mogpr_udf.py`, `whittaker_udf.py`, `peakvalley_udf.py`, `phenology_udf.py`
- Services in `src/fusets/openeo/services/` for publishing algorithms to openEO backends

### Dependencies and Environment

- **Python**: >= 3.8
- **Core dependencies**: numpy==1.23.5, xarray>=0.20.2, GPy>=1.10.0, vam.whittaker==2.0.6
- **Geospatial I/O**: rioxarray for GeoTIFF reading/writing (used in CropSAR module)
- **OpenEO integration**: Optional openeo package for cloud processing
- **Development**: Uses conda environment (`environment_whittaker.yml`)

## Key Design Patterns

1. **Transformer Pattern**: All main algorithms implement fit/transform methods following sklearn conventions
2. **Optional Dependencies**: OpenEO functionality is conditionally imported based on availability
3. **XArray Integration**: Heavy use of xarray for multi-dimensional time series data handling
4. **Modular UDFs**: Separate UDF implementations for cloud processing workflows

## Testing Structure

- Main tests in `tests/` directory
- OpenEO-specific integration tests in `tests/fusets_openeo_tests/` (excluded by default)
- Test resources in `tests/resources/`
- Uses pytest with configuration in `pytest.ini`

## Code Style Configuration

- **Black**: Line length 120, Python 3 formatting
- **Ruff**: Targets Python 3.8+, focuses on import sorting and unused imports (F401, I)
- **Pre-commit**: Automated formatting and linting on commits
- Package follows `fusets` as first-party import namespace

## Common Use Cases and Workflows

### Phenological Analysis from Time Series Data

For extracting seasonal metrics (Start/End of Season, planting indices) from satellite time series:

```python
from fusets.analytics import phenology
from fusets import whittaker

# Load time series data (NDVI, RVI, etc.)
smoothed_data = whittaker(raw_timeseries, lmbd=10000)
phenology_metrics = phenology(smoothed_data)

# Extract seasonal metrics
sos_times = phenology_metrics.da_sos_times      # Start of Season (day of year)
eos_times = phenology_metrics.da_eos_times      # End of Season (day of year)
sos_values = phenology_metrics.da_sos_values    # Vegetation values at SOS
eos_values = phenology_metrics.da_eos_values    # Vegetation values at EOS
```

### Multi-Sensor Data Fusion

For combining Sentinel-1 and Sentinel-2 data:

```python
from fusets.mogpr import MOGPRTransformer
import xarray as xr

# Prepare combined dataset with required band names
combined_data = xr.Dataset({
    'VV': s1_vv_data,      # Sentinel-1 VV band
    'VH': s1_vh_data,      # Sentinel-1 VH band
    'S2ndvi': s2_ndvi_data # Sentinel-2 NDVI
})

# Apply MOGPR fusion
mogpr = MOGPRTransformer()
fused_result = mogpr.fit_transform(combined_data)
```

### GeoTIFF Data Processing

For loading and processing GeoTIFF stacks:

```python
import rioxarray

# Load GeoTIFF stack (automatically handled by rioxarray in CropSAR module)
data_stack = rioxarray.open_rasterio("timeseries_stack.tif")
data_stack = data_stack.rename({"band": "time"})

# Process with FuseTS algorithms
result = whittaker(data_stack, lmbd=10000)
```

### Google Earth Engine Integration

While FuseTS doesn't have native GEE integration, it can process data exported from Google Earth Engine. The repository includes a dedicated workflow:

```python
# 1. Data extraction from GEE (see GEE_Data_Preparation_for_FuseTS.ipynb)
import ee
import geemap

# Extract S1/S2 data using 12-day composites
periods = generate_12day_periods(2024)  # 31 periods for full year
s1_s2_collection = process_all_periods(periods, study_area)

# Export to Google Drive or extract locally
exported_data = export_timeseries_to_drive(s1_s2_collection)

# 2. Local processing with FuseTS
import rioxarray
from fusets.mogpr import MOGPRTransformer

# Load exported GeoTIFF
data = rioxarray.open_rasterio('S1_S2_TimeSeries_2024.tif')
fusets_data = prepare_fusets_format(data)

# Apply MOGPR fusion
mogpr = MOGPRTransformer()
fused_result = mogpr.fit_transform(fusets_data)
```

**Temporal Compositing Strategy for GEE**:
- **31 periods** of 12-day composites per year
- **Period 1**: Jan 1-12, **Period 2**: Jan 13-25, etc.
- Median composites with cloud masking for S2
- Both individual period files and combined multi-band exports supported

**Required data structure for multi-sensor processing**:
- Dimensions: `(t, y, x)` (note: `t` not `time`)
- Band naming: `['VV', 'VH']` for S1, `['S2ndvi']` for S2 NDVI
- Proper coordinate naming essential for MOGPR compatibility

## Tutorial Notebooks

The repository includes comprehensive Jupyter notebooks for different workflows:

### 1. S1_S2_MOGPR_Fusion_Tutorial.ipynb
Complete end-to-end tutorial for MOGPR-based sensor fusion:
- Synthetic data generation for demonstration
- Proper xarray Dataset formatting for FuseTS
- MOGPR fusion workflow with gap-filling analysis
- Phenological metrics extraction (SOS/EOS detection)
- Visualization of fusion benefits vs single-sensor approaches
- Export capabilities for further analysis

### 2. GEE_Data_Preparation_for_FuseTS.ipynb
Google Earth Engine data extraction specifically designed for FuseTS:
- Authentication and GEE setup
- 12-day composite generation (31 periods per year)
- S1 (VV/VH) and S2 (NDVI) data collection with quality filtering
- Multiple export options (individual periods vs combined files)
- Automatic conversion to FuseTS-compatible format
- Local extraction for small study areas
- Processing metadata and timeline visualization

### Workflow Integration:
```bash
# 1. Extract data from GEE
jupyter notebook GEE_Data_Preparation_for_FuseTS.ipynb

# 2. Download exported data from Google Drive
# 3. Apply MOGPR fusion
jupyter notebook S1_S2_MOGPR_Fusion_Tutorial.ipynb
```

Both notebooks include extensive documentation, error handling, and are designed for both educational and production use.