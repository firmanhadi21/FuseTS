"""
Bridge module for integrating R sits package outputs with FuseTS

This module provides utilities to convert time series data extracted from
the R sits package into xarray formats compatible with FuseTS workflows.

Supported input formats:
- CSV files exported from sits
- NetCDF files exported from sits
- GeoTIFF stacks created by sits_cube_copy

Example workflows:
    # From CSV (point time series)
    >>> from fusets.io.sits_bridge import load_sits_csv
    >>> data = load_sits_csv("sits_timeseries.csv",
    ...                       time_col='Index',
    ...                       band_cols=['NDVI', 'EVI'])
    >>> from fusets.mogpr import MOGPRTransformer
    >>> mogpr = MOGPRTransformer()
    >>> fused = mogpr.fit_transform(data)

    # From GeoTIFF stack
    >>> from fusets.io.sits_bridge import load_sits_geotiff
    >>> data = load_sits_geotiff("sits_output/NDVI_*.tif")
    >>> from fusets import whittaker
    >>> smoothed = whittaker(data, lmbd=10000)
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import warnings

try:
    import rioxarray as rxr
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False
    warnings.warn("rioxarray not available. GeoTIFF loading will not work.")


def load_sits_csv(
    filepath: Union[str, Path],
    time_col: str = "Index",
    band_cols: Optional[List[str]] = None,
    location_cols: Optional[List[str]] = None,
    time_format: str = "%Y-%m-%d",
) -> xr.Dataset:
    """
    Load time series from sits CSV export into FuseTS-compatible xarray Dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file exported from sits
    time_col : str, default="Index"
        Name of the time/date column
    band_cols : list of str, optional
        Names of spectral band columns. If None, auto-detects numeric columns
    location_cols : list of str, optional
        Columns identifying spatial locations (e.g., ['longitude', 'latitude', 'label'])
    time_format : str, default="%Y-%m-%d"
        Format string for parsing time column

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (t, location) and variables for each band

    Examples
    --------
    # Simple NDVI time series
    >>> data = load_sits_csv("sits_ndvi.csv", band_cols=['NDVI'])

    # Multi-sensor S1+S2
    >>> data = load_sits_csv("sits_s1_s2.csv",
    ...                       band_cols=['VV', 'VH', 'NDVI', 'EVI'])
    """
    df = pd.read_csv(filepath)

    # Parse time column
    df[time_col] = pd.to_datetime(df[time_col], format=time_format)

    # Auto-detect band columns if not specified
    if band_cols is None:
        # Assume numeric columns are bands
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if location_cols:
            band_cols = [c for c in numeric_cols if c not in location_cols]
        else:
            band_cols = numeric_cols

    # Identify location columns
    if location_cols is None:
        location_cols = [c for c in df.columns if c not in [time_col] + band_cols]

    # Create multi-index if we have location data
    if location_cols:
        df = df.set_index(location_cols + [time_col])
    else:
        df = df.set_index(time_col)

    # Convert to xarray
    ds = df[band_cols].to_xarray()

    # Rename time dimension to 't' for FuseTS compatibility
    if time_col in ds.dims:
        ds = ds.rename({time_col: 't'})

    return ds


def load_sits_netcdf(filepath: Union[str, Path]) -> xr.Dataset:
    """
    Load time series from sits NetCDF export into FuseTS-compatible format.

    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file exported from sits

    Returns
    -------
    xr.Dataset
        Dataset formatted for FuseTS processing

    Examples
    --------
    >>> data = load_sits_netcdf("sits_timeseries.nc")
    >>> from fusets.mogpr import MOGPRTransformer
    >>> mogpr = MOGPRTransformer()
    >>> fused = mogpr.fit_transform(data)
    """
    ds = xr.open_dataset(filepath)

    # Ensure time dimension is named 't' for FuseTS
    time_dims = ['time', 'date', 'Index', 'datetime']
    for dim in time_dims:
        if dim in ds.dims:
            ds = ds.rename({dim: 't'})
            break

    return ds


def load_sits_geotiff(
    filepath_pattern: Union[str, Path],
    time_dimension: str = "band",
    time_coords: Optional[List] = None,
) -> xr.DataArray:
    """
    Load GeoTIFF stack created by sits into FuseTS-compatible xarray DataArray.

    Parameters
    ----------
    filepath_pattern : str or Path
        Path to GeoTIFF file(s). Can be:
        - Single multi-band GeoTIFF: "stack.tif"
        - Wildcard pattern: "NDVI_*.tif"
    time_dimension : str, default="band"
        Name of the dimension representing time
    time_coords : list, optional
        List of datetime objects or strings for time coordinates

    Returns
    -------
    xr.DataArray
        DataArray with dimensions (t, y, x) ready for FuseTS

    Examples
    --------
    # Single multi-band file
    >>> data = load_sits_geotiff("sits_output/NDVI_stack.tif")
    >>> from fusets import whittaker
    >>> smoothed = whittaker(data, lmbd=10000)

    # Multiple files with time coordinates
    >>> import pandas as pd
    >>> times = pd.date_range("2024-01-01", periods=31, freq="12D")
    >>> data = load_sits_geotiff("sits_output/NDVI_*.tif", time_coords=times)
    """
    if not HAS_RIOXARRAY:
        raise ImportError("rioxarray is required for GeoTIFF loading. "
                         "Install with: pip install rioxarray")

    # Load the data
    data = rxr.open_rasterio(filepath_pattern, masked=True)

    # Rename dimension to 't' for FuseTS
    if time_dimension in data.dims:
        data = data.rename({time_dimension: 't'})

    # Add time coordinates if provided
    if time_coords is not None:
        data = data.assign_coords({'t': time_coords})

    return data


def prepare_mogpr_format(
    s1_vv: Union[xr.DataArray, pd.Series, np.ndarray],
    s1_vh: Union[xr.DataArray, pd.Series, np.ndarray],
    s2_ndvi: Union[xr.DataArray, pd.Series, np.ndarray],
    time_coords: Optional[List] = None,
) -> xr.Dataset:
    """
    Prepare multi-sensor data for MOGPR fusion following FuseTS conventions.

    This is a convenience function for the common S1+S2 fusion case.

    Parameters
    ----------
    s1_vv : array-like
        Sentinel-1 VV polarization time series
    s1_vh : array-like
        Sentinel-1 VH polarization time series
    s2_ndvi : array-like
        Sentinel-2 NDVI time series
    time_coords : list, optional
        Time coordinates for the data

    Returns
    -------
    xr.Dataset
        Dataset with variables ['VV', 'VH', 'S2ndvi'] ready for MOGPR

    Examples
    --------
    >>> # From sits CSV extractions
    >>> df = pd.read_csv("sits_s1_s2.csv")
    >>> mogpr_data = prepare_mogpr_format(
    ...     s1_vv=df['VV'].values,
    ...     s1_vh=df['VH'].values,
    ...     s2_ndvi=df['NDVI'].values,
    ...     time_coords=pd.to_datetime(df['Index'])
    ... )
    >>> from fusets.mogpr import MOGPRTransformer
    >>> mogpr = MOGPRTransformer()
    >>> fused = mogpr.fit_transform(mogpr_data)
    """
    # Convert to DataArrays if needed
    if not isinstance(s1_vv, xr.DataArray):
        s1_vv = xr.DataArray(s1_vv, dims=['t'])
    if not isinstance(s1_vh, xr.DataArray):
        s1_vh = xr.DataArray(s1_vh, dims=['t'])
    if not isinstance(s2_ndvi, xr.DataArray):
        s2_ndvi = xr.DataArray(s2_ndvi, dims=['t'])

    # Create dataset with FuseTS naming convention
    ds = xr.Dataset({
        'VV': s1_vv,
        'VH': s1_vh,
        'S2ndvi': s2_ndvi
    })

    # Add time coordinates
    if time_coords is not None:
        ds = ds.assign_coords({'t': time_coords})

    return ds


def sits_tibble_to_xarray(
    tibble_csv: Union[str, Path],
    cube_name: str = "default",
) -> xr.Dataset:
    """
    Convert sits tibble format (typical sits output) to xarray Dataset.

    This handles the nested tibble structure that sits uses where each row
    is a location and time series are nested within columns.

    Parameters
    ----------
    tibble_csv : str or Path
        Path to CSV file exported from sits tibble
    cube_name : str, default="default"
        Name for the cube/dataset

    Returns
    -------
    xr.Dataset
        Dataset formatted for FuseTS

    Notes
    -----
    This is for advanced sits users working with the native tibble format.
    For most use cases, load_sits_csv() with pre-flattened data is simpler.
    """
    # This would need to parse the nested structure
    # Implementation depends on exact sits tibble export format
    raise NotImplementedError(
        "Direct tibble parsing not yet implemented. "
        "Please use sits_to_csv() in R to flatten the tibble first, "
        "then use load_sits_csv()."
    )


# Utility function for R users
def get_r_helper_script() -> str:
    """
    Returns R code for exporting sits data in FuseTS-compatible formats.

    Returns
    -------
    str
        R code to be saved and sourced in R

    Examples
    --------
    >>> from fusets.io.sits_bridge import get_r_helper_script
    >>> r_code = get_r_helper_script()
    >>> with open("sits_to_fusets.R", "w") as f:
    ...     f.write(r_code)
    """
    r_code = '''
# sits to FuseTS Export Helpers
# Source this file in R before exporting data

library(sits)
library(dplyr)
library(tidyr)

#' Export sits time series to FuseTS-compatible CSV
#'
#' @param sits_data sits tibble with time series
#' @param output_file path to output CSV file
#' @param bands character vector of band names to export
#'
#' @examples
#' sits_to_fusets_csv(my_timeseries, "for_fusets.csv",
#'                    bands = c("VV", "VH", "NDVI"))
sits_to_fusets_csv <- function(sits_data, output_file, bands = NULL) {

  # Extract time series from sits tibble
  ts_list <- sits_data$time_series

  # Flatten to long format
  ts_df <- bind_rows(ts_list, .id = "location_id")

  # Filter bands if specified
  if (!is.null(bands)) {
    ts_df <- ts_df %>% select(location_id, Index, all_of(bands))
  }

  # Add location metadata
  metadata <- sits_data %>%
    select(-time_series) %>%
    mutate(location_id = as.character(row_number()))

  ts_df <- ts_df %>% left_join(metadata, by = "location_id")

  # Export
  write.csv(ts_df, output_file, row.names = FALSE)
  message("Exported to ", output_file)
  message("Load in Python with: load_sits_csv('", output_file, "')")

  return(invisible(ts_df))
}

#' Export sits cube to GeoTIFF stack for FuseTS
#'
#' @param cube sits data cube
#' @param output_dir directory for GeoTIFF outputs
#' @param bands character vector of band names
#'
#' @examples
#' sits_cube_to_fusets_geotiff(my_cube, "fusets_input",
#'                              bands = c("NDVI", "EVI"))
sits_cube_to_fusets_geotiff <- function(cube, output_dir, bands = NULL) {

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Export using sits built-in function
  sits_cube_copy(cube,
                 output_dir = output_dir,
                 format = "GTiff",
                 bands = bands)

  message("Exported GeoTIFF stack to ", output_dir)
  message("Load in Python with: load_sits_geotiff('",
          file.path(output_dir, "*.tif"), "')")

  return(invisible(NULL))
}

#' Quick setup for sits + FuseTS workflow
#'
#' @param source data source ("MPC", "BDC", "AWS", etc.)
#' @param collection collection name
#' @param roi region of interest (sf object)
#' @param start_date start date
#' @param end_date end date
#' @param bands bands to retrieve
#'
#' @return sits cube ready for export
sits_for_fusets <- function(source = "MPC",
                             collection = "SENTINEL-2-L2A",
                             roi,
                             start_date,
                             end_date,
                             bands = c("B04", "B08", "B11")) {

  message("Setting up sits cube for FuseTS workflow...")

  cube <- sits_cube(
    source = source,
    collection = collection,
    roi = roi,
    start_date = start_date,
    end_date = end_date,
    bands = bands
  )

  message("Cube created. Next steps:")
  message("1. For point/polygon time series:")
  message("   ts <- sits_get_data(cube, samples = points)")
  message("   sits_to_fusets_csv(ts, 'output.csv')")
  message("")
  message("2. For spatial raster processing:")
  message("   sits_cube_to_fusets_geotiff(cube, 'output_dir')")

  return(cube)
}
'''
    return r_code
