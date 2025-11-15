
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

