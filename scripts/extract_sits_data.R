# Extract S1 + S2 Data Using sits
# Run this script in R or RStudio

library(sits)
library(sf)
library(terra)
library(dplyr)

# Source FuseTS helper functions
source("/home/unika_sianturi/work/FuseTS/scripts/sits_to_fusets.R")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Study area
study_area <- st_read("/path/to/your/study_area.shp")

# Or create from coordinates
# study_area <- st_bbox(c(xmin = 110.5, ymin = -6.95,
#                         xmax = 110.8, ymax = -6.75),
#                       crs = 4326) %>% st_as_sfc() %>% st_as_sf()

# Temporal period
start_date <- "2024-01-01"
end_date <- "2024-12-31"

# Output directory
output_dir <- "/home/unika_sianturi/work/FuseTS/data/sits_exports"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# EXTRACT SENTINEL-2 DATA
# =============================================================================

cat("Extracting Sentinel-2 data...\n")

s2_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = study_area,
  start_date = start_date,
  end_date = end_date,
  bands = c("B04", "B08")  # Red, NIR
)

# Calculate NDVI
s2_ndvi <- sits_apply(s2_cube,
  NDVI = (B08 - B04) / (B08 + B04)
)

cat("✓ S2 data extracted\n")

# =============================================================================
# EXTRACT SENTINEL-1 DATA
# =============================================================================

cat("Extracting Sentinel-1 data...\n")

s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = study_area,
  start_date = start_date,
  end_date = end_date,
  bands = c("VV", "VH")
)

cat("✓ S1 data extracted\n")

# =============================================================================
# EXPORT OPTIONS
# =============================================================================

cat("\nChoose export method:\n")
cat("  1. Point-based (faster, recommended for testing)\n")
cat("  2. Raster-based (full spatial coverage)\n")

export_method <- readline("Enter choice (1 or 2): ")

if (export_method == "1") {
  # =============================================================================
  # OPTION A: POINT-BASED EXPORT
  # =============================================================================

  cat("\nPoint-based export selected\n")

  # Sample points
  n_points <- as.numeric(readline("Number of sample points (e.g., 200): "))
  sample_points <- st_sample(study_area, size = n_points)

  # Extract time series
  cat("Extracting S1 time series...\n")
  s1_samples <- sits_get_data(s1_cube, samples = sample_points)

  cat("Extracting S2 time series...\n")
  s2_samples <- sits_get_data(s2_ndvi, samples = sample_points)

  # Export to CSV
  s1_csv <- file.path(output_dir, "s1_timeseries.csv")
  s2_csv <- file.path(output_dir, "s2_ndvi_timeseries.csv")

  sits_to_fusets_csv(s1_samples, s1_csv, bands = c("VV", "VH"))
  sits_to_fusets_csv(s2_samples, s2_csv, bands = c("NDVI"))

  cat(sprintf("\n✓ Point data exported:\n"))
  cat(sprintf("  S1: %s\n", s1_csv))
  cat(sprintf("  S2: %s\n", s2_csv))
  cat(sprintf("  Locations: %d\n", n_points))

} else if (export_method == "2") {
  # =============================================================================
  # OPTION B: RASTER-BASED EXPORT
  # =============================================================================

  cat("\nRaster-based export selected\n")

  s1_output_dir <- file.path(output_dir, "s1_rasters")
  s2_output_dir <- file.path(output_dir, "s2_rasters")

  cat("Exporting S1 rasters...\n")
  sits_cube_to_fusets_geotiff(s1_cube, s1_output_dir, bands = c("VV", "VH"))

  cat("Exporting S2 rasters...\n")
  sits_cube_to_fusets_geotiff(s2_ndvi, s2_output_dir, bands = c("NDVI"))

  cat(sprintf("\n✓ Raster data exported:\n"))
  cat(sprintf("  S1: %s\n", s1_output_dir))
  cat(sprintf("  S2: %s\n", s2_output_dir))

} else {
  cat("Invalid choice. Exiting.\n")
  quit(status = 1)
}

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("DATA EXTRACTION COMPLETE\n")
cat(strrep("=", 70) %+% "\n\n")

cat("Next steps:\n")
cat("  1. Open Python/Jupyter\n")
cat("  2. Navigate to: /home/unika_sianturi/work/FuseTS/notebooks/\n")
cat("  3. Open: Paddyfield_Phenology_S1_S2_Fusion.ipynb\n")
cat("  4. Update file paths to point to exported data\n")
cat("  5. Run Python cells for MOGPR fusion and phenology extraction\n\n")

cat("Output location: %s\n", output_dir)
cat("\nDone!\n")
