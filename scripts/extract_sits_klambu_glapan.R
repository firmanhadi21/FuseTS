# Extract S1 + S2 Data for Klambu-Glapan Area Using sits
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

cat("=======================================================================\n")
cat("  S1 + S2 Data Extraction for Klambu-Glapan Area (sits + FuseTS)\n")
cat("=======================================================================\n\n")

# Load study area
study_area_path <- "/home/unika_sianturi/work/FuseTS/data/klambu-glapan.shp"

cat("Loading study area:", study_area_path, "\n")
study_area <- st_read(study_area_path, quiet = TRUE)

# Check and print study area info
cat("✓ Study area loaded:\n")
cat("  Features:", nrow(study_area), "\n")
cat("  CRS:", st_crs(study_area)$input, "\n")
cat("  Bounding box:\n")
print(st_bbox(study_area))

# Temporal period
start_date <- "2023-11-01"
end_date <- "2025-10-30"

cat("\nTemporal period:\n")
cat("  Start:", start_date, "\n")
cat("  End:", end_date, "\n")

# Number of random sample points
n_points <- 200  # You can change this

cat("\nNumber of random sample points:", n_points, "\n")

# Output directory
output_dir <- "/home/unika_sianturi/work/FuseTS/data/sits_exports"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Output directory:", output_dir, "\n\n")

# =============================================================================
# GENERATE RANDOM SAMPLE POINTS
# =============================================================================

cat("Generating", n_points, "random points inside study area...\n")

# Generate random points within the study area polygon
set.seed(42)  # For reproducibility
sample_points <- st_sample(study_area, size = n_points, type = "random")

# Convert to sf object for easier handling
sample_points_sf <- st_as_sf(sample_points)
sample_points_sf$point_id <- 1:n_points

cat("✓ Random points generated\n")

# Optional: Save sample points for reference
sample_points_path <- file.path(output_dir, "sample_points_klambu_glapan.shp")
st_write(sample_points_sf, sample_points_path, delete_dsn = TRUE, quiet = TRUE)
cat("  Sample points saved to:", sample_points_path, "\n")

# Visualize sample points (optional)
cat("\nSample point locations:\n")
coords <- st_coordinates(sample_points)
cat("  X range:", round(min(coords[,1]), 4), "to", round(max(coords[,1]), 4), "\n")
cat("  Y range:", round(min(coords[,2]), 4), "to", round(max(coords[,2]), 4), "\n\n")

# =============================================================================
# EXTRACT SENTINEL-2 DATA
# =============================================================================

cat("-----------------------------------------------------------------------\n")
cat("STEP 1: Extracting Sentinel-2 NDVI data\n")
cat("-----------------------------------------------------------------------\n")

cat("Requesting S2 cube from Microsoft Planetary Computer...\n")

s2_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-2-L2A",
  roi = study_area,
  start_date = start_date,
  end_date = end_date,
  bands = c("B04", "B08")  # Red, NIR
)

cat("✓ S2 cube retrieved\n")
cat("  Tiles:", nrow(s2_cube), "\n")

# Calculate NDVI
cat("Calculating NDVI...\n")
s2_ndvi <- sits_apply(s2_cube,
  NDVI = (B08 - B04) / (B08 + B04)
)

cat("✓ NDVI calculated\n")

# Extract time series at sample points
cat("Extracting S2 time series at", n_points, "sample points...\n")
cat("(This may take a few minutes depending on the number of points)\n")

s2_samples <- sits_get_data(s2_ndvi, samples = sample_points)

cat("✓ S2 time series extracted\n")
cat("  Samples:", nrow(s2_samples), "\n")
cat("  Time series length:", nrow(s2_samples$time_series[[1]]), "observations\n\n")

# =============================================================================
# EXTRACT SENTINEL-1 DATA
# =============================================================================

cat("-----------------------------------------------------------------------\n")
cat("STEP 2: Extracting Sentinel-1 SAR data\n")
cat("-----------------------------------------------------------------------\n")

cat("Requesting S1 cube from Microsoft Planetary Computer...\n")

s1_cube <- sits_cube(
  source = "MPC",
  collection = "SENTINEL-1-GRD",
  roi = study_area,
  start_date = start_date,
  end_date = end_date,
  bands = c("VV", "VH")
)

cat("✓ S1 cube retrieved\n")
cat("  Tiles:", nrow(s1_cube), "\n")

# Extract time series at sample points
cat("Extracting S1 time series at", n_points, "sample points...\n")
cat("(This may take a few minutes depending on the number of points)\n")

s1_samples <- sits_get_data(s1_cube, samples = sample_points)

cat("✓ S1 time series extracted\n")
cat("  Samples:", nrow(s1_samples), "\n")
cat("  Time series length:", nrow(s1_samples$time_series[[1]]), "observations\n\n")

# =============================================================================
# EXPORT TO FUSETS-COMPATIBLE CSV
# =============================================================================

cat("-----------------------------------------------------------------------\n")
cat("STEP 3: Exporting to FuseTS-compatible CSV format\n")
cat("-----------------------------------------------------------------------\n")

# Export S1 data
s1_csv <- file.path(output_dir, "s1_klambu_glapan_timeseries.csv")
cat("Exporting S1 data to:", s1_csv, "\n")
sits_to_fusets_csv(s1_samples, s1_csv, bands = c("VV", "VH"))
cat("✓ S1 CSV exported\n")

# Export S2 NDVI data
s2_csv <- file.path(output_dir, "s2_klambu_glapan_ndvi_timeseries.csv")
cat("Exporting S2 NDVI data to:", s2_csv, "\n")
sits_to_fusets_csv(s2_samples, s2_csv, bands = c("NDVI"))
cat("✓ S2 CSV exported\n")

# =============================================================================
# DATA QUALITY SUMMARY
# =============================================================================

cat("\n")
cat("=======================================================================\n")
cat("  DATA EXTRACTION COMPLETE\n")
cat("=======================================================================\n\n")

cat("Study Area: Klambu-Glapan\n")
cat("Time Period:", start_date, "to", end_date, "\n")
cat("Sample Points:", n_points, "\n\n")

cat("Exported Files:\n")
cat("  1. S1 time series:     ", s1_csv, "\n")
cat("  2. S2 NDVI time series:", s2_csv, "\n")
cat("  3. Sample points:      ", sample_points_path, "\n\n")

# Data quality summary
s1_ts_length <- nrow(s1_samples$time_series[[1]])
s2_ts_length <- nrow(s2_samples$time_series[[1]])

cat("Data Quality:\n")
cat("  S1 observations per point: ~", s1_ts_length, "\n")
cat("  S2 observations per point: ~", s2_ts_length, "\n")
cat("  S1 temporal frequency: ~6 days (Sentinel-1A+B)\n")
cat("  S2 temporal frequency: ~5 days (Sentinel-2A+B)\n\n")

# Check for data completeness
sample_ts_s1 <- s1_samples$time_series[[1]]
sample_ts_s2 <- s2_samples$time_series[[1]]

s1_completeness <- sum(!is.na(sample_ts_s1$VV)) / nrow(sample_ts_s1) * 100
s2_completeness <- sum(!is.na(sample_ts_s2$NDVI)) / nrow(sample_ts_s2) * 100

cat("Data Completeness (sample point 1):\n")
cat("  S1 VV: ", round(s1_completeness, 1), "% (SAR = all-weather)\n", sep="")
cat("  S2 NDVI: ", round(s2_completeness, 1), "% (optical = cloud-affected)\n", sep="")
cat("\n")

if (s2_completeness < 70) {
  cat("⚠ Low S2 completeness suggests cloud contamination\n")
  cat("  → MOGPR fusion will be very beneficial!\n\n")
} else {
  cat("✓ Good S2 completeness, but MOGPR can still improve results\n\n")
}

# =============================================================================
# NEXT STEPS
# =============================================================================

cat("-----------------------------------------------------------------------\n")
cat("NEXT STEPS\n")
cat("-----------------------------------------------------------------------\n\n")

cat("1. Open Python/Jupyter:\n")
cat("   cd /home/unika_sianturi/work/FuseTS/notebooks\n")
cat("   jupyter notebook Paddyfield_Phenology_S1_S2_Fusion.ipynb\n\n")

cat("2. Update file paths in the notebook:\n")
cat("   s1_data = load_sits_csv('", s1_csv, "')\n", sep="")
cat("   s2_data = load_sits_csv('", s2_csv, "')\n\n", sep="")

cat("3. Run notebook cells:\n")
cat("   - Apply MOGPR fusion\n")
cat("   - Extract phenology metrics\n")
cat("   - Visualize results\n")
cat("   - Export phenology maps\n\n")

cat("=======================================================================\n")
cat("Done! Data ready for FuseTS processing.\n")
cat("=======================================================================\n")
