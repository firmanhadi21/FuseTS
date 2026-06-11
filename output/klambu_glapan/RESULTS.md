# Klambu–Glapan Rice Phenology via S1+S2 MOGPR Fusion — Results

**Study area:** Klambu–Glapan irrigation command area, Central Java, Indonesia
(AOI `data/klambu-glapan.shp`, ~57 × 43 km, UTM 49S / EPSG:32749)
**Period:** 2024-11-07 → 2025-11-02 (31 × 12-day composites)
**Data:** Microsoft Planetary Computer — Sentinel-2 L2A (NDVI) + Sentinel-1 RTC (VV, VH, dB)
**Method:** FuseTS MOGPR fusion → multi-season phenology (`peakvalley`)
**Generated on macOS (Apple Silicon), conda env `fusets`.**

---

## 1. Pipeline

```
MPC (odc.stac)  →  12-day datacube (VV, VH, S2ndvi)  →  MOGPR fusion  →  phenology
   scripts/klambu_glapan_mogpr.py            scripts/pv_phenology.py / map_phenology.py
```

| Script | Role |
|---|---|
| `scripts/klambu_glapan_mogpr.py` | Extraction → datacube → point MOGPR → phenology (`--points-shp` for field validation) |
| `scripts/pv_phenology.py` | Multi-season `peakvalley` (point + wall-to-wall raster) |
| `scripts/raster_mogpr_phenology.py` | Wall-to-wall MOGPR fusion on a (downsampled) cube |
| `scripts/map_phenology.py` | Phenology maps / GeoTIFFs |
| `scripts/field_zonal_mogpr.py` | Field-polygon-mean fusion from an existing cube |

Reproducible environment: `environment.yml` (FuseTS + MOGPR + MPC stack).

---

## 2. Datacube

- Dimensions **t=31, y=886, x=1162** (~1.03 M px/period), 50 m, UTM 49S.
- All periods carried `S2ndvi`, `VV`, `VH` (gaps as NaN — filled by MOGPR).
- Wet-season optical loss is severe (several periods at 0 % valid NDVI), motivating SAR fusion.
- File: `datacube_klambu_glapan.nc`

---

## 3. Point-based MOGPR + multi-season phenology

200 random sample points; **200/200 fused**; multi-season detection via `peakvalley`.

- **Mean 1.83 seasons / cropped point** (1:51, 2:48, 3:15 points) — consistent with
  irrigated double-rice.
- Mean LOS ≈ 102 d, green-up ≈ 64–78 d, peak NDVI ≈ 0.36, amplitude ≈ 0.28.
- Files: `point_mogpr_fused.csv`, `point_phenology_seasons.csv`,
  `point_phenology_pv_summary.csv`, `phenology_maps_peakvalley.png`.

> Note: single-season phenolopy metrics (`point_phenology.csv`) are retained for
> reference but **collapse the two rice cycles into one** and should not be used
> for double-crop reporting.

---

## 4. Wall-to-wall n_seasons — effect of MOGPR fusion (key result)

Controlled comparison: **both rasters at 200 m, both `peakvalley drop_thr=0.12`** —
the only difference is MOGPR fusion (S1+S2) vs optical-only Whittaker smoothing.

| (200 m, drop 0.12) | 0 | 1 | 2 | 3+ | cropped (≥1) | **double-cropped / cropped** | mean (≥1) |
|---|---|---|---|---|---|---|---|
| Whittaker-only (optical) | 54.3% | 39.5% | 5.9% | 0.3% | 46% | **13%** | 1.14 |
| **MOGPR-fused (S1+S2)** | 47.3% | 26.6% | 17.0% | 9.1% | 53% | **50%** | 1.72 |
| *point-based MOGPR (ref)* | | | | | | | *1.83* |

**Findings**
- MOGPR fusion **~quadruples** detected double-cropping among cropped pixels
  (13 % → 50 %) and raises mean seasons 1.14 → 1.72. Because resolution and
  threshold are held constant, this is attributable to the **SAR fusion**.
- The fused raster **converges to the point-based reference** (1.72 vs 1.83);
  the earlier optical-only mismatch was an artifact of cloud gaps in the wet season.
- Agronomically sensible for an irrigated double-rice command area.
- Optical-only **systematically misses the wet-season crop**; S1 radar sees through
  cloud and MOGPR carries that into NDVI.

**Outputs**
- `n_seasons_mogpr_200m.tif`, `ndvi_mogpr_fused_200m.nc` (137 min run @ ~9 px/s)
- `n_seasons_whittaker_200m.tif` (controlled comparison)
- `n_seasons_compare_whittaker_vs_mogpr.png` (side-by-side map)
- 50 m Whittaker-only multi-season GeoTIFFs: `phenology_tifs_peakvalley/`

> Full-resolution (50 m) raster MOGPR is ~32 h on this laptop — an HPC/GPU job.
> Re-run at other resolutions with `raster_mogpr_phenology.py --factor`.

---

## 5. Ground-truth validation (BulakBakal site)

Independent validation against field-recorded planting dates
(`Validasi_Data_BulakBakal/Digit merge.shp`, 31 sawah polygons near Magelang,
10 m, Sep 2025 → Jun 2026). Detected green-up (SOS) vs recorded `Tgl_Tanam`:

| config | matched | SOS median | MAE | ≤24 d |
|---|---|---|---|---|
| centroid, drop 0.15 | 23/31 | −3 d | 31.7 d | 57% |
| **centroid, drop 0.08 (best)** | **31/31** | **+9 d** | **21.3 d** | **81%** |
| field-mean, drop 0.08 | 30/31 | −13 d | 31.5 d | 73% |

- Detected green-up lags recorded transplanting by a median **+9 days** — correct
  for rice (NDVI rises after transplanting); peak (POS) ~81 d after planting.
- **Field-polygon mean did not help** — small plots dilute the signal with bunds/edges;
  centroid-pixel sampling is better here.
- **Caveat:** `drop_thr=0.08` over-segments (≈3 seasons/field for one documented
  planting); the nearest-season match flatters the MAE. A precision-aware metric and
  field-level cloud handling would tighten this.
- Outputs: `output/bulak_bakal/validation_vs_Tgl_Tanam.csv` and phenology files.

---

## 6. Limitations & next steps

- **Resolution trade-off:** raster MOGPR parity demo is at 200 m; 50 m wall-to-wall
  needs HPC/GPU.
- **peakvalley tuning:** `drop_thr` trades recall vs over-segmentation; a
  multi-season-aware validation metric would set it more rigorously.
- **Cloud extremes:** even with fusion, the Dec–Jan wet-season peak is thin in places.
- **Validation depth:** extend to EOS once `Tgl_Panen` (harvest) is recorded; use
  field-mean with cloud-weighted compositing.
- **Possible extensions:** descending-orbit S1 robustness check, multi-year SOS/EOS
  trend analysis, yield-proxy from seasonal integral.
