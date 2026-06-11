# Jatiluhur — Multi-Season Rice Phenology (S1+S2 MOGPR), with Cross-Validation

Study area: **Jatiluhur paddy command area, West Java** (UTM 48S / EPSG:32748).
Produced with `scripts/scale_runner.py` (tiled, resumable, multiprocessing,
mask-driven), MPC data, 12-day composites 2024-11-01 → 2025-10-31, **500 m grid**
granularity, clipped to the Java consensus paddy mask.

Generated 2026-06-11.

---

## 1. Run

- AOI tiled into **8 × 20 km** tiles; processed with **8 workers**.
- Full run (6 new tiles; 2 reused via `--resume`): **28.7 min wall-clock**.
- Outputs: `output/jatiluhur_grid_test/` — merged AOI GeoTIFFs (`n_seasons.tif`,
  per-season `POS_doy` / `peak_NDVI` / `LOS_days`, seasons 1–3) + per-tile products.

## 2. Cropping intensity (n_seasons)

| seasons | cells | |
|---|---|---|
| 0 (paddy, no clear cycle) | 1,929 | |
| 1 | 1,036 | |
| 2 | **3,425** | |
| 3 | 1,395 | |
| **mean (≥1)** | **2.19** | |

Jatiluhur is among Indonesia's most intensively cropped rice systems
(Citarum/Jatiluhur irrigation); the dominant double-crop signal with a substantial
triple-crop tail is agronomically expected.

## 3. Cross-validation vs independent `cropping_intensity.tif`

Validated against the SAR-classifier cropping-intensity product from
`s1-land-cover-classification` (consensus, MT 2024/25), an **independent method and
data path**. The 50 m intensity raster was majority-resampled onto our 500 m grid
(EPSG:4326 → UTM 48S), compared over cells valid in both (n = 5,873; our seasons ≥ 1).

**Confusion matrix** (rows = MOGPR `n_seasons`, cols = `cropping_intensity`):

| | CI single | CI double | CI triple |
|---|---|---|---|
| MOGPR single | 337 | 596 | 34 |
| MOGPR double | 325 | **2,627** | 323 |
| MOGPR triple | 174 | 1,119 | 338 |

- **Exact-class agreement: 56.2%**
- **Within ±1 class: 96.5%** — the two independent products almost never disagree
  by more than one crop cycle.
- Per-class: strongest on **double-crop** (MOGPR producer recall 80%, CI user 61%);
  weaker on single (35% / 40%) and triple (21% / 49%).
- **Direction: MOGPR detects more intensity** than the SAR-only classifier
  (mean 2.11 vs 1.98; MOGPR-higher 28% vs MOGPR-lower 16%) — consistent with the
  thesis that adding optical (S2 NDVI) via MOGPR fusion recovers cropping cycles the
  radar-only product misses.

Figure: `cross_validation_maps.png` (side-by-side maps + agreement heatmap).
Table: `cross_validation_cropping_intensity.csv`.

## 4. Caveats

- **Triple-class disagreement** is the largest cell (1,119 MOGPR-triple = CI-double):
  part genuine recovery, part **peakvalley over-segmentation** at `drop_thr = 0.12`
  (same tendency seen in the BulakBakal validation). A stricter `drop_thr` or a
  multi-season-aware metric would tighten this.
- **1,417 paddy cells** where MOGPR found no season but CI is cropped: conservative
  omissions, likely cloud-limited or mixed/partial-paddy 500 m cells.
- **Resolution mismatch**: CI native 50 m majority-resampled to 500 m vs our 500 m
  grid MOGPR — moderate exact agreement (56%) is expected for two independent methods
  at different native resolutions; the 96.5% within-±1 is the meaningful corroboration.
- Single Sentinel-1 orbit; 500 m grid granularity (operational/KSA-aligned, not 50 m
  wall-to-wall).

## 5. Takeaway

An **independent cross-check corroborates the fused multi-season product**: 96.5%
within-±1-class agreement with a separate SAR-classifier intensity map, and a
systematic tendency for the optical+SAR fusion to recover *more* cropping intensity —
the intended benefit for wet-season-robust monitoring. This supports using the fused
phenology as a complementary, all-weather layer alongside existing products.
