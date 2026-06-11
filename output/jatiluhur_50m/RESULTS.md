# Jatiluhur — 50 m Wall-to-Wall Rice Phenology (S1+S2 MOGPR), with Cross-Validation

Study area: **Jatiluhur paddy command area, West Java** (UTM 48S / EPSG:32748).
Produced with `scripts/scale_runner.py` (tiled, resumable, multiprocessing,
mask-driven), MPC data, 12-day composites 2024-11-01 → 2025-10-31,
**50 m per-pixel** granularity, clipped to the consensus paddy mask
(`cropping_intensity_consensus_mt2024_25/paddy_mask.tif`).

Companion to the 500 m run in `output/jatiluhur/RESULTS.md` — same AOI, mask, dates,
and `drop_thr 0.12`; the **only change is spatial granularity** (`grid 500 m` → `pixel 50 m`).

Generated 2026-06-11 (HPC, 224-core box).

---

## 1. Run

- AOI tiled into **24 × 10 km** tiles; processed with **24 workers**, BLAS pinned to 1 thread.
- Full run (no cube cache on this fresh box → every tile downloaded): **94.2 min wall-clock**.
- Outputs: `output/jatiluhur_50m/` — merged AOI GeoTIFFs (`n_seasons.tif`, per-season
  `POS_doy` / `peak_NDVI` / `LOS_days`, seasons 1–3) + per-tile products.
- Grid: 550 × 1689 px @ 50 m; **565,265** valid px, **451,426** paddy px with ≥1 cycle
  (~55× the 8,151 cells of the 500 m run, as expected: a 500 m cell = 100 × 50 m px).

## 2. Cropping intensity (n_seasons) — raw, per-pixel

| seasons | px | |
|---|---|---|
| 0 (paddy, no clear cycle) | 113,839 | |
| 1 | 53,236 | |
| 2 | **158,821** | |
| 3 | 123,076 | |
| 4 | 72,161 | |
| 5 | 31,245 | |
| 6 | 10,063 | |
| 7 | 2,408 | |
| 8–10 | 416 | |
| **mean (≥1)** | **2.80** | |

Double-crop remains the mode, but **a long tail of n > 3 appears that did not exist at
500 m**: **116,293 px (25.8% of paddy) report > 3 seasons** (up to 10). This is **per-pixel
peakvalley over-segmentation** at `drop_thr 0.12` — single 50 m pixels are noisier than the
spatially-averaged 500 m cells, so small NDVI/SAR wiggles are read as extra cycles. The 500 m
grid's coarsening suppressed this; 50 m exposes it. **This is the central finding of the 50 m run.**

## 3. Cross-validation vs independent `cropping_intensity.tif`

Validated against the SAR-classifier cropping-intensity product from
`s1-land-cover-classification` (consensus, MT 2024/25) — an **independent method and data
path**. The 50 m reference (EPSG:4326) was reprojected onto our 50 m UTM 48S grid (nearest /
majority; resolutions match, so no down-resampling). Compared over cells valid in both
(prediction ≥ 1 season and reference in {1,2,3}), **n = 451,426**. For the class comparison
our `n_seasons` is **clipped to [1,3]** (the reference only encodes single/double/triple).

**Confusion matrix** (rows = MOGPR `n_seasons` clipped, cols = `cropping_intensity`):

| | CI single | CI double | CI triple |
|---|---|---|---|
| MOGPR single | 15,977 | 32,263 | 4,996 |
| MOGPR double | 17,757 | **117,483** | 23,581 |
| MOGPR triple | 24,222 | 169,108 | 46,039 |

- **Exact-class agreement: 39.8%** (500 m: 56.2%).
- **Within ±1 class: 93.5%** (500 m: 96.5%) — the two independent products still rarely
  differ by more than one crop cycle.
- Per-class (MOGPR producer recall / CI user precision): double **74% / 37%**,
  single **30% / 28%**, triple **19% / 62%**.
- **Direction unchanged: MOGPR detects more intensity** than the SAR-only classifier
  (mean clipped 2.41 vs 2.04; MOGPR-higher 46.8% vs MOGPR-lower 13.5%) — the optical-fusion
  payoff persists at 50 m.

Figure: `cross_validation_maps.png` (MOGPR vs CI maps + row-normalised agreement heatmap).
Tables: `cross_validation_cropping_intensity.csv`, `cross_validation_metrics.json`.

## 4. Interpretation — 50 m vs 500 m

- **Spatial detail gained:** field-level phenology, no 500 m smearing across plot
  boundaries. Useful where paddy parcels are smaller than 500 m.
- **Per-pixel noise cost:** exact-class agreement falls 56% → 40%, driven almost entirely
  by the **n > 3 over-segmentation tail** (25.8% of paddy). When MOGPR triple is clipped
  from a raw 4–10, much of the "MOGPR triple → CI double" mass (169k px) is spurious extra
  cycles, not real triple-cropping.
- **`within-1` holds (93.5%)** → the disagreement is one-class over-counting, not random.
- **Recommendation:** 50 m wall-to-wall is viable for *mapping* (POS/peak/LOS), but the
  **`n_seasons` count needs a noise-robust detector at pixel scale** — raise `drop_thr`,
  add a minimum inter-peak separation / amplitude floor, or a light spatial/temporal
  smooth before peakvalley. This sharpens the WORKLOG §10 `drop_thr` item: it is now
  *required* for credible 50 m cropping-intensity, not just a refinement.

## 5. Reproduce

```bash
conda activate fusets
python scripts/scale_runner.py \
  --aoi  <.../jatiluhur_petak_4326.gpkg> \
  --mask <.../cropping_intensity_consensus_mt2024_25/paddy_mask.tif> \
  --crs EPSG:32748 --start 2024-11-01 --end 2025-10-31 --period-days 12 \
  --res 50 --granularity pixel --tile-km 10 \
  --drop-thr 0.12 --max-seasons 3 --workers 24 --resume \
  --outdir output/jatiluhur_50m

python scripts/cross_validate_cropping_intensity.py \
  --pred output/jatiluhur_50m/n_seasons.tif \
  --ref  <.../cropping_intensity_consensus_mt2024_25/cropping_intensity.tif> \
  --outdir output/jatiluhur_50m
```
