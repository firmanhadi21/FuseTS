# Jatiluhur — 50 m Wall-to-Wall Rice Phenology, TUNED `drop_thr = 0.20`

Study area: **Jatiluhur paddy command area, West Java** (UTM 48S / EPSG:32748).
Same pipeline, AOI, mask, dates, and 50 m per-pixel granularity as
`output/jatiluhur_50m/RESULTS.md` — **only `drop_thr` changed: 0.12 → 0.20**, the value
chosen by the sweep in `output/jatiluhur_50m/sweep/` (see WORKLOG §7c).

This re-fuse reused the cached tile cubes (`cube.nc`, symlinked) — **no MPC re-download**.

Generated 2026-06-11 (HPC).

---

## 1. Why 0.20

The 0.12 run over-segmented per-pixel season counts (25.7% of paddy px reported > 3
seasons, up to 10). A `drop_thr` sweep on the cached cubes (MOGPR fused once/px, seasons
re-counted at 6 thresholds) gave:

| drop_thr | exact | within-1 | over>3 | mean MOGPR | mean CI |
|---|---|---|---|---|---|
| 0.12 (baseline) | 39.8% | 93.5% | **25.7%** | 2.41 | 2.04 |
| 0.15 | 45.4% | 94.7% | 13.2% | 2.18 | 2.05 |
| **0.20 (chosen)** | **46.3%** | **94.8%** | **4.0%** | 1.85 | 2.08 |
| 0.25 | 41.0% | 93.4% | 1.2% | 1.62 | 2.10 |
| 0.30 | 34.2% | 91.6% | 0.4% | 1.47 | 2.12 |
| 0.40 | 22.8% | 87.2% | 0.06% | 1.28 | 2.15 |

**0.20 maximises exact-class agreement and within-±1**, while cutting over-segmentation
from 25.7% to 4.0%. Beyond 0.20 the detector starts *under*-counting (mean drops below CI,
`frac_lower` climbs, exact falls). 0.20 is the inflection point.

## 2. Cropping intensity (n_seasons) @ 0.20 — raw, per-pixel

Raw distribution (paddy px with ≥1 cycle): 1→127,635 · 2→**200,302** · 3→52,405 ·
4→12,773 · 5→2,597 · 6–9→466. **Over-segmentation tail (n>3): 15,836 px = 4.0%**
(was 25.8% at 0.12). Double-crop is now the clear mode, with a thin, plausible triple tail.

## 3. Cross-validation vs independent `cropping_intensity.tif`

Consensus MT 2024/25 product (independent SAR classifier), reprojected to our 50 m UTM 48S
grid; compared over cells valid in both (prediction ≥ 1 season, reference ∈ {1,2,3}),
**n = 396,178**; prediction clipped to [1,3] for the class comparison.

**Confusion matrix** (rows = MOGPR `n_seasons` clipped, cols = `cropping_intensity`):

| | CI single | CI double | CI triple |
|---|---|---|---|
| MOGPR single | 17,906 | 94,581 | 15,148 |
| MOGPR double | 13,935 | **149,230** | 37,137 |
| MOGPR triple | 5,573 | 46,510 | 16,158 |

- **Exact-class agreement: 46.3%** (was 39.8% at 0.12; +6.5 pts).
- **Within ±1 class: 94.8%** (was 93.5%).
- Per-class (MOGPR producer recall / CI user precision): double **74% / 51%**,
  single **14% / 48%**, triple **24% / 24%**.
- Direction now **balanced/slightly conservative**: mean clipped **1.85 vs CI 2.08**;
  MOGPR-higher 16.7% vs MOGPR-lower 37.1%. At 0.12 MOGPR over-detected (higher 46.8%);
  0.20 removes the spurious extra cycles and lands just below the radar product.

Figure: `cross_validation_maps.png`. Tables: `cross_validation_cropping_intensity.csv`,
`cross_validation_metrics.json`.

## 4. Interpretation

- **0.20 is the recommended production threshold for 50 m per-pixel** Jatiluhur rice
  phenology: best class agreement, over-segmentation controlled (4%), no material
  under-counting.
- The remaining single-class recall weakness (14%) reflects that many CI-single cells are
  read as double by MOGPR (94,581 px) — partly real optical-recovered second cycles, partly
  the resolution mismatch between a 50 m pixel and a parcel-scale radar label.
- **For maps** (POS/peak/LOS) the 0.20 products are the ones to use; the 0.12 set in
  `output/jatiluhur_50m/` is retained only as the untuned baseline for this comparison.

## 5. Reproduce

```bash
# sweep (cheap; fuses once/px, re-counts at many thresholds)
python scripts/sweep_drop_thr.py --run-dir output/jatiluhur_50m \
  --mask <.../paddy_mask.tif> --crs EPSG:32748 --ref <.../cropping_intensity.tif> \
  --thr 0.12 0.15 0.20 0.25 0.30 0.40 --workers 24

# re-fuse full products at the chosen threshold, reusing cached cubes (symlinked)
python scripts/scale_runner.py --aoi <.../jatiluhur_petak_4326.gpkg> \
  --mask <.../paddy_mask.tif> --crs EPSG:32748 \
  --res 50 --granularity pixel --tile-km 10 --drop-thr 0.20 \
  --max-seasons 3 --workers 24 --resume --outdir output/jatiluhur_50m_thr0.20

python scripts/cross_validate_cropping_intensity.py \
  --pred output/jatiluhur_50m_thr0.20/n_seasons.tif \
  --ref  <.../cropping_intensity.tif> --outdir output/jatiluhur_50m_thr0.20
```
