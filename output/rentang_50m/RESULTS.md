# Rentang — 50 m Wall-to-Wall Rice Phenology (S1+S2 MOGPR), `drop_thr = 0.20`

Study area: **D.I. Rentang irrigation command area, West Java** (Indramayu/Cirebon,
Cimanuk system) — UTM 49S / EPSG:32749. Same pipeline as Jatiluhur
(`scripts/scale_runner.py`, MPC, 12-day composites 2024-11-01 → 2025-10-31, 50 m per-pixel,
consensus paddy mask), using the tuned production threshold **`drop_thr = 0.20`** from the
start (see WORKLOG §7c).

AOI boundary: feature `D.I. Rentang` (kode_di `0000000182`, Pusat) extracted from the
national irrigation layer `idmai/.../VECTOR_IRRIGATION/Daerah_Irigasi_Web_IDMAI.shp` →
`data/aois/rentang_di_4326.gpkg` (865 km²).

Generated 2026-06-11 (HPC).

---

## 1. Run

- AOI → **20 × 10 km** tiles, **20 workers**, BLAS pinned to 1. Fresh AOI (no cache) →
  all tiles downloaded. Wall-clock ~96 min (12:57 → 14:34).
- Outputs: `output/rentang_50m/` — merged AOI GeoTIFFs (`n_seasons` + per-season
  `POS_doy`/`peak_NDVI`/`LOS_days`, seasons 1–3) + per-tile products.
- **362,206 paddy px with ≥1 cycle**; mean raw n_seasons (≥1) = **2.38** — high, as
  expected for the Cimanuk/Indramayu rice bowl (one of Indonesia's most intensive systems).

## 2. Cropping intensity (n_seasons) — raw, per-pixel

Raw distribution (≥1 cycle): 1→69,552 · 2→**154,259** · 3→86,850 · 4→37,782 · 5→11,122 ·
6→2,278 · 7–9→363. **Over-segmentation tail (n>3): 51,545 px = 14.2%.**

Double-crop is the mode with a strong triple tail — but the n>3 tail is **larger than
Jatiluhur's 4.0% at the same `drop_thr 0.20`** (see §4).

## 3. Cross-validation vs independent `cropping_intensity.tif`

Consensus MT 2024/25 product (independent SAR classifier), reprojected to our 50 m UTM 49S
grid; compared over cells valid in both (prediction ≥ 1, reference ∈ {1,2,3}),
**n = 362,206**; prediction clipped to [1,3].

**Confusion matrix** (rows = MOGPR `n_seasons` clipped, cols = `cropping_intensity`):

| | CI single | CI double | CI triple |
|---|---|---|---|
| MOGPR single | 17,706 | 27,424 | 24,422 |
| MOGPR double | 13,789 | **66,785** | 73,685 |
| MOGPR triple | 10,188 | 58,601 | **69,606** |

- **Exact-class agreement: 42.5%** (Jatiluhur @0.20: 46.3%).
- **Within ±1 class: 90.4%** (Jatiluhur: 94.8%).
- Per-class recall (MOGPR producer): single 25%, double 43%, **triple 50%** — best on
  triple here, reflecting the genuinely intensive system.
- Direction: mean clipped **2.19 vs CI 2.35**; MOGPR-higher 22.8% / lower 34.7% — MOGPR
  reads **slightly lower** intensity than the radar product (opposite of the over-detection
  seen at Jatiluhur 0.12), i.e. 0.20 is mildly conservative for this AOI.

Figure: `cross_validation_maps.png`. Tables: `cross_validation_cropping_intensity.csv`,
`cross_validation_metrics.json`.

## 4. Interpretation — `drop_thr` may need per-AOI calibration

The Jatiluhur-tuned `drop_thr = 0.20` **transfers only partially** to Rentang:

| | Jatiluhur @0.20 | Rentang @0.20 |
|---|---|---|
| exact | 46.3% | 42.5% |
| within-1 | 94.8% | 90.4% |
| over>3 | 4.0% | **14.2%** |
| mean MOGPR vs CI | 1.85 vs 2.08 | 2.19 vs 2.35 |

- Rentang is **more intensive** (mean CI 2.35 vs 2.08), so part of the larger n>3 tail is
  **real triple-crop**, not just per-pixel noise — consistent with the higher triple recall
  (50%) and the lower-not-higher mean vs CI.
- But 14.2% > 3 still indicates residual over-segmentation. A Rentang-specific sweep would
  likely land slightly higher than 0.20. **Takeaway: `drop_thr` is not perfectly portable
  across AOIs of different cropping intensity** — for production, run the cheap
  `sweep_drop_thr.py` per AOI (fuses once, re-counts at many thresholds) and pick locally.
- For mapping (POS/peak/LOS), the 0.20 products here are usable; treat the n>3 count with
  the caveat above.

## 5. Reproduce

```bash
# extract AOI (already done -> data/aois/rentang_di_4326.gpkg)
python scripts/scale_runner.py --aoi data/aois/rentang_di_4326.gpkg \
  --mask <.../cropping_intensity_consensus_mt2024_25/paddy_mask.tif> --crs EPSG:32749 \
  --res 50 --granularity pixel --tile-km 10 --drop-thr 0.20 \
  --max-seasons 3 --workers 20 --resume --outdir output/rentang_50m

python scripts/cross_validate_cropping_intensity.py \
  --pred output/rentang_50m/n_seasons.tif \
  --ref  <.../cropping_intensity.tif> --outdir output/rentang_50m
```
