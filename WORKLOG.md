# Worklog — S1+S2 MOGPR Rice Phenology (Klambu–Glapan → Jatiluhur → Java)

A running record of what has been built and found. Companion to
`output/klambu_glapan/RESULTS.md`, `SCALING_PLAN.md`, and
`PAPER_S1S2_MOGPR_Rice_Phenology.md`.

Last updated: 2026-06-11.

---

## 1. Environment (macOS, Apple Silicon)

- Installed **Miniforge** (`~/miniforge3`) + the **`fusets` conda env** (Python 3.11)
  from `environment.yml`: FuseTS (editable), GPy, `vam.whittaker`, numpy<2, xarray,
  rioxarray, rasterio, geopandas, **pystac-client + planetary-computer + odc-stac**,
  JupyterLab. `openeo`/`lcmap-pyccd` deliberately omitted (not needed for local MOGPR).
- Fixed a PROJ/GDAL conflict from a system **OTB 8.1.2** install via conda
  **activate/deactivate hooks** (point PROJ/GDAL at the env, neutralise OTB's PYTHONPATH).
- `conda init zsh` done; `conda activate fusets` works in new shells.
- Verified end-to-end: live MPC download + MOGPR fusion.

## 2. Pipeline scripts (`scripts/`)

| Script | Role |
|---|---|
| `klambu_glapan_mogpr.py` | MPC extraction → 12-day S1+S2 datacube → MOGPR fusion → phenology. Flags: `--points-shp` (field validation), `--mask` (paddy clip), `--crs`, dates, `--run-raster`. |
| `pv_phenology.py` | Multi-season **peakvalley** phenology (point + wall-to-wall raster). |
| `raster_mogpr_phenology.py` | Tiled wall-to-wall MOGPR fusion on a (downsampled) cube. |
| `field_zonal_mogpr.py` | Field-polygon-mean fusion from an existing cube. |
| `map_phenology.py` | Phenology maps / GeoTIFFs. |
| `scale_runner.py` | **Tiled · resumable · multiprocessing · cached** orchestrator for large AOIs. |
| `extract_sits_klambu_glapan_fixed.R` | Fixed (s1_cube_reg bug, portable paths, S1 regularize). |

## 3. Klambu–Glapan (primary demo AOI, Central Java)

- Datacube: 31 × 12-day composites (2024-11-07 → 2025-11-02), 50 m, UTM 49S,
  886 × 1162 (~1.03 M px/period); bands `S2ndvi`, `VV`, `VH` (dB).
- **Point MOGPR (200 pts):** mean **1.83 seasons/point** (double-rice). LOS ≈ 102 d,
  peak NDVI ≈ 0.36.
- **Fusion effect (controlled, 200 m, drop_thr 0.12 — only input series differs):**

  | input | double-cropped / cropped | mean seasons (≥1) |
  |---|---|---|
  | Whittaker-only (optical) | **13%** | 1.14 |
  | MOGPR-fused (S1+S2) | **50%** | 1.72 |
  | point-based MOGPR (ref) | — | 1.83 |

  → SAR–optical fusion ~quadruples detected double-cropping; converges to the point
  reference. Figure: `output/klambu_glapan/n_seasons_compare_whittaker_vs_mogpr.png`.
- Wall-to-wall 200 m MOGPR-fused cube + GeoTIFFs produced (raster MOGPR ≈ 9 px/s CPU;
  full 50 m ≈ 32 h → HPC job).
- Switched phenology to **FuseTS peakvalley** (multi-season) per request — the earlier
  single-season phenolopy collapsed the two rice cycles.

## 4. BulakBakal ground validation (31 sawah, near Magelang)

- 10 m, Sep 2025 → Jun 2026, sampled at field centroids (carrying `Tgl_Tanam`).
  Severe wet-season cloud (several 0%-NDVI periods) recovered via SAR.
- **Detected green-up vs recorded transplanting (best = centroid, drop_thr 0.08):**
  31/31 fields matched, **SOS median +9 d**, **MAE 21.3 d**, **81% within ±24 d**;
  peak ≈ 81 d after planting (agronomically correct).
- Field-polygon-mean did **not** help (small plots dilute the signal). Caveat:
  drop_thr 0.08 over-segments, so the nearest-season MAE is somewhat optimistic.
- Output: `output/bulak_bakal/validation_vs_Tgl_Tanam.csv` (git-ignored — restricted data).

## 5. Paper draft

- `PAPER_S1S2_MOGPR_Rice_Phenology.md` — full manuscript: background (Indonesia rice
  self-sufficiency, KSA, Katam, cloud problem), gap, methods, results (grounded in the
  numbers above), discussion of **benefit for the Indonesian government**, conclusions.
- **References integrity:** 33 refs, every DOI independently verified against the
  **Crossref API** (titles/authors/years/venues + 16 page-ranges spot-checked). Dropped
  unverifiable items; corrected agent metadata (Pipia 2019 exact title; no "Pipia 2022";
  FuseTS cited as software, no peer-reviewed paper).

## 6. Scaling (Klambu → Jatiluhur → Java)

- **Companion repo** `~/Github/s1-land-cover-classification` provides a **Java paddy mask**
  (~9.2–12.1 M paddy px ≈ 23,000–30,000 km²), ready AOIs (Jatiluhur, Klambu), a 500 m grid,
  and an independent `cropping_intensity.tif` (external cross-validation candidate).
- **Mask-driven principle:** fuse only rice → ~1000× reduction vs all-Java.
- **`scale_runner.py`** (built + validated): tiling, `--resume` (per-tile `.done`,
  skip finished), `--workers` multiprocessing (no GPU/SLURM), `--mask` (window-read,
  0.3 s), granularity `grid|pixel|parcel`, tile merge, and a **two-phase cube cache**
  (`cube.nc`) so the full-year download is paid once.
- **Validated:** dry-run/tiling; `fuse_raster` core (mean 1.73 seasons); resume skip
  (0.000 s on a done tile); cache (`ok:extracted` → `ok:cached`, download skipped);
  2-tile/2-worker real run + merge.
- **Compute reality:** per-tile cost is **74–128 min, dominated by MPC download** (fusion
  of ~1.5 k cells is minutes). Full Jatiluhur (8 tiles) ≈ 4–5 h at 2 workers / faster at 8.

## 7. Jatiluhur results (West Java, UTM 48S) — COMPLETE

- **Full 8-tile run done:** 500 m grid, paddy-masked, **8 workers, 28.7 min**
  wall-clock (6 new tiles; 2 skipped via `--resume`). Outputs:
  `output/jatiluhur_grid_test/` (merged AOI GeoTIFFs) + `output/jatiluhur/`.
- **Cropping intensity:** 8,151 cells, **mean 2.19 seasons/cell**
  (0:1929, 1:1036, 2:3425, 3:1395) — Indonesia's most intensive rice belt
  (Citarum/Jatiluhur), as expected.
- **Cross-validation vs independent `cropping_intensity.tif`** (SAR classifier,
  s1-land-cover repo; 50 m → 500 m majority-resampled; n = 5,873 cells):
  **56.2% exact-class**, **96.5% within ±1 class**. Strongest on double-crop
  (80% recall). **MOGPR detects more intensity** than the radar-only product
  (mean 2.11 vs 1.98; higher 28% vs lower 16%) — the optical-fusion payoff.
  See `output/jatiluhur/RESULTS.md`, `cross_validation_maps.png`, `cross_validation_cropping_intensity.csv`.
- Caveat: triple-class is where the two diverge most (partly real recovery, partly
  peakvalley over-segmentation at `drop_thr 0.12`); 1,417 conservative omissions.

## 7b. Jatiluhur 50 m wall-to-wall (West Java, UTM 48S) — COMPLETE (HPC)

- **First HPC run.** Same AOI/mask/dates/`drop_thr` as §7; only granularity changed
  (`grid 500 m` → `pixel 50 m`). 24 × 10 km tiles, **24 workers**, BLAS pinned to 1.
  No cube cache on the fresh box → all tiles downloaded: **94.2 min wall-clock** for
  the full AOI. Merged GeoTIFFs in `output/jatiluhur_50m/`.
- **Scale:** 550 × 1689 px @ 50 m; **451,426 paddy px** with ≥1 cycle (~55× the 8,151
  cells at 500 m). Mean n_seasons (≥1) = **2.80** raw.
- **Headline finding — per-pixel over-segmentation:** **25.8% of paddy px report > 3
  seasons** (up to 10), a tail that did **not** exist at 500 m. Single 50 m pixels are
  noisier than averaged 500 m cells, so peakvalley at `drop_thr 0.12` reads small wiggles
  as extra cycles.
- **Cross-validation vs `cropping_intensity.tif`** (consensus MT 2024/25; reprojected to
  50 m UTM 48S, nearest; n = 451,426): **39.8% exact** (was 56.2% at 500 m), **93.5%
  within ±1** (was 96.5%). Direction holds — **MOGPR detects more** (mean clipped 2.41 vs
  2.04; higher 46.8% vs lower 13.5%). The exact-class drop is driven entirely by the n>3
  tail, not random disagreement (within-1 stays high).
- **Takeaway:** 50 m is viable for *mapping* (POS/peak/LOS) but the **`n_seasons` count
  needs a noise-robust detector at pixel scale** (raise `drop_thr`, add min peak
  separation/amplitude, or light pre-smooth). Sharpens §10 — now *required* for credible
  50 m cropping-intensity. See `output/jatiluhur_50m/RESULTS.md`.
- **New reusable tool:** `scripts/cross_validate_cropping_intensity.py` (confusion matrix,
  exact/within-1, per-class, direction, raw-distribution incl. over-segmentation, maps) —
  the 500 m validation was ad-hoc; this makes it repeatable for any AOI.

## 7c. drop_thr tuning → 0.20 (HPC) — COMPLETE

- **Sweep** (`scripts/sweep_drop_thr.py`): re-counts seasons at 6 thresholds on the cached
  50 m tile cubes. Key trick — MOGPR (`mogpr_1D`) is independent of `drop_thr`, so each px
  is **fused once** and seasons re-counted per threshold; the whole sweep ≈ one re-fusion.
  Output `output/jatiluhur_50m/sweep/sweep_summary.csv`.

  | drop_thr | exact | within-1 | over>3 | mean MOGPR | mean CI |
  |---|---|---|---|---|---|
  | 0.12 (baseline) | 39.8% | 93.5% | **25.7%** | 2.41 | 2.04 |
  | 0.15 | 45.4% | 94.7% | 13.2% | 2.18 | 2.05 |
  | **0.20 (chosen)** | **46.3%** | **94.8%** | **4.0%** | 1.85 | 2.08 |
  | 0.25 | 41.0% | 93.4% | 1.2% | 1.62 | 2.10 |
  | 0.30 | 34.2% | 91.6% | 0.4% | 1.47 | 2.12 |
  | 0.40 | 22.8% | 87.2% | 0.06% | 1.28 | 2.15 |

  **0.20 maximises exact-class + within-±1 and cuts over-segmentation 25.7%→4.0%**; above it
  the detector under-counts (mean < CI, exact falls). 0.20 = inflection point.
- **Full re-fuse at 0.20** → `output/jatiluhur_50m_thr0.20/` (separate dir; 0.12 baseline
  kept). Reused cached `cube.nc` via symlinks — **no MPC re-download**. 24/24 tiles, all 10
  per-season products merged. Cross-val (n = 396,178): **46.3% exact, 94.8% within-1, 4.0%
  over-seg**; direction now slightly conservative (mean 1.85 vs CI 2.08; higher 16.7% /
  lower 37.1%) — the spurious extra cycles are gone. See
  `output/jatiluhur_50m_thr0.20/RESULTS.md`.
- **0.20 is the production threshold for 50 m per-pixel** going forward (incl. Rentang).

## 7d. Rentang (West Java, Indramayu/Cirebon, UTM 49S) — COMPLETE (HPC)

- **AOI sourced** from the national irrigation layer in the **idmai** repo
  (`idmai/00_PROD/INPUT_DATA/SHP/VECTOR_IRRIGATION/Daerah_Irigasi_Web_IDMAI.shp`, 21,018
  features); extracted `nama_di='D.I. Rentang'` (kode_di 0000000182, Pusat) →
  `data/aois/rentang_di_4326.gpkg` (865 km², EPSG:32749). The §10 blocker is resolved.
- **Run:** 50 m pixel, **`drop_thr 0.20`**, 20×10 km tiles / 20 workers, ~96 min. 20/20
  tiles, all per-season products merged. 362,206 paddy px, mean raw n_seasons 2.38
  (very intensive — Cimanuk rice bowl).
- **Cross-val** (n=362,206): **42.5% exact, 90.4% within-1**, mean 2.19 vs CI 2.35
  (MOGPR slightly conservative). Over-seg n>3 = **14.2%**.
- **Key finding — `drop_thr` is not fully portable:** 0.20 gave 4.0% over-seg at Jatiluhur
  but 14.2% at Rentang. Partly real (Rentang more intensive: CI 2.35 vs 2.08; triple recall
  50%), partly residual over-segmentation. **Recommend per-AOI sweep** (`sweep_drop_thr.py`
  is cheap) rather than one global threshold. See `output/rentang_50m/RESULTS.md`.

## 8. HPC notes

- User's HPC = **JupyterHub box with sudo** → effectively one **2× H100** workstation
  (per repo's `H100_SETUP_GUIDE.md`), internet available. **No SLURM needed.**
- **GPU is heavily contended / deprioritised** → run **CPU-only** (the whole pipeline
  already is). Scale via **cores + granularity + resume**, not GPU.
- Recipe: `tmux` → `conda activate fusets` → `scale_runner.py … --workers 8 --resume`.
  Use **conda, not `apt`**, for the Python geo stack (avoids PROJ/GDAL conflicts).
- For 50 m all-Java: tiled `pixel` granularity, resumable, run over time; cache makes
  re-fusion cheap. GPU (`mogpr_gpu`) optional if a card frees up.
- **Env actually built on the box (2026-06-11):** `/opt/conda` (conda 24.9 + mamba 1.5),
  **224 cores**, no GPU used. `mamba env create -f environment.yml` → `fusets`, then
  `pip install -e . --no-deps`. MPC reachable. **Caveat:** the prebuilt
  `vam.whittaker==2.0.6` wheel fails on this glibc (`undefined symbol: __log_finite`) and
  a `--no-binary` rebuild also failed — **not needed for the MOGPR/peakvalley path**
  (`scale_runner` imports only `fusets.mogpr` + peakvalley), so it does not block fusion
  runs; only the optical-only Whittaker comparison needs it. Worker sizing: set
  `--workers ≈ tile count` and pin BLAS to 1 thread (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`)
  to avoid oversubscription, since fusion is single-process per tile.

## 9. Git (fork `firmanhadi21/FuseTS`, branch `main`)

Pushed (secrets + `*.nc` datacubes always excluded; a staged-file secret-scan guard
runs before each commit):
- `4e776d2` — pipeline scripts, paper draft, scaling plan, Klambu results + figures
- `21153e6` — `scale_runner.py` + `--mask`
- `60d329e` — scale_runner two-phase cube cache
- `fd264db` — WORKLOG.md
- `298fc1c` — output data **+ BulakBakal ground-truth made public** (per maintainer
  authorization; repo is PUBLIC). Datacubes still excluded.
- `082d91d` — Jatiluhur scaled run + cross-validation vs cropping_intensity

**Always excluded:** `ee-geodetic.json`, `settings.env` (secrets — never tracked),
datacubes (`*.nc`, incl. tile `cube.nc` caches), throwaway test dirs (smoke, cache_test).
**Now public (was restricted):** BulakBakal field survey + validation outputs.

## 10. Open items / next steps

- ✅ Full Jatiluhur run + cross-validation vs `cropping_intensity.tif` — done.
- ✅ BulakBakal data published (public).
- ✅ **Jatiluhur 50 m wall-to-wall on HPC** (§7b) — done; exposed per-pixel
  over-segmentation (25.8% px > 3 seasons).
- ✅ **`drop_thr` tuned → 0.20** (§7c): sweep on cached cubes, full re-fuse, cross-val.
  Exact 39.8%→46.3%, over-seg 25.7%→4.0%. **0.20 is the production threshold for 50 m.**
- ✅ **Rentang run done** (§7d): AOI found in idmai repo; 50 m @ 0.20, 42.5% exact,
  14.2% over-seg → exposed that `drop_thr` needs **per-AOI** calibration.
- **Per-AOI `drop_thr` sweep** is now the recommended workflow before each new AOI's
  production run (sweep is cheap — fuses once, re-counts many thresholds).
- Province / all-Java at 500 m grid; 50 m wall-to-wall for more AOIs on HPC (224 cores).
- Paper: fold in the Jatiluhur 500 m **and 50 m** cross-validation results (incl. the
  resolution/over-segmentation trade-off); add figures; tighten to a venue.
- Optional: sync Demak notebooks if wanted; fix `vam.whittaker` on HPC if the optical-only
  Whittaker comparison is needed there.

Last updated: 2026-06-11 (after Jatiluhur 50 m + drop_thr→0.20 + Rentang 50 m, on HPC).
