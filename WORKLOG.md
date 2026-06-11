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

## 7. Jatiluhur results (West Java, UTM 48S)

- 2-tile validation (500 m grid): merged `n_seasons` over 2,767 cells →
  **mean 1.96 seasons/cell** (0:516, 1:547, 2:1,322, 3:318) — more intensive
  double/triple-cropping than Klambu, as expected for the Citarum/Jatiluhur system.
- **Full 8-tile run IN PROGRESS** (8 workers; 2 tiles skipped via resume) →
  `output/jatiluhur_grid_test/` AOI-wide GeoTIFFs.

## 8. HPC notes

- User's HPC = **JupyterHub box with sudo** → effectively one **2× H100** workstation
  (per repo's `H100_SETUP_GUIDE.md`), internet available. **No SLURM needed.**
- **GPU is heavily contended / deprioritised** → run **CPU-only** (the whole pipeline
  already is). Scale via **cores + granularity + resume**, not GPU.
- Recipe: `tmux` → `conda activate fusets` → `scale_runner.py … --workers 8 --resume`.
  Use **conda, not `apt`**, for the Python geo stack (avoids PROJ/GDAL conflicts).
- For 50 m all-Java: tiled `pixel` granularity, resumable, run over time; cache makes
  re-fusion cheap. GPU (`mogpr_gpu`) optional if a card frees up.

## 9. Git (fork `firmanhadi21/FuseTS`, branch `main`)

Pushed (code + docs only; secrets, `*.nc`, and restricted field data git-ignored):
- `4e776d2` — pipeline scripts, paper draft, scaling plan, Klambu results + figures
- `21153e6` — `scale_runner.py` + `--mask`
- `60d329e` — scale_runner two-phase cube cache

**Excluded by design:** `ee-geodetic.json`, `settings.env` (secrets — never tracked),
datacubes (`*.nc`), `Validasi_Data_BulakBakal/` and `output/bulak_bakal*/` (third-party
field data — pending publish permission). A staged-file secret-scan guard runs before each commit.

## 10. Open items / next steps

- Finish full Jatiluhur run; cross-validate `n_seasons` vs `cropping_intensity.tif`.
- Rentang run (needs an AOI boundary — not present in either repo).
- Province / all-Java at 500 m grid (Mac or HPC); 50 m wall-to-wall on HPC.
- Optional: decide on publishing BulakBakal data; sync Demak notebooks if wanted.
- Paper: add figures (n_seasons comparison, validation scatter), tighten to a venue.
