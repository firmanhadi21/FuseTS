# Scaling the S1+S2 MOGPR Rice-Phenology Workflow — Design Note

From a single irrigation command (Klambu–Glapan) to larger AOIs — Jatiluhur,
Rentang, a Java province, or all of Java — and where the work runs (Mac vs HPC).

Status: **plan only** (no code/runs yet). Numbers are measured where marked
"measured" and estimated (clearly labelled) otherwise.

---

## 1. Governing principle: fuse rice, not everything

Wall-to-wall MOGPR over Java at 50 m is ~51 billion pixels — infeasible. But the
companion repo `s1-land-cover-classification` already provides a **Java paddy
mask**, and only a small fraction of the island is rice:

- `consensus_paddy_survey_2024_2025/consensus_paddy_masked.tif` — **12.1 M paddy px** (measured)
- `cropping_intensity_consensus_mt2024_25/paddy_mask.tif` — **9.2 M paddy px** (measured)
- → ~23,000–30,000 km² of paddy across Java (≈ 2.5–3.2 % of the grid).

**Restricting fusion to paddy pixels is a ~1000× reduction.** This single decision
is what turns "impossible on a laptop" into "a province on a laptop, all-Java on HPC".

---

## 2. Assets to reuse (cross-repo)

From `~/Github/s1-land-cover-classification`:
- **Paddy mask** (Java, 50 m) → `--mask` to clip fusion.
- **AOIs ready to use**: `2026/scenario2/study_area_jatiluhur.gpkg` (4,079 km², measured),
  `jatiluhur_paddy_aoi.geojson` (863 km², measured), `grid_jatiluhur_500m.gpkg`,
  `jatiluhur_ftw_fields.gpkg`, `2026/irrigation_performance/klambu.gpkg`.
- **`cropping_intensity.tif`** (independent Java product) → external cross-validation of our `n_seasons`.
- **Growth-phase maps**, water/terrain masks, parallel S1 preprocessor (`s1_preprocess_parallel.py`).

From this repo (`FuseTS`):
- MOGPR (`scripts/klambu_glapan_mogpr.py`, `pv_phenology.py`, `raster_mogpr_phenology.py`).
- **HPC/GPU scaffolding already present**: `run_mogpr_h100.slurm`, `setup_h100_mogpr.sh`,
  `install_gpu_mogpr.sh`, GPU MOGPR module — the basis for the HPC path.

Missing: a **Rentang** boundary (Cirebon/Indramayu/Majalengka, ~900 km² est.) — needs a shapefile/GeoJSON.

---

## 3. Two granularities — the main lever

| Granularity | Unit | What it answers | When |
|---|---|---|---|
| **Parcel / 500 m grid** (zonal-mean fusion per cell) | ~10²–10⁵ cells | cropping intensity, planting timing per management unit | operational, KSA-aligned, **laptop-scale even for a province** |
| **50 m pixel** (per-pixel fusion over masked rice) | 10⁵–10⁷ px | wall-to-wall maps, within-field detail | research/mapping; province on laptop overnight, **all-Java needs HPC** |

The parcel/grid path is both cheaper and the *operationally correct* resolution
(it matches the BPS KSA grid-point philosophy). Default to it; reserve 50 m
wall-to-wall for map products.

---

## 4. Compute budget

Benchmarks (measured on this Mac): extraction ≈ **24 s per ~1 M px per 12-day period**
(network-bound); per-pixel MOGPR ≈ **9 px/s/core**. Mac = 10 cores (~8 usable workers).

Paddy-pixel counts: Java ≈ 9.2 M (measured); Jatiluhur paddy AOI ≈ 0.35 M (est. from 863 km²);
Java at 500 m grid ≈ 92 k paddy cells (est. from ~23,000 km²).

| Target | Granularity | Units | Mac (8 workers) | HPC |
|---|---|---|---|---|
| Jatiluhur paddy AOI (863 km²) | 50 m px | ~0.35 M | ~1.5 h ✅ | minutes |
| Jatiluhur full (4,079 km²) | 50 m paddy px | ~0.6 M (est.) | ~2.5 h ✅ | minutes |
| Rentang (~900 km², needs AOI) | 50 m paddy px | ~0.35 M | ~1.5 h ✅ | minutes |
| One province | 500 m grid/parcel | tens of k | ~20–40 min ✅ | trivial |
| All Java | 500 m grid/parcel | ~92 k | ~1–2 h ✅ | minutes |
| All Java | 50 m paddy px | ~9.2 M | ~35 h ❌ | **~1 h @ 256 cores**, or minutes on GPU ✅ |

Extraction (datacube) scales ~linearly with land area: Java ≈ 52× Klambu ≈ ~11 h of
download for one year of 31 periods, but **tile-parallelisable** to a few hours
wall-clock (Mac) and faster on HPC bandwidth.

---

## 5. Architecture

```
                 ┌─────────────── paddy mask (s1 repo) ───────────────┐
                 ▼                                                     │
 AOI ──► tiles ──► per-tile extraction (odc.stac, MPC) ──► clip to mask │
   (MGRS or grid)        (parallel, resumable)                         │
                 ▼                                                     │
        12-day datacube per tile  ──► MOGPR fusion ◄────── granularity ┘
        (VV, VH, S2ndvi)            (per-pixel OR per-grid/parcel)
                 ▼
        peakvalley multi-season phenology ──► merge tiles ──► Java mosaics
                 ▼
        cross-validate vs cropping_intensity.tif (s1 repo)
```

Design choices:
- **Tiling**: split the AOI into fixed tiles (e.g. MGRS 100 km, or a regular grid).
  Each tile is independent → trivially parallel and resumable.
- **Mask-first**: clip each tile's cube to paddy *before* fusion; skip tiles with no rice.
- **Granularity switch**: per-pixel (`raster_mogpr_phenology.py` logic) or
  per-cell zonal-mean (`field_zonal_mogpr.py` logic, generalised to a grid).
- **Resumability**: write per-tile outputs; a tile already on disk is skipped on rerun.
- **Merge**: VRT/mosaic per-tile GeoTIFFs into Java-wide `n_seasons`, `SOS`, etc.

---

## 6. Pipeline changes to implement

Add to `klambu_glapan_mogpr.py` / a new `scale_runner.py`:
- `--aoi <gpkg/geojson>` — already supported via shapefile; extend to GPKG/GeoJSON.
- `--mask <paddy.tif>` — reproject/resample mask to the cube grid; set non-paddy → NaN.
- `--granularity {pixel,grid,parcel}` and `--grid-size 500` / `--parcels <gpkg>`.
- `--tiles <n|grid.gpkg>` + `--tile-index k` (for SLURM array) + `--resume`.
- `--workers N` — multiprocessing for the Mac per-pixel path.
- Keep outputs per-tile, then a `merge_tiles.py` for the mosaic.

No new dependencies — all within the existing `fusets` conda env.

---

## 7. Mac vs HPC split

**Mac (M-series, 10 cores):**
- Jatiluhur, Rentang (50 m, mask-driven): ~1.5–2.5 h each.
- A province or all-Java at 500 m grid/parcel: ~1–2 h.
- Multiprocessing across the 8 usable workers.

**HPC (only needed for all-Java 50 m wall-to-wall paddy):**
- Embarrassingly parallel by tile → **SLURM array** (`--array=0-N`), one tile per task,
  each task runs `scale_runner.py --tile-index $SLURM_ARRAY_TASK_ID`.
- CPU estimate: 9.2 M px / 9 px/s = ~284 core-hours → ~1 h on 256 cores.
- **GPU path**: reuse `run_mogpr_h100.slurm` + GPU MOGPR — batched GP fits cut this to
  minutes–tens of minutes on a few H100s. Recommended for repeated/all-Java runs.
- Extraction on HPC: parallel tile download (mind MPC rate limits; consider a local
  Sentinel mirror or the existing `s1_preprocess_parallel.py` for the SAR side).

Sketch SLURM array (to be written):
```bash
#SBATCH --array=0-199%32        # 200 tiles, 32 concurrent
#SBATCH --cpus-per-task=8
python scale_runner.py --aoi java.gpkg --mask paddy_mask.tif \
       --granularity pixel --tiles grid_java.gpkg --tile-index $SLURM_ARRAY_TASK_ID --resume
```

---

## 8. Cross-validation (scientific value-add)

The s1 repo's independent `cropping_intensity.tif` (Java-wide, SAR-classifier-derived)
is an ideal external benchmark for our **MOGPR-fused `n_seasons`**. A pixel/zone
agreement matrix (1/2/3-crop) between the two methods would (a) validate the fused
phenology at scale without new ground data, and (b) strengthen the paper's claim
beyond the 31-field BulakBakal check.

---

## 9. Storage

Per-tile, paddy-clipped cubes are small (sparse rice). Estimate: a 50 m Java paddy
NDVI+VV+VH 31-period stack stored densely-by-tile with LZW/zlib ≈ tens of GB;
stored sparsely (paddy only) far less. Phenology mosaics (a handful of int16/float32
Java rasters at 50 m) ≈ a few GB. Plan ~100 GB scratch for an all-Java 50 m run.

---

## 10. Phased rollout

1. **Jatiluhur (Mac)** — wire `--mask` + `--aoi`, validate the scaled, mask-driven
   workflow end-to-end (~1.5 h). Cross-check vs `cropping_intensity.tif` over Jatiluhur.
2. **Rentang (Mac)** — once an AOI is provided; same recipe.
3. **One province at 500 m grid (Mac)** — operational-resolution demonstration.
4. **All-Java 50 m (HPC)** — tiled SLURM array (CPU) or GPU H100; full mosaic +
   island-wide cross-validation.

---

## 11. Risks & caveats

- **MPC rate limits / bandwidth** dominate extraction at scale → tile concurrency, retries, resume.
- **Mask vintage/resolution**: the paddy mask is a specific season's product; align its
  year and 50 m grid to the fusion period, or paddy edges will leak/clip.
- **Per-pixel MOGPR cost** is the hard limit; prefer grid/parcel granularity unless a
  50 m map is the explicit deliverable.
- **Detector tuning** (`drop_thr`) should be set per-region (amplitude differs by water
  regime); calibrate against `cropping_intensity.tif` and any field data.
- **Edge/tile artifacts**: overlap tiles slightly or post-merge smooth season counts.
- **CRS care**: paddy mask is EPSG:4326; cubes are UTM (49S Java-central, 48S West Java) —
  reproject mask per tile; West Java (Jatiluhur/Rentang) is UTM 48S (EPSG:32748).
