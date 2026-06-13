# Ingesting a New S1 Period → Growth-Stage Maps (reproducible runbook)

How to add a newly-downloaded Sentinel-1 12-day period to the VH stack and regenerate the
per-period **growth-stage maps** (6-phase + 3-phase). Validated end-to-end on **2026 period 13**
(2026-06-13). Substitute `Y`=year, `N`=period below.

Two repos involved:
- **`~/work/rice-growth-stage-mapping`** — S1 download + SNAP preprocess + mosaic + stack (steps 0–5).
- **`~/work/FuseTS`** — MOGPR fuse + phase classification → maps (steps 6–7).

---

## Prerequisites (one-time / per-period)
- **Raw S1 downloaded** into `rice-growth-stage-mapping/workspace_java_both_orbits/year_<Y>_50m/p<N>/downloads/`
  (all scenes for the period, both orbits, S1A **and S1C**). Download is via ASF (needs Earthdata
  auth in `~/.netrc`) — **done by the data owner**. Verify the count is stable before preprocessing.
- **SNAP 13** at `~/work/idmai/esa-snap-13/bin/gpt` — **must be SNAP 13** (S1C-capable; the older
  `esa-snap` cannot process S1C scenes).
- **50 m graph** `rice-growth-stage-mapping/sen1_preprocessing-gpt-50m.xml`.
- Env with rasterio + GDAL: `/opt/conda/envs/geo_ml_env/bin/python` (has `osgeo_utils.gdal_merge`).
- Box has ~2 TB RAM (4 SNAP workers × 200 G).

---

## Step 0 — confirm the download is complete
```bash
cd ~/work/rice-growth-stage-mapping
D=workspace_java_both_orbits/year_<Y>_50m/p<N>/downloads
ls $D/*.zip | wc -l           # compare to the period's scene count (≈12–16)
du -sh $D                     # re-run after ~30 s; size must be STABLE
```

## Step 1 — SNAP preprocess (parallel, SNAP-13, 50 m) → per-scene VH
```bash
/opt/conda/envs/geo_ml_env/bin/python -u s1_preprocess_parallel.py \
  --input-dir  workspace_java_both_orbits/year_<Y>_50m/p<N>/downloads \
  --output-dir workspace_java_both_orbits/year_<Y>_50m/p<N>/preprocessed \
  --graph sen1_preprocessing-gpt-50m.xml \
  --gpt-path /home/unika_sianturi/work/idmai/esa-snap-13/bin/gpt \
  --workers 4 --memory 200G --cache 150G
```
→ `p<N>/preprocessed/*_VH.tif` — **already int16, dB×100, 50 m, nodata −32768** (no convert step).
~7–10 min for 12 scenes.

## Step 2 — mosaic the period (gdal_merge) → `p<N>_mosaic.tif`
```bash
mkdir -p workspace_java_both_orbits/year_<Y>_50m/p<N>/mosaic
/opt/conda/envs/geo_ml_env/bin/python -m osgeo_utils.gdal_merge \
  -ot Int16 -of GTiff -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
  -a_nodata -32768 -n -32768 -init -32768 \
  -o workspace_java_both_orbits/year_<Y>_50m/p<N>/mosaic/p<N>_mosaic.tif \
  workspace_java_both_orbits/year_<Y>_50m/p<N>/preprocessed/*_VH.tif
```
→ `p<N>_mosaic.tif` (~90–105 MB). ~15 s.

## Step 3 — add the period to the per-year stack (incremental)
```bash
cp stacks/java_vh_<Y>_50m.tif stacks/java_vh_<Y>_50m.tif.bak   # safety
/opt/conda/envs/geo_ml_env/bin/python -u stack_period_bands.py \
  --mosaic-dir workspace_java_both_orbits/year_<Y>_50m \
  --output     stacks/java_vh_<Y>_50m.tif \
  --add-period <N> \
  --reference  stacks/java_vh_2024_50m.tif \
  --boundary   data/coastline.gpkg
```
→ `java_vh_<Y>_50m.tif` gains band `Period_<N>`. ~90 s. Verify: `bands == N`.

## Step 4 — concat into the multi-year stack (preserves `YYYY_Period_N` names)
Base = the years before `<Y>` (e.g. `java_vh_2024_2025_50m.tif`); append = the updated per-year stack.
```bash
/opt/conda/envs/fusets/bin/python ~/work/FuseTS/scripts/concat_multiyear_stack.py \
  --base   stacks/java_vh_2024_2025_50m.tif \
  --append stacks/java_vh_<Y>_50m.tif --append-year <Y> \
  --output stacks/java_vh_2024_<Y>_50m_vNEW.tif
```
→ combined stack with `<Y>_Period_<N>` as the last band. ~9 min (band-by-band). **Verify**:
```bash
/opt/conda/envs/geo_ml_env/bin/python -c "import rasterio;s=rasterio.open('stacks/java_vh_2024_<Y>_50m_vNEW.tif');print(s.count,[d for d in s.descriptions][-1])"
```

## Step 5 — swap into the live stack (back up the old one)
```bash
cd stacks
mv java_vh_2024_<Y>_50m.tif java_vh_2024_<Y>_50m.tif.bak$(prev_band_count)
mv java_vh_2024_<Y>_50m_vNEW.tif java_vh_2024_<Y>_50m.tif
```
The live stack now has the new period. **This is the only artifact the FuseTS side reads.**

## Step 6 — MOGPR fuse the year (FuseTS; caches → cheap re-runs)
```bash
cd ~/work/FuseTS
/opt/conda/envs/fusets/bin/python -u scripts/produce_annual_tiled.py \
  --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_<Y>_50m.tif \
  --mask ~/work/landcover/s1-land-cover-classification/cropping_intensity_consensus_mt2024_25/paddy_mask.tif \
  --series output/phase_model/series_0104 --year <Y> --tile-deg 0.2 --res 50 \
  --min-run 2 --workers 20 --out output/production/java_<Y>
```
Tiled, **spawn-safe**, **resumable**, **cached** (`cube.nc`+`fused.npz`). 365 paddy tiles. ~3–4 h
for ~13 periods. (Annual production total is only meaningful for a FULL year; for a partial year
the deliverable is the per-period maps.)

## Step 7 — per-period growth-stage maps (6-phase + 3-phase)
```bash
/opt/conda/envs/fusets/bin/python -u scripts/map_phases_tiled.py \
  --run-dir output/production/java_<Y> --series output/phase_model/series_0104 \
  --workers 16 --mosaic --out output/production/java_<Y>_phases
```
Reads the cached fused curves (no re-fuse) → per-tile `phase6.tif`/`phase3.tif` (one band/period) →
mosaics to `output/production/java_<Y>_phases/java_phase6_p<NN>.tif` and `java_phase3_p<NN>.tif`.
**Minutes.** → the deliverable: a 6-phase and a 3-phase Java growth-stage map for each period.

---

## Notes & gotchas (learned during p13)
- **S1C needs SNAP 13.** Using the older `esa-snap` gpt fails on S1C scenes — always `esa-snap-13`.
- **50 m graph**, not the base graph (`sen1_preprocessing-gpt-50m.xml`).
- Preprocessed VH is **already int16 dB×100 / 50 m** — no separate convert/scale step.
- Band **names matter**: the FuseTS side selects `<Y>_Period_<n>`; the concat (step 4) sets them.
  `stack_multiyear.py` uses `YYYY_PXX` (wrong format) — use `concat_multiyear_stack.py`.
- **Caching:** once step 6 fuses a year, re-runs (new threshold, 6↔3 phase, re-map) are minutes.
  Adding a *new period* to an already-fused year means re-fusing that year (the series changed) —
  but delete the stale `output/production/java_<Y>/` first, or `--resume` will skip with the old
  period count.
- **Edge effect:** the latest period is provisional (MOGPR has no future data to anchor it).
- Model accuracy: 3-phase ~70%, 6-phase ~51% (fine splits 2↔3, 4↔5 weak). Treat 6-phase
  sub-stages as lower-confidence.
