# Worklog — Rice Growth-Phase Model (MOGPR as feature-generator → production)

Resume document for the phase-classification track. Companion to `WORKLOG.md` (which
covers the separate cropping-intensity / `drop_thr` axis, now parked).

Last updated: 2026-06-13 (HPC). Latest state in §12. Scripts inventory in §4.

---

## 1. Goal (why this track exists)

Estimate **rice production** the operational (KSA / BPS) way:

> **Production = area(paddy in growth phase 4 + 5) × 5.8 t/ha**

Phase 4+5 = **generative** stage (heading → grain-fill → maturity) = *standing crop about
to be harvested* → its area is the near-term **harvest area**. The user's existing pipeline
is `VH-CNN → 6-phase → calc_luas_panen.py → production` (in `idmai/DL/vh/`).

**MOGPR's role = a feature-generator feeding a phase classifier — NOT a standalone model.**
It reconstructs a clean, cloud-free NDVI+VH phenology curve; a classifier reads the phase
off that curve. Generative phase is an **optical** signature (peak greenness → senescence),
which VH-only models can't see directly — that's the gap MOGPR fills.

**6-phase legend (idmai):** 1 flooding · 2 early-veg · 3 late-veg · **4 early-generative** ·
**5 late-generative** · 6 post-harvest.

## 2. Data & inputs

| item | path | notes |
|---|---|---|
| Ground-truth labels | `idmai/DL/vh/model_6fase_enhanced_backward/data/training_points_0104.csv` | **USE THIS** (9,467 pts, drone+field, Mar–Jun 2024, 7 Java regions, has `lokasi`/`sumber`). NOT the 101k augmented `training_points_6fase.csv`. |
| → saved with regions | `data/aois/points_0104_all.csv` | + `region`, `utm` columns |
| S1 (operational) | `~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif` | 73 named bands `<year>_Period_<n>`; **int16 = dB×100, nodata −32768**; 2024 = bands 1–31. The single-year `java_vh_2024_50m.tif` is **EMPTY over land — do not use**. |
| NDVI | Microsoft Planetary Computer Sentinel-2 | pulled per 12-day period |
| Period system | 12-day, 31/yr, period 1 = Jan 1–12 | band *i* = period *i* |

Regions (lon-binned) & generative-label counts: Karawang 265, Indramayu 313, Cirebon 511,
Brebes/Tegal 546, Pekalongan **146 (only 5.4%)**, Semarang/Demak 214, Klambu 173.

## 3. Environment (HPC, CPU-first)

- **`fusets` conda env** = the MOGPR/classifier pipeline. Added this track:
  `pip install sktime lightgbm numba` (⚠️ downgraded scikit-learn 1.9→1.7.2 for sktime).
- **`geo_ml_env`** = TensorFlow 2.18, used to run the Keras **V3 VH-CNN**. Added `py3nvml`
  (their `utils.py` imports it). Force CPU for TF: `CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3`.
- **GPU:** 2× H100 but ~92% memory held by out-of-container processes (~2.5–6 GB free, 0%
  compute). **Use CPU** — all of this is CPU-fine. `vam.whittaker` still broken on HPC glibc
  (Mac-only); not needed here.

## 4. Pipeline scripts (`scripts/`)

All committed & pushed (`origin/main`). Grouped by role:

**Data / stack:**
| script | role |
|---|---|
| `build_cube_from_snap_s1.py` | MOGPR cube from **SNAP VH (int16→dB, named-band select) + MPC NDVI** |
| `concat_multiyear_stack.py` | concat per-year stacks → multi-year, preserving `YYYY_Period_N` names (ingestion step 4) |

**Phase classifier — build & validate (leave-one-region-out):**
| script | role |
|---|---|
| `extract_point_features.py` | scalar MOGPR phenology features at points |
| `extract_point_series.py` | **full fused NDVI+VH curve** per point (+coords) |
| `train_phase_classifier.py` | scalar features + RandomForest (59% F1) |
| `train_rocket_classifier.py` | **MiniROCKET** on full curve + per-loc norm (65.5%) — the model used |
| `validate_phase_vs_drone.py` | rule-based `[POS,EOS]` baseline (22%, the failed first pass) |
| `run_v3_cnn.py` | runs the **actual idmai V3 VH-CNN** on 0104 (their utils/scaler/encoder) |
| `train_ensemble.py` / `train_v3_mogpr_ensemble.py` | VH vs OPT vs ensemble; **V3 ⊕ MOGPR** faithful ensemble |
| `weight_sweep_v3_mogpr.py` | blend-weight sweep (best w_v3≈0.5 → 71.5%) |

**Production & maps (national, tiled, spawn-safe, cached):**
| script | role |
|---|---|
| `produce_estimate.py` | snapshot: phase-4-5 area × yield at one date (1 AOI) |
| `produce_annual.py` | single-AOI annual: episode count = harvests/yr × yield |
| `produce_annual_tiled.py` | **national** annual run (Java); caches `cube.nc`+`fused.npz` |
| `map_phases_tiled.py` | **per-period 6+3-phase growth-stage maps** from cached fused curves |
| `mosaic_ci_map.py` | national cropping-intensity map (GeoTIFF+PNG) from a run-dir |
| `make_validation_points.py` | stratified field-validation points (GeoJSON/CSV/KML) |

**Count-axis (other track, parked):** `scale_runner.py`, `sweep_drop_thr.py`,
`cross_validate_cropping_intensity.py`.
**Ingestion runbook:** `INGEST_NEW_PERIOD.md` (download→SNAP-13→mosaic→add-period→concat→swap).

All phase-model evaluation uses **leave-one-region-out CV** (spatial blocking — honest cross-region
transfer; random splits leak via spatial autocorrelation).

## 5. Results so far (phase-4-5 = generative; F1, leave-one-region-out unless noted)

| model | P4-5 F1 | 3-class acc | eval |
|---|---|---|---|
| rule-based `[POS,EOS]` | 22% | ~39% | held-out — **a hard rule can't read the curve** |
| scalar features + RandomForest | 59.0% | 67.4% | held-out |
| **MiniROCKET + LightGBM** (full curve + per-loc norm) | **65.5%** | **72.9%** | held-out — best MOGPR model |
| MiniROCKET OPT-only (NDVI channels) | 67.3% | 71.8% | held-out |
| VH-only MiniROCKET (proxy) | 52.2% | 61.2% | held-out |
| **V3 VH-CNN (your real model)** | 64.0% | 64.4% | ⚠️ **in-sample + cross-domain** |
| V3 ⊕ MOGPR — OPT-only arm | **68.2%** | 72.2% | held-out |
| V3 ⊕ MOGPR — **STACK** | **66.7%** | 68.9% | V3 in-sample + OPT held-out |

**Ensemble verdict (the head-to-head finale):** **STACK − V3_only = +2.6 F1** (and +4.5
3-class acc) — adding MOGPR's *held-out* optical lifts your *in-sample-advantaged* V3 model.
Even stronger: **OPT-only alone (held-out 68.2%) beats the V3 CNN (in-sample 64.0%)** — the
optical is simply the better signal for the generative phase. Per-region the optical helps
most where V3 is weak: Klambu 80→95, Karawang 68→75, Pekalongan 7→14.

**Weight-sweep (resolved the blend question), `weight_sweep_v3_mogpr.py`:** a *simple
probability average* `blend = (1-w)·OPT + w·V3` beats both arms — **best at w_v3≈0.5 →
71.5% F1** (+3.3 vs pure optical 68.2%; pure V3 = 64.0%). This **corrects the learned
meta-stack** (66.7%): plain averaging > meta-LightGBM (which overfit on 7 regions). So V3
is **not** redundant — it adds structural info on top of the optical.
⚠️ **Caveat:** V3 probs are IN-SAMPLE → 71.5% is optimistic. **Honest held-out floor = pure
MOGPR-optical 68.2%.** True blend gain needs a held-out V3 (§7).
**Recommendation: MOGPR-optical is the essential core; a ~0.5 V3 blend adds real structural
signal — confirm the true gain with a held-out V3.**

Per-class (MiniROCKET): bare R61/P70, vegetative R86/P77, generative R64/P68.
Per-region F1 (MiniROCKET): Klambu 93, Indramayu 79, Karawang 70, Cirebon 68, Brebes 65,
Semarang/Demak 43, **Pekalongan 21** (the failure region).

### Key findings
1. **MOGPR works as a feature-generator** — 65.5% held-out F1 vs 22% for the hard rule. The
   signal is the *curve shape*; feature importances confirm it (`slope_d`+`ndvi_d` dominate,
   VH ≈ 0.11).
2. **MOGPR optical is the dominant signal:** OPT-only 67% ≫ VH-only 52%. **MOGPR supplies
   the optical dimension VH lacks (+15 pts).**
3. **Even your real V3 VH-CNN (64%, in-sample-advantaged) only matches the held-out MOGPR
   model** → a good VH model tops out where the optical would carry it further.
4. **Pekalongan + Semarang/Demak fail for *both* methods** → region/label difficulty, not
   method. Pekalongan has only 5.4% generative labels.
5. The rule-based detour failed because of a systematic timing offset; a learned model
   absorbs it. (Switching label sets / `drop_thr` did not fix it.)

Outputs live in `output/phase_model/` (features_0104.csv, series_0104.npz+meta,
*_summary.json, classifier_results.png, phase_validation_results.png, v3_cnn/).

## 6. Caveats to remember (for honesty in any write-up)

- **V3 numbers are confounded twice:** in-sample (CNN trained on these labels → optimistic)
  AND cross-domain (trained on GEE 2023/24 VH, applied to the SNAP stack → may be depressed).
  Not 1:1 comparable to the held-out MOGPR 66%.
- **Phase-4-5 area × constant** has a **double-counting** trap: a pixel stays in phase 4-5
  for several weeks — for *annual* production, count each generative episode once (per
  POS→EOS pass), not per 12-day period.
- **Flat 5.8 t/ha** is a regional average; could later be made spatially variable with
  peak/integrated NDVI.
- No ground-truth *production* validation yet — only phase classification vs drone labels.

## 7. Future work / next steps (pick up here)

1. **[DONE] V3 ⊕ MOGPR ensemble + weight-sweep** — `v3_mogpr_ensemble_summary.json`,
   `weight_sweep_summary.json`. Best = **probability blend w_v3≈0.5 → 71.5% F1** (in-sample
   V3 → optimistic); held-out floor = pure optical **68.2%**. Remaining: **true held-out V3**
   (need its train/test split) to confirm the real blend gain.
2. **Fix Pekalongan** (5.4% generative): inspect its curves/labels; try a region-specific
   model, domain adaptation, or flag it out-of-distribution.
3. **Remove the V3 cross-domain confound** — run `run_v3_cnn.py` on V3's **native GEE
   2023/24 stack** (need that stack's path) for a clean apples-to-apples VH-CNN number.
4. **Push MiniROCKET further** — more kernels, window tuning, extra channels (VH slope,
   day-of-year), 6-class/ordinal head.
5. **[DONE] Production loop** — `produce_estimate.py` (snapshot) + `produce_annual.py`
   (annual). Brebes proof AOI (10,232 ha physical paddy): snapshot 2024-06-23 → 950 ha
   generative → 5,511 t; **annual → mean 2.30 harvests/yr, 23,582 ha harvest area →
   136,778 t @ 5.8 t/ha**. Time-series shows two clean harvest pulses (Apr + Aug) =
   double-crop. Production = physical area × cropping intensity × yield; the 2.30 matches
   the count-axis (Jatiluhur 2.19 / Rentang 2.38) — the two tracks converge.
   With realistic **`min_run=2`** (flicker suppressed): mean **1.86 harvests/yr, 19,014 ha →
   110,278 t** — the defensible figure (truth between 1.86 and 2.30). Still needs sanity-check
   vs BPS / `calc_luas_panen`.
   **Java-wide:** feasible but needs porting `produce_annual` into tiled `scale_runner` +
   multi-hour NDVI download; Pekalongan accuracy caveat applies (fix §7.2 first).
6. **True held-out V3** — need the CNN's train/test split for a fully fair head-to-head.
7. **Commit** `run_v3_cnn.py` + `train_v3_mogpr_ensemble.py` + outputs (hold push for review).

## 8. Git

- Pushed: `16f3880` (phase-model scripts + outputs), `a732e9b` (Rentang count-axis).
- Uncommitted: `run_v3_cnn.py`, `train_v3_mogpr_ensemble.py`, coords edit to
  `extract_point_series.py`, `output/phase_model/v3_cnn/`, ensemble summary (when done).
- Convention: cubes (`*.nc`), logs (`output/**/*.log`) git-ignored. User reviews commits
  before push.

## 9. Scaling to a national 12-day operational product (design)

**Decision (validated this session):** combine **S1 + S2 *inside MOGPR*** — one fused signal,
one classifier. NOT two separate models blended.

```
S1 (VH, cloud-penetrating) ─┐
                            ├─► MOGPR fusion ─► clean gap-free NDVI curve ─► MiniROCKET
S2 (NDVI, cloud-blocked)   ─┘                                              ─► phase 4-5
                                                              phase-4-5 area × 5.8 t/ha = production
```
- S1 fills S2's wet-season cloud gaps; S2 gives the greenness that defines phase; MOGPR
  reconstructs one clean NDVI phenology. The classifier reads phase off it.
- This is why it beats the VH-only path (S1-only 52–64% F1 → S1+S2 fused 68%).

**Operational pipeline (per 12-day cycle):** rolling datacube ingest (append new period) →
MOGPR re-fuse paddy pixels → classify phase → phase-4-5 area × yield → production per
admin unit. `scale_runner` already provides tiling / `--resume` / multiprocessing / mask.

**Scale:** Indonesia paddy (LBS) ≈ 7.4 M ha ≈ 74,000 km² ≈ **~30 M paddy px @ 50 m**
(Java ≈ 9–12 M, already run). Full national fusion ≈ ~4–12 h on 224 cores → fits the 12-day
cadence with slack. **Compute is NOT the binding constraint.**

**Binding constraints (in priority order):**
1. **Data ingestion bandwidth (#1):** ad-hoc MPC pulls don't scale nationally every 12 days.
   Need a co-located/streaming backend — local mirror, or openEO/CDSE / Earth Engine / AWS
   S2 where data sits next to compute.
2. **National S1 preprocessing:** current VH stack is **Java-only** (`rice-growth-stage-mapping`).
   National needs the SNAP pipeline run Indonesia-wide.
3. **Model generalization off Java (biggest scientific gap):** 0104 labels are Java-only and
   the model already fails on Pekalongan (a Java region). Sumatra/Sulawesi/Kalimantan need
   their own training labels or accept degraded accuracy.
4. **Ops engineering + dedicated compute:** automation, QA, acquisition latency (S1/S2 land
   days after overpass), and a non-shared/non-GPU-contended machine.

**Bottom line:** the *method + per-pixel pipeline* is proven on Java; a national 12-day
service is an **engineering + label-collection program** (data backend, national S1 preproc,
national labels, dedicated compute), not a parameter change. Suited to a government agency
(BPS / Kementan / BRIN).

### Near-real-time data availability & latency (operational caveat)

S2 source = **MPC `sentinel-2-l2a`** (NOT GEE). Scenes for **2024 are fully available**;
the VH stack covers 2024–2026 so **2025/26 runs are also possible** (classifier is 2024-
trained → cross-year phenology assumed, unvalidated).

For a *live* 12-day product, three things make the **newest period the weak spot**:
1. **Ingestion lag** — MPC publishes S2 L2A ~**1–3 days** after acquisition. → effective
   product latency ≈ **~1 week** (lag + waiting for the period to fill).
2. **MOGPR edge effect** — the fused curve is least constrained at the *end* of the series
   (no future obs to anchor the GP), so the most-recent phase call (the "harvest-ready
   *now*?" question) is the **most uncertain**; it firms up as the next period re-anchors it.
3. **Clouds** — wet-season gaps with no future data yet to recover them.
→ **Treat the latest period's phase as provisional**; historical periods are solid.
Constellation is being refreshed (S2C launched Sep 2024), so revisit is improving, not
declining. Availability/transient MPC failures are handled as gaps (per-period try/except,
resumable) — not a blocker, but budget for retries over ~11k queries/Java-run.

### Paddy-mask choice: predicted (de facto) vs LBS (de jure)

The Java run masks to the **predicted SAR-classifier consensus paddy**
(`cropping_intensity_consensus_mt2024_25/paddy_mask.tif`, **~2.30 M ha** / 23,012 km²) —
**NOT** the official LBS registry (`LBS_Jawa_50m.shp`, **~3.48 M ha** / 34,769 km²).
Predicted is **~34% smaller** than LBS.

**Why predicted is the right denominator for *production*:** you only harvest fields that
were actually *planted* — de facto detected cultivation. **LBS (de jure registered land)
includes paddy that sat fallow or was converted** in a given year → it would *over*-count
active area.

**Honest framing of the resulting number:** the ~34% gap is **partly real** (genuinely
non-cropped land LBS still lists) and **partly the classifier's omission error** (real paddy
the SAR missed — same model family that fails on Pekalongan-type areas). Net: the predicted
mask gives a **conservative, detected-cultivation** estimate — it won't over-count fallow,
but may **miss some real paddy** → the production figure **errs LOW**, the safer direction.

→ Report the national figure as *"annual production over **detected active paddy** (~2.3 M
ha)"*. **LBS stays as a cross-check** to reconcile with official BPS area statistics if needed
(`LBS_Jawa_50m.*` in `idmai/DL/vh/model_6fase_enhanced_backward/`).

## 10. National production run — Java 2024 — DONE

Ran the full pipeline over all of Java with `scripts/produce_annual_tiled.py`
(degree-tiled, **local-UTM per tile**, paddy-skip, **`spawn`-safe** Pool, **resumable**,
**caching**). 365 paddy tiles, ~11 h, 0 errors.

**Result (2024, detected-paddy denominator):**
| | |
|---|---|
| Physical paddy | **2,278,550 ha** |
| Annual harvest area | **4,219,032 ha** |
| Mean cropping intensity | **1.85** harvests/yr |
| **Annual production** | **~24.47 Mt** @ 5.8 t/ha |

CI 1.85 matches every smaller scale (Brebes 1.86, Jatiluhur 2.19, Rentang 2.38 →
island mean 1.85) — strong internal consistency. ~24.5 Mt is **low vs BPS ~30+ Mt** (as
expected: detected mask < LBS, flat yield, ~68% model, Pekalongan-type weakness) — a
**method demonstration, not an official statistic**.

Outputs: `output/production/java/java_cropping_intensity.png` (national map),
`java_n_harvests.tif` (mosaic), `java_production_summary.json`, per-tile `n_harvests.tif`.
Committed `e88f924`.

**Production-loop scripts (all committed):**
- `produce_estimate.py` — snapshot (phase-4-5 area at one date × yield) on one AOI cube.
- `produce_annual.py` — single-AOI annual (classify all periods, count harvest episodes).
- `produce_annual_tiled.py` — **national** tiled version (the Java run).
- `weight_sweep_v3_mogpr.py` — V3⊕MOGPR blend weight sweep.

**Caching (commit `3c79e3d`):** each tile caches `cube.nc` (S2+VH, no re-download) +
`fused.npz` (fused curves, no re-fuse). Validated **122.5 s cold → 1.1 s warm** (110×). So
after a year's one-time ~11 h fuse, all re-analysis (per-period maps, 6-phase, thresholds)
is **minutes**. `fused.npz` git-ignored.

## 11. Forward plan — per-period growth-stage maps + 2026 (p13 ingestion)

**Goal:** per-period (12-day) **growth-stage maps** of Java — the operational KSA-style
product (vs the annual cropping-intensity map). Output **both 6-phase** (flooding / early-veg
/ late-veg / early-gen / late-gen / harvest — train a 6-class head) **and 3-phase** (collapse:
bare/veg/generative, the robust ~70% view). 6-class ~51%, 3-class ~70% — fine splits (2↔3,
4↔5) are the weak part; flag as lower-confidence.

**2026 status:** VH stack `java_vh_2024_2026_50m.tif` has **12 periods** (`2026_Period_1..12`,
≈ Jan–late May). User is **adding period 13** (downloading raw S1). The 12-period partial fuse
was launched then **stopped + cleared** (superseded — adding p13 means re-fusing the series).
Note: **annual cropping intensity is NOT meaningful for a partial year** (~half), so the 2026
deliverable is the **per-period growth-stage maps**, not a production total. Latest period is
**provisional** (MOGPR edge effect).

**p13 ingestion = user's SNAP pipeline** (`rice-growth-stage-mapping/s1_period_pipeline.py`:
download → SNAP preprocess → convert int16 dB×100 → mosaic → stack into the multi-band
GeoTIFF). Config: `rice-growth-stage-mapping/pipeline_config_java_both_orbits.yaml`. SNAP is
at `idmai/esa-snap-13/bin/gpt` (⚠️ `gpt` **not on PATH** — set `snap_gpt_path`). Stages for
one period: `--periods 13 --skip-download --preprocess-only` then `--convert-only`,
`--mosaic-only`, `--stack-only` (rebuilds the stack → 13 `2026_Period_*` bands).

**Steps to go (in order):**
1. **Preprocess p13** (SNAP, user's pipeline) → VH stack updated to 13 periods.
2. **Build `map_phases_tiled.py`** — reads cached `fused.npz`/`cube.nc` → classify each period
   (6-class + 3-collapse) → per-tile multi-band phase maps → mosaic per period → Java
   per-period growth-stage maps (6-phase + 3-phase). `spawn`-safe. (Not built yet.)
3. **Fuse 2026 (13 periods)** with `produce_annual_tiled --year 2026` (populates cache).
4. **Generate per-period maps** from cache → 13 Java growth-stage maps (×2 schemes), minutes.
5. As later periods (14, 15, …) download → re-ingest + re-fuse (resume/cache makes it cheap).

## 12. p13 ingested · 2026 per-period maps (running) · Semarang field validation

### Achieved
- **p13 ingested end-to-end** through the operational SNAP pipeline, now captured as the
  reproducible runbook **`INGEST_NEW_PERIOD.md`**: download (12 scenes incl S1C) →
  **SNAP-13** preprocess (`sen1_preprocessing-gpt-50m.xml`, S1C needs SNAP-13) → `gdal_merge`
  mosaic → `stack_period_bands.py --add-period 13` → **`concat_multiyear_stack.py`** (preserves
  `YYYY_Period_N` names) → **live `stacks/java_vh_2024_2026_50m.tif` = 74 bands** (2026 now has
  13 periods incl `2026_Period_13`). Original backed up `.bak73`; per-year `java_vh_2026_50m.tif`
  → 13 bands (`.bak12` saved).
- **New scripts (committed):** `INGEST_NEW_PERIOD.md`, `concat_multiyear_stack.py`,
  `map_phases_tiled.py` (validated: cache→maps ~3s/tile, emits 6-phase + 3-phase, one band/period),
  `make_validation_points.py`.

### In progress (running background jobs)
- **2026 fuse:** `produce_annual_tiled.py --year 2026 → output/production/java_2026` (PID 1515774,
  ~60/365 tiles at last check, ~3–4 h, caches `cube.nc`+`fused.npz`). Monitor `b6g01xbxj`
  **auto-chains** → `map_phases_tiled --mosaic` → **`output/production/java_2026_phases/
  java_phase6_p01..13.tif` + `java_phase3_p01..13.tif`** = the deliverable (per-period 6+3-phase
  Java growth-stage maps).
- **Validation-points monitor `bjhnhdt5u`:** waits for `java_phase6_p13.tif`, then runs
  `make_validation_points.py` → `output/validation/semarang_veg_p13.{csv,geojson,kml}`.

### Semarang field-validation design
- **Goal:** field-validate the 2026 growth-stage maps near Semarang (user is on-site).
- **TIME-LAG handling (key):** latest map = **p13 ≈ May 25–Jun 5**; survey is mid-late June+ →
  rice advances ~1 phase over the ~3–5-week lag. So **target fields at flooding/early-veg
  (phase 1–2) on p13** → they mature to **vegetation at visit**. (Do NOT pick fields already
  vegetative on p13 — they'll be generative by then.) **p14/15 not yet available** (would allow
  current-period targeting via the runbook).
- **Points:** ~100 near Semarang, bbox `110.15,-7.20,110.75,-6.80`, ≥2 km apart, paddy-only,
  phone-loadable `.kml/.geojson/.csv` with blank `observed_fase / transplant_date / notes`.
  Preliminary set (committed, CI-stratified, logistics): `output/validation/semarang_prelim.*` (102 pts).
- **Field protocol per point:** GPS + observed growth stage (1–6) + **transplant date (ask farmer)**
  + photo.
- **Validation metric:** confusion (observed vs predicted phase) **+** transplant-date phenology
  check (BulakBakal-style, robust to the lag).

### Next steps (resume here)
1. When fuse+maps finish (`b6g01xbxj`): per-period maps in `output/production/java_2026_phases/`;
   veg points auto-generated (`bjhnhdt5u`) → `output/validation/semarang_veg_p13.*`.
2. **Field survey** (user): visit points, record stage + transplant date + photo.
3. **Build a scorer** (observed vs predicted phase + phenology) once survey data returns.
4. **Ingest p14/p15** when downloaded (via `INGEST_NEW_PERIOD.md`) for current-period targeting.
5. Parked: Pekalongan fix, native-GEE V3 head-to-head, count-axis (Jatiluhur/Rentang drop_thr).

### QUEUED: full 2025 annual run (after 2026)
A detached chain (`output/production/chain_2025.log`) waits for the 2026 fuse+maps, then runs
the **full-year 2025** annual run — same steps as 2024 (it's a full year → cropping-intensity +
production total, not just per-period maps), **with caching** (automatic now):
`produce_annual_tiled --year 2025 --out output/production/java_2025` → then
`mosaic_ci_map --run-dir output/production/java_2025 --year 2025` → `java_cropping_intensity.png`
+ `java_n_harvests.tif` + `java_production_summary.json`. 2025 = 30 periods in the stack.
**To check on resume:** `tail output/production/chain_2025.log`; results in `output/production/java_2025/`.
(`mosaic_ci_map.py` also formalizes the 2024 national-map step.)

Last updated: 2026-06-13 (p13 ingested → 74-band stack; 2026 fuse+maps running; Semarang
validation points wired, lag-adjusted to phase 1–2 on p13).

## 13. 2024+2025 national runs · grid/render fixes · paper comparison · multi-season retrain (2026-06-14)

**Patrol/Indramayu gap FIXED** (`891c56d`): `map_phases_tiled._tile_crs` picked UTM zone from
tile WEST edge, but `produce_annual_tiled` uses tile CENTER → only the column straddling 108°E
(c022) was stamped 48S while data was fused 49S → rectangular hole. Fixed to center-based.

**CI map cleanups:**
- Clamp IP to 1–3 (`ccb437b`): generative-episode counter over-counts ~1% to ≥4; rice ≤3/yr.
  `mosaic_ci_map.py` now writes `java_cropping_intensity.tif` (clamped) + `java_n_harvests.tif` (raw 1–7).
- PNG render fixed twice: max-pool inflated CI=3 → switched to **majority-class (mode)** downsample
  (`65f59b5`); a plain imshow had also subsampled away thin paddy strips.
- **Grid alignment** (`b327b36`): mosaics auto-picked ~0.0004526°/50.39 m; now `reproject_match`
  to the source mask → **true 50 m (0.000449158°), pixel-aligned**. Area/production unchanged
  (computed per-tile in native UTM). Applies to `mosaic_ci_map.py` + `map_phases_tiled.py`.

**2025 full annual run DONE** (cache saved per request): 2,278,533 ha, **IP 1.94, 25.59 Mt**.
**2024 re-run** via current cached pipeline (`output/production/java_2024/`): **IP 1.85, 24.47 Mt**
== original → pipeline consistent. **2024-vs-2025 verdict: +0.10 IP is REAL** (rerun matches; raw
≥4 tail equal; 2025 has FEWER periods (30) yet higher; **S2 NDVI valid 2025 61.3% < 2024 64.9%**
yet higher → not data-coverage). Wet-year (post-El-Niño) signal. Triple-crop concentrates in
**Bengawan Solo** corridor (47% of Java CI=3 in 2024). Both years cached & comparable.

**Per-period growth-stage maps** for **2024 (31p), 2025 (30p), 2026 (13p)** — 6+3 phase, 50 m,
mask-aligned: `output/production/java_{2024,2025,2026}_phases/`.

**CI consensus (2024/2025) GeoTIFFs** in `output/production/`: `java_ci_consensus_2024_2025.tif`
(rounded 2-yr mean, IP 1.98), `java_ci_mean_…` (float), `java_ci_min_…` (floor). NOTE pixel-level
agreement only **47%** → user chose **combined-time-series CI** (re-fuse 61 periods, count/2) as
the rigorous single map — NOT yet built (needs a multi-year tweak to produce_annual_tiled).

**Follow-up report** (in landcover repo, pushed): `…/2026/laporan/LAPORAN_LANJUTAN_MOGPR_S1S2_2026.{md,tex,pdf}`
— tindak-lanjut to L1–L3+Kegiatan; BAB 4 = per-report impact; renders via their pandoc+xelatex pipeline.

**Paper comparison** (`rice-growth-stage-mapping/paper_latex/manuscript.tex` = S1-only DL, CNN
87.34% OA/4-class/in-distribution CV): NOT directly comparable to ours (S1+S2, 6/3-class,
generative-F1, leave-one-region-out). Agree on SNAP>GEE, VH, 12-day, scale. Paper maps 3.48 M ha
(≈LBS de jure) vs our 2.28 M detected. **Key cross-insight: paper's single-season collapse (21.89%)
warns our model** — trained Mar–Jun only.

**Multi-season retrain IN PROGRESS** (the seasonal-bias fix): exposure check found our 0104.csv =
**0% dry-season**; **35.5%** of 2024 generative pixels fall in dry-season periods; dry-gen wave
muted (~21% vs ~33% wet). Fix uses idmai's own **`6fase.csv` dry subset (Jul–Sep, 34,143 pts,
51% generative, native 6-class)** — NO paper data needed. Built `data/aois/points_dry6fase.csv`.
Detached chain `/tmp/multiseason.sh` (log `output/production/multiseason.log`): extract fused
series → merge w/ series_0104 → `series_multiseason` → retrain + re-map 2024 → `java_2024_phases_ms/`
+ exposure comparison. **On resume:** `tail output/production/multiseason.log`; if dry-gen RISES →
muting was bias (re-map 2025/2026 + update IP); if STAYS ~21% → real (original numbers stand).

**Unpushed (FuseTS):** none outstanding for the grid/CI/CRS commits (all pushed thru `b327b36`).
Multi-season scripts/outputs not yet committed (run in progress).

Last updated: 2026-06-14 (2024+2025 national runs done & comparable; grid/render/CRS fixed;
follow-up report written+pushed; multi-season retrain running to resolve dry-season bias).
