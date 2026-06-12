# Worklog — Rice Growth-Phase Model (MOGPR as feature-generator → production)

Resume document for the phase-classification track. Companion to `WORKLOG.md` (which
covers the separate cropping-intensity / `drop_thr` axis, now parked).

Last updated: 2026-06-12 (HPC).

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

| script | role | status |
|---|---|---|
| `build_cube_from_snap_s1.py` | MOGPR cube from **SNAP VH (int16→dB, named-band select) + MPC NDVI** | committed `16f3880` |
| `extract_point_features.py` | scalar MOGPR phenology features at points | committed |
| `extract_point_series.py` | **full fused NDVI+VH curve** per point (now also saves coords) | committed (coords edit uncommitted) |
| `train_phase_classifier.py` | scalar features + RandomForest | committed |
| `train_rocket_classifier.py` | **MiniROCKET** on full curve + per-location norm | committed |
| `train_ensemble.py` | VH-only vs OPT-only vs ensemble (proxy) | committed |
| `validate_phase_vs_drone.py` | rule-based `[POS,EOS]` baseline | committed |
| `run_v3_cnn.py` | runs the **actual idmai V3 VH-CNN** on 0104 (reuses their utils/scaler/encoder) | **UNCOMMITTED** |
| `train_v3_mogpr_ensemble.py` | **V3 ⊕ MOGPR** faithful ensemble | **UNCOMMITTED, RUNNING** |

All evaluation uses **leave-one-region-out CV** (spatial blocking — honest cross-region
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
