# TA-2027 Field Validation Plan — S1+S2 MOGPR Rice Phase / IP / Production

Purpose: convert the June-2026 phase model from a *promising demonstration* into an
*operationally trustworthy* product by validating it against **independent field data on a
spatial split**. This is the decisive next step identified in `WORK_SUMMARY_2026-06.md`.

## 1. What this validation must prove (the specific open questions)

The current model's accuracy is **optimistic** because it was validated on a random (not spatial)
split with same-campaign labels. Field validation must answer:

1. **Spatial generalization** — does accuracy hold on DIs/regions *not* in training? (The honest
   test; do **leave-one-DI-out**, not random CV.)
2. **Dry-season generative** — the multi-season fix lifted held-out dry accuracy 36.9%→78.6%, but
   on same-campaign data. Confirm with *independent* dry-season field points (esp. fase 4–5).
3. **IP / min_run calibration** — `min_run=3` was tuned to make IP physical (2.2, not 2.8). Verify
   against *actual harvest counts* tracked through a season.
4. **Production realism** — does predicted generative-area × 5.8 t/ha match observed harvest at DI
   scale (vs KSA/BPS)?
5. **Known S1 error modes** (from landcover L1/Kegiatan) — does optical fusion reduce Bandung-highland
   false positives (54% of FP) and rainfed false negatives (56% of FN)?

## 2. Sites — multi-DI, spatial + agro-climatic diversity

Align with the landcover TA-2027 roadmap (3–5 DIs) and span West/Central/East Java + rainfed:

| DI / area | Province | Role | Notes |
|---|---|---|---|
| Klambu | C. Java | irrigated, L2 baseline | reuse L2 SI/CU/RI benchmark |
| Rentang | W. Java (Indramayu/Cirebon) | intensive, 108°E zone | also tests the CRS-boundary column |
| Jatiluhur (Tarum Timur) | W. Java | large, heterogeneous | head/mid/tail gradient |
| Semarang (Demak) | C. Java | **points already generated** | `output/validation/semarang_veg_p13.*` (100 pts) |
| 1–2 East Java DIs | E. Java | Bengawan Solo basin | the triple-crop hotspot — validate IP=3 |
| + 1 rainfed / Bandung-highland transect | W. Java | error-mode test | targets FP/FN modes |

Total target ~**150–200 points/DI** (per landcover SOP), ≥ **1,000 points** overall.

## 3. Timing — two seasonal campaigns, period-tagged

The model is season-sensitive, so **both seasons must be sampled**:
- **Wet-season campaign:** ~Feb–Apr 2027 (peak vegetative→generative).
- **Dry-season campaign:** ~Jul–Sep 2027 (the under-validated window).
- Optionally a 3rd visit to **track a subset of fixed plots across the season** (for IP / harvest-count
  validation — see §6.3).

Tag every observation to its **12-day period** (the model's time unit). Apply the **lag rule** used
for Semarang: to observe a target phase at visit time, pick fields predicted to reach it given the
~1–2 period lag between the latest available S1 period and the survey date.

## 4. Sampling design

Stratify each DI by:
- **Predicted phase (1–6)** — quota per class so all phases (esp. **generative 4+5**) are represented;
  do NOT let the natural ~30% bare dominate the sample.
- **Hydraulic position** — head / middle / tail of the secondary/tertiary network (33/33/33), to
  catch irrigation-performance gradients (ties to P3A use case).
- **Confidence / ambiguity** — include low-confidence and class-boundary pixels (where 6-class is
  weak: 2↔3, 4↔5) to probe the real failure modes, not just easy fields.
- **Independence** — sample fields NOT in the idmai 0104/6fase training campaigns (avoid the
  same-campaign optimism).

Generator: `scripts/make_validation_points.py` (stratify on the per-period phase map, `--exclude`
non-target classes, `--min-dist` spacing). One run per DI per campaign.

## 5. Field protocol

Reuse the **QGIS–Mergin Maps SOP** (landcover L3; offline-first, server in Indonesia). Per point record:
- GPS (≤ 5 m; prefer ≤ 2 m), date/time → mapped to period
- **Observed phase (1–6)** + 3-class collapse
- **Water condition** (tergenang / lembab / kering)
- **Transplant date** (or estimate) + **variety** (for phenology cross-check; varieties differ ±5–7 d)
- Plot size (flag < 0.25 ha = sub-pixel at 50 m), photo, surveyor, notes
- QC: inter-surveyor Cohen's Kappa on a shared subset, agronomic-plausibility checks, outlier review

Columns already in the generated CSV/GeoJSON/KML: `observed_fase, transplant_date, notes`.

## 6. Analysis & metrics (the scorer to build)

### 6.1 Phase accuracy — spatial holdout
- Confusion matrix observed vs predicted; per-class precision/recall/F1; overall accuracy + Kappa.
- **Leave-one-DI-out CV** is the headline number (spatial generalization).
- Report **6-class and 3-class** separately, and **generative (4+5) F1** prominently (production driver).
- Break down by **season** (wet vs dry) — directly tests the multi-season fix on independent data.

### 6.2 Transplant-date → phenology consistency
- For points with a transplant date: predicted phase vs expected phase from days-after-transplant
  (rice ~110-d cycle). Quantifies temporal calibration; flags systematic lead/lag.

### 6.3 IP / production validation
- **Track a subset of fixed plots** across the season → count *actual* harvests → compare to predicted
  IP. This is the only direct test of the `min_run=3` calibration (does it over/under-count?).
- At DI scale: predicted generative-area × 5.8 t/ha vs reported harvest (KSA panel / BPS / BWS records).
  Also revisit the flat-5.8 yield assumption with per-DI yield where available.

### 6.4 Error-mode tests
- Bandung-highland transect: FP rate with vs without optical fusion.
- Rainfed fields: FN rate (the weak-VH, no-flooding-anchor case).

## 7. Success criteria (proposed)

| Metric (spatial leave-one-DI-out) | Target |
|---|---|
| 3-class overall accuracy | ≥ 80% |
| Generative (4+5) F1 | ≥ 70% |
| Dry-season 3-class accuracy | ≥ 75% (no large wet/dry gap) |
| IP vs tracked harvest count | within ±0.3 |
| DI-scale production vs reference | within ±15% |

Below target → diagnose (class, season, DI, hydraulic position) and **retrain** (§8).

## 8. Retrain & operationalization loop

1. Fold field labels into training (replacing same-campaign labels) → retrain MiniROCKET+LightGBM.
2. Re-validate on a held-out DI; iterate `min_run` against tracked harvests.
3. Re-map 2024–2027 from cache (minutes) → operational maps.
4. Integrate phase → Kc to improve the L2 SI/CU/RI irrigation indices.
5. Recalibrate the accuracy figure quoted to BWS/P3A (decision-support → contractual-grade).

## 9. Logistics

- Tie budget/personnel/BWS coordination to the landcover TA-2027 roadmap (3–5 DI pilot, ~200 pts/DI,
  Mergin server already deployed). Train 20–30 BWS/BBWS staff on collection + interpretation.
- Reuses: `make_validation_points.py`, the Semarang point set (ready), the Mergin SOP, the cached
  fused curves (instant re-map after retrain).

## 10. Immediate next action (before TA-2027)

The **Semarang survey** (100 points, lag-adjusted phase 1–2 on 2026 p13) is the pilot for this plan.
On return of filled `observed_fase` / `transplant_date`, build the scorer (§6.1–6.2) — it becomes the
template for the multi-DI campaign.
