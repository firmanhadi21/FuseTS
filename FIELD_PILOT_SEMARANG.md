# Semarang Field-Validation Pilot

Cheap, independent validation for the companion paper — and a dry run of the TA-2027 multi-DI
protocol. One short campaign around Semarang gives the paper its first *out-of-campaign*
validation point and proves the workflow end to end.

## What's prepared
- **120 points**, stratified across the 3-class scheme on the **multi-season p13 model**
  (40 bare / 40 vegetative / 40 generative; 6-class spread incl. 40 generative-stage points).
- Files (`output/validation/`):
  - `semarang_validation_p13.kml` — load on a phone/Garmin for navigation
  - `semarang_validation_p13.csv` — the field record sheet (predicted phase pre-filled)
  - `semarang_validation_p13.geojson` — GIS
- Bounding area: 110.15–110.75°E, −7.20 to −6.80 (Semarang–Demak).

## Record at each point (fill the CSV)
| column | what |
|---|---|
| `observed_fase` | growth stage **1–6** (1 flooding · 2 early-veg · 3 late-veg · 4 early-gen · 5 late-gen · 6 post-harvest) |
| `observed_date` | visit date (YYYY-MM-DD) |
| `transplant_date` | planting/transplant date (ask farmer or estimate) — **important** |
| `water_condition` | tergenang / lembab / kering |
| `notes` | anything unusual (variety, paranet, mixed plot, plot <0.25 ha) |
- GPS to within ~10 m of the point. Leave `pred_phase3`/`pred_phase6` untouched (the model's prediction).

## Timing / the lag (read this)
The model's latest period is **p13 (~late May)**. If you visit much later than that, a field's
*current* stage will have advanced past the p13 prediction, so a **direct** observed-vs-predicted
comparison is lag-affected. Two ways to stay valid:
1. **Record `transplant_date`** at every point — the scorer's transplant-date check is timing-robust
   (it compares observed stage to days-after-planting, independent of which model period is latest).
2. Or ingest the S1 period covering your visit window (see `INGEST_NEW_PERIOD.md`), re-map, and
   compare against that current prediction.
Either way, fill `transplant_date` — it is the most robust ground truth.

## Score it (when the CSV is filled)
```bash
python scripts/score_field_validation.py output/validation/semarang_validation_p13.csv
```
Outputs: 6-class and 3-class confusion + accuracy, **generative (4,5) precision/recall/F1**
(the production-relevant metric), and the transplant-date consistency check.

## What it delivers
- The paper's independent validation point (§ field validation / limitations): a real
  observed-vs-predicted confusion matrix and generative F1 on fields **not** in the training campaign.
- Proof of the protocol for the TA-2027 multi-DI campaign (`FIELD_VALIDATION_PLAN_TA2027.md`).
- A first check of whether the model's IP-driving generative class holds up on independent ground truth.
