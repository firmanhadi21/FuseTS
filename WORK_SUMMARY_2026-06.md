# Work Summary — Rice Growth-Stage / Cropping-Intensity / Production (June 2026)

Consolidated record of the S1+S2 MOGPR phase-model work. The live resume doc is
`PHASE_MODEL_WORKLOG.md` (§13–14); this file is the readable hand-over summary.

## 1. Goal
Map rice **growth stage (phase)** and derive **cropping intensity (IP)** and **production**
for Java, using **MOGPR S1+S2 fusion as a feature-generator** feeding a trained classifier
(NOT MOGPR as a standalone model). Production = area(generative, phase 4+5) × 5.8 t/ha (KSA-style).

## 2. Final deliverables (all 50 m, EPSG:4326, mask-aligned)

| Product | Path | Notes |
|---|---|---|
| Cropping-intensity map 2024 | `output/production/java_2024/java_cropping_intensity.{tif,png}` | clamped 1–3 |
| Cropping-intensity map 2025 | `output/production/java_2025/java_cropping_intensity.{tif,png}` | clamped 1–3 |
| Raw harvest count | `output/production/java_{2024,2025}/java_n_harvests.tif` | 1–7 (pre-clamp) |
| Production summary | `output/production/java_{2024,2025}/java_production_summary.json` | |
| Phase maps 2024 | `output/production/java_2024_phases_ms/` | 6-phase + 3-phase × 31 periods |
| Phase maps 2025 | `output/production/java_2025_phases_ms/` | × 30 periods |
| Phase maps 2026 | `output/production/java_2026_phases_ms/` | × 13 periods (Jan–May) |
| CI consensus 2024/2025 | `output/production/java_ci_consensus_2024_2025.tif` (+ `_mean`, `_min`) | 2-yr |
| CI=3 / Bengawan Solo fig | `output/production/ci3_highlight_2024_2025.png` | |
| Semarang validation pts | `output/validation/semarang_veg_p13.{csv,geojson,kml}` | 100 pts, lag-adjusted |
| Follow-up report | `landcover/.../2026/laporan/LAPORAN_LANJUTAN_MOGPR_S1S2_2026.{md,tex,pdf}` | Indonesian |

Per-tile cache (`cube.nc` + `fused.npz`, ~8 GB/year) saved for 2024 & 2025 → re-analysis is minutes.

## 3. Final numbers (multi-season model + min_run 3)

| Year | Detected paddy | IP | Harvest area | Production |
|---|---|---|---|---|
| 2024 | 2,278,550 ha | **2.18** | 4.96 M ha | **28.8 Mt** |
| 2025 | 2,278,533 ha | **2.22** | 5.06 M ha | **29.4 Mt** |

Harvest area ~5.0 M ha ≈ BPS luas panen (~5.34 M). **Method demonstration, not official**
(detected-mask denominator + flat 5.8 t/ha). IP 2025 > 2024 = real wet-year signal.

## 4. Model & validation
- **MOGPR** fuses SNAP VH (`java_vh_2024_2026_50m.tif`) + MPC Sentinel-2 NDVI → one curve/pixel.
- **MiniROCKET + LightGBM** classifier on the fused curve; 6-class (fase 1–6) + 3-class collapse.
- Phase F1 (leave-one-region-out, 7 regions): generative 67–68%, 3-class ~70%, 6-class ~51%.
  VH-only 52% → +MOGPR 67% (optical is the dominant signal; beats the idmai VH-CNN V3 = 64%).
- **Multi-season fix (the key correction):** original training (`series_0104`, Mar–Jun, 0% dry
  season) failed on dry season (held-out dry acc 36.9%). Added idmai `6fase.csv` dry subset
  (Jul–Sep, 34,143 pts) → `series_multiseason` → held-out dry acc **78.6%**, genF1 **91.6%**,
  no wet penalty. Rebalancing tested and rejected (loses dry signal). `compare_multiseason_f1.py`.
- **min_run 2→3** for episode counting (generative ≈ 35 d ≈ 3 periods) → IP≥4 non-physical
  18% → 2.7%. This is why IP rose from the wet-only 1.85/1.94 to 2.18/2.22.

## 5. Fixes made this cycle
- **Patrol/Indramayu gap** = CRS bug in `map_phases_tiled._tile_crs` (west-edge vs centre at the
  108°E UTM boundary). Fixed to centre-based.
- **CI clamp 1–3** (generative-episode over-count tail) + raw `n_harvests` kept separately.
- **PNG render**: max-pool inflated CI=3 → switched to majority-class (mode) downsample.
- **Grid alignment**: mosaics auto-picked ~50.39 m → `reproject_match` to source mask = true
  50 m (0.000449158°), pixel-aligned. Area/production unaffected (computed in native UTM).

## 6. Comparison with the S1-only DL paper
`rice-growth-stage-mapping/paper_latex/manuscript.tex` (CNN, 87.34% OA, 4-class, in-distribution
CV) is NOT directly comparable to ours (S1+S2, 6/3-class, generative-F1, leave-one-region-out).
Agree on SNAP>GEE, VH, 12-day, scale. Its single-season-collapse finding (21.89%) motivated our
multi-season fix. Our work is the S1+S2 fusion the paper lists as future work.

## 7. Impact on the four TA-2026 landcover reports
Binary-paddy S1 results (L1) stand. What's strengthened: the growth-stage/IP/production narrative.
We supply the validated phase model L2 admitted it lacked (32% field agreement), realise the S1+S2
fusion on the Kegiatan roadmap early, and add production estimation. Documented in the follow-up
report (BAB 4 = per-report impact).

## 8. Open items
1. **Combined-time-series CI — DONE** (`scripts/build_combined_ci.py` →
   `output/production/java_combined_2024_2025/`): stitches each pixel's cached 2024+2025 fused
   curves into one 61-period series, re-classifies (window spans the year boundary), counts
   generative episodes over 24 months ÷2. **Mean annual IP 2.215, production 29.5 Mt/yr** (vs
   per-year 2.18/2.22; simple average 2.20 — the +small uplift = boundary-spanning seasons counted
   once). A true joint MOGPR re-fuse (O(n³), ~2 days) was deemed not worth it; the stitch reuses
   cache (minutes) and delivers the boundary benefit.
2. **Semarang field survey** → build observed-vs-predicted scorer.
3. **Multi-DI field validation (TA 2027)** to confirm the multi-season model on a spatial split.
4. **Ingest p14/p15** via `INGEST_NEW_PERIOD.md`.
5. **IP reconciliation** with the landcover S1 cycle-count IP (different denominator/method).

## 9. Repro pointers
- Scripts: `scripts/{build_cube_from_snap_s1,produce_annual_tiled,map_phases_tiled,mosaic_ci_map,
  extract_point_series,compare_multiseason_f1,make_validation_points}.py`
- Series: `output/phase_model/series_multiseason.{npz,_meta.csv}` (43,610 pts = 9,467 wet + 34,143 dry)
- Adopted config: `--series series_multiseason --min-run 3`
- Pushed: FuseTS `163acdd`, landcover `44b801f`.
