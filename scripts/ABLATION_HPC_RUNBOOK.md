# Fusion ablation — HPC runbook

Run the fusion-vs-baseline ablation (does open-source MOGPR fusion earn its place, or does
"just having optical"?) on the curated 14,187-point multi-season set.

Companion files:
- `scripts/extract_point_series.py` — patched to also emit `ndvi_naive` (naive linear-interp
  optical, no SAR) + `ndvi_valid_frac` (for cloud stratification).
- `scripts/ablation_fusion.py` — the 5-arm ablation (A/B/C/D/Dopt), reusing the same
  `_window` + MiniRocket(2000) + LightGBM + leave-one-region-out as `train_v3_mogpr_ensemble.py`.
- `scripts/loro_eval.py` — canonical harness; gives the endpoint arms (VH-only → fused) for the
  sanity cross-check.

**Arms** (channels windowed around each label date; d = temporal gradient):

| Arm | Channels | Isolates |
|---|---|---|
| A · VH_only | [VH, dVH] | SAR single-sensor (status quo) |
| B · NDVI_naive | [NDVInaive, dNDVInaive] | optical, linear-interp gap-fill, **no SAR** |
| C · VH_NDVI_naive | [VH, NDVInaive] | both sensors, **no MOGPR** |
| D · VH_NDVI_mogpr | [VH, NDVIfused] | full method |
| Dopt · OPT_mogpr | [NDVIfused, dNDVIfused] | MOGPR optical-only |

**Crux contrast = Dopt vs B**: identical optical-only pipeline, only fusion vs naive interp
differs. `Dopt >> B` → MOGPR earns its place (recommend FuseTS/MOGPR to Kementan). `Dopt ≈ B` →
the value is optical presence, not MOGPR (recommend: add Sentinel-2; cheap interp suffices) —
still a clean, honest finding.

---

## 0. Get the code
```bash
cd ~/Github/FuseTS          # or wherever the HPC clone lives
git status                  # if the tree is dirty: git stash
git pull origin main        # brings the ablation commit (extract patch + ablation_fusion.py)
```

## 1. Activate the MOGPR environment
```bash
conda activate <your-mogpr-env>     # the env from environment.yml / H100_SETUP_GUIDE.md
python -c "import sktime, lightgbm, fusets, scipy; print('deps ok')"
```

## 2. Re-extract the curated set WITH the naive baseline
Heavy step: re-runs MOGPR over all 14,187 points and hits Microsoft Planetary Computer for S2.
Run in `tmux`/`screen` (or a batch job) on a node with **internet access to Planetary Computer**
— that's the usual gotcha. No GPU needed (multiprocessing CPU work).
```bash
tmux new -s extract
python scripts/extract_point_series.py \
    --points data/aois/points_combined_java.csv \
    --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
    --year 2024 --cell 0.05 --res 50 --workers 16 \
    --out output/phase_model/series_combined_java
```
Adjust `--vh-stack` to wherever the 50 m VH stack actually lives on the HPC.

Verify the patch took effect (must list `ndvi_naive`):
```bash
python -c "import numpy as np; z=np.load('output/phase_model/series_combined_java.npz'); print(sorted(z.files))"
```

## 3. Sanity cross-check (canonical harness)
```bash
python scripts/loro_eval.py output/phase_model/series_combined_java
```
Read the last line: `ABLATION generative per-region F1: VH-only X -> fused Y`. The ablation's
`A_VH_only` and `Dopt_OPT_mogpr` must reproduce X and Y. If they do, the harness is wired right.
(On the old `series_0104` set the endpoints were ~0.52 → ~0.68; on the 14,187 set they may shift —
trust the cross-check, not the old absolutes.)

## 4. Run the 5-arm ablation
```bash
python scripts/ablation_fusion.py \
    --series output/phase_model/series_combined_java \
    --window 8 \
    --out output/phase_model/ablation_combined
```
Prints the arm table + crux contrasts (Wilcoxon across regions) + cloud-stratified gain, and
writes `output/phase_model/ablation_combined_ablation.json`.

## 5. Collect the result
```bash
cat output/phase_model/ablation_combined_ablation.json
```
Keep this JSON (+ the `loro_eval` output) — it feeds Results §4.1 of the paper: the arm table,
the cloud-stratified figure, and the interpretation keyed to the decision rule.

---

## Watch-outs
- **Internet on the compute node** — step 2 needs MPC/STAC access. If batch nodes are offline,
  run the extraction on a login/data-transfer node.
- **`git pull` complains about local changes** — `git stash` first; the HPC's untracked
  outputs/series are safe and stay.
- **Extraction errors on a specific cell / S2 query** — capture the traceback; that is the one
  place setup differences bite.

## Decision rule (what the numbers mean)
- `Dopt >> B` and `D >> C` (≥ +4–5 generative-F1, significant) → fusion justified; recommend
  FuseTS/MOGPR. Cloud-stratified gain should be largest where S2 valid-fraction is lowest.
- `Dopt ≈ B` or `D ≈ C` (within ~1–2 F1, n.s.) → optical is the lever, not MOGPR; recommend adding
  Sentinel-2 with cheap interpolation, MOGPR as robustness in the cloudiest regions.
Either outcome is publishable at RSASE / Smart Agricultural Technology.
