#!/usr/bin/env python
"""Run the ACTUAL idmai V3 VH-CNN (model_6fase_enhanced_backward/model_files_V3) on the
0104 drone points, for a faithful head-to-head against the MOGPR-feature classifier.

Reuses the model's own code: `detect_phenology_stages` + `TemperatureLayer` from its
utils.py, its `feature_columns.txt` order, `scaler.joblib`, and `label_encoder.joblib`.

29 features per point (exactly extract_temporal_features_enhanced): a backward 7-period
VH window [P, P-1 ... P-6] + 6 diffs + 6 ratios + 6 phenology flags + 4 VH extrema.
VH values are fed as int16 dB×100 (the scale detect_phenology_stages thresholds expect).

CAVEATS (printed in the summary):
- The V3 model was TRAINED on the GEE 2023/2024 VH stack; here it is applied to the
  rice-growth-stage-mapping SNAP stack java_vh_2024_2026_50m.tif -> possible DOMAIN SHIFT.
- The model was likely trained on these drone labels -> this is IN-SAMPLE (optimistic),
  not the held-out leave-one-region-out used for the MOGPR classifier. Compare with care.

Run with the TF env:
  /opt/conda/envs/geo_ml_env/bin/python scripts/run_v3_cnn.py \
     --points data/aois/points_0104_all.csv \
     --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
     --model-dir ~/work/idmai/DL/vh/model_6fase_enhanced_backward \
     --out output/phase_model/v3_cnn
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import rasterio
import joblib

TO3 = {1: "bare", 6: "bare", 2: "vegetative", 3: "vegetative", 4: "generative", 5: "generative"}


def period_from_date(ts):
    doy = ts.dayofyear
    return min(31, (doy - 1) // 12 + 1)


def build_features(vh_2024, period, detect_phenology_stages, n_prev=6):
    """vh_2024: (31,) int16 dB×100 for 2024 periods 1..31; period: current period (1-based)."""
    n = n_prev + 1
    if period < n:
        return None
    vals = []
    for i in range(n):
        bidx = period - 1 - i               # 0-based band for period (P-i)
        if bidx < 0:
            return None
        v = float(vh_2024[bidx])
        if not np.isfinite(v):
            return None
        vals.append(v)
    feats = list(vals)                       # VH_t0..t6
    feats += [vals[i] - vals[i + 1] for i in range(n - 1)]               # diffs
    feats += [0.0 if abs(vals[i + 1]) < 1e-10 else vals[i] / abs(vals[i + 1])
              for i in range(n - 1)]                                     # ratios
    ph = detect_phenology_stages(vals)
    feats += [float(ph['flooding_detected']), float(ph['early_vegetative']),
              float(ph['late_vegetative']), float(ph['early_generative']),
              float(ph['late_generative']), float(ph['post_harvest'])]
    feats += [ph['vh_min_value'], ph['vh_max_value'],
              (ph['vh_min_idx'] / (n - 1)) if ph['vh_min_idx'] is not None else 0.0,
              (ph['vh_max_idx'] / (n - 1)) if ph['vh_max_idx'] is not None else 0.0]
    if any((not np.isfinite(f)) for f in feats):
        return None
    return feats


def _binary(tg, pg):
    tp = int((tg & pg).sum()); fp = int((~tg & pg).sum()); fn = int((tg & ~pg).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--vh-stack", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-files", default="model_files_V3")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.model_dir).expanduser()
    sys.path.insert(0, str(mdir))
    from utils import detect_phenology_stages, TemperatureLayer  # noqa: E402
    import tensorflow as tf  # noqa: E402

    mf = mdir / a.model_files
    scaler = joblib.load(mf / "scaler.joblib")
    le = joblib.load(mf / "label_encoder.joblib")
    model = tf.keras.models.load_model(mf / "rice_stage_model.keras",
                                       custom_objects={'TemperatureLayer': TemperatureLayer},
                                       compile=False)
    print(f"model loaded | input {model.input_shape} | label classes {list(le.classes_)}")

    d = pd.read_csv(a.points)
    d["tanggal"] = pd.to_datetime(d["tanggal"], errors="coerce")
    d = d.dropna(subset=["tanggal", "bujur", "lintang", "fase"]).reset_index(drop=True)

    # sample the 31 2024-period VH bands at all points (raw int16 dB×100)
    with rasterio.open(str(Path(a.vh_stack).expanduser())) as src:
        nod = src.nodata
        coords = list(zip(d["bujur"].values, d["lintang"].values))
        samp = np.array(list(src.sample(coords, indexes=list(range(1, 32)))), dtype="float32")  # (N,31)
    if nod is not None:
        samp[samp == nod] = np.nan

    X, keep = [], []
    for i in range(len(d)):
        f = build_features(samp[i], period_from_date(d["tanggal"].iloc[i]), detect_phenology_stages)
        if f is not None:
            X.append(f); keep.append(i)
    X = np.array(X, "float32")
    dd = d.iloc[keep].reset_index(drop=True)
    Xs = scaler.transform(X)
    probs = model.predict(Xs, batch_size=4096, verbose=0)
    pred_fase = le.inverse_transform(np.argmax(probs, axis=1)).astype(int)
    dd["pred_fase"] = pred_fase
    for k, c in enumerate(le.classes_):            # save per-class probabilities
        dd[f"v3_p{int(c)}"] = probs[:, k]
    dd["truth3"] = dd["fase"].map(TO3); dd["pred3"] = dd["pred_fase"].map(TO3)
    dd.to_csv(out / "v3_predictions.csv", index=False)

    tg = (dd.truth3 == "generative").values; pg = (dd.pred3 == "generative").values
    acc3 = float((dd.truth3 == dd.pred3).mean()); acc6 = float((dd.fase == dd.pred_fase).mean())
    per_region = []
    for r in pd.unique(dd.region):
        m = (dd.region == r).values
        per_region.append({"region": r, "n": int(m.sum()),
                           "p45_f1": _binary(tg[m], pg[m])["f1"]})
    summary = {"model": "idmai V3 VH-CNN (applied to SNAP stack)",
               "n_points": int(len(dd)), "n_skipped": int(len(d) - len(dd)),
               "evaluation": "IN-SAMPLE / cross-domain (NOT held-out) — see caveats",
               "phase45": _binary(tg, pg), "acc_3class": round(acc3, 4),
               "acc_6class": round(acc6, 4),
               "per_region": sorted(per_region, key=lambda r: -r["p45_f1"])}
    (out / "v3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nV3 VH-CNN on {len(dd)} pts ({len(d)-len(dd)} skipped, period<7 or nodata)")
    print(f"  phase-4-5  P={summary['phase45']['precision']:.1%} R={summary['phase45']['recall']:.1%} "
          f"F1={summary['phase45']['f1']:.1%}")
    print(f"  3-class acc {acc3:.1%} | 6-class acc {acc6:.1%}")
    print("  per-region phase-4-5 F1:", {r['region']: f"{r['p45_f1']:.0%}" for r in summary['per_region']})
    print("  NOTE: in-sample + cross-domain (trained on GEE VH) — not comparable 1:1 to the "
          "held-out MOGPR 66%")
    return summary


if __name__ == "__main__":
    main()
