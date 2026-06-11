#!/usr/bin/env python
"""Validate MOGPR-derived rice growth phase against drone `fase` labels (idmai 6fase).

For each labelled point we sample the (VH + NDVI) series from a cube built by
`build_cube_from_snap_s1.py` (your SNAP S1 + MPC NDVI), MOGPR-fuse it, detect seasons
(peakvalley), and classify the point's growth stage at its *observation date* from where
that date sits on the fused NDVI curve:

    label_date in [SOS, POS)   -> vegetative   (≈ fase 2–3)
    label_date in [POS, EOS]   -> generative   (≈ fase 4–5)   <-- harvest-relevant
    otherwise (trough / bare)  -> bare/other   (≈ fase 1 or 6)

Headline metric: **phase-4-5 detection** (generative vs not) — the class the
production estimate (area × t/ha) depends on. Also reports a 3-class confusion.

Truth grouping: fase {4,5}=generative, {2,3}=vegetative, {1,6}=bare/other.

Usage
-----
  python scripts/validate_phase_vs_drone.py \
      --cube output/brebes_snap_2024/cube.nc --crs EPSG:32749 \
      --points data/aois/brebes_points_proof.csv \
      --drop-thr 0.20 --out output/brebes_snap_2024
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
from pyproj import Transformer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pv_phenology import _seasons_from_pydate  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402


def _fuse_point(ndvi_v, vv_v, vh_v, t_ord, out_grid):
    nv = np.isfinite(ndvi_v)
    if nv.sum() < 2:
        return None
    data, tin = [ndvi_v[nv]], [t_ord[nv]]
    for v in (vv_v, vh_v):
        m = np.isfinite(v)
        if m.sum() >= 4:
            data.append(v[m]); tin.append(t_ord[m])
    if len(data) < 2:           # need ≥2 series to fuse (NDVI + at least one SAR)
        return None
    try:
        om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
        return np.ravel(om[0])
    except Exception:
        return None


def _classify(label_date, seasons):
    """Return 'generative' | 'vegetative' | 'bare' from curve position at label_date."""
    ld = pd.Timestamp(label_date)
    for s in seasons:
        if s["POS"] <= ld <= s["EOS"]:
            return "generative"
    for s in seasons:
        if s["SOS"] <= ld < s["POS"]:
            return "vegetative"
    return "bare"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--crs", default="EPSG:32749")
    p.add_argument("--points", required=True, help="CSV with bujur,lintang,tanggal,fase")
    p.add_argument("--drop-thr", type=float, default=0.20)
    p.add_argument("--period-days", type=int, default=12)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cube = xr.open_dataset(a.cube).rio.write_crs(a.crs)
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, a.period_days, float)
    x_py = np.array([datetime.fromordinal(int(o)) for o in out_grid], dtype=object)

    pts = pd.read_csv(a.points)
    pts["tanggal"] = pd.to_datetime(pts["tanggal"], errors="coerce")
    pts = pts.dropna(subset=["tanggal", "bujur", "lintang", "fase"])
    # reproject point lon/lat -> cube CRS, snap to nearest pixel
    tf = Transformer.from_crs("EPSG:4326", a.crs, always_xy=True)
    px, py = tf.transform(pts["bujur"].values, pts["lintang"].values)
    xs = xr.DataArray(px, dims="pt"); ys = xr.DataArray(py, dims="pt")
    samp = cube[["S2ndvi", "VV", "VH"]].sel(x=xs, y=ys, method="nearest")
    NDVI = samp["S2ndvi"].values; VV = samp["VV"].values; VH = samp["VH"].values  # (t, pt)

    truth_map = {1: "bare", 6: "bare", 2: "vegetative", 3: "vegetative", 4: "generative", 5: "generative"}
    rows = []
    for i in range(len(pts)):
        fused = _fuse_point(NDVI[:, i], VV[:, i], VH[:, i], t_ord, out_grid)
        if fused is None:
            pred = None
        else:
            seasons = _seasons_from_pydate(x_py, fused, a.drop_thr)
            pred = _classify(pts["tanggal"].iloc[i], seasons)
        rows.append({"fase": int(pts["fase"].iloc[i]),
                     "truth3": truth_map[int(pts["fase"].iloc[i])],
                     "pred3": pred})
    r = pd.DataFrame(rows)
    r.to_csv(out / "phase_validation_points.csv", index=False)

    ok = r.dropna(subset=["pred3"])
    n = len(ok); nf = len(r) - n
    # --- phase-4-5 (generative) binary ---
    tp = int(((ok.truth3 == "generative") & (ok.pred3 == "generative")).sum())
    fp = int(((ok.truth3 != "generative") & (ok.pred3 == "generative")).sum())
    fn = int(((ok.truth3 == "generative") & (ok.pred3 != "generative")).sum())
    tn = int(((ok.truth3 != "generative") & (ok.pred3 != "generative")).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc3 = float((ok.truth3 == ok.pred3).mean())
    # --- 3-class confusion ---
    cls = ["bare", "vegetative", "generative"]
    cm = pd.crosstab(ok.truth3, ok.pred3).reindex(index=cls, columns=cls, fill_value=0)

    summary = {
        "n_points": int(len(r)), "n_fused": n, "n_failed_fusion": int(nf),
        "drop_thr": a.drop_thr,
        "phase45_precision": round(prec, 4), "phase45_recall": round(rec, 4),
        "phase45_f1": round(f1, 4),
        "phase45_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "overall_3class_accuracy": round(acc3, 4),
        "confusion_3class_rows_truth_cols_pred": cm.to_dict(),
    }
    (out / "phase_validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"points: {len(r)} | fused: {n} | fusion failed: {nf}")
    print(f"PHASE 4-5 (generative): precision {prec:.1%}  recall {rec:.1%}  F1 {f1:.1%}")
    print(f"3-class accuracy: {acc3:.1%}")
    print("3-class confusion (rows=truth, cols=pred):")
    print(cm.to_string())
    print(f"wrote -> {out}/phase_validation_*.csv/.json")
    return summary


if __name__ == "__main__":
    main()
