#!/usr/bin/env python
"""Decompose the contribution of MOGPR optical fusion to VH-based phase prediction, and
build the ensemble — all in one leave-one-region-out framework on the 0104 series.

Three arms (each MiniROCKET + LightGBM, same CV, per-location-normalized VH):
  VH-only   : channels [VH, dVH]            -> stand-in for a VH-only model (your VH-CNN's info)
  OPT-only  : channels [NDVI, dNDVI]        -> the optical signal MOGPR reconstructs
  ENSEMBLE  : average of VH-only & OPT-only class probabilities (late fusion)
Plus CONCAT : channels [NDVI, VH, dNDVI] in one MiniROCKET (the 65.5% model) for reference.

The point: if OPT-only or ENSEMBLE >> VH-only, MOGPR's optical fusion *enhances* VH-based
phase classification — the answer to "will MOGPR enhance the existing VH model".

Usage
-----
  python scripts/train_ensemble.py --series output/phase_model/series_0104 \
      --window 8 --out output/phase_model
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

TO3 = {1: "bare", 6: "bare", 2: "vegetative", 3: "vegetative", 4: "generative", 5: "generative"}
CLS = ["bare", "vegetative", "generative"]


def _window(series, t_grid, label_ord, W):
    N, L = series.shape
    idx = np.abs(t_grid[None, :] - label_ord[:, None]).argmin(axis=1)
    out = np.empty((N, 2 * W + 1), series.dtype)
    for i in range(N):
        lo, hi = idx[i] - W, idx[i] + W + 1
        s = series[i]
        out[i] = np.concatenate([np.full(max(0, -lo), s[0]), s[max(0, lo):min(L, hi)],
                                 np.full(max(0, hi - L), s[-1])])[:2 * W + 1]
    return out


def _binary(tg, pg):
    tp = int((tg & pg).sum()); fp = int((~tg & pg).sum()); fn = int((tg & ~pg).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4))


def _proba_cv(Xchannels, y3, groups):
    """Leave-one-region-out out-of-fold class probabilities for given channel stack."""
    X = np.stack(Xchannels, axis=1).astype("float32")
    classes = np.array(CLS)
    proba = np.zeros((len(y3), 3), float)
    for tr, te in LeaveOneGroupOut().split(X, y3, groups):
        mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X[tr])
        Ftr, Fte = mr.transform(X[tr]).values, mr.transform(X[te]).values
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             class_weight="balanced", verbose=-1).fit(Ftr, y3[tr])
        pr = clf.predict_proba(Fte)
        # align columns to CLS order
        col = {c: i for i, c in enumerate(clf.classes_)}
        proba[te] = pr[:, [col[c] for c in CLS]]
    return proba, classes


def _score(proba, classes, y3, groups, meta):
    pred = classes[proba.argmax(1)]
    tg = (y3 == "generative"); pg = (pred == "generative")
    per_region = []
    for r in pd.unique(groups):
        m = groups == r
        per_region.append({"region": r, "n": int(m.sum()),
                           "p45_f1": _binary(tg[m], pg[m])["f1"]})
    return {"phase45": _binary(tg, pg),
            "acc_3class": round(float((y3 == pred).mean()), 4),
            "per_region": sorted(per_region, key=lambda r: -r["p45_f1"])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", required=True)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv").reset_index(drop=True)
    ndvi, vh, t_grid, label_ord = z["ndvi"], z["vh"], z["t_grid"], z["label_ord"]
    # per-location VH normalization
    vhn = vh.copy().astype("float32")
    for loc, idx in meta.groupby("lokasi").groups.items():
        ix = np.array(list(idx)); seg = vh[ix]
        vhn[ix] = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-6)
    ndvi = np.nan_to_num(ndvi); vhn = np.nan_to_num(vhn)
    W = a.window
    w_ndvi = _window(ndvi, t_grid, label_ord, W); w_vh = _window(vhn, t_grid, label_ord, W)
    d_ndvi = np.gradient(w_ndvi, axis=1).astype("float32"); d_vh = np.gradient(w_vh, axis=1).astype("float32")

    y3 = meta["fase"].map(TO3).values; groups = meta["region"].values
    print(f"{len(meta)} series | window {2*W+1} | LORO over {len(set(groups))} regions")

    arms = {}
    print("fitting VH-only arm..."); pv, cl = _proba_cv([w_vh, d_vh], y3, groups); arms["VH_only"] = pv
    print("fitting OPT-only arm..."); po, _ = _proba_cv([w_ndvi, d_ndvi], y3, groups); arms["OPT_only"] = po
    print("fitting CONCAT (NDVI+VH+dNDVI)..."); pc, _ = _proba_cv([w_ndvi, w_vh, d_ndvi], y3, groups); arms["CONCAT"] = pc
    arms["ENSEMBLE_avg"] = (pv + po) / 2.0

    res = {k: _score(v, cl, y3, groups, meta) for k, v in arms.items()}
    (out / "ensemble_summary.json").write_text(json.dumps(res, indent=2))

    print("\n=== phase-4-5 (generative), leave-one-region-out ===")
    print(f"{'arm':14s} {'P':>6} {'R':>6} {'F1':>6}  acc3")
    for k in ["VH_only", "OPT_only", "CONCAT", "ENSEMBLE_avg"]:
        s = res[k]["phase45"]
        print(f"{k:14s} {s['precision']:6.1%} {s['recall']:6.1%} {s['f1']:6.1%}  {res[k]['acc_3class']:.1%}")
    print("\nper-region phase-4-5 F1 (ENSEMBLE vs VH-only):")
    vh_f1 = {r['region']: r['p45_f1'] for r in res['VH_only']['per_region']}
    for r in res["ENSEMBLE_avg"]["per_region"]:
        print(f"   {r['region']:16s} n={r['n']:4d}  ensemble={r['p45_f1']:.0%}  vh-only={vh_f1[r['region']]:.0%}")
    print(f"\nwrote -> {out}/ensemble_summary.json")
    return res


if __name__ == "__main__":
    main()
