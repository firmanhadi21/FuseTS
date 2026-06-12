#!/usr/bin/env python
"""Faithful ensemble: your ACTUAL V3 VH-CNN ⊕ MOGPR-optical, on the 0104 points.

Does adding MOGPR's (held-out) optical signal lift your real V3 model?

Arms (3-class: bare/vegetative/generative), joined per-point on coordinates:
  V3_only   : V3 CNN class probs -> 3-class (in-sample; your model's own number)
  OPT_only  : MiniROCKET on MOGPR-fused NDVI [NDVI,dNDVI], leave-one-region-out (held-out)
  STACK     : meta-LightGBM on [V3 6 probs + OPT 3 probs], leave-one-region-out

CAVEAT: V3 probs are IN-SAMPLE (the CNN trained on these labels). So STACK > V3_only is a
*strong* signal that MOGPR adds orthogonal value (held-out optical beating an in-sample VH
model); STACK ≈ V3_only is inconclusive (V3's in-sample edge can mask the gain).

Usage
-----
  python scripts/train_v3_mogpr_ensemble.py \
      --series output/phase_model/series_0104 \
      --v3 output/phase_model/v3_cnn/v3_predictions.csv \
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


def _opt_oof(w_ndvi, d_ndvi, y3, groups):
    X = np.stack([w_ndvi, d_ndvi], axis=1).astype("float32")
    proba = np.zeros((len(y3), 3))
    for tr, te in LeaveOneGroupOut().split(X, y3, groups):
        mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X[tr])
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             class_weight="balanced", verbose=-1).fit(mr.transform(X[tr]).values, y3[tr])
        pr = clf.predict_proba(mr.transform(X[te]).values)
        col = {c: i for i, c in enumerate(clf.classes_)}
        proba[te] = pr[:, [col[c] for c in CLS]]
    return proba


def _score(pred, y3, groups):
    tg = (y3 == "generative"); pg = (pred == "generative")
    per = [{"region": r, "n": int((groups == r).sum()),
            "p45_f1": _binary(tg[groups == r], pg[groups == r])["f1"]} for r in pd.unique(groups)]
    return {"phase45": _binary(tg, pg), "acc_3class": round(float((y3 == pred).mean()), 4),
            "per_region": sorted(per, key=lambda r: -r["p45_f1"])}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--series", required=True); p.add_argument("--v3", required=True)
    p.add_argument("--window", type=int, default=8); p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv").reset_index(drop=True)
    ndvi, vh, t_grid, label_ord = z["ndvi"], z["vh"], z["t_grid"], z["label_ord"]

    # join V3 probs onto the series rows by coordinate
    v3 = pd.read_csv(a.v3)
    key = lambda df: (df["bujur"].round(6).astype(str) + "_" + df["lintang"].round(6).astype(str))
    meta["k"] = key(meta); v3["k"] = key(v3)
    pcols = [c for c in v3.columns if c.startswith("v3_p")]
    v3j = v3.drop_duplicates("k").set_index("k")[pcols]
    keep = meta["k"].isin(v3j.index).values
    meta = meta[keep].reset_index(drop=True)
    ndvi, vh, label_ord = ndvi[keep], vh[keep], label_ord[keep]
    V3P = v3j.loc[meta["k"]].values.astype("float32")          # (N,6) in fase order 1..6
    print(f"joined {len(meta)} points (V3 ∩ MOGPR)")

    # OPT features
    ndvi = np.nan_to_num(ndvi)
    w_ndvi = _window(ndvi, t_grid, label_ord, a.window)
    d_ndvi = np.gradient(w_ndvi, axis=1).astype("float32")
    y3 = meta["fase"].map(TO3).values; groups = meta["region"].values

    # V3 -> 3-class probs (bare=p1+p6, veg=p2+p3, gen=p4+p5)
    v3_3 = np.stack([V3P[:, 0] + V3P[:, 5], V3P[:, 1] + V3P[:, 2], V3P[:, 3] + V3P[:, 4]], 1)
    v3_3 = v3_3 / v3_3.sum(1, keepdims=True).clip(min=1e-9)

    print("computing OPT-only out-of-fold probs (held-out)...")
    opt = _opt_oof(w_ndvi, d_ndvi, y3, groups)

    # STACK: meta-LightGBM on [V3 6 probs + OPT 3 probs], leave-one-region-out
    Xmeta = np.hstack([V3P, opt]).astype("float32")
    stack_pred = np.empty(len(meta), dtype=object)
    for tr, te in LeaveOneGroupOut().split(Xmeta, y3, groups):
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                             class_weight="balanced", verbose=-1).fit(Xmeta[tr], y3[tr])
        stack_pred[te] = clf.predict(Xmeta[te])

    res = {
        "V3_only": _score(np.array(CLS)[v3_3.argmax(1)], y3, groups),
        "OPT_only": _score(np.array(CLS)[opt.argmax(1)], y3, groups),
        "STACK": _score(stack_pred, y3, groups),
        "caveat": "V3 probs are IN-SAMPLE; OPT/STACK CV is held-out leave-one-region-out.",
    }
    (out / "v3_mogpr_ensemble_summary.json").write_text(json.dumps(res, indent=2))
    print("\n=== phase-4-5 (generative) ===")
    print(f"{'arm':10s} {'P':>6} {'R':>6} {'F1':>6}  acc3   eval")
    ev = {"V3_only": "in-sample", "OPT_only": "held-out", "STACK": "V3 in-sample + OPT held-out"}
    for k in ["V3_only", "OPT_only", "STACK"]:
        s = res[k]["phase45"]
        print(f"{k:10s} {s['precision']:6.1%} {s['recall']:6.1%} {s['f1']:6.1%}  "
              f"{res[k]['acc_3class']:.1%}  {ev[k]}")
    print("\nper-region phase-4-5 F1 (STACK vs V3_only):")
    v3f = {r['region']: r['p45_f1'] for r in res['V3_only']['per_region']}
    for r in res["STACK"]["per_region"]:
        print(f"   {r['region']:16s} n={r['n']:4d}  stack={r['p45_f1']:.0%}  v3={v3f[r['region']]:.0%}")
    d = res["STACK"]["phase45"]["f1"] - res["V3_only"]["phase45"]["f1"]
    print(f"\nSTACK − V3_only = {d:+.1%} F1  (positive => MOGPR optical lifts your model)")
    print(f"wrote -> {out}/v3_mogpr_ensemble_summary.json")
    return res


if __name__ == "__main__":
    main()
