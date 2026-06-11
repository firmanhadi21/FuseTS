#!/usr/bin/env python
"""Temporal phase classifier: MiniROCKET on the MOGPR-fused NDVI+VH curve, windowed
around the observation date, with per-location normalization — the recommended upgrade
over scalar-features+RF.

Pipeline
--------
1. Window the fused NDVI + VH series ±W periods around each point's observation date
   (translation-aligns "where on the curve" the label sits). Edge-padded.
2. Per-location normalization: z-score VH within each lokasi (removes regional/calibration
   offsets — fixes the leave-one-region-out collapse, e.g. Pekalongan). NDVI kept (already
   physical), plus its windowed slope channel.
3. Channels fed to MiniRocketMultivariate: [NDVI, VH(loc-norm), dNDVI].
4. Head: RidgeClassifierCV (canonical) and LightGBM (boosted) — report both.
5. Leave-one-region-out CV. Targets: phase-4-5 (generative), 3-class, 6-class.

Compares to the scalar-RF baseline (phase-4-5 F1 ~59%).

Usage
-----
  python scripts/train_rocket_classifier.py \
      --series output/phase_model/series_0104 --window 8 --out output/phase_model
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import LeaveOneGroupOut
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

TO3 = {1: "bare", 6: "bare", 2: "vegetative", 3: "vegetative", 4: "generative", 5: "generative"}
CLS = ["bare", "vegetative", "generative"]


def _window(series, t_grid, label_ord, W):
    """Extract ±W samples of `series` (N,L) centered on the nearest-to-label index."""
    N, L = series.shape
    idx = np.abs(t_grid[None, :] - label_ord[:, None]).argmin(axis=1)
    out = np.empty((N, 2 * W + 1), series.dtype)
    for i in range(N):
        lo, hi = idx[i] - W, idx[i] + W + 1
        s = series[i]
        seg = np.concatenate([np.full(max(0, -lo), s[0]), s[max(0, lo):min(L, hi)],
                              np.full(max(0, hi - L), s[-1])])
        out[i] = seg[:2 * W + 1]
    return out


def _binary(truth_gen, pred_gen):
    tp = int((truth_gen & pred_gen).sum()); fp = int((~truth_gen & pred_gen).sum())
    fn = int((truth_gen & ~pred_gen).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", required=True, help="prefix: <series>.npz + <series>_meta.csv")
    p.add_argument("--window", type=int, default=8, help="±periods around label date")
    p.add_argument("--kernels", type=int, default=2000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.series + ".npz")
    meta = pd.read_csv(a.series + "_meta.csv")
    ndvi, vh, t_grid, label_ord = z["ndvi"], z["vh"], z["t_grid"], z["label_ord"]
    meta = meta.reset_index(drop=True)

    # per-location VH normalization (remove regional offsets)
    vhn = vh.copy().astype("float32")
    for loc, idx in meta.groupby("lokasi").groups.items():
        ix = np.array(list(idx))
        seg = vh[ix]; mu, sd = np.nanmean(seg), np.nanstd(seg) + 1e-6
        vhn[ix] = (seg - mu) / sd
    ndvi = np.nan_to_num(ndvi, nan=0.0); vhn = np.nan_to_num(vhn, nan=0.0)

    W = a.window
    w_ndvi = _window(ndvi, t_grid, label_ord, W)
    w_vh = _window(vhn, t_grid, label_ord, W)
    w_dndvi = np.gradient(w_ndvi, axis=1).astype("float32")
    X = np.stack([w_ndvi, w_vh, w_dndvi], axis=1).astype("float32")  # (N, 3, 2W+1)

    y6 = meta["fase"].values
    y3 = meta["fase"].map(TO3).values
    groups = meta["region"].values
    print(f"{len(meta)} series | window {2*W+1} | channels 3 | regions {len(set(groups))}")

    logo = LeaveOneGroupOut()
    results = {}
    for head_name in ("ridge", "lgbm"):
        pred3 = np.empty(len(meta), dtype=object)
        per_region = []
        for tr, te in logo.split(X, y3, groups):
            reg = meta["region"].iloc[te[0]]
            mr = MiniRocketMultivariate(num_kernels=a.kernels, random_state=0).fit(X[tr])
            Ftr, Fte = mr.transform(X[tr]).values, mr.transform(X[te]).values
            if head_name == "ridge":
                clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                        class_weight="balanced").fit(Ftr, y3[tr])
            else:
                clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                     class_weight="balanced", verbose=-1).fit(Ftr, y3[tr])
            pr = clf.predict(Fte); pred3[te] = pr
            tg = (y3[te] == "generative"); pg = (pr == "generative")
            per_region.append({"region": reg, "n": int(len(te)), "n_gen": int(tg.sum()),
                               **{f"p45_{k}": v for k, v in _binary(tg, pg).items()},
                               "acc3": round(float((y3[te] == pr).mean()), 4)})
        tg = (y3 == "generative"); pg = (pred3 == "generative")
        cm = pd.crosstab(pd.Series(y3, name="truth"), pd.Series(pred3, name="pred")).reindex(
            index=CLS, columns=CLS, fill_value=0)
        results[head_name] = {
            "phase45": _binary(tg, pg),
            "acc_3class": round(float((y3 == pred3).mean()), 4),
            "confusion_3class": cm.to_dict(),
            "per_region": per_region,
        }
        meta[f"pred3_{head_name}"] = pred3
        print(f"\n[{head_name}] phase4-5  P={results[head_name]['phase45']['precision']:.1%} "
              f"R={results[head_name]['phase45']['recall']:.1%} "
              f"F1={results[head_name]['phase45']['f1']:.1%} | 3-class acc {results[head_name]['acc_3class']:.1%}")
        print(f"[{head_name}] per-region phase-4-5 F1:")
        for r in sorted(per_region, key=lambda r: -r["p45_f1"]):
            print(f"    {r['region']:16s} n={r['n']:4d} gen={r['n_gen']:4d}  F1={r['p45_f1']:.0%}")

    results["baseline_scalar_rf_phase45_f1"] = 0.59
    (out / "rocket_summary.json").write_text(json.dumps(results, indent=2))
    meta.to_csv(out / "series_0104_predicted.csv", index=False)
    best = max(("ridge", "lgbm"), key=lambda h: results[h]["phase45"]["f1"])
    print(f"\nBEST head: {best} (phase-4-5 F1 {results[best]['phase45']['f1']:.1%}) "
          f"vs scalar-RF baseline 59%")
    print(f"wrote -> {out}/rocket_summary.json")
    return results


if __name__ == "__main__":
    main()
