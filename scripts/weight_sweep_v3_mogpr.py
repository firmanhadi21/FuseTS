#!/usr/bin/env python
"""Settle the 'optical-weighted blend' question: sweep the V3-vs-MOGPR blend weight and
find the operating point that maximises held-out phase-4-5 F1.

blend(w) = (1-w)*OPT_optical_probs + w*V3_probs   (3-class)
  w=0 -> pure MOGPR-optical (held-out, 68.2% F1)
  w=1 -> pure V3 VH-CNN     (in-sample probs)

If no w>0 beats w=0, the answer is simply 'use MOGPR-optical'.

Usage
-----
  python scripts/weight_sweep_v3_mogpr.py --series output/phase_model/series_0104 \
      --v3 output/phase_model/v3_cnn/v3_predictions.csv --out output/phase_model
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_v3_mogpr_ensemble import _window, _opt_oof, _binary, CLS, TO3  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--series", required=True); p.add_argument("--v3", required=True)
    p.add_argument("--window", type=int, default=8); p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv").reset_index(drop=True)
    ndvi, t_grid, label_ord = z["ndvi"], z["t_grid"], z["label_ord"]
    v3 = pd.read_csv(a.v3)
    key = lambda df: (df["bujur"].round(6).astype(str) + "_" + df["lintang"].round(6).astype(str))
    meta["k"] = key(meta); v3["k"] = key(v3)
    pcols = [c for c in v3.columns if c.startswith("v3_p")]
    v3j = v3.drop_duplicates("k").set_index("k")[pcols]
    keep = meta["k"].isin(v3j.index).values
    meta = meta[keep].reset_index(drop=True); ndvi, label_ord = ndvi[keep], label_ord[keep]
    V3P = v3j.loc[meta["k"]].values.astype("float32")

    ndvi = np.nan_to_num(ndvi)
    w_ndvi = _window(ndvi, t_grid, label_ord, a.window)
    d_ndvi = np.gradient(w_ndvi, axis=1).astype("float32")
    y3 = meta["fase"].map(TO3).values; groups = meta["region"].values
    print(f"{len(meta)} points | computing OPT-only out-of-fold probs (held-out)...")
    opt = _opt_oof(w_ndvi, d_ndvi, y3, groups)                 # (N,3) in CLS order, held-out

    v3_3 = np.stack([V3P[:, 0] + V3P[:, 5], V3P[:, 1] + V3P[:, 2], V3P[:, 3] + V3P[:, 4]], 1)
    v3_3 = v3_3 / v3_3.sum(1, keepdims=True).clip(min=1e-9)
    tg = (y3 == "generative")
    cls = np.array(CLS)

    rows = []
    for w in np.round(np.linspace(0, 1, 11), 2):
        blend = (1 - w) * opt + w * v3_3
        pred = cls[blend.argmax(1)]
        b = _binary(tg, pred == "generative")
        rows.append({"w_v3": float(w), **b, "acc3": round(float((y3 == pred).mean()), 4)})
    df = pd.DataFrame(rows)
    best = df.loc[df["f1"].idxmax()]
    summary = {"sweep": rows, "best_w_v3": float(best["w_v3"]),
               "best_f1": float(best["f1"]), "pure_optical_f1": float(df.iloc[0]["f1"]),
               "pure_v3_f1": float(df.iloc[-1]["f1"]),
               "gain_over_pure_optical": round(float(best["f1"] - df.iloc[0]["f1"]), 4)}
    (out / "weight_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'w_v3':>5} {'P':>6} {'R':>6} {'F1':>6} {'acc3':>6}")
    for r in rows:
        star = "  <- best" if r["w_v3"] == best["w_v3"] else ""
        print(f"{r['w_v3']:5.1f} {r['precision']:6.1%} {r['recall']:6.1%} {r['f1']:6.1%} {r['acc3']:6.1%}{star}")
    print(f"\npure optical (w=0): F1 {df.iloc[0]['f1']:.1%} | pure V3 (w=1): F1 {df.iloc[-1]['f1']:.1%}")
    print(f"BEST: w_v3={best['w_v3']:.1f} -> F1 {best['f1']:.1%} "
          f"({summary['gain_over_pure_optical']:+.1%} vs pure optical)")
    if summary["gain_over_pure_optical"] <= 0.005:
        print("=> No meaningful gain from blending. RECOMMENDATION: use pure MOGPR-optical.")
    else:
        print(f"=> Optical-weighted blend helps. RECOMMENDATION: w_v3={best['w_v3']:.1f}.")
    return summary


if __name__ == "__main__":
    main()
