#!/usr/bin/env python
"""Fusion ablation: does open-source MOGPR fusion earn its place, or does 'just having optical'?

Varies ONLY the input representation into the SAME MiniROCKET + LightGBM classifier, under the
SAME leave-one-region-out (LORO) protocol and drone-phase labels used by
train_v3_mogpr_ensemble.py. Everything else (windowing, kernels, classifier, folds) is held
constant so differences are attributable to the input alone.

Arms (channels are windowed around each label date; d = temporal gradient):
  A  VH_only        [VH,  dVH]                 -- SAR single-sensor (operational status quo)
  B  NDVI_naive     [NDVInaive, dNDVInaive]    -- optical, LINEAR-interp gap-fill, NO SAR
  C  VH_NDVI_naive  [VH,  NDVInaive]           -- both sensors, NO MOGPR (cheap multi-sensor)
  D  VH_NDVI_mogpr  [VH,  NDVIfused]           -- full method (VH + MOGPR-fused NDVI)
  Dopt OPT_mogpr    [NDVIfused, dNDVIfused]    -- MOGPR optical-only (report's ~68% anchor)

CRUX contrast = Dopt vs B: identical optical-only pipeline, the ONLY difference is
MOGPR fusion vs naive linear interpolation. Dopt >> B  => the fusion method earns its place
(recommend FuseTS/MOGPR to the ministry). Dopt ~= B  => the value is 'optical presence', not
MOGPR (recommend: add Sentinel-2; cheap interpolation suffices) -- still a clean, honest finding.

Requires series npz built by the PATCHED extract_point_series.py (adds `ndvi_naive` array and
`ndvi_valid_frac` meta column). Re-run the extractor once to regenerate the series files.

Usage
-----
  python scripts/ablation_fusion.py --series output/phase_model/series_0104 \
      --window 8 --out output/phase_model/ablation
  # pool wet + dry season (both must be re-extracted with the patch):
  python scripts/ablation_fusion.py --series output/phase_model/series_0104 \
      --series-extra output/phase_model/series_dry6fase --window 8 \
      --out output/phase_model/ablation_multiseason
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_v3_mogpr_ensemble import _window, _binary, TO3  # noqa: E402

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy optional
    wilcoxon = None


def _load(name):
    z = np.load(name + ".npz")
    m = pd.read_csv(name + "_meta.csv")
    vfrac = m["ndvi_valid_frac"].values if "ndvi_valid_frac" in m.columns \
        else np.full(len(m), np.nan)
    return (z["ndvi"], z["ndvi_naive"], z["vh"], z["t_grid"], z["label_ord"],
            m["fase"].astype(int).values, m["region"].astype(str).values, vfrac)


def _oof_pred(channels, y3, groups):
    """channels: list of (N, 2W+1) windows -> (N, C, 2W+1). Out-of-fold LORO 3-class predictions."""
    X = np.stack(channels, axis=1).astype("float32")
    pred = np.empty(len(y3), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y3, groups):
        mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X[tr])
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             class_weight="balanced", verbose=-1).fit(mr.transform(X[tr]).values, y3[tr])
        pred[te] = clf.predict(mr.transform(X[te]).values)
    return pred


def _score(pred, y3, groups):
    tg = (y3 == "generative"); pg = (pred == "generative")
    per = {str(r): _binary(tg[groups == r], pg[groups == r])["f1"] for r in pd.unique(groups)}
    return {"gen_f1": _binary(tg, pg)["f1"], "acc3": round(float((y3 == pred).mean()), 4),
            "per_region": per}


def _contrast(res, hi, lo):
    regs = list(res[hi]["per_region"].keys())
    h = np.array([res[hi]["per_region"][r] for r in regs])
    l = np.array([res[lo]["per_region"][r] for r in regs])
    p = float("nan")
    if wilcoxon is not None and np.any(h - l != 0):
        try:
            p = round(float(wilcoxon(h, l).pvalue), 4)
        except Exception:
            pass
    return {"delta_gen_f1": round(res[hi]["gen_f1"] - res[lo]["gen_f1"], 4),
            "wilcoxon_p_per_region": p, "n_regions": len(regs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--series-extra", default=None, help="optional 2nd npz to pool (e.g. dry season)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    W = a.window

    fused, naive, vh, tg, lab, fase, region, vfrac = _load(a.series)
    if a.series_extra:
        f2, n2, v2, _tg2, lab2, fa2, rg2, vf2 = _load(a.series_extra)
        L = min(fused.shape[1], f2.shape[1])
        fused = np.concatenate([fused[:, :L], f2[:, :L]]); naive = np.concatenate([naive[:, :L], n2[:, :L]])
        vh = np.concatenate([vh[:, :L], v2[:, :L]]); lab = np.concatenate([lab, lab2])
        fase = np.concatenate([fase, fa2]); region = np.concatenate([region, rg2])
        vfrac = np.concatenate([vfrac, vf2]); tg = tg[:L]

    fused = np.nan_to_num(fused); naive = np.nan_to_num(naive); vh = np.nan_to_num(vh)
    y3 = pd.Series(fase).map(TO3).values
    print(f"pool: {len(y3)} points | regions={len(np.unique(region))} | "
          f"gen%={100 * np.isin(fase, [4, 5]).mean():.1f}")

    # windowed channels (+ gradients)
    wv = _window(vh, tg, lab, W);     dv = np.gradient(wv, axis=1).astype("float32")
    wf = _window(fused, tg, lab, W);  df = np.gradient(wf, axis=1).astype("float32")
    wn = _window(naive, tg, lab, W);  dn = np.gradient(wn, axis=1).astype("float32")

    arms = {
        "A_VH_only":       [wv, dv],
        "B_NDVI_naive":    [wn, dn],
        "C_VH_NDVI_naive": [wv, wn],
        "D_VH_NDVI_mogpr": [wv, wf],
        "Dopt_OPT_mogpr":  [wf, df],
    }

    res, preds = {}, {}
    for name, ch in arms.items():
        preds[name] = _oof_pred(ch, y3, region)
        res[name] = _score(preds[name], y3, region)
        print(f"  {name:16s} genF1={res[name]['gen_f1']:.3f}  acc3={res[name]['acc3']:.3f}")

    contrasts = {
        "CRUX__Dopt_vs_B__mogpr_vs_naive_optical": _contrast(res, "Dopt_OPT_mogpr", "B_NDVI_naive"),
        "D_vs_C__fused_vs_naive_with_VH":          _contrast(res, "D_VH_NDVI_mogpr", "C_VH_NDVI_naive"),
        "Dopt_vs_A__optical_vs_SAR":               _contrast(res, "Dopt_OPT_mogpr", "A_VH_only"),
    }

    # cloud-stratified fusion gain: gen-F1(Dopt) - gen-F1(B) by S2 valid-observation fraction
    cloud = {}
    if np.isfinite(vfrac).any():
        tgb = (y3 == "generative")
        for lo, hi, name in [(-0.01, 0.3, "<30%"), (0.3, 0.6, "30-60%"), (0.6, 1.01, ">60%")]:
            sel = np.isfinite(vfrac) & (vfrac >= lo) & (vfrac < hi)
            if sel.sum() < 20:
                cloud[name] = {"n": int(sel.sum()), "note": "too few"}; continue
            fB = _binary(tgb[sel], preds["B_NDVI_naive"][sel] == "generative")["f1"]
            fD = _binary(tgb[sel], preds["Dopt_OPT_mogpr"][sel] == "generative")["f1"]
            cloud[name] = {"n": int(sel.sum()), "naive_genF1": fB, "mogpr_genF1": fD,
                           "gain": round(fD - fB, 4)}

    out = {"n": int(len(y3)), "window": W, "arms": res, "contrasts": contrasts,
           "cloud_stratified_gain": cloud}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + "_ablation.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n=== crux contrasts (generative-phase F1) ===")
    for k, v in contrasts.items():
        print(f"  {k}: dF1={v['delta_gen_f1']:+.3f}  p(region)={v['wilcoxon_p_per_region']}")
    if cloud:
        print("\n=== cloud-stratified fusion gain (Dopt - B), by S2 valid fraction ===")
        for k, v in cloud.items():
            if "gain" in v:
                print(f"  {k:7s} n={v['n']:4d}  naive={v['naive_genF1']:.3f}  "
                      f"mogpr={v['mogpr_genF1']:.3f}  gain={v['gain']:+.3f}")
    print(f"\nwrote -> {a.out}_ablation.json")
    return out


if __name__ == "__main__":
    main()
