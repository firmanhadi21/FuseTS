#!/usr/bin/env python
"""Held-out F1 comparison: wet-only vs naive-multi-season vs rebalanced training.

Fair test of whether adding dry-season labels makes the phase classifier MORE ACCURATE
(not just predict more generative). MiniROCKET is fit ONCE on the common training pool so
only the LGBM training-label composition varies. A common stratified 25% test set (by
season x fase) is held out from all three. Reports 6-class accuracy and generative (fase
4 or 5, 3-class collapse) F1, split into wet-test / dry-test / all.

  A wet-only      : series_0104 (Mar-Jun)             -> the current production model
  B naive-multi   : wet + ALL dry (45% generative)    -> the over-corrected run
  C rebalanced    : wet + dry subsampled to wet's class proportions
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_v3_mogpr_ensemble import _window  # noqa: E402
from sktime.transformations.panel.rocket import MiniRocketMultivariate  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

W = 8


def load(name):
    z = np.load(f"output/phase_model/{name}.npz")
    m = pd.read_csv(f"output/phase_model/{name}_meta.csv")
    return z, m


def main():
    zw, mw = load("series_0104")
    zd, md = load("series_dry6fase")
    L = min(zw["ndvi"].shape[1], zd["ndvi"].shape[1])
    ndvi = np.concatenate([zw["ndvi"][:, :L], zd["ndvi"][:, :L]])
    lab = np.concatenate([zw["label_ord"], zd["label_ord"]])
    tg = zw["t_grid"][:L]
    fase = np.concatenate([mw["fase"].values, md["fase"].values]).astype(int)
    season = np.array(["wet"] * len(mw) + ["dry"] * len(md))

    wn = _window(np.nan_to_num(ndvi), tg, lab, W)
    dn = np.gradient(wn, axis=1).astype("float32")
    X = np.stack([wn, dn], 1).astype("float32")
    N = len(fase)
    print(f"pool: {N} ({len(mw)} wet + {len(md)} dry) | wet gen%={100*np.isin(fase[season=='wet'],[4,5]).mean():.1f} "
          f"dry gen%={100*np.isin(fase[season=='dry'],[4,5]).mean():.1f}")

    rng = np.random.default_rng(0)
    test = np.zeros(N, bool)
    for s in ("wet", "dry"):
        for f in range(1, 7):
            idx = np.where((season == s) & (fase == f))[0]
            if len(idx) == 0:
                continue
            k = max(1, int(round(0.25 * len(idx))))
            test[rng.choice(idx, k, replace=False)] = True
    train = ~test

    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X[train])
    Xt = mr.transform(X).values
    print(f"MiniROCKET fit on {int(train.sum())} train; test={int(test.sum())}")

    wt = train & (season == "wet")
    dt = train & (season == "dry")
    A = wt.copy()
    B = train.copy()
    # C: add dry per class matching wet's class COUNTS (-> merged keeps wet's 23% gen prior, 2x data)
    wc = Counter(fase[wt])
    C = wt.copy()
    for f in range(1, 7):
        didx = np.where(dt & (fase == f))[0]
        want = min(wc.get(f, 0), len(didx))
        if want > 0:
            C[rng.choice(didx, want, replace=False)] = True

    yt = fase[test]
    gen = lambda a: np.isin(a, [4, 5]).astype(int)
    seas_t = season[test]

    def run(mask, name):
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             class_weight="balanced", verbose=-1).fit(Xt[mask], fase[mask])
        pred = clf.predict(Xt[test])
        tr_gen = 100 * np.isin(fase[mask], [4, 5]).mean()
        row = f"{name} (n_train={int(mask.sum()):>5}, gen%={tr_gen:4.1f}) | "
        for seg in ("all", "wet", "dry"):
            sel = np.ones(test.sum(), bool) if seg == "all" else (seas_t == seg)
            acc = accuracy_score(yt[sel], pred[sel])
            gf1 = f1_score(gen(yt[sel]), gen(pred[sel]))
            row += f"{seg}: acc6 {acc*100:4.1f} genF1 {gf1*100:4.1f}  "
        print(row)

    print("\n=== held-out F1 (6-class accuracy + generative[4,5] 3-class-collapse F1) ===")
    run(A, "A wet-only   ")
    run(B, "B naive-multi")
    run(C, "C rebalanced ")
    print("\n(dry-test genF1 is the key column: does multi-season actually classify dry-season "
          "generative better, vs just predicting more of it?)")


if __name__ == "__main__":
    main()
