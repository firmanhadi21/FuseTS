#!/usr/bin/env python
"""Train/validate a rice growth-phase classifier on MOGPR phenology features
(from extract_point_features.py) against idmai 0104 drone `fase` labels.

Honest design: **leave-one-region-out** cross-validation (spatial block CV) — every
point is predicted by a model that never saw its region, so the score reflects transfer
to a new area (what all-Java deployment needs). Random splits would leak via spatial
autocorrelation and inflate accuracy.

Targets reported:
  - phase-4-5 (generative, harvest-relevant): the binary that drives production = area×t/ha
  - 3-class: bare {1,6} / vegetative {2,3} / generative {4,5}
  - 6-class fase (reference)

Baseline for context: the rule-based [POS,EOS] readout (in_season & days_to_POS>=0).

Usage
-----
  python scripts/train_phase_classifier.py \
      --features output/phase_model/features_0104.csv --out output/phase_model
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

FEATS = ["ndvi_d", "slope_d", "vh_d", "n_seasons", "days_to_POS", "days_from_SOS",
         "rel_pos", "in_season", "amplitude", "peak_ndvi", "los"]
TO3 = {1: "bare", 6: "bare", 2: "vegetative", 3: "vegetative", 4: "generative", 5: "generative"}
CLS = ["bare", "vegetative", "generative"]


def _binary_scores(truth_gen, pred_gen):
    tp = int((truth_gen & pred_gen).sum()); fp = int((~truth_gen & pred_gen).sum())
    fn = int((truth_gen & ~pred_gen).sum()); tn = int((~truth_gen & ~pred_gen).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trees", type=int, default=300)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(a.features)
    d = d[d["fase"].between(1, 6)].copy()
    for c in FEATS:
        d[c] = d[c].fillna(d[c].median())
    d["truth3"] = d["fase"].map(TO3)
    X = d[FEATS].values
    y6 = d["fase"].values
    y3 = d["truth3"].values
    groups = d["region"].values

    # ---- leave-one-region-out CV ----
    logo = LeaveOneGroupOut()
    pred3 = np.empty(len(d), dtype=object)
    pred6 = np.zeros(len(d), dtype=int)
    per_region = []
    for tr, te in logo.split(X, y3, groups):
        reg = d["region"].iloc[te[0]]
        clf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, class_weight="balanced",
                                     min_samples_leaf=3, random_state=0)
        clf.fit(X[tr], y3[tr])
        pred3[te] = clf.predict(X[te])
        clf6 = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, class_weight="balanced",
                                      min_samples_leaf=3, random_state=0)
        clf6.fit(X[tr], y6[tr])
        pred6[te] = clf6.predict(X[te])
        tg = (y3[te] == "generative"); pg = (pred3[te] == "generative")
        per_region.append({"region": reg, "n": int(len(te)),
                           "n_generative": int(tg.sum()),
                           **{f"p45_{k}": v for k, v in _binary_scores(tg, pg).items()
                              if k in ("precision", "recall", "f1")},
                           "acc3": round(float((y3[te] == pred3[te]).mean()), 4)})

    d["pred3"] = pred3; d["pred6"] = pred6
    d.to_csv(out / "features_0104_predicted.csv", index=False)

    truth_gen = (y3 == "generative"); pred_gen = (pred3 == "generative")
    pooled = _binary_scores(truth_gen, pred_gen)
    acc3 = float((y3 == pred3).mean()); acc6 = float((y6 == pred6).mean())
    cm3 = pd.crosstab(pd.Series(y3, name="truth"), pd.Series(pred3, name="pred")).reindex(
        index=CLS, columns=CLS, fill_value=0)

    # rule-based baseline (what we had before): generative iff in_season & past peak
    rule_gen = (d["in_season"] == 1).values & (d["days_to_POS"] >= 0).values
    rule = _binary_scores(truth_gen, rule_gen)

    # feature importance (full-data model)
    imp = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, class_weight="balanced",
                                 min_samples_leaf=3, random_state=0).fit(X, y3)
    fi = sorted(zip(FEATS, imp.feature_importances_), key=lambda t: -t[1])

    summary = {
        "n_points": int(len(d)), "n_generative": int(truth_gen.sum()),
        "cv": "leave-one-region-out",
        "phase45_classifier": pooled,
        "phase45_rule_baseline": rule,
        "accuracy_3class": round(acc3, 4), "accuracy_6class": round(acc6, 4),
        "confusion_3class": cm3.to_dict(),
        "per_region": per_region,
        "feature_importance": [(f, round(float(v), 4)) for f, v in fi],
    }
    (out / "classifier_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"0104 points: {len(d)} | generative: {int(truth_gen.sum())} | CV: leave-one-region-out")
    print(f"PHASE 4-5  classifier:  P={pooled['precision']:.1%} R={pooled['recall']:.1%} F1={pooled['f1']:.1%}")
    print(f"PHASE 4-5  rule (old):  P={rule['precision']:.1%} R={rule['recall']:.1%} F1={rule['f1']:.1%}")
    print(f"3-class acc {acc3:.1%} | 6-class acc {acc6:.1%}")
    print("\n3-class confusion (rows=truth, cols=pred):"); print(cm3.to_string())
    print("\nper-region phase-4-5 F1:")
    for r in sorted(per_region, key=lambda r: -r["p45_f1"]):
        print(f"   {r['region']:16s} n={r['n']:4d} gen={r['n_generative']:4d}  "
              f"P={r['p45_precision']:.0%} R={r['p45_recall']:.0%} F1={r['p45_f1']:.0%}")
    print("\nfeature importance:", ", ".join(f"{f}={v:.2f}" for f, v in fi[:6]))
    print(f"wrote -> {out}/classifier_summary.json, features_0104_predicted.csv")
    return summary


if __name__ == "__main__":
    main()
