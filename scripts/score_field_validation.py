#!/usr/bin/env python
"""Score a filled field-validation sheet (observed vs predicted phase).

Reads the Semarang pilot CSV after the field team fills `observed_fase` (1-6, optionally
`observed_date`, `transplant_date`). Reports 6-class and 3-class confusion + accuracy, the
generative-class (4,5 / collapsed 3) precision-recall-F1 (the production-relevant metric), and
-- if dates are present -- a transplant-date phenology consistency check that is robust to the
lag between the latest model period and the survey date.

  python scripts/score_field_validation.py output/validation/semarang_validation_p13.csv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

SIX_TO_THREE = {1: 1, 6: 1, 2: 2, 3: 2, 4: 3, 5: 3}  # bare / vegetative / generative
NAME3 = {1: "bare", 2: "veg", 3: "gen"}
P13_DATE = pd.Timestamp("2024-05-29")  # latest model period (~late May); for lag flag


def _cm(obs, pred, labels, names):
    cm = confusion_matrix(obs, pred, labels=labels)
    hdr = "obs\\pred " + " ".join(f"{names[l]:>6}" for l in labels)
    print(hdr)
    for i, l in enumerate(labels):
        print(f"{names[l]:>8} " + " ".join(f"{cm[i,j]:>6}" for j in range(len(labels))))
    print(f"  overall accuracy: {accuracy_score(obs,pred)*100:.1f}%  (n={len(obs)})")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    df["observed_fase"] = pd.to_numeric(df.get("observed_fase"), errors="coerce")
    d = df[df["observed_fase"].isin(range(1, 7))].copy()
    if d.empty:
        print("No filled observed_fase yet. Fill the column (1-6) and re-run."); return
    d["obs6"] = d["observed_fase"].astype(int)
    d["obs3"] = d["obs6"].map(SIX_TO_THREE)
    print(f"=== Field validation — {len(d)}/{len(df)} points scored ===\n")

    if "pred_phase6" in d:
        print("-- 6-class (observed vs predicted) --")
        _cm(d["obs6"].values, d["pred_phase6"].astype(int).values, list(range(1, 7)),
            {i: str(i) for i in range(1, 7)})
        print()
    print("-- 3-class (bare / vegetative / generative) --")
    _cm(d["obs3"].values, d["pred_phase3"].astype(int).values, [1, 2, 3], NAME3)

    # generative-class metrics (the production driver)
    og = (d["obs3"] == 3).astype(int); pg = (d["pred_phase3"].astype(int) == 3).astype(int)
    p, r, f, _ = precision_recall_fscore_support(og, pg, labels=[1], average="binary", zero_division=0)
    print(f"\n  GENERATIVE (4,5): precision {p*100:.1f}%  recall {r*100:.1f}%  F1 {f*100:.1f}%")

    # timing / lag flag
    if d.get("observed_date") is not None and d["observed_date"].notna().any():
        od = pd.to_datetime(d["observed_date"], errors="coerce")
        lag = (od - P13_DATE).dt.days.dropna()
        if len(lag) and lag.median() > 18:
            print(f"\n  [!] survey median {lag.median():.0f} d after the latest model period (p13, {P13_DATE.date()}).")
            print("      Direct phase comparison is lag-affected; prefer the transplant-date check below,")
            print("      or ingest the S1 period covering the survey date and re-map before scoring.")

    # transplant-date phenology consistency (timing-robust): expected 3-class from days-after-planting
    if d.get("transplant_date") is not None and d.get("observed_date") is not None:
        td = pd.to_datetime(d["transplant_date"], errors="coerce")
        od = pd.to_datetime(d["observed_date"], errors="coerce")
        dap = (od - td).dt.days
        def exp3(x):
            if pd.isna(x): return np.nan
            if x < 55: return 2      # vegetative
            if x < 95: return 3      # generative
            return 1                 # ripening/post-harvest -> bare-ish
        d["exp3"] = dap.map(exp3)
        m = d["exp3"].notna()
        if m.any():
            agree = (d.loc[m, "exp3"] == d.loc[m, "obs3"]).mean() * 100
            print(f"\n  Transplant-date check: observed phase matches days-after-planting expectation "
                  f"in {agree:.0f}% of {int(m.sum())} dated points (field-record sanity).")
    print("\n(generative F1 and the dry-/wet-season split are the headline numbers for the paper.)")


if __name__ == "__main__":
    main()
