#!/usr/bin/env python
"""Cross-validate MOGPR multi-season `n_seasons` against an independent
`cropping_intensity.tif` (SAR-classifier product from s1-land-cover-classification).

Replicates and generalises the ad-hoc validation behind
`output/jatiluhur/RESULTS.md` (the 500 m run), so it is repeatable for the 50 m
run and any future AOI.

Method
------
- Reproject/align the reference cropping-intensity raster onto the prediction grid.
  When the two grids share a resolution (e.g. both 50 m) nearest-neighbour is exact;
  when the reference is finer it is majority-resampled.
- Compare over cells valid in BOTH products: prediction `n_seasons >= 1` and
  reference intensity in {1,2,3}.
- For the class comparison the prediction is clipped to [1,3] (the reference only
  encodes single/double/triple), but the RAW `n_seasons` distribution is reported
  separately so per-pixel over-segmentation (n>3) is visible, not hidden.

Outputs (in --outdir)
---------------------
- cross_validation_cropping_intensity.csv : 3x3 confusion matrix (rows=MOGPR, cols=CI)
- cross_validation_metrics.json           : exact/within-1, per-class, direction, raw dist
- cross_validation_maps.png               : MOGPR vs CI maps + agreement heatmap

Usage
-----
  python scripts/cross_validate_cropping_intensity.py \
      --pred output/jatiluhur_50m/n_seasons.tif \
      --ref  .../cropping_intensity_consensus_mt2024_25/cropping_intensity.tif \
      --outdir output/jatiluhur_50m
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling


def _open(path):
    da = rioxarray.open_rasterio(path)
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    return da


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="MOGPR n_seasons.tif (prediction grid)")
    p.add_argument("--ref", required=True, help="independent cropping_intensity.tif")
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-class", type=int, default=3, help="reference max class (triple)")
    a = p.parse_args(argv)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred = _open(a.pred).astype("float32")
    ref = _open(a.ref).astype("float32")

    # Choose resampling: majority if reference is finer than prediction, else nearest.
    pres = abs(float(pred.rio.resolution()[0]))
    # reference resolution in metres (reproject a tiny patch is overkill; compare after warp)
    ref_on_pred = ref.rio.reproject_match(
        pred, resampling=Resampling.mode  # majority — exact when resolutions match
    )

    pv = pred.values
    rv = ref_on_pred.values
    # align shapes (reproject_match guarantees this, but guard anyway)
    if pv.shape != rv.shape:
        ny = min(pv.shape[0], rv.shape[0]); nx = min(pv.shape[1], rv.shape[1])
        pv = pv[:ny, :nx]; rv = rv[:ny, :nx]

    K = a.max_class
    raw = pv.copy()
    # cells valid in both: prediction has a cycle, reference is paddy-with-cycle 1..K
    valid = np.isfinite(pv) & np.isfinite(rv) & (pv >= 1) & (rv >= 1) & (rv <= K)
    n = int(valid.sum())
    if n == 0:
        raise SystemExit("No overlapping valid cells between prediction and reference.")

    mog = np.clip(raw[valid], 1, K).astype(int)   # clipped to [1,K] for class compare
    ci = rv[valid].astype(int)

    # confusion matrix rows=MOGPR(1..K), cols=CI(1..K)
    cm = np.zeros((K, K), int)
    for m, c in zip(mog, ci):
        cm[m - 1, c - 1] += 1

    exact = float(np.trace(cm)) / n
    within1 = float(sum(cm[i, j] for i in range(K) for j in range(K)
                        if abs(i - j) <= 1)) / n

    per_class = {}
    for k in range(1, K + 1):
        tp = cm[k - 1, k - 1]
        recall = tp / cm[k - 1, :].sum() if cm[k - 1, :].sum() else 0.0   # MOGPR producer
        prec = tp / cm[:, k - 1].sum() if cm[:, k - 1].sum() else 0.0      # CI user
        per_class[k] = {"recall_mogpr": round(recall, 4), "precision_vs_ci": round(prec, 4)}

    mean_mogpr = float(mog.mean())
    mean_ci = float(ci.mean())
    higher = float((mog > ci).mean())
    lower = float((mog < ci).mean())

    # raw distribution incl. over-segmentation (n_seasons can exceed K at pixel scale)
    rawv = raw[np.isfinite(raw) & (raw >= 1)].astype(int)
    rawdist = {int(k): int((rawv == k).sum()) for k in np.unique(rawv)}
    over = int((rawv > K).sum())

    # ---- write CSV (match the committed 500 m format) ----
    labels = {1: "single", 2: "double", 3: "triple"}
    hdr = "," + ",".join(f"CI {labels.get(k, k)}({k})" for k in range(1, K + 1))
    lines = [hdr]
    for k in range(1, K + 1):
        lines.append(f"MOGPR {labels.get(k, k)}({k})," +
                     ",".join(str(cm[k - 1, j]) for j in range(K)))
    (outdir / "cross_validation_cropping_intensity.csv").write_text("\n".join(lines) + "\n")

    metrics = {
        "n_cells": n,
        "pred_res_m": round(pres, 2),
        "exact_class_agreement": round(exact, 4),
        "within_1_class": round(within1, 4),
        "per_class": per_class,
        "mean_mogpr_clipped": round(mean_mogpr, 3),
        "mean_ci": round(mean_ci, 3),
        "frac_mogpr_higher": round(higher, 4),
        "frac_mogpr_lower": round(lower, 4),
        "raw_n_seasons_distribution": rawdist,
        "raw_over_max_class": over,
        "raw_over_max_class_frac": round(over / int((rawv >= 1).sum()), 4),
        "confusion_matrix_rows_mogpr_cols_ci": cm.tolist(),
    }
    (outdir / "cross_validation_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- maps + heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        mog_map = np.where(valid, np.clip(raw, 1, K), np.nan)
        ci_map = np.where(valid, rv, np.nan)
        cmap = ListedColormap(["#fee08b", "#66bd63", "#1a9850"][:K])
        norm = BoundaryNorm(np.arange(0.5, K + 1.5), cmap.N)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for axi, dat, ttl in [(ax[0], mog_map, f"MOGPR n_seasons (clip 1-{K})"),
                              (ax[1], ci_map, "cropping_intensity (independent)")]:
            im = axi.imshow(dat, cmap=cmap, norm=norm)
            axi.set_title(ttl); axi.axis("off")
            fig.colorbar(im, ax=axi, ticks=range(1, K + 1), fraction=0.046)
        # agreement heatmap (row-normalised confusion)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax[2].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax[2].set_xticks(range(K)); ax[2].set_yticks(range(K))
        ax[2].set_xticklabels([labels.get(k, k) for k in range(1, K + 1)])
        ax[2].set_yticklabels([labels.get(k, k) for k in range(1, K + 1)])
        ax[2].set_xlabel("cropping_intensity"); ax[2].set_ylabel("MOGPR n_seasons")
        ax[2].set_title(f"row-normalised\nexact {exact:.1%} | ±1 {within1:.1%} | n={n:,}")
        for i in range(K):
            for j in range(K):
                ax[2].text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                           color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        fig.suptitle(f"50 m MOGPR vs independent cropping_intensity — {Path(a.pred).parent.name}")
        fig.tight_layout()
        fig.savefig(outdir / "cross_validation_maps.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # plotting must never block the numbers
        print(f"[warn] map plotting failed: {type(e).__name__}: {e}")

    # ---- console summary ----
    print(f"n={n:,} cells | exact {exact:.1%} | within-1 {within1:.1%}")
    print(f"mean MOGPR(clip) {mean_mogpr:.2f} vs CI {mean_ci:.2f} | "
          f"MOGPR higher {higher:.1%} / lower {lower:.1%}")
    print(f"raw n_seasons dist {rawdist} | over>{K}: {over:,} "
          f"({metrics['raw_over_max_class_frac']:.1%})")
    print(f"wrote -> {outdir}/cross_validation_*.csv/.json/.png")
    return metrics


if __name__ == "__main__":
    main()
