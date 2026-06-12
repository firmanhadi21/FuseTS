#!/usr/bin/env python
"""Production loop: apply the MOGPR-optical phase classifier wall-to-wall over an AOI cube,
map phase 4-5 (generative = standing crop near harvest), and estimate production.

   production = area(phase 4-5 at target date) × yield (t/ha)

Steps:
1. Train MiniROCKET(+LightGBM) on the full 0104 series (optical channels: windowed fused
   NDVI + its derivative) — the held-out-validated OPT-only model (≈68% phase-4-5 F1).
2. Load AOI cube (your SNAP VH + MPC NDVI), mask to paddy, MOGPR-fuse every paddy pixel.
3. Window each fused NDVI curve around --target-date, classify phase (bare/veg/generative).
4. Phase-4-5 area = #generative px × pixel_ha; production = area × --yield.

CAVEATS (printed): snapshot at one date (NOT annual — a pixel stays in 4-5 for weeks, so
annual totals need per-cycle counting); flat yield; model ~68% F1; no production ground
truth here (sanity-check vs BPS/calc_luas_panen separately).

Usage
-----
  python scripts/produce_estimate.py \
     --cube output/brebes_snap_2024/cube.nc --crs EPSG:32749 \
     --mask .../paddy_mask.tif --series output/phase_model/series_0104 \
     --target-date 2024-06-23 --yield 5.8 --workers 16 --out output/production/brebes
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scale_runner import apply_mask  # noqa: E402
from train_v3_mogpr_ensemble import _window, TO3, CLS  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

PIX_HA = 0.25  # 50 m × 50 m


def _fuse_chunk(args, t_ord, out_grid, target_ord, W):
    ndvi_c, vh_c = args
    feats = np.full((len(ndvi_c), 2, 2 * W + 1), np.nan, "float32")
    x_grid = out_grid
    idx = int(np.abs(x_grid - target_ord).argmin())
    lo, hi = idx - W, idx + W + 1
    for i in range(len(ndvi_c)):
        nd, vh = ndvi_c[i], vh_c[i]
        nv = np.isfinite(nd)
        if nv.sum() < 2:
            continue
        data, tin = [nd[nv]], [t_ord[nv]]
        m = np.isfinite(vh)
        if m.sum() >= 4:
            data.append(vh[m]); tin.append(t_ord[m])
        if len(data) < 2:
            continue
        try:
            om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
            fused = np.ravel(om[0]).astype("float32")
        except Exception:
            continue
        seg = np.concatenate([np.full(max(0, -lo), fused[0]), fused[max(0, lo):min(len(fused), hi)],
                              np.full(max(0, hi - len(fused)), fused[-1])])[:2 * W + 1]
        feats[i, 0] = seg
        feats[i, 1] = np.gradient(seg)
    return feats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True); p.add_argument("--crs", default="EPSG:32749")
    p.add_argument("--mask", default=None); p.add_argument("--series", required=True)
    p.add_argument("--target-date", required=True)
    p.add_argument("--yield", dest="yld", type=float, default=5.8)
    p.add_argument("--window", type=int, default=8); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W = a.window

    # 1. train OPT classifier on all 0104
    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv")
    tn = np.nan_to_num(z["ndvi"]); tg = z["t_grid"]; lo = z["label_ord"]
    wn = _window(tn, tg, lo, W); dn = np.gradient(wn, axis=1).astype("float32")
    Xtr = np.stack([wn, dn], 1).astype("float32"); ytr = meta["fase"].map(TO3).values
    print(f"training OPT classifier on {len(meta)} 0104 points...")
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(Xtr)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(mr.transform(Xtr).values, ytr)

    # 2. AOI cube -> paddy pixels
    cube = xr.open_dataset(a.cube).rio.write_crs(a.crs)
    if a.mask:
        cube = apply_mask(cube, a.mask, a.crs)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, 12, float)
    NDVI = cube["S2ndvi"].values; VH = cube["VH"].values
    valid = np.isfinite(NDVI).sum(0) >= 2
    ys, xs = np.where(valid)
    ndvi_px = NDVI[:, ys, xs].T.astype("float32"); vh_px = VH[:, ys, xs].T.astype("float32")
    target_ord = float(pd.Timestamp(a.target_date).toordinal())
    print(f"AOI {ny}x{nx} | {len(ys)} paddy px to fuse | target {a.target_date}")

    # 3. parallel MOGPR-fuse + window
    chunks = np.array_split(np.arange(len(ys)), max(1, a.workers * 4))
    args = [(ndvi_px[c], vh_px[c]) for c in chunks]
    fn = partial(_fuse_chunk, t_ord=t_ord, out_grid=out_grid, target_ord=target_ord, W=W)
    feats = np.full((len(ys), 2, 2 * W + 1), np.nan, "float32")
    with Pool(a.workers) as pool:
        for c, f in zip(chunks, pool.map(fn, args)):
            feats[c] = f
    ok = np.isfinite(feats[:, 0, W])                      # fused successfully
    print(f"fused {int(ok.sum())}/{len(ys)} px; classifying...")

    # 4. classify
    pred = np.array(["nodata"] * len(ys), dtype=object)
    if ok.sum():
        F = mr.transform(np.nan_to_num(feats[ok])).values
        pred[ok] = clf.predict(F)

    phase = np.zeros((ny, nx), "uint8")                   # 0 nodata,1 bare,2 veg,3 gen
    code = {"bare": 1, "vegetative": 2, "generative": 3}
    phase[ys, xs] = [code.get(p, 0) for p in pred]
    pmap = xr.DataArray(phase, coords={"y": cube["y"], "x": cube["x"]}, dims=("y", "x"))
    pmap.rio.write_crs(a.crs).rio.to_raster(out / "phase_map.tif", compress="lzw")

    n_gen = int((phase == 3).sum())
    area_ha = n_gen * PIX_HA
    production = area_ha * a.yld
    summary = {"aoi_cube": a.cube, "target_date": a.target_date,
               "paddy_px": int(len(ys)), "fused_px": int(ok.sum()),
               "phase_counts": {k: int((phase == v).sum()) for k, v in code.items()},
               "phase45_generative_px": n_gen,
               "phase45_area_ha": round(area_ha, 1),
               "yield_t_ha": a.yld,
               "production_t": round(production, 1),
               "caveats": ["snapshot at one date (not annual)", "flat yield",
                           "model ~68% phase-4-5 F1", "no production ground-truth here"]}
    (out / "production_summary.json").write_text(json.dumps(summary, indent=2))

    # 5. figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["white", "#cccccc", "#66bd63", "#d73027"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(phase, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
        cb = fig.colorbar(im, ticks=[0, 1, 2, 3]); cb.ax.set_yticklabels(["nodata", "bare", "vegetative", "generative(4-5)"])
        ax.set_title(f"Phase map {a.target_date} | generative {area_ha:.0f} ha -> "
                     f"{production:,.0f} t @ {a.yld} t/ha"); ax.axis("off")
        fig.tight_layout(); fig.savefig(out / "phase_map.png", dpi=120); plt.close(fig)
    except Exception as e:
        print("plot failed:", e)

    print(f"\n=== PRODUCTION ESTIMATE ({a.target_date}) ===")
    print(f"paddy px {len(ys):,} | generative (phase 4-5): {n_gen:,} px = {area_ha:,.0f} ha")
    print(f"PRODUCTION = {area_ha:,.0f} ha × {a.yld} t/ha = {production:,.0f} t")
    print(f"phase counts: {summary['phase_counts']}")
    print(f"wrote -> {out}/phase_map.tif, phase_map.png, production_summary.json")
    print("CAVEAT: snapshot (not annual); flat yield; model ~68% F1; no production GT here.")
    return summary


if __name__ == "__main__":
    main()
