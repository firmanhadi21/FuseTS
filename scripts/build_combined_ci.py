#!/usr/bin/env python
"""Combined-time-series cropping intensity (2024+2025 as ONE sequence).

Rather than averaging two annual CI maps, this stitches each pixel's CACHED per-year
fused curves (2024: 31 periods + 2025: 30 periods) into one 61-period series, re-classifies
every period with the multi-season model (so the window spans the Dec->Jan boundary), and
counts generative episodes over the full 24 months -> divided by the number of years = annual
cropping intensity. The benefit over averaging: a season straddling the year boundary is
counted ONCE (separate per-year counting splits/misses it). Cheap: reuses fused.npz (no re-fuse).

A true joint MOGPR re-fuse of 61 periods is O(n^3)/pixel (~8x single-year, ~2 days) for marginal
gain, so we reuse the cached single-year fusions and stitch.

Usage
-----
  python scripts/build_combined_ci.py --run-2024 output/production/java_2024 \
     --run-2025 output/production/java_2025 --series output/phase_model/series_multiseason \
     --min-run 3 --workers 16 --out output/production/java_combined_2024_2025
"""
import argparse
import glob
import json
import sys
import warnings
from functools import partial
from multiprocessing import get_context
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from produce_annual import _count_episodes  # noqa: E402
from produce_annual_tiled import TO3  # noqa: E402
from train_v3_mogpr_ensemble import _window  # noqa: E402

PIX_HA = 0.25
NYEARS = 2
_MR = _CLF = None


def log(m):
    import time, os
    print(f"[{time.strftime('%H:%M:%S')} pid{os.getpid()}] {m}", flush=True)


def _init(mr, clf):
    global _MR, _CLF
    _MR, _CLF = mr, clf


def _utm(tid, mask_west=103.53, td=0.2):
    c = int(tid.split("_c")[1]); lon = mask_west + c * td + td / 2.0
    return 32748 if lon < 108 else 32749


def proc(tid, cfg):
    d24 = Path(cfg["r24"]) / "tiles" / tid
    d25 = Path(cfg["r25"]) / "tiles" / tid
    f24p, f25p = d24 / "fused.npz", d25 / "fused.npz"
    if not (f24p.exists() and f25p.exists() and (d24 / "cube.nc").exists()):
        return tid, "no-cache", {}
    crs = f"EPSG:{_utm(tid)}"; W = cfg["window"]
    z24, z25 = np.load(f24p), np.load(f25p)
    f24, ys24, xs24, g24 = z24["fused"], z24["ys"], z24["xs"], z24["out_grid"]
    f25, ys25, xs25, g25 = z25["fused"], z25["ys"], z25["xs"], z25["out_grid"]

    # match pixels present (and finite) in BOTH years by (y,x)
    ok24 = np.isfinite(f24).all(1); ok25 = np.isfinite(f25).all(1)
    idx25 = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys25, xs25))}
    rows24, rows25, oy, ox = [], [], [], []
    for i in range(len(ys24)):
        if not ok24[i]:
            continue
        j = idx25.get((int(ys24[i]), int(xs24[i])))
        if j is None or not ok25[j]:
            continue
        rows24.append(i); rows25.append(j); oy.append(int(ys24[i])); ox.append(int(xs24[i]))
    if not rows24:
        return tid, "no-overlap", {}
    F = np.concatenate([np.nan_to_num(f24[rows24]), np.nan_to_num(f25[rows25])], axis=1)  # (n,61)
    oy = np.array(oy); ox = np.array(ox); L = F.shape[1]

    is_gen = np.zeros((F.shape[0], L), bool)
    for pidx in range(L):
        lo_, hi_ = pidx - W, pidx + W + 1
        seg = np.stack([np.concatenate([np.full(max(0, -lo_), F[i, 0]),
                       F[i, max(0, lo_):min(L, hi_)], np.full(max(0, hi_ - L), F[i, -1])])[:2 * W + 1]
                       for i in range(F.shape[0])])
        X = np.stack([seg, np.gradient(seg, axis=1)], 1).astype("float32")
        is_gen[:, pidx] = (_CLF.predict(_MR.transform(X).values) == "generative")
    ep2yr = _count_episodes(is_gen, cfg["min_run"]).astype("uint8")  # 24-month total

    cube = xr.open_dataset(d24 / "cube.nc").rio.write_crs(crs)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    hmap = np.zeros((ny, nx), "uint8"); hmap[oy, ox] = np.clip(ep2yr, 0, 255)
    out_t = Path(cfg["out"]) / "tiles" / tid; out_t.mkdir(parents=True, exist_ok=True)
    xr.DataArray(hmap, coords={"y": cube["y"], "x": cube["x"]}, dims=("y", "x")) \
        .rio.write_crs(crs).rio.to_raster(out_t / "n_harvests_2yr.tif", compress="lzw")
    return tid, "ok", {"px": int(F.shape[0]), "ep2yr_sum": int(ep2yr.sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-2024", dest="r24", required=True)
    p.add_argument("--run-2025", dest="r25", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--mask", default="/home/unika_sianturi/work/landcover/s1-land-cover-classification/"
                   "cropping_intensity_consensus_mt2024_25/paddy_mask.tif")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--yield", dest="yld", type=float, default=5.8)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); (out / "tiles").mkdir(parents=True, exist_ok=True)

    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    from lightgbm import LGBMClassifier
    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv")
    wn = _window(np.nan_to_num(z["ndvi"]), z["t_grid"], z["label_ord"], a.window)
    Xtr = np.stack([wn, np.gradient(wn, axis=1)], 1).astype("float32")
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(Xtr)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(mr.transform(Xtr).values,
                                                                  meta["fase"].map(TO3).values)
    log(f"3-class classifier trained on {len(meta)} multi-season points")

    tids = [Path(t).parent.name for t in sorted(glob.glob(f"{a.r24}/tiles/*/fused.npz"))]
    log(f"{len(tids)} tiles -> combined 24-month CI | min_run {a.min_run} | workers {a.workers}")
    fn = partial(proc, cfg=dict(r24=a.r24, r25=a.r25, mask=a.mask, window=a.window,
                                min_run=a.min_run, out=str(out)))
    nok = 0
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(mr, clf)) as pool:
        for i, (tid, status, st) in enumerate(pool.imap_unordered(fn, tids), 1):
            if status == "ok":
                nok += 1
            if i % 50 == 0 or status != "ok":
                log(f"[{i}/{len(tids)}] {tid}: {status}")
    log(f"{nok} tiles ok; mosaicking ...")

    # mosaic 2-year totals, snap to mask grid, then annual CI = total / NYEARS
    from rioxarray.merge import merge_arrays
    from rasterio.enums import Resampling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    tifs = sorted(glob.glob(f"{out}/tiles/*/n_harvests_2yr.tif"))
    arrs = [rioxarray.open_rasterio(t).rio.reproject("EPSG:4326") for t in tifs]
    mos = merge_arrays(arrs, nodata=0)
    mask = rioxarray.open_rasterio(a.mask)
    mos = mos.rio.reproject_match(mask.rio.clip_box(*mos.rio.bounds()), resampling=Resampling.nearest)
    mos.rio.to_raster(out / "java_n_harvests_2yr.tif", compress="lzw")

    tot = mos.isel(band=0).values.astype("float32")
    paddy = tot > 0
    annual = np.where(paddy, tot / NYEARS, np.nan)              # mean annual IP (float)
    ci = np.where(paddy, np.clip(np.rint(annual), 1, 3), 0).astype("uint8")  # clamped 1-3 for map
    ci_da = mos.isel(band=0).copy(); ci_da.values = ci
    ci_da.attrs.pop("_FillValue", None); ci_da.encoding.pop("_FillValue", None)
    ci_da.rio.write_nodata(0).rio.to_raster(out / "java_cropping_intensity_combined.tif", compress="lzw")

    mean_ip = float(np.nanmean(annual))
    harvest_ha = float(tot[paddy].sum()) / NYEARS * PIX_HA
    paddy_ha = int(paddy.sum()) * PIX_HA  # note: 4326 px-area approx; authoritative paddy ~2.28M ha
    summary = {"product": "combined_2024_2025", "years": NYEARS, "min_run": a.min_run,
               "mean_annual_IP": round(mean_ip, 3), "annual_harvest_ha": round(harvest_ha, 1),
               "annual_production_t": round(harvest_ha * a.yld, 1), "yield_t_ha": a.yld}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # PNG: majority-class (mode) downsample
    f = 10
    cnt = np.stack([(ci_da == k).coarsen(x=f, y=f, boundary="pad").sum().values for k in (1, 2, 3)])
    vals = np.where(cnt.sum(0) > 0, cnt.argmax(0) + 1, np.nan)
    ext = [float(ci_da.x.min()), float(ci_da.x.max()), float(ci_da.y.min()), float(ci_da.y.max())]
    cmap = ListedColormap(["#fee08b", "#66bd63", "#1a9850"]); norm = BoundaryNorm([.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(16, 6)); ax.set_facecolor("#f5f5f5")
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3], shrink=0.7)
    cb.ax.set_yticklabels(["1", "2", "3"]); cb.set_label("annual cropping intensity (24-mo /2)")
    ax.set_title("Java rice cropping intensity 2024-2025 combined (MOGPR S1+S2, 61-period count /2)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout(); fig.savefig(out / "java_cropping_intensity_combined.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE -> {out} | mean annual IP {mean_ip:.3f} | production {harvest_ha*a.yld/1e6:.2f} Mt/yr")


if __name__ == "__main__":
    main()
