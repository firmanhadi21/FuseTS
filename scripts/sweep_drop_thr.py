#!/usr/bin/env python
"""Sweep peakvalley `drop_thr` on already-cached tile cubes, to tame the per-pixel
season over-segmentation seen in the 50 m run (WORKLOG §7b / §10).

Key efficiency: in `_fuse_series` the expensive step is the MOGPR fit
(`mogpr_1D`), which is INDEPENDENT of `drop_thr`; only the cheap peakvalley
season-count (`_seasons_from_pydate`) uses it. So we MOGPR-fuse each pixel ONCE
and re-count seasons at every threshold — the whole sweep costs ~one re-fusion,
not one-per-threshold. No re-download (uses the cached `cube.nc`).

For each threshold it writes a merged AOI `n_seasons_thr{T}.tif`, runs the
existing cross-validation against `cropping_intensity.tif`, and collects a
summary table so you can pick the threshold that best balances exact-class
agreement against the n>max over-segmentation tail.

Usage
-----
  python scripts/sweep_drop_thr.py \
      --run-dir output/jatiluhur_50m \
      --mask <.../paddy_mask.tif> --crs EPSG:32748 \
      --ref  <.../cropping_intensity.tif> \
      --thr 0.12 0.15 0.20 0.25 0.30 0.40 \
      --workers 24
"""
import argparse
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scale_runner import apply_mask  # noqa: E402
from pv_phenology import _seasons_from_pydate  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402
import cross_validate_cropping_intensity as xval  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _fuse_once(ndvi_v, vv_v, vh_v, t_ord, out_grid):
    """MOGPR-fuse one pixel's series ONCE; return the fused NDVI series or None."""
    nv = np.isfinite(ndvi_v)
    if nv.sum() < 2:
        return None
    data, tin = [ndvi_v[nv]], [t_ord[nv]]
    for v in (vv_v, vh_v):
        m = np.isfinite(v)
        if m.sum() >= 4:
            data.append(v[m]); tin.append(t_ord[m])
    try:
        om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
        return np.ravel(om[0])
    except Exception:
        return None


def fuse_tile(tile_dir, mask, crs, thrs):
    """Fuse one cached tile once/pixel; write n_seasons_thr{T}.tif per threshold."""
    cube_path = Path(tile_dir) / "cube.nc"
    if not cube_path.exists():
        return tile_dir, "no-cube"
    cube = xr.open_dataset(cube_path).load().rio.write_crs(crs)
    if mask:
        cube = apply_mask(cube, mask, crs)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, 12, float)
    x_py = np.array([datetime.fromordinal(int(o)) for o in out_grid], dtype=object)
    ndvi = cube["S2ndvi"].values; vv = cube["VV"].values; vh = cube["VH"].values
    out = {t: np.full((ny, nx), np.nan, np.float32) for t in thrs}
    valid = np.isfinite(ndvi).sum(axis=0) >= 2
    ys, xs = np.where(valid)
    for yi, xi in zip(ys, xs):
        fused = _fuse_once(ndvi[:, yi, xi], vv[:, yi, xi], vh[:, yi, xi], t_ord, out_grid)
        if fused is None:
            continue
        for t in thrs:
            out[t][yi, xi] = len(_seasons_from_pydate(x_py, fused, t))
    for t in thrs:
        da = xr.DataArray(out[t], coords={"y": cube["y"], "x": cube["x"]}, dims=("y", "x"))
        da.rio.write_crs(crs).rio.to_raster(Path(tile_dir) / f"n_seasons_thr{t:.2f}.tif",
                                            compress="lzw")
    return tile_dir, f"ok ({int(valid.sum())} px)"


def merge_threshold(run_dir, thr, crs):
    from rioxarray.merge import merge_arrays
    tiles = sorted((Path(run_dir) / "tiles").glob(f"*/n_seasons_thr{thr:.2f}.tif"))
    arrs = [rioxarray.open_rasterio(p) for p in tiles]
    mosaic = merge_arrays(arrs)
    sweep = Path(run_dir) / "sweep"
    sweep.mkdir(exist_ok=True)
    outp = sweep / f"n_seasons_thr{thr:.2f}.tif"
    mosaic.rio.write_crs(crs).rio.to_raster(outp, compress="lzw")
    return outp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="existing run dir with tiles/*/cube.nc")
    p.add_argument("--mask", required=True)
    p.add_argument("--crs", default="EPSG:32748")
    p.add_argument("--ref", required=True, help="cropping_intensity.tif for cross-val")
    p.add_argument("--thr", type=float, nargs="+",
                   default=[0.12, 0.15, 0.20, 0.25, 0.30, 0.40])
    p.add_argument("--max-class", type=int, default=3)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args(argv)

    run_dir = Path(a.run_dir)
    tile_dirs = sorted(str(d) for d in (run_dir / "tiles").iterdir()
                       if (d / "cube.nc").exists())
    log(f"{len(tile_dirs)} cached tiles | thresholds {a.thr} | workers {a.workers}")

    t0 = time.time()
    fn = partial(fuse_tile, mask=a.mask, crs=a.crs, thrs=a.thr)
    with Pool(a.workers) as pool:
        for i, (td, status) in enumerate(pool.imap_unordered(fn, tile_dirs), 1):
            log(f"  [{i}/{len(tile_dirs)}] {Path(td).name}: {status}")
    log(f"fusion done in {(time.time()-t0)/60:.1f} min; merging + cross-validating")

    sweep = run_dir / "sweep"; sweep.mkdir(exist_ok=True)
    rows = []
    for thr in a.thr:
        merged = merge_threshold(run_dir, thr, a.crs)
        outdir = sweep / f"thr{thr:.2f}"
        m = xval.main(["--pred", str(merged), "--ref", a.ref,
                       "--outdir", str(outdir), "--max-class", str(a.max_class)])
        rows.append({
            "drop_thr": thr, "n_cells": m["n_cells"],
            "exact": m["exact_class_agreement"], "within1": m["within_1_class"],
            "mean_mogpr_clip": m["mean_mogpr_clipped"], "mean_ci": m["mean_ci"],
            "over_max_frac": m["raw_over_max_class_frac"],
            "frac_higher": m["frac_mogpr_higher"], "frac_lower": m["frac_mogpr_lower"],
        })
        log(f"  thr {thr:.2f}: exact {m['exact_class_agreement']:.1%} | "
            f"within1 {m['within_1_class']:.1%} | over>{a.max_class} "
            f"{m['raw_over_max_class_frac']:.1%} | mean {m['mean_mogpr_clipped']:.2f}")

    df = pd.DataFrame(rows)
    csv = sweep / "sweep_summary.csv"
    df.to_csv(csv, index=False)
    log(f"wrote {csv}")
    # pick a recommendation: highest exact-class among thresholds whose over-tail <= 10%
    ok = df[df["over_max_frac"] <= 0.10]
    best = (ok if len(ok) else df).sort_values("exact", ascending=False).iloc[0]
    log(f"RECOMMEND drop_thr={best['drop_thr']:.2f} "
        f"(exact {best['exact']:.1%}, over>{a.max_class} {best['over_max_frac']:.1%})")
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
