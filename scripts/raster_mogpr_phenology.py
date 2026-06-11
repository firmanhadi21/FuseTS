#!/usr/bin/env python
"""
Wall-to-wall MOGPR fusion + multi-season phenology on a (downsampled) datacube.

Raster MOGPR fits a coregionalized GP per pixel (~9 px/s), so full 50 m over the
Klambu AOI is ~32 h. This script coarsens the cube by --factor first so the run
is tractable, then runs MOGPR tiled, saves the fused NDVI cube, and runs
peakvalley on the FUSED NDVI (true parity with the point-based product).

    python scripts/raster_mogpr_phenology.py \
        --cube output/klambu_glapan/datacube_klambu_glapan.nc \
        --factor 4 --outdir output/klambu_glapan        # 50m -> 200m
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401

from fusets import mogpr


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--factor", type=int, default=4, help="spatial coarsening factor (50m*factor)")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--crs", default="EPSG:32749")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cube = xr.open_dataset(a.cube).rio.write_crs(a.crs)
    vars_ = [v for v in ["VV", "VH", "S2ndvi"] if v in cube]
    if a.factor > 1:
        cube = cube.coarsen(y=a.factor, x=a.factor, boundary="trim").mean()
    ny, nx, nt = cube.sizes["y"], cube.sizes["x"], cube.sizes["t"]
    npx = ny * nx
    log(f"raster MOGPR on {npx:,}px ({ny}x{nx}, {50*a.factor}m), {nt} t, vars={vars_}")
    log(f"  est ~{npx/9/60:.0f} min @ ~9 px/s")

    fused_ndvi = np.full((nt, ny, nx), np.nan, np.float32)
    t0, done = time.time(), 0
    for y0 in range(0, ny, a.tile):
        for x0 in range(0, nx, a.tile):
            sub = cube[vars_].isel(y=slice(y0, y0 + a.tile), x=slice(x0, x0 + a.tile))
            try:
                f = mogpr(sub, variables=vars_, time_dimension="t")
                f = f.compute() if hasattr(f, "compute") else f
                fused_ndvi[:, y0:y0 + sub.sizes["y"], x0:x0 + sub.sizes["x"]] = \
                    f["S2ndvi_FUSED"].transpose("t", "y", "x").values
            except Exception as e:
                log(f"  tile y{y0} x{x0} failed: {e}")
            done += sub.sizes["y"] * sub.sizes["x"]
            log(f"  {done:,}/{npx:,}px  ({100*done/npx:.0f}%)  elapsed {(time.time()-t0)/60:.0f}m")

    fda = xr.DataArray(fused_ndvi, dims=("t", "y", "x"),
                       coords={"t": cube["t"], "y": cube["y"], "x": cube["x"]},
                       name="S2ndvi_FUSED").rio.write_crs(a.crs)
    fp = outdir / f"ndvi_mogpr_fused_{50*a.factor}m.nc"
    fda.to_netcdf(fp, encoding={"S2ndvi_FUSED": {"zlib": True, "complevel": 4}})
    log(f"fused NDVI cube saved: {fp}  ({(time.time()-t0)/60:.0f} min total)")

    # multi-season peakvalley on the FUSED NDVI (parity with points)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pv_phenology import _seasons_from_pydate
    x_py = np.array([pd.Timestamp(t).to_pydatetime() for t in cube["t"].values], dtype=object)
    nseas = xr.apply_ufunc(
        lambda ts: float(len(_seasons_from_pydate(x_py, np.asarray(ts, float), 0.12))),
        fda, input_core_dims=[["t"]], vectorize=True, output_dtypes=[np.float32],
    ).rio.write_crs(a.crs)
    nseas.rio.to_raster(outdir / f"n_seasons_mogpr_{50*a.factor}m.tif", compress="lzw")
    v = nseas.values; v = v[np.isfinite(v)]
    dist = {int(k): int((v == k).sum()) for k in range(0, 5)}
    log(f"FUSED n_seasons distribution: {dist}  mean(>=1)={v[v>=1].mean():.2f}")
    log("Done.")


if __name__ == "__main__":
    main()
