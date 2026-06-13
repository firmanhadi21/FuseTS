#!/usr/bin/env python
"""Per-period (12-day) growth-stage maps of Java from CACHED fused curves.

Reads each tile's cached `fused.npz` (+ `cube.nc` for geometry) produced by
`produce_annual_tiled.py`, classifies the growth phase at EVERY period, and writes
per-tile multi-band phase rasters, then mosaics per period into Java-wide maps.

Outputs **both schemes** (6-class head, 3 is the collapse):
  6-phase: 1 flooding · 2 early-veg · 3 late-veg · 4 early-gen · 5 late-gen · 6 post-harvest
  3-phase: 1 bare{1,6} · 2 vegetative{2,3} · 3 generative{4,5}   (robust ~70%)

Because fusion is cached, this is CHEAP (seconds/tile) — no download, no re-fuse.
6-class accuracy ~51% (fine splits 2↔3, 4↔5 weak); 3-class ~70%. Latest period is
provisional (MOGPR edge effect).

Usage
-----
  python scripts/map_phases_tiled.py --run-dir output/production/java_2026 \
     --series output/phase_model/series_0104 --workers 16 --out output/production/java_2026_phases
"""
import argparse
import glob
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
from train_v3_mogpr_ensemble import _window  # noqa: E402

# 6-class -> 3-class collapse
SIX_TO_THREE = {1: 1, 6: 1, 2: 2, 3: 2, 4: 3, 5: 3}
_MR = _CLF = None


def log(m):
    import time, os
    print(f"[{time.strftime('%H:%M:%S')} pid{os.getpid()}] {m}", flush=True)


def _init(mr, clf):
    global _MR, _CLF
    _MR, _CLF = mr, clf


def _tile_crs(tid):
    c = int(tid.split("_c")[1]); lon = 103.53 + c * 0.2
    return f"EPSG:{32748 if lon < 108 else 32749}"


def map_tile(tdir, cfg):
    tdir = Path(tdir)
    fused_p = tdir / "fused.npz"; cube_p = tdir / "cube.nc"
    if not (fused_p.exists() and cube_p.exists()):
        return tdir.name, "no-cache"
    crs = _tile_crs(tdir.name)
    fz = np.load(fused_p); fused, ys, xs = fz["fused"], fz["ys"], fz["xs"]
    cube = xr.open_dataset(cube_p).rio.write_crs(crs)
    ny, nx = cube.sizes["y"], cube.sizes["x"]; L = fused.shape[1]; W = cfg["window"]
    ok = np.isfinite(fused).all(1)
    if ok.sum() == 0:
        return tdir.name, "no-fuse"
    Fok = np.nan_to_num(fused[ok]); oy, ox = ys[ok], xs[ok]
    phase6 = np.zeros((L, ny, nx), "uint8")
    for pidx in range(L):
        lo_, hi_ = pidx - W, pidx + W + 1
        seg = np.stack([np.concatenate([np.full(max(0, -lo_), Fok[i, 0]),
                       Fok[i, max(0, lo_):min(L, hi_)], np.full(max(0, hi_ - L), Fok[i, -1])])[:2 * W + 1]
                       for i in range(Fok.shape[0])])
        X = np.stack([seg, np.gradient(seg, axis=1)], 1).astype("float32")
        fase = _CLF.predict(_MR.transform(X).values).astype("uint8")     # 1..6
        phase6[pidx, oy, ox] = fase
    phase3 = np.zeros_like(phase6)
    for k, v in SIX_TO_THREE.items():
        phase3[phase6 == k] = v
    coords = {"band": np.arange(1, L + 1), "y": cube["y"], "x": cube["x"]}
    for name, arr in [("phase6", phase6), ("phase3", phase3)]:
        xr.DataArray(arr, coords=coords, dims=("band", "y", "x")) \
            .rio.write_crs(crs).rio.to_raster(tdir / f"{name}.tif", compress="lzw")
    return tdir.name, f"ok L={L} px={int(ok.sum())}"


def mosaic_period(run_dir, scheme, period, out, crs="EPSG:4326"):
    from rioxarray.merge import merge_arrays
    tifs = sorted(glob.glob(f"{run_dir}/tiles/*/{scheme}.tif"))
    arrs = []
    for t in tifs:
        try:
            arrs.append(rioxarray.open_rasterio(t).isel(band=period - 1).rio.reproject(crs))
        except Exception:
            pass
    if not arrs:
        return None
    mos = merge_arrays(arrs, nodata=0)
    p = Path(out) / f"java_{scheme}_p{period:02d}.tif"
    mos.rio.to_raster(p, compress="lzw")
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="dir with tiles/*/fused.npz + cube.nc")
    p.add_argument("--series", required=True)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--mosaic", action="store_true", help="also mosaic per-period Java maps")
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # train 6-class classifier on 0104
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    from lightgbm import LGBMClassifier
    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv")
    wn = _window(np.nan_to_num(z["ndvi"]), z["t_grid"], z["label_ord"], a.window)
    dn = np.gradient(wn, axis=1).astype("float32"); Xtr = np.stack([wn, dn], 1).astype("float32")
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(Xtr)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(mr.transform(Xtr).values,
                                                                   meta["fase"].astype(int).values)
    log(f"trained 6-class classifier on {len(meta)} points")

    tiles = sorted(glob.glob(f"{a.run_dir}/tiles/*/fused.npz"))
    tdirs = [str(Path(t).parent) for t in tiles]
    log(f"{len(tdirs)} cached tiles -> per-period phase maps | workers {a.workers}")
    fn = partial(map_tile, cfg=dict(window=a.window))
    ctx = get_context("spawn")
    nL = 0
    with ctx.Pool(a.workers, initializer=_init, initargs=(mr, clf)) as pool:
        for i, (tid, status) in enumerate(pool.imap_unordered(fn, tdirs), 1):
            if status.startswith("ok"):
                nL = max(nL, int(status.split("L=")[1].split()[0]))
            log(f"[{i}/{len(tdirs)}] {tid}: {status}")

    if a.mosaic and nL:
        log(f"mosaicking {nL} periods x 2 schemes -> Java maps")
        for scheme in ("phase6", "phase3"):
            for per in range(1, nL + 1):
                mp = mosaic_period(a.run_dir, scheme, per, out)
                log(f"  {scheme} p{per:02d}: {'ok' if mp else 'empty'}")
    log(f"done -> {out} (per-tile phase6/phase3.tif; --mosaic for Java maps)")


if __name__ == "__main__":
    main()
