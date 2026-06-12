#!/usr/bin/env python
"""Java-wide (tiled, resumable) annual rice production via the MOGPR-optical phase model.

Per tile: build a MOGPR cube (SNAP VH + MPC NDVI) in the tile's LOCAL UTM, mask to paddy,
MOGPR-fuse every paddy pixel, classify phase at all 31 periods, count generative episodes
(= harvests/yr), → tile n_harvests.tif + summary. National total = Σ tiles.

  production = Σ_pixels(harvests/yr) × pixel_ha × yield

Degree-tiled (Java spans UTM 48S+49S → each tile uses its local zone). A fast paddy-mask
window check skips ocean/non-paddy tiles. Resumable via per-tile `.done`.

CAVEATS: ~68% F1 model, fails on Pekalongan-type regions; flat yield; single-UTM-per-tile;
no production ground-truth. This is a first national estimate, accuracy uneven.

Usage
-----
  python scripts/produce_annual_tiled.py \
     --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
     --mask .../paddy_mask.tif --series output/phase_model/series_0104 \
     --year 2024 --tile-deg 0.2 --res 50 --yield 5.8 --min-run 2 \
     --workers 16 --out output/production/java [--limit-tiles N] [--dry-run]
"""
import argparse
import json
import sys
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
import rasterio
from rasterio.warp import transform_bounds

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from klambu_glapan_mogpr import generate_periods, s2_ndvi_composite  # noqa: E402
from scale_runner import get_catalog, apply_mask  # noqa: E402
from build_cube_from_snap_s1 import _load_vh_db  # noqa: E402
from train_v3_mogpr_ensemble import _window, TO3  # noqa: E402
from produce_annual import _count_episodes  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

PIX_HA = 0.25


def log(m):
    import time, os
    print(f"[{time.strftime('%H:%M:%S')} pid{os.getpid()}] {m}", flush=True)


def paddy_tiles(mask_path, tile_deg, limit=None):
    """Degree grid over the mask extent; keep tiles that contain paddy (mask>0)."""
    with rasterio.open(mask_path) as ds:
        w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    tiles = []
    ny = int(np.ceil((n - s) / tile_deg)); nx = int(np.ceil((e - w) / tile_deg))
    m = rioxarray.open_rasterio(mask_path, masked=True, chunks={"x": 2048, "y": 2048})
    for r in range(ny):
        for c in range(nx):
            x0, y0 = w + c * tile_deg, s + r * tile_deg
            bb = [x0, y0, min(x0 + tile_deg, e), min(y0 + tile_deg, n)]
            try:
                sub = m.rio.clip_box(*bb).load()
                if float(np.nansum(sub.values > 0)) < 1:
                    continue
            except Exception:
                continue
            utm = 32748 if (bb[0] + bb[2]) / 2 < 108 else 32749
            tiles.append({"id": f"r{r:03d}_c{c:03d}", "bbox": bb, "utm": utm})
            if limit and len(tiles) >= limit:
                return tiles
    return tiles


_MR = _CLF = None


def _init_worker(mr, clf):
    """Pool initializer: receive the fitted classifier once per (spawned) worker."""
    global _MR, _CLF
    _MR, _CLF = mr, clf


def process_tile(tile, cfg, mr=None, clf=None):
    mr = mr if mr is not None else _MR
    clf = clf if clf is not None else _CLF
    out = Path(cfg["out"]); tdir = out / "tiles" / tile["id"]
    done = tdir / ".done"
    if cfg["resume"] and done.exists():
        return tile["id"], "skip", json.loads(done.read_text())
    tdir.mkdir(parents=True, exist_ok=True)
    crs = f"EPSG:{tile['utm']}"; bbox = tile["bbox"]; W = cfg["window"]
    cube_path = tdir / "cube.nc"; fused_path = tdir / "fused.npz"
    no_cache = cfg.get("no_cache", False)

    def _build_cube():
        """Download S2 NDVI + load VH -> tile cube (the ~expensive, cacheable step)."""
        vh, periods = _load_vh_db(cfg["vh_stack"], cfg["year"], bbox, crs, cfg["res"])
        if not np.isfinite(vh.values).any():
            return None
        ref = vh.isel(band=0, drop=True)
        pinfo = {p["period"]: p for p in generate_periods(f"{cfg['year']}-01-01", f"{cfg['year']}-12-31", 12)}
        cat = get_catalog()
        nd_sl, vh_sl, tc = [], [], []
        for bi, pn in enumerate(periods):
            prd = pinfo.get(pn)
            if prd is None:
                continue
            try:
                s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                     datetime=f"{prd['start']}/{prd['end']}",
                                     query={"eo:cloud_cover": {"lt": 90}}).items())
                nd = s2_ndvi_composite(cat, s2, bbox, crs, cfg["res"]) if s2 else None
            except Exception:
                nd = None
            nd = nd.rio.reproject_match(ref) if nd is not None else xr.full_like(ref, np.nan, "float32")
            nd_sl.append(nd.assign_coords(y=ref.y, x=ref.x)); vh_sl.append(vh.isel(band=bi, drop=True))
            tc.append(np.datetime64(prd["center"]))
        return xr.Dataset({"S2ndvi": xr.concat(nd_sl, "t").assign_coords(t=tc),
                           "VH": xr.concat(vh_sl, "t").assign_coords(t=tc)}).rio.write_crs(crs)

    # ---- fused curves: from fused.npz cache, else from cube.nc cache, else download+fuse ----
    if not no_cache and fused_path.exists() and cube_path.exists():
        fz = np.load(fused_path)
        fused, ys, xs, out_grid = fz["fused"], fz["ys"], fz["xs"], fz["out_grid"]
        cube = xr.open_dataset(cube_path).rio.write_crs(crs)            # geometry only
    else:
        if not no_cache and cube_path.exists():
            cube_full = xr.open_dataset(cube_path).rio.write_crs(crs)   # skip MPC download
        else:
            try:
                cube_full = _build_cube()
            except Exception as e:
                done.write_text(json.dumps({"status": "empty", "err": str(e)})); return tile["id"], "empty", {}
            if cube_full is None:
                done.write_text(json.dumps({"status": "empty"})); return tile["id"], "empty", {}
            if not no_cache:
                enc = {v: {"zlib": True, "complevel": 4} for v in cube_full.data_vars}
                cube_full.to_netcdf(cube_path, encoding=enc)            # cache cube -> no re-download
        cube = apply_mask(cube_full, cfg["mask"], crs)
        NDVI = cube["S2ndvi"].values; VH = cube["VH"].values
        tc = list(cube["t"].values)
        ys, xs = np.where(np.isfinite(NDVI).sum(0) >= 2)
        if len(ys) == 0:
            done.write_text(json.dumps({"status": "no_paddy"})); return tile["id"], "empty", {}
        t_ord = np.array([pd.Timestamp(t).toordinal() for t in tc], float)
        out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, 12, float)
        ndvi_px = NDVI[:, ys, xs].T; vh_px = VH[:, ys, xs].T
        fused = np.full((len(ys), len(out_grid)), np.nan, "float32")
        for i in range(len(ys)):
            nd = ndvi_px[i]; vh_v = vh_px[i]; nv = np.isfinite(nd)
            if nv.sum() < 2:
                continue
            data, tin = [nd[nv]], [t_ord[nv]]; mm = np.isfinite(vh_v)
            if mm.sum() >= 4:
                data.append(vh_v[mm]); tin.append(t_ord[mm])
            if len(data) < 2:
                continue
            try:
                om, *_ = mogpr_1D(data, tin, 0, out_grid, 1); fused[i] = np.ravel(om[0]).astype("float32")
            except Exception:
                pass
        if not no_cache:
            np.savez_compressed(fused_path, fused=fused, ys=ys, xs=xs, out_grid=out_grid)  # cache -> no re-fuse

    L = len(out_grid)
    ok = np.isfinite(fused).all(1)
    if ok.sum() == 0:
        done.write_text(json.dumps({"status": "no_fuse"})); return tile["id"], "empty", {}
    Fok = np.nan_to_num(fused[ok])
    is_gen = np.zeros((Fok.shape[0], L), bool)
    for pidx in range(L):
        lo_, hi_ = pidx - W, pidx + W + 1
        seg = np.stack([np.concatenate([np.full(max(0, -lo_), Fok[i, 0]),
                       Fok[i, max(0, lo_):min(L, hi_)], np.full(max(0, hi_ - L), Fok[i, -1])])[:2 * W + 1]
                       for i in range(Fok.shape[0])])
        X = np.stack([seg, np.gradient(seg, axis=1)], 1).astype("float32")
        is_gen[:, pidx] = (clf.predict(mr.transform(X).values) == "generative")
    episodes = _count_episodes(is_gen, cfg["min_run"])
    nh = np.zeros(len(ys), int); nh[ok] = episodes
    ny_, nx_ = cube.sizes["y"], cube.sizes["x"]
    hmap = np.zeros((ny_, nx_), "uint8"); hmap[ys, xs] = np.clip(nh, 0, 255)
    xr.DataArray(hmap, coords={"y": cube["y"], "x": cube["x"]}, dims=("y", "x")) \
        .rio.write_crs(crs).rio.to_raster(tdir / "n_harvests.tif", compress="lzw")
    stat = {"status": "ok", "utm": tile["utm"], "paddy_px": int(ok.sum()),
            "physical_ha": round(int(ok.sum()) * PIX_HA, 1),
            "harvest_ha": round(float(episodes.sum()) * PIX_HA, 1),
            "production_t": round(float(episodes.sum()) * PIX_HA * cfg["yld"], 1),
            "mean_ci": round(float(episodes.mean()), 3)}
    done.write_text(json.dumps(stat))
    return tile["id"], "ok", stat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vh-stack", required=True); p.add_argument("--mask", required=True)
    p.add_argument("--series", required=True); p.add_argument("--year", type=int, default=2024)
    p.add_argument("--tile-deg", type=float, default=0.2); p.add_argument("--res", type=float, default=50.0)
    p.add_argument("--yield", dest="yld", type=float, default=5.8); p.add_argument("--min-run", type=int, default=2)
    p.add_argument("--window", type=int, default=8); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True); p.add_argument("--limit-tiles", type=int, default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--no-cache", action="store_true", help="don't read/write cube.nc + fused.npz")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    log("finding paddy tiles...")
    tiles = paddy_tiles(a.mask, a.tile_deg, a.limit_tiles)
    log(f"{len(tiles)} paddy tiles of {a.tile_deg}°")
    if a.dry_run:
        log("dry-run: exiting."); return tiles

    # train classifier once
    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv")
    wn = _window(np.nan_to_num(z["ndvi"]), z["t_grid"], z["label_ord"], a.window)
    dn = np.gradient(wn, axis=1).astype("float32"); Xtr = np.stack([wn, dn], 1).astype("float32")
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(Xtr)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(mr.transform(Xtr).values, meta["fase"].map(TO3).values)
    log("classifier trained; processing tiles...")

    cfg = dict(out=a.out, vh_stack=a.vh_stack, mask=a.mask, year=a.year, res=a.res,
               yld=a.yld, min_run=a.min_run, window=a.window, resume=a.resume, no_cache=a.no_cache)
    fn = partial(process_tile, cfg=cfg)
    stats = []
    # 'spawn' avoids the fork-after-numba(MiniROCKET) deadlock; initializer ships the
    # fitted classifier to each fresh worker once (not per-task).
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(mr, clf)) as pool:
        for i, (tid, status, stat) in enumerate(pool.imap_unordered(fn, tiles), 1):
            if status in ("ok", "skip") and stat.get("status") == "ok":
                stats.append(stat)
            log(f"[{i}/{len(tiles)}] {tid}: {status}"
                + (f" {stat.get('production_t',0):,.0f} t" if stat.get("status") == "ok" else ""))

    tot = {"tiles_ok": len(stats),
           "physical_paddy_ha": round(sum(s["physical_ha"] for s in stats), 1),
           "annual_harvest_area_ha": round(sum(s["harvest_ha"] for s in stats), 1),
           "annual_production_t": round(sum(s["production_t"] for s in stats), 1),
           "mean_cropping_intensity": round(
               sum(s["harvest_ha"] for s in stats) / max(1e-9, sum(s["physical_ha"] for s in stats)), 3),
           "yield_t_ha": a.yld, "min_run": a.min_run}
    (out / "java_production_summary.json").write_text(json.dumps(tot, indent=2))
    log(f"=== JAVA TOTAL === paddy {tot['physical_paddy_ha']:,.0f} ha | "
        f"CI {tot['mean_cropping_intensity']:.2f} | harvest {tot['annual_harvest_area_ha']:,.0f} ha | "
        f"PRODUCTION {tot['annual_production_t']:,.0f} t")
    return tot


if __name__ == "__main__":
    main()
