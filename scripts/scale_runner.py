#!/usr/bin/env python
"""
scale_runner.py — tiled, mask-driven, resumable MOGPR rice-phenology at scale.

CPU-only (GPy MOGPR), multiprocessing across tiles, no GPU and no scheduler — so
it runs anywhere (Mac, a JupyterHub box, any Linux node) and tolerates being
killed/preempted: each tile is a checkpoint, and --resume skips finished tiles.

Granularity (the main cost lever):
  pixel  : per-pixel MOGPR over masked rice at native --res (50 m) -> wall-to-wall maps
  grid   : coarsen to --grid-size (e.g. 500 m), then per-cell MOGPR -> cheap, KSA-aligned
  parcel : zonal-mean per polygon in --parcels -> per-field records

Pipeline per tile: MPC extract (S1+S2 12-day cube) -> paddy mask -> MOGPR fuse
-> multi-season peakvalley phenology -> write tile outputs + .done marker.
Then merge tiles into AOI-wide products.

Examples
--------
  # 500 m grid over Java paddy (cheap), 8 workers, resumable:
  python scripts/scale_runner.py --aoi java.gpkg --mask paddy_mask.tif \
      --crs EPSG:32748 --granularity grid --grid-size 500 --workers 8 \
      --outdir output/java_grid

  # 50 m wall-to-wall paddy, tiled, run in tmux over time:
  python scripts/scale_runner.py --aoi jatiluhur.gpkg --mask paddy_mask.tif \
      --crs EPSG:32748 --granularity pixel --tile-km 20 --workers 12 \
      --outdir output/jatiluhur_pixel

  # list tiles only (plan), no processing:
  python scripts/scale_runner.py --aoi java.gpkg --crs EPSG:32748 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401
from shapely.geometry import box
from rasterio.warp import transform_bounds

# reuse existing building blocks
sys.path.insert(0, str(Path(__file__).resolve().parent))
from klambu_glapan_mogpr import s2_ndvi_composite, s1_db_composite, generate_periods  # noqa: E402
from pv_phenology import _seasons_from_pydate  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_CAT = None  # per-process STAC catalogue


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} pid{os.getpid()}] {m}", flush=True)


def get_catalog():
    """Lazily build one signed MPC client per worker process."""
    global _CAT
    if _CAT is None:
        import planetary_computer as pc
        import pystac_client
        _CAT = pystac_client.Client.open(MPC_STAC, modifier=pc.sign_inplace)
    return _CAT


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def load_aoi(path, crs):
    g = gpd.read_file(path).to_crs(crs)
    return g, g.geometry.unary_union


def generate_tiles(geom_crs, crs, tile_m):
    """Regular grid of tile_m tiles (in target CRS) intersecting the AOI."""
    minx, miny, maxx, maxy = geom_crs.bounds
    tiles, k = [], 0
    y = miny
    row = 0
    while y < maxy:
        x = minx
        col = 0
        while x < maxx:
            cell = box(x, y, min(x + tile_m, maxx), min(y + tile_m, maxy))
            if cell.intersects(geom_crs):
                w, s, e, n = cell.bounds
                wgs = transform_bounds(crs, "EPSG:4326", w, s, e, n)
                tiles.append({"id": f"r{row:03d}_c{col:03d}", "utm": (w, s, e, n),
                              "wgs84": list(wgs)})
                k += 1
            x += tile_m
            col += 1
        y += tile_m
        row += 1
    return tiles


# ---------------------------------------------------------------------------
# Per-tile extraction
# ---------------------------------------------------------------------------
def build_tile_cube(bbox_wgs84, cfg):
    cat = get_catalog()
    periods = generate_periods(cfg["start"], cfg["end"], cfg["period_days"])
    orbit_q = {} if cfg["s1_orbit"] == "any" else {"sat:orbit_state": {"eq": cfg["s1_orbit"]}}
    slices = []
    for p in periods:
        win = f"{p['start']}/{p['end']}"
        try:
            s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox_wgs84, datetime=win,
                                 query={"eo:cloud_cover": {"lt": cfg["max_cloud"]}}).items())
            s1 = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox_wgs84, datetime=win,
                                 query={"sar:instrument_mode": {"eq": "IW"}, **orbit_q}).items())
            ndvi = s2_ndvi_composite(cat, s2, bbox_wgs84, cfg["crs"], cfg["res"]) if s2 else None
            vv, vh = s1_db_composite(cat, s1, bbox_wgs84, cfg["crs"], cfg["res"]) if s1 else (None, None)
        except Exception as e:
            log(f"  period {p['period']} error: {e}")
            continue
        if ndvi is None and vv is None:
            continue
        ref = ndvi if ndvi is not None else vv
        nan_t = xr.full_like(ref, np.nan, dtype="float32")
        dd = {"S2ndvi": ndvi if ndvi is not None else nan_t,
              "VV": vv.rio.reproject_match(ref) if vv is not None else nan_t,
              "VH": vh.rio.reproject_match(ref) if vh is not None else nan_t}
        dd = {kk: vvv.drop_vars("spatial_ref", errors="ignore").assign_coords(y=ref.y, x=ref.x)
              for kk, vvv in dd.items()}
        slices.append(xr.Dataset(dd).assign_coords(t=np.datetime64(p["center"])))
    if not slices:
        return None
    cube = xr.concat(slices, dim="t").sortby("t").transpose("t", "y", "x").rio.write_crs(cfg["crs"])
    return cube


def apply_mask(cube, mask_path, crs):
    ref = cube["S2ndvi"].isel(t=0)
    m = rioxarray.open_rasterio(mask_path, masked=True, chunks={"x": 2048, "y": 2048}).squeeze()
    minx, miny, maxx, maxy = ref.rio.bounds()
    mb = transform_bounds(crs, m.rio.crs, minx, miny, maxx, maxy)
    try:
        m = m.rio.clip_box(*mb).compute()
    except Exception:
        return cube  # tile outside mask footprint -> leave as-is
    m = m.rio.reproject_match(ref)
    return cube.where((m > 0).values)


# ---------------------------------------------------------------------------
# Fusion + phenology
# ---------------------------------------------------------------------------
def _fuse_series(ndvi_v, vv_v, vh_v, t_ord, out_grid, x_py, drop_thr, max_seasons):
    """MOGPR-fuse one unit's series, return a metric vector [n, (POS,peak,LOS)*K]."""
    data, tin = [], []
    nv = np.isfinite(ndvi_v)
    if nv.sum() < 2:
        return None
    data.append(ndvi_v[nv]); tin.append(t_ord[nv])      # NDVI is master (index 0)
    for v in (vv_v, vh_v):
        m = np.isfinite(v)
        if m.sum() >= 4:
            data.append(v[m]); tin.append(t_ord[m])
    try:
        om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
        fused = np.ravel(om[0])
    except Exception:
        return None
    seasons = _seasons_from_pydate(x_py, fused, drop_thr)
    out = np.full(1 + max_seasons * 3, np.nan, np.float32)
    out[0] = len(seasons)
    for k, s in enumerate(seasons[:max_seasons]):
        out[1 + k * 3] = s["POS_doy"]; out[2 + k * 3] = s["peak_NDVI"]; out[3 + k * 3] = s["LOS_days"]
    return out


def fuse_raster(cube, cfg):
    """Per-pixel (or coarsened-cell) MOGPR fusion -> metric Dataset on the cube grid."""
    if cfg["granularity"] == "grid":
        f = max(1, round(cfg["grid_size"] / cfg["res"]))
        cube = cube.coarsen(y=f, x=f, boundary="trim").mean()
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, cfg["period_days"], float)
    x_py = np.array([datetime.fromordinal(int(o)) for o in out_grid], dtype=object)
    K = cfg["max_seasons"]; nm = 1 + K * 3
    ndvi = cube["S2ndvi"].values; vv = cube["VV"].values; vh = cube["VH"].values
    metrics = np.full((nm, ny, nx), np.nan, np.float32)
    valid = np.isfinite(ndvi).sum(axis=0) >= 2
    ys, xs = np.where(valid)
    for yi, xi in zip(ys, xs):
        out = _fuse_series(ndvi[:, yi, xi], vv[:, yi, xi], vh[:, yi, xi],
                           t_ord, out_grid, x_py, cfg["drop_thr"], K)
        if out is not None:
            metrics[:, yi, xi] = out
    names = ["n_seasons"]
    for i in range(1, K + 1):
        names += [f"s{i}_POS_doy", f"s{i}_peak_NDVI", f"s{i}_LOS_days"]
    ds = xr.Dataset({nm_: (("y", "x"), metrics[k]) for k, nm_ in enumerate(names)},
                    coords={"y": cube["y"], "x": cube["x"]}).rio.write_crs(cfg["crs"])
    return ds, int(valid.sum())


def fuse_parcels(cube, parcels, cfg):
    """Zonal-mean MOGPR per parcel polygon -> season-record rows."""
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, cfg["period_days"], float)
    x_py = np.array([datetime.fromordinal(int(o)) for o in out_grid], dtype=object)
    rows = []
    for idx, geom in zip(parcels.index, parcels.geometry):
        try:
            clip = {s: cube[s].rio.clip([geom], cfg["crs"], all_touched=True, drop=True)
                            .mean(dim=("y", "x")).values for s in ("S2ndvi", "VV", "VH")}
        except Exception:
            continue
        out = _fuse_series(clip["S2ndvi"], clip["VV"], clip["VH"],
                           t_ord, out_grid, x_py, cfg["drop_thr"], cfg["max_seasons"])
        if out is None:
            continue
        c = geom.centroid
        rows.append({"parcel": idx, "x": c.x, "y": c.y, "n_seasons": int(out[0]),
                     "s1_POS_doy": out[1], "s1_peak_NDVI": out[2], "s1_LOS_days": out[3]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------
def process_tile(tile, cfg):
    tdir = Path(cfg["outdir"]) / "tiles" / tile["id"]
    done = tdir / ".done"
    if cfg["resume"] and done.exists():
        return tile["id"], "skip", 0
    tdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        cube = build_tile_cube(tile["wgs84"], cfg)
        if cube is None:
            (tdir / ".empty").write_text("no data")
            done.write_text("empty")
            return tile["id"], "empty", 0
        if cfg["mask"]:
            cube = apply_mask(cube, cfg["mask"], cfg["crs"])

        if cfg["granularity"] == "parcel":
            parcels = gpd.read_file(cfg["parcels"]).to_crs(cfg["crs"])
            parcels = parcels[parcels.intersects(box(*tile["utm"]))]
            df = fuse_parcels(cube, parcels, cfg)
            df.to_csv(tdir / "parcels.csv", index=False)
            n = len(df)
        else:
            ds, n = fuse_raster(cube, cfg)
            for v in ds.data_vars:
                ds[v].rio.write_crs(cfg["crs"]).rio.to_raster(tdir / f"{v}.tif", compress="lzw")
        done.write_text(json.dumps({"units": int(n), "secs": round(time.time() - t0)}))
        return tile["id"], "ok", n
    except Exception as e:
        (tdir / ".error").write_text(repr(e))
        return tile["id"], f"error: {e}", 0


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_outputs(cfg, tiles):
    out = Path(cfg["outdir"])
    if cfg["granularity"] == "parcel":
        dfs = [pd.read_csv(out / "tiles" / t["id"] / "parcels.csv")
               for t in tiles if (out / "tiles" / t["id"] / "parcels.csv").exists()]
        if dfs:
            allp = pd.concat(dfs, ignore_index=True).drop_duplicates("parcel")
            allp.to_csv(out / "parcels_phenology.csv", index=False)
            log(f"merged {len(allp)} parcels -> {out/'parcels_phenology.csv'}")
        return
    from rioxarray.merge import merge_arrays
    metrics = set()
    for t in tiles:
        d = out / "tiles" / t["id"]
        if d.exists():
            metrics |= {f.stem for f in d.glob("*.tif")}
    for m in sorted(metrics):
        arrs = [rioxarray.open_rasterio(out / "tiles" / t["id"] / f"{m}.tif", masked=True)
                for t in tiles if (out / "tiles" / t["id"] / f"{m}.tif").exists()]
        if not arrs:
            continue
        mosaic = merge_arrays(arrs)
        mosaic.rio.to_raster(out / f"{m}.tif", compress="lzw")
        log(f"merged {m}: {len(arrs)} tiles -> {out/f'{m}.tif'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", required=True)
    p.add_argument("--mask", default=None, help="paddy/land-cover mask raster (>0 = keep)")
    p.add_argument("--crs", default="EPSG:32749", help="target UTM CRS (Java-central 49S; West Java 48S)")
    p.add_argument("--start", default="2024-11-01")
    p.add_argument("--end", default="2025-10-31")
    p.add_argument("--period-days", type=int, default=12)
    p.add_argument("--res", type=float, default=50.0)
    p.add_argument("--granularity", choices=["pixel", "grid", "parcel"], default="grid")
    p.add_argument("--grid-size", type=float, default=500.0, help="grid cell size (m) for --granularity grid")
    p.add_argument("--parcels", default=None, help="polygon layer for --granularity parcel")
    p.add_argument("--tile-km", type=float, default=20.0, help="processing tile size (km)")
    p.add_argument("--max-cloud", type=float, default=90.0)
    p.add_argument("--s1-orbit", default="ascending", choices=["ascending", "descending", "any"])
    p.add_argument("--drop-thr", type=float, default=0.12)
    p.add_argument("--max-seasons", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", default=None, help="output dir (required unless --dry-run)")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--limit-tiles", type=int, default=None, help="process only the first N tiles (testing)")
    p.add_argument("--dry-run", action="store_true", help="list tiles and exit")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = vars(a).copy()
    gdf, geom = load_aoi(a.aoi, a.crs)
    tiles = generate_tiles(geom, a.crs, a.tile_km * 1000)
    if a.limit_tiles:
        tiles = tiles[:a.limit_tiles]
    if a.dry_run:
        log(f"AOI {Path(a.aoi).name} -> {len(tiles)} tiles of {a.tile_km} km")
        for t in tiles[:20]:
            print(f"  {t['id']}  wgs84={[round(v,3) for v in t['wgs84']]}")
        log(f"dry-run: {len(tiles)} tiles total. Exiting.")
        return
    if not a.outdir:
        sys.exit("--outdir is required unless --dry-run")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "tiles_manifest.json").write_text(json.dumps(tiles, indent=2))
    log(f"AOI {Path(a.aoi).name} -> {len(tiles)} tiles of {a.tile_km} km "
        f"| granularity={a.granularity} | workers={a.workers} | resume={a.resume}")
    done0 = sum(1 for t in tiles if (out / "tiles" / t["id"] / ".done").exists())
    log(f"{done0}/{len(tiles)} tiles already done (will skip on resume)")
    t0 = time.time()
    fn = partial(process_tile, cfg=cfg)
    results = []
    if a.workers <= 1:
        for t in tiles:
            results.append(fn(t)); log(f"  tile {results[-1][0]}: {results[-1][1]}")
    else:
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, tiles), 1):
                results.append(r)
                log(f"  [{i}/{len(tiles)}] tile {r[0]}: {r[1]} ({r[2]} units)")
    ok = sum(1 for r in results if r[1] == "ok")
    log(f"processed: {ok} ok, {len(results)-ok} skip/empty/error in {(time.time()-t0)/60:.1f} min")

    log("merging tiles…")
    merge_outputs(cfg, tiles)
    log(f"DONE. Outputs in {out}/")


if __name__ == "__main__":
    main()
