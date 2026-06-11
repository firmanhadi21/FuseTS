#!/usr/bin/env python
"""Build a MOGPR-ready (t, y, x) cube from a locally **SNAP-preprocessed** Sentinel-1
VH stack + **MPC** Sentinel-2 NDVI — so MOGPR fuses on YOUR operational S1 instead of
the S1 that `scale_runner` pulls from MPC.

Source stack (rice-growth-stage-mapping):
  `stacks/java_vh_2024_2026_50m.tif` — 73 bands named `<year>_Period_<n>`
  (2024 = bands 1–31, 2025 = 32–62, 2026 = 63–73), EPSG:4326 ~50 m.
  **Stored int16 = Gamma0 VH dB × 100, nodata = −32768** (e.g. −1535 → −15.35 dB).
  This builder selects one year's bands by name, **rescales ×0.01 → dB, masks −32768**.

Notes:
  - Give the **calibrated backscatter** stack, NOT a Heikin-Ashi product (HA pre-smooths
    and breaks MOGPR's noise model).
  - 2024 has no operational VV/NDVI → fusion is **VH (yours) + NDVI (MPC S2)**, a valid
    2-series SAR↔optical fusion. Pass `--vv-stack` if you have a matching VV stack.

Output matches `scale_runner.build_tile_cube` (dims t,y,x; vars S2ndvi, VV, VH; CRS set),
a drop-in for `fuse_raster` / phenology:

    from scale_runner import fuse_raster
    cube = xr.open_dataset("cube.nc").rio.write_crs(cfg["crs"])
    ds, n = fuse_raster(cube, cfg)            # cfg["granularity"]="pixel", drop_thr, ...

Usage
-----
  python scripts/build_cube_from_snap_s1.py \
      --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
      --year 2024 --aoi data/aois/rentang_di_4326.gpkg \
      --crs EPSG:32749 --res 50 --out output/rentang_snap/cube.nc
  # or a bbox instead of --aoi:  --bbox 108.10 -6.68 108.53 -6.28
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import rasterio
import rioxarray  # noqa: F401
import geopandas as gpd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from klambu_glapan_mogpr import generate_periods, s2_ndvi_composite  # noqa: E402
from scale_runner import get_catalog  # noqa: E402

DB_SCALE = 0.01  # int16 stores dB × 100


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load_vh_db(path, year, bbox_wgs84, crs, res):
    """Open the VH stack, select `year`'s 31 period-bands by name, window to bbox,
    reproject to target grid, rescale int16(dB×100) → dB (nodata already NaN).

    Returns (DataArray[band,y,x] in dB, list-of-period-numbers aligned to band order).
    """
    with rasterio.open(path) as ds:
        desc = list(ds.descriptions)
    tag = f"{year}_Period_"
    sel = sorted(((i + 1, int(d.split("_Period_")[1])) for i, d in enumerate(desc)
                  if d and d.startswith(tag)), key=lambda t: t[1])
    da = rioxarray.open_rasterio(path, masked=True, chunks={"x": 2048, "y": 2048})
    if sel:
        bands = [b for b, _ in sel]
        periods = [pn for _, pn in sel]
        da = da.sel(band=bands)
    else:  # un-named single-year stack: assume sequential periods
        periods = list(range(1, da.sizes["band"] + 1))
        log(f"  no '{tag}*' band names found — assuming bands are periods 1..{len(periods)}")
    da = da.rio.clip_box(*bbox_wgs84, crs="EPSG:4326")
    da = da.rio.reproject(crs, resolution=res)
    da = (da * DB_SCALE).load()  # masked=True already turned -32768 into NaN
    return da, periods


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vh-stack", required=True, help="multi-year VH dB×100 int16 stack (named bands)")
    p.add_argument("--vv-stack", default=None, help="optional matching VV stack (same naming/scale)")
    p.add_argument("--year", type=int, default=2024, help="which <year>_Period_* bands to use")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--aoi", help="AOI vector (any CRS); its bbox is used")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="bbox in WGS84 lon/lat")
    p.add_argument("--crs", default="EPSG:32749", help="target UTM CRS")
    p.add_argument("--res", type=float, default=50.0)
    p.add_argument("--period-days", type=int, default=12)
    p.add_argument("--max-cloud", type=float, default=90.0, help="S2 cloud-cover filter (%)")
    p.add_argument("--out", required=True, help="output cube.nc path")
    a = p.parse_args(argv)

    bbox = (list(gpd.read_file(a.aoi).to_crs("EPSG:4326").total_bounds) if a.aoi else list(a.bbox))
    log(f"bbox(wgs84)={[round(x, 4) for x in bbox]} | year={a.year} crs={a.crs} res={a.res}")

    # period_number -> {start,end,center}
    pinfo = {pp["period"]: pp for pp in generate_periods(f"{a.year}-01-01", f"{a.year}-12-31", a.period_days)}

    vh, periods = _load_vh_db(a.vh_stack, a.year, bbox, a.crs, a.res)
    log(f"VH: {vh.sizes['band']} bands (periods {periods[0]}..{periods[-1]}) "
        f"on {vh.sizes['y']}x{vh.sizes['x']} grid ({a.crs})")
    vfin = int(np.isfinite(vh.values).sum())
    if vfin == 0:
        raise SystemExit("No valid VH pixels in this AOI — wrong stack/year/bbox?")
    vv = None
    if a.vv_stack:
        vv, _ = _load_vh_db(a.vv_stack, a.year, bbox, a.crs, a.res)

    ref = vh.isel(band=0, drop=True)
    cat = get_catalog()
    slices = []
    for bi, pn in enumerate(periods):
        prd = pinfo.get(pn)
        if prd is None:
            continue
        vh_i = vh.isel(band=bi, drop=True)
        win = f"{prd['start']}/{prd['end']}"
        try:
            s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=win,
                                 query={"eo:cloud_cover": {"lt": a.max_cloud}}).items())
            ndvi = s2_ndvi_composite(cat, s2, bbox, a.crs, a.res) if s2 else None
        except Exception as e:
            log(f"  period {pn} S2 error: {e}")
            ndvi = None
        ndvi = ndvi.rio.reproject_match(ref) if ndvi is not None else xr.full_like(ref, np.nan, "float32")
        vv_i = (vv.isel(band=bi, drop=True).rio.reproject_match(ref)
                if vv is not None and bi < vv.sizes["band"] else xr.full_like(ref, np.nan, "float32"))
        dd = {"S2ndvi": ndvi, "VV": vv_i, "VH": vh_i}
        dd = {k: v.drop_vars("spatial_ref", errors="ignore").assign_coords(y=ref.y, x=ref.x)
              for k, v in dd.items()}
        slices.append(xr.Dataset(dd).assign_coords(t=np.datetime64(prd["center"])))
        nd_ok = int(np.isfinite(dd["S2ndvi"].values).sum())
        log(f"  period {pn:2d} {prd['center']}: NDVI px={nd_ok} ({'S2' if nd_ok else 'gap→SAR'})")

    cube = xr.concat(slices, dim="t").sortby("t").transpose("t", "y", "x").rio.write_crs(a.crs)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in cube.data_vars}
    cube.to_netcdf(out, encoding=enc)
    log(f"wrote {out} | dims {dict(cube.sizes)} | "
        f"NDVI finite {int(np.isfinite(cube['S2ndvi'].values).sum())} | "
        f"VH finite {int(np.isfinite(cube['VH'].values).sum())}")
    log("fuse: cube=xr.open_dataset(out).rio.write_crs(crs); fuse_raster(cube, cfg)")
    return cube


if __name__ == "__main__":
    main()
