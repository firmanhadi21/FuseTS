#!/usr/bin/env python
"""
Klambu-Glapan S1+S2 MOGPR fusion & phenology — full pipeline (no Jupyter).

Pipeline
--------
1. EXTRACT : Build a full-AOI 12-day datacube (VV, VH dB + Sentinel-2 NDVI)
             from Microsoft Planetary Computer onto a UTM 49S / 50 m grid.
             Saved to <outdir>/datacube_klambu_glapan.nc
2. FUSE    : Point-based MOGPR over N random sample points (fast, validates the
             whole chain). Fused NDVI/VV/VH + phenology metrics -> CSV + plots.
3. RASTER  : Optional wall-to-wall MOGPR over the cube (tiled). Off by default
             (--run-raster) because ~1M pixels is heavy on a laptop.

Run
---
    conda activate fusets
    python scripts/klambu_glapan_mogpr.py                  # extract + point fusion
    python scripts/klambu_glapan_mogpr.py --skip-extract   # reuse saved cube
    python scripts/klambu_glapan_mogpr.py --run-raster      # also do raster MOGPR

All knobs are CLI flags — see `python scripts/klambu_glapan_mogpr.py --help`.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
from shapely.geometry import Point
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import planetary_computer as pc
import pystac_client
import odc.stac

from fusets.mogpr import mogpr_1D
from fusets.analytics import phenology

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default=str(repo / "data" / "klambu-glapan.shp"),
                   help="AOI shapefile (default: data/klambu-glapan.shp)")
    p.add_argument("--start", default="2024-11-01", help="start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-10-31", help="end date YYYY-MM-DD")
    p.add_argument("--period-days", type=int, default=12, help="composite period length")
    p.add_argument("--res", type=float, default=50.0, help="target resolution (m)")
    p.add_argument("--crs", default="EPSG:32749", help="target CRS (UTM 49S)")
    p.add_argument("--buffer", type=float, default=500.0, help="AOI buffer (m)")
    p.add_argument("--max-cloud", type=float, default=90.0,
                   help="max scene cloud%% to consider (per-pixel SCL masking still applied)")
    p.add_argument("--s1-orbit", default="ascending",
                   choices=["ascending", "descending", "any"], help="S1 orbit state")
    p.add_argument("--n-points", type=int, default=200, help="MOGPR sample points")
    p.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    p.add_argument("--mask", default=None,
                   help="paddy/land-cover mask raster; non-positive pixels set to NaN "
                        "so fusion only runs on rice (reprojected to the cube grid)")
    p.add_argument("--points-shp", default=None,
                   help="sample at the centroids of this polygon/point layer "
                        "(carries its attributes) instead of random points — for validation")
    p.add_argument("--outdir", default=str(repo / "output" / "klambu_glapan"),
                   help="output directory")
    p.add_argument("--skip-extract", action="store_true",
                   help="reuse an existing saved datacube instead of downloading")
    p.add_argument("--run-raster", action="store_true",
                   help="ALSO run wall-to-wall raster MOGPR (slow; tiled)")
    p.add_argument("--raster-tile", type=int, default=256, help="raster MOGPR tile size (px)")
    p.add_argument("--n-plot", type=int, default=6, help="example point plots to save")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_aoi(path, buffer_m, target_crs):
    aoi = gpd.read_file(path)
    geom_utm = aoi.to_crs(target_crs).buffer(buffer_m)
    aoi_utm = gpd.GeoDataFrame(geometry=geom_utm, crs=target_crs)
    bbox = aoi_utm.to_crs("EPSG:4326").total_bounds.tolist()
    geom_wgs84 = aoi_utm.to_crs("EPSG:4326").geometry.unary_union
    return bbox, aoi_utm, geom_wgs84


def generate_periods(start_str, end_str, days):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    periods, n, cur = [], 1, start
    while cur <= end:
        pend = min(cur + timedelta(days=days - 1), end)
        periods.append({"period": n, "start": cur.strftime("%Y-%m-%d"),
                        "end": pend.strftime("%Y-%m-%d"),
                        "center": cur + timedelta(days=days // 2)})
        cur = pend + timedelta(days=1)
        n += 1
    return periods


def s2_ndvi_composite(catalog, items, bbox, crs, res):
    """Cloud-masked (SCL) median NDVI composite for a list of S2 items."""
    ds = odc.stac.load(items, bands=["red", "nir", "SCL"], bbox=bbox, crs=crs,
                       resolution=res, groupby="solar_day",
                       chunks={"time": 1, "x": 1024, "y": 1024})
    # SCL keep: 4 veg, 5 bare, 6 water, 7 unclassified, 11 snow
    clear = ds["SCL"].isin([4, 5, 6, 7, 11])
    red = ds["red"].where(clear)
    nir = ds["nir"].where(clear)
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.where(np.isfinite(ndvi))
    return ndvi.median("time", skipna=True).compute()


def s1_db_composite(catalog, items, bbox, crs, res):
    """Median VV/VH (dB) composite for a list of S1-RTC items."""
    ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, crs=crs,
                       resolution=res, groupby="solar_day",
                       chunks={"time": 1, "x": 1024, "y": 1024})
    vv = (10 * np.log10(ds["vv"].where(ds["vv"] > 0))).median("time", skipna=True)
    vh = (10 * np.log10(ds["vh"].where(ds["vh"] > 0))).median("time", skipna=True)
    return vv.compute(), vh.compute()


# ---------------------------------------------------------------------------
# Stage 1 — extraction
# ---------------------------------------------------------------------------
def build_datacube(args, bbox, geom_wgs84, periods, ncpath):
    catalog = pystac_client.Client.open(MPC_STAC, modifier=pc.sign_inplace)
    log(f"Connected to MPC. Building {len(periods)} x {args.period_days}-day composites "
        f"over AOI bbox {[round(b, 3) for b in bbox]}")

    orbit_q = {} if args.s1_orbit == "any" else {"sat:orbit_state": {"eq": args.s1_orbit}}
    slices, ref = [], None
    for p in periods:
        win = f"{p['start']}/{p['end']}"
        try:
            s2 = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                     datetime=win,
                                     query={"eo:cloud_cover": {"lt": args.max_cloud}}).items())
            s1 = list(catalog.search(collections=["sentinel-1-rtc"], bbox=bbox,
                                     datetime=win,
                                     query={"sar:instrument_mode": {"eq": "IW"}, **orbit_q}).items())
            ndvi = s2_ndvi_composite(catalog, s2, bbox, args.crs, args.res) if s2 else None
            vv, vh = s1_db_composite(catalog, s1, bbox, args.crs, args.res) if s1 else (None, None)
        except Exception as e:
            log(f"  period {p['period']:02d} {win}  ERROR: {e}")
            continue

        if ndvi is None and vv is None:
            log(f"  period {p['period']:02d} {win}  no data, skipped")
            continue
        ref = ndvi if ndvi is not None else vv
        nan_t = xr.full_like(ref, np.nan, dtype="float32")
        dd = {
            "S2ndvi": ndvi if ndvi is not None else nan_t,
            "VV": vv.rio.reproject_match(ref) if vv is not None else nan_t,
            "VH": vh.rio.reproject_match(ref) if vh is not None else nan_t,
        }
        # strip per-array CRS coord and snap every band onto the reference grid
        dd = {k: v.drop_vars("spatial_ref", errors="ignore")
                  .assign_coords(y=ref.y, x=ref.x)
              for k, v in dd.items()}
        ds = xr.Dataset(dd).assign_coords(t=np.datetime64(p["center"]))
        slices.append(ds)
        nd = float(np.isfinite(ndvi).mean()) * 100 if ndvi is not None else float("nan")
        log(f"  period {p['period']:02d} {win}  S2={len(s2):2d} S1={len(s1):2d}  "
            f"NDVI_valid={nd:4.0f}%  vars={list(dd)}")

    if not slices:
        log("No periods produced data — aborting.")
        sys.exit(1)

    cube = xr.concat(slices, dim="t").sortby("t").transpose("t", "y", "x")
    cube = cube.rio.write_crs(args.crs)

    # Optional: clip to a paddy/land-cover mask so fusion only runs on rice.
    if args.mask:
        import rioxarray as _rxr
        from rasterio.warp import transform_bounds
        ref = cube["S2ndvi"].isel(t=0)
        # window-read only the AOI footprint of a possibly continent-wide mask
        m = _rxr.open_rasterio(args.mask, masked=True, chunks={"x": 2048, "y": 2048}).squeeze()
        minx, miny, maxx, maxy = ref.rio.bounds()
        mb = transform_bounds(ref.rio.crs, m.rio.crs, minx, miny, maxx, maxy)
        m = m.rio.clip_box(*mb).compute()
        m = m.rio.reproject_match(ref)              # mask -> cube grid (handles CRS)
        keep = (m > 0).values
        cube = cube.where(keep)
        frac = float(np.nanmean(keep)) * 100
        log(f"Applied paddy mask {Path(args.mask).name}: {frac:.1f}% of grid kept as rice")

    ncpath.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in cube.data_vars}
    cube.to_netcdf(ncpath, encoding=enc)
    log(f"Datacube saved: {ncpath}  dims={dict(cube.sizes)}  vars={list(cube.data_vars)}")
    return cube


def quicklook(cube, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if "S2ndvi" in cube:
        cube["S2ndvi"].mean("t").plot(ax=axes[0], cmap="YlGn", vmin=0, vmax=0.9,
                                      cbar_kwargs={"label": "mean NDVI"})
        axes[0].set_title("Mean NDVI (period composites)")
        (np.isfinite(cube["S2ndvi"]).sum("t")).plot(ax=axes[1], cmap="viridis",
                                                    cbar_kwargs={"label": "n valid periods"})
        axes[1].set_title("S2 NDVI temporal availability")
    if "VV" in cube:
        cube["VV"].mean("t").plot(ax=axes[2], cmap="Greys_r",
                                  cbar_kwargs={"label": "mean VV (dB)"})
        axes[2].set_title("Mean VV backscatter")
    for a in axes:
        a.set_aspect("equal")
    fig.tight_layout()
    fp = outdir / "quicklook_maps.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"Quicklook maps: {fp}")


# ---------------------------------------------------------------------------
# Stage 2 — point-based MOGPR + phenology
# ---------------------------------------------------------------------------
def sample_points(aoi_utm, n, seed):
    """Uniform random points inside the AOI (rejection sampling, deterministic)."""
    geom = aoi_utm.geometry.unary_union
    minx, miny, maxx, maxy = geom.bounds
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if geom.contains(Point(x, y)):
            out.append((x, y))
    return np.array(out)


def point_mogpr(cube, coords, signals, time_step_days=12):
    """Run per-point MOGPR; returns fused DataArrays per signal on a common time grid."""
    xs = xr.DataArray(coords[:, 0], dims="point")
    ys = xr.DataArray(coords[:, 1], dims="point")
    series = {s: cube[s].sel(x=xs, y=ys, method="nearest").values  # (t, point)
              for s in signals if s in cube}
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], dtype=float)

    npts = coords.shape[0]
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, time_step_days, dtype=float)
    out_dates = [datetime.fromordinal(int(o)) for o in out_grid]
    fused = {s: np.full((len(out_grid), npts), np.nan) for s in series}

    n_ok = 0
    for j in range(npts):
        data_in, time_in, names = [], [], []
        for s in series:
            v = series[s][:, j]
            m = np.isfinite(v)
            if m.sum() >= 4:                       # need a few obs to fit
                data_in.append(v[m]); time_in.append(t_ord[m]); names.append(s)
        if len(data_in) < 1:
            continue
        try:
            out_mean, *_ = mogpr_1D(data_in, time_in, 0, out_grid, 1)
            for k, s in enumerate(names):
                fused[s][:, j] = np.ravel(out_mean[k])
            n_ok += 1
        except Exception:
            continue
        if (j + 1) % 25 == 0:
            log(f"    MOGPR points {j + 1}/{npts}")
    log(f"  MOGPR fitted {n_ok}/{npts} points")
    return out_dates, fused, series, t_ord


def point_phenology(out_dates, fused_ndvi):
    """fusets phenology on point-fused NDVI (reshaped as time x 1 x n_points)."""
    times = pd.to_datetime(out_dates)
    da = xr.DataArray(
        fused_ndvi[:, None, :],                          # (time, y=1, x=npoint)
        dims=("time", "y", "x"),
        coords={"time": times, "y": [0], "x": np.arange(fused_ndvi.shape[1])},
        name="NDVI",
    )
    ph = phenology(da)
    out = {}
    for key, name in [("sos_times", "SOS_doy"), ("eos_times", "EOS_doy"),
                      ("los_values", "LOS_steps"), ("pos_values", "peak_NDVI"),
                      ("pos_times", "POS_doy"), ("vos_values", "trough_NDVI"),
                      ("aos_values", "amplitude"), ("sios_values", "seasonal_integral"),
                      ("roi_values", "greenup_rate"), ("rod_values", "senescence_rate")]:
        if key in ph:
            out[name] = np.ravel(ph[key].values)
    return pd.DataFrame(out)


def plot_examples(out_dates, fused, series, t_ord, signals, outdir, n_plot):
    raw_dates = [datetime.fromordinal(int(o)) for o in t_ord]
    for j in range(min(n_plot, next(iter(series.values())).shape[1])):
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = {"S2ndvi": "green", "VV": "navy", "VH": "darkred"}
        for s in signals:
            if s not in series:
                continue
            ax.plot(raw_dates, series[s][:, j], "o", ms=4, alpha=0.5,
                    color=colors.get(s, "gray"), label=f"{s} raw")
            if s in fused:
                ax.plot(out_dates, fused[s][:, j], "-", lw=2,
                        color=colors.get(s, "gray"), label=f"{s} MOGPR")
        ax.set_title(f"Point {j}: MOGPR fusion")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"point_{j:03d}_mogpr.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    log(f"  saved {min(n_plot, npts_of(series))} example point plots")


def npts_of(series):
    return next(iter(series.values())).shape[1]


# ---------------------------------------------------------------------------
# Stage 3 — optional raster MOGPR (tiled)
# ---------------------------------------------------------------------------
def raster_mogpr(cube, outdir, tile, crs):
    from fusets import mogpr as mogpr_xr
    log(f"RASTER MOGPR (tiled {tile}px) — this is the slow path…")
    vars_present = [v for v in ["VV", "VH", "S2ndvi"] if v in cube]
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    fused_full = {f"{v}_FUSED": np.full((cube.sizes["t"], ny, nx), np.nan, np.float32)
                  for v in vars_present}
    for y0 in range(0, ny, tile):
        for x0 in range(0, nx, tile):
            sub = cube.isel(y=slice(y0, y0 + tile), x=slice(x0, x0 + tile))
            try:
                f = mogpr_xr(sub[vars_present], variables=vars_present, time_dimension="t")
                for v in vars_present:
                    fused_full[f"{v}_FUSED"][:, y0:y0 + sub.sizes["y"], x0:x0 + sub.sizes["x"]] = \
                        f[f"{v}_FUSED"].values
            except Exception as e:
                log(f"    tile y{y0} x{x0} failed: {e}")
            log(f"    tile y{y0}:{y0 + tile} x{x0}:{x0 + tile} done")
    out = xr.Dataset(
        {k: (("t", "y", "x"), v) for k, v in fused_full.items()},
        coords={"t": cube["t"], "y": cube["y"], "x": cube["x"]},
    ).rio.write_crs(crs)
    fp = outdir / "datacube_fused_raster.nc"
    out.to_netcdf(fp, encoding={k: {"zlib": True, "complevel": 4} for k in out.data_vars})
    log(f"  raster fused cube saved: {fp}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ncpath = outdir / "datacube_klambu_glapan.nc"

    log("=" * 70)
    log("Klambu-Glapan MOGPR pipeline")
    log("=" * 70)
    bbox, aoi_utm, geom_wgs84 = load_aoi(args.aoi, args.buffer, args.crs)
    periods = generate_periods(args.start, args.end, args.period_days)

    # --- Stage 1: extraction ---
    if args.skip_extract and ncpath.exists():
        log(f"Loading existing datacube: {ncpath}")
        cube = xr.open_dataset(ncpath).rio.write_crs(args.crs)
    else:
        cube = build_datacube(args, bbox, geom_wgs84, periods, ncpath)
    quicklook(cube, outdir)

    signals = [s for s in ["S2ndvi", "VV", "VH"] if s in cube]
    log(f"Datacube ready: dims={dict(cube.sizes)} signals={signals}")

    # --- Stage 2: point-based MOGPR + phenology ---
    log("-" * 70)
    attrs = None
    if args.points_shp:
        gp = gpd.read_file(args.points_shp).to_crs(args.crs)
        cent = gp.geometry.centroid
        coords = np.column_stack([cent.x.values, cent.y.values])
        attrs = gp.drop(columns=gp.geometry.name).reset_index(drop=True)
        log(f"Point-based MOGPR at {len(coords)} centroids from {Path(args.points_shp).name}")
    else:
        log(f"Point-based MOGPR over {args.n_points} random sample points")
        coords = sample_points(aoi_utm, args.n_points, args.seed)
    log(f"  {len(coords)} points")
    if attrs is not None:
        attrs.insert(0, "point", np.arange(len(coords)))
        attrs.insert(1, "x", coords[:, 0]); attrs.insert(2, "y", coords[:, 1])
        attrs.to_csv(outdir / "field_attributes.csv", index=False)
    out_dates, fused, series, t_ord = point_mogpr(cube, coords, signals, args.period_days)

    # save fused time series (long format)
    rows = []
    for j in range(coords.shape[0]):
        for i, d in enumerate(out_dates):
            row = {"point": j, "x": coords[j, 0], "y": coords[j, 1], "date": d}
            for s in fused:
                row[f"{s}_fused"] = fused[s][i, j]
            rows.append(row)
    fused_csv = outdir / "point_mogpr_fused.csv"
    pd.DataFrame(rows).to_csv(fused_csv, index=False)
    log(f"  fused time series: {fused_csv}")

    plot_examples(out_dates, fused, series, t_ord, signals, outdir, args.n_plot)

    if "S2ndvi" in fused:
        try:
            ph = point_phenology(out_dates, fused["S2ndvi"])
            ph.insert(0, "y", coords[:, 1]); ph.insert(0, "x", coords[:, 0])
            ph.insert(0, "point", np.arange(coords.shape[0]))
            ph_csv = outdir / "point_phenology.csv"
            ph.to_csv(ph_csv, index=False)
            log(f"  phenology metrics: {ph_csv}")
            with pd.option_context("display.width", 120):
                log("  phenology summary:\n" + ph.describe().round(2).to_string())
        except Exception as e:
            log(f"  phenology step skipped ({type(e).__name__}: {e})")

    # --- Stage 3: optional raster ---
    if args.run_raster:
        log("-" * 70)
        raster_mogpr(cube, outdir, args.raster_tile, args.crs)

    log("=" * 70)
    log(f"DONE in {time.time() - t0:.0f}s. Outputs in {outdir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()
