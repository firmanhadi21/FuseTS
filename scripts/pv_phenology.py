#!/usr/bin/env python
"""
Multi-season phenology for rice using FuseTS `peakvalley`.

Unlike the single-season phenolopy metrics, this detects EVERY growing cycle in
a fused-NDVI time series (rice in Klambu-Glapan / Central Java is typically
double- or triple-cropped). For each point it runs `peakvalley_f` on the
MOGPR-fused NDVI and extracts one record per detected season:

    point, season, SOS, POS, EOS, peak_NDVI, trough_NDVI, amplitude, LOS_days

Reusable from the main pipeline (`extract_point_seasons`) or as a CLI that reads
an existing point_mogpr_fused.csv:

    python scripts/pv_phenology.py --outdir output/klambu_glapan
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusets.peakvalley import peakvalley_f


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_point_seasons(dates, ndvi, drop_thr=0.15, rec_r=1.0, slope_thr=-0.007):
    """dates: 1D datetimes, ndvi: 1D fused NDVI -> list of per-season metric dicts."""
    x = np.array([pd.Timestamp(d).to_pydatetime() for d in dates], dtype=object)
    return _seasons_from_pydate(x, np.asarray(ndvi, dtype=float),
                                drop_thr, rec_r, slope_thr)


def _seasons_from_pydate(x, y, drop_thr=0.15, rec_r=1.0, slope_thr=-0.007):
    """x: object array of python datetimes; y: float NDVI (same length)."""
    valid = np.isfinite(y)
    if valid.sum() < 5:
        return []
    xv, yv = x[valid], y[valid]
    try:
        _, pairs = peakvalley_f(xv, yv, drop_thr, rec_r, slope_thr)
    except Exception:
        return []
    pairs = np.atleast_2d(np.asarray(pairs))
    if pairs.size == 0 or pairs.shape[-1] != 2:
        return []
    # peakvalley_f returns [peak_idx, valley_idx] pairs. Green-up start (SOS) is the
    # NDVI minimum between the previous valley and this peak; EOS is the valley.
    seasons = []
    prev_vl = 0
    for pk, vl in pairs:
        pk, vl = int(pk), int(vl)
        if vl <= pk:
            continue
        lo = min(prev_vl, pk)
        sos_idx = lo + int(np.argmin(yv[lo:pk + 1])) if pk > lo else pk
        sos, pos, eos = xv[sos_idx], xv[pk], xv[vl]
        peak, trough, sos_val = float(yv[pk]), float(yv[vl]), float(yv[sos_idx])
        seasons.append({
            "SOS": pd.Timestamp(sos), "POS": pd.Timestamp(pos), "EOS": pd.Timestamp(eos),
            "SOS_doy": sos.timetuple().tm_yday, "POS_doy": pos.timetuple().tm_yday,
            "EOS_doy": eos.timetuple().tm_yday, "peak_NDVI": round(peak, 3),
            "trough_NDVI": round(trough, 3),
            "amplitude": round(peak - min(sos_val, trough), 3),
            "LOS_days": (eos - sos).days, "greenup_days": (pos - sos).days,
        })
        prev_vl = vl
    return seasons


def run(outdir, aoi_path, drop_thr=0.15, rec_r=1.0, slope_thr=-0.007,
        ndvi_col="S2ndvi_fused", crs="EPSG:32749"):
    outdir = Path(outdir)
    fused_csv = outdir / "point_mogpr_fused.csv"
    if not fused_csv.exists():
        log(f"fused CSV not found: {fused_csv}")
        return
    df = pd.read_csv(fused_csv, parse_dates=["date"])
    if ndvi_col not in df.columns:
        log(f"{ndvi_col} not in fused CSV (have {list(df.columns)})")
        return

    rows = []
    pts = df.groupby("point")
    for pid, g in pts:
        g = g.sort_values("date")
        seasons = extract_point_seasons(g["date"].values, g[ndvi_col].values,
                                        drop_thr, rec_r, slope_thr)
        x, y = float(g["x"].iloc[0]), float(g["y"].iloc[0])
        for si, s in enumerate(seasons, 1):
            rows.append({"point": pid, "x": x, "y": y, "season": si, **s})
    if not rows:
        log("no seasons detected"); return
    sea = pd.DataFrame(rows)
    sea_csv = outdir / "point_phenology_seasons.csv"
    sea.to_csv(sea_csv, index=False)
    log(f"seasons CSV: {sea_csv}  ({len(sea)} season-records over "
        f"{sea['point'].nunique()} points)")

    # per-point summary
    summ = (sea.groupby("point")
               .agg(n_seasons=("season", "max"),
                    x=("x", "first"), y=("y", "first"),
                    mean_peak_NDVI=("peak_NDVI", "mean"),
                    mean_LOS_days=("LOS_days", "mean"),
                    mean_amplitude=("amplitude", "mean"))
               .reset_index())
    summ.to_csv(outdir / "point_phenology_pv_summary.csv", index=False)
    nseas = summ["n_seasons"]
    log(f"seasons/point: mean={nseas.mean():.2f}  "
        f"dist={ {int(k): int(v) for k, v in nseas.value_counts().sort_index().items()} }")

    _maps(sea, summ, aoi_path, crs, outdir)
    return sea, summ


def _maps(sea, summ, aoi_path, crs, outdir):
    aoi = gpd.read_file(aoi_path).to_crs(crs)
    s1 = sea[sea.season == 1]
    s2 = sea[sea.season == 2]
    panels = [
        (summ, "n_seasons", "Number of seasons", "turbo"),
        (s1, "POS_doy", "Season 1 — Peak DOY", "twilight"),
        (s2, "POS_doy", "Season 2 — Peak DOY", "twilight"),
        (s1, "peak_NDVI", "Season 1 — Peak NDVI", "YlGn"),
        (s2, "peak_NDVI", "Season 2 — Peak NDVI", "YlGn"),
        (summ, "mean_LOS_days", "Mean season length (days)", "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    for ax, (data, col, label, cmap) in zip(axes, panels):
        aoi.boundary.plot(ax=ax, color="black", linewidth=0.6)
        if col in data.columns and len(data):
            sc = ax.scatter(data["x"], data["y"], c=data[col], cmap=cmap, s=26,
                            edgecolor="k", linewidth=0.2)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Klambu-Glapan multi-season phenology (FuseTS peakvalley on MOGPR-fused NDVI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fp = outdir / "phenology_maps_peakvalley.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"peakvalley maps: {fp}")


def run_raster(outdir, aoi_path, drop_thr=0.15, rec_r=1.0, slope_thr=-0.007,
               lam=5000.0, max_seasons=3, crs="EPSG:32749"):
    """Wall-to-wall multi-season peakvalley on the Whittaker-smoothed NDVI cube."""
    import xarray as xr
    import rioxarray  # noqa: F401
    from fusets import whittaker

    outdir = Path(outdir)
    ncpath = outdir / "datacube_klambu_glapan.nc"
    if not ncpath.exists():
        log(f"datacube not found: {ncpath}"); return
    cube = xr.open_dataset(ncpath)
    rcrs = cube.rio.crs or crs
    ndvi = cube["S2ndvi"]
    log(f"raster peakvalley on {dict(ndvi.sizes)} — Whittaker smoothing (lambda={lam:.0f})…")
    try:
        ndvi = whittaker(ndvi, smoothing_lambda=lam, time_dimension="t")
    except Exception as e:
        log(f"  whittaker failed ({e}); using raw composites")
    ndvi = ndvi.clip(-0.2, 1.0).transpose("t", "y", "x")

    x_py = np.array([pd.Timestamp(t).to_pydatetime() for t in cube["t"].values], dtype=object)
    nm = 1 + max_seasons * 3  # n_seasons + (POS_doy, peak, LOS) per season

    def _cb(ts):
        s = _seasons_from_pydate(x_py, np.asarray(ts, float), drop_thr, rec_r, slope_thr)
        out = np.full(nm, np.nan, np.float32)
        out[0] = len(s)
        for i, se in enumerate(s[:max_seasons]):
            out[1 + i * 3] = se["POS_doy"]
            out[2 + i * 3] = se["peak_NDVI"]
            out[3 + i * 3] = se["LOS_days"]
        return out

    log("  running per-pixel season detection (vectorized; a few minutes)…")
    t0 = time.time()
    res = xr.apply_ufunc(
        _cb, ndvi,
        input_core_dims=[["t"]], output_core_dims=[["metric"]],
        vectorize=True, output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"metric": nm}},
    )
    res = res.compute()
    log(f"  done in {time.time() - t0:.0f}s")

    names = ["n_seasons"]
    for i in range(1, max_seasons + 1):
        names += [f"s{i}_POS_doy", f"s{i}_peak_NDVI", f"s{i}_LOS_days"]
    layers = {nm_: res.isel(metric=k).rio.write_crs(rcrs) for k, nm_ in enumerate(names)}

    tif_dir = outdir / "phenology_tifs_peakvalley"
    tif_dir.mkdir(exist_ok=True)
    for nm_, da in layers.items():
        da.rio.to_raster(tif_dir / f"{nm_}.tif", compress="lzw")
    log(f"  GeoTIFFs: {tif_dir}/  ({len(layers)} layers)")

    aoi = gpd.read_file(aoi_path).to_crs(rcrs)
    panel = [("n_seasons", "Number of seasons", "turbo", (0, max_seasons)),
             ("s1_POS_doy", "Season 1 — Peak DOY", "twilight", (1, 365)),
             ("s2_POS_doy", "Season 2 — Peak DOY", "twilight", (1, 365)),
             ("s1_peak_NDVI", "Season 1 — Peak NDVI", "YlGn", (0, 1)),
             ("s2_peak_NDVI", "Season 2 — Peak NDVI", "YlGn", (0, 1)),
             ("s1_LOS_days", "Season 1 — Length (days)", "viridis", (40, 160))]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11)); axes = axes.ravel()
    for ax, (nm_, label, cmap, rng) in zip(axes, panel):
        layers[nm_].where(np.isfinite(layers[nm_])).plot.imshow(
            ax=ax, cmap=cmap, vmin=rng[0], vmax=rng[1],
            add_colorbar=True, cbar_kwargs={"shrink": 0.8})
        aoi.boundary.plot(ax=ax, color="black", linewidth=0.5)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Klambu-Glapan multi-season phenology — WALL-TO-WALL "
                 "(FuseTS peakvalley on Whittaker-smoothed NDVI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fp = outdir / "phenology_maps_peakvalley_raster.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"  raster multi-season figure: {fp}")


def parse_args(argv=None):
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=str(repo / "output" / "klambu_glapan"))
    p.add_argument("--aoi", default=str(repo / "data" / "klambu-glapan.shp"))
    p.add_argument("--drop-thr", type=float, default=0.15,
                   help="min NDVI drop to count a season")
    p.add_argument("--rec-r", type=float, default=1.0, help="recovery ratio")
    p.add_argument("--slope-thr", type=float, default=-0.007, help="onset slope threshold")
    p.add_argument("--crs", default="EPSG:32749")
    p.add_argument("--raster", action="store_true",
                   help="ALSO compute wall-to-wall multi-season maps")
    p.add_argument("--raster-only", action="store_true",
                   help="ONLY compute the wall-to-wall raster maps")
    p.add_argument("--lambda", dest="lam", type=float, default=5000.0,
                   help="Whittaker smoothing for the raster path")
    p.add_argument("--max-seasons", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    log("Multi-season phenology via FuseTS peakvalley")
    if not a.raster_only:
        run(a.outdir, a.aoi, a.drop_thr, a.rec_r, a.slope_thr, crs=a.crs)
    if a.raster or a.raster_only:
        run_raster(a.outdir, a.aoi, a.drop_thr, a.rec_r, a.slope_thr,
                   lam=a.lam, max_seasons=a.max_seasons, crs=a.crs)
    log("Done.")


if __name__ == "__main__":
    main()
