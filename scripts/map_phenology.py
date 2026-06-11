#!/usr/bin/env python
"""
Map Klambu-Glapan phenology results.

Produces two kinds of maps from the outputs of klambu_glapan_mogpr.py:

1. RASTER (wall-to-wall): Whittaker-smooth the 12-day NDVI datacube along time,
   run FuseTS phenology per pixel, and export each metric as a GeoTIFF plus a
   multi-panel PNG. This is the spatial phenology product.
2. POINTS: if point_phenology.csv exists, plot the per-point metrics as colored
   scatter maps over the AOI outline.

Run
---
    conda activate fusets
    python scripts/map_phenology.py                 # uses output/klambu_glapan
    python scripts/map_phenology.py --no-raster      # point maps only (fast)
    python scripts/map_phenology.py --lambda 5000    # Whittaker smoothing strength
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
import geopandas as gpd
import rioxarray  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusets import whittaker
from fusets.analytics import phenology

# metric key -> (nice label, colormap, value range or None)
METRICS = [
    ("sos_times", "Start of Season (DOY)", "twilight", (1, 365)),
    ("pos_times", "Peak of Season (DOY)", "twilight", (1, 365)),
    ("eos_times", "End of Season (DOY)", "twilight", (1, 365)),
    ("los_values", "Length of Season (steps)", "viridis", None),
    ("pos_values", "Peak NDVI", "YlGn", (0, 1)),
    ("aos_values", "Amplitude", "plasma", (0, 1)),
    ("sios_values", "Seasonal Integral", "copper_r", None),
    ("roi_values", "Green-up Rate", "Greens", None),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_args(argv=None):
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=str(repo / "output" / "klambu_glapan"))
    p.add_argument("--aoi", default=str(repo / "data" / "klambu-glapan.shp"))
    p.add_argument("--lambda", dest="lam", type=float, default=5000.0,
                   help="Whittaker smoothing_lambda (higher = smoother)")
    p.add_argument("--no-raster", action="store_true", help="skip wall-to-wall raster maps")
    return p.parse_args(argv)


def raster_phenology(outdir, aoi_path, lam):
    ncpath = outdir / "datacube_klambu_glapan.nc"
    if not ncpath.exists():
        log(f"datacube not found ({ncpath}); skipping raster maps")
        return
    cube = xr.open_dataset(ncpath)
    crs = cube.rio.crs or "EPSG:32749"
    ndvi = cube["S2ndvi"]  # (t, y, x)
    log(f"NDVI cube {dict(ndvi.sizes)} — Whittaker smoothing (lambda={lam:.0f})…")
    try:
        ndvi_s = whittaker(ndvi, smoothing_lambda=lam, time_dimension="t")
    except Exception as e:
        log(f"  whittaker failed ({e}); using raw composites")
        ndvi_s = ndvi
    da = ndvi_s.rename({"t": "time"})
    da = da.assign_coords(time=cube["t"].values).clip(-0.2, 1.0)

    log("Computing per-pixel phenology (this takes a few minutes)…")
    t0 = time.time()
    ph = phenology(da)
    log(f"  phenology done in {time.time() - t0:.0f}s; vars={list(ph.data_vars)}")

    # GeoTIFF exports
    tif_dir = outdir / "phenology_tifs"
    tif_dir.mkdir(exist_ok=True)
    for key, label, *_ in METRICS:
        if key not in ph:
            continue
        da_m = ph[key].rio.write_crs(crs)
        da_m.rio.to_raster(tif_dir / f"{key}.tif", compress="lzw")
    log(f"  GeoTIFFs written to {tif_dir}/")

    # AOI outline in cube CRS
    aoi = gpd.read_file(aoi_path).to_crs(crs)

    present = [(k, lbl, cm, rng) for (k, lbl, cm, rng) in METRICS if k in ph]
    n = len(present)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, label, cmap, rng) in zip(axes, present):
        d = ph[key].astype("float32").where(np.isfinite(ph[key]))
        kw = dict(cmap=cmap, add_colorbar=True, cbar_kwargs={"shrink": 0.8})
        if rng:
            kw.update(vmin=rng[0], vmax=rng[1])
        else:
            vals = d.values[np.isfinite(d.values)]
            if vals.size:
                kw.update(vmin=np.nanpercentile(vals, 2), vmax=np.nanpercentile(vals, 98))
        d.plot.imshow(ax=ax, **kw)
        aoi.boundary.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Klambu-Glapan Phenology (Whittaker-smoothed S2 NDVI, 2024-11 → 2025-10)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fp = outdir / "phenology_maps_raster.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  raster map figure: {fp}")


def point_maps(outdir, aoi_path):
    csv = outdir / "point_phenology.csv"
    if not csv.exists():
        log(f"point phenology CSV not found ({csv}); skipping point maps")
        return
    df = pd.read_csv(csv)
    crs = "EPSG:32749"
    aoi = gpd.read_file(aoi_path).to_crs(crs)
    cols = [("SOS_doy", "Start of Season (DOY)", "twilight"),
            ("POS_doy", "Peak of Season (DOY)", "twilight"),
            ("EOS_doy", "End of Season (DOY)", "twilight"),
            ("LOS_steps", "Length of Season", "viridis"),
            ("peak_NDVI", "Peak NDVI", "YlGn"),
            ("amplitude", "Amplitude", "plasma"),
            ("seasonal_integral", "Seasonal Integral", "copper_r")]
    cols = [c for c in cols if c[0] in df.columns]
    if not cols:
        log("no phenology metric columns in CSV; skipping point maps")
        return
    n = len(cols); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, label, cmap) in zip(axes, cols):
        aoi.boundary.plot(ax=ax, color="black", linewidth=0.6)
        sc = ax.scatter(df["x"], df["y"], c=df[col], cmap=cmap, s=22,
                        edgecolor="k", linewidth=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Klambu-Glapan Point Phenology (MOGPR-fused NDVI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fp = outdir / "phenology_maps_points.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  point map figure: {fp}")


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    log("=" * 66)
    log("Mapping Klambu-Glapan phenology")
    log("=" * 66)
    point_maps(outdir, args.aoi)
    if not args.no_raster:
        raster_phenology(outdir, args.aoi, args.lam)
    log("Done.")


if __name__ == "__main__":
    main()
