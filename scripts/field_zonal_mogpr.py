#!/usr/bin/env python
"""
Field-polygon-mean MOGPR fusion from an existing datacube.

Instead of sampling one centroid pixel per field (which at 10 m can straddle
bunds/adjacent plots), this averages NDVI/VV/VH over each field polygon, then
runs MOGPR per field. Output matches point_mogpr_fused.csv so pv_phenology.py
can consume it directly.

    python scripts/field_zonal_mogpr.py \
        --cube output/bulak_bakal/datacube_klambu_glapan.nc \
        --fields "Validasi_Data_BulakBakal/Digit merge.shp" \
        --attrs output/bulak_bakal/field_attributes.csv \
        --outdir output/bulak_bakal_fieldmean
"""
from __future__ import annotations

import argparse
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401

from fusets.mogpr import mogpr_1D


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--attrs", default=None, help="field_attributes.csv to copy across")
    p.add_argument("--outdir", required=True)
    p.add_argument("--crs", default="EPSG:32749")
    p.add_argument("--period-days", type=int, default=12)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cube = xr.open_dataset(a.cube).rio.write_crs(a.crs)
    signals = [s for s in ["S2ndvi", "VV", "VH"] if s in cube]
    fields = gpd.read_file(a.fields).to_crs(a.crs).reset_index(drop=True)
    log(f"cube {dict(cube.sizes)}  signals={signals}  fields={len(fields)}")

    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], dtype=float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, a.period_days, dtype=float)
    out_dates = [datetime.fromordinal(int(o)) for o in out_grid]

    rows, n_ok = [], 0
    for i, geom in enumerate(fields.geometry):
        # zonal-mean time series per band (all_touched for sub-pixel fields)
        series = {}
        for s in signals:
            try:
                clip = cube[s].rio.clip([geom], a.crs, all_touched=True, drop=True)
                series[s] = clip.mean(dim=[d for d in clip.dims if d != "t"]).values
            except Exception:
                series[s] = np.full(cube.sizes["t"], np.nan)
        cen = geom.centroid

        data_in, time_in, names = [], [], []
        for s in signals:
            v = series[s]; m = np.isfinite(v)
            if m.sum() >= 4:
                data_in.append(v[m]); time_in.append(t_ord[m]); names.append(s)
        fused = {s: np.full(len(out_grid), np.nan) for s in signals}
        if data_in:
            try:
                out_mean, *_ = mogpr_1D(data_in, time_in, 0, out_grid, 1)
                for k, s in enumerate(names):
                    fused[s] = np.ravel(out_mean[k])
                n_ok += 1
            except Exception:
                pass
        for di, d in enumerate(out_dates):
            row = {"point": i, "x": cen.x, "y": cen.y, "date": d}
            for s in signals:
                row[f"{s}_fused"] = fused[s][di]
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "point_mogpr_fused.csv", index=False)
    log(f"fused {n_ok}/{len(fields)} fields -> {outdir/'point_mogpr_fused.csv'}")

    if a.attrs and Path(a.attrs).exists():
        shutil.copy(a.attrs, outdir / "field_attributes.csv")
        log(f"copied attributes -> {outdir/'field_attributes.csv'}")


if __name__ == "__main__":
    main()
