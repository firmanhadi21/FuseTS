#!/usr/bin/env python
"""Extract MOGPR phenology features at labelled drone points (idmai 0104), for a
*trained* phase classifier (replaces the brittle [POS,EOS] rule).

For each point: sample its VH (your SNAP stack) + NDVI (MPC S2) 2024 series, MOGPR-fuse,
and compute features describing where the observation date sits on the fused curve.
Points are processed in small lon/lat cells (only VH+NDVI over each cell is loaded), so
it scales to Java-wide points without building wall-to-wall cubes.

Features per point (at the observation date d):
  ndvi_d        fused NDVI at d
  slope_d       d/dt of fused NDVI at d (per day)        -> rising (veg) vs falling (late gen)
  vh_d          VH (dB) at d
  days_to_POS   signed days from nearest season's peak   -> <0 before peak, >0 after
  days_from_SOS days since nearest season's green-up
  rel_pos       (d-SOS)/(EOS-SOS) within nearest season  -> 0=start .. 1=harvest
  in_season     d within [SOS,EOS] of some season
  amplitude, peak_ndvi, los  of the nearest season
  n_seasons

Output CSV: one row per point with features + `fase`, `region`, `lokasi`.

Usage
-----
  python scripts/extract_point_features.py \
      --points data/aois/points_0104_all.csv \
      --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
      --year 2024 --cell 0.05 --res 50 --workers 16 \
      --out output/phase_model/features_0104.csv
"""
import argparse
import sys
import warnings
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
from pyproj import Transformer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from klambu_glapan_mogpr import generate_periods, s2_ndvi_composite  # noqa: E402
from scale_runner import get_catalog  # noqa: E402
from build_cube_from_snap_s1 import _load_vh_db  # noqa: E402
from pv_phenology import _seasons_from_pydate  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')} pid{__import__('os').getpid()}] {m}", flush=True)


def _features(fused, vh_series, t_ord_vh, out_grid, label_ord, drop_thr):
    x_py = np.array([datetime.fromordinal(int(o)) for o in out_grid], dtype=object)
    ndvi_d = float(np.interp(label_ord, out_grid, fused))
    slope_d = float((np.interp(label_ord + 12, out_grid, fused)
                     - np.interp(label_ord - 12, out_grid, fused)) / 24.0)
    vh_d = float(np.interp(label_ord, t_ord_vh, vh_series)) if np.isfinite(vh_series).any() else np.nan
    seasons = _seasons_from_pydate(x_py, fused, drop_thr)
    f = {"ndvi_d": ndvi_d, "slope_d": slope_d, "vh_d": vh_d, "n_seasons": len(seasons),
         "days_to_POS": np.nan, "days_from_SOS": np.nan, "rel_pos": np.nan,
         "in_season": 0, "amplitude": np.nan, "peak_ndvi": np.nan, "los": np.nan}
    if seasons:
        ld = pd.Timestamp.fromordinal(int(label_ord))
        # nearest season by |days to POS|
        s = min(seasons, key=lambda s: abs((ld - s["POS"]).days))
        f["days_to_POS"] = (ld - s["POS"]).days
        f["days_from_SOS"] = (ld - s["SOS"]).days
        los = max(1, (s["EOS"] - s["SOS"]).days)
        f["rel_pos"] = (ld - s["SOS"]).days / los
        f["in_season"] = int(s["SOS"] <= ld <= s["EOS"])
        f["amplitude"] = s.get("amplitude", np.nan)
        f["peak_ndvi"] = s["peak_NDVI"]
        f["los"] = los
    return f


def _process_cell(item, vh_stack, year, res, drop_thr):
    (utm, _cx, _cy), df = item
    crs = f"EPSG:{utm}"
    pad = 0.01
    bbox = [df.bujur.min() - pad, df.lintang.min() - pad, df.bujur.max() + pad, df.lintang.max() + pad]
    try:
        vh, periods = _load_vh_db(vh_stack, year, bbox, crs, res)
    except Exception as e:
        log(f"cell {utm},{_cx},{_cy}: VH load failed {e}")
        return []
    ref = vh.isel(band=0, drop=True)
    pinfo = {pp["period"]: pp for pp in generate_periods(f"{year}-01-01", f"{year}-12-31", 12)}
    cat = get_catalog()
    ndvi_slices, vh_keep, t_centers = [], [], []
    for bi, pn in enumerate(periods):
        prd = pinfo.get(pn)
        if prd is None:
            continue
        win = f"{prd['start']}/{prd['end']}"
        try:
            s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=win,
                                 query={"eo:cloud_cover": {"lt": 90}}).items())
            nd = s2_ndvi_composite(cat, s2, bbox, crs, res) if s2 else None
        except Exception:
            nd = None
        nd = nd.rio.reproject_match(ref) if nd is not None else xr.full_like(ref, np.nan, "float32")
        ndvi_slices.append(nd.assign_coords(y=ref.y, x=ref.x))
        vh_keep.append(vh.isel(band=bi, drop=True))
        t_centers.append(np.datetime64(prd["center"]))
    NDVI = xr.concat(ndvi_slices, dim="t").assign_coords(t=t_centers)
    VH = xr.concat(vh_keep, dim="t").assign_coords(t=t_centers)
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in t_centers], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, 12, float)

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rows = []
    for _, p in df.iterrows():
        px, py = tf.transform(p.bujur, p.lintang)
        nd_v = NDVI.sel(x=px, y=py, method="nearest").values
        vh_v = VH.sel(x=px, y=py, method="nearest").values
        nv = np.isfinite(nd_v)
        if nv.sum() < 2:
            continue
        data, tin = [nd_v[nv]], [t_ord[nv]]
        m = np.isfinite(vh_v)
        if m.sum() >= 4:
            data.append(vh_v[m]); tin.append(t_ord[m])
        if len(data) < 2:
            continue
        try:
            om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
            fused = np.ravel(om[0])
        except Exception:
            continue
        lab_ord = pd.Timestamp(p.tanggal).toordinal()
        feat = _features(fused, vh_v, t_ord, out_grid, lab_ord, drop_thr)
        feat.update({"lokasi": p.lokasi, "region": p.region, "fase": int(p.fase),
                     "sumber": p.get("sumber", "")})
        rows.append(feat)
    log(f"cell {utm},{_cx:.2f},{_cy:.2f}: {len(rows)}/{len(df)} pts featurised")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", required=True)
    p.add_argument("--vh-stack", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cell", type=float, default=0.05, help="processing cell size (deg)")
    p.add_argument("--res", type=float, default=50.0)
    p.add_argument("--drop-thr", type=float, default=0.10, help="low: detect seasons for features")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    d = pd.read_csv(a.points)
    d["tanggal"] = pd.to_datetime(d["tanggal"], errors="coerce")
    d = d.dropna(subset=["tanggal", "bujur", "lintang", "fase"])
    d["cx"] = (d.bujur / a.cell).round() * a.cell
    d["cy"] = (d.lintang / a.cell).round() * a.cell
    cells = list(d.groupby(["utm", "cx", "cy"]))
    log(f"{len(d)} points -> {len(cells)} cells | workers {a.workers}")

    fn = partial(_process_cell, vh_stack=a.vh_stack, year=a.year, res=a.res, drop_thr=a.drop_thr)
    allrows = []
    with Pool(a.workers) as pool:
        for rows in pool.imap_unordered(fn, cells):
            allrows.extend(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(allrows)
    df.to_csv(out, index=False)
    log(f"wrote {out}: {len(df)} featurised points | fase dist {df['fase'].value_counts().sort_index().to_dict()}")
    return df


if __name__ == "__main__":
    main()
