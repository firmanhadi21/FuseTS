#!/usr/bin/env python
"""Extract the full MOGPR-fused NDVI + VH time series at labelled drone points (0104),
for a *temporal* phase model (MiniROCKET) instead of hand-crafted scalars.

Same per-cell extraction as extract_point_features.py, but per point it keeps the whole
fused NDVI curve and the VH series resampled onto the common 12-day grid, plus the
observation-date ordinal. The temporal model windows around that date downstream.

All cells share the 2024 31×12-day grid, so series have a fixed length L.

Output:
  <out>.npz       : ndvi (N,L) float32, vh (N,L) float32, t_grid (L,), label_ord (N,)
  <out>_meta.csv  : fase, region, lokasi, sumber  (row-aligned with the npz)

Usage
-----
  python scripts/extract_point_series.py \
      --points data/aois/points_0104_all.csv \
      --vh-stack ~/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif \
      --year 2024 --cell 0.05 --res 50 --workers 16 \
      --out output/phase_model/series_0104
"""
import argparse
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
from pyproj import Transformer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from klambu_glapan_mogpr import generate_periods, s2_ndvi_composite  # noqa: E402
from scale_runner import get_catalog  # noqa: E402
from build_cube_from_snap_s1 import _load_vh_db  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402


def log(m):
    import time, os
    print(f"[{time.strftime('%H:%M:%S')} pid{os.getpid()}] {m}", flush=True)


def _process_cell(item, vh_stack, year, res):
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
            fused = np.ravel(om[0]).astype("float32")
        except Exception:
            continue
        # VH resampled onto the same grid (interp over finite VH)
        vh_grid = np.interp(out_grid, t_ord[m], vh_v[m]).astype("float32") if m.sum() >= 2 \
            else np.full(out_grid.shape, np.nan, "float32")
        # --- ABLATION additions (backward-compatible) ---
        # naive linear-interpolated NDVI (optical-only, NO SAR) onto the same grid: the cheap
        # gap-fill baseline to contrast against MOGPR fusion. `nv`/`nd_v`/`t_ord` are the finite
        # S2 NDVI observations used above as MOGPR input, so this uses IDENTICAL optical data.
        ndvi_naive = np.interp(out_grid, t_ord[nv], nd_v[nv]).astype("float32")
        # S2 valid-observation fraction (for cloud-stratified analysis of the fusion benefit)
        ndvi_valid_frac = float(nv.sum()) / float(len(nd_v))
        rows.append((fused, vh_grid, out_grid.astype("float32"),
                     float(pd.Timestamp(p.tanggal).toordinal()),
                     int(p.fase), p.region, p.lokasi, p.get("sumber", ""),
                     float(p.bujur), float(p.lintang),
                     ndvi_naive, ndvi_valid_frac))
    log(f"cell {utm},{_cx:.2f},{_cy:.2f}: {len(rows)}/{len(df)} series")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", required=True)
    p.add_argument("--vh-stack", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cell", type=float, default=0.05)
    p.add_argument("--res", type=float, default=50.0)
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

    fn = partial(_process_cell, vh_stack=a.vh_stack, year=a.year, res=a.res)
    allrows = []
    with Pool(a.workers) as pool:
        for rows in pool.imap_unordered(fn, cells):
            allrows.extend(rows)

    # all share the same grid length; align to the modal length just in case
    L = int(np.median([len(r[0]) for r in allrows]))
    keep = [r for r in allrows if len(r[0]) == L and len(r[1]) == L and len(r[10]) == L]
    ndvi = np.stack([r[0] for r in keep]); vh = np.stack([r[1] for r in keep])
    ndvi_naive = np.stack([r[10] for r in keep])          # ablation baseline (naive optical fill)
    t_grid = keep[0][2]; label_ord = np.array([r[3] for r in keep], float)
    meta = pd.DataFrame([(r[4], r[5], r[6], r[7], r[8], r[9], r[11]) for r in keep],
                        columns=["fase", "region", "lokasi", "sumber", "bujur", "lintang",
                                 "ndvi_valid_frac"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", ndvi=ndvi, vh=vh, ndvi_naive=ndvi_naive,
                        t_grid=t_grid, label_ord=label_ord)
    meta.to_csv(str(out) + "_meta.csv", index=False)
    log(f"wrote {out}.npz ({len(meta)} series, L={L}) + meta | fase {meta['fase'].value_counts().sort_index().to_dict()}")
    return out


if __name__ == "__main__":
    main()
