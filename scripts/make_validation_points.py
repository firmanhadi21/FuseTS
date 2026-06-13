#!/usr/bin/env python
"""Generate field-validation points for the growth-stage maps — stratified, spaced, in paddy,
within a drive of a base town (e.g. Semarang). Outputs phone-loadable GeoJSON + CSV (+ KML).

Stratifies by a raster (the per-period phase map when available, else the 2024 cropping-
intensity as a spatial-spread proxy): N points per stratum value, each ≥ --min-dist apart,
only on paddy (mask>0), inside the bbox. Each point carries the predicted/stratum value plus
blank field-record columns (observed_fase, transplant_date, notes) for the field protocol.

Usage
-----
  # preliminary (spread by 2024 cropping intensity), near Semarang:
  python scripts/make_validation_points.py --strat output/production/java/java_n_harvests.tif \
     --mask .../paddy_mask.tif --bbox 110.15 -7.20 110.75 -6.80 \
     --n-per 8 --min-dist 0.02 --out output/validation/semarang_prelim

  # final (stratified by predicted p13 phase, when maps exist):
  python scripts/make_validation_points.py --strat output/production/java_2026_phases/java_phase6_p13.tif \
     --mask .../paddy_mask.tif --bbox 110.15 -7.20 110.75 -6.80 --n-per 8 --min-dist 0.02 \
     --out output/validation/semarang_p13
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strat", required=True, help="stratifier raster (phase map or cropping-intensity)")
    p.add_argument("--mask", required=True, help="paddy mask (>0 = paddy)")
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--n-per", type=int, default=8, help="points per stratum value")
    p.add_argument("--min-dist", type=float, default=0.02, help="min spacing between points (deg, ~2km)")
    p.add_argument("--exclude", type=int, nargs="*", default=[0], help="stratum values to skip (e.g. 0 nodata)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    w, s, e, n = a.bbox
    strat = rioxarray.open_rasterio(a.strat, masked=True, chunks={"x": 2048, "y": 2048}).rio.clip_box(w, s, e, n).squeeze()
    mask = rioxarray.open_rasterio(a.mask, masked=True, chunks={"x": 2048, "y": 2048}).rio.reproject_match(strat).squeeze()
    sv = strat.values
    paddy = np.isfinite(sv) & np.isfinite(mask.values) & (mask.values > 0)
    xs = strat.x.values; ys = strat.y.values

    chosen = []  # (lon, lat, stratum)
    for val in sorted(set(np.unique(sv[paddy]).astype(int).tolist()) - set(a.exclude)):
        cand = np.argwhere(paddy & (sv == val))
        rng.shuffle(cand)
        picked = []
        for (r, c) in cand:
            lon, lat = float(xs[c]), float(ys[r])
            if all((lon - L) ** 2 + (lat - La) ** 2 >= a.min_dist ** 2 for L, La, _ in picked):
                picked.append((lon, lat, val))
                if len(picked) >= a.n_per:
                    break
        chosen.extend(picked)

    # CSV
    lines = ["id,lon,lat,pred_stratum,observed_fase,transplant_date,notes"]
    feats = []
    for i, (lon, lat, val) in enumerate(chosen, 1):
        lines.append(f"VP{i:03d},{lon:.6f},{lat:.6f},{int(val)},,,")
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": {"id": f"VP{i:03d}", "pred_stratum": int(val),
                                     "observed_fase": "", "transplant_date": "", "notes": ""}})
    (str(out) + ".csv") and Path(str(out) + ".csv").write_text("\n".join(lines) + "\n")
    Path(str(out) + ".geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))
    # KML (phone nav)
    kml = ['<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>']
    for i, (lon, lat, val) in enumerate(chosen, 1):
        kml.append(f'<Placemark><name>VP{i:03d} (s{int(val)})</name>'
                   f'<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>')
    kml.append("</Document></kml>")
    Path(str(out) + ".kml").write_text("\n".join(kml))

    from collections import Counter
    dist = dict(sorted(Counter(int(v) for _, _, v in chosen).items()))
    print(f"{len(chosen)} validation points | per-stratum: {dist}")
    print(f"wrote {out}.csv / .geojson / .kml")
    return chosen


if __name__ == "__main__":
    main()
