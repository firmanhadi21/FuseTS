#!/usr/bin/env python
"""Study-area figure for the companion manuscript, on an OpenStreetMap basemap.

Reproduces the previous fig_study_area (Java paddy in grey + 14,187 combined-survey
drone phase points coloured by season + Bengawan Solo corridor box) but renders it in
Web Mercator (EPSG:3857) over OSM tiles via contextily.

Run with the `fusets` conda env (has rioxarray, geopandas, contextily).
"""
import numpy as np
import pandas as pd
import rioxarray
import contextily as ctx
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MASK = "/home/unika_sianturi/work/landcover/s1-land-cover-classification/cropping_intensity_consensus_mt2024_25/paddy_mask.tif"
PTS = "/home/unika_sianturi/work/FuseTS/data/aois/points_combined_java.csv"
OUT = "/home/unika_sianturi/work/rice-growth-stage-mapping/paper_latex/companion_figures/fig_study_area.png"

to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# --- paddy mask: coarsen, then reproject to Web Mercator for OSM overlay ---
m = rioxarray.open_rasterio(MASK, masked=True).isel(band=0)
mm = (m > 0).astype("float32").fillna(0.0).coarsen(x=20, y=20, boundary="pad").sum()
mm = mm.rio.write_crs("EPSG:4326").rio.write_nodata(0.0)
mm = mm.rio.reproject("EPSG:3857")
bg = mm.values.astype(float)
bg = np.where(np.isfinite(bg), bg, 0.0)
l, b, r, t = mm.rio.bounds()  # already EPSG:3857
ext = [l, r, b, t]

# --- points coloured by season (wet = Mar-Jun, dry = the Oct campaign ~43%) ---
p = pd.read_csv(PTS)
d = pd.to_datetime(p.tanggal, errors="coerce")
p["season"] = np.where(d.dt.month.isin([3, 4, 5, 6]), "wet", "dry")
px, py = to3857.transform(p.bujur.values, p.lintang.values)
p["mx"], p["my"] = px, py

fig, ax = plt.subplots(figsize=(13, 4.2))

# paddy extent (grey), drawn above the basemap
ax.imshow(np.ma.masked_where(bg <= 0, bg), cmap="Greys", vmin=0, vmax=bg.max() or 1,
          alpha=0.55, extent=ext, interpolation="nearest", zorder=2)

for s, c in [("wet", "#2c7fb8"), ("dry", "#e6550d")]:
    q = p[p.season == s]
    ax.scatter(q.mx, q.my, s=3, c=c, alpha=0.6, edgecolors="none", zorder=3,
               label=f"{s} season (n={len(q):,})")

# Bengawan Solo corridor: lon 110.7-112.7, lat -7.75 to -6.85
bx0, by0 = to3857.transform(110.7, -7.75)
bx1, by1 = to3857.transform(112.7, -6.85)
ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                        ec="#1f78b4", lw=1.4, ls="--", zorder=4))
tx, ty = to3857.transform(111.0, -7.95)
ax.text(tx, ty, "Bengawan Solo", color="#1f78b4", fontsize=8, zorder=4)

# axis window (EPSG:3857), matching the original lon 105-114.8, lat -8.9 to -5.6
x0, y0 = to3857.transform(105.0, -8.9)
x1, y1 = to3857.transform(114.8, -5.6)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)

# OSM basemap behind everything
ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik,
                attribution_size=5, zorder=0)

ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower left", fontsize=8, markerscale=3, framealpha=0.9)
ax.set_title("Study area: Java rice paddy (grey) and combined-survey drone phase "
             "observations (n=14,187), on OpenStreetMap")
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT)
