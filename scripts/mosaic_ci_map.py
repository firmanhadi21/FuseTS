#!/usr/bin/env python
"""Mosaic per-tile n_harvests.tif from a produce_annual_tiled run-dir into a national
cropping-intensity map (GeoTIFF + PNG) and print the annual totals.

Usage
-----
  python scripts/mosaic_ci_map.py --run-dir output/production/java_2025 --year 2025
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import rioxarray  # noqa: F401
from rioxarray.merge import merge_arrays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PIX_HA = 0.25


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--yield", dest="yld", type=float, default=5.8)
    a = p.parse_args(argv)
    rd = Path(a.run_dir)

    tifs = sorted(glob.glob(f"{rd}/tiles/*/n_harvests.tif"))
    print(f"mosaicking {len(tifs)} tiles -> EPSG:4326 ...", flush=True)
    arrs = []
    for t in tifs:
        try:
            arrs.append(rioxarray.open_rasterio(t).rio.reproject("EPSG:4326"))
        except Exception:
            pass
    mos = merge_arrays(arrs, nodata=0)
    mos.rio.to_raster(rd / "java_n_harvests.tif", compress="lzw")  # raw episode count (1-7)

    # Cropping intensity is physically 1-3 (IP100/200/300). The generative-episode counter
    # over-counts a small tail (~1%) to >=4 when the fused curve flickers; clamp it to 3.
    ci = mos.where(mos < 4, 3).rio.write_crs(mos.rio.crs)
    ci.rio.to_raster(rd / "java_cropping_intensity.tif", compress="lzw")

    # PNG: block max-pool (skipna) so thin paddy strips survive downsampling to display res
    # (a plain imshow of the full-res raster subsamples and drops most strips).
    cm = rioxarray.open_rasterio(rd / "java_cropping_intensity.tif", masked=True).isel(band=0)
    da = cm.coarsen(x=10, y=10, boundary="pad").max()
    ext = [float(cm.x.min()), float(cm.x.max()), float(cm.y.min()), float(cm.y.max())]
    cmap = ListedColormap(["#fee08b", "#66bd63", "#1a9850"])  # 1,2,3
    norm = BoundaryNorm([.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(16, 6)); ax.set_facecolor("#f5f5f5")
    im = ax.imshow(np.ma.masked_invalid(da.values), cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3], shrink=0.7)
    cb.ax.set_yticklabels(["1 (IP100)", "2 (IP200)", "3 (IP300)"]); cb.set_label("harvests / yr (cropping intensity)")
    ax.set_title(f"Java rice cropping intensity {a.year} (MOGPR S1+S2 phase model, clamped 1-3, detected paddy)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout(); fig.savefig(rd / "java_cropping_intensity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # totals from per-tile summaries
    ph = hv = 0.0
    for f in glob.glob(f"{rd}/tiles/*/.done"):
        try:
            d = json.loads(Path(f).read_text())
            if d.get("status") == "ok":
                ph += d.get("physical_ha", 0); hv += d.get("harvest_ha", 0)
        except Exception:
            pass
    summary = {"year": a.year, "physical_paddy_ha": round(ph, 1), "annual_harvest_area_ha": round(hv, 1),
               "mean_cropping_intensity": round(hv / max(1e-9, ph), 3),
               "annual_production_t": round(hv * a.yld, 1), "yield_t_ha": a.yld}
    (rd / "java_production_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"=== {a.year} === paddy {ph:,.0f} ha | CI {summary['mean_cropping_intensity']:.2f} | "
          f"harvest {hv:,.0f} ha | production {summary['annual_production_t']/1e6:.2f} Mt", flush=True)
    print(f"wrote {rd}/java_cropping_intensity.{{tif,png}} (clamped 1-3), java_n_harvests.tif (raw), "
          f"java_production_summary.json")


if __name__ == "__main__":
    main()
