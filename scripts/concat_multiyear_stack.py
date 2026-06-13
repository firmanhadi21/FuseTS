#!/usr/bin/env python
"""Concatenate a base multi-year VH stack with an updated per-year stack into the combined
multi-year stack used by the phase pipeline, **preserving `YYYY_Period_N` band names**.

Used by the new-period ingestion (see INGEST_NEW_PERIOD.md). The base keeps its existing band
descriptions; the appended per-year stack's bands are (re)named `<append-year>_Period_<i>`,
which is the naming `produce_annual_tiled._load_vh_db` selects on.

All inputs must share grid/dtype (int16, nodata -32768, same width/height/CRS).

Usage
-----
  python scripts/concat_multiyear_stack.py \
      --base   /…/stacks/java_vh_2024_2025_50m.tif \
      --append /…/stacks/java_vh_2026_50m.tif --append-year 2026 \
      --output /…/stacks/java_vh_2024_2026_50m_vNEW.tif
"""
import argparse
import rasterio


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base multi-year stack (band names preserved)")
    p.add_argument("--append", required=True, help="per-year stack to append")
    p.add_argument("--append-year", type=int, required=True, help="year label for appended bands")
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    with rasterio.open(a.base) as sb, rasterio.open(a.append) as sa:
        if (sb.width, sb.height) != (sa.width, sa.height):
            raise SystemExit(f"grid mismatch: base {sb.width}x{sb.height} vs append {sa.width}x{sa.height}")
        n = sb.count + sa.count
        names = list(sb.descriptions[:sb.count]) + \
            [f"{a.append_year}_Period_{i}" for i in range(1, sa.count + 1)]
        prof = sb.profile.copy()
        prof.update(count=n, compress="LZW", tiled=True, bigtiff="YES")
        print(f"writing {n} bands ({sb.count} base + {sa.count} {a.append_year}) -> {a.output}", flush=True)
        with rasterio.open(a.output, "w", **prof) as dst:
            bi = 1
            for src in (sb, sa):
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), bi)
                    dst.set_band_description(bi, names[bi - 1])
                    if bi % 10 == 0 or bi > sb.count:
                        print(f"  band {bi}/{n}  {names[bi - 1]}", flush=True)
                    bi += 1
    print(f"DONE: {n} bands, last = {names[-1]}", flush=True)


if __name__ == "__main__":
    main()
