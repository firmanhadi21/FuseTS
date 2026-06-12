#!/usr/bin/env python
"""Annual production loop: classify phase at EVERY 12-day period over the year and count
each generative (phase 4-5) episode once per pixel = harvest events. Avoids the snapshot
double-counting; a double-cropped pixel contributes 2 harvests.

   annual harvest area = Σ_pixels (n_generative_episodes × pixel_ha)
   annual production    = annual harvest area × yield (t/ha)

n_generative_episodes per pixel = number of maximal runs of 'generative' in its per-period
phase sequence (a run ≥ --min-run periods, to suppress single-period flicker). This is the
phase-based cropping intensity (= luas panen / physical area).

Outputs: n_harvests map (= cropping intensity), per-period generative-area time series,
annual area + production. Same MOGPR-optical classifier as produce_estimate.py.

CAVEATS: model ~68% F1; flat yield; episode-counting is sensitive to --min-run; no
production ground-truth here (validate vs BPS / calc_luas_panen separately).

Usage
-----
  python scripts/produce_annual.py --cube output/brebes_snap_2024/cube.nc --crs EPSG:32749 \
     --mask .../paddy_mask.tif --series output/phase_model/series_0104 \
     --yield 5.8 --min-run 1 --workers 16 --out output/production/brebes_annual
"""
import argparse
import json
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from scale_runner import apply_mask  # noqa: E402
from train_v3_mogpr_ensemble import _window, TO3  # noqa: E402
from fusets.mogpr import mogpr_1D  # noqa: E402
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier

PIX_HA = 0.25


def _fuse_chunk(args, t_ord, out_grid):
    """Return the full fused NDVI curve (len L) per pixel; NaN row if fusion fails."""
    ndvi_c, vh_c = args
    L = len(out_grid)
    fused_c = np.full((len(ndvi_c), L), np.nan, "float32")
    for i in range(len(ndvi_c)):
        nd, vh = ndvi_c[i], vh_c[i]
        nv = np.isfinite(nd)
        if nv.sum() < 2:
            continue
        data, tin = [nd[nv]], [t_ord[nv]]
        m = np.isfinite(vh)
        if m.sum() >= 4:
            data.append(vh[m]); tin.append(t_ord[m])
        if len(data) < 2:
            continue
        try:
            om, *_ = mogpr_1D(data, tin, 0, out_grid, 1)
            fused_c[i] = np.ravel(om[0]).astype("float32")
        except Exception:
            pass
    return fused_c


def _count_episodes(is_gen_seq, min_run):
    """# of maximal runs of True with length >= min_run, per row (N,L) -> (N,)."""
    N, L = is_gen_seq.shape
    out = np.zeros(N, int)
    for i in range(N):
        run = 0
        for v in is_gen_seq[i]:
            if v:
                run += 1
            else:
                if run >= min_run:
                    out[i] += 1
                run = 0
        if run >= min_run:
            out[i] += 1
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True); p.add_argument("--crs", default="EPSG:32749")
    p.add_argument("--mask", default=None); p.add_argument("--series", required=True)
    p.add_argument("--yield", dest="yld", type=float, default=5.8)
    p.add_argument("--min-run", type=int, default=1)
    p.add_argument("--window", type=int, default=8); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W = a.window

    # train OPT classifier on 0104
    z = np.load(a.series + ".npz"); meta = pd.read_csv(a.series + "_meta.csv")
    tn = np.nan_to_num(z["ndvi"]); tg = z["t_grid"]; lo = z["label_ord"]
    wn = _window(tn, tg, lo, W); dn = np.gradient(wn, axis=1).astype("float32")
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(np.stack([wn, dn], 1).astype("float32"))
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(
        mr.transform(np.stack([wn, dn], 1).astype("float32")).values, meta["fase"].map(TO3).values)
    print(f"trained OPT classifier on {len(meta)} points")

    cube = xr.open_dataset(a.cube).rio.write_crs(a.crs)
    if a.mask:
        cube = apply_mask(cube, a.mask, a.crs)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    t_ord = np.array([pd.Timestamp(t).toordinal() for t in cube["t"].values], float)
    out_grid = np.arange(int(t_ord.min()), int(t_ord.max()) + 1, 12, float)
    L = len(out_grid)
    NDVI = cube["S2ndvi"].values; VH = cube["VH"].values
    ys, xs = np.where(np.isfinite(NDVI).sum(0) >= 2)
    ndvi_px = NDVI[:, ys, xs].T.astype("float32"); vh_px = VH[:, ys, xs].T.astype("float32")
    print(f"AOI {ny}x{nx} | {len(ys)} paddy px to fuse")

    chunks = np.array_split(np.arange(len(ys)), max(1, a.workers * 4))
    fn = partial(_fuse_chunk, t_ord=t_ord, out_grid=out_grid)
    fused = np.full((len(ys), L), np.nan, "float32")
    with Pool(a.workers) as pool:
        for c, f in zip(chunks, pool.map(fn, [(ndvi_px[c], vh_px[c]) for c in chunks])):
            fused[c] = f
    ok = np.isfinite(fused).all(1)
    print(f"fused {int(ok.sum())}/{len(ys)} px; classifying all {L} periods...")

    # classify phase at every period (window around each grid point)
    Fok = np.nan_to_num(fused[ok])
    dF = np.gradient(Fok, axis=1).astype("float32")
    is_gen = np.zeros((Fok.shape[0], L), bool)
    per_period_gen = np.zeros(L, int)
    for pidx in range(L):
        lo_, hi_ = pidx - W, pidx + W + 1
        segN = np.stack([np.concatenate([np.full(max(0, -lo_), Fok[i, 0]),
                         Fok[i, max(0, lo_):min(L, hi_)], np.full(max(0, hi_ - L), Fok[i, -1])])[:2 * W + 1]
                         for i in range(Fok.shape[0])])
        segD = np.gradient(segN, axis=1).astype("float32")
        pr = clf.predict(mr.transform(np.stack([segN, segD], 1).astype("float32")).values)
        g = (pr == "generative")
        is_gen[:, pidx] = g
        per_period_gen[pidx] = int(g.sum())

    episodes = _count_episodes(is_gen, a.min_run)            # n harvests per fused pixel
    n_harv = np.zeros(len(ys), int); n_harv[ok] = episodes

    # maps
    hmap = np.zeros((ny, nx), "uint8"); hmap[ys, xs] = np.clip(n_harv, 0, 255)
    xr.DataArray(hmap, coords={"y": cube["y"], "x": cube["x"]}, dims=("y", "x")) \
        .rio.write_crs(a.crs).rio.to_raster(out / "n_harvests.tif", compress="lzw")

    area_ha = float(episodes.sum()) * PIX_HA
    production = area_ha * a.yld
    paddy_ha = int(ok.sum()) * PIX_HA
    summary = {
        "aoi_cube": a.cube, "min_run": a.min_run, "yield_t_ha": a.yld,
        "paddy_px_fused": int(ok.sum()), "physical_paddy_ha": round(paddy_ha, 1),
        "n_harvests_distribution": {int(k): int(v) for k, v in
                                    zip(*np.unique(episodes, return_counts=True))},
        "mean_cropping_intensity": round(float(episodes.mean()), 3),
        "annual_harvest_area_ha": round(area_ha, 1),
        "annual_production_t": round(production, 1),
        "per_period_generative_px": per_period_gen.tolist(),
        "period_dates": [str(pd.Timestamp.fromordinal(int(o)).date()) for o in out_grid],
        "caveats": ["model ~68% F1", "flat yield", f"episode count sensitive to min_run={a.min_run}",
                    "no production ground-truth here"],
    }
    (out / "annual_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        im = ax[0].imshow(np.where(hmap > 0, hmap, np.nan), cmap="YlOrRd", vmin=0, vmax=3)
        ax[0].set_title(f"Harvests/yr (cropping intensity) mean={episodes.mean():.2f}"); ax[0].axis("off")
        fig.colorbar(im, ax=ax[0], fraction=.046)
        dates = [pd.Timestamp.fromordinal(int(o)) for o in out_grid]
        ax[1].plot(dates, np.array(per_period_gen) * PIX_HA)
        ax[1].set_title("Generative (phase 4-5) area over the year"); ax[1].set_ylabel("ha")
        ax[1].tick_params(axis="x", rotation=30)
        fig.suptitle(f"Annual: {area_ha:,.0f} ha harvested → {production:,.0f} t @ {a.yld} t/ha "
                     f"(physical paddy {paddy_ha:,.0f} ha)")
        fig.tight_layout(); fig.savefig(out / "annual.png", dpi=120); plt.close(fig)
    except Exception as e:
        print("plot failed:", e)

    print(f"\n=== ANNUAL PRODUCTION ===")
    print(f"physical paddy {paddy_ha:,.0f} ha | mean cropping intensity {episodes.mean():.2f} harvests/yr")
    print(f"annual harvest area {area_ha:,.0f} ha → production {production:,.0f} t @ {a.yld} t/ha")
    print(f"harvests/yr dist: {summary['n_harvests_distribution']}")
    print(f"wrote -> {out}/n_harvests.tif, annual.png, annual_summary.json")
    return summary


if __name__ == "__main__":
    main()
