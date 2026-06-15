#!/usr/bin/env python
"""Generate the 6 new figures for the S1+S2 fusion companion paper.

Outputs -> rice-growth-stage-mapping/paper_latex/companion_figures/
1 fig_fusion_concept   2 fig_workflow   3a fig_phase_ablation  3b fig_confusion_seasonal
4 fig_ip_minrun        5 fig_perperiod_generative              6 fig_perregion_f1
Each figure is isolated (try/except) so one failure does not block the others.
"""
import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

OUT = Path("/home/unika_sianturi/work/rice-growth-stage-mapping/paper_latex/companion_figures")
OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(m, flush=True)


def fig1_fusion_concept():
    d = Path("output/production/java_2024/tiles/r007_c054")
    cube = xr.open_dataset(d / "cube.nc"); z = np.load(d / "fused.npz")
    fused, ys, xs, og = z["fused"], z["ys"], z["xs"], z["out_grid"]
    t = [pd.Timestamp(v) for v in cube["t"].values]
    t_ord = np.array([x.toordinal() for x in t], float)
    nd = cube["S2ndvi"].values; vh = cube["VH"].values
    ok = np.isfinite(fused).all(1)
    pick = None
    for i in np.where(ok)[0]:
        s = nd[:, ys[i], xs[i]]; v = np.isfinite(s)
        if 0.3 < v.mean() < 0.75 and np.nanmax(s) - np.nanmin(s) > 0.3:
            pick = i; break
    if pick is None:
        pick = np.where(ok)[0][0]
    y, x = ys[pick], xs[pick]
    nds = nd[:, y, x]; vhs = vh[:, y, x]
    og_dt = [pd.Timestamp.fromordinal(int(o)) for o in og]
    nv = np.isfinite(nds); mv = np.isfinite(vhs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([t[k] for k in np.where(nv)[0]], nds[nv], "o", color="#2c7fb8", ms=6, label="S2 NDVI obs (with cloud gaps)")
    ax.plot(og_dt, fused[pick], "-", color="#1a9850", lw=2.2, label="MOGPR fused curve")
    ax.set_ylabel("NDVI"); ax.set_xlabel("date (2024)")
    ax2 = ax.twinx()
    ax2.plot([t[k] for k in np.where(mv)[0]], vhs[mv], "s", color="#d7301f", ms=4, alpha=.7, label="S1 VH obs (dB)")
    ax2.set_ylabel("VH (dB)", color="#d7301f"); ax2.tick_params(axis="y", colors="#d7301f")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper right", fontsize=8)
    ax.set_title("MOGPR fuses S1 VH + S2 NDVI into one gap-filled curve (sample paddy pixel)")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "fig_fusion_concept.png", dpi=160); plt.close(fig)
    log("ok fig1_fusion_concept")


def fig2_workflow():
    fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3)
    boxes = [(0.3, "S1 VH\n(SNAP 50 m)", "#deebf7"), (0.3, "S2 NDVI\n(MPC)", "#fee8c8")]
    # two inputs feed fusion
    def box(x, y, w, h, txt, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=c, ec="#444"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    box(0.2, 1.9, 1.8, 0.8, "S1 VH\n(SNAP 50 m)", "#deebf7")
    box(0.2, 0.3, 1.8, 0.8, "S2 NDVI\n(MPC)", "#fee8c8")
    chain = [(2.7, "MOGPR\nfusion", "#e5f5e0"), (5.0, "MiniROCKET\n+ LightGBM", "#e5f5e0"),
             (7.3, "6/3 phase\nper period", "#c7e9c0"), (9.0, "count generative\nepisodes (min_run 3)", "#c7e9c0"),
             (11.2, "IP &\nproduction", "#a1d99b")]
    for x, txt, c in chain:
        box(x, 1.1, 1.6, 0.8, txt, c)
    arrows = [(2.0, 2.3, 2.7, 1.7), (2.0, 0.7, 2.7, 1.3), (4.3, 1.5, 5.0, 1.5),
              (6.6, 1.5, 7.3, 1.5), (8.9, 1.5, 9.0, 1.5), (10.6, 1.5, 11.2, 1.5)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="#444", lw=1.4))
    ax.set_title("Workflow: fuse -> classify -> count -> map", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_workflow.png", dpi=160); plt.close(fig)
    log("ok fig2_workflow")


def fig3a_ablation():
    methods = ["Rule\n[POS,EOS]", "Scalar\n+ RF", "VH-only", "MiniROCKET\n(fused)", "S1-CNN\n(in-samp)", "VH+MOGPR", "MOGPR\noptical(held)"]
    f1 = [22, 59, 52, 65.5, 64, 67, 68.2]
    cols = ["#bdbdbd", "#bdbdbd", "#fdae61", "#74add1", "#bdbdbd", "#1a9850", "#1a9850"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b = ax.bar(methods, f1, color=cols, ec="#333")
    for r, v in zip(b, f1): ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:g}", ha="center", fontsize=8)
    ax.set_ylabel("Generative-phase F1 (%)"); ax.set_ylim(0, 80)
    ax.set_title("Optical fusion lifts generative-phase F1 (leave-one-region-out)")
    ax.axhline(52, ls="--", color="#fdae61", lw=1, alpha=.7)
    fig.tight_layout(); fig.savefig(OUT / "fig_phase_ablation.png", dpi=160); plt.close(fig)
    log("ok fig3a_ablation")


def _load_pool(W=8):
    from train_v3_mogpr_ensemble import _window
    zw = np.load("output/phase_model/series_0104.npz"); mw = pd.read_csv("output/phase_model/series_0104_meta.csv")
    zd = np.load("output/phase_model/series_dry6fase.npz"); md = pd.read_csv("output/phase_model/series_dry6fase_meta.csv")
    L = min(zw["ndvi"].shape[1], zd["ndvi"].shape[1])
    ndvi = np.concatenate([zw["ndvi"][:, :L], zd["ndvi"][:, :L]])
    lab = np.concatenate([zw["label_ord"], zd["label_ord"]]); tg = zw["t_grid"][:L]
    fase = np.concatenate([mw["fase"].values, md["fase"].values]).astype(int)
    region = np.concatenate([mw["region"].values, md["region"].values])
    season = np.array(["wet"] * len(mw) + ["dry"] * len(md))
    wn = _window(np.nan_to_num(ndvi), tg, lab, W)
    X = np.stack([wn, np.gradient(wn, axis=1)], 1).astype("float32")
    return X, fase, region, season


def _mr_clf():
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    from lightgbm import LGBMClassifier
    return MiniRocketMultivariate(num_kernels=2000, random_state=0), LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced", verbose=-1)


def fig3b_confusion():
    from sklearn.metrics import confusion_matrix
    X, fase, region, season = _load_pool()
    rng = np.random.default_rng(0); N = len(fase); test = np.zeros(N, bool)
    for s in ("wet", "dry"):
        for f in range(1, 7):
            idx = np.where((season == s) & (fase == f))[0]
            if len(idx): test[rng.choice(idx, max(1, int(.25 * len(idx))), replace=False)] = True
    train = ~test
    mr, _ = _mr_clf(); mr.fit(X[train]); Xt = mr.transform(X).values
    g = lambda a: np.isin(a, [4, 5]).astype(int)  # 0 non-gen, 1 gen (binary slice of 3-class)
    dry = (season[test] == "dry")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, mask, name in [(axes[0], train & (season == "wet"), "Wet-only model"),
                           (axes[1], train, "Multi-season model")]:
        _, clf = _mr_clf(); clf.fit(Xt[mask], fase[mask]); pred = clf.predict(Xt[test])
        cm = confusion_matrix(g(fase[test][dry]), g(pred[dry]))
        cmn = cm / cm.sum(1, keepdims=True) * 100
        im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i,j]:.0f}%", ha="center", va="center",
                        color="white" if cmn[i, j] > 50 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-gen", "gen"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["non-gen", "gen"])
        ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.set_title(f"{name}\n(DRY-season test points)")
    fig.suptitle("Generative detection on held-out DRY-season points", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_seasonal.png", dpi=160); plt.close(fig)
    log("ok fig3b_confusion")


def fig4_ip_minrun():
    from produce_annual import _count_episodes
    from collections import Counter
    tiles = sorted(glob.glob("output/production/java_2024/tiles/*/phase3.tif"))
    res = {m: Counter() for m in (2, 3, 4)}; ntot = 0
    for t in tiles:
        a = rioxarray.open_rasterio(t).values; L = a.shape[0]
        flat = a.reshape(L, -1).T; fp = flat[(flat > 0).any(1)]; isg = (fp == 3)
        ntot += len(fp)
        for m in (2, 3, 4):
            for k, c in Counter(np.clip(_count_episodes(isg, m), 0, 8).tolist()).items():
                res[m][int(k)] += c
    ks = list(range(0, 7)); width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate((2, 3, 4)):
        vals = [100 * res[m].get(k, 0) / ntot for k in ks]
        ax.bar([k + (i - 1) * width for k in ks], vals, width, label=f"min_run={m}")
    ax.set_xlabel("cropping intensity (harvests/yr per pixel)"); ax.set_ylabel("% of paddy pixels")
    ax.set_xticks(ks); ax.legend(); ax.axvspan(3.5, 6.5, color="#d7301f", alpha=.08)
    ax.text(5, ax.get_ylim()[1] * .8, "non-physical\n(IP>=4)", color="#d7301f", ha="center", fontsize=8)
    ax.set_title("min_run=3 removes non-physical IP>=4 (multi-season model, 2024)")
    fig.tight_layout(); fig.savefig(OUT / "fig_ip_minrun.png", dpi=160); plt.close(fig)
    log("ok fig4_ip_minrun")


def fig5_perperiod():
    def frac(dirn):
        out = []
        for p in range(1, 32):
            f = f"output/production/{dirn}/java_phase3_p{p:02d}.tif"
            if not Path(f).exists():
                out.append(np.nan); continue
            v = rioxarray.open_rasterio(f).isel(band=0).values
            pad = v > 0; out.append(100 * (v == 3).sum() / max(1, pad.sum()))
        return np.array(out)
    wet = frac("java_2024_phases"); ms = frac("java_2024_phases_ms")
    per = np.arange(1, 32)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(per, wet, "-o", ms=3, color="#fdae61", label="wet-only model")
    ax.plot(per, ms, "-o", ms=3, color="#1a9850", label="multi-season model")
    ax.axvspan(16, 27, color="#3690c0", alpha=.08); ax.text(21.5, ax.get_ylim()[1] * .92, "dry season", color="#3690c0", ha="center", fontsize=8)
    ax.set_xlabel("12-day period (2024)"); ax.set_ylabel("% paddy in generative phase"); ax.legend()
    ax.set_title("Per-period generative fraction: multi-season recovers the dry-season wave")
    fig.tight_layout(); fig.savefig(OUT / "fig_perperiod_generative.png", dpi=160); plt.close(fig)
    log("ok fig5_perperiod")


def fig6_perregion():
    from sklearn.metrics import f1_score
    X, fase, region, season = _load_pool()
    regions = ["Karawang", "Indramayu", "Cirebon", "Brebes_Tegal", "Pekalongan", "Semarang_Demak", "Klambu"]
    mr, _ = _mr_clf(); mr.fit(X); Xt = mr.transform(X).values
    g = lambda a: np.isin(a, [4, 5]).astype(int)
    f1s = []
    for r in regions:
        te = region == r; tr = ~te
        _, clf = _mr_clf(); clf.fit(Xt[tr], fase[tr]); pred = clf.predict(Xt[te])
        f1s.append(100 * f1_score(g(fase[te]), g(pred)) if te.sum() else np.nan)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b = ax.bar(regions, f1s, color="#1a9850", ec="#333")
    for rr, v in zip(b, f1s):
        if np.isfinite(v): ax.text(rr.get_x() + rr.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(np.nanmean(f1s), ls="--", color="#666", label=f"mean {np.nanmean(f1s):.0f}")
    ax.set_ylabel("Generative F1 (%)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("Leave-one-region-out generative F1 by Java region (spatial holdout)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "fig_perregion_f1.png", dpi=160); plt.close(fig)
    log("ok fig6_perregion")


if __name__ == "__main__":
    for fn in (fig1_fusion_concept, fig2_workflow, fig3a_ablation, fig3b_confusion,
               fig4_ip_minrun, fig5_perperiod, fig6_perregion):
        try:
            fn()
        except Exception as e:
            log(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    log("done -> " + str(OUT))
