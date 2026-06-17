"""Significance of the fusion gain under LORO: McNemar (paired, per-instance),
Wilcoxon signed-rank (per-region), bootstrap 95% CIs. Run: python scripts/loro_significance.py (fusets).
Mirrors loro_eval exactly (MiniROCKET 2000 kernels on full X; per-region balanced LightGBM)."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from train_v3_mogpr_ensemble import _window
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier
from scipy.stats import chi2, binom, wilcoxon
from sklearn.metrics import f1_score, accuracy_score

pre = "output/phase_model/series_combined"; W = 8; S3 = {1: 1, 6: 1, 2: 2, 3: 2, 4: 3, 5: 3}
z = np.load(pre + ".npz"); m = pd.read_csv(pre + "_meta.csv")
fase = m["fase"].astype(int).values; region = m["region"].astype(str).values
tg = z["t_grid"]; lab = z["label_ord"]
def feats(a):
    w = _window(np.nan_to_num(a), tg, lab, W); return np.stack([w, np.gradient(w, axis=1)], 1).astype("float32")
def loro_preds(X):
    mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X); Xt = mr.transform(X).values
    pred = np.zeros(len(fase), int)
    for r in sorted(set(region)):
        te = region == r
        pred[te] = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  class_weight="balanced", verbose=-1).fit(Xt[~te], fase[~te]).predict(Xt[te])
    return pred
pf = loro_preds(feats(z["ndvi"]))  # fused
pv = loro_preds(feats(z["vh"]))    # VH-only
s3 = np.vectorize(S3.get); g = lambda a: np.isin(a, [4, 5]).astype(int)

def mcnemar(corr_a, corr_b):  # a=VH-only, b=fused; c = fused-only-correct
    b = int(np.sum(corr_a & ~corr_b)); c = int(np.sum(~corr_a & corr_b)); n = b + c
    if n == 0: return b, c, 1.0
    p = chi2.sf((abs(b - c) - 1) ** 2 / n, 1) if n >= 25 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))
    return b, c, p

print(f"=== fusion significance, LORO n={len(fase)} (sanity: 3cls acc fused {accuracy_score(s3(fase),s3(pf))*100:.1f}%, VH {accuracy_score(s3(fase),s3(pv))*100:.1f}%) ===")
ta3 = s3(fase)
b, c, p = mcnemar(s3(pv) == ta3, s3(pf) == ta3)
print(f"McNemar 3-class:   discordant VH-only-correct={b}, fused-correct={c}, p={p:.2e}")
gt = np.isin(fase, [4, 5])
b, c, p = mcnemar(np.isin(pv, [4, 5]) == gt, np.isin(pf, [4, 5]) == gt)
print(f"McNemar gen-binary: VH-only-correct={b}, fused-correct={c}, p={p:.2e}")
regs = sorted(set(region))
fv = [100 * f1_score(g(fase[region == r]), g(pv[region == r]), zero_division=0) for r in regs]
ff = [100 * f1_score(g(fase[region == r]), g(pf[region == r]), zero_division=0) for r in regs]
w, pw = wilcoxon(ff, fv)
print(f"per-region gen F1: VH {np.mean(fv):.1f}+/-{np.std(fv):.1f} -> fused {np.mean(ff):.1f}+/-{np.std(ff):.1f}; Wilcoxon p={pw:.4f} (n={len(regs)})")
rng = np.random.RandomState(0); N = len(fase); idx = np.arange(N); d3 = []; dg = []
for _ in range(2000):
    s = rng.choice(idx, N, replace=True)
    d3.append(accuracy_score(ta3[s], s3(pf)[s]) - accuracy_score(ta3[s], s3(pv)[s]))
    dg.append(f1_score(g(fase[s]), g(pf[s]), zero_division=0) - f1_score(g(fase[s]), g(pv[s]), zero_division=0))
d3 = np.array(d3) * 100; dg = np.array(dg) * 100
print(f"bootstrap 3-class acc gain: {d3.mean():.1f} [{np.percentile(d3,2.5):.1f}, {np.percentile(d3,97.5):.1f}]")
print(f"bootstrap gen-binary F1 gain: {dg.mean():.1f} [{np.percentile(dg,2.5):.1f}, {np.percentile(dg,97.5):.1f}]")
np.savez("output/phase_model/loro_preds_mrlgbm.npz", pred=pf, fase=fase, region=region)
print("saved fused per-instance preds -> output/phase_model/loro_preds_mrlgbm.npz")
