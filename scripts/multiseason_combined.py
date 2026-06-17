"""Multi-season training test on the SAME 14,187 series_combined set (unifies datasets).

Does adding dry-season labels improve held-out DRY-season prediction? Held-out test =
stratified 25% of dry points (by fase). MiniROCKET fit once on the common training pool
(wet + dry-train); only the LightGBM training composition varies. Fused features.
Run: python scripts/multiseason_combined.py   (fusets env)
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from train_v3_mogpr_ensemble import _window
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

pre = "output/phase_model/series_combined"; W = 8
z = np.load(pre + ".npz"); m = pd.read_csv(pre + "_meta.csv")
fase = m["fase"].astype(int).values
tg = z["t_grid"]; lab = z["label_ord"]

# season via coordinate join to the points file (meta has no date)
p = pd.read_csv("data/aois/points_combined_java.csv")
for d in (m, p):
    d["k"] = d["bujur"].round(6).astype(str) + "_" + d["lintang"].round(6).astype(str)
p["season"] = np.where(pd.to_datetime(p["tanggal"], errors="coerce").dt.month == 10, "dry", "wet")
season = m["k"].map(p.groupby("k")["season"].agg(lambda s: s.mode().iloc[0])).values

def feats(arr):
    w = _window(np.nan_to_num(arr), tg, lab, W)
    return np.stack([w, np.gradient(w, axis=1)], 1).astype("float32")

X = feats(z["ndvi"])  # fused MOGPR curve
rng = np.random.RandomState(0)
dry = np.where(season == "dry")[0]; wet = np.where(season == "wet")[0]
test = []
for f in range(1, 7):
    idx = dry[fase[dry] == f]; rng.shuffle(idx); test += idx[:int(round(0.25 * len(idx)))].tolist()
test = np.array(sorted(test))
dry_train = np.setdiff1d(dry, test)
trainA = wet                                   # wet-only
trainB = np.concatenate([wet, dry_train])      # combined
mr = MiniRocketMultivariate(num_kernels=2000, random_state=0).fit(X[trainB])
Xt = mr.transform(X).values
g = lambda a: np.isin(a, [4, 5]).astype(int)
S3 = {1: 1, 6: 1, 2: 2, 3: 2, 4: 3, 5: 3}
s3 = lambda a: np.vectorize(S3.get)(a)
yt = fase[test]
print(f"=== multi-season on series_combined (n={len(fase)}; wet {len(wet)}, dry {len(dry)}; dry-test {len(test)}) ===")
for tr, name in [(trainA, "A wet-only "), (trainB, "B combined ")]:
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", verbose=-1).fit(Xt[tr], fase[tr])
    pr = clf.predict(Xt[test])
    print(f"{name}: dry-test 6cls {accuracy_score(yt, pr)*100:4.1f}  "
          f"3cls {accuracy_score(s3(yt), s3(pr))*100:4.1f}  "
          f"genF1 {f1_score(g(yt), g(pr), zero_division=0)*100:4.1f}  (n_train={len(tr)})")
