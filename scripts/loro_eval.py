"""Leave-one-region-out evaluation for any series (per-class + per-region + VH-vs-fused)."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from train_v3_mogpr_ensemble import _window
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
pre=sys.argv[1]; W=8; S3={1:1,6:1,2:2,3:2,4:3,5:3}
z=np.load(pre+".npz"); m=pd.read_csv(pre+"_meta.csv")
fase=m["fase"].astype(int).values; region=m["region"].astype(str).values
tg=z["t_grid"]; lab=z["label_ord"]
def feats(arr): 
    w=_window(np.nan_to_num(arr),tg,lab,W); return np.stack([w,np.gradient(w,axis=1)],1).astype("float32")
def loro(X):
    mr=MiniRocketMultivariate(num_kernels=2000,random_state=0).fit(X); Xt=mr.transform(X).values
    regs=sorted(set(region)); o=[]; p=[]; per=[]; g=lambda a:np.isin(a,[4,5]).astype(int)
    for r in regs:
        te=region==r; tr=~te
        clf=LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,class_weight="balanced",verbose=-1).fit(Xt[tr],fase[tr])
        pr=clf.predict(Xt[te]); o+=fase[te].tolist(); p+=pr.tolist()
        per.append(100*f1_score(g(fase[te]),g(pr),zero_division=0))
    return np.array(o),np.array(p),per
print(f"=== LORO {pre}  n={len(fase)}  regions={len(set(region))} ===")
o,p,per=loro(feats(z["ndvi"]))   # fused
o3=np.vectorize(S3.get)(o); p3=np.vectorize(S3.get)(p)
P,R,F,Su=precision_recall_fscore_support(o,p,labels=range(1,7),zero_division=0)
nm=["flood","e-veg","l-veg","e-gen","l-gen","post"]
print("6-class P/R/F1:"); [print(f"  {nm[i]:>6} P{P[i]*100:4.0f} R{R[i]*100:4.0f} F1{F[i]*100:4.0f} n{Su[i]}") for i in range(6)]
print(f"  6-class acc {accuracy_score(o,p)*100:.1f}%")
P,R,F,Su=precision_recall_fscore_support(o3,p3,labels=[1,2,3],zero_division=0)
[print(f"  {n:>4} P{P[i]*100:4.0f} R{R[i]*100:4.0f} F1{F[i]*100:4.0f} n{Su[i]}") for i,n in enumerate(["bare","veg","gen"])]
print(f"  3-class acc {accuracy_score(o3,p3)*100:.1f}%")
print(f"  generative per-region F1: {np.mean(per):.1f} +/- {np.std(per):.1f}  range {min(per):.0f}-{max(per):.0f}")
ov,pv,perv=loro(feats(z["vh"]))
print(f"\nABLATION generative per-region F1: VH-only {np.mean(perv):.1f}+/-{np.std(perv):.1f}  ->  fused {np.mean(per):.1f}+/-{np.std(per):.1f}")
