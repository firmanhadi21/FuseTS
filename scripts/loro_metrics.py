"""Leave-one-region-out per-class metrics (multi-season model) for the paper."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from train_v3_mogpr_ensemble import _window
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, confusion_matrix
W=8; S3={1:1,6:1,2:2,3:2,4:3,5:3}
zw=np.load("output/phase_model/series_0104.npz"); mw=pd.read_csv("output/phase_model/series_0104_meta.csv")
zd=np.load("output/phase_model/series_dry6fase.npz"); md=pd.read_csv("output/phase_model/series_dry6fase_meta.csv")
L=min(zw["ndvi"].shape[1],zd["ndvi"].shape[1])
ndvi=np.concatenate([zw["ndvi"][:,:L],zd["ndvi"][:,:L]]); lab=np.concatenate([zw["label_ord"],zd["label_ord"]])
tg=zw["t_grid"][:L]; fase=np.concatenate([mw["fase"].values,md["fase"].values]).astype(int)
region=np.concatenate([mw["region"].values,md["region"].values])
wn=_window(np.nan_to_num(ndvi),tg,lab,W); X=np.stack([wn,np.gradient(wn,axis=1)],1).astype("float32")
mr=MiniRocketMultivariate(num_kernels=2000,random_state=0).fit(X); Xt=mr.transform(X).values
regions=["Karawang","Indramayu","Cirebon","Brebes_Tegal","Pekalongan","Semarang_Demak","Klambu"]
obs6=[]; pred6=[]; perfold=[]
for r in regions:
    te=region==r; tr=~te
    if te.sum()==0: continue
    clf=LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,class_weight="balanced",verbose=-1).fit(Xt[tr],fase[tr])
    p=clf.predict(Xt[te]); o=fase[te]
    obs6+=o.tolist(); pred6+=p.tolist()
    g=lambda a:np.isin(a,[4,5]).astype(int)
    perfold.append((r,100*f1_score(g(o),g(p),zero_division=0),int(te.sum())))
obs6=np.array(obs6); pred6=np.array(pred6); obs3=np.vectorize(S3.get)(obs6); pred3=np.vectorize(S3.get)(pred6)
print("=== pooled leave-one-region-out (multi-season) ===")
print("6-class per-class P/R/F1:")
P,R,F,Sup=precision_recall_fscore_support(obs6,pred6,labels=[1,2,3,4,5,6],zero_division=0)
nm={1:"flood",2:"e-veg",3:"l-veg",4:"e-gen",5:"l-gen",6:"post"}
for i,k in enumerate([1,2,3,4,5,6]): print(f"  {nm[k]:>6}: P{P[i]*100:4.0f} R{R[i]*100:4.0f} F1{F[i]*100:4.0f} n{Sup[i]}")
print(f"  6-class overall acc {accuracy_score(obs6,pred6)*100:.1f}%")
print("3-class per-class P/R/F1:")
P,R,F,Sup=precision_recall_fscore_support(obs3,pred3,labels=[1,2,3],zero_division=0)
for i,k in enumerate(["bare","veg","gen"]): print(f"  {k:>4}: P{P[i]*100:4.0f} R{R[i]*100:4.0f} F1{F[i]*100:4.0f} n{Sup[i]}")
print(f"  3-class overall acc {accuracy_score(obs3,pred3)*100:.1f}%")
ff=[f for _,f,_ in perfold]
print(f"\nper-region generative F1: mean {np.mean(ff):.1f} +/- {np.std(ff):.1f}  range {min(ff):.0f}-{max(ff):.0f}")
for r,f,n in perfold: print(f"  {r:>15}: {f:4.0f}  (n={n})")

# --- VH-only vs fused ablation under the SAME LORO protocol ---
def loro_genF1(feat):
    Xa=np.stack([feat,np.gradient(feat,axis=1)],1).astype("float32")
    mr2=MiniRocketMultivariate(num_kernels=2000,random_state=0).fit(Xa); Xa=mr2.transform(Xa).values
    g=lambda a:np.isin(a,[4,5]).astype(int); per=[]
    for r in regions:
        te=region==r; tr=~te
        if te.sum()==0: continue
        clf=LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,class_weight="balanced",verbose=-1).fit(Xa[tr],fase[tr])
        per.append(100*f1_score(g(fase[te]),g(clf.predict(Xa[te])),zero_division=0))
    return np.mean(per),np.std(per)
vh_w=_window(np.nan_to_num(np.concatenate([zw["vh"][:,:L],zd["vh"][:,:L]])),tg,lab,W)
fu_w=wn
print("\n=== ablation (per-region generative F1, mean+/-sd) ===")
m,s=loro_genF1(vh_w); print(f"  VH-only : {m:.1f} +/- {s:.1f}")
m,s=loro_genF1(fu_w); print(f"  fused   : {m:.1f} +/- {s:.1f}")
