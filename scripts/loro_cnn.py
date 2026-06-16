"""1D-CNN vs MiniROCKET+LightGBM under identical leave-one-region-out (fused features).
Run with geo_ml_env python (TF 2.18). Fair-comparison check for the companion paper."""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
pre=sys.argv[1] if len(sys.argv)>1 else "output/phase_model/series_combined"; W=8; S3={1:1,6:1,2:2,3:2,4:3,5:3}
z=np.load(pre+".npz"); m=pd.read_csv(pre+"_meta.csv")
fase=m["fase"].astype(int).values; region=m["region"].astype(str).values
tg=z["t_grid"]; lab=z["label_ord"]
def _window(series,t_grid,label_ord,Wn):
    N,L=series.shape; idx=np.abs(t_grid[None,:]-label_ord[:,None]).argmin(1); out=np.empty((N,2*Wn+1),series.dtype)
    for i in range(N):
        lo,hi=idx[i]-Wn,idx[i]+Wn+1; s=series[i]
        out[i]=np.concatenate([np.full(max(0,-lo),s[0]),s[max(0,lo):min(L,hi)],np.full(max(0,hi-L),s[-1])])[:2*Wn+1]
    return out
wn=_window(np.nan_to_num(z["ndvi"]),tg,lab,W); dn=np.gradient(wn,axis=1)
X=np.stack([wn,dn],-1).astype("float32")    # (N, 2W+1, 2)
y=fase-1                                      # 0..5
def make():
    inp=tf.keras.Input((X.shape[1],2)); x=inp
    for f in (32,32): x=tf.keras.layers.Conv1D(f,3,activation="relu",padding="same")(x)
    x=tf.keras.layers.GlobalMaxPool1D()(x); x=tf.keras.layers.Dense(64,activation="relu")(x)
    x=tf.keras.layers.Dropout(0.3)(x); out=tf.keras.layers.Dense(6,activation="softmax")(x)
    mdl=tf.keras.Model(inp,out); mdl.compile("adam","sparse_categorical_crossentropy"); return mdl
regs=sorted(set(region)); O=[]; P=[]; per=[]; g=lambda a:np.isin(a,[4,5]).astype(int)
for r in regs:
    te=region==r; tr=~te
    mu=X[tr].mean((0,1),keepdims=True); sd=X[tr].std((0,1),keepdims=True)+1e-6
    Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd
    cw=compute_class_weight("balanced",classes=np.arange(6),y=y[tr]); cw={i:w for i,w in enumerate(cw)}
    tf.keras.utils.set_random_seed(0); mdl=make()
    mdl.fit(Xtr,y[tr],epochs=30,batch_size=64,verbose=0,class_weight=cw)
    pr=mdl.predict(Xte,verbose=0).argmax(1)+1; o=fase[te]
    O+=o.tolist(); P+=pr.tolist(); per.append(100*f1_score(g(o),g(pr),zero_division=0))
O=np.array(O); P=np.array(P); O3=np.vectorize(S3.get)(O); P3=np.vectorize(S3.get)(P)
print(f"=== 1D-CNN LORO ({pre}) n={len(fase)} ===")
print(f"  6-class acc {accuracy_score(O,P)*100:.1f}%   3-class acc {accuracy_score(O3,P3)*100:.1f}%")
print(f"  generative per-region F1: {np.mean(per):.1f} +/- {np.std(per):.1f}  range {min(per):.0f}-{max(per):.0f}")
print("  (compare MiniROCKET+LightGBM: 6cls 57.2%, 3cls 69.8%, gen 59.3+/-7.9)")
