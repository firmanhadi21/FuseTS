"""Per-pixel classifier confidence map (LightGBM max class-probability) for one period."""
import glob, sys, warnings, numpy as np, pandas as pd, xarray as xr, rioxarray
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from train_v3_mogpr_ensemble import _window
from produce_annual_tiled import TO3
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from lightgbm import LGBMClassifier
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUN="output/production/java_2024"; PIDX=7; W=8  # period p08
MASK="/home/unika_sianturi/work/landcover/s1-land-cover-classification/cropping_intensity_consensus_mt2024_25/paddy_mask.tif"
def utm(tid): c=int(tid.split("_c")[1]); return 32748 if 103.53+c*0.2+0.1<108 else 32749
z=np.load("output/phase_model/series_combined.npz"); m=pd.read_csv("output/phase_model/series_combined_meta.csv")
wn=_window(np.nan_to_num(z["ndvi"]),z["t_grid"],z["label_ord"],W); Xtr=np.stack([wn,np.gradient(wn,axis=1)],1).astype("float32")
mr=MiniRocketMultivariate(num_kernels=2000,random_state=0).fit(Xtr)
clf=LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,class_weight="balanced",verbose=-1).fit(mr.transform(Xtr).values,m["fase"].map(TO3).values)
print("classifier trained",flush=True)
tiles=sorted(glob.glob(f"{RUN}/tiles/*/fused.npz")); arrs=[]
for i,fp in enumerate(tiles):
    d=fp.rsplit("/",1)[0]; tid=d.split("/")[-1]; crs=f"EPSG:{utm(tid)}"
    z2=np.load(fp); fused,ys,xs=z2["fused"],z2["ys"],z2["xs"]; L=fused.shape[1]
    ok=np.isfinite(fused).all(1)
    if ok.sum()==0: continue
    F=np.nan_to_num(fused[ok]); lo,hi=PIDX-W,PIDX+W+1
    seg=np.stack([np.concatenate([np.full(max(0,-lo),F[k,0]),F[k,max(0,lo):min(L,hi)],np.full(max(0,hi-L),F[k,-1])])[:2*W+1] for k in range(F.shape[0])])
    X=np.stack([seg,np.gradient(seg,axis=1)],1).astype("float32")
    conf=clf.predict_proba(mr.transform(X).values).max(1)
    cube=xr.open_dataset(f"{d}/cube.nc").rio.write_crs(crs); ny,nx=cube.sizes["y"],cube.sizes["x"]
    g=np.full((ny,nx),np.nan,"float32"); g[ys[ok],xs[ok]]=conf
    da=xr.DataArray(g,coords={"y":cube["y"],"x":cube["x"]},dims=("y","x")).rio.write_crs(crs)
    try: arrs.append(da.rio.reproject("EPSG:4326"))
    except Exception: pass
    if i%80==0: print(f"{i}/{len(tiles)}",flush=True)
mos=merge_arrays(arrs); mask=rioxarray.open_rasterio(MASK)
mos=mos.rio.reproject_match(mask.rio.clip_box(*mos.rio.bounds()),resampling=Resampling.bilinear)
v=mos.isel(band=0).values if mos.ndim==3 else mos.values
f=10; H,Wd=v.shape; vp=np.pad(v,((0,(-H)%f),(0,(-Wd)%f)),constant_values=np.nan); h,w=vp.shape
vv=np.nanmean(vp.reshape(h//f,f,w//f,f),axis=(1,3))
ext=[float(mos.x.min()),float(mos.x.max()),float(mos.y.min()),float(mos.y.max())]
fig,ax=plt.subplots(figsize=(15,4.6)); ax.set_facecolor("#f5f5f5")
im=ax.imshow(np.ma.masked_invalid(vv),cmap="viridis",vmin=0.4,vmax=1.0,extent=ext,interpolation="nearest")
cb=fig.colorbar(im,ax=ax,shrink=.85); cb.set_label("classifier confidence (max class probability)")
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title("Per-pixel classification confidence, Java 2024 P08 (~late March)")
fig.tight_layout(); fig.savefig("/home/unika_sianturi/work/rice-growth-stage-mapping/paper_latex/companion_figures/fig_confidence_map.png",dpi=150,bbox_inches="tight")
print(f"wrote fig_confidence_map.png | mean conf {np.nanmean(v):.2f}",flush=True)
