"""Run a compact source-only InceptionTime-style baseline on HANOI-HUST."""
from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from electrical_fm.hanoi_hust_baselines import aggregate_group_predictions, component_targets, compute_multilabel_metrics

ROOT=Path(__file__).resolve().parents[1]
WAVEFORM_CACHE=ROOT/"artifacts/hanoi_hust_window/source_window_waveforms.npy"
WAVEFORM_METADATA=ROOT/"artifacts/hanoi_hust_window/source_window_waveforms_metadata.json"
MANIFESTS=ROOT/"results/g2/split_manifests/hanoi_hust_g2_manifests.json"
OUTPUT=ROOT/"results/g3/hanoi_hust_inceptiontime_bearing_grouped.json"
PREDICTIONS=ROOT/"results/g3/unit_level_predictions/hanoi_hust_inceptiontime_bearing_grouped.npz"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); s=path.with_suffix(".writing"); s.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8"); os.replace(s,path)
def atomic_npz(path:Path,arrays:dict[str,np.ndarray])->None:
    path.parent.mkdir(parents=True,exist_ok=True); s=path.with_suffix(".writing")
    with s.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(s,path)

class InceptionBlock(nn.Module):
    def __init__(self,cin:int,cout:int)->None:
        super().__init__(); bottleneck=max(4,cout//4)
        self.b=nn.Sequential(nn.Conv1d(cin,bottleneck,1,bias=False),nn.BatchNorm1d(bottleneck),nn.ReLU())
        self.c1=nn.Conv1d(bottleneck,cout,9,padding=4,bias=False); self.c2=nn.Conv1d(bottleneck,cout,19,padding=9,bias=False); self.c3=nn.Conv1d(bottleneck,cout,39,padding=19,bias=False)
        self.pool=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),nn.Conv1d(cin,cout,1,bias=False))
        self.bn=nn.BatchNorm1d(cout*4); self.act=nn.ReLU()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        z=self.b(x); return self.act(self.bn(torch.cat((self.c1(z),self.c2(z),self.c3(z),self.pool(x)),dim=1)) )

class InceptionTime(nn.Module):
    def __init__(self)->None:
        super().__init__(); self.blocks=nn.Sequential(InceptionBlock(1,8),InceptionBlock(32,8),InceptionBlock(32,8)); self.head=nn.Linear(32,3)
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.head(self.blocks(x).mean(dim=-1))

def seed(v:int)->None:
    random.seed(v); np.random.seed(v); torch.manual_seed(v); torch.use_deterministic_algorithms(True,warn_only=True)
def prep(x:np.ndarray)->np.ndarray:
    x=np.asarray(x[:,::64],dtype=np.float32); x-=x.mean(1,keepdims=True); return x/np.maximum(x.std(1,keepdims=True),1e-6)
def fit(xtr:np.ndarray,y:np.ndarray,xte:np.ndarray,s:int,epochs:int)->np.ndarray:
    seed(s); m=InceptionTime(); pos=y.sum(0); neg=len(y)-pos; loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/np.maximum(pos,1),dtype=torch.float32)); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(xtr[:,None,:]),torch.from_numpy(y.astype(np.float32))); ld=DataLoader(ds,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(s),num_workers=0); m.train()
    for _ in range(epochs):
        for xb,yb in ld: opt.zero_grad(set_to_none=True); l=loss(m(xb),yb); l.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.sigmoid(m(torch.from_numpy(xte[:,None,:]))).numpy().astype(np.float64)
def run(*,count:int=10,epochs:int=3)->dict[str,Any]:
    meta=json.loads(WAVEFORM_METADATA.read_text(encoding="utf-8")); row=meta["metadata"]; wav=np.load(WAVEFORM_CACHE,mmap_mode="r",allow_pickle=False); states=np.asarray(row["state"],dtype=str); bearings=np.asarray(row["bearing_id"],dtype=str); labels=component_targets(states); manifests=json.loads(MANIFESTS.read_text(encoding="utf-8"))["bearing_grouped"][:count]
    rows=[]; fs=[]; fg=[]; ft=[]; fp=[]; fy=[]
    for sid,man in enumerate(manifests):
        tr=np.flatnonzero(np.isin(bearings,man["train_groups"])); te=np.flatnonzero(np.isin(bearings,man["test_groups"])); p=fit(prep(np.asarray(wav[tr])),labels[tr],prep(np.asarray(wav[te])),20260730+sid,epochs); g=aggregate_group_predictions(labels[te],p,bearings[te]); rows.append({"split_id":sid,"train_window_count":int(len(tr)),"test_window_count":int(len(te)),"metrics":compute_multilabel_metrics(g["targets"],g["probabilities"],g["predictions"])}); fs.extend([sid]*len(g["groups"])); fg.extend(g["groups"].tolist()); ft.append(g["targets"]); fp.append(g["probabilities"]); fy.append(g["predictions"])
    atomic_npz(PREDICTIONS,{"split_id":np.asarray(fs,dtype=np.int16),"groups":np.asarray(fg,dtype="U"),"targets":np.concatenate(ft).astype(np.int8),"probabilities":np.concatenate(fp).astype(np.float32),"predictions":np.concatenate(fy).astype(np.int8)})
    names=("mean_component_auroc","mean_component_aupr","mean_component_balanced_accuracy","mean_component_macro_f1","mean_brier_score","exact_set_accuracy","hamming_loss"); summary={n:{"mean":float(np.mean([r["metrics"][n] for r in rows])),"std":float(np.std([r["metrics"][n] for r in rows],ddof=1)) if len(rows)>1 else 0.0} for n in names}
    out={"stage":"hanoi_hust_as_g3_inceptiontime","schema_version":1,"status":"completed","method_level":"R2","information_budget":"I0_source_only","independent_metric_unit":"physical_bearing","candidate":{"family":"InceptionTime_style","epochs":epochs,"downsample":64,"batch_size":256,"input":"raw_one_channel_window"},"waveform_cache":{"path":WAVEFORM_CACHE.relative_to(ROOT).as_posix(),"sha256":sha(WAVEFORM_CACHE)},"manifest_sha256":sha(MANIFESTS),"split_count":len(rows),"summary":summary,"splits":rows,"predictions":{"path":PREDICTIONS.relative_to(ROOT).as_posix(),"sha256":sha(PREDICTIONS)}}; atomic_json(OUTPUT,out); return out
if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=10); p.add_argument("--epochs",type=int,default=3); a=p.parse_args(); print(json.dumps(run(count=a.count,epochs=a.epochs),indent=2,ensure_ascii=False))
