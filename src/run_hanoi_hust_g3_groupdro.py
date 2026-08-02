"""Run a source-only GroupDRO-style logistic baseline on HANOI-HUST features."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from electrical_fm.hanoi_hust_baselines import aggregate_group_predictions, component_targets, compute_multilabel_metrics

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/"artifacts/hanoi_hust/source_features.npz"; MANIFESTS=ROOT/"results/g2/split_manifests/hanoi_hust_g2_manifests.json"; OUTPUT=ROOT/"results/g3/hanoi_hust_groupdro_bearing_grouped.json"; PREDICTIONS=ROOT/"results/g3/unit_level_predictions/hanoi_hust_groupdro_bearing_grouped.npz"
def sha(path:Path)->str:
    import hashlib
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); s=path.with_suffix('.writing'); s.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); os.replace(s,path)
def atomic_npz(path:Path,a:dict[str,np.ndarray])->None:
    path.parent.mkdir(parents=True,exist_ok=True); s=path.with_suffix('.writing')
    with s.open('wb') as f: np.savez_compressed(f,**a)
    os.replace(s,path)
def fit_head(x:np.ndarray,y:np.ndarray,domains:np.ndarray,seed:int)->LogisticRegression:
    scaler=StandardScaler().fit(x); z=scaler.transform(x); unique=np.unique(domains); weights=np.ones(len(x),dtype=float)/len(x)
    model=None
    for _ in range(6):
        model=LogisticRegression(C=10.0,class_weight='balanced',max_iter=5000,random_state=seed,solver='lbfgs'); model.fit(z,y,sample_weight=weights)
        losses=np.zeros(len(unique))
        for i,d in enumerate(unique):
            mask=domains==d
            if np.unique(y[mask]).size<2: losses[i]=log_loss(y[mask],model.predict_proba(z[mask])[:,1],labels=[0,1])
            else: losses[i]=log_loss(y[mask],model.predict_proba(z[mask])[:,1],labels=[0,1])
        losses=np.exp(losses-np.max(losses)); losses/=losses.sum(); weights=np.zeros(len(x))
        for i,d in enumerate(unique): weights[domains==d]=losses[i]/np.sum(domains==d)
    model._g3_scaler=scaler  # type: ignore[attr-defined]
    return model
def run(*,count:int=10)->dict[str,Any]:
    with np.load(CACHE,allow_pickle=False) as z: arrays={k:np.asarray(z[k]) for k in z.files}
    x=np.asarray(arrays['envelope_log_power'],dtype=np.float32).reshape(len(arrays['state']),-1); y=component_targets(arrays['state'].astype(str)); bearings=np.asarray([f"{s}{b}" for s,b in zip(arrays['state'].astype(str),arrays['bearing_type'],strict=True)],dtype=str); domains=np.asarray(arrays['load_w'],dtype=str); manifests=json.loads(MANIFESTS.read_text(encoding='utf-8'))['bearing_grouped'][:count]
    rows=[]; fs=[]; fg=[]; ft=[]; fp=[]; fy=[]
    for sid,m in enumerate(manifests):
        tr=np.asarray(m['train_indices'],dtype=int); te=np.asarray(m['test_indices'],dtype=int); probs=np.zeros((len(te),3))
        for h in range(3):
            model=fit_head(x[tr],y[tr,h],domains[tr],20260730+sid); probs[:,h]=model.predict_proba(model._g3_scaler.transform(x[te]))[:,1]  # type: ignore[attr-defined]
        g=aggregate_group_predictions(y[te],probs,bearings[te]); rows.append({'split_id':sid,'train_record_count':int(len(tr)),'test_record_count':int(len(te)),'metrics':compute_multilabel_metrics(g['targets'],g['probabilities'],g['predictions'])}); fs.extend([sid]*len(g['groups'])); fg.extend(g['groups'].tolist()); ft.append(g['targets']); fp.append(g['probabilities']); fy.append(g['predictions'])
    atomic_npz(PREDICTIONS,{'split_id':np.asarray(fs,dtype=np.int16),'groups':np.asarray(fg,dtype='U'),'targets':np.concatenate(ft).astype(np.int8),'probabilities':np.concatenate(fp).astype(np.float32),'predictions':np.concatenate(fy).astype(np.int8)})
    names=('mean_component_auroc','mean_component_aupr','mean_component_balanced_accuracy','mean_component_macro_f1','mean_brier_score','exact_set_accuracy','hamming_loss'); summary={n:{'mean':float(np.mean([r['metrics'][n] for r in rows])),'std':float(np.std([r['metrics'][n] for r in rows],ddof=1)) if len(rows)>1 else 0.0} for n in names}; out={'stage':'hanoi_hust_as_g3_groupdro','schema_version':1,'status':'completed','method_level':'R2','information_budget':'I0_source_only','independent_metric_unit':'physical_bearing','candidate':{'family':'GroupDRO_style_logistic','representation':'envelope_log_power','rounds':6,'domain':'load_w','C':10.0},'cache_sha256':sha(CACHE),'manifest_sha256':sha(MANIFESTS),'split_count':len(rows),'summary':summary,'splits':rows,'predictions':{'path':PREDICTIONS.relative_to(ROOT).as_posix(),'sha256':sha(PREDICTIONS)}}; atomic_json(OUTPUT,out); return out
if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--count',type=int,default=10); a=p.parse_args(); print(json.dumps(run(count=a.count),indent=2,ensure_ascii=False))
