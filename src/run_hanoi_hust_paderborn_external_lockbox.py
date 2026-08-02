"""One-shot HANOI-to-Paderborn boundary check with a frozen source model."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from electrical_fm.hanoi_hust_features import record_feature_blocks

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/hanoi_hust/source_features.npz'
EXT_ROOT=ROOT/'external_lockbox_paderborn_20260730/extracted'
OUT=ROOT/'results/g3/hanoi_hust_paderborn_external_lockbox.json'
CACHE=ROOT/'results/g3/hanoi_hust_paderborn_external_features.npz'
LABELS={'K001':np.array([0,0,0],dtype=np.int8),'KA01':np.array([0,1,0],dtype=np.int8),'KI01':np.array([1,0,0],dtype=np.int8)}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def _vibration(path:Path)->np.ndarray:
 d=loadmat(path,squeeze_me=True,struct_as_record=False); s=next(v for k,v in d.items() if not k.startswith('__')); candidates=[]
 for y in np.ravel(s.Y):
  if getattr(y,'Name','')=='vibration_1': candidates.append(np.asarray(y.Data,dtype=np.float64).reshape(-1))
 if len(candidates)!=1: raise RuntimeError(f'vibration_1 not unique in {path}')
 return candidates[0]
def _features(path:Path)->dict[str,np.ndarray]:
 signal=_vibration(path)
 if signal.size < 200000: raise RuntimeError(f'signal too short: {path} {signal.size}')
 signal=np.asarray(resample_poly(signal,4,5),dtype=np.float32)
 blocks=[]
 for offset in (0,12800,25600):
  b,_=record_feature_blocks(signal,offset=offset); blocks.append(b)
 return {name:np.concatenate([b[name] for b in blocks]).astype(np.float32) for name in ('statistics','fixed_log_power','envelope_log_power')}
def main()->None:
 files=[]
 for code in LABELS:
  paths=sorted((EXT_ROOT/code).rglob('*.mat'))
  if len(paths)!=80: raise RuntimeError(f'{code}: expected 80 files, found {len(paths)}')
  files.extend((code,p) for p in paths)
 names=('statistics','fixed_log_power','envelope_log_power'); ext={n:[] for n in names}; codes=[]; paths=[]
 for i,(code,path) in enumerate(files):
  b=_features(path)
  for n in names: ext[n].append(b[n])
  codes.append(code); paths.append(path.name)
 arrays={n:np.stack(ext[n]).astype(np.float32) for n in names}; y=np.stack([LABELS[c] for c in codes]); np.savez_compressed(CACHE,**arrays,codes=np.asarray(codes),names=np.asarray(paths),targets=y)
 with np.load(SOURCE,allow_pickle=False) as z: sx=np.asarray(z['envelope_log_power'],dtype=np.float32).reshape(len(z['state']),-1); sy=np.stack([np.array([s=='I',s=='O',s=='B'],dtype=np.int8) for s in z['state'].astype(str)])
 scaler=StandardScaler().fit(sx); zsx=scaler.transform(sx); probs=np.zeros((len(y),3),dtype=np.float64)
 for h in range(3):
  m=LogisticRegression(C=10.0,class_weight='balanced',max_iter=5000,random_state=20260730); m.fit(zsx,sy[:,h]); probs[:,h]=m.predict_proba(scaler.transform(arrays['envelope_log_power']))[:,1]
 hard=(probs>=0.5).astype(np.int8); unit=[]
 for code in LABELS:
  mask=np.asarray(codes)==code; p=probs[mask].mean(axis=0); t=LABELS[code]; unit.append({'bearing_code':code,'record_count':int(mask.sum()),'target':t.tolist(),'probability':p.tolist(),'prediction':(p>=0.5).astype(int).tolist(),'exact_set':bool(np.all(t==(p>=0.5)))})
 observed=[h for h in range(3) if len(np.unique(y[:,h]))>1]; metrics={}
 for h in observed:
  unit_p=np.stack([u['probability'] for u in unit])[:,h]; unit_t=np.stack([u['target'] for u in unit])[:,h]; unit_h=(unit_p>=0.5).astype(np.int8); metrics[['inner','outer','ball'][h]]={'auroc':float(roc_auc_score(unit_t,unit_p)),'aupr':float(average_precision_score(unit_t,unit_p)),'balanced_accuracy':float(balanced_accuracy_score(unit_t,unit_h)),'brier':float(brier_score_loss(unit_t,unit_p)),'macro_f1':float(f1_score(unit_t,unit_h,average='macro',zero_division=0))}
 metrics['exact_set_accuracy']=float(np.mean([u['exact_set'] for u in unit])); metrics['hamming_loss']=float(np.mean(np.stack([u['target'] for u in unit]) != np.stack([u['prediction'] for u in unit])))
 out={'stage':'hanoi_hust_as_g3_paderborn_external_lockbox','schema_version':1,'status':'completed','information_budget':'I0_source_only','independent_metric_unit':'physical_bearing_code','source_model':{'cache_sha256':sha(SOURCE),'representation':'envelope_log_power','classifier':'logistic_l2','C':10.0,'scaler':'HANOI_source_only'},'lockbox':{'source':'https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download','archive_sha256':{'K001.rar':'0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5','KA01.rar':'6a6be1e11132730cc6f560d51eacedcbfd5fd74b829e9d8d3728c6c8a7cd4c0e','KI01.rar':'b1dd6d99bb64d556f889fefaedb7e6e672900f5f015125615feaae776f055348'},'bearing_count':3,'record_count':240},'external_feature_cache':{'path':CACHE.relative_to(ROOT).as_posix(),'sha256':sha(CACHE)},'metrics':metrics,'unit_rows':unit}; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
