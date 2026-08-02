"""Paired split-level comparison of source-only GroupDRO and fixed logistic."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
from electrical_fm.hanoi_hust_baselines import compute_multilabel_metrics

ROOT=Path(__file__).resolve().parents[1]
G=ROOT/"results/g3/unit_level_predictions/hanoi_hust_groupdro_bearing_grouped.npz"
L=ROOT/"results/g2/unit_level_predictions/hanoi_hust_bearing_grouped_fixed_prespecified.npz"
OUT=ROOT/"results/g3/hanoi_hust_groupdro_vs_logistic_paired.json"
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main()->None:
 g=np.load(G); l=np.load(L)
 if not (np.array_equal(g['split_id'],l['split_id']) and np.array_equal(g['groups'],l['groups']) and np.array_equal(g['targets'],l['targets'])): raise RuntimeError('prediction axes are not paired')
 diffs=[]; rows=[]
 for sid in np.unique(g['split_id']):
  a=g['split_id']==sid; gm=compute_multilabel_metrics(g['targets'][a],g['probabilities'][a]); lm=compute_multilabel_metrics(l['targets'][a],l['probabilities'][a]); d=float(gm['mean_component_auroc']-lm['mean_component_auroc']); diffs.append(d); rows.append({'split_id':int(sid),'groupdro_auroc':gm['mean_component_auroc'],'logistic_auroc':lm['mean_component_auroc'],'difference':d})
 diffs=np.asarray(diffs,float); rng=np.random.default_rng(20260730); draws=200000; signs=rng.choice(np.array([-1.0,1.0]),size=(draws,len(diffs))); null=np.mean(signs*diffs,axis=1); observed=float(np.mean(diffs)); p=float((np.count_nonzero(np.abs(null)>=abs(observed))+1)/(draws+1))
 out={'stage':'hanoi_hust_as_g3_groupdro_comparison','schema_version':1,'paired_split_count':len(diffs),'groupdro_result_sha256':sha(G),'logistic_result_sha256':sha(L),'observed_mean_auroc_difference':observed,'p_two_sided_sign_permutation':p,'mean_groupdro_auroc':float(np.mean([r['groupdro_auroc'] for r in rows])),'mean_logistic_auroc':float(np.mean([r['logistic_auroc'] for r in rows])),'split_rows':rows}; OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
