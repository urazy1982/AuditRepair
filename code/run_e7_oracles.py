#!/usr/bin/env python3
"""Secondary E7-RC2 oracle ladder with original E1 uncertainty warm-up.
Gold-informed oracles are retrospective only. O4 is calibration-adaptive.
The corrected group acceptance predicate and terminal PAC are retained.
"""
from __future__ import annotations
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np,pandas as pd

def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 r=a.root;out=a.out;out.mkdir(exist_ok=True,parents=True)
 sys.path.insert(0,str(r/'AuditRepair-v8.3/study/corrected'))
 sys.path.insert(0,str(r/'AuditRepair-v8.4/code'))
 from audit_repair.core import Oracle,candidate_groups_disjoint,dawid_skene,run_group_repair,uncertainty_fill
 from fast_pac import pac_labels_fast as pac
 p=r/'results/E7_RC2_ANALYSIS_PROTOCOL.json'
 pp=json.loads(p.read_text());assert filehash(p)==filehash(out/'E7_RC2_ANALYSIS_PROTOCOL.json')
 stat=[];allrows=[]
 names=['pac_only','same_audit_ds','legacy_guarded','O1_shortlist','O2_disagreement','O3_complete_unanimity','O4_baseline_prefix','O5_all_errors']
 for pool in ['ag_news_5000','ag_news_holdout']:
  d=pd.read_csv(r/'e7_collected_rc2/validated'/f'e7_rc2_four_family_{pool}.csv')
  L0=d[['src_qwen','src_llama','src_mistral','src_phi']].to_numpy(int);gold0=d.gold.to_numpy(int);N=len(gold0)
  for seed in range(20):
   s0,sp,seq=np.random.SeedSequence([20261030,seed]).spawn(3)
   perm=np.random.default_rng(s0).permutation(N);L=L0[perm];gold=gold0[perm]
   post0=dawid_skene(L,4)
   res=run_group_repair(L,4,gold,400,alpha=.05,seed=sp,ref_share=.3,min_size=30,stop_after_fails=10)
   guarded,info=res['guarded'];aud=info['audited_labels'];post=info['post'];base=post.argmax(1);U=1-post.max(1)
   assert all(base[i]==y for i,y in aud.items()) and len(aud)<=400
   warm={};uncertainty_fill(L,4,Oracle(gold,120),warm)
   post_ref=dawid_skene(L,4,warm);ref=post_ref.argmax(1)
   for i,y in warm.items():ref[i]=y
   shortlist=sorted(candidate_groups_disjoint(L,ref,post_ref,min_size=30),key=lambda g:-len(g.items))[:10]
   o1=base.copy();beneficial=0
   for group in shortlist:
    rem=np.array([int(i) for i in group.items if i not in aud],dtype=int)
    gain=int((gold[rem]==group.alt).sum()-(gold[rem]==base[rem]).sum())
    if gain>0:o1[rem]=group.alt;beneficial+=1
   wrong=base!=gold
   for i in aud:wrong[i]=False
   dissent=np.array([len(set(v[v>=0]))>=2 for v in L])
   unanimous=np.all(L>=0,axis=1)&np.all(L==L[:,:1],axis=1)
   o2=base.copy();o2[wrong&dissent]=gold[wrong&dissent]
   o3=base.copy();o3[wrong&unanimous]=gold[wrong&unanimous]
   og=U.copy()
   for i,conf in info['repair_conf'].items():og[i]=1-conf
   for rep,cal_seed in enumerate(seq.spawn(10)):
    _,refinfo=pac(base,U,gold,eps=.03,alpha=.1,m=500,already=aud,seed=cal_seed)
    o4=base.copy();o4[wrong&(U<=refinfo['u_hat'])]=gold[wrong&(U<=refinfo['u_hat'])]
    arms={'pac_only':(post0.argmax(1),1-post0.max(1),{}),'same_audit_ds':(base,U,aud),
      'legacy_guarded':(guarded,og,aud),'O1_shortlist':(o1,U,aud),
      'O2_disagreement':(o2,U,aud),'O3_complete_unanimity':(o3,U,aud),
      'O4_baseline_prefix':(o4,U,aud),'O5_all_errors':(gold.copy(),U,aud)}
    for method in names:
     y,u,owned=arms[method]
     if method=='same_audit_ds':z=refinfo
     else:_,z=pac(y,u,gold,eps=.03,alpha=.1,m=500,already=owned,seed=cal_seed)
     assert method!='O5_all_errors' or z['threshold_new']==0
     allrows.append({'pool':pool,'seed':seed,'rep':rep,'method':method,
       'cost':len(owned)+z['expert_new'],'error':z['error'],'threshold_new':z['threshold_new'],
       'pre_audits':len(owned),'calibration_new':z['calibration_new'],
       'certified_groups':info['groups_repaired'],'beneficial_shortlist':beneficial,
       'oracle_is_calibration_adaptive':int(method=='O4_baseline_prefix')})
   stat.append({'pool':pool,'seed':seed,'audited':len(aud),'groups_certified':info['groups_repaired'],
                'beneficial_shortlisted_groups':beneficial,'wrong_disagreement':int((wrong&dissent).sum()),
                'wrong_unanimous':int((wrong&unanimous).sum())})
   print('E7 oracle',pool,seed,'cert',info['groups_repaired'],'benefit',beneficial,flush=True)
   # Checkpoint now, to preserve partial work if interrupted
   (out/'e7_rc2_e1_oracle_rows_checkpoint.csv').parent.mkdir(exist_ok=True,parents=True)
   pd.DataFrame(allrows).to_csv(out/'e7_rc2_e1_oracle_rows_checkpoint.csv',index=False)
 df=pd.DataFrame(allrows)
 assert len(df)==20*2*10*8
 df.to_csv(out/'e7_rc2_e1_oracle_rows.csv',index=False)
 pd.DataFrame(stat).to_csv(out/'e7_rc2_e1_preprocessing.csv',index=False)
 rng=np.random.default_rng(20261104);agg=[]
 for pool in ['ag_news_5000','ag_news_holdout']:
  ds=df[df.pool==pool].pivot(index=['seed','rep'],columns='method',values='cost')
  for method in names:
   vals=(ds['same_audit_ds']-ds[method]).groupby(level='seed').mean().to_numpy()
   boots=vals[rng.integers(0,20,size=(4000,20))].mean(1)
   lo,hi=np.quantile(boots,[.025,.975]);n=5000 if pool=='ag_news_5000' else 2600
   agg.append({'pool':pool,'method':method,'n':n,'seeds':20,'reps':200,'mean_labels_saved':float(vals.mean()),
               'share_of_pool_saved':float(vals.mean()/n),'ci95_low':float(lo),'ci95_high':float(hi)})
 pd.DataFrame(agg).to_csv(out/'e7_rc2_e1_oracle_summary.csv',index=False)
 print(pd.DataFrame(agg).to_string(index=False),flush=True)
 (out/'E7_RC2_E1_STATUS.json').write_text(json.dumps({'status':'complete','rows':len(df),'scope':'Secondary E1 oracle diagnostics after the four-family collection. O4 is gold+calibration adaptive, not a deployable or guaranteed PAC procedure.'},indent=2)+'\n')
if __name__=='__main__':main()
