#!/usr/bin/env python3
"""Reproduce the E7-RC2 same-item, four-family paired repair/score study.

Do not overwrite original panels or protocol. Uses corrected v8.3 guarded
remainder acceptance, original v8.1 seed streams, and v8.4 fast PAC evaluator.
All performance analysis occurs after frozen gold-blind collection has qualified.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd


def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jdump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');tmp.replace(p)

def setup(r):
 sys.path.insert(0,str(r/'AuditRepair-v8.3/study/frozen_v8_1'))
 sys.path.insert(0,str(r/'AuditRepair-v8.3/study/corrected'))
 sys.path.insert(0,str(r/'AuditRepair-v8.4/code'))
 from audit_repair.core import dawid_skene
 from fast_pac import pac_labels_fast
 from factorial import preprocess_generator
 return dawid_skene,pac_labels_fast,preprocess_generator


def arrays(r,pool):
 src=r/'e7_collected_rc2/validated'/f'e7_rc2_four_family_{pool}.csv'
 original=r/'AuditRepair-v8.3/study/frozen_v8_1/data'/f'{pool}.csv'
 df=pd.read_csv(src,dtype={'id':str});old=pd.read_csv(original,dtype={'id':str})
 assert len(df)==len(old) and list(df.id)==list(old.id) and list(df.text)==list(old.text) and list(df.gold)==list(old.gold)
 assert [x for x in df.columns if x.startswith('src_')]==['src_qwen','src_llama','src_mistral','src_phi']
 L=df[['src_qwen','src_llama','src_mistral','src_phi']].to_numpy(int)
 gold=df.gold.to_numpy(int)
 assert L.min()>=-1 and L.max()<=3 and gold.min()>=0 and gold.max()<=3
 cache=r/'AuditRepair-v8.3/study/frozen_v8_1/feature_cache'/f'{pool}.npz'
 with np.load(cache) as z:X,neighbors=z['features'],z['neighbors']
 assert len(X)==len(gold) and len(neighbors)==len(gold)
 return L,gold,X,neighbors


def calc_one(r,pool,seed,imports,out):
 ds,pac,pre=imports
 target=out/'checkpoints'/pool/f'seed_{seed:02}.json'
 if target.exists():
  ck=json.loads(target.read_text());assert ck['complete'] and ck['seed']==seed and ck['pool']==pool and len(ck['rows'])==250
  return ck,True
 L0,g0,X0,neigh0=arrays(r,pool);N=len(g0);K=4
 data_seed,pre_seed,calib_seed=np.random.SeedSequence([20261030,seed]).spawn(3)
 perm=np.random.default_rng(data_seed).permutation(N)
 inv=np.empty_like(perm);inv[perm]=np.arange(N)
 L,g,X=L0[perm],g0[perm],X0[perm]
 neighbors=inv[neigh0[perm]]
 assert not np.any(neighbors==np.arange(N)[:,None])
 states={name:pre(L,K,g,X,neighbors,name,pre_seed) for name in ['G0_uniform','G1','G2']}
 for name in ['G1','G2']:
  np.testing.assert_array_equal(states['G0_uniform']['warm_indices'],states[name]['warm_indices'])
  np.testing.assert_allclose(states['G0_uniform']['text_probabilities'],states[name]['text_probabilities'],atol=0,rtol=0)
 base0=ds(L,K);pred0=base0.argmax(1);u0=1-base0.max(1)
 rows=[];ss=[]
 for gen,st in states.items():
  d=st['diagnostics'];ss.append({'pool':pool,'seed':seed,'generator':gen,**d})
  assert len(st['audits'])<=400 and d['harmful_accepted_vs_ref']==0 or len(st['audits'])<=400
  for score,U in st['scores'].items():
   if score=='S0':U_C=1-st['post'][np.arange(N),st['pred']]
   else:U_C=1-st['text_probabilities'][np.arange(N),st['pred']]
   assert (np.isfinite(U_C).all() and (U_C>=0).all() and (U_C<=1).all())
   if not np.any(st['pred']!=st['base']):np.testing.assert_array_equal(U,U_C)
 for rep,pac_seed in enumerate(calib_seed.spawn(10)):
  _,info=pac(pred0,u0,g,eps=.03,alpha=.1,m=500,seed=pac_seed)
  rows.append({'pool':pool,'seed':seed,'rep':rep,'generator':'none','score':'none','role':'pac_only',
               'pre_audits':0,'cost':info['expert_new'],'error':info['error'],'cutoff':info['u_hat'],
               'calibration_new':info['calibration_new'],'threshold_new':info['threshold_new'],
               'accepted_state':0,'changed_predictions':0,'certified_groups':0,'nonbeneficial_vs_ref':0})
  for gen,st in states.items():
   audits=st['audits']; base=st['base'];repair=st['pred'];candidate=st['oracle_candidate_labels']
   diag=st['diagnostics'];active=int(diag['certified_groups']>0);changed=int(np.sum(repair!=base))
   assert bool(active)==bool(changed) or (active and not changed)
   for score,U in st['scores'].items():
    UC=1-(st['post'][np.arange(N),repair] if score=='S0' else st['text_probabilities'][np.arange(N),repair])
    _,ai=pac(base,U,g,eps=.03,alpha=.1,m=500,already=audits,seed=pac_seed)
    def add(role,inf):
     rows.append({'pool':pool,'seed':seed,'rep':rep,'generator':gen,'score':score,'role':role,
                  'pre_audits':len(audits),'cost':len(audits)+inf['expert_new'],
                  'error':inf['error'],'cutoff':inf['u_hat'],'calibration_new':inf['calibration_new'],
                  'threshold_new':inf['threshold_new'],'accepted_state':active,
                  'changed_predictions':changed,'certified_groups':diag['certified_groups'],
                  'nonbeneficial_vs_ref':diag['harmful_accepted_vs_ref'],
                  'net_gain_vs_ref':diag['net_gain_vs_ref'],
                  'net_gain_vs_same_audit':diag['net_gain_vs_same_audit']})
    add('A_base',ai)
    if changed:
     _,bi=pac(repair,U,g,eps=.03,alpha=.1,m=500,already=audits,seed=pac_seed)
     _,ci=pac(repair,UC,g,eps=.03,alpha=.1,m=500,already=audits,seed=pac_seed)
    else:bi=ci=ai
    add('B_repair_fixed',bi);add('C_repair_rescored',ci)
    if np.array_equal(candidate,base):oi=ai
    else:_,oi=pac(candidate,U,g,eps=.03,alpha=.1,m=500,already=audits,seed=pac_seed)
    add('O1_gold_shortlist',oi)
 assert len(rows)==250,(len(rows),pool,seed)
 outstates=[]
 for gen,st in states.items():
  outstates.append({'generator':gen,'diagnostics':st['diagnostics'],
      'guarded_group_summaries':[{'group_index':x['group_index'],'size':x['size'],
        'audited':x['audited'],'stop_reason':x['stop_reason'],'guarded':x['guarded'],
        'true_remainder_gain_vs_frozen_ref':x['true_remainder_gain_vs_frozen_ref'],
        'effect_vs_same_audit':x['effect_vs_same_audit']}
        for x in st['group_log']],
      'audits_original_row_ids':sorted(int(perm[i]) for i in st['audits']),
      'warm_original_row_ids':[int(perm[i]) for i in st['warm_indices']]})
 obj={'complete':True,'pool':pool,'seed':seed,'rows':rows,'states':outstates,'protocol_sha256':h(out/'E7_RC2_ANALYSIS_PROTOCOL.json'),
      'corrected_core_sha256':h(r/'AuditRepair-v8.3/study/corrected/audit_repair/core.py'),
      'factorial_sha256':h(r/'AuditRepair-v8.3/study/corrected/factorial.py'),
      'source_csv_sha256':h(r/'e7_collected_rc2/validated'/f'e7_rc2_four_family_{pool}.csv')}
 jdump(target,obj)
 return obj,False


def summarize(out,r):
 cps=[]
 for pool in ['ag_news_5000','ag_news_holdout']:
  for seed in range(20):
   cp=json.loads((out/'checkpoints'/pool/f'seed_{seed:02}.json').read_text());assert cp['complete']
   assert cp['protocol_sha256']==h(out/'E7_RC2_ANALYSIS_PROTOCOL.json')
   cps.append(cp)
 rows=[row for ck in cps for row in ck['rows']]
 df=pd.DataFrame(rows)
 assert len(df)==10000
 assert df.groupby(['pool','seed','rep']).size().eq(25).all()
 df.to_csv(out/'e7_rc2_all_pac_arms.csv',index=False)
 active=[]
 for cp in cps:
  for st in cp['states']:
   active.append({'pool':cp['pool'],'seed':cp['seed'],'generator':st['generator'],**st['diagnostics']})
 pd.DataFrame(active).to_csv(out/'e7_rc2_preprocessing_funnel.csv',index=False)
 paired=[]
 for (pool,gen,score),p in df[df.role!='pac_only'].groupby(['pool','generator','score'],sort=True):
  pivot=p.pivot(index=['seed','rep'],columns='role',values='cost')
  er=p.pivot(index=['seed','rep'],columns='role',values='error')
  seeds=p.groupby('seed')['accepted_state'].max()
  for seedrep,row in pivot.iterrows():
   idx=(pool,gen,score)
   paired.append({'pool':pool,'generator':gen,'score':score,'seed':seedrep[0],'rep':seedrep[1],
       'active':int(seeds.loc[seedrep[0]]),
       'A_cost':int(row['A_base']),'B_cost':int(row['B_repair_fixed']),'C_cost':int(row['C_repair_rescored']),'O1_cost':int(row['O1_gold_shortlist']),
       'label_saved':int(row['A_base']-row['B_repair_fixed']),
       'rescoring_saved':int(row['B_repair_fixed']-row['C_repair_rescored']),
       'joint_saved':int(row['A_base']-row['C_repair_rescored']),
       'oracle_saved':int(row['A_base']-row['O1_gold_shortlist']),
       'A_error':float(er.loc[seedrep,'A_base']),'C_error':float(er.loc[seedrep,'C_repair_rescored'])})
 dif=pd.DataFrame(paired);dif.to_csv(out/'e7_rc2_paired_comparisons.csv',index=False)
 stat=[];rng=np.random.default_rng(20261104)
 for (pool,gen,score),p in dif.groupby(['pool','generator','score'],sort=True):
  arr=p.groupby('seed')[['label_saved','rescoring_saved','joint_saved','oracle_saved']].mean().to_numpy()
  assert arr.shape==(20,4)
  boots=arr[rng.integers(0,20,size=(4000,20))].mean(axis=1)
  v={'pool':pool,'generator':gen,'score':score,'n':5000 if pool=='ag_news_5000' else 2600,'seeds':20,'pairs':len(p),
     'accepted_seeds':int(p.groupby('seed')['active'].max().sum()),
     'joint_nonzero_pairs':int((p.joint_saved!=0).sum()),
     'joint_savings_ge_1pct_pairs':int((p.joint_saved>=.01*v) .sum()) if False else 0,
     'joint_improve_count':int((p.joint_saved>0).sum()),'joint_harm_count':int((p.joint_saved<0).sum()),
     'A_violations':int((p.A_error>.03).sum()),'C_violations':int((p.C_error>.03).sum())}
  n=v['n'];v['joint_savings_ge_1pct_pairs']=int((p.joint_saved>=.01*n).sum());v['joint_added_ge_1pct_pairs']=int((p.joint_saved<=-.01*n).sum())
  for j,kind in enumerate(['label','rescoring','joint','oracle']):
   v[f'mean_{kind}_saved']=float(arr[:,j].mean())
   v[f'{kind}_ci95_low'],v[f'{kind}_ci95_high']=map(float,np.quantile(boots[:,j],[.025,.975]))
  stat.append(v)
 ss=pd.DataFrame(stat);ss.to_csv(out/'e7_rc2_seed_cluster_summary.csv',index=False)
 base=df[df.role=='pac_only'].groupby('pool').agg(mean_pac_only_cost=('cost','mean'),mean_pac_only_error=('error','mean')).reset_index()
 baseline=df[df.role=='A_base'].groupby(['pool','generator','score']).agg(mean_same_audit_cost=('cost','mean'),mean_same_audit_error=('error','mean')).reset_index()
 baseline=baseline.merge(base,on='pool');baseline.to_csv(out/'e7_rc2_total_cost.csv',index=False)
 # Original panel v8.4 E5 is a matched historical computational reference,
 # not a new independent dataset nor a pre-registered replication.
 original=pd.read_csv(r/'AuditRepair-v8.4/results/E5/e5_seedcluster_summary.csv')
 old=original[original.pool.isin(['ag_news_5000','ag_news_holdout'])]
 comparison=ss[['pool','generator','score','accepted_seeds','mean_label_saved','mean_joint_saved']].merge(old[['pool','generator','score','accepted_seeds','mean_label_saving','mean_joint_saving']],on=['pool','generator','score'],suffixes=('_e7','_original'))
 comparison['joint_savings_delta_e7_minus_original']=comparison.mean_joint_saved-comparison.mean_joint_saving
 comparison.to_csv(out/'e7_rc2_vs_original_panel.csv',index=False)
 status={'status':'complete','protocol_sha256':h(out/'E7_RC2_ANALYSIS_PROTOCOL.json'),
       'new_panel_qualifies':True,'panels':2,'seeds_per_panel':20,'pac_reps':10,
       'generator_score_cells':12,'preprocessing_states':120,'evaluated_arm_records':len(df),'paired_comparisons':len(dif),
       'new_annotation_requests':15200,'old_qwen_llama_reused':True,
       'scope':'Same AG News items, amended Mistral/Phi constrained decoding, observational panel contrast, fixed seeds and audit policy'}
 jdump(out/'E7_RC2_COMPLETION.json',status)
 print('\nCOMPLETE',json.dumps(status,indent=2));
 print('\nSUMMARY:\n',ss[['pool','generator','score','accepted_seeds','mean_label_saved','mean_joint_saved','joint_ci95_low','joint_ci95_high']].to_string(index=False))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pool',choices=['all','ag_news_5000','ag_news_holdout'],default='all');ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--seed-stop',type=int,default=20);a=ap.parse_args()
 r=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 config_source=r/'results/E7_RC2_ANALYSIS_PROTOCOL.json';target=out/'E7_RC2_ANALYSIS_PROTOCOL.json'
 if not target.exists():target.write_bytes(config_source.read_bytes())
 assert target.read_bytes()==config_source.read_bytes(),'Frozen E7 protocol changed'
 config=json.loads(target.read_text())
 for k,rel in config['source_paths'].items():assert h(r/rel)==config['source_hashes'][k],f'Source/protocol hash drift {k}'
 assert 0<=a.seed_start<a.seed_stop<=20
 imports=setup(r)
 pools=['ag_news_5000','ag_news_holdout'] if a.pool=='all' else [a.pool]
 for pool in pools:
  for seed in range(a.seed_start,a.seed_stop):
   t=time.monotonic();ck,reused=calc_one(r,pool,seed,imports,out)
   print(f'E7 {pool} seed={seed:02} active={sum(s["diagnostics"]["certified_groups"] for s in ck["states"])} time={time.monotonic()-t:.1f}s cached={reused}',flush=True)
 if a.pool=='all' and a.seed_start==0 and a.seed_stop==20:summarize(out,r)
 else:print('PARTIAL: no summary until full panel/seed grid completed',flush=True)
if __name__=='__main__':main()
