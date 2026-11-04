"""Independent structural and selected numerical audit of E7-RC2 release."""
from pathlib import Path
import argparse,hashlib,json,sys
import pandas as pd,numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(r,out):
 assert json.loads((out/'E7_RC2_COMPLETION.json').read_text())['status']=='complete'
 cfg=json.loads((out/'E7_RC2_ANALYSIS_PROTOCOL.json').read_text())
 for rel,h in cfg['source_hashes'].items():
  file=r/cfg['source_paths'][rel]
  assert sha(file)==h,(rel,file)
 rows=[];states=[]
 for pool,n in [('ag_news_5000',5000),('ag_news_holdout',2600)]:
  for seed in range(20):
   cp=json.loads((out/'checkpoints'/pool/f'seed_{seed:02d}.json').read_text())
   assert cp['complete'] and cp['pool']==pool and cp['seed']==seed and cp['protocol_sha256']==sha(out/'E7_RC2_ANALYSIS_PROTOCOL.json')
   assert len(cp['rows'])==250 and len(cp['states'])==3
   for st in cp['states']:
    d=st['diagnostics'];assert 0<=d['audited']<=400 and d['certified_groups']==sum(x['guarded'] for x in st['guarded_group_summaries'])
    states.append((pool,seed,st['generator'],d['certified_groups']))
   rows.extend(cp['rows'])
 df=pd.DataFrame(rows)
 assert len(df)==10000
 assert df[['pool','seed','rep','generator','score','role']].duplicated().sum()==0
 assert df.groupby(['pool','seed','rep']).size().eq(25).all()
 assert df.cost.le(df.pool.map({'ag_news_5000':5000,'ag_news_holdout':2600})).all()
 assert df.cost.ge(df.pre_audits).all()
 assert (df.cost==df.pre_audits+df.calibration_new+df.threshold_new).all()
 assert (df.error>=0).all() and (df.error<=1).all()
 inactive=0
 for key,grp in df[df.generator!='none'].groupby(['pool','seed','rep','generator','score']):
  assert set(grp.role)=={'A_base','B_repair_fixed','C_repair_rescored','O1_gold_shortlist'}
  row=grp.set_index('role')
  if int(row.loc['A_base','accepted_state'])==0:
   assert row.loc['A_base','cost']==row.loc['B_repair_fixed','cost']==row.loc['C_repair_rescored','cost']
   assert row.loc['A_base','error']==row.loc['B_repair_fixed','error']==row.loc['C_repair_rescored','error']
   inactive+=1
 d=pd.read_csv(out/'e7_rc2_paired_comparisons.csv')
 assert len(d)==2400 and (d[['pool','seed','rep','generator','score']].duplicated().sum()==0)
 assert (d.A_cost-d.B_cost==d.label_saved).all()
 assert (d.B_cost-d.C_cost==d.rescoring_saved).all()
 assert (d.A_cost-d.C_cost==d.joint_saved).all()
 ss=pd.read_csv(out/'e7_rc2_seed_cluster_summary.csv');assert len(ss)==12
 for _,s in ss.iterrows():
  sub=d[(d.pool==s['pool'])&(d.generator==s.generator)&(d.score==s.score)]
  assert len(sub)==200 and abs(float(s.mean_joint_saved)-sub.joint_saved.mean())<1e-9
 # Spot-check the threshold-optimized PAC implementation against the v8.3
 # original Python capital loop, using actual E7 source matrices and ties.
 sys.path.insert(0,str(r/'AuditRepair-v8.3/study/corrected'))
 sys.path.insert(0,str(r/'AuditRepair-v8.4/code'))
 from audit_repair.core import dawid_skene
 from audit_repair.pac import pac_labels
 from fast_pac import pac_labels_fast
 cases=0
 for pool in ['ag_news_5000','ag_news_holdout']:
  data=pd.read_csv(r/'e7_collected_rc2/validated'/f'e7_rc2_four_family_{pool}.csv')
  L=data[['src_qwen','src_llama','src_mistral','src_phi']].to_numpy(int)
  gold=data.gold.to_numpy(int)
  post=dawid_skene(L,4);pred=post.argmax(1);U=1-post.max(1)
  for seed in [0,17,44]:
   _,slow=pac_labels(pred,U,gold,eps=.03,alpha=.1,m=500,seed=seed)
   _,fast=pac_labels_fast(pred,U,gold,eps=.03,alpha=.1,m=500,seed=seed)
   for k in ['expert_new','calibration_new','threshold_new','error','u_hat']:
    assert slow[k]==fast[k],(pool,seed,k,slow[k],fast[k])
   cases+=1
 result={'status':'passed','preprocessing_states':len(states),'accepted_group_actions':int(sum(x[-1] for x in states)),
    'pac_arm_records':len(df),'paired_evaluations':len(d),'generator_score_cells':len(ss),'inactive_generator_score_repetitions':inactive,
    'independent_original_pac_spotchecks':cases,'all_results_accounting_and_pairing_checked':True}
 (out/'E7_RC2_INDEPENDENT_RESULT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();verify(x.root,x.out)
