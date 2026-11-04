from pathlib import Path
import hashlib,json,csv,collections,re
import numpy as np, pandas as pd
R=Path(__file__).resolve().parents[1]
P=R/'e7_collected_rc2'
F=R/'AuditRepair-v8.4/E7_tasks'
A=R/'e7_rc2'
O=R/'e7_collected'
DATA=R/'AuditRepair-v8.3/study/frozen_v8_1/data'
classes=['World','Sports','Business','Sci/Tech']
report={'quality':{},'rows':{},'sources':{},'status':'checking'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def reqhash(row):return hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
original=json.loads((F/'E7_GOLD_BLIND_MANIFEST.json').read_text());amend=json.loads((A/'E7_RC2_MANIFEST.json').read_text())
assert sha(F/'E7_GOLD_BLIND_MANIFEST.json')==amend['parent_manifest_sha256']
for panel,n in [('ag_news_5000',5000),('ag_news_holdout',2600)]:
 src=DATA/(panel+'.csv'); old=pd.read_csv(src,dtype={'id':str})
 joined=pd.read_csv(P/'validated'/f'e7_rc2_four_family_{panel}.csv',dtype={'id':str})
 assert len(old)==len(joined)==n and old.id.is_unique and joined.id.is_unique
 assert old.id.tolist()==joined.id.tolist() and old.text.fillna('').tolist()==joined.text.fillna('').tolist()
 assert old.gold.tolist()==joined.gold.tolist()
 report['rows'][panel]={'count':n,'gold_exact':True,'text_exact':True}
 for family in ['qwen','llama','mistral','phi']:
  stem=f'{panel}_{family}'; new=family in ['mistral','phi']
  spec=amend['tasks'][f'{stem}_tasks.jsonl'] if new else original['tasks'][f'{stem}_tasks.jsonl']
  task=(A if new else F)/'tasks'/f'{stem}_tasks.jsonl' if new else F/f'{stem}_tasks.jsonl'
  assert sha(task)==spec['sha256'];assert sha(src)==spec['source_file_sha256']
  responses=(P if new else O)/'responses'/f'{stem}_responses.jsonl'
  tasks=load_jsonl(task);response_rows=load_jsonl(responses)
  assert len(tasks)==len(response_rows)==n
  byid={str(r['item_id']):r for r in response_rows};assert len(byid)==n
  labels={};invalid=0;style=collections.Counter();reasons=collections.Counter()
  for t in tasks:
   iid=str(t['item_id']);r=byid[iid];assert r['request_sha256']==reqhash(t)
   s=r.get('response_text');style[t['style']]+=1
   valid=(r['status']=='received' and r.get('finish_reason')!='length' and isinstance(s,str))
   ans=None
   if valid:
    s=s.strip()
    if new:
     if r['finish_reason']=='stop' and s in classes:ans=classes.index(s)
    elif t['style']=='direct':
     if s in classes:ans=classes.index(s)
    else:
     lines=s.splitlines()
     if lines and lines[-1].startswith('Answer: '):
      v=lines[-1][8:].strip()
      if v in classes:ans=classes.index(v)
   if ans is None: invalid+=1;ans=-1;reasons[(r.get('status'),r.get('finish_reason'))]+=1
   labels[iid]=ans
  obs=[labels[iid] for iid in joined.id]
  assert obs==joined['src_'+family].astype(int).tolist(),stem
  assert invalid/n<=0.05,stem
  report['quality'][stem]={'count':n,'unusable':invalid,'rate':invalid/n,'style_counts':dict(style),'version':'RC2' if new else 'original'}
 report['sources'][panel]={'old_df_sha256':sha(src),'new_df_sha256':sha(P/'validated'/f'e7_rc2_four_family_{panel}.csv')}
report['status']='passed';report['conclusion']='All original and RC2 request/response hashes match; labels, text and gold reproduced from frozen input; all eight 5% gates passed.'
out=R/'audit';out.mkdir(exist_ok=True);(out/'independent_pre_gold_join_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
