#!/usr/bin/env python3
"""Validate mixed E7 original Qwen/Llama and versioned Mistral/Phi before gold join."""
from pathlib import Path
import argparse,hashlib,json,sys
P=Path

def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def rh(task):return sha(json.dumps(task,sort_keys=True).encode())
def parse_original(r,style,classes):
    if r.get('status')!='received' or r.get('finish_reason')=='length':return None
    text=r.get('response_text');
    if not isinstance(text,str):return None
    s=text.strip()
    if style=='direct':return s if s in classes else None
    if style=='reason':
      # Preserve the original strict final line contract (don't accept extra text).
      lines=s.splitlines()
      if lines and lines[-1].startswith('Answer: '):
        c=lines[-1][8:].strip()
        return c if c in classes else None
    return None
def parse_v2(r,classes):
    s=r.get('response_text')
    return s.strip() if r.get('status')=='received' and r.get('finish_reason')=='stop' and isinstance(s,str) and s.strip() in classes else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=P,required=True);ap.add_argument('--amendment',type=P,required=True);ap.add_argument('--old',type=P,required=True);ap.add_argument('--new',type=P,required=True);ap.add_argument('--out',type=P,required=True);a=ap.parse_args()
    orig=a.repo/'AuditRepair-v8.4/E7_tasks';older=json.loads((orig/'E7_GOLD_BLIND_MANIFEST.json').read_text());v2=json.loads((a.amendment/'E7_RC2_MANIFEST.json').read_text());assert sha((orig/'E7_GOLD_BLIND_MANIFEST.json').read_bytes())==v2['parent_manifest_sha256']
    classes=older['classes'];assert classes==v2['classes'];a.out.mkdir(parents=True,exist_ok=True)
    report={'status':'failed_pre_gold','amendment':'E7-RC2','original_failed_collection_preserved':True,'no_gold_joined':True,'assessments':{},'failure_reasons':[]}
    all_labels={}
    for pool,n in [('ag_news_5000',5000),('ag_news_holdout',2600)]:
      for family in ('qwen','llama','mistral','phi'):
        stem=f'{pool}_{family}';name=stem+'_tasks.jsonl';new_source=family in ('mistral','phi')
        task_path=(a.amendment/'tasks'/name) if new_source else (orig/name)
        spec=v2['tasks'][name] if new_source else older['tasks'][name]
        assert sha(task_path.read_bytes())==spec['sha256'],'Task hash mismatch'
        inp=(a.new if new_source else a.old)/'responses'/f'{stem}_responses.jsonl'
        if not inp.exists():report['failure_reasons'].append(f'Missing {inp}');continue
        tasks=load(task_path);responses=load(inp)
        if len(tasks)!=n or len(responses)!=n:report['failure_reasons'].append(f'Incomplete {stem}: {len(responses)}/{n}');continue
        indexed={str(r['item_id']):r for r in responses}
        if len(indexed)!=n:report['failure_reasons'].append(f'Duplicate {stem}');continue
        label_map={};invalid=0
        for task in tasks:
          sid=str(task['item_id']);r=indexed.get(sid)
          if r is None or r['request_sha256']!=rh(task):raise SystemExit(f'Request hash drift {stem} {sid}')
          c=parse_v2(r,classes) if new_source else parse_original(r,task['style'],classes)
          if c is None:invalid+=1;label_map[sid]=-1
          else:label_map[sid]=classes.index(c)
        report['assessments'][stem]={'response_count':n,'unusable':invalid,'unusable_rate':invalid/n,'passed':invalid/n<=.05,'collection_version':'RC2' if new_source else 'original'}
        all_labels[stem]=label_map
        if invalid/n>.05:report['failure_reasons'].append(f'Quality gate {stem} {invalid}/{n}')
    valid=not report['failure_reasons'] and len(report['assessments'])==8
    report['status']='passed_pre_gold' if valid else 'failed_pre_gold'
    reportpath=a.out/'E7_RC2_PRE_GOLD_GATE.json';reportpath.write_text(json.dumps(report,indent=2)+'\n')
    print('Pre-gold result',report['status']);
    for k,v in report['assessments'].items():print(k,v['unusable'],'/',v['response_count'])
    if not valid:raise SystemExit(2)
    # Only NOW read gold from source CSV. It is checked against the original frozen manifest first.
    import pandas as pd
    for pool,n in [('ag_news_5000',5000),('ag_news_holdout',2600)]:
      src=a.repo/'AuditRepair-v8.3/study/frozen_v8_1/data'/f'{pool}.csv'
      hash_expected=older['tasks'][f'{pool}_qwen_tasks.jsonl']['source_file_sha256']
      assert sha(src.read_bytes())==hash_expected, f'Gold/source file hash mismatch for {pool}'
      frame=pd.read_csv(src,dtype={'id':str}); assert len(frame)==n and frame.id.is_unique
      final=frame[['id','text','gold']].copy()
      for family in ('qwen','llama','mistral','phi'):final['src_'+family]=[all_labels[f'{pool}_{family}'][str(k)] for k in frame.id]
      final.to_csv(a.out/f'e7_rc2_four_family_{pool}.csv',index=False)
    report['status']='passed_and_joined';report['no_gold_joined']=False
    reportpath.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: all eight quality gates and joined panels')
if __name__=='__main__':main()
