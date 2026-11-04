#!/usr/bin/env python3
"""E7 first validate all model responses WITHOUT gold, then join to fixed items.

Requires all eight source/panel outputs, exactly one per task. 5% invalid gate
per family AND per panel. Pass/fail written before opening original gold files.
"""
from pathlib import Path
import argparse,hashlib,json,re
import pandas as pd

def sha(b):return hashlib.sha256(b).hexdigest()
def parse(text,style,classes):
    if not isinstance(text,str):return None
    t=text.strip()
    if style=='direct':return classes.index(t) if t in classes else None
    match=re.search(r'(?m)^Answer: (.+)$',t)
    if style=='reason' and match:
        labels=[x for x in re.finditer(r'(?m)^Answer: (.+)$',t)]
        c=labels[-1].group(1).strip()
        return classes.index(c) if c in classes and labels[-1].end()==len(t) else None
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',type=Path,required=True)
    ap.add_argument('--responses',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.tasks/'E7_GOLD_BLIND_MANIFEST.json').read_text());classes=manifest['classes']
    assessments={};predictions={}
    for name,meta in manifest['tasks'].items():
        tasks=a.tasks/name
        if sha(tasks.read_bytes())!=meta['sha256']:raise SystemExit(f'Task hash drift {name}')
        stem=name.removesuffix('_tasks.jsonl')
        path=a.responses/f'{stem}_responses.jsonl'
        if not path.exists():raise SystemExit(f'Missing {path}')
        task_lines=[json.loads(x) for x in tasks.read_text().splitlines() if x.strip()]
        response_lines=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if len(response_lines)!=len(task_lines):raise SystemExit(f'Incomplete responses {stem}: {len(response_lines)}/{len(task_lines)}')
        keyed={r['item_id']:r for r in response_lines}
        if len(keyed)!=len(response_lines):raise SystemExit(f'Duplicate responses in {stem}')
        unusable=0;labels={}
        for t in task_lines:
            key=t['item_id'];r=keyed.get(key)
            if not r or r['request_sha256']!=sha(json.dumps(t,sort_keys=True).encode()):raise SystemExit(f'Request mismatch for {stem}:{key}')
            val=parse(r.get('response_text'),t['style'],classes) if r['status']=='received' and r.get('finish_reason')!='length' else None
            if val is None:unusable+=1;val=-1
            labels[key]=val
        rate=unusable/len(task_lines)
        assessments[stem]={'items':len(task_lines),'unusable':unusable,'unusable_rate':rate,'passed':rate<=.05}
        predictions[stem]=labels
    status='passed_pre_gold' if all(a['passed'] for a in assessments.values()) else 'failed_pre_gold'
    record={'status':status,'assessments':assessments,'source':'All checks computed without reading reference gold or historical labels'}
    (a.out/'pre_gold_quality_gate.json').write_text(json.dumps(record,indent=2)+'\n')
    if status!='passed_pre_gold':
        print('Quality gate failed; no gold joined. Details in',a.out/'pre_gold_quality_gate.json')
        raise SystemExit(2)
    # Gold is consulted ONLY after the independent source quality gate passes.
    for pool in ('ag_news_5000','ag_news_holdout'):
        src=a.reference/f'study/frozen_v8_1/data/{pool}.csv'
        first=manifest['tasks'][f'{pool}_qwen_tasks.jsonl']
        if sha(src.read_bytes())!=first['source_file_sha256']:raise SystemExit('Source CSV changed since prompt freeze')
        old=pd.read_csv(src,dtype={'id':str})
        new=old[['id','text','gold']].copy()
        for fam in ('qwen','llama','mistral','phi'):
            labels=predictions[f'{pool}_{fam}']
            new[f'src_{fam}']=[labels[str(i)] for i in old.id]
        new.to_csv(a.out/f'e7_four_family_{pool}.csv',index=False)
    print('PASS pre-gold quality gate and joined panel files',a.out)
if __name__=='__main__':main()
