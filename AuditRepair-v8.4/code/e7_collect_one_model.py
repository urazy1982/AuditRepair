#!/usr/bin/env python3
"""Sequential, restartable OpenAI-compatible vLLM collector for one gold-blind worklist.

Never sends gold. Refuses existing output rows with mismatched request hashes;
records failures explicitly (no silent omission). The operator must pin/verify
served checkpoint externally; model ID only is insufficient to prove revision.
"""
from pathlib import Path
import argparse, hashlib, json, os, time, urllib.request, urllib.error

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args()
    rows=[json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
    if not rows:raise SystemExit('Empty tasks')
    ids=[r['item_id'] for r in rows]
    if len(set(ids))!=len(ids):raise SystemExit('Duplicate task IDs')
    a.output.parent.mkdir(exist_ok=True,parents=True)
    existing={}
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            if not line.strip():continue
            record=json.loads(line)
            if record['item_id'] in existing:raise SystemExit('Duplicate completed response')
            existing[record['item_id']]=record
    # Validate server's model ID, but this is NOT a commit-fingerprint verification.
    req=urllib.request.Request(a.base_url.rstrip('/')+'/models')
    try:
        with urllib.request.urlopen(req,timeout=a.timeout) as res:models=json.loads(res.read())
    except Exception as error:raise SystemExit(f'Server preflight failed: {error}')
    served={str(obj.get('id','')) for obj in models.get('data',[])}
    if rows[0]['model'] not in served:raise SystemExit(f'Expected model not served: {rows[0]["model"]}; available: {served}')
    with a.output.open('a',encoding='utf-8') as fp:
        for k,r in enumerate(rows):
            sid=str(r['item_id']);request_sha=sha(json.dumps(r,sort_keys=True).encode())
            if sid in existing:
                if existing[sid]['request_sha256']!=request_sha:raise SystemExit('Cached response task changed')
                continue
            payload={'model':r['model'],'messages':r['messages'],**r['generation']}
            http=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            result={'panel':r['panel'],'family':r['family'],'model':r['model'],
                    'revision_expected':r['revision'],'item_id':sid,'style':r['style'],
                    'request_sha256':request_sha,'response_text':None,'status':'request_failed'}
            try:
                with urllib.request.urlopen(http,timeout=a.timeout) as ans:
                    content=json.loads(ans.read())
                result['response_text']=content['choices'][0]['message']['content']
                result['status']='received'
                result['finish_reason']=content['choices'][0].get('finish_reason')
            except Exception as error:
                result['error']=str(error)[:400]
            fp.write(json.dumps(result,ensure_ascii=False)+'\n');fp.flush()
            if (k+1)%100==0:print('completed responses',k+1,'/',len(rows),flush=True)
    print('Output',a.output)
if __name__=='__main__':main()
