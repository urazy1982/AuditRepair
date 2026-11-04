#!/usr/bin/env python3
"""Fail-closed, restartable constrained-choice E7 collector (stdlib only)."""
import argparse,hashlib,json,pathlib,urllib.request,urllib.error,time,os
P=pathlib.Path

def digest(b):return hashlib.sha256(b).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def request_hash(task):return digest(json.dumps(task,sort_keys=True).encode())
def exact_response(record,choices):
    t=record.get('response_text')
    return record.get('status')=='received' and record.get('finish_reason')=='stop' and isinstance(t,str) and t.strip() in choices

def api(base,endpoint,payload=None,timeout=180):
    url=base.rstrip('/')+'/'+endpoint.lstrip('/')
    req=urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as res:return json.loads(res.read())
def predict(base,task,timeout):
    payload={'model':task['model'],'messages':task['messages'],**task['generation']}
    response=api(base,'chat/completions',payload,timeout)
    c=response['choices'][0]
    return c['message'].get('content'),c.get('finish_reason')
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=P,required=True);p.add_argument('--tasks',type=P,required=True);p.add_argument('--output',type=P,required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());name=a.tasks.name;spec=manifest['tasks'][name];assert digest(a.tasks.read_bytes())==spec['sha256'],'Task hash drift'
    original=rows(a.tasks);assert len(original)==spec['count'];choices=set(manifest['classes']);ids=[str(x['item_id']) for x in original];assert len(ids)==len(set(ids))
    model=original[0]['model'];server=api(a.base_url,'models',timeout=a.timeout)
    assert model in [x['id'] for x in server['data']],f'Model ID mismatch: {model}'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    existing={}
    if a.output.exists():
      for record in rows(a.output):
        sid=str(record['item_id']);assert sid not in existing,'Duplicate cached record';existing[sid]=record
    for task in original:
        sid=str(task['item_id']);cached=existing.get(sid)
        if cached:assert cached['request_sha256']==request_hash(task),f'Existing task mismatch at {sid}'
    # Probe *real* endpoint before generating many rows. The structured-output field is mandatory.
    probe_task=original[0];probe_text,probe_finish=predict(a.base_url,probe_task,a.timeout)
    assert probe_finish=='stop' and isinstance(probe_text,str) and probe_text.strip() in choices,(
        f'Constrained choice preflight failed: finish={probe_finish} content={probe_text!r}; do NOT fall back to unconstrained output')
    print('Constrained-choice preflight PASS',model,repr(probe_text),flush=True)
    succeeded=sum(exact_response(r,choices) for r in existing.values())
    with a.output.open('a',encoding='utf8') as fout:
      for i,task in enumerate(original,1):
        sid=str(task['item_id']);cached=existing.get(sid)
        if cached:
          # A cached failed row is deliberately not retried in-place; use a fresh amendment run if it exceeds the gate.
          continue
        rec={'item_id':sid,'panel':task['panel'],'family':task['family'],'model':task['model'],'revision_expected':task['revision'],
            'style':task['style'],'amendment':task['amendment'],'request_sha256':request_hash(task),'status':'request_failed',
            'response_text':None,'finish_reason':None}
        try:
          text,finish=predict(a.base_url,task,a.timeout)
          rec.update(status='received',response_text=text,finish_reason=finish)
        except Exception as e:rec['error']=str(e)[:300]
        fout.write(json.dumps(rec,ensure_ascii=False)+'\n');fout.flush()
        if i%100==0:print('Processed',i,'/',len(original),flush=True)
    output=rows(a.output);assert len(output)==len(original),'Output incomplete'
    passed=sum(exact_response(r,choices) for r in output)
    print('FINISHED',name,'valid',passed,'/',len(output),'rate',passed/len(output))
if __name__=='__main__':main()
