#!/usr/bin/env python3
"""Gold-blind worklist hash, response coverage and request integrity check."""
import argparse,hashlib,json
from pathlib import Path

def digest(b):return hashlib.sha256(b).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--responses',type=Path);a=p.parse_args()
m=json.loads((a.tasks/'E7_GOLD_BLIND_MANIFEST.json').read_text())
assert a.family in {'qwen','llama','mistral','phi'}
for panel in ('ag_news_5000','ag_news_holdout'):
 name=f'{panel}_{a.family}_tasks.jsonl'
 f=a.tasks/name; assert digest(f.read_bytes())==m['tasks'][name]['sha256'],f'Task hash mismatch: {name}'
 tasks=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
 assert len(tasks)==m['tasks'][name]['n']
 assert len({r['item_id'] for r in tasks})==len(tasks)
 assert all(not ({'gold','label','target','src_a_direct','src_b_direct'}&set(r)) for r in tasks)
 if a.responses:
  resp=a.responses/f'{panel}_{a.family}_responses.jsonl'
  assert resp.exists(),f'Missing {resp}'
  rows=[json.loads(s) for s in resp.read_text().splitlines() if s.strip()]
  indexed={r['item_id']:r for r in rows}
  assert len(indexed)==len(rows)==len(tasks),f'Incomplete/duplicates {name}: {len(rows)}/{len(tasks)}'
  for r in tasks:
   got=indexed[r['item_id']];expected=digest(json.dumps(r,sort_keys=True).encode())
   assert got['request_sha256']==expected
  failed=sum(r.get('status')!='received' for r in rows)
  print(panel,a.family,'complete',len(rows),'HTTP failures',failed)
 else:print(panel,a.family,'worklist hash and gold isolation PASS',len(tasks))
