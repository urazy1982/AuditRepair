#!/usr/bin/env python3
"""Prepare a versioned, gold-blind E7 response-contract amendment for Mistral/Phi."""
import argparse,hashlib,json,pathlib,copy,datetime
P=pathlib.Path

def digest(b):return hashlib.sha256(b).hexdigest()
def load_jsonl(path):return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=P,required=True);p.add_argument('--out',type=P,required=True);a=p.parse_args()
    root=a.repo.resolve(); orig=root/'AuditRepair-v8.4/E7_tasks'; manifest=json.loads((orig/'E7_GOLD_BLIND_MANIFEST.json').read_text());classes=manifest['classes']
    assert classes==['World','Sports','Business','Sci/Tech']
    a.out.mkdir(parents=True,exist_ok=True); tasks=a.out/'tasks';tasks.mkdir(exist_ok=True)
    existing=a.out/'E7_RC2_MANIFEST.json'
    if existing.exists():
        frozen=json.loads(existing.read_text(encoding='utf8'))
        assert frozen['parent_manifest_sha256']==digest((orig/'E7_GOLD_BLIND_MANIFEST.json').read_bytes()),'Parent changed since freeze'
        for name,meta in frozen['tasks'].items():
            assert (tasks/name).exists() and digest((tasks/name).read_bytes())==meta['sha256'], f'Frozen amendment drift: {name}'
        print('Existing E7-RC2 protocol retained without modification:',existing)
        return
    specs={}
    for family in ('mistral','phi'):
      for pool,n in [('ag_news_5000',5000),('ag_news_holdout',2600)]:
        source_name=f'{pool}_{family}_tasks.jsonl';src=orig/source_name; original=manifest['tasks'][source_name]
        assert digest(src.read_bytes())==original['sha256'],f'Original task drift: {source_name}'
        old=load_jsonl(src);assert len(old)==n and len({r['item_id'] for r in old})==n
        amended=[]
        for r in old:
          t=copy.deepcopy(r); t['amendment']='e7_response_contract_v2_choice';t['historical_style']=r['style']
          # Keep original text and original two prompt styles except the output contract.
          txt=t['messages'][-1]['content']
          if family=='mistral':
            # direct ends in 'Answer with ...'; reasoning ends with text and asks for final Answer line.
            txt += '\nOutput contract override: respond ONLY with one exact category name from World, Sports, Business, Sci/Tech. No Answer prefix, no explanation.'
          else:
            txt += '\nOutput contract override: respond ONLY with one exact category name from World, Sports, Business, Sci/Tech. No Answer prefix, no explanation.'
          t['messages'][-1]['content']=txt
          t['generation']={**r['generation'],'max_tokens':32,'structured_outputs':{'choice':classes}}
          amended.append(t)
        dst=tasks/source_name;write_jsonl(dst,amended)
        # A source-file hash belongs to the original tasks, making gold-isolation check versioned.
        specs[source_name]={'count':n,'sha256':digest(dst.read_bytes()),'original_sha256':original['sha256'],'source_file_sha256':original['source_file_sha256'],'family':family,'panel':pool}
    plan={'amendment_id':'E7-RC2','status':'prepared_gold_blind_not_annotated','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'parent_manifest_sha256':digest((orig/'E7_GOLD_BLIND_MANIFEST.json').read_bytes()),'classes':classes,
       'models':[x for x in manifest['models'] if x['family'] in ('mistral','phi')],
       'change':'Versioned full-family re-collection with constrained class-name-only decoding; original response files preserved',
       'parsing':'strip leading/trailing whitespace; exact match one of four class names; finish_reason stop only',
       'gold_not_used':True,'original_qwen_llama_retained':True,'new_request_count':15200,'tasks':specs}
    (a.out/'E7_RC2_MANIFEST.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared 15200 gold-blind requests in',a.out)
if __name__=='__main__':main()
