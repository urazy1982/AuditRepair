#!/usr/bin/env python3
"""Create gold-blind four-model AG News annotation worklists for an external RTX 5090.

Does not connect to GPU, read model secrets, or call a model. Gold and historical
responses are deliberately omitted. The published class order is kept fixed.
"""
from pathlib import Path
import argparse, hashlib, json, os
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ.get('V83_REFERENCE',str(ROOT.parent/'AuditRepair-v8.3')))
MODELS=[
 ('qwen','Qwen/Qwen2.5-7B-Instruct-AWQ','b25037543e9394b818fdfca67ab2a00ecc7dd641'),
 ('llama','hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4','db1f81ad4b8c7e39777509fac66c652eb0a52f91'),
 ('mistral','mistralai/Mistral-7B-Instruct-v0.3','c170c708c41dac9275d15a8fff4eca08d52bab71'),
 ('phi','microsoft/Phi-3.5-mini-instruct','2fe192450127e6a83f7441aef6e3ca586c338b77')]

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,default=REF);ap.add_argument('--out',type=Path,default=ROOT/'E7_tasks')
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    classes=json.loads((args.reference/'study/frozen_v8_1/data/ag_news_5000.csv.meta.json').read_text())['classes']
    if classes!=['World','Sports','Business','Sci/Tech']:raise SystemExit('Pinned class order changed')
    manifest={'status':'prepared_not_annotated','dataset':'fancyzhx/ag_news','classes':classes,
        'source':'same fixed 5000 development and 2600 holdout item IDs; excludes gold and old model responses',
        'prompt_scheme':'per item and family: two direct and two reasoning requests; each family alternates style over items',
        'models':[{'family':f,'model_id':m,'revision':rev} for f,m,rev in MODELS],
        'tasks':{},'quality_gate':'each family <=5% invalid/failed on each panel, before gold joins',
        'model_revision_verification':'operator must independently confirm server loaded the pinned revision; /models ID alone does not establish commit',
        'existing_holdout_status':'previously inspected in historical study; new model annotations are independent collection, not blind confirmation of prior results'}
    for pool in ('ag_news_5000','ag_news_holdout'):
        df=pd.read_csv(args.reference/f'study/frozen_v8_1/data/{pool}.csv',dtype={'id':str})
        if df['id'].duplicated().any() or df.text.isna().any():raise AssertionError('Invalid input ID/text')
        for j,(fam,model,revision) in enumerate(MODELS):
            path=args.out/f'{pool}_{fam}_tasks.jsonl'
            with path.open('w') as fp:
                for k,row in df.iterrows():
                    style='direct' if (k+j)%2==0 else 'reason'
                    text=str(row['text'])[:2000]
                    if style=='direct':
                        prompt=(f'Classify the text into exactly one of these categories: {", ".join(classes)}. '
                                f'Text: {text} Answer with the category name only, nothing else.')
                    else:
                        prompt=(f'You are an expert annotator. Possible categories: {", ".join(classes)}. '
                                f'Read the text, think about which category fits best, then end your reply with '
                                f'a final line of exactly the form "Answer: <category>". Text: {text}')
                    record={'panel':pool,'item_id':str(row['id']),'family':fam,'style':style,
                            'model':model,'revision':revision,'messages':[{'role':'user','content':prompt}],
                            'generation':{'temperature':0,'seed':0,'max_tokens':16 if style=='direct' else 512}}
                    # Absolute prohibition on attaching gold or historical model labels.
                    assert not ({'gold','src_a_direct','src_a_reason','src_b_direct','src_b_reason'} & set(record))
                    fp.write(json.dumps(record,ensure_ascii=False)+'\n')
            manifest['tasks'][path.name]={'n':len(df),'sha256':sha(path.read_bytes()),'source_file_sha256':sha((args.reference/f'study/frozen_v8_1/data/{pool}.csv').read_bytes())}
    (args.out/'E7_GOLD_BLIND_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Prepared',len(manifest['tasks']),'gold-blind worklists with',sum(v['n'] for v in manifest['tasks'].values()),'requests')
if __name__=='__main__':main()
