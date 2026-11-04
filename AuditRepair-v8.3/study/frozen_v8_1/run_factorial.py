"""Run the amended E2/E3 common-warm-up factorial on all seven cached pools.

python run_factorial.py --workers 3
The six cells use identical score vectors for repaired and no-repair arms.
The G0 reference here uses uniform warm-up and is explicitly distinct from E1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from audit_repair.core import dawid_skene
from audit_repair.pac import pac_labels
from experiment_common import (ROOT, array_hash, atomic_json, changed_diagnostics,
                               check_pac_accounting, file_hash, load_pool, protocol,
                               streams, summarize_rows)
from factorial import build_features, preprocess_generator

GENERATORS = ('G0_uniform','G1','G2')
SCORES = ('S0','S1')
ROLES = ('same_audit','repair','oracle')
CODE_FILES = ['factorial.py','run_factorial.py','experiment_common.py',
              'audit_repair/core.py','audit_repair/pac.py']


def feature_cache(pool):
    p,sha = protocol(validate=False)
    out = ROOT/'feature_cache'
    out.mkdir(exist_ok=True)
    target,meta = out/f'{pool}.npz',out/f'{pool}.json'
    code_hash = file_hash(ROOT/'factorial.py')
    if target.exists() and meta.exists():
        info=json.loads(meta.read_text())
        if info['protocol_sha256']!=sha or info['code_sha256']!=code_hash or info['cache_sha256']!=file_hash(target):
            raise ValueError('Feature cache differs from frozen protocol/code')
        return target,info
    df,_,_,_ = load_pool(pool,with_text=True)
    start=time.monotonic()
    x,neighbors=build_features(df.text.tolist(),df.id.tolist())
    tmp=out/f'{pool}.tmp.npz'
    np.savez_compressed(tmp,features=x,neighbors=neighbors)
    tmp.replace(target)
    info={'pool':pool,'protocol_sha256':sha,'code_sha256':code_hash,
          'features_sha256':array_hash(x),'neighbors_sha256':array_hash(neighbors),
          'cache_sha256':file_hash(target),'n':len(df),'dimensions':x.shape[1],
          'neighbor_count':neighbors.shape[1],'wall_seconds':time.monotonic()-start,
          'created_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(meta,info)
    print(f"features {pool}: {x.shape}, {info['wall_seconds']:.1f}s",flush=True)
    return target,info


def retained_changes(pred,base,gold,U,info,audits,seed,m):
    draws=np.random.default_rng(seed).integers(0,len(gold),m)
    mask=(pred!=base)&(U<info['u_hat'])
    mask[np.unique(draws)]=False
    if audits: mask[list(audits)]=False
    return {'retained_automatic_changes':int(mask.sum()),
            'retained_automatic_corrections':int((mask&(pred==gold)&(base!=gold)).sum()),
            'retained_automatic_new_errors':int((mask&(pred!=gold)&(base==gold)).sum())}


def one_seed(pool,seed,out_dir):
    started=time.monotonic()
    p,sha=protocol(validate=False);a=p['parameters']
    code={f:file_hash(ROOT/f) for f in CODE_FILES}
    feature_path=ROOT/'feature_cache'/f'{pool}.npz'
    feature_info=json.loads((ROOT/'feature_cache'/f'{pool}.json').read_text())
    if (feature_info['protocol_sha256']!=sha or
            feature_info['code_sha256']!=code['factorial.py'] or
            feature_info['cache_sha256']!=file_hash(feature_path)):
        raise ValueError('Worker feature cache failed frozen provenance check')
    target=Path(out_dir)/'checkpoints'/pool/f'seed_{seed:02d}.json'
    if target.exists():
        cached=json.loads(target.read_text())
        if cached['protocol_sha256']!=sha or cached['code_sha256']!=code:
            raise ValueError('Checkpoint protocol or implementation changed')
        expected_grid={(r,'pac_only') for r in range(a['pac_reps'])}
        expected_grid|={(r,f'{g}_{s}_{role}') for r in range(a['pac_reps']) for g in GENERATORS for s in SCORES for role in ROLES}
        observed_grid={(row['rep'],row['method']) for row in cached['rows']}
        if (cached['status']!='complete' or cached['pool']!=pool or cached['seed']!=seed or
                len(cached['rows'])!=19*a['pac_reps'] or observed_grid!=expected_grid or
                any(row['pool']!=pool or row['seed']!=seed for row in cached['rows']) or
                cached.get('feature_cache_sha256')!=feature_info['cache_sha256']):
            raise ValueError('Incomplete factorial checkpoint')
        return cached,True
    _,L0,gold0,K=load_pool(pool)
    cache=np.load(feature_path)
    X0,neighbors0=cache['features'],cache['neighbors']
    data_seed,pre_seed,pac_streams=streams(a['root_seed'],seed,a['pac_reps'])
    perm=np.random.default_rng(data_seed).permutation(len(gold0))
    inverse=np.empty_like(perm);inverse[perm]=np.arange(len(perm))
    L,gold,X=L0[perm],gold0[perm],X0[perm]
    neighbors=inverse[neighbors0[perm]]
    if (neighbors==np.arange(len(perm))[:,None]).any(): raise AssertionError('Neighbor permutation introduced self')
    states={g:preprocess_generator(L,K,gold,X,neighbors,g,pre_seed) for g in GENERATORS}
    warm=states['G0_uniform']['warm_indices']
    q=states['G0_uniform']['text_probabilities']
    for state in states.values():
        np.testing.assert_array_equal(state['warm_indices'],warm)
        np.testing.assert_allclose(state['text_probabilities'],q,atol=0,rtol=0)
        if len(state['audits'])>a['B']: raise AssertionError('Audit budget exceeded')
    post0=dawid_skene(L,K)
    pred0=post0.argmax(1);u0=1-post0.max(1)
    frozen={}
    for g,state in states.items():
        frozen[g]={'base':array_hash(state['base']),'repair':array_hash(state['pred']),
                   'oracle':array_hash(state['oracle_candidate_labels']),
                   'scores':{s:array_hash(state['scores'][s]) for s in SCORES},
                   'audits':array_hash(np.array(sorted(state['audits']),dtype=np.int64)),
                   'warmup':array_hash(warm)}
    frozen_utc=datetime.now(timezone.utc).isoformat()
    rows=[]
    for rep,stream in enumerate(pac_streams):
        _,info=pac_labels(pred0,u0,gold,eps=a['epsilon'],alpha=a['alpha_final'],m=a['m'],seed=stream)
        accounting=check_pac_accounting(info,u0,{},len(gold),a['m'],stream)
        rows.append({'pool':pool,'n':len(gold),'seed':seed,'rep':rep,'method':'pac_only',
                     'generator':'none','score':'S0','role':'pac_only','pre_audits':0,
                     'expert_total':info['expert_new'],**info,**accounting,'violated':info['error']>a['epsilon']})
        for g,state in states.items():
            aud=state['audits'];base=state['base'];diag=state['diagnostics']
            preds={'same_audit':base,'repair':state['pred'],'oracle':state['oracle_candidate_labels']}
            for score in SCORES:
                U=state['scores'][score]
                _,base_info=pac_labels(base,U,gold,eps=a['epsilon'],alpha=a['alpha_final'],m=a['m'],already=aud,seed=stream)
                for role in ROLES:
                    pred=preds[role]
                    if role=='same_audit': inf=base_info
                    else:
                        _,inf=pac_labels(pred,U,gold,eps=a['epsilon'],alpha=a['alpha_final'],m=a['m'],already=aud,seed=stream)
                    accounting=check_pac_accounting(inf,U,aud,len(gold),a['m'],stream)
                    row={'pool':pool,'n':len(gold),'seed':seed,'rep':rep,
                         'method':f'{g}_{score}_{role}','generator':g,'score':score,'role':role,
                         'pre_audits':len(aud),'expert_total':len(aud)+inf['expert_new'],
                         'paired_cost_delta':inf['expert_new']-base_info['expert_new'],
                         'baseline_u_hat':base_info['u_hat'],**inf,**accounting,
                         'violated':inf['error']>a['epsilon'],
                         'groups_certified':diag['certified_groups'],
                         'harmful_vs_frozen_ref':diag['harmful_accepted_vs_ref'],
                         'oracle_beneficial_groups':diag['oracle_beneficial_groups_vs_same_audit'],
                         **changed_diagnostics(pred,base,gold,U,base_info['u_hat'],aud),
                         **retained_changes(pred,base,gold,U,inf,aud,stream,a['m'])}
                    rows.append(row)
    audits={g:{'original_row_ids':[int(perm[i]) for i in sorted(st['audits'])],
                'warmup_original_row_ids':[int(perm[i]) for i in st['warm_indices']],
                'group_log':st['group_log'],'groups':st['groups'],
                'classifier_info':st['classifier_info'],'diagnostics':st['diagnostics']}
            for g,st in states.items()}
    answer={'status':'complete','pool':pool,'seed':seed,'protocol_sha256':sha,'code_sha256':code,
            'feature_cache_sha256':feature_info['cache_sha256'],
            'frozen_arrays_utc':frozen_utc,'frozen_arrays':frozen,'permutation_sha256':array_hash(perm),
            'preprocessing':audits,'rows':rows,'wall_seconds':time.monotonic()-started}
    atomic_json(target,answer)
    return answer,False


def primary_contrasts(rows,out):
    df=pd.DataFrame(rows)
    p,_=protocol(validate=False);a=p['parameters']
    rng=np.random.default_rng(a['bootstrap_seed'])
    result=[]
    for pool,part in df[df.role=='repair'].groupby('pool',sort=True):
        part=part.copy();part['cell']=part.generator+'_'+part.score
        wide=part.pivot(index=['seed','rep'],columns='cell',values='paired_cost_delta')
        candidates={
            'primary_generator_G1_S0_vs_G0_uniform_S0':wide['G1_S0']-wide['G0_uniform_S0'],
            'primary_score_G0_uniform_S1_vs_S0':wide['G0_uniform_S1']-wide['G0_uniform_S0'],
            'secondary_G1_by_S1_interaction':(wide['G1_S1']-wide['G1_S0'])-(wide['G0_uniform_S1']-wide['G0_uniform_S0']),
        }
        for name,d in candidates.items():
            vals=d.groupby(level='seed').mean().to_numpy()
            b=rng.choice(vals,size=(a['bootstrap_reps'],len(vals)),replace=True).mean(1)
            lo,hi=np.quantile(b,[.025,.975])
            result.append({'pool':pool,'contrast':name,'seeds':len(vals),
                           'mean_difference_in_incremental_repair_cost':float(vals.mean()),
                           'ci95_low':float(lo),'ci95_high':float(hi),
                           'negative_favors':'first named intervention',
                           'scope':'fixed-pool algorithmic variability; no selection of best cell'})
    pd.DataFrame(result).to_csv(Path(out)/'prespecified_contrasts.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--out',type=Path,default=ROOT/'results/E2_E3')
    ap.add_argument('--pools',nargs='+')
    ap.add_argument('--seed-start',type=int,default=0)
    ap.add_argument('--seed-stop',type=int)
    args=ap.parse_args();p,sha=protocol();a=p['parameters']
    selected=args.pools or p['pools'];stop=args.seed_stop if args.seed_stop is not None else a['seeds']
    if not set(selected)<=set(p['pools']) or not 0<=args.seed_start<stop<=a['seeds']:
        raise ValueError('Requested tasks outside frozen scope')
    args.out.mkdir(parents=True,exist_ok=True)
    for pool in selected: feature_cache(pool)
    run={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha,
         'selected_pools':selected,'seed_range':[args.seed_start,stop],
         'expected_tasks':len(selected)*(stop-args.seed_start),'completed_tasks':0}
    atomic_json(args.out/'run_status.json',run)
    rows=[];pre=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fs={ex.submit(one_seed,pool,seed,args.out):(pool,seed) for pool in selected for seed in range(args.seed_start,stop)}
            for f in as_completed(fs):
                answer,cached=f.result();rows.extend(answer['rows'])
                for g,st in answer['preprocessing'].items():
                    pre.append({'pool':answer['pool'],'seed':answer['seed'],'generator':g,**st['diagnostics']})
                run['completed_tasks']+=1;atomic_json(args.out/'run_status.json',run)
                print(f"E2/E3 {run['completed_tasks']}/{run['expected_tasks']} {answer['pool']} seed={answer['seed']} {'cached' if cached else 'completed'} {answer['wall_seconds']:.1f}s",flush=True)
        rows.sort(key=lambda r:(p['pools'].index(r['pool']),r['seed'],r['rep'],r['method']))
        pairs=[]
        for g in GENERATORS:
            for s in SCORES:
                pairs.extend([(f'{g}_{s}_repair',f'{g}_{s}_same_audit'),(f'{g}_{s}_oracle',f'{g}_{s}_same_audit'),(f'{g}_{s}_same_audit','pac_only')])
        summarize_rows(rows,args.out,pairs)
        pd.DataFrame(pre).sort_values(['pool','seed','generator']).to_csv(args.out/'preprocessing.csv',index=False)
        primary_contrasts(rows,args.out)
        run.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),
                   full_frozen_scope_complete=set(selected)==set(p['pools']) and args.seed_start==0 and stop==a['seeds'])
        atomic_json(args.out/'run_status.json',run)
    except Exception as error:
        run.update(status='failed',error=repr(error));atomic_json(args.out/'run_status.json',run);raise


if __name__=='__main__': main()
