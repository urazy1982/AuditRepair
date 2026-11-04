"""Validate all completed experiment records and apply the frozen paired analysis.

One bootstrap index matrix is shared by all contrasts within each real pool.
This preserves joint pairing and gives identical intervals to identical effects.
No cells, seeds, or adverse outcomes are selected out after looking at results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_common import ROOT, atomic_json, file_hash, protocol


def read_phase(path,p,sha,phase):
    status=json.loads((path/'run_status.json').read_text())
    if status['status']!='complete' or not status.get('full_frozen_scope_complete'):
        raise ValueError(f'{phase} full frozen scope is not complete')
    records=[];checks=[]
    for checkpoint in sorted((path/'checkpoints').glob('*/*.json')):
        d=json.loads(checkpoint.read_text())
        if d['status']!='complete' or d['protocol_sha256']!=sha:
            raise ValueError(f'Invalid checkpoint: {checkpoint}')
        for name,value in d['code_sha256'].items():
            if file_hash(ROOT/name)!=value:
                raise ValueError(f'Code changed after experiment: {name}')
        identity=d.get('pool',d.get('case'))
        if checkpoint.parent.name!=identity or checkpoint.stem!=f"seed_{d['seed']:02d}":
            raise ValueError('Checkpoint file identity mismatch')
        for r in d['rows']:
            if r['pool']!=identity or r['seed']!=d['seed']:
                raise ValueError('Row identity mismatch')
            if not 0<=r['expert_total']<=r['n']:
                raise ValueError('Expert cost outside finite pool')
            if r['expert_total']!=r['pre_audits']+r['calibration_new']+r['threshold_new']:
                raise ValueError('Unique-cost partition mismatch')
            if bool(r['violated'])!=(r['error']>p['parameters']['epsilon']):
                raise ValueError('Violation indicator mismatch')
        records.extend(d['rows'])
        checks.append({'checkpoint':str(checkpoint.relative_to(path)),
                       'sha256':file_hash(checkpoint),'rows':len(d['rows'])})
    df=pd.DataFrame(records)
    if phase in ('E1','E2_E3'):
        if phase=='E1':
            from run_e1 import METHODS
            methods=set(METHODS)
        else:
            from run_factorial import GENERATORS,SCORES,ROLES
            methods={'pac_only'}|{f'{g}_{s}_{r}' for g in GENERATORS for s in SCORES for r in ROLES}
        key=['pool','seed','rep','method']
        expected={(pool,seed,rep,method) for pool in p['pools']
                  for seed in range(p['parameters']['seeds'])
                  for rep in range(p['parameters']['pac_reps']) for method in methods}
    else:
        from run_e4 import METHODS,SHARED_SOURCE_COUNTS,SLICE_FRACTIONS,case_name,_payload_hash
        key=['pool','seed','rep','eta','method']
        expected={(case_name(k,f),seed,rep,float(eta),method) for k in SHARED_SOURCE_COUNTS
                  for f in SLICE_FRACTIONS for seed in range(p['E4']['seeds'])
                  for rep in range(p['E4']['pac_reps']) for eta in p['E4']['eta'] for method in METHODS}
        for checkpoint in sorted((path/'checkpoints').glob('*/*.json')):
            d=json.loads(checkpoint.read_text())
            if d['payload_sha256']!=_payload_hash(d): raise ValueError('E4 payload checksum mismatch')
    got=set(df[key].itertuples(index=False,name=None))
    if got!=expected or len(df)!=len(expected):
        raise ValueError(f'{phase} missing or duplicated experimental cells')
    # Every method uses the identical draw set within a paired repetition.
    if df.groupby(['pool','seed','rep']).draws_sha256.nunique().max()!=1:
        raise ValueError('Paired calibration streams differ')
    if phase=='E1':
        wide=df.pivot(index=['pool','seed','rep'],columns='method',values='expert_total')
        if not (wide.O0_identity==wide.same_audit_ds).all(): raise ValueError('O0 identity mismatch')
        full=df[df.method=='O5_all_errors']
        if not (full.expert_total==full.minimum_fixed_cost).all() or not (full.error==0).all():
            raise ValueError('O5 did not attain fixed-query floor')
    return df,checks


def interval(values,indices):
    values=np.asarray(values,float)
    if not np.isfinite(values).all(): raise ValueError('Incomplete paired effect')
    lo,hi=np.quantile(values[indices].mean(1),[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def effects(df,p,indices,phase):
    output=[]
    for pool in p['pools']:
        part=df[df.pool==pool]
        means=part.groupby(['seed','method']).expert_total.mean().unstack('method').sort_index()
        n=int(part.n.iloc[0])
        if phase=='E1':
            comparisons=[(x,'same_audit_ds') for x in means if x not in ('same_audit_ds','pac_only')]
            comparisons.append(('same_audit_ds','pac_only'))
        else:
            comparisons=[]
            for g in ['G0_uniform','G1','G2']:
                for s in ['S0','S1']:
                    comparisons += [(f'{g}_{s}_repair',f'{g}_{s}_same_audit'),
                                    (f'{g}_{s}_oracle',f'{g}_{s}_same_audit'),
                                    (f'{g}_{s}_repair','pac_only'),
                                    (f'{g}_{s}_same_audit','pac_only')]
        for arm,ref in comparisons:
            v=(means[ref]-means[arm]).to_numpy() # positive values are savings
            mean,lo,hi=interval(v,indices[pool])
            output.append({'pool':pool,'n':n,'arm':arm,'reference':ref,
                           'arm_mean_cost':float(means[arm].mean()),'reference_mean_cost':float(means[ref].mean()),
                           'mean_reviews_saved':mean,'ci95_saved_low':lo,'ci95_saved_high':hi,
                           'saving_pct_of_pool':100*mean/n,'ci95_pct_low':100*lo/n,'ci95_pct_high':100*hi/n,
                           'preprocessing_seeds':len(v),'pac_reps_per_seed':p['parameters']['pac_reps'],
                           'point_estimate_meets_1pct':mean>=.01*n,
                           'lower_bound_meets_1pct':lo>=.01*n,
                           'ci_rules_out_1pct_saving':hi<.01*n,
                           'interval_scope':'conditional on fixed panel; paired preprocessing seed bootstrap'})
    return pd.DataFrame(output)


def fixed_factorial_contrasts(df,p,indices):
    records=[]
    for pool in p['pools']:
        part=df[(df.pool==pool)&(df.role=='repair')].copy()
        part['cell']=part.generator+'_'+part.score
        w=part.groupby(['seed','cell']).paired_cost_delta.mean().unstack('cell').sort_index()
        contrasts={
            'primary_generator_G1_S0_vs_G0_uniform_S0':w['G1_S0']-w['G0_uniform_S0'],
            'primary_score_G0_uniform_S1_vs_S0':w['G0_uniform_S1']-w['G0_uniform_S0'],
            'secondary_G1_by_score_interaction':(w['G1_S1']-w['G1_S0'])-(w['G0_uniform_S1']-w['G0_uniform_S0']),
        }
        for name,v in contrasts.items():
            mean,lo,hi=interval(v.to_numpy(),indices[pool])
            records.append({'pool':pool,'contrast':name,'difference_in_incremental_cost':mean,
                            'ci95_low':lo,'ci95_high':hi,'negative_favors':'first named intervention'})
    return pd.DataFrame(records)


def reach_summary(df):
    records=[]
    for (pool,method),d in df.groupby(['pool','method'],sort=True):
        if 'corrected' not in d: continue
        corr=d.corrected.fillna(0).astype(float)
        count=int((corr>0).sum())
        frac=d.corrected_le/corr.replace(0,np.nan)
        records.append({'pool':pool,'method':method,'runs':len(d),
                        'runs_with_corrections':count,'runs_without_corrections':len(d)-count,
                        'mean_corrected':float(corr.mean()),
                        'mean_introduced_errors':float(d.introduced_errors.fillna(0).mean()),
                        'mean_share_of_corrections_le_when_defined':float(frac.mean()) if count else None,
                        'pooled_share_of_corrections_le':float(d.corrected_le.sum()/corr.sum()) if corr.sum()>0 else None,
                        'mean_share_changed_le_when_defined':float(d.share_changed_le.mean()) if d.share_changed_le.notna().any() else None,
                        'reach_region':'U<=baseline first-failing cutoff; includes reviewed boundary',
                        'mean_retained_automatic_corrections':float(d.retained_automatic_corrections.mean()) if 'retained_automatic_corrections' in d else None})
    return pd.DataFrame(records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--out',type=Path,default=ROOT/'analysis')
    args=ap.parse_args();p,sha=protocol()
    args.out.mkdir(parents=True,exist_ok=True)
    e1,check1=read_phase(args.results/'E1',p,sha,'E1')
    fact,check2=read_phase(args.results/'E2_E3',p,sha,'E2_E3')
    e4,check4=read_phase(args.results/'E4',p,sha,'E4')
    # The unchanged PAC-only arm also pairs across the two real-data experiments.
    index=['pool','seed','rep']
    b1=e1[e1.method=='pac_only'].set_index(index).sort_index()
    b2=fact[fact.method=='pac_only'].set_index(index).sort_index()
    for metric in ['expert_total','pre_audits','calibration_new','threshold_new','error','violated']:
        np.testing.assert_array_equal(b1[metric].to_numpy(),b2[metric].to_numpy())
    rng=np.random.default_rng(p['parameters']['bootstrap_seed'])
    n=p['parameters']['seeds'];b=p['parameters']['bootstrap_reps']
    indices={pool:rng.integers(0,n,size=(b,n)) for pool in p['pools']}
    effect1=effects(e1,p,indices,'E1');effect2=effects(fact,p,indices,'E2_E3')
    effect1.to_csv(args.out/'E1_effects.csv',index=False)
    effect2.to_csv(args.out/'E2_E3_effects.csv',index=False)
    fixed_factorial_contrasts(fact,p,indices).to_csv(args.out/'E2_E3_primary_contrasts.csv',index=False)
    reach_summary(e1).to_csv(args.out/'E1_reach.csv',index=False)
    reach_summary(fact).to_csv(args.out/'E2_E3_reach.csv',index=False)
    for phase,df in [('E1',e1),('E2_E3',fact),('E4',e4)]:
        d=df.groupby(['pool','method']).agg(records=('violated','size'),
                                            preprocessing_seeds=('seed','nunique'),
                                            observed_final_error_violations=('violated','sum'),
                                            mean_final_error=('error','mean'),
                                            maximum_final_error=('error','max'))
        d.to_csv(args.out/f'{phase}_observed_error.csv')
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha,
              'analysis_code_sha256':file_hash(Path(__file__)),
              'status':'complete_and_integrity_checked',
              'rows':{'E1':len(e1),'E2_E3':len(fact),'E4':len(e4)},
              'real_pool_items':int(e1.groupby('pool').n.first().sum()),
              'checks':{'all_expected_cells':True,'current_code_hashes_match':True,
                        'unique_cost_partition':True,'paired_calibration_hashes':True,
                        'PAC_only_identical_across_real_experiments':True,
                        'O0_identity':True,'O5_cost_floor':True},
              'bootstrap':'4000 resamples of20 paired preprocessing seed means; one shared matrix per pool across all arms and contrasts; pointwise intervals, not simultaneous winner intervals.',
              'bootstrap_matrix_sha256':{pool:__import__('hashlib').sha256(a.tobytes()).hexdigest() for pool,a in indices.items()},
              'source_checkpoints':{'E1':check1,'E2_E3':check2,'E4':check4},
              'note':'Phase-level initial summaries retain their original bootstrap Monte Carlo draws. This unified analysis is authoritative for real-pool intervals and jointly retains arm pairing. E4 paired/reach intervals already use shared seed matrices and remain in results/E4.'}
    atomic_json(args.out/'analysis_manifest.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['source_checkpoints','bootstrap_matrix_sha256']},indent=2))


if __name__=='__main__': main()
