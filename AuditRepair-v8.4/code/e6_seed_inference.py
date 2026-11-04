#!/usr/bin/env python3
"""E6 seed-clustered inference (synthetic populations re-generated each seed).

Within a cell policies share synthetic population and PAC streams, so policy
comparisons are matched by preprocessing seed. Five PAC repetitions nested in
one seed are never counted as five independent populations.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import beta

def cp(k,n,alpha=.05):
    if n==0:return (None,None)
    return (0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)))

def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=root/'results/E6_full')
    args=p.parse_args();d=pd.read_csv(args.results/'e6_rows.csv');state=pd.read_csv(args.results/'e6_states.csv')
    rng=np.random.default_rng(20261104);rowout=[];pairs=[]
    if d.groupby(['cell_id','policy','seed']).size().ne(5).any():raise SystemExit('Incomplete nested PAC repetition grid')
    for (cid,policy),part in d.groupby(['cell_id','policy'],sort=True):
        by=part.groupby('seed')
        means=by.saved.mean().sort_index()
        vals=means.to_numpy();n=len(vals)
        b=vals[rng.integers(0,n,(4000,n))].mean(axis=1)
        sl=state[(state.cell_id==cid)&(state.policy==policy)]
        if len(sl)!=n:raise AssertionError('Missing seed-state')
        events={'accepted_group':int((sl.certified>0).sum()),
            'any_cost_change':int(by.saved.apply(lambda x:(x!=0).any()).sum()),
            'large_saving':int(by.saved.apply(lambda x:(x>=50).any()).sum()),
            'large_cost_increase':int(by.saved.apply(lambda x:(x<=-50).any()).sum())}
        vals_ci={name:{'k':k,'ci':cp(k,n)} for name,k in events.items()}
        row={'cell_id':cid,'policy':policy,'size':int(part['size'].iloc[0]),'gain':float(part.gain.iloc[0]),
             'position':part.position.iloc[0],'shortlist':int(part.groups.iloc[0]),
             'independent_seeds':n,'nested_PAC_reps':5,
             'mean_savings':float(vals.mean()),'seed_cluster_CI95_low':float(np.quantile(b,.025)),
             'seed_cluster_CI95_high':float(np.quantile(b,.975)),
             'mean_group_audits':float(sl.group_audits.mean()),
             'mean_failed_group_audits':float(sl.failed_group_audits.mean()),
             'mean_certified_groups':float(sl.certified.mean()),
             'mean_positive_available_groups':float(sl.positive_candidates.mean()),
             'mean_changed_items':float(sl.changed_items.mean())}
        for name,o in vals_ci.items():
            row[f'{name}_event_seeds']=o['k']
            row[f'{name}_rate']=o['k']/n
            row[f'{name}_exact95_low']=o['ci'][0]
            row[f'{name}_exact95_high']=o['ci'][1]
        rowout.append(row)
    out=pd.DataFrame(rowout)
    out.to_csv(args.results/'e6_seedcluster_inference.csv',index=False)
    for cid,part in d.groupby('cell_id'):
        wide=part.pivot_table(index=['seed','rep'],columns='policy',values='saved')
        seedwide=wide.groupby(level='seed').mean()
        if seedwide.isna().any().any():raise AssertionError('Missing matched policy')
        for name,left,right in [('cost_aware_vs_size','cost_aware_no_gold','size'),('random_vs_size','random','size')]:
            differences=(seedwide[left]-seedwide[right]).to_numpy()
            n=len(differences); b=differences[rng.integers(0,n,(4000,n))].mean(axis=1)
            pairs.append({'cell_id':cid,'contrast':name,'seeds':n,'mean_paired_difference_in_savings':float(differences.mean()),
                          'cluster_bootstrap95_low':float(np.quantile(b,.025)),
                          'cluster_bootstrap95_high':float(np.quantile(b,.975))})
    pd.DataFrame(pairs).to_csv(args.results/'e6_policy_contrasts.csv',index=False)
    manifest={'status':'complete','cells':int(d.cell_id.nunique()),'policies':sorted(d.policy.unique().tolist()),
              'seeds_per_cell_policy':int(d.groupby(['cell_id','policy']).seed.nunique().min()),
              'analysis':'full-seed paired bootstrap, exact beta-binomial event intervals on any-event-per-seed; fixed synthetic-generating distribution'}
    (args.results/'e6_inference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
