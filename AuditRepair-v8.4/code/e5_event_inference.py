#!/usr/bin/env python3
"""Seed-level rare-event and signed-tail inference from completed E5 pairs.

One Bernoulli observation per preprocessing seed: whether ANY of its ten
calibration repetitions had the event. This avoids treating nested PAC draws
as independent seed replicates. Does not infer across new annotation panels.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import beta

def cp_bounds(k,n,alpha=.05):
    if n<=0:return (None,None)
    lower=0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    upper=1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lower,upper

def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,default=root/'results/E5/e5_paired_rows.csv')
    p.add_argument('--out',type=Path,default=root/'results/E5/e5_seed_event_uncertainty.csv')
    a=p.parse_args();df=pd.read_csv(a.rows);out=[]
    for (pool,gen,score),part in df.groupby(['pool','generator','score'],sort=True):
        states=[];n=int(part.n.iloc[0]);thr=.01*n
        for seed,block in part.groupby('seed'):
            if len(block)!=10:raise AssertionError('Incomplete nested calibration')
            active=int(block.accepted_state.max());d=block.delta_joint
            states.append({'accepted':active,'any_change':int((d!=0).any()),
                           'large_save':int((d>=thr).any()),
                           'large_cost_increase':int((d<=-thr).any()),
                           'mean_savings':float(d.mean())})
        s=pd.DataFrame(states)
        for groupname,subset in [('all_seeds',s),('accepted_only',s[s.accepted==1])]:
            for event in ('accepted','any_change','large_save','large_cost_increase'):
                if groupname=='accepted_only' and event=='accepted':continue
                k=int(subset[event].sum());nn=len(subset);lo,hi=cp_bounds(k,nn)
                out.append({'pool':pool,'generator':gen,'score':score,'denominator':groupname,
                            'event':event,'independent_seed_units':nn,'event_seeds':k,
                            'rate':k/nn if nn else np.nan,'CI95_low':lo,'CI95_high':hi,
                            'assumptions':'independent preprocessing seed randomizations conditional on fixed annotation pool; nested PAC repetitions counted as any-event within seed'})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(a.out,index=False)
    print('E5 event report',a.out,'rows',len(out))
if __name__=='__main__':main()
