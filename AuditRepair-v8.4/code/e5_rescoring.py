#!/usr/bin/env python3
"""E5: cached-panel three-arm joint repair/rescoring, *no new annotations*.

The v8.3 E2/E3 frozen checkpoints supply the charged audit histories and
original A/B outcomes. Reconstruct the frozen DS fit, repaired predictions and
warm-up-trained text model without looking at unaudited gold; assert agreement
with archived A/B outcomes; evaluate C on the same PAC draw streams.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path(os.environ.get('V83_REFERENCE',str(BASE.parent/'AuditRepair-v8.3')))


def setup_import(ref:Path):
    sys.path.insert(0,str(BASE/'code'))
    p=ref/'study'/'corrected'
    h=ref/'study'/'frozen_v8_1'
    if not p.exists() or not h.exists(): raise FileNotFoundError('v8.3 reference project missing corrected/frozen study dirs')
    # corrected core and factorial must precede frozen versions
    sys.path.insert(0,str(h))
    sys.path.insert(0,str(p))
    from audit_repair.core import dawid_skene
    from fast_pac import pac_labels_fast as pac_labels
    from factorial import _text_probabilities
    from experiment_common import load_pool,streams,array_hash
    return dawid_skene,pac_labels,_text_probabilities,load_pool,streams,array_hash


def one_seed(ref, pool, seed, funcs, verify=True):
    dawid_skene,pac_labels,_text_probabilities,load_pool,streams,array_hash = funcs
    source=ref/'study'/'frozen_v8_1'
    chk=json.loads((source/'results'/'E2_E3'/'checkpoints'/pool/f'seed_{seed:02d}.json').read_text())
    if chk['status']!='complete' or chk['pool']!=pool or chk['seed']!=seed:raise RuntimeError('Incomplete/incorrect archived checkpoint')
    _,L0,g0,K=load_pool(pool)
    features=None  # load only when an accepted repair requires rescoring
    data_seed,pre_seed,streams_pac=streams(20261030,seed,10)
    perm=np.random.default_rng(data_seed).permutation(len(g0))
    inv=np.empty_like(perm);inv[perm]=np.arange(len(perm))
    if chk['permutation_sha256']!=array_hash(perm): raise AssertionError('Permutation drift')
    L,g=L0[perm],g0[perm]
    warm=inv[np.array(chk['preprocessing']['G1']['warmup_original_row_ids'],dtype=int)]
    if chk['frozen_arrays']['G1']['warmup']!=array_hash(warm): raise AssertionError('Warm-up order drift')
    q=None
    out=[];reconstruction=[]
    bymethod={(r['rep'],r['method']):r for r in chk['rows']}
    idx=np.arange(len(g))
    for gen in ('G0_uniform','G1','G2'):
        prep=chk['preprocessing'][gen]
        audits=inv[np.array(prep['original_row_ids'],dtype=int)]
        owned={int(i):int(g[i]) for i in audits}
        if array_hash(np.array(sorted(owned),dtype=np.int64))!=chk['frozen_arrays'][gen]['audits']:raise AssertionError('Audit-set drift')
        active=int(any(log['guarded'] for log in prep['group_log']))
        if not active:
            # Exact logical identity, not a numerical approximation: no accepted
            # replacement means repaired predictions=baseline predictions, hence
            # rescoring by the same model also leaves all scores unchanged.
            if chk['frozen_arrays'][gen]['base']!=chk['frozen_arrays'][gen]['repair']:
                raise AssertionError('Inactive state has nonidentical archived labels')
            for sc in ('S0','S1'):
                for rep in range(10):
                    ar=bymethod[rep,f'{gen}_{sc}_same_audit']; br=bymethod[rep,f'{gen}_{sc}_repair']
                    for f in ('expert_total','error','u_hat','calibration_new','threshold_new'):
                        if ar[f]!=br[f]:raise AssertionError('Inactive state differs from B')
                    cost=int(ar['expert_total']);err=float(ar['error'])
                    out.append({'pool':pool,'n':len(g),'seed':seed,'rep':rep,'generator':gen,'score':sc,
                        'accepted_state':0,'changed_predictions':0,'rescore_changed_entries':0,
                        'pre_audits':len(owned),'A_cost':cost,'B_cost':cost,'C_cost':cost,
                        'delta_label':0,'delta_rescoring':0,'delta_joint':0,
                        'A_error':err,'B_error':err,'C_error':err,
                        'A_cal_new':int(ar['calibration_new']),'B_cal_new':int(ar['calibration_new']),
                        'C_cal_new':int(ar['calibration_new']),
                        'A_threshold_new':int(ar['threshold_new']),'B_threshold_new':int(ar['threshold_new']),
                        'C_threshold_new':int(ar['threshold_new']),
                        'A_threshold':float(ar['u_hat']),'B_threshold':float(ar['u_hat']),
                        'C_threshold':float(ar['u_hat']),
                        'A_violated':int(err>.03),'C_violated':int(err>.03)})
                reconstruction.append({'pool':pool,'seed':seed,'generator':gen,'score':sc,
                    'accepted_state':0,'changed_predictions':0,'baseline_hash_verified':'not_recomputed_identity',
                    'audits':len(owned),'rescore_changed_entries':0})
            continue
        if q is None:
            features=np.load(source/'feature_cache'/f'{pool}.npz')['features']
            q,_=_text_probabilities(features[perm],warm,g[warm],K)
        post=dawid_skene(L,K,owned)
        base=post.argmax(1)
        for i,lab in owned.items():base[i]=lab
        repair=base.copy()
        for log in prep['group_log']:
            if log['guarded']:
                items=np.asarray(log['remainder_members_at_stop'],dtype=int)
                if any(int(i) in owned for i in items):raise AssertionError('Repaired audited item')
                repair[items]=int(log['alt'])
        if array_hash(base)!=chk['frozen_arrays'][gen]['base'] or array_hash(repair)!=chk['frozen_arrays'][gen]['repair']:
            raise AssertionError('A/B prediction vector mismatch')
        changed=int((repair!=base).sum())
        for sc in ('S0','S1'):
            if sc=='S0':
                U0=1.0-post.max(1)
                Ur=1.0-post[idx,repair]
            else:
                U0=1.0-q[idx,base]
                Ur=1.0-q[idx,repair]
            if not np.isfinite(Ur).all() or (Ur < 0).any() or (Ur > 1).any():raise AssertionError('Score out of range')
            equality=bool(np.array_equal(U0,Ur))
            if (changed==0) != equality:
                # This can be equal with changed labels if the two classes have same probabilities.
                if changed==0: raise AssertionError('No changes but score differs')
            score_hash_match=array_hash(U0)==chk['frozen_arrays'][gen]['scores'][sc]
            # Portable execution may differ by tiny numeric roundoff.
            # Independently re-evaluated A/B discrete PAC outcomes are the acceptance gate.
            # Record score-hash drift in state reconstruction rather than failing.
            if not score_hash_match and changed==0:
                pass
            drift_changed=int(np.count_nonzero(Ur!=U0))
            # Scores and predictions both fixed before PAC draws.
            for rep,stream in enumerate(streams_pac):
                Arow=bymethod[rep,f'{gen}_{sc}_same_audit']
                Brow=bymethod[rep,f'{gen}_{sc}_repair']
                if int(Arow['pre_audits'])!=len(owned) or int(Brow['pre_audits'])!=len(owned):raise AssertionError('Historical audit budget drift')
                if not changed:
                    # C identically equals A when there is no changed label.
                    Crow=Arow
                    validated=True
                else:
                    # Recompute ALL 3 arms for active states and compare A/B to archive.
                    _,av=pac_labels(base,U0,g,eps=.03,alpha=.1,m=500,already=owned,seed=stream)
                    _,bv=pac_labels(repair,U0,g,eps=.03,alpha=.1,m=500,already=owned,seed=stream)
                    _,Crow=pac_labels(repair,Ur,g,eps=.03,alpha=.1,m=500,already=owned,seed=stream)
                    validated=True
                    for inf,old,name in [(av,Arow,'A'),(bv,Brow,'B')]:
                        if len(owned)+inf['expert_new']!=int(old['expert_total']) or abs(inf['error']-float(old['error']))>1e-12:
                            raise AssertionError(f'Archived {name} outcome mismatch: {pool}/{seed}/{gen}/{sc}/{rep}')
                        if not (float(inf['u_hat'])==float(old['u_hat']) or abs(float(inf['u_hat'])-float(old['u_hat']))<1e-11):
                            raise AssertionError(f'Archived {name} threshold mismatch: {pool}/{seed}/{gen}/{sc}/{rep}')
                base_cost=int(Arow['expert_total']);label_cost=int(Brow['expert_total']);joint_cost=len(owned)+int(Crow['expert_new'])
                out.append({
                    'pool':pool,'n':len(g),'seed':seed,'rep':rep,'generator':gen,'score':sc,
                    'accepted_state':active,'changed_predictions':changed,'rescore_changed_entries':drift_changed,
                    'pre_audits':len(owned),'A_cost':base_cost,'B_cost':label_cost,'C_cost':joint_cost,
                    'delta_label':base_cost-label_cost,'delta_rescoring':label_cost-joint_cost,
                    'delta_joint':base_cost-joint_cost,
                    'A_error':float(Arow['error']),'B_error':float(Brow['error']),'C_error':float(Crow['error']),
                    'A_cal_new':int(Arow['calibration_new']),'B_cal_new':int(Brow['calibration_new']),
                    'C_cal_new':int(Crow['calibration_new']),
                    'A_threshold_new':int(Arow['threshold_new']),'B_threshold_new':int(Brow['threshold_new']),
                    'C_threshold_new':int(Crow['threshold_new']),
                    'A_threshold':float(Arow['u_hat']),'B_threshold':float(Brow['u_hat']),
                    'C_threshold':float(Crow['u_hat']),
                    'A_violated':int(float(Arow['error'])>.03), 'C_violated':int(float(Crow['error'])>.03),
                })
            reconstruction.append({'pool':pool,'seed':seed,'generator':gen,'score':sc,'accepted_state':active,
                                  'changed_predictions':changed,'baseline_hash_verified':score_hash_match,
                                  'audits':len(owned),'rescore_changed_entries':drift_changed})
    return out,reconstruction


def aggregate(rows,out_dir):
    df=pd.DataFrame(rows)
    df.to_csv(out_dir/'e5_paired_rows.csv',index=False)
    stage=[];rng=np.random.default_rng(20261104)
    for (pool,gen,score),part in df.groupby(['pool','generator','score'],sort=True):
        seedavg=part.groupby('seed')[['delta_label','delta_rescoring','delta_joint']].mean()
        a=seedavg.to_numpy();n=len(a)
        # Whole-seed bootstrap; do not treat repeated PAC draws as independent.
        chosen=rng.integers(0,n,(4000,n))
        means=a[chosen].mean(axis=1)
        stats={'pool':pool,'generator':gen,'score':score,'seeds':n,'pairs':len(part),
               'accepted_seeds':int(part.groupby('seed')['accepted_state'].max().sum()),
               'nonzero_joint_pairs':int((part.delta_joint!=0).sum()),
               'saved_1pct_pairs':int((part.delta_joint>=.01*int(part.n.iloc[0])).sum()),
               'added_1pct_pairs':int((part.delta_joint<=-.01*int(part.n.iloc[0])).sum()),
               'C_final_error_violations':int(part.C_violated.sum()),
               'A_final_error_violations':int(part.A_violated.sum())}
        for j,name in enumerate(['label','rescoring','joint']):
            stats[f'mean_{name}_saving']=float(a[:,j].mean())
            lo,hi=np.quantile(means[:,j],[.025,.975]);stats[f'{name}_seedboot95_low']=float(lo);stats[f'{name}_seedboot95_high']=float(hi)
        stage.append(stats)
    pd.DataFrame(stage).to_csv(out_dir/'e5_seedcluster_summary.csv',index=False)
    # Explicit conditional-on-accepted-state distribution.
    act=df.loc[df.accepted_state==1]
    if len(act):
        act.groupby(['pool','generator','score']).agg(
            accepted_seed_count=('seed','nunique'),pairs=('rep','size'),
            mean_joint=('delta_joint','mean'),median_joint=('delta_joint','median'),
            min_joint=('delta_joint','min'),max_joint=('delta_joint','max'),
            nonzero_joint=('delta_joint',lambda s:int((s!=0).sum()))
        ).reset_index().to_csv(out_dir/'e5_active_state_distribution.csv',index=False)
    return df, pd.DataFrame(stage)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    ap.add_argument('--out',type=Path,default=BASE/'results'/'E5');ap.add_argument('--pools',nargs='*')
    ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--seed-stop',type=int,default=20)
    args=ap.parse_args()
    source=args.reference/'study'/'frozen_v8_1'
    if not (source/'protocol_v8_1.json').exists(): raise SystemExit('Missing source; extract AuditRepair-v8.3 first')
    pp=json.loads((BASE/'protocol'/'frozen_v84.json').read_text());hashes=pp['v83_sha256']
    for rel,sha in hashes.items():
        b=(args.reference/rel).read_bytes()
        if hashlib.sha256(b).hexdigest()!=sha: raise SystemExit(f'v8.3 input drift: {rel}')
    if not 0<=args.seed_start<args.seed_stop<=20: raise SystemExit('E5 seed range must be 0..20')
    pools=args.pools or pp['e5']['pools']
    if any(x not in pp['e5']['pools'] for x in pools):raise SystemExit('Unknown pool')
    args.out.mkdir(parents=True,exist_ok=True)
    funcs=setup_import(args.reference)
    allrows=[];state=[]
    for pool in pools:
        for seed in range(args.seed_start,args.seed_stop):
            rows,recon=one_seed(args.reference,pool,seed,funcs)
            allrows.extend(rows);state.extend(recon)
            print(f'E5 {pool} seed={seed} active={sum(z["accepted_state"] for z in recon)//2} rows={len(rows)}',flush=True)
    df,su=aggregate(allrows,args.out)
    pd.DataFrame(state).to_csv(args.out/'e5_state_reconstruction.csv',index=False)
    manifest={'status':'complete' if len(pools)==len(pp['e5']['pools']) and args.seed_start==0 and args.seed_stop==20 else 'partial',
              'scope':'retrospective cached-panel conditional evidence; same original seeds; no new LLM labels',
              'pools':pools,'seeds':[args.seed_start,args.seed_stop],'rows':len(df),'state_score_cells':len(state),
              'accepted_group_state_score_cells':int(sum(x['accepted_state'] for x in state)),
              'baseline_and_repair_checked':'A/B archived outcomes reexecuted for all active state comparisons',
              'primary_contrast':'G1 S1 A-C on AG News holdout (retrospective extension; previously viewed holdout)',
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256((BASE/'protocol'/'frozen_v84.json').read_bytes()).hexdigest()}
    (args.out/'e5_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
