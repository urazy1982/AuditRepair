"""Amended E1 oracle family on cached real annotation pools, with immutable checkpoints.

Run: python run_e1.py --workers 3
Defaults are the frozen 7 pools x 20 preprocessing seeds x 10 paired PAC repetitions.
Results are conditional on the observed panels; O4 is an adaptive diagnostic only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from audit_repair.core import Oracle, candidate_groups_disjoint, dawid_skene, run_group_repair, uncertainty_fill
from audit_repair.pac import pac_labels
from experiment_common import (ROOT, array_hash, atomic_json, changed_diagnostics,
                               check_pac_accounting, file_hash, load_pool, protocol,
                               streams, summarize_rows)

METHODS = ['pac_only','same_audit_ds','O0_identity','legacy_guarded','O1_shortlist',
           'O2_disagreement','O3_complete_unanimity','O4_baseline_prefix','O5_all_errors']


def e1_masks(L, base, gold, audits):
    dissent = np.array([len(set(row[row >= 0])) >= 2 for row in L], dtype=bool)
    unanimous = (L >= 0).all(axis=1) & (L == L[:, :1]).all(axis=1)
    wrong = base != gold
    if audits:
        wrong[list(audits)] = False
    return wrong, dissent, unanimous


def one_seed(pool, seed, out_dir):
    started = time.monotonic()
    p, protocol_sha = protocol(validate=False)
    a = p['parameters']
    code = {f:file_hash(ROOT/f) for f in ['run_e1.py','experiment_common.py','audit_repair/core.py','audit_repair/pac.py']}
    dest = Path(out_dir) / 'checkpoints' / pool / f'seed_{seed:02d}.json'
    if dest.exists():
        cached = json.loads(dest.read_text())
        if cached['protocol_sha256'] != protocol_sha or cached['code_sha256'] != code:
            raise ValueError(f'Existing checkpoint has different protocol/code: {dest}')
        if cached['status'] != 'complete' or len(cached['rows']) != len(METHODS)*a['pac_reps']:
            raise ValueError(f'Incomplete checkpoint: {dest}')
        return cached, True
    _, L0, gold0, K = load_pool(pool)
    ss_data, ss_pre, pac_streams = streams(a['root_seed'], seed, a['pac_reps'])
    perm = np.random.default_rng(ss_data).permutation(len(gold0))
    L, gold = L0[perm], gold0[perm]
    post0 = dawid_skene(L, K)
    res = run_group_repair(L,K,gold,a['B'],alpha=a['alpha_repair'],seed=ss_pre,
                           ref_share=a['warmup']/a['B'],min_size=a['min_group_size'],
                           stop_after_fails=a['stop_after_fails'])
    guarded, guarded_info = res['guarded']
    audits = guarded_info['audited_labels']
    post = guarded_info['post']
    base = post.argmax(1)
    U = 1-post.max(1)
    if not all(int(base[i]) == y for i,y in audits.items()):
        raise AssertionError('Same-audit DS predictions are not clamped')
    warm = {}
    uncertainty_fill(L,K,Oracle(gold,a['warmup']),warm)
    post_ref = dawid_skene(L,K,warm)
    ref = post_ref.argmax(1)
    for i,y in warm.items(): ref[i]=y
    shortlist = sorted(candidate_groups_disjoint(L,ref,post_ref,min_size=a['min_group_size']),
                       key=lambda g:-len(g.items))[:a['max_groups']]
    y_shortlist = base.copy()
    groups = []
    for rank,g in enumerate(shortlist):
        rem = np.array([int(i) for i in g.items if int(i) not in audits],dtype=int)
        gain = int((gold[rem]==g.alt).sum()-(gold[rem]==base[rem]).sum())
        if gain > 0: y_shortlist[rem] = g.alt
        groups.append({'rank':rank,'cur':g.cur,'alt':g.alt,'size':len(g.items),
                       'remainder_size':len(rem),'same_audit_net_gain':gain})
    wrong, dissent, unanimous = e1_masks(L,base,gold,audits)
    y_dis,y_uni,y_all = base.copy(),base.copy(),gold.copy()
    y_dis[wrong & dissent] = gold[wrong & dissent]
    y_uni[wrong & unanimous] = gold[wrong & unanimous]
    ug = U.copy()
    for i,confidence in guarded_info['repair_conf'].items(): ug[i] = 1-confidence
    fixed = {
        'pac_only':(post0.argmax(1),1-post0.max(1),{}),
        'same_audit_ds':(base,U,audits),'O0_identity':(base.copy(),U.copy(),dict(audits)),
        'legacy_guarded':(guarded,ug,audits),'O1_shortlist':(y_shortlist,U,audits),
        'O2_disagreement':(y_dis,U,audits),'O3_complete_unanimity':(y_uni,U,audits),
        'O5_all_errors':(y_all,U,audits),
    }
    # This timestamp and digest are recorded before any evaluation calibration draws.
    frozen_inputs = {name:{'labels_sha256':array_hash(y),'scores_sha256':array_hash(u),
                          'audited_ids_sha256':array_hash(np.array(sorted(aud),dtype=np.int64))}
                     for name,(y,u,aud) in fixed.items()}
    frozen_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for rep,stream in enumerate(pac_streams):
        final_base, info_base = pac_labels(base,U,gold,eps=a['epsilon'],alpha=a['alpha_final'],
                                           m=a['m'],already=audits,seed=stream)
        y_prefix = base.copy()
        mask = wrong & (U <= info_base['u_hat'])
        y_prefix[mask] = gold[mask]
        arms = dict(fixed)
        arms['O4_baseline_prefix'] = (y_prefix,U,audits)
        for method in METHODS:
            pred,u,aud = arms[method]
            if method == 'same_audit_ds': final,info = final_base,info_base
            else:
                final,info = pac_labels(pred,u,gold,eps=a['epsilon'],alpha=a['alpha_final'],
                                       m=a['m'],already=aud,seed=stream)
            accounting = check_pac_accounting(info,u,aud,len(gold),a['m'],stream)
            if method == 'O0_identity' and (info != info_base or not np.array_equal(final,final_base)):
                raise AssertionError('Identity oracle changed output')
            if method == 'O5_all_errors' and not np.array_equal(final,gold):
                raise AssertionError('Full-gold oracle did not remain perfect')
            if method == 'O5_all_errors' and info['threshold_new'] != 0:
                raise AssertionError('Full-gold oracle did not attain the prespecified fixed-query cost floor')
            row = {'pool':pool,'n':len(gold),'method':method,'seed':seed,'rep':rep,
                   'pre_audits':len(aud),'expert_total':len(aud)+info['expert_new'],
                   'baseline_u_hat':info_base['u_hat'], **info, **accounting,
                   'violated':info['error'] > a['epsilon'],
                   'frozen_label_PAC_interpretation':method != 'O4_baseline_prefix',
                   'attains_fixed_cost_floor':info['threshold_new'] == 0,
                   **changed_diagnostics(pred,base,gold,U,info_base['u_hat'],audits)}
            rows.append(row)
    answer = {'status':'complete','pool':pool,'seed':seed,'protocol_sha256':protocol_sha,
              'code_sha256':code,'frozen_arrays_utc':frozen_utc,'frozen_arrays':frozen_inputs,
              'permutation_sha256':array_hash(perm),'audits_original_row_ids':[int(perm[i]) for i in sorted(audits)],
              'preprocessing':{'audits':len(audits),'groups_certified':guarded_info['groups_repaired'],
                               'harmful_vs_frozen_ref':guarded_info['harmful_group_repairs'],
                               'beneficial_shortlist_groups':sum(g['same_audit_net_gain']>0 for g in groups),
                               'baseline_errors':int(wrong.sum()),'disagreement_errors':int((wrong&dissent).sum()),
                               'complete_unanimity_errors':int((wrong&unanimous).sum()),
                               'other_missing_vote_errors':int((wrong&~dissent&~unanimous).sum())},
              'shortlist':groups,'rows':rows,'wall_seconds':time.monotonic()-started}
    atomic_json(dest,answer)
    return answer,False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--out',type=Path,default=ROOT/'results/E1')
    ap.add_argument('--pools',nargs='+')
    ap.add_argument('--seed-start',type=int,default=0)
    ap.add_argument('--seed-stop',type=int)
    args = ap.parse_args()
    p,sha = protocol()
    selected = args.pools or p['pools']
    if not set(selected)<=set(p['pools']): raise ValueError('Pool outside frozen protocol')
    stop = args.seed_stop if args.seed_stop is not None else p['parameters']['seeds']
    if not 0<=args.seed_start<stop<=p['parameters']['seeds']: raise ValueError('Invalid seed range')
    args.out.mkdir(parents=True,exist_ok=True)
    tasks = [(pool,seed) for pool in selected for seed in range(args.seed_start,stop)]
    run = {'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha,
           'expected_tasks':len(tasks),'completed_tasks':0,'status':'running',
           'selected_pools':selected,'seed_range':[args.seed_start,stop],
           'interpretation':'complete protocol only if all seven pools and all twenty seeds finish'}
    atomic_json(args.out/'run_status.json',run)
    rows=[];states=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fs={ex.submit(one_seed,pool,seed,args.out):(pool,seed) for pool,seed in tasks}
            for future in as_completed(fs):
                answer,cached=future.result();rows.extend(answer['rows']);states.append(answer['preprocessing']|{'pool':answer['pool'],'seed':answer['seed']})
                run['completed_tasks']+=1
                atomic_json(args.out/'run_status.json',run)
                print(f"E1 {run['completed_tasks']}/{len(tasks)} {answer['pool']} seed={answer['seed']} {'cached' if cached else 'completed'} {answer['wall_seconds']:.1f}s",flush=True)
        rows.sort(key=lambda r:(p['pools'].index(r['pool']),r['seed'],r['rep'],METHODS.index(r['method'])))
        summarize_rows(rows,args.out)
        import pandas as pd
        pd.DataFrame(states).sort_values(['pool','seed']).to_csv(args.out/'preprocessing.csv',index=False)
        run['status']='complete'
        run['full_frozen_scope_complete']=set(selected)==set(p['pools']) and args.seed_start==0 and stop==p['parameters']['seeds']
        run['rows']=len(rows)
        run['completed_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(args.out/'run_status.json',run)
    except Exception as error:
        run['status']='failed';run['error']=repr(error);atomic_json(args.out/'run_status.json',run)
        raise


if __name__=='__main__': main()
