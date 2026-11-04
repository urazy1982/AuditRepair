#!/usr/bin/env python3
"""E6 frozen 54-cell controlled useful-group conversion experiment.

A synthetic generator plants groups using OBSERVABLE group IDs and frozen score
features. The true gains are determined by the synthetic population, not exposed
to candidate selection, test ordering, audit allocation or the PAC score rule.
All policies share group-specific audit permutations and paired PAC draw streams.
The group-confidence-sequence uses the corrected v8.3 exact acceptance predicate.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ.get('V83_REFERENCE',str(ROOT.parent/'AuditRepair-v8.3')))
SEEDROOT=20261104
N=5000
B=400
WARM=120
GROUP_B=196


def setup_import(reference):
    sys.path.insert(0,str(ROOT/'code'))
    sys.path.insert(0,str(reference/'study/corrected'))
    from audit_repair.core import WorLowerCS
    from fast_pac import pac_labels_fast
    return WorLowerCS,pac_labels_fast


def cells():
    return [dict(cell_id=i,size=s,gain=g,position=pos,groups=count)
            for i,(s,g,pos,count) in enumerate(itertools.product(
                (50,100,300),(0.0,0.2,0.4),('low','near','high'),(1,10)))]


def make_pool(cell,seed):
    xseed=np.random.SeedSequence([SEEDROOT,cell['cell_id'],seed])
    poolseed,warmseed,orderseed,randrankseed=xseed.spawn(4)
    rng=np.random.default_rng(poolseed)
    n=N;size=cell['size'];count=cell['groups'];mean=cell['gain']
    # Scores have discrete values and ties to exercise PAC threshold behavior.
    U=np.round(rng.uniform(0,1,n)*64)/64
    pred=np.zeros(n,dtype=np.int64)
    gold=np.where(rng.random(n)<(.015+.20*U**2),rng.choice([1,2],n),0).astype(np.int64)
    membership=np.full(n,-1,dtype=int)
    chosen=rng.permutation(n)[:size*count]
    groups=[];realized=[]
    for i in range(count):
        items=np.asarray(chosen[i*size:(i+1)*size],dtype=int)
        membership[items]=i
        zero=int(round(size*.4));nonzero=size-zero
        gain_count=int(round(mean*size))
        if (nonzero+gain_count)%2:
            # Counterbalance integer rounding, preserving mean to 1/size.
            gain_count-=1
        plus=(nonzero+gain_count)//2
        minus=nonzero-plus
        if min(plus,minus,zero)<0:raise AssertionError('Invalid gain histogram')
        candidate_gold=np.r_[np.ones(plus,dtype=int),np.zeros(minus,dtype=int),np.full(zero,2,dtype=int)]
        gold[items]=rng.permutation(candidate_gold)
        centers={'low':.13,'near':.47,'high':.85}
        # Observable per-group feature shift, fully defined before audits.
        center=centers[cell['position']]+.08*((i/(count-1))-.5 if count>1 else 0)
        U[items]=np.round(np.clip(center+rng.uniform(-.085,.085,len(items)),0,1)*64)/64
        groups.append(items)
        realized.append(int((gold[items]==1).sum()-(gold[items]==0).sum()))
    warm=np.random.default_rng(warmseed).permutation(n)[:WARM]
    order_rng=np.random.default_rng(orderseed)
    # Fixed per-group permutation, regardless of which policy later visits it.
    orders={i:order_rng.permutation(items) for i,items in enumerate(groups)}
    random_order=np.random.default_rng(randrankseed).permutation(count).tolist()
    return {'gold':gold,'pred':pred,'U':U,'membership':membership,'groups':groups,
            'warm':warm,'orders':orders,'random_order':random_order,'true_group_gains':realized}


def ranking(pop,policy):
    groups=pop['groups'];U=pop['U']
    if policy=='size':return sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))
    if policy=='cost_aware_no_gold':
        # Frozen, gold-blind proxy for potential automatic-retention gain, not
        # a fitted PAC threshold or a learned estimate of true correction value.
        return sorted(range(len(groups)),key=lambda i:(
            -int(np.sum(U[groups[i]] <= .60)),float(np.median(U[groups[i]])),i))
    if policy=='random':return list(pop['random_order'])
    raise ValueError('unknown policy')


def audit_policy(pop,policy,WorLowerCS):
    gold=pop['gold'];pred0=pop['pred'];U=pop['U']
    owned={int(i):int(gold[i]) for i in pop['warm']}
    protected=set()
    pred=pred0.copy();accepted=[];tested=[]
    rank=ranking(pop,policy)
    failed=0;used=0
    for gi in rank:
        if used>=GROUP_B or failed>=10: break
        rest=np.asarray([int(i) for i in pop['groups'][gi] if int(i) not in owned],dtype=int)
        if len(rest)<30:continue
        cs=WorLowerCS(len(rest),alpha=.05/len(pop['groups']))
        audit_order=[int(i) for i in pop['orders'][gi] if int(i) not in owned]
        decision='abstain';stop='exhausted'
        for j in audit_order:
            if used>=GROUP_B:stop='budget';break
            y=int(gold[j]);owned[j]=y;used+=1
            d=int(y==1)-int(y==0)
            cs.update((d+1)/2)
            if cs.remainder_is_positive():decision='repair';stop='certified';break
            if cs.t>=10 and cs.mean < .45:stop='futility';break
            if cs.t>=80 or cs.t >= len(rest)/2:
                stop='audit_cap' if cs.t>=80 else 'fraction_cap';break
        remained=[int(i) for i in rest if int(i) not in owned]
        true_gain=int(sum(int(gold[i]==1)-int(gold[i]==0) for i in remained))
        if decision=='repair':
            for i in remained:pred[i]=1
            accepted.append(gi);protected.update(int(i) for i in rest)
            failed=0
        else:failed+=1
        tested.append({'group':gi,'audits':cs.t,'decision':decision,'stop_reason':stop,
                       'true_unaudited_gain':true_gain,'score_mean':float(U[rest].mean()),
                       'lower_index':cs.lower_grid_index if hasattr(cs,'lower_grid_index') else None})
    # Complete the same total B=400 with observable score-based fill.
    candidates=[int(i) for i in np.lexsort((np.arange(len(U)),-U)) if int(i) not in owned and int(i) not in protected]
    for i in candidates[:B-len(owned)]:owned[i]=int(gold[i])
    if len(owned)!=B:raise AssertionError('Unexpected total audit shortfall')
    # The synthetic aggregator is deliberately held fixed after audits;
    # all paid labels are set to gold by PAC via its 'already' argument.
    return {'base':pred0.copy(),'repair':pred,'owned':owned,'tested':tested,
            'accepted_groups':accepted,'tested_count':len(tested),
            'successful_gain':sum(t['true_unaudited_gain'] for t in tested if t['decision']=='repair'),
            'group_audits':used,'failed_group_audits':sum(t['audits'] for t in tested if t['decision']!='repair')}


def one_cell_seed(cell,seed,WorLowerCS,pac_labels):
    pop=make_pool(cell,seed)
    if any(int(g)!=int(round(cell['gain']*cell['size'])) for g in pop['true_group_gains']):
        raise AssertionError('Planted gain mismatch')
    gold=pop['gold'];U=pop['U'];rows=[];states=[]
    # Both arms use the same calibration draws within a policy, and all policies
    # use the same draw-seed specification for paired policy contrasts.
    for policy in ('size','cost_aware_no_gold','random'):
        st=audit_policy(pop,policy,WorLowerCS)
        audits=st['owned'];base=st['base'];repaired=st['repair']
        changed=int(np.count_nonzero(base!=repaired))
        if not st['accepted_groups'] and changed:raise AssertionError('Repaired without certification')
        for rep in range(5):
            seedstream=np.random.SeedSequence([SEEDROOT,cell['cell_id'],seed,rep,17])
            _,A=pac_labels(base,U,gold,eps=.03,alpha=.10,m=500,already=audits,seed=seedstream)
            if changed:
                _,C=pac_labels(repaired,U,gold,eps=.03,alpha=.10,m=500,already=audits,seed=seedstream)
            else:C=A
            ac=B+A['expert_new'];cc=B+C['expert_new']
            rows.append({'cell_id':cell['cell_id'],'seed':seed,'rep':rep,'policy':policy,
                'size':cell['size'],'gain':cell['gain'],'position':cell['position'],'groups':cell['groups'],
                'A_cost':ac,'repair_cost':cc,'saved':ac-cc,'A_error':A['error'],'repair_error':C['error'],
                'accepted_groups':len(st['accepted_groups']),'groups_tested':st['tested_count'],
                'group_audits':st['group_audits'],'failed_group_audits':st['failed_group_audits'],
                'changed_items':changed,'certified_gain':st['successful_gain'],
                'above_baseline_1pct':int(ac-cc>=.01*N),'cost_increase_1pct':int(ac-cc<=-.01*N),
                'A_u_hat':A['u_hat'],'repair_u_hat':C['u_hat'],
                'A_violated':int(A['error']>.03),'repair_violated':int(C['error']>.03)})
        states.append({'cell_id':cell['cell_id'],'seed':seed,'policy':policy,
            'size':cell['size'],'gain':cell['gain'],'position':cell['position'],'groups':cell['groups'],
            'group_audits':st['group_audits'],'failed_group_audits':st['failed_group_audits'],
            'tested':st['tested_count'],'certified':len(st['accepted_groups']),
            'successful_gain':st['successful_gain'],'changed_items':changed,
            'true_available_gain':sum(pop['true_group_gains']),
            'positive_candidates':sum(g>0 for g in pop['true_group_gains'])})
    return rows,states


def output_checkpoint(out,cell,seed,rows,states,protocol_hash):
    d=out/'checkpoints'/f'cell_{cell["cell_id"]:02d}'
    d.mkdir(parents=True,exist_ok=True)
    final=d/f'seed_{seed:03d}.json'
    obj={'version':'v8.4','protocol_sha256':protocol_hash,'cell':cell,'seed':seed,
         'status':'complete','rows':rows,'states':states}
    b=json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    tmp=final.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(final)


def summarize(out,cells_,seedstart,seedstop):
    rows=[];states=[]
    for cell in cells_:
        for seed in range(seedstart,seedstop):
            r=json.loads((out/'checkpoints'/f'cell_{cell["cell_id"]:02d}'/f'seed_{seed:03d}.json').read_text())
            if r['status']!='complete':raise AssertionError('Incomplete checkpoint')
            rows.extend(r['rows']);states.extend(r['states'])
    df=pd.DataFrame(rows);st=pd.DataFrame(states)
    df.to_csv(out/'e6_rows.csv',index=False)
    st.to_csv(out/'e6_states.csv',index=False)
    report=[]
    for keys,block in df.groupby(['cell_id','policy'],sort=True):
        seedave=block.groupby('seed').saved.mean()
        report.append({'cell_id':keys[0],'policy':keys[1],
            'size':int(block['size'].iloc[0]),'gain':float(block['gain'].iloc[0]),
            'position':block['position'].iloc[0],'groups':int(block['groups'].iloc[0]),
            'preprocessing_seeds':seedave.size,'PAC_pairs':len(block),
            'accepted_seed_fraction':float((block.groupby('seed').accepted_groups.max()>0).mean()),
            'mean_expert_labels_saved':float(seedave.mean()),
            'median_seed_mean_saving':float(seedave.median()),
            'fraction_cost_changed':float((block.saved!=0).mean()),
            'fraction_large_saving':float(block.above_baseline_1pct.mean()),
            'fraction_large_cost_increase':float(block.cost_increase_1pct.mean()),
            'max_expert_labels_saved':int(block.saved.max()),'max_expert_labels_added':int(-block.saved.min()),
            'mean_group_audits':float(block.group_audits.mean()),
            'mean_failed_group_audits':float(block.failed_group_audits.mean()),
            'mean_certified_groups':float(block.accepted_groups.mean()),
            'error_violations':int(block.repair_violated.sum())})
    pd.DataFrame(report).to_csv(out/'e6_summary.csv',index=False)
    return len(rows),len(states),pd.DataFrame(report)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,default=REF)
    ap.add_argument('--out',type=Path,default=ROOT/'results/E6')
    ap.add_argument('--seed-start',type=int,default=0)
    ap.add_argument('--seed-stop',type=int,default=2)
    ap.add_argument('--cell-start',type=int,default=0)
    ap.add_argument('--cell-stop',type=int,default=54)
    a=ap.parse_args()
    if not 0<=a.seed_start<a.seed_stop<=100 or not 0<=a.cell_start<a.cell_stop<=54:raise SystemExit('Outside frozen seed or cell scope')
    protocol=json.loads((ROOT/'protocol/frozen_v84.json').read_text())
    if protocol['e6']['N']!=N or protocol['e6']['total_independent_seeds_per_cell']!=100:raise AssertionError('Protocol disagreement')
    phash=hashlib.sha256((ROOT/'protocol/frozen_v84.json').read_bytes()).hexdigest()
    WorLowerCS,pac_labels=setup_import(a.reference)
    chosen=cells()[a.cell_start:a.cell_stop]
    a.out.mkdir(exist_ok=True,parents=True)
    started=time.monotonic()
    for cell in chosen:
        for seed in range(a.seed_start,a.seed_stop):
            ck=a.out/'checkpoints'/f'cell_{cell["cell_id"]:02d}'/f'seed_{seed:03d}.json'
            if ck.exists():
                obj=json.loads(ck.read_text())
                if obj['protocol_sha256']!=phash or obj['status']!='complete':raise AssertionError('Old/out of sync checkpoint')
                continue
            rows,states=one_cell_seed(cell,seed,WorLowerCS,pac_labels)
            output_checkpoint(a.out,cell,seed,rows,states,phash)
        print('E6 complete',cell,'seeds',a.seedstart if hasattr(a,'seedstart') else a.seed_start,'..',a.seed_stop,flush=True)
    nr,ns,summary=summarize(a.out,chosen,a.seed_start,a.seed_stop)
    status={'status':'complete' if a.seed_start==0 and a.seed_stop==100 and a.cell_start==0 and a.cell_stop==54 else 'pilot_partial',
            'cells':len(chosen),'seed_range':[a.seed_start,a.seed_stop],'per_cell_independent_seeds':a.seed_stop-a.seed_start,
            'PAC_pairs':nr,'policy_states':ns,'wall_seconds':round(time.monotonic()-started,2),
            'protocol_sha256':phash,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scientific_scope':'synthetic generator with planted observable group IDs, not LLM annotations; gold unseen by policy except bought audits'}
    (a.out/'e6_manifest.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
