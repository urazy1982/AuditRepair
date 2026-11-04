#!/usr/bin/env python3
"""Independent synthetic gate for the v8 factorial, never reads study pools."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE))
f=importlib.import_module('factorial')
r=importlib.import_module('run_factorial')
c=importlib.import_module('audit_repair.core')


def model(L,K,clamp=None,**kwargs):
    post=np.column_stack([np.full(len(L),.9),np.full(len(L),.1)])
    for i,y in (clamp or {}).items():
        post[i]=0.;post[i,y]=1.
    return post


def main():
    files=['factorial.py','run_factorial.py','experiment_common.py','audit_repair/core.py','audit_repair/pac.py']
    hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    n=620
    L=np.column_stack([np.zeros(n,int),np.ones(n,int)])
    gold=np.ones(n,int);gold[-70:]=0
    X=np.column_stack([np.linspace(-1,1,n),np.ones(n)])
    neighbors=np.array([np.array([j for j in range(16) if j!=i][:15]) for i in range(n)])
    warm=np.arange(120)
    assertions=0; chronology=[]
    def check(condition,message):
        nonlocal assertions
        assertions+=1
        if not condition: raise AssertionError(message)
    original_oracle=f.Oracle
    original_make=f._generate_groups
    original_text=f._text_probabilities
    original_gain=f._gain
    original_fill=f.uncertainty_fill
    for generator in ('G0_uniform','G1','G2'):
        queries=[]; events=[]; fill_done=[False]; freeze=[None]
        class RecordingOracle(original_oracle):
            def query(self,i):
                check(not fill_done[0],'Oracle queried after final fill completed')
                y=super().query(i);queries.append(int(i));return y
        def make(*args,**kwargs):
            check(len(set(queries))==120,'Shortlist did not freeze after exactly 120 warm-up labels')
            groups=original_make(*args,**kwargs)
            freeze[0]=[(g.cur,g.alt,tuple(g.items)) for g in groups]
            events.append('shortlist_frozen')
            return groups
        def text(features,warm_ids,labels,K):
            check(set(warm_ids)==set(queries),'Text model saw anything beyond warm-up items')
            check(np.array_equal(labels,gold[warm_ids]),'Text model labels do not match charged warm-up')
            events.append('text_model_fitted_from_warmup')
            return original_text(features,warm_ids,labels,K)
        def fill(*args,**kwargs):
            result=original_fill(*args,**kwargs)
            fill_done[0]=True;events.append('all_adaptive_audits_finished')
            return result
        def gain(*args,**kwargs):
            check(fill_done[0],'Full-gold diagnostic invoked before adaptive audits finished')
            return original_gain(*args,**kwargs)
        with patch.object(f,'Oracle',RecordingOracle),patch.object(f,'_generate_groups',make),patch.object(f,'_text_probabilities',text),patch.object(f,'_gain',gain),patch.object(f,'uncertainty_fill',fill),patch.object(f,'dawid_skene',model),patch.object(c,'dawid_skene',model):
            state=f.preprocess_generator(L,2,gold,X,neighbors,generator,seed=912,warm_indices=warm)
        check(events==['text_model_fitted_from_warmup','shortlist_frozen','all_adaptive_audits_finished'],'Unexpected audit chronology')
        check(len(state['audits'])==len(set(queries))<=400,'Unique audit budget accounting')
        check(state['diagnostics']['group_phase_audits']<=196,'Group phase exceeds prespecified budget')
        group_members=[i for g in state['groups'] for i in g['members']]
        check(len(group_members)==len(set(group_members)),'Overlapping frozen candidate groups')
        check(not set(group_members)&set(warm),'Warm-up item in certification population')
        check(all(not s.flags.writeable for s in state['scores'].values()),'Score arrays not frozen')
        check(len(state['group_log'])>0,'Fixture did not exercise group audits')
        for group in state['group_log']:
            check(group['alpha']==.05/max(1,len(state['groups'])),'Incorrect familywise alpha allocation')
            check(set(group['members'])==set(group['remainder_members_at_stop'])|{h['item'] for h in group['audit_history']},'Remainder membership mismatch')
            if group['guarded']:
                check(group['remainder_lb_exact']>0,'Accepted without positive bound')
                check(not set(group['remainder_members_at_stop'])&set(state['audits']),'Accepted remainder later audited')
        chronology.append({'generator':generator,'events':events,'groups':len(state['groups']),'tested':len(state['group_log']),'accepted':state['diagnostics']['certified_groups'],'audits':len(state['audits'])})

    # Exercise the actual 19-arm orchestration on a synthetic pool through mocks
    # at its data boundary. Cache loading is intercepted; no real pool is read.
    params={'B':400,'root_seed':20261104,'pac_reps':3,'epsilon':.05,'alpha_final':.1,'m':80}
    protocol={'parameters':params}
    captured=[]
    actual_pac=r.pac_labels
    original_read_text=Path.read_text
    original_file_hash=r.file_hash
    def synthetic_metadata_read(path,*args,**kwargs):
        if path.name=='synthetic_verification_only.json' and path.parent.name=='feature_cache':
            return json.dumps({'protocol_sha256':'synthetic-verification-protocol','code_sha256':hashes['factorial.py'],'cache_sha256':'synthetic-cache-hash'})
        return original_read_text(path,*args,**kwargs)
    def synthetic_cache_hash(path):
        if Path(path).name=='synthetic_verification_only.npz':return 'synthetic-cache-hash'
        return original_file_hash(path)
    def observed_pac(pred,U,gold_input,**kwargs):
        draws=np.random.default_rng(kwargs['seed']).integers(0,len(pred),kwargs['m'])
        result=actual_pac(pred,U,gold_input,**kwargs)
        already=kwargs.get('already') or {}
        adjusted=U.copy()
        if already: adjusted[list(already)]=-1.
        queried=set(np.flatnonzero(adjusted>=result[1]['u_hat']))|set(draws)|set(already)
        check(len(queried)==len(already)+result[1]['expert_new'],'Independent cost union differs')
        captured.append({'seed_key':kwargs['seed'].spawn_key,'draws_hash':hashlib.sha256(draws.tobytes()).hexdigest(),'score_hash':hashlib.sha256(np.asarray(U).tobytes()).hexdigest(),'owned':len(already)})
        return result
    with tempfile.TemporaryDirectory(prefix='synthetic_factorial_gate_',dir=HERE) as tmp:
        with patch.object(Path,'read_text',synthetic_metadata_read),patch.object(r,'file_hash',synthetic_cache_hash),patch.object(r,'protocol',return_value=(protocol,'synthetic-verification-protocol')),patch.object(r,'load_pool',return_value=(None,L,gold,2)),patch.object(r.np,'load',return_value={'features':X,'neighbors':neighbors}),patch.object(r,'dawid_skene',model),patch.object(f,'dawid_skene',model),patch.object(c,'dawid_skene',model),patch.object(r,'pac_labels',observed_pac):
            answer,cached=r.one_seed('synthetic_verification_only',0,Path(tmp))
        check(not cached,'Synthetic call unexpectedly reused a checkpoint')
        check(len(answer['rows'])==19*3,'Incorrect factorial role/score/generator count')
        for rep in range(3):
            rows=[row for row in answer['rows'] if row['rep']==rep]
            check(len(rows)==19,'Missing arm in paired replication')
            check(len({row['draws_sha256'] for row in rows})==1,'PAC draws not paired across arms')
            for generator in ('G0_uniform','G1','G2'):
                for score in ('S0','S1'):
                    cell=[row for row in rows if row['generator']==generator and row['score']==score]
                    baseline=next(row for row in cell if row['role']=='same_audit')
                    check(all(row['pre_audits']==baseline['pre_audits'] for row in cell),'Within-cell owned audits differ')
                    check(all(row['paired_cost_delta']==row['expert_total']-baseline['expert_total'] for row in cell),'Paired cost contrast differs from direct subtraction')
        check(len({entry['draws_hash'] for entry in captured})==3,'Distinct repetition streams unexpectedly share a full draw sequence')
        preprocessing=answer['preprocessing']
        warmups=[entry['warmup_original_row_ids'] for entry in preprocessing.values()]
        check(all(ids==warmups[0] for ids in warmups),'Warm-up differs across generators')
    check(all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items()),'Source changed during gate')
    report={'status':'passed_under_tested_conditions','artifact_kind':'independent_synthetic_engineering_gate_not_study_results','assertion_count':assertions,'chronology':chronology,'synthetic_orchestration_rows':57,'pac_calls':len(captured),'source_sha256':hashes,'limitations':['Deterministic observable base model substitutes for DS estimation to isolate chronology, pairing and costs.','Does not certify effectiveness, real-data results, annotation variability, or the statistical theorem.','The reviewed runner freezes each generator state before PAC, reuses one score vector for its roles, and uses common PAC draws within a replication.']}
    target=HERE/'factorial_independent_gate_report.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'assertions':assertions,'report':str(target)}))

if __name__=='__main__':main()
