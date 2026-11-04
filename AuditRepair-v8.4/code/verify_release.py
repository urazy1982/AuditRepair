#!/usr/bin/env python3
"""Validate the externally frozen parent identity and all v8.4 completed outputs."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ.get('V83_REFERENCE',str(ROOT.parent/'AuditRepair-v8.3')))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=ROOT/'protocol/frozen_v84.json';expected=(ROOT/'protocol/frozen_v84.sha256').read_text().split()[0]
    if digest(p)!=expected:raise AssertionError('Protocol checksum mismatch')
    protocol=json.loads(p.read_text())
    for rel,exp in protocol['v83_sha256'].items():
        if digest(REF/rel)!=exp:raise AssertionError(f'Parent reference changed: {rel}')
    e5=ROOT/'results/E5'
    m=json.loads((e5/'e5_manifest.json').read_text())
    if m['status']!='complete' or m['rows']!=8400:raise AssertionError('E5 incomplete')
    if digest(ROOT/'protocol/frozen_e5_as_executed.json')!=m['protocol_sha256']:
        raise AssertionError('E5 performed under a different recorded frozen protocol')
    x=pd.read_csv(e5/'e5_paired_rows.csv')
    if len(x)!=8400 or x.groupby(['pool','generator','score','seed']).size().ne(10).any():raise AssertionError('E5 paired grid')
    if not ((x.A_cost-x.B_cost==x.delta_label)&(x.B_cost-x.C_cost==x.delta_rescoring)&(x.A_cost-x.C_cost==x.delta_joint)).all():
        raise AssertionError('E5 cost identity')
    if (x.query('accepted_state==0').delta_joint!=0).any():raise AssertionError('Inactive E5 identity')
    a=pd.read_csv(e5/'e5_state_reconstruction.csv')
    if len(a)!=840 or int(a.accepted_state.sum())!=62:raise AssertionError('E5 accepted state count drift')
    e6=ROOT/'results/E6_full'
    em=json.loads((e6/'e6_manifest.json').read_text())
    if em['status']!='complete' or em['cells']!=54 or em['per_cell_independent_seeds']!=100:
        raise AssertionError('E6 not fully completed')
    if em['protocol_sha256']!=expected:raise AssertionError('E6 wrong frozen protocol')
    rows=pd.read_csv(e6/'e6_rows.csv');st=pd.read_csv(e6/'e6_states.csv')
    if len(rows)!=81000 or len(st)!=16200:raise AssertionError('E6 row cardinality')
    if rows.groupby(['cell_id','seed','policy']).size().ne(5).any() or st.groupby(['cell_id','seed','policy']).size().ne(1).any():
        raise AssertionError('E6 nesting')
    if not (rows.A_cost-rows.repair_cost==rows.saved).all():raise AssertionError('E6 cost identity')
    if not ((rows.group_audits<=196)&(rows.group_audits>=0)&(rows.A_cost<=5000)&(rows.repair_cost<=5000)).all():
        raise AssertionError('E6 budget bounds')
    if not ((st.certified<=st.tested)&(st.tested<=st.groups)).all():raise AssertionError('E6 group counts')
    e8=json.loads((ROOT/'results/E8/e8_independent_audit.json').read_text())
    if e8['status']!='passed' or e8['wor']['violations'] or e8['terminal_pac']['violations']:
        raise AssertionError('E8 independent checks fail')
    em7=json.loads((ROOT/'E7_tasks/E7_GOLD_BLIND_MANIFEST.json').read_text())
    if len(em7['tasks'])!=8 or sum(v['n'] for v in em7['tasks'].values())!=30400:raise AssertionError('E7 worklist incomplete')
    for name,info in em7['tasks'].items():
        if digest(ROOT/'E7_tasks'/name)!=info['sha256']:raise AssertionError('E7 task drift')
    report={'status':'passed','parent_input_checkpoint_hashes':len(protocol['v83_sha256']),
            'e5_pairs':len(x),'e5_active_state_score_cells':int(a.accepted_state.sum()),
            'e6_pairs':len(rows),'e6_states':len(st),'e6_design_cells':int(rows.cell_id.nunique()),
            'e8_small_WoR_paths':e8['wor']['paths'],'e8_PAC_decimal_cases':e8['terminal_pac']['scenarios'],
            'e7_gold_blind_requests':sum(v['n'] for v in em7['tasks'].values()),
            'new_LLM_annotation_files':0,
            'scope':'completed CPU interventions on cached panels plus fresh synthetic populations; no newly collected annotation panel'}
    target=ROOT/'results/release_check.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
