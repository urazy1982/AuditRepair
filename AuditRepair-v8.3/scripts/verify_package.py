#!/usr/bin/env python3
"""Independent v8.3 top-level integrity and result checks.
Does not alter frozen experiment files or claim any new statistical evidence.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MUTABLE = {'study/results/release_check.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    with path.open(encoding='utf-8', newline='') as f:
        return sum(1 for _ in f) - 1


def verify():
    problems = []
    stats = {}
    def require(ok, msg):
        if not ok:
            problems.append(msg)

    manifest_path = ROOT/'RELEASE_MANIFEST_SHA256.json'
    require(manifest_path.exists(), 'Missing RELEASE_MANIFEST_SHA256.json')
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        require(man.get('release_version') == '8.3.0', 'Wrong release version')
        for rel, expected in man.get('files', {}).items():
            f = ROOT/rel
            require(f.is_file(), 'Missing file: ' + rel)
            if f.is_file():
                require(digest(f) == expected, 'SHA256 mismatch: '+rel)
        stats['release_manifest_files_checked'] = len(man.get('files', {}))
    orig = ROOT/'study/frozen_v8_1'
    for phase, expected, expected_rows in [('E1',140,12600), ('E2_E3',140,26600), ('E4',40,3000)]:
        stage = orig/'results'/phase
        files = list((stage/'checkpoints').glob('*/*.json'))
        require(len(files) == expected, f'{phase}: {len(files)} checkpoints, expected {expected}')
        require(records(stage/'rows.csv') == expected_rows, f'{phase}: main row count differs')
        status = json.loads((stage/'run_status.json').read_text())
        require(status.get('status')=='complete', f'{phase}: incomplete status')
        stats[phase] = {'checkpoints':len(files), 'rows':expected_rows}
    stats['main_records'] = sum(a['rows'] for a in [stats['E1'],stats['E2_E3'],stats['E4']])
    for name in ('E1_effects.csv','E1_observed_error.csv','E1_reach.csv','E2_E3_effects.csv',
                 'E2_E3_observed_error.csv','E2_E3_primary_contrasts.csv','E2_E3_reach.csv','E4_observed_error.csv'):
        a=orig/'analysis'/name
        b=ROOT/'validation/main_wsl/analysis'/name
        require(a.is_file() and b.is_file() and digest(a)==digest(b), f'Main-analysis cross-workstation mismatch: {name}')
    stats['main_authoritative_csvs_matched'] = 8
    a=json.loads((ROOT/'study/results/acceptance_replay_summary.json').read_text())
    require(a.get('status')=='passed_no_archived_decision_changed', 'Acceptance replay did not pass')
    require(a.get('total_preprocessing_states')==600 and a.get('total_audit_prefixes')==52824, 'Replay coverage incorrect')
    require(a.get('accepted_groups_corrected')==a.get('accepted_groups_archived')==55, 'Changed accepted-group total')
    require(a.get('changed_acceptance_predicates')==0, 'Replayed acceptance change')
    stats['acceptances']=55
    import pandas as pd
    cell=pd.read_csv(ROOT/'study/results/active_cell_classification.csv')
    require(len(cell)==42, f'Active-cell count incorrect: {len(cell)}')
    mech=pd.read_csv(ROOT/'study/results/active_mechanisms.csv')
    require(len(mech)==620,'Accepted-pair mechanism rows != 620')
    d=pd.read_csv(ROOT/'study/results/active_cost_distributions.csv')
    x=d[d['scope']=='all_prespecified_preprocessing_seeds']
    require(len(x)==10,'Expected 10 active cell distribution summaries')
    require(int(x.saving_repetitions.sum())==33 and int(x.cost_increase_repetitions.sum())==30,'Active tail signs mismatch')
    require(int(x.paired_repetitions.sum())==2000,'Active 10-cell repetition denominator incorrect')
    stats['active_pairs']=len(mech)
    p=json.loads((ROOT/'study/results/power_manifest.json').read_text())
    require(p.get('status')=='complete' and p.get('matched_orders')==30000 and p.get('level_runs')==60000,'Power run not complete')
    stats['power_matched_orders']=p.get('matched_orders')
    ws=json.loads((ROOT/'validation/targeted_wsl/CURRENT_STATUS.json').read_text())
    require(ws.get('verdict')=='verified_complete_within_recovery_scope','WSL target verification status not complete')
    require(ws.get('total_scoped_PAC_arm_records')==7660 and ws.get('changed_acceptance_predicates')==0,'WSL scope changed')
    stats['WSL_verified_PAC_records']=7660
    stats['historical_archive']='historical_support/AuditRepair_v7_9_code_data_results.zip (repository root)'
    # Verify no obvious private material in tracked files by filename.
    for pth in ROOT.rglob('*'):
        if pth.is_file() and (pth.name in ('.env','id_rsa','id_ed25519') or pth.suffix.lower() in ('.pem','.p12','.pfx','.key')):
            problems.append('Potential secret-bearing file name: '+str(pth.relative_to(ROOT)))
    report={'release':'8.3.0','status':'passed' if not problems else 'failed','checks':stats,'problems':problems,
            'scope':'Sealed file identity, study output grid, cross-workstation authoritative CSV agreement and amended results. Does not certify dataset redistribution rights or prove probability guarantees.'}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if problems:
        raise SystemExit(1)

if __name__=='__main__':
    verify()
