"""Frozen input validation, seed streams, PAC evaluation and paired summaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ('infinity' if obj > 0 else '-infinity')
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def protocol(validate=True):
    path = ROOT / 'protocol_v8_1.json'
    expected = (ROOT / 'protocol_v8_1.sha256').read_text().split()[0]
    actual = file_hash(path)
    if actual != expected:
        raise ValueError('Protocol hash mismatch; refusing to run')
    p = json.loads(path.read_text())
    if validate:
        for relative, data in p['input_files'].items():
            if file_hash(ROOT / relative) != data['sha256']:
                raise ValueError(f'Input changed after freeze: {relative}')
        for name, expected_sha in p['historical_core_sha256'].items():
            if file_hash(ROOT / 'audit_repair' / name) != expected_sha:
                raise ValueError(f'Historical core changed after freeze: {name}')
    return p, actual


def load_pool(name, with_text=False):
    fname = 'banking77_with_text' if name == 'banking77' and with_text else name
    path = ROOT / 'data' / (fname + '.csv')
    df = pd.read_csv(path, dtype={'id': str})
    if df['id'].isna().any() or df['id'].duplicated().any():
        raise ValueError(f'Nonunique or missing IDs: {name}')
    source_columns = [c for c in df if c.startswith('src_')]
    if not source_columns:
        raise ValueError(f'No source labels: {name}')
    numeric = df[source_columns + ['gold']].to_numpy(float)
    if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError('Missing or fractional labels')
    labels = df[source_columns].to_numpy(dtype=int)
    gold = df['gold'].to_numpy(dtype=int)
    k = len(json.loads(Path(str(path) + '.meta.json').read_text())['classes'])
    if gold.min() < 0 or gold.max() >= k or labels.min() < -1 or labels.max() >= k:
        raise ValueError('Labels outside declared vocabulary')
    if with_text and ('text' not in df or df['text'].isna().any()):
        raise ValueError(f'Missing text: {name}')
    return df, labels, gold, k


def streams(root, seed, reps):
    data, pre, pac = np.random.SeedSequence([root, seed]).spawn(3)
    return data, pre, pac.spawn(reps)


def changed_diagnostics(pred, baseline, gold, uncertainty, threshold, already):
    changed = pred != baseline
    if already:
        changed[np.fromiter(already, dtype=int)] = False
    ids = np.flatnonzero(changed)
    n = len(ids)
    below = uncertainty[ids] < threshold
    equal = uncertainty[ids] == threshold
    above = uncertainty[ids] > threshold
    corrected = (baseline[ids] != gold[ids]) & (pred[ids] == gold[ids])
    harmed = (baseline[ids] == gold[ids]) & (pred[ids] != gold[ids])
    return {
        'changed': n,
        'corrected': int(corrected.sum()),
        'introduced_errors': int(harmed.sum()),
        'net_corrected': int(corrected.sum() - harmed.sum()),
        'changed_below': int(below.sum()), 'changed_at': int(equal.sum()),
        'changed_above': int(above.sum()),
        'share_changed_below': float(below.mean()) if n else None,
        'share_changed_at': float(equal.mean()) if n else None,
        'share_changed_le': float((below | equal).mean()) if n else None,
        'share_changed_above': float(above.mean()) if n else None,
        'corrected_le': int((corrected & (below | equal)).sum()),
        'introduced_errors_le': int((harmed & (below | equal)).sum()),
    }


def check_pac_accounting(info, uncertainty, already, n, m, seed):
    # Independent set calculation, not the implementation's returned counters.
    draws = np.random.default_rng(seed).integers(0, n, m)
    audit = set(map(int, already))
    u = np.asarray(uncertainty, dtype=float).copy()
    if audit:
        u[list(audit)] = -1.0
    calibration = set(map(int, np.unique(draws)))
    reviewed = set(map(int, np.flatnonzero(u >= info['u_hat'])))
    union = audit | calibration | reviewed
    if len(union) != len(audit) + info['expert_new']:
        raise AssertionError('Union-based expert accounting mismatch')
    if info['calibration_new'] != len(calibration - audit):
        raise AssertionError('Calibration accounting mismatch')
    if info['threshold_new'] != len(reviewed - audit - calibration):
        raise AssertionError('Threshold accounting mismatch')
    return {
        'unique_calibration_draws': len(calibration),
        'calibration_reused_audits': len(calibration & audit),
        'minimum_fixed_cost': len(audit | calibration),
        'draws_sha256': array_hash(draws),
    }


def summarize_rows(rows, out_dir, comparison_pairs=None):
    out_dir = Path(out_dir)
    df = pd.DataFrame(rows)
    if not len(df):
        raise ValueError('No results to summarize')
    df.to_csv(out_dir / 'rows.csv', index=False)
    metrics = [c for c in ['expert_total', 'pre_audits', 'calibration_new', 'threshold_new',
                          'error', 'changed', 'corrected', 'introduced_errors', 'net_corrected',
                          'share_changed_below', 'share_changed_at', 'share_changed_le',
                          'groups_certified', 'harmful_vs_frozen_ref', 'paired_cost_delta'] if c in df]
    summary = df.groupby(['pool', 'method'])[metrics].mean()
    summary['runs'] = df.groupby(['pool', 'method']).size()
    summary['seeds'] = df.groupby(['pool', 'method'])['seed'].nunique()
    summary['observed_final_error_violations'] = df.groupby(['pool', 'method'])['violated'].sum()
    summary.to_csv(out_dir / 'summary.csv')
    p, _ = protocol(validate=False)
    rng = np.random.default_rng(p['parameters']['bootstrap_seed'])
    comparisons = []
    for pool, part in df.groupby('pool', sort=True):
        wide = part.pivot(index=['seed', 'rep'], columns='method', values='expert_total')
        pairs = comparison_pairs
        if pairs is None:
            pairs = [(m, 'same_audit_ds') for m in wide if m not in ('same_audit_ds', 'pac_only')]
            pairs += [('same_audit_ds', 'pac_only')]
        for arm, reference in pairs:
            if arm not in wide or reference not in wide:
                continue
            vals = (wide[arm] - wide[reference]).groupby(level='seed').mean().to_numpy()
            if not np.isfinite(vals).all():
                raise ValueError('Unpaired or missing results')
            boots = rng.choice(vals, size=(p['parameters']['bootstrap_reps'], len(vals)), replace=True).mean(1)
            low, high = np.quantile(boots, [.025, .975])
            n = int(part.n.iloc[0])
            comparisons.append({'pool': pool, 'method': arm, 'reference': reference,
                                'seeds': len(vals), 'mean_cost_delta': float(vals.mean()),
                                'ci95_low': float(low), 'ci95_high': float(high),
                                'saving_fraction': float(-vals.mean()/n),
                                'mean_saving_meets_1pct': bool(-vals.mean() >= .01*n),
                                'ci_lower_saving_meets_1pct': bool(-high >= .01*n),
                                'interval_scope': 'algorithmic seed randomness conditional on fixed pool'})
    pd.DataFrame(comparisons).to_csv(out_dir / 'paired_differences.csv', index=False)
    return summary, comparisons
