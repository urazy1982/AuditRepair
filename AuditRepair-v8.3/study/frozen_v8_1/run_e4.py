"""Correctness-conditioned synthetic coupling sweep (amended E4).

Run after protocol/code review: python run_e4.py --workers 3

Each of the 40 synthetic-pool/preprocessing seeds is frozen once. Five score
permutations preserve its exact correct/error score multisets, and all arms and
eta settings reuse the same five PAC draw streams. Eta is a rank-mixing weight,
not an asserted Pearson correlation. The score intervention uses synthetic gold
and is a mechanism experiment, not a deployable scoring method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from audit_repair.core import (Oracle, candidate_groups_disjoint, dawid_skene,
                               run_group_repair, uncertainty_fill)
from audit_repair.pac import pac_labels
from audit_repair.simulate import Confusion, simulate
from experiment_common import (ROOT, array_hash, atomic_json, changed_diagnostics,
                               check_pac_accounting, clean_json, file_hash, protocol)


METHODS = ('same_audit_ds', 'guarded_fixed_scores', 'O1_shortlist')
SHARED_SOURCE_COUNTS = (3, 4)  # Both settings retain four total source columns.
SLICE_FRACTIONS = (0.2, 0.35)  # Fractions within true class 0, not class prevalence.
N_ITEMS = 5000
N_CLASSES = 4
FLIP_PROBABILITY = 0.85
SEED_NAMESPACE = 404
SCHEMA_VERSION = 1


def case_name(shared_sources, slice_frac):
    if shared_sources not in SHARED_SOURCE_COUNTS or slice_frac not in SLICE_FRACTIONS:
        raise ValueError('E4 setting outside the frozen simulation specification')
    return f'shared{shared_sources}_slice{int(round(100 * slice_frac)):02d}'


def e4_streams(root_seed, shared_sources, slice_frac, seed, reps):
    """Independent named children, reused without mutation across paired arms."""
    case_name(shared_sources, slice_frac)
    if seed < 0 or reps < 1:
        raise ValueError('Invalid seed or repetition count')
    entropy = [int(root_seed), SEED_NAMESPACE, int(shared_sources),
               int(round(100 * slice_frac)), int(seed)]
    data, preprocessing, coupling, pac = np.random.SeedSequence(entropy).spawn(4)
    return data, preprocessing, coupling, pac.spawn(reps), entropy


def disagreement_fraction(labels):
    """1 - modal valid-vote share; fewer than two observed votes gives zero."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError('Source labels must be a nonempty-width matrix')
    if not np.isfinite(labels).all() or np.any(labels < -1) or np.any(labels != np.floor(labels)):
        raise ValueError('Source votes must be integer labels or -1 for missing')
    valid = labels >= 0
    counts = ((labels[:, :, None] == labels[:, None, :]) & valid[:, None, :]).sum(2)
    nvalid = valid.sum(1)
    result = np.zeros(len(labels), dtype=float)
    use = nvalid >= 2
    result[use] = 1.0 - counts[use].max(1) / nvalid[use]
    return result


def _audit_mask(n, audits):
    mask = np.zeros(n, dtype=bool)
    ids = np.asarray(list(audits), dtype=int)
    if len(ids):
        if np.any(ids < 0) or np.any(ids >= n):
            raise ValueError('Audited index outside the population')
        mask[ids] = True
    return mask


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RankStratum:
    correct: bool
    indices: np.ndarray
    sorted_scores: np.ndarray
    random_rank: np.ndarray
    disagreement_rank: np.ndarray


@dataclass(frozen=True)
class ScoreCoupling:
    original_scores: np.ndarray
    correctness: np.ndarray
    audited: np.ndarray
    disagreement: np.ndarray
    strata: tuple[RankStratum, ...]

    def scores(self, eta):
        """Assign the same ascending score multiset to a mixed-rank ordering."""
        eta = float(eta)
        if not np.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError('eta must be in [0,1]')
        result = self.original_scores.copy()
        for stratum in self.strata:
            mixture = (1.0 - eta) * stratum.random_rank + eta * stratum.disagreement_rank
            # lexsort's last key is primary; random ranks resolve all mixture ties.
            order = np.lexsort((stratum.indices, stratum.random_rank, mixture))
            result[stratum.indices[order]] = stratum.sorted_scores
        self.assert_preserved(result)
        return result

    def assert_preserved(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.original_scores.shape or not np.isfinite(scores).all():
            raise AssertionError('Invalid permuted score array')
        if not np.array_equal(scores[self.audited], self.original_scores[self.audited]):
            raise AssertionError('Coupling intervention changed known audit scores')
        for stratum in self.strata:
            if not np.array_equal(np.sort(scores[stratum.indices]), stratum.sorted_scores):
                raise AssertionError('Correctness-conditioned score multiset changed')

    def manifest(self):
        return {
            'original_scores_sha256': array_hash(self.original_scores),
            'correctness_sha256': array_hash(self.correctness),
            'disagreement_sha256': array_hash(self.disagreement),
            'audited_mask_sha256': array_hash(self.audited),
            'strata': [
                {'correct': s.correct, 'n': len(s.indices),
                 'indices_sha256': array_hash(s.indices),
                 'score_multiset_sha256': array_hash(s.sorted_scores),
                 'random_rank_sha256': array_hash(s.random_rank),
                 'disagreement_rank_sha256': array_hash(s.disagreement_rank)}
                for s in self.strata
            ],
            'tie_rule': 'average disagreement ranks; mixture ties use fixed random ordinal rank then item index',
        }


def make_score_coupling(scores, baseline, gold, disagreement, audits, seed):
    scores = np.asarray(scores, dtype=float)
    baseline, gold = np.asarray(baseline), np.asarray(gold)
    disagreement = np.asarray(disagreement, dtype=float)
    if scores.ndim != 1 or not (scores.shape == baseline.shape == gold.shape == disagreement.shape):
        raise ValueError('Scores, labels, and disagreement must be matching vectors')
    if not np.isfinite(scores).all() or not np.isfinite(disagreement).all():
        raise ValueError('Scores and disagreement must be finite')
    if np.any((disagreement < 0) | (disagreement > 1)):
        raise ValueError('Disagreement fractions must be in [0,1]')
    audited = _audit_mask(len(gold), audits)
    if np.any((scores[~audited] < 0) | (scores[~audited] > 1)):
        raise ValueError('Unaudited uncertainty scores must be probabilities')
    if not np.array_equal(baseline[audited], gold[audited]):
        raise ValueError('Baseline predictions are not clamped on audited items')
    correct = baseline == gold
    random_values = np.random.default_rng(seed).random(len(gold))
    strata = []
    for correctness in (False, True):
        ids = np.flatnonzero((correct == correctness) & ~audited)
        size = len(ids)
        random_order = np.lexsort((ids, random_values[ids]))
        random_rank = np.empty(size, dtype=float)
        random_rank[random_order] = np.arange(size, dtype=float) / max(size - 1, 1)
        disagreement_rank = ((rankdata(disagreement[ids], method='average') - 1.0)
                             / max(size - 1, 1))
        strata.append(RankStratum(bool(correctness), _readonly(ids),
                                 _readonly(np.sort(scores[ids])), _readonly(random_rank),
                                 _readonly(disagreement_rank)))
    return ScoreCoupling(_readonly(scores), _readonly(correct), _readonly(audited),
                         _readonly(disagreement), tuple(strata))


def _correlation(x, y):
    """Return an explicit reason instead of NaN for an undefined correlation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Correlation requires matching finite vectors')
    if len(x) < 2:
        return None, 'fewer_than_two_items'
    if np.ptp(x) == 0 and np.ptp(y) == 0:
        return None, 'both_constant'
    if np.ptp(x) == 0:
        return None, 'constant_scores'
    if np.ptp(y) == 0:
        return None, 'constant_disagreement'
    xc, yc = x - x.mean(), y - y.mean()
    value = np.dot(xc / np.linalg.norm(xc), yc / np.linalg.norm(yc))
    return float(np.clip(value, -1, 1)), None


def score_quality(scores, correctness, audited):
    """Baseline error-score AUC and Brier loss, on unaudited items only."""
    use = ~np.asarray(audited, dtype=bool)
    scores = np.asarray(scores, dtype=float)[use]
    errors = ~np.asarray(correctness, dtype=bool)[use]
    n_error, n_correct = int(errors.sum()), int((~errors).sum())
    if n_error and n_correct:
        ranks = rankdata(scores, method='average')
        auc = (ranks[errors].sum() - n_error * (n_error + 1) / 2) / (n_error * n_correct)
        reason = None
    else:
        auc, reason = None, 'only_one_correctness_class' if len(scores) else 'no_unaudited_items'
    return {'n_unaudited': len(scores), 'n_errors': n_error, 'n_correct': n_correct,
            'baseline_error_auc': float(auc) if auc is not None else None,
            'baseline_error_auc_undefined': reason,
            'baseline_error_brier': float(np.mean((scores - errors) ** 2)) if len(scores) else None}


def coupling_diagnostics(coupling, scores):
    coupling.assert_preserved(scores)
    result = score_quality(scores, coupling.correctness, coupling.audited)
    masks = {'unaudited': ~coupling.audited,
             'correct': ~coupling.audited & coupling.correctness,
             'incorrect': ~coupling.audited & ~coupling.correctness}
    for name, mask in masks.items():
        x, y = np.asarray(scores)[mask], coupling.disagreement[mask]
        pearson, pearson_reason = _correlation(x, y)
        spearman, spearman_reason = _correlation(rankdata(x), rankdata(y))
        result.update({f'{name}_n': int(mask.sum()),
                       f'{name}_pearson_rho': pearson,
                       f'{name}_pearson_undefined': pearson_reason,
                       f'{name}_spearman_rho': spearman,
                       f'{name}_spearman_undefined': spearman_reason})
    original = score_quality(coupling.original_scores, coupling.correctness, coupling.audited)
    if result['baseline_error_auc'] != original['baseline_error_auc']:
        raise AssertionError('AUC changed under correctness-conditioned score permutation')
    brier, reference_brier = result['baseline_error_brier'], original['baseline_error_brier']
    if brier is not None and not np.isclose(brier, reference_brier, rtol=0, atol=1e-14):
        raise AssertionError('Brier loss changed beyond floating-point summation tolerance')
    result['score_multisets_exactly_preserved'] = True
    return result


def shortlist_oracle(baseline, gold, audits, shortlist):
    """Apply a frozen group only when its final unaudited net gain is positive."""
    baseline, gold = np.asarray(baseline), np.asarray(gold)
    labels = baseline.copy()
    audited = set(map(int, audits))
    used = set()
    records = []
    for rank, group in enumerate(shortlist):
        members = [int(i) for i in group.items]
        if len(set(members)) != len(members) or used.intersection(members):
            raise ValueError('O1 requires the original disjoint frozen shortlist')
        used.update(members)
        remaining = np.array([i for i in members if i not in audited], dtype=int)
        gain = int((gold[remaining] == group.alt).sum() - (gold[remaining] == baseline[remaining]).sum())
        if gain > 0:
            labels[remaining] = group.alt
        records.append({'rank': rank, 'cur': int(group.cur), 'alt': int(group.alt),
                        'members': members, 'size': len(members),
                        'remainder_size': len(remaining), 'same_audit_net_gain': gain,
                        'oracle_applied': gain > 0})
    return labels, records


@dataclass
class FrozenPreprocessing:
    baseline: np.ndarray
    guarded: np.ndarray
    oracle_shortlist: np.ndarray
    scores: np.ndarray
    audits: dict
    reference: np.ndarray
    shortlist: list
    diagnostics: dict


def legacy_preprocessing(labels, gold, n_classes, parameters, seed):
    """Run legacy G0 once; none of its audit choices uses the E4 score sweep."""
    a = parameters
    result = run_group_repair(
        labels, n_classes, gold, a['B'], alpha=a['alpha_repair'], seed=seed,
        ref_share=a['warmup'] / a['B'], min_size=a['min_group_size'],
        max_groups=a['max_groups'], group_share=a['group_share_remaining'],
        k_max=a['max_audits_per_group'], max_audit_frac=a['max_group_audit_fraction'],
        futility_after=a['futility_after'], futility_mean=a['futility_mean'],
        stop_after_fails=a['stop_after_fails'], disjoint=True, order='size')
    guarded, info = result['guarded']
    audits = {int(i): int(y) for i, y in info['audited_labels'].items()}
    post = info['post']
    baseline, scores = post.argmax(1), 1.0 - post.max(1)
    if audits:
        ids = np.array(sorted(audits), dtype=int)
        if not np.array_equal(baseline[ids], gold[ids]) or not np.array_equal(guarded[ids], gold[ids]):
            raise AssertionError('Preprocessing failed to preserve known gold labels')
        scores[ids] = -1.0
    # Reconstruct the identical deterministic warm-up reference and frozen shortlist.
    warm = {}
    uncertainty_fill(labels, n_classes, Oracle(gold, a['warmup']), warm)
    post_ref = dawid_skene(labels, n_classes, warm)
    reference = post_ref.argmax(1)
    for i, y in warm.items():
        reference[i] = y
    shortlist = sorted(candidate_groups_disjoint(labels, reference, post_ref,
                                                 min_size=a['min_group_size']),
                       key=lambda group: -len(group.items))[:a['max_groups']]
    oracle, groups = shortlist_oracle(baseline, gold, audits, shortlist)
    guarded_harms_vs_baseline = 0
    for record in info['log']:
        if record['guarded']:
            rem = np.array([int(i) for i in record['members'] if int(i) not in audits], dtype=int)
            gain = int((gold[rem] == record['alt']).sum() - (gold[rem] == baseline[rem]).sum())
            guarded_harms_vs_baseline += int(gain <= 0)
    diagnostics = {
        'pre_audits': len(audits), 'groups_certified': info['groups_repaired'],
        'harmful_vs_frozen_ref': info['harmful_group_repairs'],
        'harmful_vs_same_audit_baseline': guarded_harms_vs_baseline,
        'shortlist_groups': len(groups),
        'beneficial_shortlist_groups': sum(g['oracle_applied'] for g in groups),
        'baseline_errors': int((baseline != gold).sum()),
        'guarded_net_gain_vs_baseline': int((guarded == gold).sum() - (baseline == gold).sum()),
        'oracle_net_gain_vs_baseline': int((oracle == gold).sum() - (baseline == gold).sum()),
    }
    return FrozenPreprocessing(baseline, guarded, oracle, scores, audits, reference,
                               groups, diagnostics)


def reach_diagnostics(pred, baseline, gold, scores, threshold, audits):
    """Reach of frozen changes relative to the same-score baseline first failure."""
    result = changed_diagnostics(pred, baseline, gold, scores, threshold, audits)
    eligible = ~_audit_mask(len(gold), audits)
    corrected = (baseline != gold) & (pred == gold) & eligible
    introduced = (baseline == gold) & (pred != gold) & eligible
    below, at = scores < threshold, scores == threshold
    for name, mask in (('corrected', corrected), ('introduced_errors', introduced)):
        denominator = int(mask.sum())
        result[f'{name}_below'] = int((mask & below).sum())
        result[f'{name}_at'] = int((mask & at).sum())
        result[f'{name}_above'] = int((mask & (scores > threshold)).sum())
        result[f'share_{name}_below'] = float((mask & below).sum() / denominator) if denominator else None
        result[f'share_{name}_le'] = float((mask & (below | at)).sum() / denominator) if denominator else None
    return result


def _code_hashes():
    names = ('run_e4.py', 'experiment_common.py', 'audit_repair/core.py',
             'audit_repair/pac.py', 'audit_repair/simulate.py')
    return {name: file_hash(ROOT / name) for name in names}


def _payload_hash(payload):
    body = {key: value for key, value in payload.items() if key != 'payload_sha256'}
    encoded = json.dumps(clean_json(body), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_checkpoint(path, name, seed, protocol_sha, code, etas, reps):
    cached = json.loads(path.read_text())
    if cached.get('protocol_sha256') != protocol_sha or cached.get('code_sha256') != code:
        raise ValueError(f'Existing E4 checkpoint has different protocol/code: {path}')
    if cached.get('payload_sha256') != _payload_hash(cached):
        raise ValueError(f'E4 checkpoint content checksum mismatch: {path}')
    if (cached.get('schema_version') != SCHEMA_VERSION or cached.get('status') != 'complete'
            or cached.get('case') != name or cached.get('seed') != seed):
        raise ValueError(f'Incomplete or mismatched E4 checkpoint: {path}')
    expected = {(float(eta), rep, method) for eta in etas for rep in range(reps) for method in METHODS}
    rows = cached.get('rows', [])
    observed = {(row.get('eta'), row.get('rep'), row.get('method')) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(f'Missing or duplicate E4 checkpoint rows: {path}')
    for row in rows:
        if row.get('case') != name or row.get('seed') != seed:
            raise ValueError(f'Wrong task identity inside E4 checkpoint: {path}')
        for key in ('u_hat', 'baseline_u_hat'):
            if row.get(key) == 'infinity':
                row[key] = float('inf')
            elif row.get(key) == '-infinity':
                row[key] = float('-inf')
    return cached


def one_seed(shared_sources, slice_frac, seed, out_dir):
    """Run or reuse one complete (shared-source count, slice fraction, seed) task."""
    started = time.monotonic()
    p, protocol_sha = protocol(validate=False)
    a, e4 = p['parameters'], p['E4']
    if not 0 <= seed < e4['seeds']:
        raise ValueError('Seed outside frozen E4 range')
    etas = tuple(float(eta) for eta in e4['eta'])
    if len(set(etas)) != len(etas) or 0.0 not in etas or 1.0 not in etas:
        raise ValueError('E4 requires unique eta values including 0 and 1')
    if any(not np.isfinite(eta) or not 0 <= eta <= 1 for eta in etas):
        raise ValueError('Invalid frozen eta grid')
    reps = int(e4['pac_reps'])
    name = case_name(shared_sources, slice_frac)
    code = _code_hashes()
    dest = Path(out_dir) / 'checkpoints' / name / f'seed_{seed:02d}.json'
    if dest.exists():
        return _read_checkpoint(dest, name, seed, protocol_sha, code, etas, reps), True

    data_stream, pre_stream, score_stream, pac_streams, entropy = e4_streams(
        a['root_seed'], shared_sources, slice_frac, seed, reps)
    labels, gold = simulate(n=N_ITEMS, K=N_CLASSES, seed=data_stream,
                            confusions=[Confusion(0, 1, slice_frac, tuple(range(shared_sources)),
                                                  flip_prob=FLIP_PROBABILITY)])
    state = legacy_preprocessing(labels, gold, N_CLASSES, a, pre_stream)
    coupling = make_score_coupling(state.scores, state.baseline, gold,
                                   disagreement_fraction(labels), state.audits, score_stream)
    scores_by_eta = {eta: coupling.scores(eta) for eta in etas}
    diagnostics_by_eta = {eta: coupling_diagnostics(coupling, scores_by_eta[eta]) for eta in etas}
    predictions = {'same_audit_ds': state.baseline, 'guarded_fixed_scores': state.guarded,
                   'O1_shortlist': state.oracle_shortlist}
    frozen = {
        'source_labels_sha256': array_hash(labels), 'gold_sha256': array_hash(gold),
        'reference_labels_sha256': array_hash(state.reference),
        'audit_ids_sha256': array_hash(np.array(sorted(state.audits), dtype=np.int64)),
        'label_vectors': {method: array_hash(pred) for method, pred in predictions.items()},
        'scores_by_eta': {format(eta, '.12g'): array_hash(scores) for eta, scores in scores_by_eta.items()},
        'score_coupling': coupling.manifest(),
    }
    # Every label vector, rank ordering and score vector is recorded before PAC evaluation.
    frozen_utc = datetime.now(timezone.utc).isoformat()
    rows = []
    for rep, pac_stream in enumerate(pac_streams):
        draws = np.random.default_rng(pac_stream).integers(0, len(gold), a['m'])
        draws_hash = array_hash(draws)
        already_or_calibrated = _audit_mask(len(gold), state.audits)
        already_or_calibrated[draws] = True
        for eta in etas:
            scores = scores_by_eta[eta]
            final_base, info_base = pac_labels(state.baseline, scores, gold, eps=a['epsilon'],
                                               alpha=a['alpha_final'], m=a['m'],
                                               already=state.audits, seed=pac_stream)
            baseline_cost = len(state.audits) + info_base['expert_new']
            for method in METHODS:
                pred = predictions[method]
                if method == 'same_audit_ds':
                    final, info = final_base, info_base
                else:
                    final, info = pac_labels(pred, scores, gold, eps=a['epsilon'],
                                             alpha=a['alpha_final'], m=a['m'],
                                             already=state.audits, seed=pac_stream)
                accounting = check_pac_accounting(info, scores, state.audits, len(gold), a['m'], pac_stream)
                if accounting['draws_sha256'] != draws_hash:
                    raise AssertionError('E4 lost paired calibration draws')
                if not np.array_equal(final[already_or_calibrated], gold[already_or_calibrated]):
                    raise AssertionError('PAC failed to retain purchased gold labels')
                kept = ~already_or_calibrated & (scores < info['u_hat'])
                corrected = (state.baseline != gold) & (pred == gold)
                introduced = (state.baseline == gold) & (pred != gold)
                expert_total = len(state.audits) + info['expert_new']
                guarded_arm = method == 'guarded_fixed_scores'
                row = {
                    'case': name, 'pool': name, 'shared_sources': shared_sources,
                    'slice_frac': slice_frac, 'source_count': labels.shape[1], 'n': len(gold),
                    'seed': seed, 'rep': rep, 'eta': eta, 'method': method,
                    'pre_audits': len(state.audits), 'expert_total': expert_total,
                    'baseline_expert_total': baseline_cost,
                    'paired_cost_delta': expert_total - baseline_cost,
                    'saving_fraction': (baseline_cost - expert_total) / len(gold),
                    'baseline_u_hat': info_base['u_hat'], **info, **accounting,
                    'violated': bool(info['error'] > a['epsilon']),
                    'labels_sha256': frozen['label_vectors'][method],
                    'scores_sha256': frozen['scores_by_eta'][format(eta, '.12g')],
                    'frozen_label_PAC_interpretation': True,
                    'groups_certified': state.diagnostics['groups_certified'] if guarded_arm else 0,
                    'groups_applied': (state.diagnostics['groups_certified'] if guarded_arm else
                                       state.diagnostics['beneficial_shortlist_groups'] if method == 'O1_shortlist' else 0),
                    'harmful_vs_frozen_ref': state.diagnostics['harmful_vs_frozen_ref'] if guarded_arm else None,
                    'harmful_vs_same_audit_baseline': state.diagnostics['harmful_vs_same_audit_baseline'] if guarded_arm else 0,
                    **reach_diagnostics(pred, state.baseline, gold, scores, info_base['u_hat'], state.audits),
                    'retained_corrected_without_expert_query': int((kept & corrected).sum()),
                    'retained_introduced_errors': int((kept & introduced).sum()),
                    'final_corrected_vs_pac_baseline': int(((final_base != gold) & (final == gold)).sum()),
                    'final_introduced_errors_vs_pac_baseline': int(((final_base == gold) & (final != gold)).sum()),
                    **diagnostics_by_eta[eta],
                }
                rows.append(row)
    # Detect accidental mutation or a concurrent edit instead of committing mixed inputs.
    if any(array_hash(pred) != frozen['label_vectors'][method] for method, pred in predictions.items()):
        raise AssertionError('A frozen label vector changed during E4 evaluation')
    if any(array_hash(scores) != frozen['scores_by_eta'][format(eta, '.12g')]
           for eta, scores in scores_by_eta.items()):
        raise AssertionError('A frozen score vector changed during E4 evaluation')
    if _code_hashes() != code or protocol(validate=False)[1] != protocol_sha:
        raise ValueError('Code or protocol changed during E4 evaluation')
    answer = {
        'schema_version': SCHEMA_VERSION, 'experiment': 'E4', 'status': 'complete',
        'case': name, 'shared_sources': shared_sources, 'slice_frac': slice_frac, 'seed': seed,
        'protocol_sha256': protocol_sha, 'code_sha256': code,
        'seed_entropy': entropy,
        'stream_children': {'simulation': 0, 'preprocessing': 1, 'coupling_ranks': 2, 'PAC_parent': 3},
        'pac_spawn_keys': [list(stream.spawn_key) for stream in pac_streams],
        'frozen_arrays_utc': frozen_utc, 'frozen_arrays': frozen,
        'audited_labels': state.audits, 'preprocessing': state.diagnostics,
        'shortlist': state.shortlist,
        'coupling_diagnostics': [{'eta': eta, **diagnostics_by_eta[eta]} for eta in etas],
        'rows': rows, 'wall_seconds': time.monotonic() - started,
        'interpretation': 'Gold-conditioned score intervention frozen before PAC; eta is not a target correlation.',
    }
    answer['payload_sha256'] = _payload_hash(answer)
    atomic_json(dest, answer)
    return answer, False


def _mean_interval(values, indices):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError('Bootstrap estimates require finite paired seed means')
    if len(values) < 2:
        return float(values.mean()), None, None
    lo, hi = np.quantile(values[indices].mean(1), [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def summarize_e4(rows, p):
    """Paired repetition means first; jointly resample whole synthetic seed clusters."""
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No E4 results to summarize')
    keys = ['case', 'seed', 'rep', 'eta', 'method']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate E4 result rows')
    etas, reps = tuple(map(float, p['E4']['eta'])), int(p['E4']['pac_reps'])
    expected = {(eta, rep, method) for eta in etas for rep in range(reps) for method in METHODS}
    for (_, _), part in df.groupby(['case', 'seed']):
        actual = set(zip(part.eta, part.rep, part.method))
        if len(part) != len(expected) or actual != expected:
            raise ValueError('Unpaired or incomplete E4 seed cluster')
    metric_names = [
        'expert_total', 'pre_audits', 'calibration_new', 'threshold_new', 'error',
        'paired_cost_delta', 'saving_fraction', 'changed', 'corrected', 'introduced_errors',
        'net_corrected', 'share_changed_below', 'share_changed_le', 'share_corrected_below',
        'share_corrected_le', 'retained_corrected_without_expert_query', 'retained_introduced_errors',
        'final_corrected_vs_pac_baseline', 'final_introduced_errors_vs_pac_baseline',
        'groups_certified', 'groups_applied', 'harmful_vs_frozen_ref', 'harmful_vs_same_audit_baseline',
        'unaudited_pearson_rho', 'unaudited_spearman_rho', 'correct_pearson_rho',
        'incorrect_pearson_rho', 'baseline_error_auc', 'baseline_error_brier']
    metric_names = [name for name in metric_names if name in df]
    for metric in metric_names:
        df[metric] = pd.to_numeric(df[metric], errors='raise')
    group = df.groupby(['case', 'eta', 'method'], sort=True)
    summary = group[metric_names].mean()
    summary['runs'] = group.size()
    summary['seed_clusters'] = group.seed.nunique()
    summary['observed_final_error_violations'] = group.violated.sum()
    comparisons, reach_intervals = [], []
    n_boot = int(p['parameters']['bootstrap_reps'])
    if n_boot < 1:
        raise ValueError('Bootstrap repetition count must be positive')
    for name, part in df.groupby('case', sort=True):
        settings = part[['shared_sources', 'slice_frac', 'n']].drop_duplicates()
        if len(settings) != 1:
            raise ValueError('Inconsistent synthetic setting or population size within case')
        settings = settings.iloc[0]
        seeds = np.array(sorted(part.seed.unique()))
        cluster_cost = part.groupby(['seed', 'eta', 'method']).expert_total.mean().unstack(['eta', 'method']).reindex(seeds)
        if not np.isfinite(cluster_cost.to_numpy()).all():
            raise ValueError('Missing paired seed-level costs')
        rng = np.random.default_rng(np.random.SeedSequence([
            p['parameters']['bootstrap_seed'], SEED_NAMESPACE,
            int(settings.shared_sources), int(round(100 * settings.slice_frac))]))
        bootstrap_indices = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
        n = int(settings.n)
        for method in METHODS[1:]:
            effects = {eta: (cluster_cost[(eta, method)] - cluster_cost[(eta, METHODS[0])]).to_numpy()
                       for eta in etas}
            for eta in etas:
                for comparison in ('within_eta', 'incremental_effect_change_from_eta0'):
                    if comparison != 'within_eta' and eta == 0:
                        continue
                    values = effects[eta] if comparison == 'within_eta' else effects[eta] - effects[0.0]
                    estimate, low, high = _mean_interval(values, bootstrap_indices)
                    comparisons.append({
                        'case': name, 'eta': eta, 'method': method, 'comparison': comparison,
                        'reference_method': METHODS[0], 'reference_eta': eta if comparison == 'within_eta' else 0.0,
                        'seed_clusters': len(seeds), 'pac_repetitions_per_seed': reps,
                        'mean_cost_delta': estimate, 'ci95_low': low, 'ci95_high': high,
                        'mean_saving_fraction': -estimate / n,
                        'ci95_saving_low': -high / n if high is not None else None,
                        'ci95_saving_high': -low / n if low is not None else None,
                        'mean_saving_meets_1pct': -estimate >= 0.01 * n if comparison == 'within_eta' else None,
                        'ci_lower_saving_meets_1pct': -high >= 0.01 * n if high is not None and comparison == 'within_eta' else None,
                        'interval_unavailable_reason': 'fewer_than_two_seed_clusters' if len(seeds) < 2 else None,
                        'interval_scope': 'seed-cluster variability under the specified synthetic generator and annotation mechanism',
                    })
            # Zero-opportunity seeds are retained in costs. Reach fractions condition on
            # positive opportunity and explicitly report the eligible cluster count.
            for metric in ('share_changed_below', 'share_changed_le', 'share_corrected_below', 'share_corrected_le'):
                if metric not in part:
                    continue
                values = (part[part.method == method].groupby(['seed', 'eta'])[metric]
                          .mean().unstack('eta').reindex(seeds))
                defined = values.notna()
                if not np.array_equal(defined.to_numpy(), np.repeat(defined.iloc[:, :1].to_numpy(), len(etas), axis=1)):
                    raise ValueError('Reach denominator changed across eta for a frozen label vector')
                eligible = values.loc[defined.iloc[:, 0]]
                size = len(eligible)
                reach_indices = rng.integers(0, size, size=(n_boot, size)) if size else None
                for eta in etas:
                    estimate, low, high = _mean_interval(eligible[eta].to_numpy(), reach_indices) if size else (None, None, None)
                    reach_intervals.append({
                        'case': name, 'eta': eta, 'method': method, 'metric': metric,
                        'mean': estimate, 'ci95_low': low, 'ci95_high': high,
                        'eligible_seed_clusters': size, 'all_seed_clusters': len(seeds),
                        'pac_repetitions_per_seed': reps,
                        'interval_unavailable_reason': 'no_positive_denominator' if size == 0 else 'fewer_than_two_eligible_seed_clusters' if size < 2 else None,
                        'interval_scope': 'mean seed-level reach conditional on a positive denominator; eligible whole seeds resampled jointly across eta',
                    })
    return summary.reset_index(), pd.DataFrame(comparisons), pd.DataFrame(reach_intervals)


def run_e4(*, workers=3, out_dir=None, seed_start=0, seed_stop=None):
    """Callable driver; a seed subset is clearly marked as partial protocol scope."""
    p, protocol_sha = protocol()
    stop = int(p['E4']['seeds']) if seed_stop is None else int(seed_stop)
    if workers < 1 or not 0 <= seed_start < stop <= p['E4']['seeds']:
        raise ValueError('Invalid worker count or E4 seed range')
    out_dir = Path(out_dir) if out_dir is not None else ROOT / 'results' / 'E4'
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(shared, fraction, seed) for shared in SHARED_SOURCE_COUNTS
             for fraction in SLICE_FRACTIONS for seed in range(seed_start, stop)]
    run = {
        'experiment': 'E4', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha, 'code_sha256': _code_hashes(),
        'expected_tasks': len(tasks), 'completed_tasks': 0, 'status': 'running',
        'seed_range': [seed_start, stop], 'eta': p['E4']['eta'],
        'interpretation': 'Full frozen scope requires both shared-source settings, both slice fractions, and all ten seeds.',
    }
    atomic_json(out_dir / 'run_status.json', run)
    rows, states, coupling_rows = [], [], []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(one_seed, shared, fraction, seed, out_dir): (shared, fraction, seed)
                       for shared, fraction, seed in tasks}
            for future in as_completed(futures):
                answer, cached = future.result()
                rows.extend(answer['rows'])
                identity = {'case': answer['case'], 'seed': answer['seed'],
                            'shared_sources': answer['shared_sources'], 'slice_frac': answer['slice_frac']}
                states.append({**identity, **answer['preprocessing']})
                coupling_rows.extend({**identity, **record} for record in answer['coupling_diagnostics'])
                run['completed_tasks'] += 1
                atomic_json(out_dir / 'run_status.json', run)
                print(f"E4 {run['completed_tasks']}/{len(tasks)} {answer['case']} seed={answer['seed']} "
                      f"{'cached' if cached else 'completed'} {answer['wall_seconds']:.1f}s", flush=True)
        rows.sort(key=lambda row: (row['shared_sources'], row['slice_frac'], row['seed'],
                                   row['eta'], row['rep'], METHODS.index(row['method'])))
        summary, paired, reach = summarize_e4(rows, p)
        pd.DataFrame(rows).to_csv(out_dir / 'rows.csv', index=False)
        summary.to_csv(out_dir / 'summary.csv', index=False)
        paired.to_csv(out_dir / 'paired_differences.csv', index=False)
        reach.to_csv(out_dir / 'reach_intervals.csv', index=False)
        pd.DataFrame(states).sort_values(['case', 'seed']).to_csv(out_dir / 'preprocessing.csv', index=False)
        pd.DataFrame(coupling_rows).sort_values(['case', 'seed', 'eta']).to_csv(out_dir / 'coupling_diagnostics.csv', index=False)
        run.update({'status': 'complete', 'rows': len(rows),
                    'full_frozen_scope_complete': seed_start == 0 and stop == p['E4']['seeds'],
                    'completed_utc': datetime.now(timezone.utc).isoformat()})
        atomic_json(out_dir / 'run_status.json', run)
        return run
    except Exception as error:
        run.update({'status': 'failed', 'error': repr(error)})
        atomic_json(out_dir / 'run_status.json', run)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'E4')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-stop', type=int)
    args = parser.parse_args(argv)
    return run_e4(workers=args.workers, out_dir=args.out,
                  seed_start=args.seed_start, seed_stop=args.seed_stop)


if __name__ == '__main__':
    main()
