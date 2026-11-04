"""Deterministic E4 construction/accounting fixtures; no full E4 study outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from audit_repair.core import Group
from experiment_common import clean_json
from run_e4 import (METHODS, FrozenPreprocessing, _correlation, coupling_diagnostics,
                    disagreement_fraction, e4_streams, make_score_coupling, one_seed,
                    reach_diagnostics, score_quality, shortlist_oracle, summarize_e4)


class ScoreCouplingTests(unittest.TestCase):
    def fixture(self):
        gold = np.arange(14) % 2
        baseline = gold.copy()
        baseline[[0, 2, 4, 6, 8, 10]] = 1 - gold[[0, 2, 4, 6, 8, 10]]
        audits = {12: int(gold[12]), 13: int(gold[13])}
        scores = np.array([.1, .01, .3, .05, .3, .1, .6, .15, .7, .3, .9, .4, -1., -1.])
        disagreement = np.array([.0, .5, .25, .25, .75, .0, .5, .75, .25, .5, .0, .25, .5, .25])
        return scores, baseline, gold, disagreement, audits

    def test_valid_vote_disagreement_including_missing_votes(self):
        labels = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 2, 3],
                           [-1, -1, 2, 3], [-1, 1, 1, 1], [-1, -1, -1, 2], [-1, -1, -1, -1]])
        np.testing.assert_array_equal(disagreement_fraction(labels), [0, .25, .5, .75, .5, 0, 0, 0])
        with self.assertRaises(ValueError):
            disagreement_fraction(np.array([[0, 0.5]]))

    def test_exact_stratum_multisets_audits_auc_and_brier_are_preserved(self):
        scores, baseline, gold, disagreement, audits = self.fixture()
        inputs = [array.copy() for array in (scores, baseline, gold, disagreement)]
        coupling = make_score_coupling(scores, baseline, gold, disagreement, audits, 193)
        original_quality = score_quality(scores, coupling.correctness, coupling.audited)
        for eta in (0, .25, .5, .75, 1):
            result = coupling.scores(eta)
            np.testing.assert_array_equal(result[list(audits)], scores[list(audits)])
            for correct in (False, True):
                mask = ((baseline == gold) == correct) & ~coupling.audited
                np.testing.assert_array_equal(np.sort(result[mask]), np.sort(scores[mask]))
            diagnostic = coupling_diagnostics(coupling, result)
            self.assertTrue(diagnostic['score_multisets_exactly_preserved'])
            self.assertEqual(diagnostic['baseline_error_auc'], original_quality['baseline_error_auc'])
            self.assertAlmostEqual(diagnostic['baseline_error_brier'], original_quality['baseline_error_brier'], places=14)
        for actual, original in zip((scores, baseline, gold, disagreement), inputs):
            np.testing.assert_array_equal(actual, original)

    def test_eta_zero_ignores_disagreement_and_eta_one_sorts_it(self):
        scores, baseline, gold, disagreement, audits = self.fixture()
        first = make_score_coupling(scores, baseline, gold, disagreement, audits, 193)
        reversed_disagreement = make_score_coupling(scores, baseline, gold, 1 - disagreement, audits, 193)
        np.testing.assert_array_equal(first.scores(0), reversed_disagreement.scores(0))
        endpoint = first.scores(1)
        for stratum in first.strata:
            ids = stratum.indices
            for i in ids:
                for j in ids:
                    if disagreement[i] < disagreement[j]:
                        self.assertLessEqual(endpoint[i], endpoint[j])
        # Reusing an eta or visiting them in another order cannot consume RNG state.
        saved = first.scores(.25)
        first.scores(1)
        first.scores(.5)
        np.testing.assert_array_equal(first.scores(.25), saved)
        second = make_score_coupling(scores, baseline, gold, disagreement, audits, 193)
        np.testing.assert_array_equal(first.scores(.5), second.scores(.5))

    def test_constant_disagreement_empty_strata_and_undefined_correlations(self):
        scores, baseline, gold, _, audits = self.fixture()
        coupling = make_score_coupling(scores, baseline, gold, np.zeros(len(gold)), audits, 8)
        for eta in (.25, .5, .75, 1):
            np.testing.assert_array_equal(coupling.scores(eta), coupling.scores(0))
        result = coupling_diagnostics(coupling, coupling.scores(1))
        self.assertIsNone(result['unaudited_pearson_rho'])
        self.assertEqual(result['unaudited_pearson_undefined'], 'constant_disagreement')
        all_correct = make_score_coupling(scores, gold, gold, np.zeros(len(gold)), audits, 8)
        quality = coupling_diagnostics(all_correct, all_correct.scores(.5))
        self.assertIsNone(quality['baseline_error_auc'])
        self.assertEqual(quality['incorrect_n'], 0)
        tiny = make_score_coupling(np.array([.2, -1.]), np.array([0, 1]), np.array([1, 1]),
                                   np.array([.5, 0.]), {1: 1}, 3)
        np.testing.assert_array_equal(tiny.scores(.75), [.2, -1.])
        self.assertEqual(_correlation([1], [2]), (None, 'fewer_than_two_items'))

    def test_preservation_check_rejects_cross_stratum_and_audit_changes(self):
        scores, baseline, gold, disagreement, audits = self.fixture()
        coupling = make_score_coupling(scores, baseline, gold, disagreement, audits, 193)
        broken = coupling.scores(.5)
        i, j = next((int(i), int(j)) for i in coupling.strata[0].indices
                    for j in coupling.strata[1].indices if broken[i] != broken[j])
        broken[i], broken[j] = broken[j], broken[i]
        with self.assertRaises(AssertionError):
            coupling.assert_preserved(broken)
        broken = coupling.scores(.5)
        broken[12] = 0
        with self.assertRaises(AssertionError):
            coupling.assert_preserved(broken)
        with self.assertRaises(ValueError):
            coupling.scores(1.01)

    def test_auc_matches_independent_pairwise_tie_count(self):
        scores = np.array([.1, .3, .3, .8, -1.])
        correct = np.array([True, False, True, False, True])
        audited = np.array([False, False, False, False, True])
        quality = score_quality(scores, correct, audited)
        positive, negative = scores[~correct & ~audited], scores[correct & ~audited]
        pairs = [float(a > b) + .5 * float(a == b) for a in positive for b in negative]
        self.assertEqual(quality['baseline_error_auc'], np.mean(pairs))


class OracleAndAccountingTests(unittest.TestCase):
    def test_oracle_uses_final_unaudited_positive_net_gain(self):
        baseline = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        gold = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1])
        groups = [Group(0, 0, 1, np.array([0, 1, 2, 3]), 0),
                  Group(0, 0, 1, np.array([4, 5]), 0),
                  Group(0, 0, 1, np.array([6, 7]), 0),
                  Group(0, 1, 0, np.array([8]), 0)]
        result, records = shortlist_oracle(baseline, gold, {0: 1, 8: 1}, groups)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 0, 0, 0, 0, 1])
        self.assertEqual([record['same_audit_net_gain'] for record in records], [1, 0, -2, 0])
        self.assertEqual([record['oracle_applied'] for record in records], [True, False, False, False])
        with self.assertRaises(ValueError):
            shortlist_oracle(baseline, gold, {}, [groups[0], groups[0]])

    def test_reach_includes_boundary_only_in_le_metric_and_reports_empty(self):
        baseline, gold = np.zeros(4, dtype=int), np.ones(4, dtype=int)
        result = reach_diagnostics(gold, baseline, gold, np.array([.1, .2, .2, .8]), .2, {})
        self.assertEqual(result['corrected_below'], 1)
        self.assertEqual(result['corrected_at'], 2)
        self.assertEqual(result['share_corrected_below'], .25)
        self.assertEqual(result['share_corrected_le'], .75)
        empty = reach_diagnostics(baseline, baseline, gold, np.zeros(4), .2, {})
        self.assertIsNone(empty['share_corrected_le'])

    def test_namespaced_streams_are_replayable_and_separate(self):
        data, pre, coupling, pac, entropy = e4_streams(20261030, 3, .2, 0, 5)
        self.assertEqual(entropy, [20261030, 404, 3, 20, 0])
        draws = [np.random.default_rng(stream).integers(0, 5000, 50) for stream in pac]
        for stream, expected in zip(pac, draws):
            np.testing.assert_array_equal(np.random.default_rng(stream).integers(0, 5000, 50), expected)
        self.assertEqual(len({tuple(stream.spawn_key) for stream in (data, pre, coupling, *pac)}), 8)
        other = e4_streams(20261030, 4, .2, 0, 5)[3][0]
        self.assertFalse(np.array_equal(draws[0], np.random.default_rng(other).integers(0, 5000, 50)))

    def test_checkpoint_resume_and_full_pairing_on_fixed_toy_state(self):
        # simulate() and all preprocessing are replaced, so this does not run an E4 population.
        gold = np.arange(16) % 4
        baseline = gold.copy()
        baseline[[2, 4, 6, 8]] = (gold[[2, 4, 6, 8]] + 1) % 4
        guarded, oracle = baseline.copy(), baseline.copy()
        guarded[2], guarded[3] = gold[2], (gold[3] + 1) % 4
        oracle[[2, 4, 6]] = gold[[2, 4, 6]]
        audits = {0: int(gold[0]), 1: int(gold[1])}
        scores = np.linspace(.01, .8, 16)
        scores[:2] = -1
        labels = np.tile(gold[:, None], (1, 4))
        labels[::2, 0] = (labels[::2, 0] + 1) % 4
        diagnostics = {'groups_certified': 1, 'harmful_vs_frozen_ref': 0,
                       'harmful_vs_same_audit_baseline': 1, 'beneficial_shortlist_groups': 1}
        state = FrozenPreprocessing(baseline, guarded, oracle, scores, audits, baseline.copy(), [], diagnostics)
        p = {'parameters': {'root_seed': 20261030, 'epsilon': .2, 'alpha_final': .1, 'm': 40},
             'E4': {'seeds': 10, 'pac_reps': 2, 'eta': [0, .5, 1]}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('run_e4.protocol', return_value=(p, 'fixture-protocol')):
                with patch('run_e4.simulate', return_value=(labels, gold)) as simulation:
                    with patch('run_e4.legacy_preprocessing', return_value=state) as preprocessing:
                        result, cached = one_seed(3, .2, 0, directory)
                self.assertFalse(cached)
                self.assertEqual(simulation.call_count, 1)
                self.assertEqual(preprocessing.call_count, 1)
                self.assertEqual(len(result['rows']), 3 * 3 * 2)
                for rep in range(2):
                    rows = [row for row in result['rows'] if row['rep'] == rep]
                    self.assertEqual(len({row['draws_sha256'] for row in rows}), 1)
                    self.assertEqual(len({row['minimum_fixed_cost'] for row in rows}), 1)
                for method in METHODS:
                    self.assertEqual(len({row['labels_sha256'] for row in result['rows'] if row['method'] == method}), 1)
                for eta in p['E4']['eta']:
                    self.assertEqual(len({row['scores_sha256'] for row in result['rows'] if row['eta'] == eta}), 1)
                for row in result['rows']:
                    self.assertEqual(row['paired_cost_delta'], row['expert_total'] - row['baseline_expert_total'])
                    self.assertEqual(row['expert_total'], len(audits) + row['calibration_new'] + row['threshold_new'])
                with patch('run_e4.simulate', side_effect=AssertionError('A cached task must not simulate')):
                    resumed, cached = one_seed(3, .2, 0, directory)
                self.assertTrue(cached)
                self.assertEqual(clean_json(resumed), clean_json(result))
                path = Path(directory) / 'checkpoints/shared3_slice20/seed_00.json'
                damaged = json.loads(path.read_text())
                damaged['rows'][0]['expert_total'] += 1
                path.write_text(json.dumps(damaged))
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    one_seed(3, .2, 0, directory)


class ClusterSummaryTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for seed in (0, 1):
            for rep in (0, 1):
                for eta in (0., 1.):
                    guarded_delta = (-5 if rep == 0 else -15) if seed == 0 else (0 if rep == 0 else 4)
                    if eta == 1 and seed == 0:
                        guarded_delta -= 10
                    for method in METHODS:
                        delta = 0 if method == METHODS[0] else guarded_delta if method == METHODS[1] else -20 + 16 * seed
                        rows.append({'case': 'shared3_slice20', 'shared_sources': 3, 'slice_frac': .2,
                                     'n': 1000, 'seed': seed, 'rep': rep, 'eta': eta, 'method': method,
                                     'expert_total': 100 + delta, 'violated': False,
                                     'share_changed_below': None if method == METHODS[0] else .2 + .2 * eta,
                                     'share_changed_le': None if method == METHODS[0] else .4 + .2 * eta,
                                     'share_corrected_below': None, 'share_corrected_le': None})
        p = {'E4': {'eta': [0, 1], 'pac_reps': 2},
             'parameters': {'bootstrap_seed': 20261104, 'bootstrap_reps': 4000}}
        return rows, p

    def test_repetitions_are_averaged_before_paired_seed_bootstrap(self):
        rows, p = self.fixture()
        summary, comparisons, reach = summarize_e4(rows, p)
        within = comparisons[(comparisons.method == METHODS[1]) & (comparisons.eta == 0)
                             & (comparisons.comparison == 'within_eta')].iloc[0]
        # The two independent seed means are -10 and +2; four PAC runs are not four clusters.
        self.assertEqual(within.seed_clusters, 2)
        self.assertEqual(within.mean_cost_delta, -4)
        self.assertEqual((within.ci95_low, within.ci95_high), (-10, 2))
        change = comparisons[(comparisons.method == METHODS[1])
                             & (comparisons.comparison == 'incremental_effect_change_from_eta0')].iloc[0]
        self.assertEqual(change.mean_cost_delta, -5)
        self.assertEqual((change.ci95_low, change.ci95_high), (-10, 0))
        invariant = comparisons[(comparisons.method == METHODS[2])
                                & (comparisons.comparison == 'incremental_effect_change_from_eta0')].iloc[0]
        self.assertEqual((invariant.mean_cost_delta, invariant.ci95_low, invariant.ci95_high), (0, 0, 0))
        empty = reach[reach.metric == 'share_corrected_le']
        self.assertTrue((empty.eligible_seed_clusters == 0).all())
        self.assertTrue(empty['mean'].isna().all())
        self.assertTrue((empty.interval_unavailable_reason == 'no_positive_denominator').all())
        self.assertTrue((summary.seed_clusters == 2).all())

    def test_missing_arm_or_eta_rows_are_rejected(self):
        rows, p = self.fixture()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize_e4(rows[:-1], p)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize_e4(rows + [copy.deepcopy(rows[0])], p)

    def test_one_seed_is_reported_without_degenerate_confidence_interval(self):
        rows, p = self.fixture()
        _, comparisons, _ = summarize_e4([row for row in rows if row['seed'] == 0], p)
        self.assertTrue(comparisons.ci95_low.isna().all())
        self.assertTrue((comparisons.interval_unavailable_reason == 'fewer_than_two_seed_clusters').all())


if __name__ == '__main__':
    unittest.main()
