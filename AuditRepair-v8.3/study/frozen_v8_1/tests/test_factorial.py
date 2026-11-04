"""Synthetic tests for the E2/E3 information boundary and paired comparisons.

Run from v8_experiments with:
    python -m unittest discover -s tests -p test_factorial.py -v

No cached study pools are read. The pipeline tests use the real Oracle,
WorLowerCS, and uncertainty_fill, with a controlled DS posterior to isolate
certificate targets and final-baseline changes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import factorial as f

core = importlib.import_module(f.Oracle.__module__)


def neighbor_rows(n: int, anchor_start: int) -> np.ndarray:
    """All rows consult 15 distinct anchor rows, excluding themselves."""
    return np.asarray([
        [j for j in range(anchor_start, n) if j != i][:15]
        for i in range(n)
    ], dtype=int)


def fixed_ds(L, K, clamp=None, **kwargs):
    """Controlled predictions with genuine clamping, independent of hidden gold."""
    labels = L[:, 0].copy()
    for i, y in (clamp or {}).items():
        labels[i] = y
    return np.eye(K)[labels]


def improving_ds(L, K, clamp=None, **kwargs):
    """Model whose final refit resolves label 0 after the group stage."""
    labels = (np.ones(len(L), dtype=int) if len(clamp or {}) > 120
              else L[:, 0].copy())
    for i, y in (clamp or {}).items():
        labels[i] = y
    return np.eye(K)[labels]


def positive_pool():
    n = 800
    L = np.zeros((n, 4), dtype=int)
    L[n // 2:] = 1
    gold = np.ones(n, dtype=int)
    X = np.column_stack((np.linspace(-1, 1, n), np.ones(n)))
    return L, gold, X, neighbor_rows(n, n // 2)


class FeatureTests(unittest.TestCase):
    def test_neighbors_exclude_self_and_resolve_distance_ties_by_id(self):
        texts = (["alpha repeated shared words"] * 10 +
                 ["beta repeated shared words"] * 10 + ["uniquetoken"])
        ids = [f"item-{20-i:02d}" for i in range(len(texts))]
        X, neighbors = f.build_features(texts, ids)
        self.assertEqual(neighbors.shape, (21, 15))
        self.assertTrue(np.isfinite(X).all())
        np.testing.assert_array_equal(X[-1], np.zeros(X.shape[1]))
        for i in range(len(texts)):
            self.assertNotIn(i, neighbors[i])
            self.assertEqual(len(set(neighbors[i])), 15)
        for i in range(20):
            cluster = range(10) if i < 10 else range(10, 20)
            expected = sorted((j for j in cluster if j != i), key=lambda j: ids[j])
            np.testing.assert_array_equal(neighbors[i, :9], expected)
        # The zero row has equally distant nonself neighbours.
        expected_zero = sorted(range(20), key=lambda j: ids[j])[:15]
        np.testing.assert_array_equal(neighbors[-1], expected_zero)

    def test_duplicate_ids_fail_instead_of_ambiguous_ties(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            f.build_features(["shared terms", "shared terms"], [1, "1"])


class ClassifierTests(unittest.TestCase):
    def test_missing_classes_and_one_class_fallback_have_fixed_smoothing(self):
        X = np.arange(20, dtype=float).reshape(10, 2) / 10
        warm = np.arange(6)
        q, info = f._text_probabilities(X, warm, np.array([0, 0, 1, 1, 0, 1]), 3)
        np.testing.assert_allclose(q.sum(axis=1), 1)
        np.testing.assert_allclose(q[:, 2], 1e-6 / (1 + 3e-6))
        self.assertEqual(info["observed_classes"], [0, 1])
        self.assertFalse(info["one_class_fallback"])
        q, info = f._text_probabilities(X, warm, np.full(6, 2), 4)
        expected = np.full(4, 1e-6)
        expected[2] += 1
        expected /= 1 + 4e-6
        np.testing.assert_allclose(q, np.tile(expected, (10, 1)))
        self.assertTrue(info["one_class_fallback"])

    def test_classifier_convergence_warning_is_a_failure(self):
        class NonConverging:
            def fit(self, *args):
                warnings.warn("iteration limit", ConvergenceWarning)

        with patch.object(f, "LogisticRegression", return_value=NonConverging()):
            with self.assertRaisesRegex(RuntimeError, "failed to converge"):
                f._text_probabilities(np.ones((6, 2)), np.arange(6),
                                      np.array([0, 1, 0, 1, 0, 1]), 2)


class GroupDefinitionTests(unittest.TestCase):
    def test_g2_ranks_by_prewarm_confusions_before_group_size(self):
        n, K = 300, 3
        warm = np.arange(120)
        ref = np.full(n, 2, dtype=int)
        ref[120:220], ref[220:] = 0, 1
        initial = ref.copy()
        initial[:10], initial[10:110], initial[110:120] = 0, 1, 2
        L = np.repeat(ref[:, None], 4, axis=1)
        q = np.tile([0.0, 0.0, 1.0], (n, 1))
        groups = f._generate_groups(
            L, ref, np.eye(K)[ref], neighbor_rows(n, 120), q,
            warm, np.full(120, 2), initial, "G2", K,
        )
        self.assertEqual([(g.cur, g.alt, len(g.items)) for g in groups],
                         [(1, 2, 80), (0, 2, 100)])
        self.assertEqual([g.confusion_count for g in groups], [100, 10])

    def test_warmup_exclusion_precedes_minimum_group_size(self):
        n, K = 200, 2
        ref = np.ones(n, dtype=int)
        ref[:40] = 0
        L = np.repeat(ref[:, None], 4, axis=1)
        L[:40, 0] = 1
        warm = np.r_[np.arange(20), np.arange(100, 200)]
        q = np.tile([0.0, 1.0], (n, 1))
        for generator in ("G0_uniform", "G1", "G2"):
            with self.subTest(generator=generator):
                groups = f._generate_groups(
                    L, ref, np.eye(K)[ref], neighbor_rows(n, 40), q,
                    warm, ref[warm], ref, generator, K,
                )
                self.assertEqual(groups, [])  # Only 20 of the 40 remain eligible.


class PipelineTests(unittest.TestCase):
    def run_state(self, L, gold, X, neighbors, generator, *, warm=None,
                  K=2, ds=improving_ds, seed=20261030):
        with patch.object(f, "dawid_skene", side_effect=ds), \
             patch.object(core, "dawid_skene", side_effect=ds):
            return f.preprocess_generator(L, K, gold, X, neighbors, generator,
                                          seed, warm_indices=warm)

    def test_common_warmup_paired_scores_and_certificate_target(self):
        L, gold, X, neighbors = positive_pool()
        original = [a.copy() for a in (L, gold, X, neighbors)]
        seed = np.random.SeedSequence(90210)
        states = [self.run_state(L, gold, X, neighbors, g, seed=seed)
                  for g in ("G0_uniform", "G1", "G2")]
        self.assertEqual(seed.n_children_spawned, 0)
        for state in states:
            np.testing.assert_array_equal(state["warm_indices"], states[0]["warm_indices"])
            np.testing.assert_allclose(state["text_probabilities"], states[0]["text_probabilities"])
            self.assertEqual(len(state["audits"]), 400)
            self.assertLessEqual(state["diagnostics"]["group_phase_audits"], 196)
            members = [i for g in state["groups"] for i in g["members"]]
            self.assertEqual(len(set(members)), len(members))
            self.assertFalse(set(members) & set(state["warm_audits"]))
            for i, y in state["audits"].items():
                self.assertEqual(state["base"][i], y)
                self.assertEqual(state["pred"][i], y)
            np.testing.assert_allclose(state["scores"]["S0"], 1 - state["post"].max(axis=1))
            np.testing.assert_allclose(
                state["scores"]["S1"],
                1 - state["text_probabilities"][np.arange(len(L)), state["base"]],
            )
            self.assertFalse(state["scores"]["S0"].flags.writeable)
            self.assertFalse(state["scores"]["S1"].flags.writeable)
        for array, previous in zip((L, gold, X, neighbors), original):
            np.testing.assert_array_equal(array, previous)

        for state in states[1:]:
            self.assertGreater(state["diagnostics"]["certified_groups"], 0)
            for log in state["group_log"]:
                self.assertTrue(log["guarded"])
                self.assertGreater(log["true_remainder_gain_vs_frozen_ref"], 0)
                self.assertEqual(log["effect_vs_same_audit"], 0)
                self.assertFalse(set(log["remainder_members_at_stop"]) & set(state["audits"]))
                # G1/G2 can propose a repair even though every source is unanimous.
                self.assertTrue(np.all(L[log["members"]] == L[log["members"], :1]))
            self.assertGreater(state["diagnostics"]["net_gain_vs_ref"], 0)
            self.assertEqual(state["diagnostics"]["net_gain_vs_same_audit"], 0)
            np.testing.assert_array_equal(state["pred"], state["base"])

    def test_unrevealed_truth_cannot_change_selection_training_or_stopping(self):
        L, gold, X, neighbors = positive_pool()
        first = self.run_state(L, gold, X, neighbors, "G2")
        changed = gold.copy()
        untouched = np.ones(len(L), dtype=bool)
        untouched[list(first["audits"])] = False
        changed[untouched] = 0
        second = self.run_state(L, changed, X, neighbors, "G2")
        self.assertEqual(first["audits"], second["audits"])
        for key in ("ref", "base", "pred", "text_probabilities"):
            np.testing.assert_array_equal(first[key], second[key])
        for score in ("S0", "S1"):
            np.testing.assert_array_equal(first["scores"][score], second["scores"][score])
        for before, after in zip(first["group_log"], second["group_log"]):
            for key in ("members", "audit_history", "decision", "stop_reason", "remainder_lb"):
                self.assertEqual(before[key], after[key])
        self.assertEqual(first["diagnostics"]["harmful_accepted_vs_ref"], 0)
        self.assertGreater(second["diagnostics"]["harmful_accepted_vs_ref"], 0)

    def test_futility_and_odd_group_fraction_cap(self):
        n = 200
        L = np.ones((n, 4), dtype=int)
        L[:31] = 0
        X = np.ones((n, 2))
        neighbors = neighbor_rows(n, 31)
        warm = np.arange(80, 200)
        for target_gold, stop, count in ((0, "futility", 10), (2, "fraction_cap", 16)):
            with self.subTest(stop=stop):
                gold = np.ones(n, dtype=int)
                gold[:31] = target_gold
                state = self.run_state(L, gold, X, neighbors, "G1", K=3,
                                       warm=warm, ds=fixed_ds)
                self.assertEqual(len(state["group_log"]), 1)
                log = state["group_log"][0]
                self.assertEqual(log["stop_reason"], stop)
                self.assertEqual(log["audited"], count)
                self.assertFalse(log["guarded"])
                self.assertEqual(state["diagnostics"]["certified_groups"], 0)

    def test_group_budget_and_bonferroni_use_the_frozen_shortlist(self):
        n, K = 1600, 6
        labels = np.full(n, 4, dtype=int)
        for c in range(4):
            labels[c * 300:(c + 1) * 300] = c
        L = np.repeat(labels[:, None], 4, axis=1)
        gold = np.full(n, 4, dtype=int)
        gold[:1200] = 5  # Neither current nor proposed class: all gains are zero.
        state = self.run_state(
            L, gold, np.ones((n, 2)), neighbor_rows(n, 1200), "G1", K=K,
            warm=np.arange(1480, 1600), ds=fixed_ds,
        )
        self.assertEqual(len(state["groups"]), 4)
        self.assertEqual([log["audited"] for log in state["group_log"]], [80, 80, 36])
        self.assertEqual(state["group_log"][-1]["stop_reason"], "budget")
        self.assertEqual(state["diagnostics"]["group_phase_audits"], 196)
        self.assertEqual(len(state["audits"]), 400)
        self.assertTrue(all(log["alpha"] == 0.05 / 4 for log in state["group_log"]))


if __name__ == "__main__":
    unittest.main()
