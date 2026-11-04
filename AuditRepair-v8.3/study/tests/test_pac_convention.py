"""Regression for the terminal-capital PAC convention used in v8.1 and v8.2.

The forced draw sequence is an implementation fixture, not a probability sample.
It deliberately crosses 1/alpha and later falls below it. A sticky replacement
would produce a different threshold and cost, so this test guards the convention.
"""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected"))
from audit_repair.pac import pac_labels


class FixedDraws:
    def __init__(self, draws):
        self.draws = draws

    def integers(self, low, high, size):
        assert (low, high, size) == (0, 60, 25)
        return self.draws.copy()


class PacConventionTest(unittest.TestCase):
    def test_first_crossing_then_fall_is_not_rejected_at_terminal_time(self):
        draws = np.asarray([0] * 20 + [1] * 5)
        epsilon, alpha = 0.2, 0.2
        total = squared = log_capital = 0.0
        capitals = []
        for t, z in enumerate([0] * 20 + [1] * 5):
            mean = (epsilon / 2 + total) / (t + 1)
            variance = (0.25 + squared) / (t + 1)
            gap = epsilon - mean
            stake = max(gap / (variance + gap * gap), 0.0) if gap > 0 else 0.0
            stake = min(stake, 0.5 / (1 - epsilon))
            log_capital += math.log1p(stake * (epsilon - z))
            capitals.append(math.exp(log_capital))
            total += z
            squared += (z - mean) ** 2
        self.assertGreater(max(capitals), 1 / alpha)
        self.assertLess(capitals[-1], 1 / alpha)
        self.assertEqual(next(i + 1 for i, x in enumerate(capitals) if x >= 1 / alpha), 15)

        prediction = np.zeros(60, dtype=int)
        gold = prediction.copy()
        gold[1] = 1
        scores = np.full(60, 0.5)
        with patch("audit_repair.pac.np.random.default_rng", return_value=FixedDraws(draws)):
            labels, info = pac_labels(prediction, scores, gold, eps=epsilon, alpha=alpha, m=25)
        self.assertEqual(info["u_hat"], 0.5)
        self.assertEqual(info["expert_new"], 60)
        np.testing.assert_array_equal(labels, gold)
        # Under a sticky counterfactual the sole threshold would certify and
        # only the two distinct calibration items would be purchased.
        self.assertEqual(len(set(draws)), 2)


if __name__ == "__main__":
    unittest.main()
