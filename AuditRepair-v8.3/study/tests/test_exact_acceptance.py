"""Focused regression checks for the v8.2 strict remainder decision.

Run from the revision folder: python3 -m unittest discover -s tests -p test_exact_acceptance.py -v
The arithmetic fixtures deliberately isolate the final decision from the
statistical question of whether a confidence sequence reaches a given index.
"""
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corrected.audit_repair.core import WorLowerCS


def decision_state(N, j, gain_sum, t=1, grid=401):
    """Install a selected-index decision fixture; not a simulated audit path."""
    test = WorLowerCS(N, grid=grid)
    test.t = t
    test._gain_sum = gain_sum
    test.s = (gain_sum + t) / 2
    test._lower_index = j
    test._lower = float(test.m[j])
    return test


class ExactAcceptanceTests(unittest.TestCase):
    def test_reviewers_zero_margin_regression(self):
        test = decision_state(50, 204, 1)
        # The old floating sign would accept this exact nominal zero.
        self.assertGreater(test.remainder_gain_lb(), 0)
        self.assertEqual(Fraction(50) * (2 * Fraction(204, 400) - 1) - 1, 0)
        self.assertEqual(test.remainder_exact_numerators()[0], 0)
        self.assertFalse(test.remainder_is_positive())

    def test_adjacent_grid_points_keep_strict_boundary(self):
        self.assertFalse(decision_state(50, 203, 1).remainder_is_positive())
        self.assertFalse(decision_state(50, 204, 1).remainder_is_positive())
        self.assertTrue(decision_state(50, 205, 1).remainder_is_positive())

    def test_acceptance_does_not_consult_rounded_diagnostic(self):
        zero = decision_state(50, 204, 1)
        positive = decision_state(50, 205, 1)
        with patch.object(zero, "remainder_gain_lb", return_value=float("inf")):
            self.assertFalse(zero.remainder_is_positive())
        with patch.object(positive, "remainder_gain_lb", return_value=-float("inf")):
            self.assertTrue(positive.remainder_is_positive())

    def test_integer_gain_accumulation_and_selected_index(self):
        test = WorLowerCS(50)
        gains = (1, 0, -1, 1, 1, 0, -1)
        for step, gain in enumerate(gains, 1):
            test.update((gain + 1) / 2)
            self.assertIs(type(test.audited_gain_sum), int)
            self.assertEqual(test.audited_gain_sum, sum(gains[:step]))
            never = [j for j, rejected in enumerate(test._rejected) if not rejected]
            expected_index = max(never[0] - 1, 0) if never else 400
            self.assertEqual(test.lower_grid_index, expected_index)
            self.assertEqual(test.lower, test.m[expected_index])

    def test_no_empty_remainder_or_unaudited_acceptance(self):
        self.assertFalse(decision_state(50, 400, 0, t=0).remainder_is_positive())
        self.assertFalse(decision_state(50, 400, 0, t=50).remainder_is_positive())

    def test_general_bounded_values_are_not_silently_quantized(self):
        test = WorLowerCS(50)
        test.update(0.3)  # Valid input for a generic bounded-mean CS.
        self.assertEqual(test.t, 1)
        with self.assertRaisesRegex(ValueError, "requires x"):
            test.remainder_is_positive()

    def test_fraction_reference_and_unbounded_integer_arithmetic(self):
        # A separate Fraction calculation checks signs, including population
        # sizes beyond int64 multiplication safety and grid floats below j/q.
        represented_guard_changes_a_decision = False
        for N in (30, 50, 101, 5000, 2**60 + 1):
            for j in (1, 7, 57, 199, 200, 201, 204, 205, 233, 301, 399):
                nominal_total = Fraction(N) * (2 * Fraction(j, 400) - 1)
                gain = nominal_total.numerator // nominal_total.denominator
                t = max(1, abs(gain))
                test = decision_state(N, j, gain, t=t)
                nominal = nominal_total - gain
                represented = Fraction(N) * (2 * Fraction.from_float(test.lower) - 1) - gain
                numerator, represented_numerator = test.remainder_exact_numerators()
                self.assertIs(type(numerator), int)
                self.assertIs(type(represented_numerator), int)
                self.assertEqual(Fraction(numerator, 400), nominal)
                self.assertEqual(Fraction(represented_numerator, test.lower.as_integer_ratio()[1]), represented)
                self.assertEqual(test.remainder_is_positive(), 0 < t < N and min(nominal, represented) > 0)
                represented_guard_changes_a_decision |= nominal > 0 and represented <= 0
        self.assertTrue(represented_guard_changes_a_decision)

    def test_record_distinguishes_exact_and_float_quantities(self):
        record = decision_state(50, 204, 1).remainder_certificate_record()
        self.assertNotIn("remainder_lb_exact", record)
        self.assertEqual(record["lower_grid_index"], 204)
        self.assertEqual(record["audited_gain_sum"], 1)
        self.assertEqual(record["remainder_nominal_numerator"], 0)
        self.assertGreater(record["remainder_lb_float"], 0)
        self.assertFalse(record["remainder_is_positive"])


if __name__ == "__main__":
    unittest.main()
