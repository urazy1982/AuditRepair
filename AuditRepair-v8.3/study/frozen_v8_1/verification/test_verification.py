"""Meaningful harness checks, including detection of deliberately injected defects.

All mutations are confined to temporary copies inside this verification directory.
No study input or original source file is modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_verification", HERE / "verify_guarantees.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class IndependentOracleTests(unittest.TestCase):
    def test_zero_calibration_requires_all_unknown_labels(self):
        final, info, audit = v.manual_pac([0, 0, 0], [.1, .2, .3], [1, 0, 1], [], {1: 0})
        self.assertEqual(final.tolist(), [1, 0, 1])
        self.assertEqual(info["expert_new"], 2)
        self.assertEqual(info["calibration_new"], 0)
        self.assertEqual(info["threshold_new"], 2)
        self.assertEqual(audit["new_items"], [0, 2])

    def test_duplicate_draws_and_owned_labels_count_once(self):
        final, info, audit = v.manual_pac([0, 0, 0], [.1, .2, .3], [0, 0, 0], [0]*30, {0: 0})
        self.assertTrue(np.array_equal(final, [0, 0, 0]))
        self.assertEqual(info["u_hat"], float("inf"))
        self.assertEqual(info["expert_new"], 0)
        self.assertEqual(info["calibration_new"], 0)
        self.assertEqual(audit["sampled_items"], [0])

    def test_exact_boundary_impossibility(self):
        F = v.F
        state = v.rational_wor_step(2, [F(0), F(1, 2), F(1)], 0, F(0), F(0), [F(1)]*3, [F(1)]*3, F(1))
        self.assertIsNone(state[3][0])
        self.assertEqual(state[3][1], F(1))  # initial bet is zero at null .5
        self.assertEqual(state[3][2], F(1))

    def test_clopper_pearson_zero_event_upper_is_exact(self):
        interval = v.cp_interval(0, 100)
        self.assertAlmostEqual(interval["upper"], 1 - .05**.01, places=12)
        self.assertEqual(interval["lower"], 0)

    def test_statistical_status_is_three_way(self):
        self.assertEqual(v.empirical_classification({"lower": .12, "upper": .2}, .1), "fail")
        self.assertEqual(v.empirical_classification({"lower": .01, "upper": .08}, .1), "pass_with_margin")
        self.assertEqual(v.empirical_classification({"lower": .03, "upper": .13}, .1), "inconclusive")
        self.assertEqual(v.empirical_classification({"lower": .03, "upper": .1}, .1), "inconclusive")

    def test_json_never_serializes_infinite_json_numbers(self):
        encoded = json.dumps(v.json_safe({"threshold": float("inf"), "ratio": v.F(2, 3)}), allow_nan=False)
        self.assertEqual(json.loads(encoded)["threshold"], "Infinity")


class MutationDetectionTests(unittest.TestCase):
    def test_cost_defect_is_detected_and_cli_exits_nonzero(self):
        source = HERE.parent
        with tempfile.TemporaryDirectory(prefix="harness_mutation_", dir=HERE) as directory:
            root = Path(directory)
            package = root / "audit_repair"
            package.mkdir()
            for name in ("__init__.py", "core.py", "pac.py"):
                shutil.copyfile(source / "audit_repair" / name, package / name)
            target = package / "pac.py"
            old = '"threshold_new": len(new_labels - calibration_new),'
            new = '"threshold_new": len(new_labels - calibration_new) + 1,'
            code = target.read_text()
            self.assertIn(old, code)
            target.write_text(code.replace(old, new))
            report = root / "mutation_report.json"
            proc = subprocess.run([sys.executable, str(HERE / "verify_guarantees.py"),
                "--source-root", str(root), "--output", str(report), "--profile", "quick",
                "--max-n", "1", "--wor-reps", "1", "--search-reps", "1", "--pac-reps", "1"],
                text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, msg=proc.stdout+proc.stderr)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["status"], "failed")
            failure_ids = {failure["check"] for failure in payload["failures"]}
            self.assertIn("pac_manual_threshold_and_cost", failure_ids)
            self.assertIn("pac_cost_partition", failure_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
