"""Synthetic integration checks for the factorial runner; no study pools read.

The preprocessing fixture deliberately uses a small audit set so accounting
cannot pass merely because the entire pool has already been audited. Its
algorithm is covered separately in test_factorial.py. PAC and its accounting
check run unchanged here.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_common as common
import run_factorial as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.out = self.root / "results"
        self.out.mkdir()
        self.n = 48
        self.pool = "synthetic"
        self.p = {"pools": [self.pool], "parameters": {
            "root_seed": 20261030, "pac_reps": 2, "epsilon": 0.3,
            "alpha_final": 0.1, "m": 31, "B": 400,
            "bootstrap_seed": 20261104, "bootstrap_reps": 100,
        }}
        self.sha = "synthetic-frozen-protocol"
        for relative in runner.CODE_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic provenance fixture\n")

        self.gold = np.arange(self.n) % 2
        self.L = np.repeat(self.gold[:, None], 4, axis=1)
        self.X = np.column_stack((np.arange(self.n), np.arange(self.n) / self.n))
        self.neighbors = np.asarray([
            (i + np.arange(1, 16)) % self.n for i in range(self.n)
        ])
        self.cache_path = self.root / "feature_cache" / f"{self.pool}.npz"
        self.cache_path.parent.mkdir()
        np.savez_compressed(self.cache_path, features=self.X, neighbors=self.neighbors)
        common.atomic_json(self.cache_path.with_suffix(".json"), {
            "pool": self.pool, "protocol_sha256": self.sha,
            "code_sha256": common.file_hash(self.root / "factorial.py"),
            "cache_sha256": common.file_hash(self.cache_path),
            "features_sha256": common.array_hash(self.X),
            "neighbors_sha256": common.array_hash(self.neighbors),
            "n": self.n, "dimensions": 2, "neighbor_count": 15,
        })
        frame = pd.DataFrame({"id": [f"id-{i}" for i in range(self.n)]})
        self.pre_calls, self.pac_calls = [], []
        actual_pac = runner.pac_labels

        def preprocess(L, K, gold, X, neighbors, generator, seed):
            self.pre_calls.append({
                "L": L.copy(), "X": X.copy(), "neighbors": neighbors.copy(),
                "seed": seed, "generator": generator,
            })
            audits = {i: int(gold[i]) for i in range(5)}
            base = gold.copy()
            base[8::4] = 1 - base[8::4]
            repair = base.copy()
            repair[8::8] = gold[8::8]
            u0, u1 = np.linspace(0, 1, len(gold)), np.linspace(1, 0, len(gold))
            u0.setflags(write=False)
            u1.setflags(write=False)
            return {
                "base": base, "pred": repair, "oracle_candidate_labels": gold.copy(),
                "audits": audits, "warm_indices": np.arange(3),
                "text_probabilities": np.tile([0.55, 0.45], (len(gold), 1)),
                "scores": {"S0": u0, "S1": u1}, "groups": [], "group_log": [],
                "classifier_info": {"fixture": True},
                "diagnostics": {"certified_groups": 1, "harmful_accepted_vs_ref": 0,
                                "oracle_beneficial_groups_vs_same_audit": 1},
            }

        def trace_pac(pred, uncertainty, gold, **kwargs):
            before = common.array_hash(uncertainty)
            answer = actual_pac(pred, uncertainty, gold, **kwargs)
            self.assertEqual(common.array_hash(uncertainty), before)
            self.pac_calls.append({"score_id": id(uncertainty), "score_hash": before,
                                   "seed": kwargs["seed"]})
            return answer

        self.pre_mock = self.stack.enter_context(patch.object(
            runner, "preprocess_generator", side_effect=preprocess,
        ))
        self.stack.enter_context(patch.object(runner, "ROOT", self.root))
        self.stack.enter_context(patch.object(runner, "protocol", return_value=(self.p, self.sha)))
        self.stack.enter_context(patch.object(runner, "load_pool", return_value=(frame, self.L, self.gold, 2)))
        self.stack.enter_context(patch.object(runner, "dawid_skene", side_effect=lambda L, K: np.eye(K)[L[:, 0]]))
        self.stack.enter_context(patch.object(runner, "pac_labels", side_effect=trace_pac))

    def run_seed(self):
        return runner.one_seed(self.pool, 4, self.out)

    def test_remapping_nineteen_methods_paired_draws_and_audit_cost(self):
        answer, cached = self.run_seed()
        self.assertFalse(cached)
        self.assertEqual(len(self.pre_calls), 3)
        for call in self.pre_calls:
            original_rows = call["X"][:, 0].astype(int)
            np.testing.assert_array_equal(call["L"], self.L[original_rows])
            np.testing.assert_array_equal(
                original_rows[call["neighbors"]], self.neighbors[original_rows],
            )
            self.assertFalse(np.any(call["neighbors"] == np.arange(self.n)[:, None]))
        self.assertTrue(all(call["seed"] is self.pre_calls[0]["seed"] for call in self.pre_calls))
        self.assertEqual(self.pre_calls[0]["seed"].n_children_spawned, 0)

        expected = {"pac_only"} | {
            f"{g}_{s}_{role}" for g in runner.GENERATORS
            for s in runner.SCORES for role in runner.ROLES
        }
        self.assertEqual(len(expected), 19)
        rows = pd.DataFrame(answer["rows"])
        self.assertEqual(len(rows), 38)
        for _, frame in rows.groupby("rep"):
            self.assertEqual(set(frame.method), expected)
            self.assertEqual(frame.draws_sha256.nunique(), 1)
        self.assertEqual(rows.draws_sha256.nunique(), 2)
        self.assertTrue((rows.loc[rows.role != "pac_only", "pre_audits"] == 5).all())
        self.assertTrue((rows.expert_total == rows.pre_audits + rows.calibration_new + rows.threshold_new).all())
        for (rep, generator, score), frame in rows[rows.role != "pac_only"].groupby(["rep", "generator", "score"]):
            base_cost = frame.loc[frame.role == "same_audit", "expert_total"].iloc[0]
            np.testing.assert_array_equal(frame.paired_cost_delta, frame.expert_total - base_cost)

        # Each cell's score vector is reused for all three roles in both reps.
        counts = {}
        for call in self.pac_calls:
            counts.setdefault(call["score_id"], []).append(call["score_hash"])
            self.assertEqual(call["seed"].n_children_spawned, 0)
        self.assertEqual(sorted(map(len, counts.values())), [2] + [6] * 6)
        self.assertTrue(all(len(set(hashes)) == 1 for hashes in counts.values()))

    def test_complete_checkpoint_resume_is_idempotent(self):
        first, cached = self.run_seed()
        self.assertFalse(cached)
        target = self.out / "checkpoints" / self.pool / "seed_04.json"
        before = target.read_bytes()
        self.pre_mock.reset_mock()
        second, cached = self.run_seed()
        self.assertTrue(cached)
        self.pre_mock.assert_not_called()
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(second, common.clean_json(first))

    def test_wrong_seed_or_duplicate_method_checkpoint_is_rejected(self):
        self.run_seed()
        target = self.out / "checkpoints" / self.pool / "seed_04.json"
        original = target.read_text()
        for corruption in ("seed", "method"):
            with self.subTest(corruption=corruption):
                altered = json.loads(original)
                if corruption == "seed":
                    altered["seed"] = 5
                else:
                    altered["rows"][1]["method"] = altered["rows"][0]["method"]
                common.atomic_json(target, altered)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    self.run_seed()
                target.write_text(original)

    def test_modified_feature_cache_is_rejected_even_on_resume(self):
        self.run_seed()
        with self.cache_path.open("ab") as file:
            file.write(b"modified after checkpoint")
        self.pre_mock.reset_mock()
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_seed()
        self.pre_mock.assert_not_called()

    def test_primary_contrasts_use_incremental_cost_not_absolute_cost_or_best_cell(self):
        rows = []
        seed_deltas = [
            {"G0_uniform_S0": 10, "G1_S0": 4, "G0_uniform_S1": 5, "G1_S1": -3},
            {"G0_uniform_S0": 20, "G1_S0": 12, "G0_uniform_S1": 13, "G1_S1": 1},
        ]
        for seed, values in enumerate(seed_deltas):
            values.update({"G2_S0": -9999, "G2_S1": -99999})
            for rep in range(2):
                for number, (cell, delta) in enumerate(values.items()):
                    generator, score = cell.rsplit("_", 1)
                    rows.append({
                        "pool": self.pool, "seed": seed, "rep": rep,
                        "generator": generator, "score": score, "role": "repair",
                        "paired_cost_delta": delta + (2 * rep - 1),
                        "expert_total": 1000 * number + 100 * seed + rep,
                    })
        runner.primary_contrasts(rows, self.out)
        frame = pd.read_csv(self.out / "prespecified_contrasts.csv").set_index("contrast")
        expected = {
            "primary_generator_G1_S0_vs_G0_uniform_S0": -7,
            "primary_score_G0_uniform_S1_vs_S0": -6,
            "secondary_G1_by_S1_interaction": -3,
        }
        self.assertEqual(set(frame.index), set(expected))
        for name, value in expected.items():
            self.assertEqual(frame.loc[name, "mean_difference_in_incremental_repair_cost"], value)
            self.assertEqual(frame.loc[name, "seeds"], 2)


if __name__ == "__main__":
    unittest.main()
