#!/usr/bin/env python3
"""Independent CPU implementation checks; NEVER a study E1--E4 runner.

Only synthetic, fixed verification fixtures are used. Exhaustive probabilities are
exact counts for ternary finite populations, not claims for every real population.
The implementation under test is imported read-only from --source-root.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import defaultdict
from fractions import Fraction as F
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Checks:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.assertion_count = 0

    def check(self, condition, check_id, details=None):
        self.assertion_count += 1
        if not condition:
            if len(self.failures) < 40:
                self.failures.append({"check": check_id, "details": details})
            return False
        return True

    def warn(self, check_id, message):
        self.warnings.append({"check": check_id, "message": message})


def load_source(source_root):
    root = Path(source_root).resolve()
    if not (root / "audit_repair" / "core.py").is_file():
        raise FileNotFoundError(f"Not an Audit-Repair source root: {root}")
    sys.path.insert(0, str(root))
    core = importlib.import_module("audit_repair.core")
    pac = importlib.import_module("audit_repair.pac")
    for module in (core, pac):
        if root not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f"Imported the wrong source module: {module.__file__}")
    return core, pac


def digest_array(a):
    a = np.asarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return json_safe(float(x))
    if isinstance(x, float) and not math.isfinite(x):
        return "Infinity" if x > 0 else "-Infinity" if x < 0 else "NaN"
    if isinstance(x, F):
        return {"numerator": x.numerator, "denominator": x.denominator, "decimal": float(x)}
    return x


def cp_interval(k, n, confidence=.95):
    """One-sided CP lower and upper bounds, each with stated confidence."""
    tail = 1 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n-k+1))
    upper = 1.0 if k == n else float(beta.ppf(confidence, k+1, n-k))
    return {"events": int(k), "trials": int(n), "estimate": k/n,
            "one_sided_confidence": confidence, "lower": lower, "upper": upper}


def empirical_classification(interval, nominal):
    if interval["lower"] > nominal:
        return "fail"
    if interval["upper"] < nominal:
        return "pass_with_margin"
    return "inconclusive"


def rational_wor_step(N, grid, t, s, ss, capitals, maxima, x):
    """Rational-arithmetic reference from the conditional mean and product formula.

    None represents +infinity after a logically impossible null. No logarithms,
    NumPy betting code, or production classes participate in this calculation.
    """
    remaining = N-t
    regularized_mean = (F(1, 2) + s)/(t+1)
    regularized_variance = (F(1, 4) + ss)/(t+1)
    next_capitals, next_maxima = [], []
    for null, capital, maximum in zip(grid, capitals, maxima):
        conditional_null_mean = (N*null-s)/remaining
        if capital is None or conditional_null_mean < 0 or (conditional_null_mean == 0 and x > 0):
            new_capital = None
        elif conditional_null_mean == 0 or conditional_null_mean >= 1:
            new_capital = capital
        else:
            gap = regularized_mean - conditional_null_mean
            stake = max(F(0), gap/(regularized_variance + gap*gap)) if gap > 0 else F(0)
            stake = min(stake, F(1, 2)/conditional_null_mean)
            factor = 1 + stake*(x-conditional_null_mean)
            if factor < 0:
                raise AssertionError("Negative reference betting factor")
            new_capital = capital*factor
        next_capitals.append(new_capital)
        next_maxima.append(None if maximum is None or new_capital is None else max(maximum, new_capital))
    return t+1, s+x, ss+(x-regularized_mean)**2, next_capitals, next_maxima


def clone_cs(cs):
    result = copy.copy(cs)
    result.log_cap = cs.log_cap.copy()
    result._rejected = cs._rejected.copy()
    return result


def exact_wor_check(core, checks, max_n=8, grid_size=401, reference_grid_size=41):
    if (grid_size-1) % (reference_grid_size-1):
        raise ValueError("Reference grid must be a subset of production grid")
    alphas = [F(1, 20), F(1, 5), F(1, 2)]
    selected = np.linspace(0, grid_size-1, reference_grid_size, dtype=int)
    grid = [F(j, reference_grid_size-1) for j in range(reference_grid_size)]
    half_idx = grid.index(F(1, 2))
    totals = defaultdict(lambda: {"orders": 0, "cs_miss": 0, "harmful_first_acceptance": 0,
                                   "accepted": 0, "scalar_ever_reject": 0})
    max_log_error = 0.0
    threshold_roundoff_cases = 0
    prefix_count = 0
    source_comparisons = 0
    total_leaves = 0

    for N in range(1, max_n+1):
        initial_cs = [core.WorLowerCS(N, alpha=float(a), grid=grid_size) for a in alphas]
        initial_scalar = [core.WorBettingTest(N, alpha=float(a)) for a in alphas]
        initial_ref = (0, F(0), F(0), [F(1)]*len(grid), [F(1)]*len(grid))

        def visit(ref, cs_objects, scalar_objects, counts, path, first_accept, scalar_ever, maximum_lower):
            nonlocal prefix_count, max_log_error, threshold_roundoff_cases, source_comparisons, total_leaves
            t, s, ss, capital, maximum = ref
            if t == N:
                total_leaves += 1
                mu = s/N
                for ai, alpha in enumerate(alphas):
                    row = totals[(N, counts, alpha)]
                    row["orders"] += 1
                    lower_index = int(round(cs_objects[ai].lower*(grid_size-1)))
                    row["cs_miss"] += maximum_lower[ai] > mu
                    row["scalar_ever_reject"] += scalar_ever[ai]
                    acceptance = first_accept[ai]
                    if acceptance is not None:
                        row["accepted"] += 1
                        stop_t, observed_sum = acceptance
                        true_remaining_gain = (2*s-N)-(2*observed_sum-stop_t)
                        row["harmful_first_acceptance"] += true_remaining_gain <= 0
                return
            for symbol, x in enumerate((F(0), F(1, 2), F(1))):
                child_ref = rational_wor_step(N, grid, t, s, ss, capital, maximum, x)
                nt, ns, nss, ncaps, nmax = child_ref
                child_cs = [clone_cs(c) for c in cs_objects]
                child_scalar = [copy.copy(c) for c in scalar_objects]
                next_accept = list(first_accept)
                next_ever = list(scalar_ever)
                next_maximum_lower = list(maximum_lower)
                child_path = path+(symbol,)
                prefix_count += 1
                ref_logs = np.array([math.inf if v is None else math.log(float(v)) for v in ncaps])
                for ai, (alpha, cs, scalar) in enumerate(zip(alphas, child_cs, child_scalar)):
                    cs.update(float(x))
                    scalar.xs = scalar.xs.copy()
                    scalar.update(float(x))
                    next_ever[ai] |= scalar.rejected
                    next_maximum_lower[ai] = max(next_maximum_lower[ai],F.from_float(cs.lower))
                    finite = np.isfinite(ref_logs)
                    actual = cs.log_cap[selected]
                    equal_infinity = np.array_equal(np.isposinf(actual), np.isposinf(ref_logs))
                    error = float(np.max(np.abs(actual[finite]-ref_logs[finite]))) if finite.any() else 0.0
                    max_log_error = max(max_log_error, error)
                    source_comparisons += len(grid)
                    checks.check(equal_infinity and error < 2e-10, "wor_rational_capital",
                                 {"N": N, "path": child_path, "alpha": float(alpha), "max_log_error": error})
                    expected_rejected = np.array([v is None or v >= 1/alpha for v in nmax])
                    differences = np.flatnonzero(cs._rejected[selected] != expected_rejected)
                    for j in differences:
                        gap = math.inf if nmax[j] is None else abs(math.log(float(nmax[j]))-math.log(float(1/alpha)))
                        if gap < 2e-10:
                            threshold_roundoff_cases += 1
                        else:
                            checks.check(False, "wor_sticky_rejection_rational", {"N": N, "path": child_path,
                                         "null": str(grid[j]), "alpha": float(alpha), "log_threshold_gap": gap})
                    checks.check(abs(float(ns)-cs.s) < 1e-10 and abs(float(nss)-cs.ss) < 1e-10,
                                 "wor_predictable_state", {"N": N, "path": child_path})
                    lower_index = int(round(cs.lower*(grid_size-1)))
                    exact_remainder = N*(2*F(lower_index, grid_size-1)-1)-(2*ns-nt)
                    checks.check(abs(cs.remainder_gain_lb()-float(exact_remainder)) < 2e-10,
                                 "remainder_algebra", {"N": N, "path": child_path})
                    scalar_log = ref_logs[half_idx]
                    checks.check((math.isinf(scalar_log) and scalar.log_cap == scalar_log) or
                                 (math.isfinite(scalar_log) and abs(scalar.log_cap-scalar_log) < 2e-10),
                                 "scalar_wor_rational", {"N": N, "path": child_path})
                    # Optional stopping is first strict positive remainder bound; never at exhaustion.
                    if next_accept[ai] is None and nt < N and cs.remainder_gain_lb() > 0:
                        next_accept[ai] = (nt, ns)
                next_counts = list(counts)
                next_counts[symbol] += 1
                visit(child_ref, child_cs, child_scalar, tuple(next_counts), child_path,
                      next_accept, next_ever, next_maximum_lower)
        visit(initial_ref, initial_cs, initial_scalar, (0, 0, 0), (), [None]*len(alphas), [False]*len(alphas), [F(0)]*len(alphas))

    rows = []
    for (N, counts, alpha), row in sorted(totals.items(), key=lambda z: (z[0][0], z[0][1], z[0][2])):
        expected_orders = math.factorial(N)
        for count in counts:
            expected_orders //= math.factorial(count)
        checks.check(row["orders"] == expected_orders, "exact_order_count", {"N": N, "counts": counts})
        miss = F(row["cs_miss"], row["orders"])
        bad = F(row["harmful_first_acceptance"], row["orders"])
        checks.check(miss <= alpha, "exact_cs_coverage", {"N": N, "counts": counts, "probability": miss, "alpha": alpha})
        checks.check(bad <= alpha, "exact_remainder_stopping", {"N": N, "counts": counts, "probability": bad, "alpha": alpha})
        mu = F(counts[1]+2*counts[2], 2*N)
        scalar_probability = F(row["scalar_ever_reject"], row["orders"]) if mu <= F(1, 2) else None
        if scalar_probability is not None:
            checks.check(scalar_probability <= alpha, "exact_scalar_null_coverage", {"N": N, "counts": counts, "probability": scalar_probability, "alpha": alpha})
        rows.append({"N": N, "counts_0_half_1": counts, "mean": mu, "alpha": alpha,
                     **row, "miscoverage_probability": miss, "harmful_probability": bad,
                     "scalar_null_reject_probability": scalar_probability})
    worst = max(rows, key=lambda r: r["miscoverage_probability"]/r["alpha"])
    worst_bad = max(rows, key=lambda r: r["harmful_probability"]/r["alpha"])
    return {"status": "completed", "max_N": max_n, "values": [0, .5, 1],
            "production_grid": grid_size, "rational_reference_grid": reference_grid_size,
            "reference_scope": "Exact rational capitals on a nested null grid; production-grid coverage exhaustively counted over every ternary order.",
            "distinct_population_count": len(rows)//len(alphas), "population_alpha_combinations": len(rows),
            "distinct_full_orders": total_leaves, "distinct_prefixes": prefix_count,
            "capital_point_comparisons_across_alpha": source_comparisons,
            "max_absolute_log_capital_error": max_log_error,
            "threshold_roundoff_cases_within_tolerance": threshold_roundoff_cases,
            "worst_miscoverage_over_alpha": float(worst["miscoverage_probability"]/worst["alpha"]),
            "worst_harmful_probability_over_alpha": float(worst_bad["harmful_probability"]/worst_bad["alpha"]),
            "populations": rows}


def grid_inversion_check(core, checks):
    cases = 0
    for bits in itertools.product((False, True), repeat=8):
        mask = np.array((*bits, False))  # the endpoint 1 is never rejected by a valid path
        cs = core.WorLowerCS(100, alpha=.05, grid=9)
        cs.log_cap[:] = np.where(mask, 1000., -1000.)
        cs.update(0.)
        first_unrejected = next(i for i, rejected in enumerate(mask) if not rejected)
        expected = F(max(first_unrejected-1, 0), 8)
        checks.check(cs.lower == float(expected), "nonmonotone_discrete_prefix", {"mask": mask.tolist(), "expected": expected})
        cs.log_cap[:] = -1000.
        cs.update(0.)
        checks.check(cs.lower == float(expected), "sticky_after_capital_drop", {"mask": mask.tolist()})
        # Actual conservative statement: if a grid ceiling null is not rejected,
        # lower <= its predecessor. No monotonicity between nulls is assumed.
        for j in np.flatnonzero(~mask):
            bound = F(max(int(j)-1, 0), 8)
            checks.check(cs.lower <= float(bound), "unrejected_grid_null_dominates", {"mask": mask.tolist(), "null_index": int(j)})
        cases += 1
    # Arbitrary continuous rejection sets need not have a monotone boundary.
    # Let all points below 1 be rejected except the hole [.2,.3]. On grid
    # {0,.5,1}, the implemented lower is .5, while inf(unrejected set) is .2.
    coarse = F(1, 2)
    continuous_first_unrejected = F(1, 5)
    checks.check(coarse > continuous_first_unrejected, "counterexample_to_unjustified_continuous_comparison")
    checks.warn("plan_grid_oracle", "The draft's bisection oracle is invalid for arbitrary nonmonotone rejection sets. This package checks the discrete prefix and one-unrejected-grid-null implication; it does not claim domination of a continuous first-unrejected bound.")
    return {"status": "completed", "exhaustive_nonmonotone_masks": cases,
            "sticky_drop_cases": cases, "bisection_used": False,
            "statement_checked": "lower is the predecessor of the first never-rejected grid null; any never-rejected true grid ceiling controls the reported lower bound.",
            "counterexample_to_draft_continuous_comparison": {"grid": [0, .5, 1], "unrejected_continuous_set": "[0.2,0.3] union {1}", "grid_lower": .5, "continuous_first_unrejected": .2},
            "limitation": "The constructed continuous pattern is a logical counterexample for arbitrary rejection sets, not a claimed path generated by this betting implementation."}


def deterministic_base_model(L, K, clamp=None, **kwargs):
    """Fixed observable reference for structural and full-search verification only."""
    post = np.full((len(L), K), .1/(K-1))
    post[:, 0] = .9
    for i, y in (clamp or {}).items():
        post[int(i)] = 0.
        post[int(i), int(y)] = 1.
    return post


def group_structure_check(core, checks):
    L = np.array([[0,1,1,2], [0,1,-1,2], [0,1,1,2], [0,2,2,1],
                  [0,2,2,-1], [0,-1,-1,0], [1,0,0,2], [1,2,0,-1]])
    cur = np.array([0,0,0,0,0,0,1,1])
    post = np.full((len(L), 3), 1/3)
    groups = core.candidate_groups_disjoint(L, cur, post, min_size=1)
    actual = {(g.cur, g.alt, tuple(g.source)): g.items.tolist() for g in groups}
    expected = {(0,1,(1,2)): [0,2], (0,1,(1,)): [1], (0,2,(1,2)): [3,4],
                (1,0,(1,2)): [6], (1,0,(2,)): [7]}
    checks.check(actual == expected, "candidate_ties_missing_and_disjoint", {"actual": str(actual), "expected": str(expected)})
    all_members = [int(i) for g in groups for i in g.items]
    checks.check(len(set(all_members)) == len(all_members), "candidate_disjoint_membership")

    snapshots, queries = [], []
    original_oracle = core.Oracle
    original_make = core.candidate_groups_disjoint

    class RecordingOracle(original_oracle):
        def query(self, i):
            value = super().query(i)
            queries.append(int(i))
            return value

    def recording_make(labels, reference, posterior, min_size=30):
        frozen = original_make(labels, reference, posterior, min_size=min_size)
        snapshots.append({"queries_at_freeze": tuple(queries), "reference": reference.copy(),
                          "groups": [(g.cur, g.alt, g.items.copy()) for g in frozen]})
        return frozen

    fixture_L = np.column_stack([np.zeros(80, int), np.ones(80, int)])
    gold = np.ones(80, int); gold[:4] = 0
    with patch.object(core, "dawid_skene", deterministic_base_model), patch.object(core, "Oracle", RecordingOracle), patch.object(core, "candidate_groups_disjoint", recording_make):
        result = core.run_group_repair(fixture_L, 2, gold, budget=20, ref_share=.2,
                                      group_share=1., min_size=2, max_groups=4, k_max=3, seed=71)
    checks.check(len(snapshots) == 1, "single_candidate_freeze", {"calls": len(snapshots)})
    snapshot = snapshots[0]
    checks.check(len(set(snapshot["queries_at_freeze"])) == 4, "freeze_after_warmup", {"queries": snapshot["queries_at_freeze"]})
    ref = snapshot["reference"]
    for current, alternative, members in snapshot["groups"]:
        for i in set(snapshot["queries_at_freeze"]).intersection(members.tolist()):
            known_gain = int(gold[i] == alternative)-int(gold[i] == ref[i])
            checks.check(ref[i] == gold[i] and known_gain <= 0, "warmup_known_nonpositive_gain", {"item": i, "gain": known_gain})
    checked_remainders = 0
    for record in result["guarded"][1]["log"]:
        history = record["audit_history"]
        independently_computed_s = sum((int(gold[h["item"]] == record["alt"])-int(gold[h["item"]] == ref[h["item"]])+1)/2 for h in history)
        bound = record["size"]*(2*record["lower_mean"]-1)-(2*independently_computed_s-len(history))
        checks.check(abs(bound-record["remainder_lb_exact"]) < 1e-10, "group_log_remainder_reconstruction")
        ignored_known = set(record["known_at_start"])-{h["item"] for h in history}
        checks.check(all(int(gold[i] == record["alt"])-int(gold[i] == ref[i]) <= 0 for i in ignored_known), "known_unsampled_remainder_conservatism")
        checked_remainders += 1

    # Gold changes that cannot be observed must not change allocation or decisions.
    owned = result["guarded"][1]["audited_labels"]
    gold_alt = gold.copy()
    hidden = sorted(set(range(len(gold)))-set(owned))
    gold_alt[hidden] = 1-gold_alt[hidden]
    with patch.object(core, "dawid_skene", deterministic_base_model):
        result_alt = core.run_group_repair(fixture_L, 2, gold_alt, budget=20, ref_share=.2,
                                          group_share=1., min_size=2, max_groups=4, k_max=3, seed=71)
    fields = ("members", "audit_history", "alpha", "guarded", "unguarded", "stop_reason", "remainder_lb_exact")
    signatures = lambda r: [{f: g[f] for f in fields} for g in r["guarded"][1]["log"]]
    checks.check(signatures(result) == signatures(result_alt) and owned == result_alt["guarded"][1]["audited_labels"], "hidden_gold_no_decision_leakage", {"perturbed_hidden_items": len(hidden)})

    # Deliberately overlap equal candidates. Previously revealed positive gains
    # must be absent from the next group's population before its random order.
    overlap_groups = [core.Group(s, 0, 1, np.arange(40), 0.) for s in range(3)]
    with patch.object(core, "dawid_skene", deterministic_base_model), patch.object(core, "candidate_groups", return_value=overlap_groups):
        overlap = core.run_group_repair(np.zeros((40, 3), int), 2, np.ones(40, int), budget=8,
                                        group_share=1., min_size=2, k_max=1, disjoint=False, max_groups=3, seed=17)
    earlier = set()
    overlap_count = 0
    for record in overlap["guarded"][1]["log"]:
        checks.check(not (earlier & set(record["members"])), "exclude_previous_group_audits", {"previous": sorted(earlier), "members": record["members"]})
        earlier.update(h["item"] for h in record["audit_history"])
        overlap_count += 1
    checks.check(overlap_count == 3, "overlap_fixture_exercised")

    # Exercise an actually accepted repair and its preservation through final fit.
    accept_L = np.zeros((160, 2), int); accept_L[:120, 1] = 1
    accept_gold = np.zeros(160, int); accept_gold[:120] = 1
    with patch.object(core, "dawid_skene", deterministic_base_model):
        accepted = core.run_group_repair(accept_L, 2, accept_gold, budget=100, group_share=1.,
                                        min_size=10, max_groups=1, k_max=80, seed=9)
    final, info = accepted["guarded"]
    accepted_records = [g for g in info["log"] if g["guarded"]]
    checks.check(len(accepted_records) == 1, "accepted_repair_fixture_exercised", {"accepted": len(accepted_records)})
    for record in accepted_records:
        remaining = set(record["members"])-set(info["audited_labels"])
        checks.check(record["remainder_lb_exact"] > 0 and record["audited"] < record["size"], "strict_remainder_acceptance")
        checks.check(all(final[i] == record["alt"] for i in remaining), "final_refit_preserves_accepted_remainder")
        checks.check(info["repaired"] == len(remaining), "repaired_item_accounting")
    checks.check(all(final[i] == y for i,y in info["audited_labels"].items()), "audits_are_gold_in_final_output")
    return {"status": "completed", "hand_checked_candidate_groups": len(expected), "freeze_calls": len(snapshots),
            "warmup_audits": 4, "remainder_records_reconstructed": checked_remainders,
            "hidden_gold_items_perturbed": len(hidden), "overlap_groups_checked": overlap_count,
            "accepted_groups_exercised": len(accepted_records),
            "limitation": "The base estimator is replaced by a deterministic observable model to isolate search, freezing, oracle accounting, and repair logic. This does not validate Dawid-Skene estimation or newly proposed G1/G2/S1 implementations."}


def manual_pac(pred, scores, gold, draws, already, epsilon=F(1,5), alpha=F(1,5)):
    """Independent scalar rational test + explicit sets for oracle cost accounting."""
    p = [int(v) for v in pred]
    u = [float(v) for v in scores]
    for i, y in already.items():
        p[i], u[i] = int(y), -1.
    candidates = sorted(set(v for v in u if v >= 0))
    certificate = []
    capitals = []
    for threshold in candidates:
        capital, s, ss = F(1), F(0), F(0)
        for t, i in enumerate(draws):
            z = F(int(p[i] != int(gold[i]) and u[i] <= threshold))
            mu = (epsilon/2+s)/(t+1)
            variance = (F(1,4)+ss)/(t+1)
            gap = epsilon-mu
            stake = F(0) if gap <= 0 else min(gap/(variance+gap*gap), F(1,2)/(1-epsilon))
            capital *= 1+stake*(epsilon-z)
            s += z
            ss += (z-mu)**2
        capitals.append(capital)
        certificate.append(capital >= 1/alpha)
    first_failure = next((j for j, ok in enumerate(certificate) if not ok), None)
    threshold = math.inf if first_failure is None else candidates[first_failure]
    threshold_indices = {i for i, score in enumerate(u) if score >= threshold}
    sampled_indices = set(int(i) for i in draws)
    final = p.copy()
    for i in threshold_indices | sampled_indices:
        final[i] = int(gold[i])
    new = (threshold_indices | sampled_indices)-set(already)
    calibration = sampled_indices-set(already)
    return np.array(final), {"expert_new": len(new), "u_hat": threshold,
        "calibration_new": len(calibration), "threshold_new": len(new-calibration),
        "error": sum(a != int(b) for a,b in zip(final,gold))/len(gold)}, {
        "candidates": candidates, "certified": certificate, "capitals": [float(c) for c in capitals],
        "threshold_items": sorted(threshold_indices), "sampled_items": sorted(sampled_indices), "new_items": sorted(new)}


class FixedDraws:
    def __init__(self, draws):
        self.draws = np.array(draws, dtype=int)
    def integers(self, low, high, size):
        if low != 0 or size != len(self.draws) or np.any(self.draws >= high):
            raise AssertionError("Unexpected calibration sampling request")
        return self.draws.copy()


def fixed_pac_check(pac, checks):
    fixtures = []
    def add(name, p, u, y, draws, already=None):
        fixtures.append((name, np.array(p), np.array(u, dtype=float), np.array(y), list(draws), already or {}))
    add("zero_draws", [0]*6, [.1,.1,.2,.3,.4,.5], [0,1,0,1,0,1], [])
    add("all_correct_all_thresholds_certified", [0]*8, np.arange(8)/8, [0]*8, [0,0,1,2]*10)
    add("all_wrong_first_threshold_fails", [0]*8, np.arange(8)/8, [1]*8, [0,1,2,3]*10)
    add("first_failing_threshold_with_ties", [0]*10, [0,0,.2,.2,.4,.4,.6,.6,.8,.8], [0,0,0,0,0,0,1,1,1,1], list(range(10))*6)
    add("known_calibration_threshold_overlap", [0]*10, [0,.1,.2,.3,.4,.5,.6,.7,.8,.9], [0,1,0,1,0,1,0,1,0,1], [0,1,1,2,3,6,9]*7, {0:0,2:0,9:1})
    add("all_owned_no_candidate_threshold", [0]*4, [.2]*4, [1,0,1,0], [0,1,2,3]*10, {0:1,1:0,2:1,3:0})
    rng = np.random.default_rng(3391)
    for case in range(12):
        p = rng.integers(0,3,12); y = rng.integers(0,3,12); u = rng.integers(0,5,12)/5
        known = {int(i): int(y[i]) for i in rng.choice(12, case % 4, replace=False)}
        add(f"fixed_random_{case}", p, u, y, rng.integers(0,12,48), known)
    rows = []
    for name, p, u, y, draws, already in fixtures:
        before = [digest_array(a) for a in (p,u,y)]
        expected_final, expected_info, oracle = manual_pac(p,u,y,draws,already)
        with patch.object(pac.np.random, "default_rng", return_value=FixedDraws(draws)):
            final, info = pac.pac_labels(p,u,y,eps=.2,alpha=.2,m=len(draws),already=already,seed=0)
        checks.check(np.array_equal(final, expected_final), "pac_manual_labels", {"fixture": name})
        for key in expected_info:
            checks.check(info[key] == expected_info[key], "pac_manual_threshold_and_cost", {"fixture": name, "key": key, "expected": expected_info[key], "actual": info[key]})
        checks.check(before == [digest_array(a) for a in (p,u,y)], "pac_input_freeze_no_mutation", {"fixture": name})
        checks.check(info["expert_new"] == info["calibration_new"]+info["threshold_new"], "pac_cost_partition", {"fixture": name})
        rows.append({"fixture": name, "n": len(p), "m": len(draws), "owned": len(already), "result": info, **oracle})
    # Inject capital increments to isolate the threshold-selection control flow.
    # The synthetic certificate mask [True,False,True] must stop at its first
    # failure; it is not claimed to be a realized statistical sample path.
    injected_increments = np.log(np.array([6.,1.,6.]))
    with patch.object(pac.np.random,"default_rng",return_value=FixedDraws([2])), patch.object(pac.np,"log1p",return_value=injected_increments):
        branch_final,branch_info=pac.pac_labels(np.zeros(3,int),np.array([.1,.2,.3]),np.ones(3,int),eps=.2,alpha=.2,m=1)
    checks.check(branch_info["u_hat"]==.2 and branch_info["expert_new"]==2 and branch_final.tolist()==[0,1,1],
                 "pac_first_failure_nonmonotone_control_flow",branch_info)
    return {"status": "completed", "fixed_input_cases": len(rows), "nonmonotone_control_flow_injections":1, "epsilon": .2, "alpha_final": .2,
            "reference": "scalar exact rational capital for every threshold; manually formed union of gold-query index sets",
            "fixtures": rows}


def stream_check(runner_path, checks):
    file = Path(runner_path)
    tree = ast.parse(file.read_text())
    source = ast.unparse(tree)
    seed_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"pac_labels", "run_group_repair"}:
            seed = next((ast.unparse(k.value) for k in node.keywords if k.arg == "seed"), None)
            seed_calls.append({"function": node.func.id, "seed_expression": seed})
    expected = {("pac_labels", "pac_streams[r]"), ("run_group_repair", "ss_pre")}
    checks.check(expected.issubset({(x["function"],x["seed_expression"]) for x in seed_calls}), "pipeline_separate_seed_arguments", {"calls": seed_calls})
    spawn_three = any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "spawn"
                      and [ast.unparse(t) for t in n.targets] == ["(ss_data, ss_pre, ss_pac)"]
                      and [ast.literal_eval(a) for a in n.value.args] == [3]
                      for n in ast.walk(tree))
    checks.check(spawn_three and "pac_streams = ss_pac.spawn(a.pac_reps)" in source,
                 "pipeline_distinct_seedsequence_children")
    children = np.random.SeedSequence([20261104, 883]).spawn(3)
    pac_children = children[2].spawn(6)
    paths = [s.spawn_key for s in children[:2]] + [s.spawn_key for s in pac_children]
    checks.check(len(paths) == len(set(paths)), "verification_random_stream_paths_unique")
    checks.warn("stream_scope", "AST seed inspection verifies the named existing runner and distinct stream wiring, not statistical independence of arbitrary callers or future v8 score-training code.")
    return {"status": "completed", "file": str(file), "seed_calls": seed_calls,
            "verification_spawn_keys": paths, "limitation": "No claim of independence can be established solely by inspecting integer seeds; scores/predictions must be frozen before independent calibration draws."}


def wor_mc_check(core, checks, reps, seed, alpha_repair):
    fixtures = [(50, 25, 25, 0), (100, 55, 45, 0), (100,40,40,20), (300,153,147,0)]
    streams = np.random.SeedSequence([seed,100]).spawn(len(fixtures))
    rows = []
    for (N,ones,zeros,halves), stream in zip(fixtures,streams):
        values = np.array([1.]*ones+[0.]*zeros+[.5]*halves)
        rng = np.random.default_rng(stream)
        miss, harmful, accepted, numerical_zero_margin = 0, 0, 0, 0
        for _ in range(reps):
            order = rng.permutation(values)
            cs = core.WorLowerCS(N, alpha=alpha_repair)
            missed = False
            first_acceptance_recorded = False
            for t,x in enumerate(order,1):
                cs.update(float(x))
                missed |= F.from_float(cs.lower) > F(2*ones+halves,2*N)
                if not first_acceptance_recorded and t < N and cs.remainder_gain_lb() > 0:
                    accepted += 1
                    harmful += (2*order[t:]-1).sum() <= 0
                    exact_grid_margin = N*(2*F(round(cs.lower*400),400)-1)-(2*F(cs.s)-cs.t)
                    numerical_zero_margin += exact_grid_margin == 0
                    first_acceptance_recorded = True
            miss += missed
        interval = cp_interval(miss,reps)
        bad = cp_interval(harmful,reps)
        rows.append({"N":N,"ones":ones,"zeros":zeros,"halves":halves,
                     "miscoverage": interval,"harmful_first_acceptance":bad,"accepted":accepted,
                     "floating_point_positive_margin_with_zero_exact_grid_margin":numerical_zero_margin,
                     "miscoverage_comparison":empirical_classification(interval,alpha_repair),
                     "harmful_comparison":empirical_classification(bad,alpha_repair)})
        # A statistical regression signal; exact mathematical checks
        # are the mechanism for deterministic reproduced defects.
        checks.check(interval["lower"] <= alpha_repair, "wor_mc_excess_regression", rows[-1])
    return {"status":"completed","alpha_repair":alpha_repair,"repetitions_per_fixture":reps,
            "total_trajectories":reps*len(fixtures),"fixtures":rows}


def full_search_mc_check(core, checks, reps, seed, alpha_repair):
    streams = np.random.SeedSequence([seed,200]).spawn(3)
    rows = []
    for case, stream in enumerate(streams):
        n=300
        L=np.zeros((n,4),int)
        if case < 2:
            for j in range(3): L[j*100:(j+1)*100,j+1]=1
        else:
            L[:180,1]=1; L[60:240,2]=1; L[180:,3]=1
        gold=np.zeros(n,int)
        data_rng=np.random.default_rng(np.random.SeedSequence([seed,201,case]))
        for j,p in enumerate((.5,.4,.8)):
            gold[j*100+data_rng.choice(100,int(100*p),replace=False)]=1
        rep_streams=stream.spawn(reps)
        bad=accepted=groups_checked=warmup_cases=0
        with patch.object(core,"dawid_skene",deterministic_base_model):
            for rep_stream in rep_streams:
                ref_share=.15 if case==1 else 0.
                result=core.run_group_repair(L,2,gold,budget=160,alpha=alpha_repair,ref_share=ref_share,
                                             group_share=1.,min_size=12,max_groups=8,k_max=65,
                                             max_audit_frac=.5,disjoint=case<2,seed=rep_stream)
                info=result["guarded"][1]
                bad_run=False
                for record in info["log"]:
                    groups_checked+=1
                    if not record["guarded"]: continue
                    accepted+=1
                    # Every frozen candidate has a common current label; calculate
                    # applied gain from labels without trusting true_gain log fields.
                    excluded=set(record["known_at_start"])|{h["item"] for h in record["audit_history"]}
                    remainder=np.array(sorted(set(record["members"])-excluded),dtype=int)
                    gain=int(((gold[remainder]==record["alt"]).astype(int)-(gold[remainder]==record["cur"]).astype(int)).sum())
                    bad_run |= gain <= 0
                    checks.check(record["remainder_lb_exact"]>0 and record["audited"]<record["size"], "mc_acceptance_guard")
                    checks.check(not (set(remainder)&set(info["audited_labels"])), "accepted_remainder_not_later_audited")
                    checks.check(all(result["guarded"][0][i]==record["alt"] for i in remainder),"accepted_remainder_labels_preserved")
                    warmup_cases+=bool(record["known_at_start"])
                bad+=bad_run
        interval=cp_interval(bad,reps)
        label=("disjoint_mixed_gain","disjoint_warmup_mixed_gain","overlapping_mixed_gain")[case]
        row={"fixture":label,"trials":reps,"groups_tested":groups_checked,"accepted_groups":accepted,
             "accepted_groups_with_known_warmup":warmup_cases,"any_harmful_acceptance":interval,
             "comparison":empirical_classification(interval,alpha_repair),
             "frozen_labels_sha256":digest_array(L),"gold_sha256":digest_array(gold)}
        rows.append(row)
        checks.check(interval["lower"]<=alpha_repair,"search_mc_familywise_excess_regression",row)
    return {"status":"completed","alpha_repair":alpha_repair,"repetitions_per_fixture":reps,
            "total_search_runs":3*reps,"fixtures":rows,
            "limitation":"The full production search and certifier run with a deterministic observable base estimator. These are synthetic implementation checks, not E4 coupling-sweep experiments."}


def make_pac_pools(seed):
    n=240
    rng=np.random.default_rng(np.random.SeedSequence([seed,301]))
    scores=np.repeat(np.arange(12)/12,20)
    pred=np.zeros(n,int)
    gold=np.zeros(n,int); gold[190:]=1
    pools=[("ordered_errors",pred.copy(),scores.copy(),gold.copy(),{})]
    gold=np.zeros(n,int); gold[:45]=1
    pools.append(("misranked_errors",pred.copy(),scores.copy(),gold.copy(),{}))
    gold=np.zeros(n,int); gold[rng.choice(n,17,replace=False)]=1
    pools.append(("all_scores_tied",pred.copy(),np.full(n,.5),gold.copy(),{}))
    gold=rng.integers(0,2,n); pred=gold.copy()
    error_indices=rng.choice(n,40,replace=False); pred[error_indices]=1-pred[error_indices]
    owned={int(i):int(gold[i]) for i in rng.choice(n,24,replace=False)}
    pools.append(("frozen_repaired_with_owned_audits",pred.copy(),scores[::-1].copy(),gold.copy(),owned))
    return pools


def pac_mc_check(pac, checks, reps, seed, alpha_final, epsilon, m):
    pools=make_pac_pools(seed)
    streams=np.random.SeedSequence([seed,300]).spawn(len(pools))
    rows=[]
    for (name,pred,scores,gold,owned),stream in zip(pools,streams):
        before=[digest_array(a) for a in (pred,scores,gold)]
        for a in (pred,scores,gold): a.setflags(write=False)
        bad=0; expert_total=0; nontrivial_savings=0; max_error=0.
        calibration_draw_count=m
        rep_streams=stream.spawn(reps)
        for rep_stream in rep_streams:
            final,info=pac.pac_labels(pred,scores,gold,eps=epsilon,alpha=alpha_final,
                                     m=calibration_draw_count,already=owned,seed=rep_stream)
            actual_error=float(np.mean(final!=gold))
            bad+=actual_error>epsilon+1e-12
            max_error=max(max_error,actual_error)
            expert_total+=len(owned)+info["expert_new"]
            nontrivial_savings+=len(owned)+info["expert_new"]<len(gold)
            checks.check(actual_error==info["error"],"pac_mc_error_recomputed")
            checks.check(info["expert_new"]==info["calibration_new"]+info["threshold_new"],"pac_mc_cost_partition")
            checks.check(all(final[i]==y for i,y in owned.items()),"pac_mc_owned_remain_gold")
        checks.check(before==[digest_array(a) for a in (pred,scores,gold)],"pac_mc_inputs_frozen")
        interval=cp_interval(bad,reps)
        row={"fixture":name,"n":len(gold),"m":calibration_draw_count,"already_owned":len(owned),
             "frozen_input_sha256":dict(zip(("pred","scores","gold"),before)),
             "violation":interval,"comparison":empirical_classification(interval,alpha_final),
             "mean_total_unique_experts":expert_total/reps,"runs_saving_at_least_one_label":nontrivial_savings,
             "maximum_final_error":max_error,"seed_spawn_key":stream.spawn_key}
        rows.append(row)
        checks.check(interval["lower"]<=alpha_final,"pac_mc_excess_regression",row)
    return {"status":"completed","alpha_final":alpha_final,"epsilon":epsilon,
            "repetitions_per_frozen_pool":reps,"total_pac_runs":reps*len(pools),"pools":rows,
            "limitation":"Each frozen synthetic input is conditional and fixed. This does not test annotation-panel variability, audit-trained score leakage, or a general population guarantee."}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root",type=Path,default=ROOT/"v8_experiments")
    ap.add_argument("--runner-path",type=Path,default=HERE/"run_certified_v7_9_reference.py")
    ap.add_argument("--output",type=Path,default=HERE/"verification_report.json")
    ap.add_argument("--profile",choices=("quick","full"),default="full")
    ap.add_argument("--max-n",type=int,default=8)
    ap.add_argument("--reference-grid",type=int,default=41)
    ap.add_argument("--wor-reps",type=int)
    ap.add_argument("--search-reps",type=int)
    ap.add_argument("--pac-reps",type=int)
    ap.add_argument("--seed",type=int,default=20261104)
    ap.add_argument("--alpha-repair",type=float,default=.05)
    ap.add_argument("--alpha-final",type=float,default=.10)
    ap.add_argument("--epsilon",type=float,default=.03)
    ap.add_argument("--pac-m",type=int,default=500)
    args=ap.parse_args()
    if not (1<=args.max_n<=8): ap.error("--max-n must be between 1 and 8 for bounded exhaustive checking")
    if args.reference_grid < 3 or (401-1) % (args.reference_grid-1) or (args.reference_grid-1)%2:
        ap.error("--reference-grid must include .5 and divide the 401-point production grid")
    for name in ("alpha_repair","alpha_final","epsilon"):
        if not 0<getattr(args,name)<1: ap.error(f"--{name.replace('_','-')} must lie in (0,1)")
    defaults={"quick":(100,100,300),"full":(1000,1000,10000)}[args.profile]
    wor_reps=args.wor_reps if args.wor_reps is not None else defaults[0]
    search_reps=args.search_reps if args.search_reps is not None else defaults[1]
    pac_reps=args.pac_reps if args.pac_reps is not None else defaults[2]
    if args.pac_m<0: ap.error("--pac-m must be nonnegative")
    if min(wor_reps,search_reps,pac_reps)<=0: ap.error("All repetition counts must be positive")
    start=time.monotonic()
    core,pac=load_source(args.source_root)
    checks=Checks()
    source_files=[args.source_root/"audit_repair"/"core.py",args.source_root/"audit_repair"/"pac.py",args.runner_path]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    report={"schema_version":1,"artifact_kind":"implementation_verification_not_study_results",
            "source_root":str(args.source_root.resolve()),"source_sha256":hashes,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python":sys.version,"numpy":np.__version__,"configuration":{"profile":args.profile,
            "seed":args.seed,"max_N":args.max_n,"reference_grid":args.reference_grid,"production_grid":401,
            "wor_reps":wor_reps,"search_reps":search_reps,"pac_reps":pac_reps,
            "alpha_repair":args.alpha_repair,"alpha_final":args.alpha_final,"epsilon":args.epsilon,"pac_m":args.pac_m},
            "probability_targets":{"repair_familywise_failure_budget":args.alpha_repair,
            "final_error_failure_budget":args.alpha_final,
            "joint_union_bound_failure_budget":min(1.,args.alpha_repair+args.alpha_final),
            "interpretation":"Repair and final PAC probabilities are separate. The union-bound sum applies only to the conjunction of both stage guarantees; final error alone uses alpha_final."},"sections":{}}

    tasks=[("exact_wor_and_remainder",lambda:exact_wor_check(core,checks,args.max_n,401,args.reference_grid)),
           ("grid_prefix_inversion",lambda:grid_inversion_check(core,checks)),
           ("frozen_candidates_and_remainders",lambda:group_structure_check(core,checks)),
           ("fixed_input_pac_manual_oracle",lambda:fixed_pac_check(pac,checks)),
           ("random_stream_wiring",lambda:stream_check(args.runner_path,checks)),
           ("wor_monte_carlo",lambda:wor_mc_check(core,checks,wor_reps,args.seed,args.alpha_repair)),
           ("full_search_monte_carlo",lambda:full_search_mc_check(core,checks,search_reps,args.seed,args.alpha_repair)),
           ("frozen_input_pac_monte_carlo",lambda:pac_mc_check(pac,checks,pac_reps,args.seed,args.alpha_final,args.epsilon,args.pac_m))]
    for name,fn in tasks:
        print(f"START {name}",flush=True)
        section_start=time.monotonic()
        try:
            report["sections"][name]=fn()
            report["sections"][name]["elapsed_seconds"]=round(time.monotonic()-section_start,3)
        except Exception as exc:
            import traceback
            checks.check(False,name+"_execution_error",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
            report["sections"][name]={"status":"error","error":str(exc)}
        print(f"DONE {name} {time.monotonic()-section_start:.2f}s",flush=True)
        # Checkpoints stay in the same verification directory. They are not a completed report.
        report["status"]="in_progress"
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+"\n")
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    checks.check(unchanged,"source_files_unchanged")
    report.update({"status":"failed" if checks.failures else "passed_under_tested_conditions",
                   "elapsed_seconds":round(time.monotonic()-start,3),"assertion_count":checks.assertion_count,
                   "failure_count_retained":len(checks.failures),"failures":checks.failures,"warnings":checks.warnings,
                   "source_files_unchanged":unchanged,
                   "limitations":["These checks are not proofs of the general statistical guarantees.",
                   "Exhaustive coverage is limited to ternary finite populations with N at most 8; every distinct ordering has its exact combinatorial probability. Mean coverage compares the exact rational value of the reported floating-point lower endpoint to the exact population mean and tracks the maximum over every prefix.",
                   "Exact rational capital agreement is checked on the configured nested reference grid; the complete 401-point production grid is used for exhaustive coverage and all simulations.",
                   "The grid check does not use bisection or assume monotonicity across nulls, and it does not assert a continuous-bound dominance property.",
                   "One-sided Clopper-Pearson bounds describe each fixed simulation fixture; a bound above nominal is inconclusive rather than proof of failure.",
                   "Simulation classification is fail when the one-sided 95% lower bound exceeds nominal, pass_with_margin when its upper bound is below nominal, and inconclusive otherwise. Several fixtures are tested without simultaneous adjustment, so simulation flags can occur by chance; deterministic defects have separate check IDs.",
                   "Inputs are assumed valid: truthful owned labels, finite nonnegative uncertainties for unowned items, permissible alpha/epsilon, and frozen predictions/scores.",
                   "G1/G2/S1 text features, cross-fitting, model fitting, real annotation panels, and E1-E4 study outcomes are outside this verification run."]})
    args.output.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"report":str(args.output),"elapsed_seconds":report["elapsed_seconds"],
                      "assertions":checks.assertion_count,"failures":len(checks.failures)}),flush=True)
    return 1 if checks.failures else 0


if __name__=="__main__":
    raise SystemExit(main())
