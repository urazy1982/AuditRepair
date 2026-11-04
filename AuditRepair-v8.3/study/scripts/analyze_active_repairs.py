#!/usr/bin/env python3
"""Retrospective v8.2 analysis of the frozen v8.1 factorial's active repairs.

Read-only with respect to frozen_v8_1 and corrected.  This does not sample new
PAC repetitions or collect annotations.  The 31 accepted preprocessing states
are reconstructed only to recover item-level mechanism information.  Their
saved array hashes must agree exactly before that information is used.

Run from any directory:
    python3 scripts/analyze_active_repairs.py

The output is a descriptive amendment after the v8.1 results and peer review
were seen.  The original pointwise, whole-preprocessing-seed bootstrap
intervals are retained for the all-seed mean effects.  Frequencies and the
accepted-state-only summaries do not receive independent-repetition CIs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Do not leave bytecode in the immutable inputs.  Match the CPU runner's
# single-thread defaults, without overriding explicitly selected settings.
sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parents[1]
GENERATORS = ("G0_uniform", "G1", "G2")
SCORES = ("S0", "S1")
CLASSIFICATIONS = (
    "no_certified_action_in_sampled_seeds",
    "certified_action_but_all_paired_costs_identical",
    "certified_action_and_some_nonzero_cost_differences",
)
MECHANISMS = (
    "no_actual_prediction_change_vs_final_DS",
    "all_changed_predictions_strictly_above_baseline_threshold",
    "prefix_reached_but_sampled_prefix_losses_unchanged",
    "sampled_prefix_losses_changed_threshold_unchanged",
    "threshold_changed_unique_purchase_union_unchanged",
    "expert_cost_decreased",
    "expert_cost_increased",
)
STATE_CLASSES = (
    "no_accepted_group",
    "accepted_but_no_actual_prediction_change_vs_final_DS",
    "accepted_actual_changes_with_zero_net_gain_vs_final_DS",
    "accepted_actual_changes_with_negative_net_gain_vs_final_DS",
    "accepted_actual_changes_with_positive_net_gain_vs_final_DS",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("infinity" if value > 0 else "-infinity")
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class Checks:
    def __init__(self):
        self.counts = Counter()

    def require(self, condition, name: str, detail=None):
        self.counts[name] += 1
        if not bool(condition):
            raise AssertionError(f"{name}: {detail}")


def thresholds(value):
    return float(value)


def state_key(pool, seed, generator):
    return f"{pool}/seed_{seed:02d}/{generator}"


def paired_key(pool, seed, generator, score, rep):
    return f"{state_key(pool, seed, generator)}/{score}/rep_{rep:02d}"


def classify_state(diagnostics):
    if diagnostics["certified_groups"] == 0:
        return STATE_CLASSES[0]
    if diagnostics["changed_items_vs_same_audit"] == 0:
        return STATE_CLASSES[1]
    gain = diagnostics["net_gain_vs_same_audit"]
    return STATE_CLASSES[2] if gain == 0 else STATE_CLASSES[3] if gain < 0 else STATE_CLASSES[4]


def quantiles(values, prefix):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {f"{prefix}_{name}": None for name in ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max")}
    return {f"{prefix}_{name}": float(value) for name, value in zip(
        ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"),
        np.quantile(v, [0, .01, .05, .25, .5, .75, .95, .99, 1]),
    )}


def seed_summary(part, diagnostics):
    savings = part.reviews_saved.to_numpy(dtype=int)
    cut = int(part.practical_threshold_labels.iloc[0])
    first = part.iloc[0]
    return {
        "state_id": first.state_id, "pool": first.pool, "generator": first.generator,
        "score": first.score, "seed": int(first.seed), "n": int(first.n),
        "paired_repetitions": len(part),
        "has_accepted_group": bool(first.has_accepted_group),
        "accepted_groups": int(diagnostics["certified_groups"]),
        "net_gain_vs_frozen_reference": int(diagnostics["net_gain_vs_ref"]),
        "net_gain_vs_final_same_audit_DS": int(diagnostics["net_gain_vs_same_audit"]),
        "actual_changed_predictions_vs_final_DS": int(diagnostics["changed_items_vs_same_audit"]),
        "mean_reviews_saved": float(savings.mean()),
        "changed_cost_repetitions": int(np.count_nonzero(savings)),
        "saving_repetitions": int((savings > 0).sum()),
        "cost_increase_repetitions": int((savings < 0).sum()),
        "practical_threshold_labels": cut,
        "savings_at_least_1pct_pool_repetitions": int((savings >= cut).sum()),
        "increases_at_least_1pct_pool_repetitions": int((savings <= -cut).sum()),
        "changed_cost_fraction": float(np.mean(savings != 0)),
        "saving_fraction": float(np.mean(savings > 0)),
        "cost_increase_fraction": float(np.mean(savings < 0)),
        "practical_saving_fraction": float(np.mean(savings >= cut)),
        "practical_increase_fraction": float(np.mean(savings <= -cut)),
        **quantiles(savings, "repetition_savings"),
    }


def distribution(part, seed_part, scope, original_effect, checks):
    savings = part.reviews_saved.to_numpy(dtype=int)
    seed_values = seed_part.mean_reviews_saved.to_numpy(dtype=float)
    cut, n = int(part.practical_threshold_labels.iloc[0]), int(part.n.iloc[0])
    checks.require(np.isclose(savings.mean(), seed_values.mean(), atol=1e-12, rtol=0),
                   "equal_replication_agrees_with_cluster_mean")
    all_seeds = scope == "all_prespecified_preprocessing_seeds"
    mean = float(original_effect["mean_reviews_saved"]) if all_seeds else float(seed_values.mean())
    if all_seeds:
        checks.require(np.isclose(mean, savings.mean(), atol=1e-12, rtol=0),
                       "archive_mean_matches_integer_paired_effects")
    baseline_mean = float(part.baseline_expert_total.mean())
    result = {
        "pool": part.pool.iloc[0], "generator": part.generator.iloc[0],
        "score": part.score.iloc[0], "scope": scope, "n": n,
        "preprocessing_seeds": len(seed_part), "paired_repetitions": len(part),
        "repetitions_per_seed": int(part.groupby("seed").size().iloc[0]),
        "accepted_preprocessing_seeds": int(seed_part.has_accepted_group.sum()),
        "accepted_group_actions": int(seed_part.accepted_groups.sum()),
        "changed_cost_repetitions": int((savings != 0).sum()),
        "identical_cost_repetitions": int((savings == 0).sum()),
        "saving_repetitions": int((savings > 0).sum()),
        "cost_increase_repetitions": int((savings < 0).sum()),
        "changed_cost_fraction": float(np.mean(savings != 0)),
        "saving_fraction": float(np.mean(savings > 0)),
        "cost_increase_fraction": float(np.mean(savings < 0)),
        "practical_threshold_labels": cut,
        "savings_at_least_1pct_pool_repetitions": int((savings >= cut).sum()),
        "increases_at_least_1pct_pool_repetitions": int((savings <= -cut).sum()),
        "practical_saving_fraction": float(np.mean(savings >= cut)),
        "practical_increase_fraction": float(np.mean(savings <= -cut)),
        "seeds_with_any_cost_change": int((seed_part.changed_cost_repetitions > 0).sum()),
        "seeds_with_any_saving": int((seed_part.saving_repetitions > 0).sum()),
        "seeds_with_any_cost_increase": int((seed_part.cost_increase_repetitions > 0).sum()),
        "seeds_with_practical_saving": int((seed_part.savings_at_least_1pct_pool_repetitions > 0).sum()),
        "seeds_with_practical_increase": int((seed_part.increases_at_least_1pct_pool_repetitions > 0).sum()),
        "seeds_with_nonzero_mean": int((seed_values != 0).sum()),
        "baseline_mean_expert_total": baseline_mean,
        "repair_mean_expert_total": float(part.repair_expert_total.mean()),
        "mean_reviews_saved": mean,
        "mean_savings_pct_of_pool": 100 * mean / n,
        "mean_savings_pct_of_baseline_cost_ratio_of_means": 100 * mean / baseline_mean,
        "original_ci95_saved_low": float(original_effect["ci95_saved_low"]) if all_seeds else None,
        "original_ci95_saved_high": float(original_effect["ci95_saved_high"]) if all_seeds else None,
        "bootstrap_interval_degenerate": bool(original_effect["ci95_saved_low"] == original_effect["ci95_saved_high"]) if all_seeds else None,
        "mean_interval_scope": ("original pointwise percentile bootstrap of 20 preprocessing-seed means; fixed panel; no equivalence claim"
                                if all_seeds else "not computed: descriptive conditioning on observed acceptance"),
        "frequency_scope": "empirical frequencies of clustered repetitions; no independent-repetition CI",
        **quantiles(savings, "repetition_savings"),
        **quantiles(savings[savings != 0], "nonzero_repetition_savings"),
        **quantiles(seed_values, "seed_mean_savings"),
    }
    for column in ("changed_cost_fraction", "practical_saving_fraction", "practical_increase_fraction"):
        vals = seed_part[column].to_numpy()
        result[f"seed_{column}_min"] = float(vals.min())
        result[f"seed_{column}_median"] = float(np.median(vals))
        result[f"seed_{column}_max"] = float(vals.max())
    return result


def reconstruct(d, generator, common, frozen_factorial, corrected_factorial,
                cache_data, checks):
    pool, seed = d["pool"], d["seed"]
    _, labels0, gold0, k = common.load_pool(pool)
    x0, neighbors0 = cache_data[pool]
    p, _ = common.protocol(validate=False)
    a = p["parameters"]
    data_seed, pre_seed, pac_seeds = common.streams(a["root_seed"], seed, a["pac_reps"])
    permutation = np.random.default_rng(data_seed).permutation(len(gold0))
    checks.require(common.array_hash(permutation) == d["permutation_sha256"],
                   "active_permutation_hash_matches", (pool, seed, generator))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    labels, gold, x = labels0[permutation], gold0[permutation], x0[permutation]
    neighbors = inverse[neighbors0[permutation]]
    state = frozen_factorial.preprocess_generator(labels, k, gold, x, neighbors, generator, pre_seed)
    expected = d["frozen_arrays"][generator]
    arrays = {"base": state["base"], "repair": state["pred"],
              "oracle": state["oracle_candidate_labels"],
              "audits": np.array(sorted(state["audits"]), dtype=np.int64),
              "warmup": state["warm_indices"],
              "S0": state["scores"]["S0"], "S1": state["scores"]["S1"]}
    obtained = {name: common.array_hash(array) for name, array in arrays.items()}
    for name, value in obtained.items():
        wanted = expected["scores"][name] if name in SCORES else expected[name]
        checks.require(value == wanted, "active_saved_array_hash_matches",
                       (pool, seed, generator, name, wanted, value))
    archived = d["preprocessing"][generator]
    checks.require(state["diagnostics"] == archived["diagnostics"],
                   "active_frozen_diagnostics_match", (pool, seed, generator))
    checks.require(state["group_log"] == archived["group_log"],
                   "active_frozen_complete_group_logs_match", (pool, seed, generator))
    checks.require(state["groups"] == archived["groups"],
                   "active_frozen_complete_shortlist_matches", (pool, seed, generator))
    checks.require([int(permutation[i]) for i in sorted(state["audits"])] == archived["original_row_ids"],
                   "active_frozen_original_audit_ids_match", (pool, seed, generator))
    corrected_hashes = None
    if corrected_factorial is not None:
        corrected = corrected_factorial.preprocess_generator(labels, k, gold, x, neighbors, generator, pre_seed)
        corrected_arrays = {"base": corrected["base"], "repair": corrected["pred"],
                            "oracle": corrected["oracle_candidate_labels"],
                            "audits": np.array(sorted(corrected["audits"]), dtype=np.int64),
                            "warmup": corrected["warm_indices"],
                            "S0": corrected["scores"]["S0"], "S1": corrected["scores"]["S1"]}
        corrected_hashes = {name: common.array_hash(array) for name, array in corrected_arrays.items()}
        for name, value in corrected_hashes.items():
            checks.require(value == obtained[name], "corrected_active_saved_array_hash_matches",
                           (pool, seed, generator, name))
        checks.require(corrected["diagnostics"] == state["diagnostics"],
                       "corrected_active_diagnostics_match", (pool, seed, generator))
        checks.require(len(corrected["group_log"]) == len(state["group_log"]),
                       "corrected_active_group_log_count_matches", (pool, seed, generator))
        for old, new in zip(state["group_log"], corrected["group_log"]):
            for key in ("group_index", "decision", "guarded", "stop_reason", "audited", "audit_history",
                        "known_at_start", "remainder_members_at_stop", "lower_mean",
                        "true_remainder_gain_vs_frozen_ref", "effect_vs_same_audit"):
                checks.require(old[key] == new[key], "corrected_active_trajectory_matches",
                               (pool, seed, generator, old["group_index"], key))
    return state, gold, permutation, pac_seeds, {
        "state_id": state_key(pool, seed, generator),
        "pool": pool, "seed": seed, "generator": generator,
        "frozen_reconstructed_hashes": obtained,
        "corrected_reconstructed_hashes": corrected_hashes,
        "saved_hashes_all_exact": True,
        "frozen_full_group_logs_exact": True,
        "corrected_saved_arrays_and_trajectories_exact": corrected_factorial is not None,
    }


def inspect_state(d, generator, state, gold, permutation, pac_seeds, m, checks, common):
    pool, seed, n = d["pool"], d["seed"], len(gold)
    sid = state_key(pool, seed, generator)
    base, pred, ref = state["base"], state["pred"], state["ref"]
    audit_ids = np.array(sorted(state["audits"]), dtype=int)
    audit_mask = np.zeros(n, dtype=bool)
    audit_mask[audit_ids] = True
    accepted = [g for g in state["group_log"] if g["guarded"]]
    group_rows, item_rows, mechanism_rows = [], [], []
    diag = state["diagnostics"]
    for group in accepted:
        gid = f"{sid}/group_{group['group_index']:02d}"
        rem = np.asarray(group["remainder_members_at_stop"], dtype=int)
        checks.require(not audit_mask[rem].any(), "accepted_remainder_excluded_from_final_audits", gid)
        changed = pred[rem] != base[rem]
        corrected = (base[rem] != gold[rem]) & (pred[rem] == gold[rem])
        introduced = (base[rem] == gold[rem]) & (pred[rem] != gold[rem])
        i2c_ref = (ref[rem] != gold[rem]) & (pred[rem] == gold[rem])
        c2i_ref = (ref[rem] == gold[rem]) & (pred[rem] != gold[rem])
        gain_ref = int(i2c_ref.sum() - c2i_ref.sum())
        gain_base = int(corrected.sum() - introduced.sum())
        checks.require(gain_ref == group["true_remainder_gain_vs_frozen_ref"], "accepted_group_reference_gain_matches", gid)
        checks.require(gain_base == group["effect_vs_same_audit"], "accepted_group_same_audit_gain_matches", gid)
        gain_sum = sum(int(h["gain"]) for h in group["audit_history"])
        grid = np.linspace(0., 1., 401)
        j_candidates = np.flatnonzero(grid == group["lower_mean"])
        checks.require(len(j_candidates) == 1, "group_lower_index_identified", gid)
        j = int(j_candidates[0])
        numerator = int(group["size"]) * (j - 200) - 200 * gain_sum
        group_rows.append({
            "group_id": gid, "state_id": sid, "pool": pool, "seed": seed, "generator": generator,
            "group_index": group["group_index"], "frozen_reference_label": group["cur"],
            "replacement_label": group["alt"], "group_alpha": group["alpha"],
            "population_size": group["size"], "audits_at_acceptance": group["audited"],
            "accepted_remainder_size": len(rem), "lower_grid_index": j,
            "grid_denominator": 400, "observed_gain_sum": gain_sum,
            "nominal_remainder_margin_numerator_over_200": numerator,
            "nominal_remainder_margin": numerator / 200,
            "incorrect_to_correct_vs_frozen_reference": int(i2c_ref.sum()),
            "correct_to_incorrect_vs_frozen_reference": int(c2i_ref.sum()),
            "net_gain_vs_frozen_reference": gain_ref,
            "actual_changed_predictions_vs_final_DS": int(changed.sum()),
            "incorrect_to_correct_vs_final_same_audit_DS": int(corrected.sum()),
            "correct_to_incorrect_vs_final_same_audit_DS": int(introduced.sum()),
            "wrong_to_different_wrong_vs_final_DS": int((changed & ~corrected & ~introduced).sum()),
            "net_gain_vs_final_same_audit_DS": gain_base,
            "final_DS_already_equals_replacement": int((base[rem] == pred[rem]).sum()),
            "final_DS_differs_from_frozen_reference_on_remainder": int((base[rem] != ref[rem]).sum()),
            "membership_sha256": group["membership_id"],
            "remainder_permuted_indices_sha256": common.array_hash(rem),
            "remainder_original_row_indices_sha256": common.array_hash(permutation[rem]),
        })
        for pos, i in enumerate(rem):
            item_rows.append({
                "group_id": gid, "state_id": sid, "pool": pool, "seed": seed,
                "generator": generator, "group_index": group["group_index"],
                "permuted_row_index": int(i), "original_row_index": int(permutation[i]),
                "gold": int(gold[i]), "frozen_reference": int(ref[i]),
                "final_same_audit_DS": int(base[i]), "accepted_prediction": int(pred[i]),
                "actual_prediction_changed_vs_final_DS": bool(changed[pos]),
                "corrected_vs_final_DS": bool(corrected[pos]),
                "introduced_error_vs_final_DS": bool(introduced[pos]),
                "S0_frozen": float(state["scores"]["S0"][i]),
                "S1_frozen": float(state["scores"]["S1"][i]),
            })
    checks.require(sum(x["net_gain_vs_frozen_reference"] for x in group_rows) == diag["net_gain_vs_ref"],
                   "accepted_group_reference_gains_sum_to_state", sid)
    checks.require(sum(x["net_gain_vs_final_same_audit_DS"] for x in group_rows) == diag["net_gain_vs_same_audit"],
                   "accepted_group_same_audit_gains_sum_to_state", sid)
    checks.require(sum(x["actual_changed_predictions_vs_final_DS"] for x in group_rows) == diag["changed_items_vs_same_audit"],
                   "accepted_group_changed_items_sum_to_state", sid)

    rows = {(r["score"], r["rep"], r["role"]): r for r in d["rows"] if r.get("generator") == generator}
    changed = (pred != base) & ~audit_mask
    corrected = (base != gold) & (pred == gold) & changed
    introduced = (base == gold) & (pred != gold) & changed
    wrong_to_wrong = changed & ~corrected & ~introduced
    loss_changed = ((base != gold) != (pred != gold)) & ~audit_mask
    group_ids = ";".join(g["group_id"] for g in group_rows)
    for score in SCORES:
        u = state["scores"][score].copy()
        u[audit_mask] = -1.
        candidates = np.unique(u[u >= 0])
        for rep, pac_seed in enumerate(pac_seeds):
            b, r = rows[(score, rep, "same_audit")], rows[(score, rep, "repair")]
            bid = paired_key(pool, seed, generator, score, rep)
            baseline_u, repair_u = thresholds(b["u_hat"]), thresholds(r["u_hat"])
            checks.require(baseline_u == thresholds(r["baseline_u_hat"]), "recorded_baseline_threshold_matches", bid)
            for value in (baseline_u, repair_u):
                checks.require(math.isinf(value) or np.any(candidates == value), "archived_threshold_is_exact_candidate", (bid, value))
            draws = np.random.default_rng(pac_seed).integers(0, n, m)
            checks.require(common.array_hash(draws) == b["draws_sha256"] == r["draws_sha256"],
                           "active_draw_hash_matches", bid)
            draw_mask = np.zeros(n, dtype=bool)
            draw_mask[draws] = True
            baseline_review, repair_review = (u >= baseline_u) & ~audit_mask, (u >= repair_u) & ~audit_mask
            baseline_purchased = audit_mask | draw_mask | baseline_review
            repair_purchased = audit_mask | draw_mask | repair_review
            checks.require(int(baseline_purchased.sum()) == b["expert_total"], "baseline_unique_union_matches", bid)
            checks.require(int(repair_purchased.sum()) == r["expert_total"], "repair_unique_union_matches", bid)
            removed = baseline_purchased & ~repair_purchased
            added = repair_purchased & ~baseline_purchased
            saved = int(removed.sum() - added.sum())
            checks.require(saved == b["expert_total"] - r["expert_total"] == -r["paired_cost_delta"],
                           "set_difference_matches_paired_savings", bid)
            final_b, final_r = base.copy(), pred.copy()
            final_b[baseline_purchased] = gold[baseline_purchased]
            final_r[repair_purchased] = gold[repair_purchased]
            checks.require(float((final_b != gold).mean()) == b["error"], "baseline_final_error_reconstructed", bid)
            checks.require(float((final_r != gold).mean()) == r["error"], "repair_final_error_reconstructed", bid)
            region_masks = {"below": u < baseline_u, "at": u == baseline_u, "above": u > baseline_u}
            positions = {}
            for name, mask in (("changed", changed), ("corrected", corrected),
                               ("introduced_errors", introduced), ("wrong_to_wrong", wrong_to_wrong)):
                positions[name] = int(mask.sum())
                for region, region_mask in region_masks.items():
                    positions[f"{name}_{region}_baseline_threshold"] = int((mask & region_mask).sum())
                checks.require(sum(positions[f"{name}_{region}_baseline_threshold"] for region in region_masks) == positions[name],
                               "strict_equal_above_exhaustive_partition", (bid, name))
            for key in ("changed", "corrected", "introduced_errors"):
                checks.require(positions[key] == r[key], "archived_change_count_matches", (bid, key))
            for region in region_masks:
                checks.require(positions[f"changed_{region}_baseline_threshold"] == r[f"changed_{region}"],
                               "archived_changed_position_matches", (bid, region))
            checks.require(positions["corrected_below_baseline_threshold"] + positions["corrected_at_baseline_threshold"] == r["corrected_le"],
                           "archived_corrected_prefix_count_matches", bid)
            checks.require(positions["introduced_errors_below_baseline_threshold"] + positions["introduced_errors_at_baseline_threshold"] == r["introduced_errors_le"],
                           "archived_introduced_prefix_count_matches", bid)
            changed_loss_prefix_draws = loss_changed[draws] & (u[draws] <= baseline_u)
            improved_prefix_draws = corrected[draws] & (u[draws] <= baseline_u)
            worsened_prefix_draws = introduced[draws] & (u[draws] <= baseline_u)
            strict_locality = bool(changed.any() and not (changed & (u <= baseline_u)).any())
            sample_prefix_invariant = not changed_loss_prefix_draws.any()
            threshold_changed = baseline_u != repair_u
            review_changed = not np.array_equal(baseline_review, repair_review)
            purchase_changed = not np.array_equal(baseline_purchased, repair_purchased)
            final_labels_changed = bool(np.any(final_b != final_r))
            if strict_locality:
                checks.require(not threshold_changed and not review_changed and not purchase_changed and not final_labels_changed,
                               "current_factorial_structural_locality_verified", bid)
            if sample_prefix_invariant:
                checks.require(not threshold_changed and not review_changed and not purchase_changed,
                               "current_factorial_sample_prefix_invariance_verified", bid)
            if not changed.any():
                mechanism = MECHANISMS[0]
            elif strict_locality:
                mechanism = MECHANISMS[1]
            elif sample_prefix_invariant:
                mechanism = MECHANISMS[2]
            elif not threshold_changed:
                mechanism = MECHANISMS[3]
            elif not purchase_changed:
                mechanism = MECHANISMS[4]
            elif saved > 0:
                mechanism = MECHANISMS[5]
            elif saved < 0:
                mechanism = MECHANISMS[6]
            else:
                raise AssertionError(f"Unclassified nonnested fixed-score purchase change: {bid}")
            kept = changed & ~repair_purchased
            checks.require(int(kept.sum()) == r["retained_automatic_changes"], "archived_retained_changes_match", bid)
            checks.require(int((corrected & ~repair_purchased).sum()) == r["retained_automatic_corrections"], "archived_retained_corrections_match", bid)
            checks.require(int((introduced & ~repair_purchased).sum()) == r["retained_automatic_new_errors"], "archived_retained_introduced_errors_match", bid)
            mechanism_rows.append({
                "pair_id": bid, "state_id": sid, "accepted_group_ids": group_ids,
                "pool": pool, "seed": seed, "generator": generator, "score": score, "rep": rep,
                "n": n, "accepted_groups": len(accepted),
                "accepted_remainder_items": diag["repaired_unaudited_items"],
                "net_gain_vs_frozen_reference": diag["net_gain_vs_ref"],
                "net_gain_vs_final_same_audit_DS": diag["net_gain_vs_same_audit"],
                **positions,
                "baseline_threshold": baseline_u, "repair_threshold": repair_u,
                "baseline_threshold_rank": int(np.searchsorted(candidates, baseline_u)),
                "repair_threshold_rank": int(np.searchsorted(candidates, repair_u)),
                "threshold_rank_change": int(np.searchsorted(candidates, repair_u) - np.searchsorted(candidates, baseline_u)),
                "threshold_changed": threshold_changed,
                "baseline_review_set_size_excluding_audits": int(baseline_review.sum()),
                "repair_review_set_size_excluding_audits": int(repair_review.sum()),
                "review_set_items_removed": int((baseline_review & ~repair_review).sum()),
                "review_set_items_added": int((repair_review & ~baseline_review).sum()),
                "review_set_changed": review_changed,
                "baseline_unique_expert_total": int(baseline_purchased.sum()),
                "repair_unique_expert_total": int(repair_purchased.sum()),
                "unique_expert_purchases_removed": int(removed.sum()),
                "unique_expert_purchases_added": int(added.sum()),
                "unique_purchase_union_changed": purchase_changed, "reviews_saved": saved,
                "calibration_loss_change_events_through_baseline_threshold": int(changed_loss_prefix_draws.sum()),
                "calibration_improvement_events_through_baseline_threshold": int(improved_prefix_draws.sum()),
                "calibration_worsening_events_through_baseline_threshold": int(worsened_prefix_draws.sum()),
                "calibration_loss_change_unique_items_through_baseline_threshold": int((loss_changed & draw_mask & (u <= baseline_u)).sum()),
                "all_changed_predictions_strictly_above_baseline_threshold": strict_locality,
                "all_sampled_loss_sequences_identical_through_baseline_threshold": sample_prefix_invariant,
                "mechanism_class": mechanism,
                "retained_automatic_changes": int(kept.sum()),
                "retained_automatic_corrections": int((corrected & ~repair_purchased).sum()),
                "retained_automatic_introduced_errors": int((introduced & ~repair_purchased).sum()),
                "final_labels_differ_on_items": int((final_b != final_r).sum()),
                "baseline_final_errors": int((final_b != gold).sum()),
                "repair_final_errors": int((final_r != gold).sum()),
                "draws_sha256": r["draws_sha256"],
            })
    return group_rows, item_rows, mechanism_rows


def make_plot(rep_df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nonzero = rep_df.groupby(["pool", "generator", "score"]).reviews_saved.apply(lambda x: bool((x != 0).any()))
    selected = [key for key, flag in nonzero.items() if flag]
    fig, axes = plt.subplots(len(selected), 1, figsize=(8.2, 9.4), sharex=True, constrained_layout=True)
    names = {"ag_news_5000": "AG News development", "ag_news_holdout": "AG News holdout"}
    for ax, key in zip(np.atleast_1d(axes), selected):
        pool, generator, score = key
        part = rep_df[(rep_df.pool == pool) & (rep_df.generator == generator) & (rep_df.score == score)]
        x = part.seed.to_numpy() + (part.rep.to_numpy() - 4.5) * .055
        y = part.reviews_saved.to_numpy()
        colors = np.where(y > 0, "#086788", np.where(y < 0, "#c64b2e", "#b9bdc5"))
        ax.scatter(x[y == 0], y[y == 0], c=colors[y == 0], s=11, alpha=.5, linewidths=0)
        ax.scatter(x[y != 0], y[y != 0], c=colors[y != 0], s=27, alpha=.9, linewidths=.3, edgecolors="white")
        means = part.groupby("seed").reviews_saved.mean()
        ax.plot(means.index, means.values, "_", color="#262b32", markersize=12, markeredgewidth=1.7, label="Seed mean")
        cut = int(part.practical_threshold_labels.iloc[0])
        for sign in (-1, 1):
            ax.axhline(sign * cut, color="#677584", linewidth=.8, linestyle="--", zorder=0)
        ax.axhline(0, color="#727a85", linewidth=.7, zorder=0)
        ax.set_title(f"{names[pool]} · {generator} · {score}  |  {int((y != 0).sum())}/200 changed costs", fontsize=11, loc="left", fontweight="bold")
        ax.set_ylabel("Expert labels saved\n(negative = extra labels)", fontsize=9)
        ax.set_ylim(-420, 540)
        ax.set_yticks([-400, -200, 0, 200, 400])
        ax.grid(axis="y", alpha=.15)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.text(.01, .97, f"Dashed lines: ±{cut} labels (1% of pool)\nBlack marks: preprocessing-seed means", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="#48515e")
    axes[-1].set_xticks(range(20))
    axes[-1].set_xlabel("Preprocessing seed (10 paired PAC repetitions within each seed)", fontsize=10)
    axes[-1].set_xlim(-.7, 19.7)
    fig.suptitle("Observed cost changes include benefits and increases", fontsize=14, fontweight="bold")
    fig.savefig(out / "active_cost_distribution.png", dpi=220)
    fig.savefig(out / "active_cost_distribution.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=REVISION / "frozen_v8_1")
    parser.add_argument("--out", type=Path, default=REVISION / "results")
    parser.add_argument("--skip-corrected-comparison", action="store_true",
                        help="Recover frozen mechanisms only; record that corrected pipeline comparison was not run.")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    frozen, out = args.frozen.resolve(), args.out.resolve()
    if out == frozen or frozen in out.parents:
        raise ValueError("Outputs must be outside the immutable frozen project")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(frozen))
    common = importlib.import_module("experiment_common")
    factorial = importlib.import_module("factorial")
    corrected = None
    corrected_sources = {}
    if not args.skip_corrected_comparison:
        sys.path.insert(0, str(REVISION))
        corrected = importlib.import_module("corrected.factorial")
        corrected_sources = {str(p.relative_to(REVISION)): sha(p) for p in (
            REVISION / "corrected/factorial.py", REVISION / "corrected/audit_repair/core.py")}
    checks = Checks()
    p, protocol_sha = common.protocol(validate=True)
    a = p["parameters"]
    inputs = {}
    for rel in ("protocol_v8_1.json", "factorial_execution_freeze.json", "analysis/E2_E3_effects.csv",
                "analysis/analysis_manifest.json", "results/E2_E3/preprocessing.csv"):
        inputs[rel] = sha(frozen / rel)
    checkpoints, all_rows, pre = {}, [], {}
    for path in sorted((frozen / "results/E2_E3/checkpoints").glob("*/*.json")):
        d = json.loads(path.read_text())
        key = (d["pool"], d["seed"])
        checks.require(d["status"] == "complete" and d["protocol_sha256"] == protocol_sha,
                       "checkpoint_complete_and_frozen", key)
        checks.require(key not in checkpoints, "checkpoint_unique_identity", key)
        checks.require(path.parent.name == key[0] and path.stem == f"seed_{key[1]:02d}",
                       "checkpoint_path_identity", key)
        for name, value in d["code_sha256"].items():
            checks.require(sha(frozen / name) == value, "checkpoint_code_hash_matches", (key, name))
            inputs[name] = value
        inputs[str(path.relative_to(frozen))] = sha(path)
        checkpoints[key] = d
        all_rows.extend(d["rows"])
        for generator in GENERATORS:
            pre[(*key, generator)] = d["preprocessing"][generator]["diagnostics"]
    checks.require(set(checkpoints) == {(pool, seed) for pool in p["pools"] for seed in range(a["seeds"])},
                   "complete140_checkpoint_grid")
    frame = pd.DataFrame(all_rows)
    expected_methods = {"pac_only"} | {f"{g}_{s}_{role}" for g in GENERATORS for s in SCORES for role in ("same_audit", "repair", "oracle")}
    expected_rows = {(pool, seed, rep, method) for pool in p["pools"] for seed in range(a["seeds"])
                     for rep in range(a["pac_reps"]) for method in expected_methods}
    observed_rows = set(frame[["pool", "seed", "rep", "method"]].itertuples(index=False, name=None))
    checks.require(observed_rows == expected_rows and len(frame) == len(expected_rows), "complete26600_row_grid")
    checks.require(frame.groupby(["pool", "seed", "rep"]).draws_sha256.nunique().eq(1).all(),
                   "all_factorial_repetitions_are_paired")
    original = pd.read_csv(frozen / "analysis/E2_E3_effects.csv")
    state_rows = []
    for (pool, seed, generator), diagnostic in sorted(pre.items()):
        state_rows.append({
            "state_id": state_key(pool, seed, generator), "pool": pool, "seed": seed,
            "generator": generator, "state_classification": classify_state(diagnostic),
            "accepted_groups": diagnostic["certified_groups"],
            "accepted_remainder_items": diagnostic["repaired_unaudited_items"],
            "actual_changed_predictions_vs_final_DS": diagnostic["changed_items_vs_same_audit"],
            "net_gain_vs_frozen_reference": diagnostic["net_gain_vs_ref"],
            "net_gain_vs_final_same_audit_DS": diagnostic["net_gain_vs_same_audit"],
        })
    state_classes = pd.DataFrame(state_rows)
    state_class_summary_rows = []
    for (pool, generator), part in state_classes.groupby(["pool", "generator"], sort=True):
        state_class_summary_rows.append({
            "pool": pool, "generator": generator, "preprocessing_seeds": len(part),
            **{f"n_{name}": int(part.state_classification.eq(name).sum()) for name in STATE_CLASSES},
            "accepted_group_actions": int(part.accepted_groups.sum()),
            "sum_net_gain_vs_frozen_reference_across_states": int(part.net_gain_vs_frozen_reference.sum()),
            "sum_net_gain_vs_final_same_audit_DS_across_states": int(part.net_gain_vs_final_same_audit_DS.sum()),
        })
    state_class_summary = pd.DataFrame(state_class_summary_rows)
    classification_rows, repetition_rows = [], []
    for pool in p["pools"]:
        for generator in GENERATORS:
            diagnostics = [pre[(pool, seed, generator)] for seed in range(a["seeds"])]
            accepted_seeds = sum(d["certified_groups"] > 0 for d in diagnostics)
            accepted_groups = sum(d["certified_groups"] for d in diagnostics)
            for score in SCORES:
                part = frame[(frame.pool == pool) & (frame.generator == generator) & (frame.score == score)]
                b = part[part.role == "same_audit"].set_index(["seed", "rep"]).sort_index()
                r = part[part.role == "repair"].set_index(["seed", "rep"]).sort_index()
                checks.require(b.index.equals(r.index) and len(r) == a["seeds"] * a["pac_reps"],
                               "cell_has_all200_paired_repetitions", (pool, generator, score))
                saved = (b.expert_total - r.expert_total).astype(int)
                checks.require(np.array_equal(saved.to_numpy(), -r.paired_cost_delta.to_numpy()),
                               "archived_paired_cost_delta_agrees", (pool, generator, score))
                n = int(r.n.iloc[0])
                category = CLASSIFICATIONS[0] if not accepted_seeds else CLASSIFICATIONS[2] if (saved != 0).any() else CLASSIFICATIONS[1]
                if not accepted_seeds:
                    checks.require((saved == 0).all() and r.changed.eq(0).all(), "inactive_cell_identity", (pool, generator, score))
                classification_rows.append({
                    "pool": pool, "generator": generator, "score": score, "n": n,
                    "classification": category, "preprocessing_seeds": a["seeds"],
                    "paired_repetitions": len(r), "accepted_preprocessing_seeds": accepted_seeds,
                    "accepted_group_actions": accepted_groups,
                    **{f"seeds_{name}": sum(classify_state(d) == name for d in diagnostics) for name in STATE_CLASSES},
                    "repetitions_with_accepted_action": accepted_seeds * a["pac_reps"],
                    "repetitions_with_cost_change": int((saved != 0).sum()),
                    "seeds_with_cost_change": int((saved != 0).groupby(level="seed").any().sum()),
                })
                if accepted_seeds:
                    for (seed, rep), value in saved.items():
                        base_row, repair_row = b.loc[(seed, rep)], r.loc[(seed, rep)]
                        diag = pre[(pool, seed, generator)]
                        repetition_rows.append({
                            "pair_id": paired_key(pool, seed, generator, score, rep),
                            "state_id": state_key(pool, seed, generator),
                            "pool": pool, "seed": int(seed), "rep": int(rep), "generator": generator,
                            "score": score, "n": n, "has_accepted_group": diag["certified_groups"] > 0,
                            "accepted_groups": diag["certified_groups"],
                            "baseline_expert_total": int(base_row.expert_total),
                            "repair_expert_total": int(repair_row.expert_total), "reviews_saved": int(value),
                            "practical_threshold_labels": (n + 99) // 100,
                            "baseline_threshold": thresholds(base_row.u_hat),
                            "repair_threshold": thresholds(repair_row.u_hat),
                        })
    classification = pd.DataFrame(classification_rows)
    repetitions = pd.DataFrame(repetition_rows)
    seed_rows = []
    for (pool, generator, score, seed), part in repetitions.groupby(["pool", "generator", "score", "seed"], sort=True):
        seed_rows.append(seed_summary(part, pre[(pool, seed, generator)]))
    seeds = pd.DataFrame(seed_rows)
    distribution_rows = []
    for (pool, generator, score), part in repetitions.groupby(["pool", "generator", "score"], sort=True):
        select = (seeds.pool == pool) & (seeds.generator == generator) & (seeds.score == score)
        seed_part = seeds[select]
        archive = original[(original.pool == pool) & (original.arm == f"{generator}_{score}_repair") &
                           (original.reference == f"{generator}_{score}_same_audit")]
        checks.require(len(archive) == 1, "one_authoritative_mean_interval", (pool, generator, score))
        distribution_rows.append(distribution(part, seed_part, "all_prespecified_preprocessing_seeds", archive.iloc[0], checks))
        distribution_rows.append(distribution(part[part.has_accepted_group], seed_part[seed_part.has_accepted_group],
                                              "accepted_preprocessing_seeds_only", archive.iloc[0], checks))
    distributions = pd.DataFrame(distribution_rows)
    active_states = [(pool, seed, generator) for (pool, seed, generator), d in pre.items() if d["certified_groups"] > 0]
    active_states.sort()
    cache_data = {}
    for pool in sorted({x[0] for x in active_states}):
        cache_path, cache_meta = frozen / "feature_cache" / f"{pool}.npz", frozen / "feature_cache" / f"{pool}.json"
        meta = json.loads(cache_meta.read_text())
        checks.require(meta["protocol_sha256"] == protocol_sha and meta["code_sha256"] == sha(frozen / "factorial.py") and
                       meta["cache_sha256"] == sha(cache_path), "active_feature_cache_hash_matches", pool)
        for path in (cache_path, cache_meta, frozen / "data" / f"{pool}.csv", frozen / "data" / f"{pool}.csv.meta.json"):
            inputs[str(path.relative_to(frozen))] = sha(path)
        with np.load(cache_path) as values:
            cache_data[pool] = (values["features"], values["neighbors"])
    group_rows, item_rows, mechanism_rows, recovery = [], [], [], []
    for ordinal, (pool, seed, generator) in enumerate(active_states, 1):
        d = checkpoints[(pool, seed)]
        state, gold, permutation, pac_seeds, recovered = reconstruct(d, generator, common, factorial, corrected, cache_data, checks)
        groups, items, mechanisms = inspect_state(d, generator, state, gold, permutation, pac_seeds, a["m"], checks, common)
        group_rows.extend(groups); item_rows.extend(items); mechanism_rows.extend(mechanisms); recovery.append(recovered)
        print(f"Active reconstruction {ordinal}/{len(active_states)} {pool} seed={seed} {generator}: saved hashes exact", flush=True)
    groups, items, mechanisms = pd.DataFrame(group_rows), pd.DataFrame(item_rows), pd.DataFrame(mechanism_rows)
    expected_active_pair_ids = set(repetitions[repetitions.has_accepted_group].pair_id)
    checks.require(set(mechanisms.pair_id) == expected_active_pair_ids and len(mechanisms) == len(expected_active_pair_ids),
                   "mechanism_rows_cover_every_accepted_state_repetition")
    checks.require(len(groups) == sum(d["certified_groups"] for d in pre.values()), "accepted_groups_complete")
    joined = mechanisms.merge(repetitions[["pair_id", "reviews_saved"]], on="pair_id", suffixes=("_mechanism", "_archive"), validate="one_to_one")
    checks.require(joined.reviews_saved_mechanism.eq(joined.reviews_saved_archive).all(), "mechanism_savings_match_every_archived_pair")
    mechanism_summary_rows = []
    for (pool, generator, score), part in mechanisms.groupby(["pool", "generator", "score"], sort=True):
        summary = {"pool": pool, "generator": generator, "score": score,
                   "accepted_preprocessing_seeds": int(part.seed.nunique()),
                   "accepted_state_paired_repetitions": len(part),
                   "all_prespecified_paired_repetitions": a["seeds"] * a["pac_reps"],
                   "threshold_changed_repetitions": int(part.threshold_changed.sum()),
                   "review_set_changed_repetitions": int(part.review_set_changed.sum()),
                   "purchase_union_changed_repetitions": int(part.unique_purchase_union_changed.sum()),
                   "final_labels_changed_repetitions": int((part.final_labels_differ_on_items > 0).sum()),
                   "mean_reviews_saved_conditional_on_acceptance": float(part.reviews_saved.mean())}
        for kind in MECHANISMS:
            summary[f"n_{kind}"] = int(part.mechanism_class.eq(kind).sum())
        for metric in ("changed", "corrected", "introduced_errors", "wrong_to_wrong"):
            for region in ("below", "at", "above"):
                column = f"{metric}_{region}_baseline_threshold"
                summary[f"mean_{column}_per_accepted_pair"] = float(part[column].mean())
                summary[f"sum_{column}_occurrences_across_repetitions"] = int(part[column].sum())
        mechanism_summary_rows.append(summary)
    mechanism_summary = pd.DataFrame(mechanism_summary_rows)
    for name, table in (
        ("active_cell_classification.csv", classification),
        ("active_state_classification.csv", state_classes),
        ("active_state_classification_summary.csv", state_class_summary),
        ("active_cost_distributions.csv", distributions),
        ("active_seed_summaries.csv", seeds),
        ("active_repetition_effects.csv", repetitions),
        ("active_accepted_groups.csv", groups),
        ("active_remainder_items.csv", items),
        ("active_mechanisms.csv", mechanisms),
        ("active_mechanism_summary.csv", mechanism_summary),
    ):
        table.to_csv(out / name, index=False)
    if not args.skip_plot:
        make_plot(repetitions, out)
    for rel, expected in inputs.items():
        checks.require(sha(frozen / rel) == expected, "frozen_analysis_input_unchanged", rel)
    for rel, expected in corrected_sources.items():
        checks.require(sha(REVISION / rel) == expected, "corrected_analysis_source_unchanged", rel)
    partition = {category: int(classification.classification.eq(category).sum()) for category in CLASSIFICATIONS}
    total = {
        "factorial_cells": len(classification), "active_cells": int((classification.accepted_preprocessing_seeds > 0).sum()),
        "accepted_preprocessing_states": len(active_states), "accepted_group_actions": len(groups),
        "active_cell_all_seed_paired_repetitions": len(repetitions),
        "accepted_state_paired_repetitions": len(mechanisms),
        "nonzero_cost_differences": int(repetitions.reviews_saved.ne(0).sum()),
        "saving_repetitions": int(repetitions.reviews_saved.gt(0).sum()),
        "cost_increase_repetitions": int(repetitions.reviews_saved.lt(0).sum()),
        "practical_saving_repetitions": int(repetitions.reviews_saved.ge(repetitions.practical_threshold_labels).sum()),
        "practical_cost_increase_repetitions": int(repetitions.reviews_saved.le(-repetitions.practical_threshold_labels).sum()),
        "maximum_labels_saved_in_one_repetition": int(repetitions.reviews_saved.max()),
        "maximum_extra_labels_in_one_repetition": int(-repetitions.reviews_saved.min()),
        "accepted_group_reference_gain_sum": int(groups.net_gain_vs_frozen_reference.sum()),
        "accepted_group_final_DS_gain_sum": int(groups.net_gain_vs_final_same_audit_DS.sum()),
        "accepted_groups_with_positive_reference_gain": int(groups.net_gain_vs_frozen_reference.gt(0).sum()),
        "accepted_groups_with_positive_final_DS_gain": int(groups.net_gain_vs_final_same_audit_DS.gt(0).sum()),
        "accepted_groups_with_zero_final_DS_gain": int(groups.net_gain_vs_final_same_audit_DS.eq(0).sum()),
        "accepted_groups_with_negative_final_DS_gain": int(groups.net_gain_vs_final_same_audit_DS.lt(0).sum()),
        "actual_changed_item_occurrences_across31_states": int(groups.actual_changed_predictions_vs_final_DS.sum()),
        "accepted_groups_minimum_net_gain_vs_final_DS": int(groups.net_gain_vs_final_same_audit_DS.min()),
    }
    report = {
        "schema_version": 1, "analysis_version": "v8.2 retrospective active-repair amendment",
        "status": "complete_with_exact_frozen_array_reconstruction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "protocol_v8_1_sha256": protocol_sha,
        "analysis_script_sha256": sha(Path(__file__)),
        "cell_partition": partition, "totals": total,
        "preprocessing_state_classification_counts": {name: int(state_classes.state_classification.eq(name).sum()) for name in STATE_CLASSES},
        "accepted_pair_mechanism_counts": {name: int(mechanisms.mechanism_class.eq(name).sum()) for name in MECHANISMS},
        "checks_passed": int(sum(checks.counts.values())), "check_categories": dict(checks.counts),
        "failures": [], "new_annotations": 0, "new_PAC_repetitions": 0,
        "corrected_preprocessing_comparison_run": corrected is not None,
        "frozen_input_sha256": inputs, "corrected_source_sha256": corrected_sources,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": importlib.import_module("scipy").__version__, "scikit_learn": importlib.import_module("sklearn").__version__,
                        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "limitations": [
            "This amendment was designed after the fixed v8.1 outcomes and peer review were seen; it is not a new prospective efficacy experiment.",
            "Costs and thresholds are archived outcomes. New calculations recover their set and item mechanisms; no new PAC samples were generated.",
            "200 paired repetitions per cell are nested within 20 preprocessing seeds. Frequencies are descriptive; no independent-repetition standard errors or binomial CIs are reported.",
            "Accepted-state conditioning is a descriptive diagnostic, not a randomized comparison or an estimate of a new deployment policy.",
            "Original pointwise seed-bootstrap intervals are retained only for all-seed mean effects. A degenerate interval does not establish equivalence or exclude unobserved realizations.",
            "The classification is exhaustive for observed active pairs. Structural locality is sufficient, not necessary; failing it does not by itself explain a cost change.",
            "Gains and changed-item totals across states count occurrences. The same original pool item can appear under multiple seeds; these are not unique corrected items.",
            "A nonnegative frozen-reference certificate does not certify improvement against final same-audit DS or a cost reduction. Those comparisons are evaluated separately here.",
        ],
    }
    write_json(out / "active_reconstruction_checks.json", {
        "status": report["status"], "analysis_script_sha256": report["analysis_script_sha256"],
        "corrected_source_sha256": corrected_sources, "states": recovery,
        "exactly_reconstructed_active_states": len(recovery), "frozen_arrays_checked_per_state": 7,
        "corrected_arrays_checked_per_state": 7 if corrected is not None else 0,
        "checks_passed": report["checks_passed"], "check_categories": report["check_categories"], "failures": [],
    })
    write_json(out / "active_analysis_report.json", report)
    nonzero_cells = distributions[(distributions.scope == "all_prespecified_preprocessing_seeds") & (distributions.changed_cost_repetitions > 0)]
    summary_lines = [
        "v8.2 active-repair analysis of frozen v8.1 factorial", "",
        f"Cell partition: {partition}",
        f"Accepted actions: {len(groups)} groups in {len(active_states)} preprocessing states; {len(mechanisms)} paired score/repetition comparisons.",
        "All 10 active cells remain represented, with all 20 seeds and all 10 PAC repetitions each.",
        "",
        nonzero_cells[["pool", "generator", "score", "mean_reviews_saved", "changed_cost_repetitions", "saving_repetitions",
                       "cost_increase_repetitions", "savings_at_least_1pct_pool_repetitions", "increases_at_least_1pct_pool_repetitions",
                       "repetition_savings_min", "repetition_savings_max"]].to_string(index=False),
        "", "Mechanisms among accepted-state paired repetitions:",
        json.dumps(report["accepted_pair_mechanism_counts"], indent=2),
        "", "Gain totals count occurrences across preprocessing states, not distinct items:",
        f"frozen-reference gain sum={total['accepted_group_reference_gain_sum']}; final same-audit DS gain sum={total['accepted_group_final_DS_gain_sum']}.",
        f"{total['accepted_groups_with_positive_final_DS_gain']} positive, {total['accepted_groups_with_zero_final_DS_gain']} zero, {total['accepted_groups_with_negative_final_DS_gain']} negative group gains against final same-audit DS.",
        "", "All recorded input arrays reconstructed with exact hashes; no tolerance substitution.",
        f"Corrected preprocessing comparison: {corrected is not None}; all saved outputs and stopping trajectories matched where run.",
        "No new annotations and no new PAC repetitions. No practical-equivalence or rare-event exclusion claim.",
    ]
    (out / "active_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(json.dumps(clean({"status": report["status"], "cell_partition": partition, "totals": total,
                           "mechanisms": report["accepted_pair_mechanism_counts"], "checks_passed": report["checks_passed"],
                           "elapsed_seconds": report["elapsed_seconds"]}), indent=2))


if __name__ == "__main__":
    main()
