#!/usr/bin/env python3
"""Read-only final-margin audit of frozen E1/E2_E3/E4 acceptances.

E2/E3 histories are read from checkpoints. E1 and E4 did not preserve their
histories, so ONLY preprocessing states reporting an acceptance are recovered
with frozen seeds/code, and their outputs must match saved hashes. No PAC call,
score-coupling sweep, or full study runner is executed; no frozen file is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction_json(x):
    return {"numerator": x.numerator, "denominator": x.denominator,
            "fraction": str(x), "decimal": float(x)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path, default=HERE/"numerical_acceptance_audit.json")
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    import audit_repair.core as core
    import audit_repair.pac as pac
    import experiment_common as common
    import run_e1 as e1
    import run_e4 as e4

    started = time.monotonic()
    checks = Counter()
    failures = []

    def require(condition, name, detail=None):
        checks[name] += 1
        if not condition:
            raise AssertionError({"check": name, "detail": detail})

    # Guard against an accidental calibration call during targeted recovery.
    def no_pac(*_args, **_kwargs):
        raise AssertionError("A PAC call is outside this targeted numerical audit")

    protocol, protocol_sha = common.protocol(validate=True)
    params = protocol["parameters"]
    stages = ("E1", "E2_E3", "E4")
    paths = {stage: sorted((root/"results"/stage/"checkpoints").glob("*/*.json")) for stage in stages}
    checkpoint_hashes = {str(p.relative_to(root)): sha(p) for ps in paths.values() for p in ps}
    source_names = ("audit_repair/core.py", "audit_repair/pac.py", "audit_repair/simulate.py",
                    "experiment_common.py", "run_e1.py", "factorial.py", "run_factorial.py", "run_e4.py")
    source_hashes = {name: sha(root/name) for name in source_names}
    q = 400
    grid = np.linspace(0., 1., q+1)
    require(np.array_equal(core.WorLowerCS(1).m, grid), "actual_production_grid_matches_401_points")
    records = []
    state_records = []
    inventory = {}

    def audit_group(log, stage, identity, rank, recovery):
        history = log["audit_history"]
        gains = [int(h["gain"]) for h in history]
        require(all(h["gain"] == int(h["gain"]) and int(h["gain"]) in (-1, 0, 1)
                    for h in history), "gain_history_is_integer_ternary", identity)
        N, t = int(log["size"]), int(log["audited"])
        require(len(history) == t, "audit_history_length_matches", identity)
        require(0 < t < N, "acceptance_has_unaudited_remainder", identity)
        lower = float(log["lower_mean"])
        matches = np.flatnonzero(grid == lower)
        require(len(matches) == 1, "stored_lower_exactly_matches_one_grid_float", identity)
        j = int(matches[0])
        gain_sum = sum(gains)
        nominal = Fraction(N*(2*j-q)-q*gain_sum, q)
        represented = N*(2*Fraction.from_float(lower)-1)-gain_sum
        floating = N*(2*lower-1)-gain_sum
        require(floating == log["remainder_lb_exact"], "stored_float_margin_reconstructed", identity)
        require(floating > 0, "stored_accepted_margin_positive", identity)
        # Verify the saved stopping-state lower index using only its recorded
        # finite audit path. This is not a group search or a study rerun.
        test = core.WorLowerCS(N=N, alpha=float(log["alpha"]))
        earlier_float_acceptances = 0
        for step, z in enumerate(gains, 1):
            test.update((z+1)/2)
            if step < t and step < N and test.remainder_gain_lb() > 0:
                earlier_float_acceptances += 1
        require(test.lower == lower, "recorded_path_recreates_lower_index", identity)
        require(test.remainder_gain_lb() == floating, "recorded_path_recreates_float_margin", identity)
        require(earlier_float_acceptances == 0, "first_positive_acceptance_stopping_time", identity)
        records.append({"experiment": stage, **identity, "group_rank": rank,
                        "history_source": recovery, "population_size": N, "audit_count": t,
                        "audited_gain_sum": gain_sum, "audit_gains": gains,
                        "lower_grid_index": j, "grid_denominator": q,
                        "stored_lower_mean": lower, "stored_lower_hex": lower.hex(),
                        "stored_float_margin": floating,
                        "exact_nominal_grid_margin": fraction_json(nominal),
                        "exact_represented_grid_margin": fraction_json(represented),
                        "floating_minus_nominal_margin": fraction_json(Fraction.from_float(floating)-nominal),
                        "float_positive_but_nominal_nonpositive": nominal <= 0,
                        "float_positive_but_represented_nonpositive": represented <= 0})

    def recover_e1(d):
        pool, seed = d["pool"], d["seed"]
        _, labels0, gold0, n_classes = common.load_pool(pool)
        data_stream, pre_stream, _ = common.streams(params["root_seed"], seed, params["pac_reps"])
        perm = np.random.default_rng(data_stream).permutation(len(gold0))
        require(common.array_hash(perm) == d["permutation_sha256"], "E1_permutation_hash")
        labels, gold = labels0[perm], gold0[perm]
        result = core.run_group_repair(labels, n_classes, gold, params["B"],
                                      alpha=params["alpha_repair"], seed=pre_stream,
                                      ref_share=params["warmup"]/params["B"],
                                      min_size=params["min_group_size"],
                                      stop_after_fails=params["stop_after_fails"])
        guarded, info = result["guarded"]
        audits = info["audited_labels"]
        base = info["post"].argmax(1)
        scores = 1-info["post"].max(1)
        guarded_scores = scores.copy()
        for i, confidence in info["repair_conf"].items():
            guarded_scores[i] = 1-confidence
        for name, pred, uncertainty in (("same_audit_ds", base, scores),
                                        ("legacy_guarded", guarded, guarded_scores)):
            saved = d["frozen_arrays"][name]
            require(common.array_hash(pred) == saved["labels_sha256"], "E1_saved_prediction_hash", name)
            require(common.array_hash(uncertainty) == saved["scores_sha256"], "E1_saved_score_hash", name)
            require(common.array_hash(np.array(sorted(audits), dtype=np.int64)) == saved["audited_ids_sha256"],
                    "E1_saved_audit_ids_hash", name)
        original_ids = [int(perm[i]) for i in sorted(audits)]
        require(original_ids == d["audits_original_row_ids"], "E1_saved_original_audit_ids")
        require(len(audits) == d["preprocessing"]["audits"], "E1_saved_audit_count")
        require(info["groups_repaired"] == d["preprocessing"]["groups_certified"], "E1_saved_acceptance_count")
        return info["log"]

    def recover_e4(d):
        data_stream, pre_stream, _, _, entropy = e4.e4_streams(
            params["root_seed"], d["shared_sources"], d["slice_frac"], d["seed"], protocol["E4"]["pac_reps"])
        require(entropy == d["seed_entropy"], "E4_seed_entropy_matches")
        labels, gold = e4.simulate(n=e4.N_ITEMS, K=e4.N_CLASSES, seed=data_stream,
                                  confusions=[e4.Confusion(0, 1, d["slice_frac"],
                                              tuple(range(d["shared_sources"])), flip_prob=e4.FLIP_PROBABILITY)])
        frozen = d["frozen_arrays"]
        require(common.array_hash(labels) == frozen["source_labels_sha256"], "E4_saved_source_panel_hash")
        require(common.array_hash(gold) == frozen["gold_sha256"], "E4_saved_gold_hash")
        captured = []

        def capture(*args, **kwargs):
            result = core.run_group_repair(*args, **kwargs)
            captured.append(result["guarded"][1])
            return result

        with patch.object(e4, "run_group_repair", capture):
            state = e4.legacy_preprocessing(labels, gold, e4.N_CLASSES, params, pre_stream)
        require(len(captured) == 1, "E4_single_preprocessing_recovery")
        info = captured[0]
        require(state.audits == {int(i): int(y) for i, y in d["audited_labels"].items()}, "E4_saved_audit_labels")
        require(common.array_hash(np.array(sorted(state.audits), dtype=np.int64)) == frozen["audit_ids_sha256"],
                "E4_saved_audit_ids_hash")
        require(common.array_hash(state.reference) == frozen["reference_labels_sha256"], "E4_saved_reference_hash")
        require(common.array_hash(state.scores) == frozen["score_coupling"]["original_scores_sha256"], "E4_saved_original_scores_hash")
        for name, pred in (("same_audit_ds", state.baseline), ("guarded_fixed_scores", state.guarded),
                           ("O1_shortlist", state.oracle_shortlist)):
            require(common.array_hash(pred) == frozen["label_vectors"][name], "E4_saved_prediction_hash", name)
        require(info["groups_repaired"] == d["preprocessing"]["groups_certified"], "E4_saved_acceptance_count")
        return info["log"]

    with patch.object(pac, "pac_labels", no_pac), patch.object(e1, "pac_labels", no_pac), patch.object(e4, "pac_labels", no_pac):
        for stage in stages:
            counts = Counter(checkpoints=len(paths[stage]))
            print(f"START {stage}: {len(paths[stage])} checkpoints", flush=True)
            for path in paths[stage]:
                rel = str(path.relative_to(root))
                d = json.loads(path.read_text())
                try:
                    require(d["status"] == "complete", "checkpoint_complete", rel)
                    require(d["protocol_sha256"] == protocol_sha, "checkpoint_protocol_matches", rel)
                    require(all(source_hashes[name] == value for name, value in d["code_sha256"].items()),
                            "checkpoint_code_matches", rel)
                    states = d["preprocessing"].items() if stage == "E2_E3" else [("legacy_guarded", d["preprocessing"])]
                    for generator, state in states:
                        counts["preprocessing_states"] += 1
                        n_accepted = int(state["diagnostics"]["certified_groups"] if stage == "E2_E3" else state["groups_certified"])
                        counts["reported_accepted_groups"] += n_accepted
                        ident = {"case_or_pool": d.get("pool", d.get("case")), "seed": d["seed"],
                                 "generator": generator, "checkpoint": rel,
                                 "checkpoint_sha256": checkpoint_hashes[rel]}
                        if stage == "E2_E3":
                            logs = state["group_log"]
                            counts["stored_group_records"] += len(logs)
                            recovery = "stored_checkpoint_audit_history"
                        elif n_accepted:
                            logs = recover_e1(d) if stage == "E1" else recover_e4(d)
                            counts["recovered_preprocessing_states"] += 1
                            recovery = "accepted_state_only_preprocessing_recovery_matched_saved_hashes"
                            print(f"  recovered {stage} {ident['case_or_pool']} seed={ident['seed']}", flush=True)
                        else:
                            counts["zero_acceptance_states_no_replay"] += 1
                            continue
                        selected = [(rank, group) for rank, group in enumerate(logs) if group["guarded"]]
                        require(len(selected) == n_accepted, "accepted_log_count_matches_checkpoint", ident)
                        if n_accepted:
                            state_records.append({"experiment": stage, **ident, "reported_acceptances": n_accepted,
                                                  "history_source": recovery, "saved_identity_checks_passed": True})
                        for rank, group in selected:
                            audit_group(group, stage, ident, rank, recovery)
                            counts["audited_accepted_groups"] += 1
                except Exception as error:
                    failures.append({"checkpoint": rel, "exception": type(error).__name__, "message": str(error)})
            inventory[stage] = dict(counts)

    require(all(sha(root/name) == value for name, value in source_hashes.items()), "frozen_sources_unchanged")
    require(all(sha(root/name) == value for name, value in checkpoint_hashes.items()), "frozen_checkpoints_unchanged")
    require(all(paths[stage] == sorted((root/"results"/stage/"checkpoints").glob("*/*.json")) for stage in stages),
            "checkpoint_inventory_unchanged")
    affected = [r for r in records if r["float_positive_but_nominal_nonpositive"]]
    represented_affected = [r for r in records if r["float_positive_but_represented_nonpositive"]]

    def exact_value(row):
        x = row["exact_nominal_grid_margin"]
        return Fraction(x["numerator"], x["denominator"])

    minimum_by_stage = {}
    for stage in stages:
        candidates = [r for r in records if r["experiment"] == stage and exact_value(r) > 0]
        minimum_by_stage[stage] = min(candidates, key=exact_value) if candidates else None
    positives = [r for r in records if exact_value(r) > 0]
    expected = sum(s.get("reported_accepted_groups", 0) for s in inventory.values())
    if len(records) != expected:
        failures.append({"check": "all_reported_acceptances_audited", "reported": expected, "audited": len(records)})
    report = {"schema_version": 1, "artifact_kind": "frozen_study_final_acceptance_numerical_audit",
              "status": "incomplete_or_mismatch" if failures else "affected_acceptance_found" if affected or represented_affected else "no_affected_acceptances_under_audited_arithmetic",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic()-started, 3),
              "protocol_sha256": protocol_sha, "script_sha256": sha(__file__), "source_sha256": source_hashes,
              "checkpoint_sha256": checkpoint_hashes, "inventory": inventory,
              "reported_accepted_groups": expected, "audited_accepted_groups": len(records),
              "float_positive_nominal_nonpositive_count": len(affected),
              "float_positive_nominal_zero_count": sum(exact_value(r) == 0 for r in affected),
              "float_positive_nominal_negative_count": sum(exact_value(r) < 0 for r in affected),
              "float_positive_represented_nonpositive_count": len(represented_affected),
              "minimum_positive_exact_nominal_margin": min(positives, key=exact_value) if positives else None,
              "minimum_positive_exact_nominal_margin_by_stage": minimum_by_stage,
              "checks": dict(checks), "assertion_count": sum(checks.values()), "failures": failures,
              "accepted_states": state_records, "accepted_group_records": records, "affected_records": affected,
              "method": {"grid_points": 401, "nominal_grid_denominator": q,
                         "exact_margin": "[N*(2*j-q) - q*sum(z)]/q, z in {-1,0,1}",
                         "represented_grid_margin": "N*(2*Fraction.from_float(stored_lower)-1) - sum(z)",
                         "actual_lower_mapping": "Require stored lower to equal exactly one np.linspace(0,1,401) float; use that point's integer index j.",
                         "recovery_scope": "E1/E4 omitted audit histories. Only states with reported acceptance are replayed through unchanged preprocessing, with saved audit, prediction and score hashes checked. E4 also checks source panel, gold and reference hashes. No PAC call is allowed."},
              "recommended_future_implementation": {
                  "nominal_integer_gate": "Store the selected lower grid index j and sum(z) as Python integers. With q=grid_points-1, compute D=N*(2*j-q)-q*sum(z). Accept only when t<N and D>0.",
                  "conservative_dual_gate": "To respect both nominal and actually represented grid values, obtain a,b=float(grid[j]).as_integer_ratio(). Also require D_rep=N*(2*a-b)-b*sum(z)>0. Requiring both is equivalent to exact remainder positivity at min(j/q, represented_grid_value).",
                  "legacy_subset_option": "A transition that must never add an acceptance to the frozen floating rule can additionally retain its original float-margin >0 condition; the integer guards then only remove questionable acceptances.",
                  "storage_and_types": "Carry j from inversion, not by rounding the bound later. Update an integer gain accumulator from the observed {-1,0,1} gain. Use Python integers, or establish overflow bounds for fixed-width integers. Do not quantize arbitrary [0,1] observations to this lattice.",
                  "outward_rounding_alternative": "Use verified interval or directed-rounding arithmetic for every operation in the final remainder expression and require the interval's lower endpoint >0. A single nextafter on the final rounded answer is insufficient to bound all earlier rounding errors.",
                  "scope": "These guards fix the final remainder arithmetic and its nominal-grid boundary. They do not certify floating-point stake, conditional-null mean, log-capital accumulation, impossible-null tolerance, or threshold rejection. No general floating-point theorem is asserted."},
              "limitations": ["This audit addresses only actually accepted groups in the completed frozen checkpoints; it does not replace the statistical guarantee or prove accuracy of every numerical operation.",
                              "E1/E4 accepted histories are recovered, not directly stored; saved-hash agreement and deterministic frozen code identify the recovered trajectories.",
                              "The exact nominal margin is conditional on the observed selected grid index. Rejection decisions are checked for exact replay, not independently proven to be rounding-safe.",
                              "Checkpoints, sources, main verification code and main verification report are not modified; only this separate audit artifact is created."]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "accepted_groups": len(records), "affected": len(affected),
                      "recovery_failures": len(failures), "elapsed_seconds": report["elapsed_seconds"], "output": str(args.output)}), flush=True)
    return 1 if failures else 2 if affected or represented_affected else 0


if __name__ == "__main__":
    raise SystemExit(main())
