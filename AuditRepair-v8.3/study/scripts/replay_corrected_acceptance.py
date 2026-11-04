#!/usr/bin/env python3
"""Replay frozen v8.1 repair decisions using the corrected v8.2 publication gate.

All factorial histories are replayed prefix by prefix. Every E1/E4 preprocessing
state is rerun with the corrected core because those checkpoints did not save
group histories. Their labels, scores and audit sets must match archived hashes.
No PAC evaluations or annotation calls are made. Frozen files are never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

sys.dont_write_bytecode = True
REVISION = Path(__file__).resolve().parents[1]
FROZEN = REVISION / "frozen_v8_1"
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(REVISION))

import numpy as np
import audit_repair.core as legacy_core
import audit_repair.pac as legacy_pac
import experiment_common as common
import run_e4 as e4
from corrected.audit_repair import core as corrected_core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def no_pac(*_args, **_kwargs):
    raise AssertionError("PAC calls are outside this acceptance replay")


def compare_group(log, stage, identity, rank):
    """Independent Fraction signs and unchanged capitals at every observed prefix."""
    N, t = int(log["size"]), int(log["audited"])
    history = log["audit_history"]
    require(len(history) == t, "Audit history length disagrees with log")
    old = legacy_core.WorLowerCS(N, alpha=float(log["alpha"]))
    new = corrected_core.WorLowerCS(N, alpha=float(log["alpha"]))
    changes, earlier_old, earlier_new = [], 0, 0
    exact_gain = 0
    for step, observation in enumerate(history, 1):
        gain = observation["gain"]
        require(type(gain) is int and gain in (-1, 0, 1), "Audit gains must be ternary integers")
        exact_gain += gain
        old.update((gain + 1) / 2)
        new.update((gain + 1) / 2)
        require(old.lower == new.lower and np.array_equal(old.log_cap, new.log_cap)
                and np.array_equal(old._rejected, new._rejected), "Betting path changed")
        require(old.t == new.t and old.s == new.s and old.ss == new.ss,
                "Betting sufficient statistics changed")
        require(new.audited_gain_sum == exact_gain, "Integer gain accumulator changed")
        # Fraction arithmetic is a separately expressed reference calculation.
        nominal = N * (2 * Fraction(new.lower_grid_index, 400) - 1) - exact_gain
        represented = N * (2 * Fraction.from_float(new.lower) - 1) - exact_gain
        reference = step < N and min(nominal, represented) > 0
        old_accept = step < N and old.remainder_gain_lb() > 0
        new_accept = new.remainder_is_positive()
        require(new_accept == reference, "Publication gate differs from rational reference")
        if old_accept != new_accept:
            changes.append({"step": step, "legacy": old_accept, "corrected": new_accept,
                            "nominal_margin": str(nominal), "represented_margin": str(represented)})
        if step < t:
            earlier_old += int(old_accept)
            earlier_new += int(new_accept)
    require(old.lower == float(log["lower_mean"]), "Stored/recovered lower bound changed")
    legacy_float = log.get("remainder_lb_exact", log.get("remainder_lb_float"))
    require(old.remainder_gain_lb() == legacy_float, "Stored/recovered float margin changed")
    require(earlier_old == 0 and earlier_new == 0, "A policy continued after a prior acceptance")
    require(new.remainder_is_positive() == bool(log["guarded"]),
            "Stopping decision disagrees with publication gate")
    require(not changes, "Legacy/corrected acceptance differ: " + str(changes))
    record = new.remainder_certificate_record()
    return {
        "experiment": stage, **identity, "group_rank": rank,
        "population_size": N, "audited": t, "alpha_group": log["alpha"],
        "stop_reason": log["stop_reason"], "accepted": bool(log["guarded"]),
        "prefixes_checked": t, "predicate_changes": len(changes),
        "betting_path_identical": True,
        "audit_history_sha256": hashlib.sha256(json.dumps(history, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest(),
        **record,
    }


def recover_historical(task):
    """Run corrected preprocessing, retain original implementation for reference checks."""
    stage, relative = task
    started = time.monotonic()
    checkpoint = FROZEN / relative
    d = json.loads(checkpoint.read_text())
    protocol, _ = common.protocol(validate=False)
    params = protocol["parameters"]
    hash_checks = 0

    def match(array, expected, name):
        nonlocal hash_checks
        require(common.array_hash(array) == expected, f"{relative}: {name} does not match archive")
        hash_checks += 1

    with patch.object(legacy_pac, "pac_labels", no_pac), patch.object(e4, "pac_labels", no_pac):
        if stage == "E1":
            pool, seed = d["pool"], d["seed"]
            identity = {"pool": pool, "seed": seed, "generator": "historical_G0"}
            _, labels0, gold0, K = common.load_pool(pool)
            data_stream, pre_stream, _ = common.streams(params["root_seed"], seed, params["pac_reps"])
            perm = np.random.default_rng(data_stream).permutation(len(gold0))
            match(perm, d["permutation_sha256"], "permutation")
            labels, gold = labels0[perm], gold0[perm]
            result = corrected_core.run_group_repair(
                labels, K, gold, params["B"], alpha=params["alpha_repair"], seed=pre_stream,
                ref_share=params["warmup"] / params["B"], min_size=params["min_group_size"],
                stop_after_fails=params["stop_after_fails"],
            )
            guarded, info = result["guarded"]
            audits = info["audited_labels"]
            base = info["post"].argmax(1)
            scores = 1 - info["post"].max(1)
            guarded_scores = scores.copy()
            for i, confidence in info["repair_conf"].items():
                guarded_scores[i] = 1 - confidence
            for name, pred, uncertainty in (("same_audit_ds", base, scores),
                                            ("legacy_guarded", guarded, guarded_scores)):
                saved = d["frozen_arrays"][name]
                match(pred, saved["labels_sha256"], name + " labels")
                match(uncertainty, saved["scores_sha256"], name + " scores")
                match(np.array(sorted(audits), dtype=np.int64), saved["audited_ids_sha256"], name + " audits")
            require([int(perm[i]) for i in sorted(audits)] == d["audits_original_row_ids"],
                    "Original audit row IDs changed")
            require(len(audits) == d["preprocessing"]["audits"], "Audit count changed")
            require(info["groups_repaired"] == d["preprocessing"]["groups_certified"],
                    "Accepted-group count changed")
        else:
            identity = {"pool": d["case"], "seed": d["seed"], "generator": "historical_G0"}
            data_stream, pre_stream, score_stream, _, entropy = e4.e4_streams(
                params["root_seed"], d["shared_sources"], d["slice_frac"], d["seed"],
                protocol["E4"]["pac_reps"],
            )
            require(entropy == d["seed_entropy"], "E4 seed entropy changed")
            labels, gold = e4.simulate(
                n=e4.N_ITEMS, K=e4.N_CLASSES, seed=data_stream,
                confusions=[e4.Confusion(0, 1, d["slice_frac"], tuple(range(d["shared_sources"])),
                                         flip_prob=e4.FLIP_PROBABILITY)],
            )
            frozen = d["frozen_arrays"]
            match(labels, frozen["source_labels_sha256"], "source panel")
            match(gold, frozen["gold_sha256"], "gold")
            captured = []

            def capture(*args, **kwargs):
                result = corrected_core.run_group_repair(*args, **kwargs)
                captured.append(result["guarded"][1])
                return result

            with patch.object(e4, "run_group_repair", capture):
                state = e4.legacy_preprocessing(labels, gold, e4.N_CLASSES, params, pre_stream)
            require(len(captured) == 1, "E4 preprocessing must use one guarded trajectory")
            info = captured[0]
            require(state.audits == {int(i): int(y) for i, y in d["audited_labels"].items()},
                    "E4 audit labels changed")
            match(np.array(sorted(state.audits), dtype=np.int64), frozen["audit_ids_sha256"], "audits")
            match(state.reference, frozen["reference_labels_sha256"], "frozen reference")
            match(state.scores, frozen["score_coupling"]["original_scores_sha256"], "original scores")
            for name, pred in (("same_audit_ds", state.baseline), ("guarded_fixed_scores", state.guarded),
                               ("O1_shortlist", state.oracle_shortlist)):
                match(pred, frozen["label_vectors"][name], name + " labels")
            coupling = e4.make_score_coupling(state.scores, state.baseline, gold,
                                               e4.disagreement_fraction(labels), state.audits, score_stream)
            require(coupling.manifest() == frozen["score_coupling"], "E4 score-coupling inputs changed")
            for eta in protocol["E4"]["eta"]:
                match(coupling.scores(eta), frozen["scores_by_eta"][format(eta, ".12g")], "coupled scores")
            require(info["groups_repaired"] == d["preprocessing"]["groups_certified"],
                    "E4 accepted-group count changed")
        groups = [compare_group(log, stage, identity, rank) for rank, log in enumerate(info["log"])]
    state_record = {
        "experiment": stage, **identity, "history_source": "corrected_complete_preprocessing",
        "checkpoint": relative, "tested_groups": len(groups),
        "accepted_groups": sum(g["accepted"] for g in groups),
        "prefixes_checked": sum(g["prefixes_checked"] for g in groups),
        "predicate_changes": 0, "archive_output_hash_checks": hash_checks,
        "archive_outputs_identical": True, "elapsed_seconds": time.monotonic() - started,
    }
    return state_record, groups


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=REVISION / "results")
    args = parser.parse_args()
    require(args.workers >= 1, "workers must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    created = datetime.now(timezone.utc).isoformat()
    protocol, old_protocol_sha = common.protocol(validate=True)
    revision_protocol_sha = sha(REVISION / "protocol_v8_2.json")
    frozen_names = ("audit_repair/core.py", "audit_repair/pac.py", "audit_repair/simulate.py",
                    "experiment_common.py", "run_e1.py", "factorial.py", "run_factorial.py", "run_e4.py")
    old_code = {name: sha(FROZEN / name) for name in frozen_names}
    new_code = {name: sha(REVISION / name) for name in ("corrected/audit_repair/core.py", "corrected/factorial.py")}
    require(sha(REVISION / "corrected/audit_repair/pac.py") == old_code["audit_repair/pac.py"],
            "Final PAC implementation must remain unchanged")
    checkpoints = {stage: sorted((FROZEN / "results" / stage / "checkpoints").glob("*/*.json"))
                   for stage in ("E1", "E2_E3", "E4")}
    require([len(checkpoints[s]) for s in ("E1", "E2_E3", "E4")] == [140, 140, 40],
            "Replay requires the complete frozen checkpoint inventory")
    checkpoint_hashes = {str(p.relative_to(FROZEN)): sha(p) for paths in checkpoints.values() for p in paths}
    expected_acceptances = Counter()
    for stage, paths in checkpoints.items():
        for path in paths:
            d = json.loads(path.read_text())
            require(d["status"] == "complete" and d["protocol_sha256"] == old_protocol_sha,
                    "Checkpoint completeness/protocol mismatch")
            require(all(sha(FROZEN / key) == value for key, value in d["code_sha256"].items()),
                    "Archived implementation hash mismatch")
            if stage == "E2_E3":
                expected_acceptances[stage] += sum(s["diagnostics"]["certified_groups"]
                                                   for s in d["preprocessing"].values())
            else:
                expected_acceptances[stage] += d["preprocessing"]["groups_certified"]
    states, groups, failures = [], [], []
    # Factorial group histories include every tested path, including inactive cells.
    for path in checkpoints["E2_E3"]:
        d = json.loads(path.read_text())
        for generator, state in d["preprocessing"].items():
            identity = {"pool": d["pool"], "seed": d["seed"], "generator": generator}
            try:
                rows = [compare_group(log, "E2_E3", identity, rank)
                        for rank, log in enumerate(state["group_log"])]
                require(sum(g["accepted"] for g in rows) == state["diagnostics"]["certified_groups"],
                        "Factorial accepted-group count changed")
                groups.extend(rows)
                states.append({
                    "experiment": "E2_E3", **identity, "history_source": "complete_archived_group_histories",
                    "checkpoint": str(path.relative_to(FROZEN)), "tested_groups": len(rows),
                    "accepted_groups": sum(g["accepted"] for g in rows),
                    "prefixes_checked": sum(g["prefixes_checked"] for g in rows),
                    "predicate_changes": 0, "archive_output_hash_checks": 0,
                    "archive_outputs_identical": "implied_by_identical_all_prefix_decisions",
                    "elapsed_seconds": None,
                })
            except Exception as error:
                failures.append({"experiment": "E2_E3", **identity, "error": str(error)})
    print(json.dumps({"stage": "E2_E3", "states": len(states), "failures": len(failures)}), flush=True)
    historical_tasks = [(stage, str(path.relative_to(FROZEN))) for stage in ("E1", "E4")
                        for path in checkpoints[stage]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(recover_historical, task): task for task in historical_tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                state, rows = future.result()
                states.append(state)
                groups.extend(rows)
            except Exception as error:
                failures.append({"experiment": task[0], "checkpoint": task[1], "error": str(error),
                                 "traceback": traceback.format_exc()})
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(json.dumps({"historical_states_completed": completed, "of": len(futures),
                                  "failures": len(failures), "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
    states.sort(key=lambda r: (r["experiment"], r["pool"], r["seed"], r["generator"]))
    groups.sort(key=lambda r: (r["experiment"], r["pool"], r["seed"], r["generator"], r["group_rank"]))
    for name, expected in old_code.items():
        require(sha(FROZEN / name) == expected, "Frozen source changed during replay")
    for name, expected in new_code.items():
        require(sha(REVISION / name) == expected, "Corrected source changed during replay")
    require(sha(REVISION / "protocol_v8_2.json") == revision_protocol_sha, "Amendment changed during replay")
    require(all(sha(FROZEN / name) == expected for name, expected in checkpoint_hashes.items()),
            "Frozen checkpoint changed during replay")
    stage_counts = {}
    for stage in checkpoints:
        ss = [s for s in states if s["experiment"] == stage]
        gg = [g for g in groups if g["experiment"] == stage]
        stage_counts[stage] = {
            "checkpoints": len(checkpoints[stage]), "preprocessing_states": len(ss),
            "tested_groups": len(gg), "audit_prefixes": sum(g["prefixes_checked"] for g in gg),
            "accepted_groups_archived": expected_acceptances[stage],
            "accepted_groups_corrected": sum(g["accepted"] for g in gg),
            "changed_decisions": sum(g["predicate_changes"] for g in gg),
            "archive_output_hash_checks": sum(s["archive_output_hash_checks"] for s in ss),
        }
        if stage_counts[stage]["accepted_groups_corrected"] != expected_acceptances[stage]:
            failures.append({"experiment": stage, "error": "accepted-group total mismatch"})
    nominal_minimum = {}
    for stage in checkpoints:
        accepted = [g for g in groups if g["experiment"] == stage and g["accepted"]]
        if accepted:
            margin = min(Fraction(g["remainder_nominal_numerator"], g["remainder_nominal_denominator"])
                         for g in accepted)
            nominal_minimum[stage] = {"fraction": str(margin), "decimal": float(margin)}
    summary = {
        "schema_version": 1, "artifact_kind": "v8_2_corrected_acceptance_replay",
        "status": "failed" if failures else "passed_no_archived_decision_changed",
        "started_utc": created, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "v8_1_protocol_sha256": old_protocol_sha, "v8_2_protocol_sha256": revision_protocol_sha,
        "script_sha256": sha(__file__), "frozen_source_sha256": old_code,
        "corrected_source_sha256": new_code, "checkpoint_sha256": checkpoint_hashes,
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "workers": args.workers, "stages": stage_counts,
        "total_preprocessing_states": len(states), "total_tested_groups": len(groups),
        "total_audit_prefixes": sum(g["prefixes_checked"] for g in groups),
        "accepted_groups_archived": sum(expected_acceptances.values()),
        "accepted_groups_corrected": sum(g["accepted"] for g in groups),
        "changed_acceptance_predicates": sum(g["predicate_changes"] for g in groups),
        "minimum_positive_nominal_remainder_margin_by_stage": nominal_minimum,
        "frozen_sources_and_checkpoints_unchanged": True,
        "final_pac_code_unchanged": True, "pac_calls": 0,
        "failures": failures,
        "decision_rule": "0<t<N and N*(2*j-q)-q*sum(d)>0 and N*(2*a-b)-b*sum(d)>0; q=400; a/b is exact float(grid[j])",
        "method": [
            "Replay every recorded factorial group prefix through both legacy and corrected CS; compare all capital arrays, sticky rejection masks, sufficient statistics and decisions exactly.",
            "Independently calculate nominal and represented remainder margins with Fraction and check the corrected predicate.",
            "Execute every E1/E4 preprocessing state using the corrected core; match archived audit IDs, labels and score hashes; replay every recovered group prefix through both CS implementations.",
            "Recreate every E4 score intervention and match its archived vector hash.",
            "The active-factorial mechanism analysis separately reconstructs all accepted states with frozen and corrected factorial code; see active_reconstruction_checks.json.",
        ],
        "interpretation": "The corrected strict predicate changes no recorded repair decision. Unchanged preprocessing inputs, PAC implementation and calibration streams imply the archived downstream outcomes are retained; this replay does not rerun 42,200 PAC evaluation rows.",
        "limitations": [
            "A targeted final-inequality correction, not a proof that all floating-point betting arithmetic is outward rounded.",
            "E1/E4 histories are reconstructed from frozen inputs and seeds, rather than originally stored; exact output hashes identify the recovered paths.",
            "Algorithmic replay of existing panels/seeds is reproducibility evidence, not an additional sample or independent annotation panel.",
        ],
    }
    write_csv(args.out_dir / "acceptance_replay_groups.csv", groups)
    write_csv(args.out_dir / "acceptance_replay_states.csv", states)
    common.atomic_json(args.out_dir / "acceptance_replay_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "total_preprocessing_states", "total_tested_groups",
                    "total_audit_prefixes", "accepted_groups_corrected", "changed_acceptance_predicates",
                    "elapsed_seconds")}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
