"""Validate the packaged evidence and run the two targeted regression files.

This does not rerun empirical experiments. It verifies source/output identity
and cross-table accounting before trusting the supplied results.
"""
from pathlib import Path
import hashlib
import io
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    import pandas as pd
    checks = 0
    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    original = read("provenance/frozen_v8_1_inventory.json")
    for name, digest in original.items():
        require(sha(ROOT/"frozen_v8_1"/name) == digest, "Frozen file changed: "+name)
    require(sha(ROOT/"protocol_v8_2.json") == (ROOT/"protocol_v8_2.sha256").read_text().split()[0], "Amendment hash mismatch")
    a = read("results/acceptance_replay_summary.json")
    require(a["status"] == "passed_no_archived_decision_changed", "Acceptance replay not passed")
    require(a["total_preprocessing_states"] == 600 and a["total_audit_prefixes"] == 52824, "Acceptance replay incomplete")
    require(a["accepted_groups_archived"] == a["accepted_groups_corrected"] == 55 and a["changed_acceptance_predicates"] == 0, "Acceptance results changed")
    require(a["script_sha256"] == sha(ROOT/"scripts/replay_corrected_acceptance.py"), "Replay source mismatch")
    for name, digest in a["corrected_source_sha256"].items():
        require(sha(ROOT/name) == digest, "Corrected source mismatch: "+name)
    require(sha(ROOT/"corrected/audit_repair/pac.py") == sha(ROOT/"frozen_v8_1/audit_repair/pac.py"), "Final PAC implementation changed")

    b = read("results/active_analysis_report.json")
    require(not b["failures"] and b["corrected_preprocessing_comparison_run"], "Active reconstruction not passed")
    require(b["analysis_script_sha256"] == sha(ROOT/"scripts/analyze_active_repairs.py"), "Active analysis source mismatch")
    require(sorted(b["cell_partition"].values()) == [3,7,32], "Unexpected cell partition")
    require(b["totals"]["accepted_group_actions"] == 31 and b["totals"]["accepted_state_paired_repetitions"] == 620, "Active scope incomplete")
    require(sum(b["accepted_pair_mechanism_counts"].values()) == 620, "Mechanism partition incomplete")
    for name, digest in b["frozen_input_sha256"].items():
        require(sha(ROOT/"frozen_v8_1"/name) == digest, "Active source mismatch: "+name)
    for name, digest in b["corrected_source_sha256"].items():
        require(sha(ROOT/name) == digest, "Active corrected source mismatch: "+name)
    groups = pd.read_csv(ROOT/"results/active_accepted_groups.csv")
    require(len(groups)==31 and (groups.net_gain_vs_final_same_audit_DS>0).all(), "Accepted-group gain summary mismatch")
    mechanisms = pd.read_csv(ROOT/"results/active_mechanisms.csv")
    require(len(mechanisms)==620, "Mechanism table incomplete")
    distribution = pd.read_csv(ROOT/"results/active_cost_distributions.csv")
    all_seeds = distribution[distribution.scope == "all_prespecified_preprocessing_seeds"]
    require(len(all_seeds)==10 and int(all_seeds.paired_repetitions.sum())==2000, "Active-cell denominators mismatch")
    require(int(all_seeds.saving_repetitions.sum())==33 and int(all_seeds.cost_increase_repetitions.sum())==30, "Tail signs mismatch")
    require(int(all_seeds.savings_at_least_1pct_pool_repetitions.sum())==22 and int(all_seeds.increases_at_least_1pct_pool_repetitions.sum())==24, "Practical tail counts mismatch")

    p = read("results/power_manifest.json")
    require(p["status"]=="complete" and p["populations"]==30 and p["matched_orders"]==30000 and p["level_runs"]==60000, "Power extension incomplete")
    require(p["nested_pair_violations"]==0, "Matched-level nesting failure")
    for name, digest in p["source_sha256"].items():
        require(sha(ROOT/name)==digest, "Power source mismatch: "+name)
    for name, digest in p["output_sha256"].items():
        require(sha(ROOT/"results"/name)==digest, "Power output mismatch: "+name)
    q=read("results/publication_reporting_check.json")
    require(q["status"]=="passed" and q["script_sha256"]==sha(ROOT/"scripts/export_publication_tables.py"), "Publication export source mismatch")
    for block, metadata in q["blocks"].items():
        require(sha(ROOT/metadata["input"])==metadata["input_sha256"], "Publication input changed: "+block)
        require(sha(ROOT/metadata["output"])==metadata["output_sha256"], "Publication output changed: "+block)
        frame = pd.read_csv(ROOT/metadata["output"])
        require("ci_rules_out_1pct_saving" not in frame.columns, "Deprecated inferential flag survives")

    manifest_path=ROOT/"RELEASE_MANIFEST_SHA256.json"
    release_files=0
    require(manifest_path.is_file(), "Missing required RELEASE_MANIFEST_SHA256.json; this is not a sealed release")
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest["files"].items():
        require(sha(ROOT/name)==digest, "Release file changed: "+name)
    release_files=len(manifest["files"])
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_*.py")
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    require(result.wasSuccessful(), stream.getvalue())
    require(result.testsRun==9, "Expected eight acceptance tests and one PAC convention test")
    report={"status":"passed", "checks":checks, "frozen_files_verified":len(original),
            "release_manifest_present":manifest_path.exists(), "release_files_verified":release_files,
            "regression_tests":result.testsRun,"regression_failures":len(result.failures),"regression_errors":len(result.errors),
            "frozen_main_study_unchanged":True,"cell_partition":b["cell_partition"],
            "accepted_state_pairs":620,"power_matched_orders":30000,
            "protocol_sha256":sha(ROOT/"protocol_v8_2.json"),
            "source_sha256":sha(Path(__file__)),
            "scope":"Integrity, cross-table accounting and focused regression; not a new experiment or universal arithmetic proof."}
    (ROOT/"results/release_check.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
