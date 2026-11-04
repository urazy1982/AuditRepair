"""Export explicit-denominator v8.2 tables from untouched frozen summaries.

This performs reporting corrections only. It does not recompute or strengthen
the archived pointwise seed-bootstrap intervals.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "frozen_v8_1" / "analysis"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    inventory = {}
    for block in ("E1", "E2_E3"):
        path = source / f"{block}_effects.csv"
        d = pd.read_csv(path)
        original = d.copy()
        d = d.drop(columns=["ci_rules_out_1pct_saving", "lower_bound_meets_1pct"], errors="raise")
        d = d.rename(columns={
            "point_estimate_meets_1pct": "observed_mean_saving_at_least_1pct_of_pool",
            "saving_pct_of_pool": "mean_saving_percent_of_pool",
            "ci95_pct_low": "pointwise_bootstrap_low_percent_of_pool",
            "ci95_pct_high": "pointwise_bootstrap_high_percent_of_pool",
            "ci95_saved_low": "pointwise_bootstrap_low_labels",
            "ci95_saved_high": "pointwise_bootstrap_high_labels",
        })
        d["mean_saving_percent_of_mean_baseline_cost"] = 100 * d.mean_reviews_saved / d.reference_mean_cost
        d["baseline_cost_denominator_definition"] = "ratio of mean saving to mean baseline cost; not mean of per-repetition ratios"
        d["pointwise_bootstrap_is_degenerate"] = d.pointwise_bootstrap_low_labels == d.pointwise_bootstrap_high_labels
        d["interval_interpretation"] = "pointwise preprocessing-seed bootstrap conditional on the fixed panel; no equivalence or exclusion of unseen algorithmic outcomes"
        d["experiment_block"] = block
        d["source_version"] = "frozen_v8_1"
        np.testing.assert_array_equal(d.mean_reviews_saved, original.mean_reviews_saved)
        np.testing.assert_array_equal(d.pointwise_bootstrap_low_labels, original.ci95_saved_low)
        np.testing.assert_array_equal(d.pointwise_bootstrap_high_labels, original.ci95_saved_high)
        np.testing.assert_allclose(d.mean_saving_percent_of_pool, 100*d.mean_reviews_saved/d.n, atol=1e-12, rtol=0)
        target = out / f"publication_{block}_effects.csv"
        d.to_csv(target, index=False)
        inventory[block] = {"input": str(path.relative_to(ROOT)), "input_sha256": sha(path),
                            "output": str(target.relative_to(ROOT)), "output_sha256": sha(target),
                            "rows": len(d), "means_and_intervals_unchanged": True}
        if block == "E1":
            selected = d[(d.arm == "O5_all_errors") & (d.reference == "same_audit_ds")]
            columns = ["pool", "n", "reference_mean_cost", "arm_mean_cost", "mean_reviews_saved",
                       "mean_saving_percent_of_pool", "mean_saving_percent_of_mean_baseline_cost",
                       "pointwise_bootstrap_low_labels", "pointwise_bootstrap_high_labels"]
            assert len(selected) == 7
            selected[columns].to_csv(out / "publication_E1_oracle_denominators.csv", index=False)
    report = {
        "status": "passed", "protocol_sha256": sha(ROOT / "protocol_v8_2.json"),
        "script_sha256": sha(Path(__file__)), "blocks": inventory,
        "removed_fields": ["ci_rules_out_1pct_saving", "lower_bound_meets_1pct"],
        "replacement_statement": "No practically important mean saving was observed in the sampled realizations of these repair configurations.",
        "scope_of_statement": "The 42 same-audit/same-score certified-repair contrasts only; not all score-only or oracle contrasts in the files.",
        "warning": "A bootstrap interval from all-zero seed means cannot rule out rare events, bound their frequency, or establish practical equivalence."
    }
    (out / "publication_reporting_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "rows": sum(x["rows"] for x in inventory.values()),
                      "removed_exclusion_flags": True, "means_and_intervals_unchanged": True}))


if __name__ == "__main__":
    main()
