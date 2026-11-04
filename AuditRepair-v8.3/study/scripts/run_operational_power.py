#!/usr/bin/env python3
"""Matched finite-group acceptance curves at alpha=.05 and .005.

This consumes the frozen v8.2 amendment, uses the corrected production CS,
and never reads benchmark gold or generates LLM annotations.  It measures an
isolated group with its full per-group allowance, not the power of the whole
shortlist policy.  The two levels share each complete audit permutation.

Run from any working directory:
    python3 scripts/run_operational_power.py --workers 2
An existing completed run is reused only after checking source/input/output
hashes.  Use --out for a separate reproduction, not to overwrite the release.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corrected"))
from audit_repair.core import WorLowerCS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty results table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_binomial_interval(successes: int, n: int) -> tuple[float, float]:
    """Two-sided, pointwise, equal-tailed Clopper--Pearson 95% interval."""
    return (
        float(beta.ppf(.025, successes, n - successes + 1)) if successes else 0.,
        float(beta.ppf(.975, successes + 1, n - successes)) if successes < n else 1.,
    )


def population(N: int, requested_gain: float, zero_share: float) -> np.ndarray:
    """Exact original safeguard-validity integer construction, including round."""
    n0 = int(round(zero_share * N))
    rest = N - n0
    npos = int(round((requested_gain * N + rest) / 2))
    npos = min(max(npos, 0), rest)
    return np.asarray([1] * npos + [-1] * (rest - npos) + [0] * n0,
                      dtype=np.int64)


def simulate_one_level(gains: np.ndarray, alpha: float, stop: dict) -> dict:
    """Use the corrected production predicate; retain operational stop order."""
    N = len(gains)
    cs = WorLowerCS(N=N, alpha=alpha, grid=stop["grid_points"], c=stop["bet_cap_c"])
    accepted = False
    reason = "population_exhausted"
    for t, gain in enumerate(gains, 1):
        cs.update((int(gain) + 1) / 2)
        if cs.remainder_is_positive():
            accepted, reason = True, "certified"
            break
        if t >= stop["futility_after"] and cs.mean < stop["futility_mean_x"]:
            reason = "futility"
            break
        if t >= stop["max_draws"] or 2 * t >= N:
            reason = "audit_cap" if t >= stop["max_draws"] else "fraction_cap"
            break
    true_remainder_gain = int(np.sum(gains[cs.t:], dtype=np.int64))
    nominal, represented = cs.remainder_exact_numerators()
    if cs.audited_gain_sum != int(np.sum(gains[:cs.t], dtype=np.int64)):
        raise AssertionError("Production exact gain sum differs from independent sum")
    if accepted != (0 < cs.t < N and nominal > 0 and represented > 0):
        raise AssertionError("Stop-state acceptance is inconsistent with exact predicate")
    return {
        "accepted": int(accepted),
        "beneficial_accepted": int(accepted and true_remainder_gain > 0),
        "nonbeneficial_accepted": int(accepted and true_remainder_gain <= 0),
        "draws": cs.t,
        "stop_reason": reason,
        "true_remainder_gain": true_remainder_gain,
        "lower_grid_index": cs.lower_grid_index,
        "audited_gain_sum": cs.audited_gain_sum,
        "nominal_remainder_numerator": nominal,
        "represented_remainder_numerator": represented,
    }


def simulate_population(task: tuple[int, dict, dict]) -> tuple[dict, list[dict]]:
    index, composition, plan = task
    d = population(composition["N"], composition["requested_mean_gain"],
                   composition["zero_gain_fraction"])
    rng = np.random.default_rng(np.random.SeedSequence([plan["root_seed"], index]))
    rows = []
    for rep in range(plan["independent_WOR_orders_per_population"]):
        order = rng.permutation(len(d))
        path = d[order]
        row = {
            **composition,
            "rep": rep,
            "order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest(),
        }
        outcomes = [simulate_one_level(path, alpha, plan["stopping"])
                    for alpha in plan["levels"]]
        # Lower alpha has a higher rejection boundary.  On this common path it
        # cannot accept unless the higher alpha accepts no later.  Futility and
        # caps depend on the same observations, not on alpha.
        if outcomes[1]["accepted"] and not outcomes[0]["accepted"]:
            raise AssertionError("Nested-alpha acceptance failed on a matched path")
        if outcomes[1]["accepted"] and outcomes[1]["draws"] < outcomes[0]["draws"]:
            raise AssertionError("Stricter alpha accepted before the looser alpha")
        for suffix, outcome in zip(("a050", "a005"), outcomes):
            row.update({f"{name}_{suffix}": value for name, value in outcome.items()})
        rows.append(row)
    return composition, rows


def summarize(composition: dict, rows: list[dict], plan: dict) -> tuple[list[dict], dict]:
    n = len(rows)
    summaries = []
    for suffix, alpha, label in zip(("a050", "a005"), plan["levels"], plan["level_labels"]):
        accepted = sum(r[f"accepted_{suffix}"] for r in rows)
        beneficial = sum(r[f"beneficial_accepted_{suffix}"] for r in rows)
        nonbeneficial = sum(r[f"nonbeneficial_accepted_{suffix}"] for r in rows)
        lo, hi = exact_binomial_interval(accepted, n)
        blo, bhi = exact_binomial_interval(beneficial, n)
        hlo, hhi = exact_binomial_interval(nonbeneficial, n)
        accepted_rows = [r for r in rows if r[f"accepted_{suffix}"]]
        gains = [r[f"true_remainder_gain_{suffix}"] for r in accepted_rows]
        summaries.append({
            **composition, "alpha_group": alpha, "level_interpretation": label,
            "orders": n, "accepted": accepted, "acceptance_probability": accepted / n,
            "acceptance_ci95_low": lo, "acceptance_ci95_high": hi,
            "beneficial_accepted": beneficial, "beneficial_acceptance_probability": beneficial / n,
            "beneficial_ci95_low": blo, "beneficial_ci95_high": bhi,
            "nonbeneficial_accepted": nonbeneficial,
            "nonbeneficial_acceptance_probability": nonbeneficial / n,
            "nonbeneficial_ci95_low": hlo, "nonbeneficial_ci95_high": hhi,
            "mean_draws_all_orders": float(np.mean([r[f"draws_{suffix}"] for r in rows])),
            "mean_draws_given_acceptance": float(np.mean([r[f"draws_{suffix}"] for r in accepted_rows])) if accepted else None,
            "mean_true_remainder_gain_given_acceptance": float(np.mean(gains)) if accepted else None,
            "min_true_remainder_gain_given_acceptance": min(gains) if accepted else None,
            "median_true_remainder_gain_given_acceptance": float(np.median(gains)) if accepted else None,
            "max_true_remainder_gain_given_acceptance": max(gains) if accepted else None,
            "stopped_futility": sum(r[f"stop_reason_{suffix}"] == "futility" for r in rows),
            "stopped_audit_cap": sum(r[f"stop_reason_{suffix}"] == "audit_cap" for r in rows),
            "stopped_fraction_cap": sum(r[f"stop_reason_{suffix}"] == "fraction_cap" for r in rows),
        })
    both = sum(r["accepted_a050"] and r["accepted_a005"] for r in rows)
    high_only = sum(r["accepted_a050"] and not r["accepted_a005"] for r in rows)
    low_only = sum(r["accepted_a005"] and not r["accepted_a050"] for r in rows)
    neither = n - both - high_only - low_only
    lo, hi = exact_binomial_interval(high_only, n)
    if low_only:
        raise AssertionError("Cannot use nested-pair interval when low-only acceptances exist")
    pair = {
        **composition, "matched_orders": n, "both_accept": both,
        "only_a050_accepts": high_only, "only_a005_accepts": low_only,
        "neither_accepts": neither,
        "acceptance_probability_difference_a005_minus_a050": -high_only / n,
        "difference_ci95_low": -hi, "difference_ci95_high": -lo,
        "difference_interval_method": "negative exact binomial interval for nested paired lost-acceptance indicator",
    }
    return summaries, pair


def plot_curves(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.3), sharex=True, sharey=True)
    styles = [(0.05, "#2166ac", "o", ".05: illustrative single-group level"),
              (0.005, "#b2182b", "s", ".005: per-group level for a 10-group shortlist")]
    for row, zero in enumerate((0., .4)):
        for col, N in enumerate((50, 100, 300)):
            ax = axes[row, col]
            for alpha, color, marker, label in styles:
                part = sorted((x for x in rows if x["N"] == N and x["zero_gain_fraction"] == zero
                               and x["alpha_group"] == alpha), key=lambda x: x["actual_mean_gain"])
                x = np.asarray([p["actual_mean_gain"] for p in part])
                y = 100 * np.asarray([p["acceptance_probability"] for p in part])
                low = 100 * np.asarray([p["acceptance_ci95_low"] for p in part])
                high = 100 * np.asarray([p["acceptance_ci95_high"] for p in part])
                ax.errorbar(x, y, yerr=np.maximum(np.vstack([y-low, high-y]), 0),
                            color=color, marker=marker, linewidth=1.5, markersize=4,
                            capsize=2, label=label)
            ax.set_title(f"N = {N}; {int(100*zero)}% zero-gain items", fontsize=10)
            ax.set_xlim(-.015, .415)
            ax.set_ylim(-1, 101)
            ax.set_xticks([0, .1, .2, .3, .4])
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("Actual mean gain per item", fontsize=10)
            if col == 0:
                ax.set_ylabel("Acceptance probability (%)", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, .94))
    fig.suptitle("Matched fixed-group acceptance curves with corrected arithmetic", fontsize=13, y=.995)
    fig.text(.5, .005,
             "1,000 matched orders per population; pointwise exact 95% intervals. Full per-group allowance; not whole-shortlist power.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .035, 1, .88))
    fig.savefig(out / "power_acceptance_curves.pdf")
    fig.savefig(out / "power_acceptance_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol_v8_2.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocol = json.loads(args.protocol.read_text())
    plan = protocol["power_extension"]
    # The script suffixes and plot labels are deliberately tied to this frozen
    # comparison; reject edits that silently change the estimand.
    if plan["levels"] != [.05, .005] or plan["stopping"]["fraction_cap"] != "ceil(N/2)":
        raise ValueError("Unexpected frozen alpha comparison or fraction cap")
    if not plan["matched_levels_on_same_order"] or not plan["fixed_sample_size"]:
        raise ValueError("This runner requires the frozen paired, fixed-size design")
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "protocol_v8_2.json": args.protocol,
        "scripts/run_operational_power.py": Path(__file__).resolve(),
        "corrected/audit_repair/core.py": ROOT / "corrected/audit_repair/core.py",
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    manifest_path = args.out / "power_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("status") != "complete" or old["source_sha256"] != source_hashes:
            raise ValueError("Existing power run has different sources or is incomplete; use a fresh --out")
        for name, digest in old["output_sha256"].items():
            if sha256(args.out / name) != digest:
                raise ValueError(f"Existing power output has changed: {name}")
        print(json.dumps({"status": "cached_outputs_verified", "populations": old["populations"],
                          "matched_orders": old["matched_orders"], "level_runs": old["level_runs"]}))
        return
    compositions = []
    for N in plan["population_sizes"]:
        for zero in plan["zero_gain_fractions"]:
            for gain in plan["requested_mean_gains"]:
                d = population(N, gain, zero)
                compositions.append({
                    "population_id": len(compositions), "N": N,
                    "zero_gain_fraction": zero, "requested_mean_gain": gain,
                    "actual_mean_gain": float(np.mean(d)),
                    "n_positive": int(np.sum(d == 1)), "n_negative": int(np.sum(d == -1)),
                    "n_zero": int(np.sum(d == 0)), "total_gain": int(np.sum(d)),
                    "population_sha256": hashlib.sha256(np.asarray(d, dtype="<i8").tobytes()).hexdigest(),
                })
    # This record is written BEFORE generating any audit order.
    execution_freeze = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "power_extension": plan,
        "populations": compositions,
        "seed_scheme": "One np.random.default_rng(np.random.SeedSequence([root_seed, population_id])) per population; sequential rng.permutation(N) for 1000 repetitions; same order for both levels.",
        "ci_method": "Pointwise two-sided equal-tailed Clopper-Pearson 95% intervals; paired acceptance difference uses nested lost-acceptance indicators, not independent alpha samples.",
        "original_population_constructor_source_sha256": "49453e74f52b47f75f854c24d55652871b72c83279fcd46df949167dba66f2d1",
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "workers": args.workers,
    }
    freeze_path = args.out / "power_execution_freeze.json"
    if freeze_path.exists():
        raise FileExistsError("An unfinished frozen power run exists; use a fresh --out")
    write_json(freeze_path, execution_freeze)
    print(json.dumps({"event": "execution_frozen_before_orders", "sha256": sha256(freeze_path),
                      "populations": len(compositions), "orders_per_population": plan["independent_WOR_orders_per_population"]}), flush=True)
    start = time.monotonic()
    by_population = {}
    tasks = [(i, composition, plan) for i, composition in enumerate(compositions)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(simulate_population, task) for task in tasks]
        for future in as_completed(futures):
            composition, rows = future.result()
            by_population[composition["population_id"]] = rows
            print(json.dumps({"event": "population_complete", "completed": len(by_population),
                              "population_id": composition["population_id"], "N": composition["N"],
                              "zero_gain_fraction": composition["zero_gain_fraction"],
                              "actual_mean_gain": composition["actual_mean_gain"]}), flush=True)
    all_trials, summaries, pairs = [], [], []
    for composition in compositions:
        rows = by_population[composition["population_id"]]
        all_trials.extend(rows)
        summary, pair = summarize(composition, rows, plan)
        summaries.extend(summary)
        pairs.append(pair)
    write_csv(args.out / "power_populations.csv", compositions)
    write_csv(args.out / "power_trials.csv", all_trials)
    write_csv(args.out / "power_population_summary.csv", summaries)
    write_csv(args.out / "power_paired_summary.csv", pairs)
    plot_curves(summaries, args.out)
    after_hashes = {name: sha256(path) for name, path in source_paths.items()}
    if after_hashes != source_hashes:
        raise AssertionError("Source or protocol changed during the frozen power run")
    n_orders = len(all_trials)
    acceptance_by_alpha = {str(a): sum(r["accepted"] for r in summaries if r["alpha_group"] == a)
                           for a in plan["levels"]}
    harm_by_alpha = {str(a): sum(r["nonbeneficial_accepted"] for r in summaries if r["alpha_group"] == a)
                    for a in plan["levels"]}
    report = (
        "V8.2 MATCHED FIXED-GROUP ACCEPTANCE STUDY\n\n"
        f"{len(compositions)} prespecified finite populations; {n_orders:,} matched random audit orders; {2*n_orders:,} level-specific group runs.\n"
        "The alpha=.05 and alpha=.005 curves use the same permutations and the corrected production remainder predicate.\n"
        "Alpha=.005 is alpha_repair/10 for a maximum-size shortlist. Actual shorter shortlists use alpha_repair/g.\n"
        "Every isolated group receives its full per-group allowance: at most min(80,ceil(N/2)) observations, with the unchanged futility rule.\n"
        "No warm-up, shortlist ranking, shared group budget or final PAC cost is simulated. These curves are not whole-policy power.\n\n"
        f"Total accepted counts across this deliberately heterogeneous fixture grid: {json.dumps(acceptance_by_alpha, sort_keys=True)}.\n"
        f"Total nonbeneficial accepted remainders across the grid: {json.dumps(harm_by_alpha, sort_keys=True)}.\n"
        "These totals are inventory counts; pooling them is not an operational acceptance probability or an error guarantee.\n"
        "Report the population-specific rows and their actual integer gains. At N=50, requested .1 and .3 yield actual .12 and .28.\n\n"
        "Uncertainty: exact pointwise two-sided binomial 95% intervals for each fixed population/level.\n"
        "The paired difference is .005 minus .05; its interval is the negative exact interval for lost-acceptance indicators.\n"
        "There are no .005-only acceptances under the matched paths; this nesting is checked for every order.\n"
        "Intervals quantify Monte Carlo uncertainty only, not new-panel variation, joint coverage across the grid, or proof of the statistical guarantee.\n"
        "No simulation count or plotted setting was selected after viewing outcomes.\n"
    )
    (args.out / "power_summary.txt").write_text(report)
    output_names = ["power_execution_freeze.json", "power_populations.csv", "power_trials.csv",
                    "power_population_summary.csv", "power_paired_summary.csv", "power_acceptance_curves.pdf",
                    "power_acceptance_curves.png", "power_summary.txt"]
    manifest = {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "output_sha256": {name: sha256(args.out / name) for name in output_names},
        "populations": len(compositions), "matched_orders": n_orders, "level_runs": 2 * n_orders,
        "nested_pair_violations": 0, "source_unchanged": True,
        "acceptance_counts_by_alpha": acceptance_by_alpha,
        "nonbeneficial_acceptance_counts_by_alpha": harm_by_alpha,
        "elapsed_seconds": time.monotonic() - start,
        "scope": plan["scope"], "inference": execution_freeze["ci_method"],
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "populations": len(compositions),
                      "matched_orders": n_orders, "elapsed_seconds": manifest["elapsed_seconds"],
                      "nonbeneficial_acceptance_counts_by_alpha": harm_by_alpha}), flush=True)


if __name__ == "__main__":
    main()
