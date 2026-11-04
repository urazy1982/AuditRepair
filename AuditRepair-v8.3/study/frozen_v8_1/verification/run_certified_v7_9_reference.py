"""Certified comparison: expert labels needed for  P(final error <= eps) >= 1 - alpha.

Every arm ends with PAC Labels; all expert labels (pre-PAC audits + PAC draws + PAC threshold)
are counted.
  pac_only            DS predictions -> PAC
  ds_unc_audits+pac   B DS-uncertainty audits, DS re-fit -> PAC
  ds_same_audits+pac  DS re-fit with exactly the audits of the guarded arm -> PAC
  guarded+pac         certified group repair (budget B) -> PAC
  unguarded+pac       SAME candidates and audit evidence, no per-repair safeguard -> same PAC
                      (tests whether local certification adds value beyond better predictions)
Usage: python run_certified.py --sim --scenario clear --eps 0.03 --alpha 0.1 --B 400 --seeds 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_repair.core import Oracle, dawid_skene, run_group_repair, uncertainty_fill
from audit_repair.pac import pac_labels
from audit_repair.simulate import SCENARIOS, simulate
from run_pilot import load_labels


def unc(post):
    return 1.0 - post.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--scenario", default="clear")
    ap.add_argument("--labels")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.1, help="alpha_final for the PAC stage")
    ap.add_argument("--alpha_repair", type=float, default=0.05,
                    help="family-wise level for the repair certificates; a joint claim (every "
                         "accepted repair beneficial AND final error <= eps) holds with "
                         "probability >= 1 - (alpha_repair + alpha)")
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--ref_share", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--pac_reps", type=int, default=1,
                    help="independent repetitions of PAC randomness per annotation dataset")
    ap.add_argument("--root_seed", type=int, default=20261030)
    ap.add_argument("--boot", type=int, default=2000, help="cluster-bootstrap resamples")
    ap.add_argument("--min_size", type=int, default=30, help="minimum candidate group size")
    ap.add_argument("--out", default="certified.csv")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run only the five original arms for same-environment regression checks")
    a = ap.parse_args()
    rows, gstats, certificates = [], [], []
    for seed in range(a.seeds):
        # Independent random streams per dataset: data generation / item order, preprocessing
        # (warm-up + group audits), and PAC certification. PAC streams are shared ACROSS methods
        # (paired comparison) but never with data or preprocessing.
        ss_data, ss_pre, ss_pac = np.random.SeedSequence([a.root_seed, seed]).spawn(3)
        pac_streams = ss_pac.spawn(a.pac_reps)
        if a.sim:
            L, gold = simulate(seed=ss_data, **SCENARIOS[a.scenario]); K = int(gold.max()) + 1
            original_rows = np.arange(len(gold))
        else:
            L, gold, K, _ = load_labels(a.labels)
            perm = np.random.default_rng(ss_data).permutation(len(gold)); L, gold = L[perm], gold[perm]
            original_rows = perm
        def run_pac(name, pre, pred, U, already=None):
            for r in range(a.pac_reps):
                res = pac_labels(pred, U, gold, eps=a.eps, alpha=a.alpha, m=a.m,
                                 already=already, seed=pac_streams[r])
                rows.append({"method": name, "seed": seed, "rep": r, "pre_audits": pre,
                             "expert_total": pre + res[1]["expert_new"], "error": res[1]["error"],
                             "violated": res[1]["error"] > a.eps})

        post = dawid_skene(L, K)
        run_pac("pac_only", 0, post.argmax(1), unc(post))

        orc = Oracle(gold, a.B); cl = {}
        p2 = uncertainty_fill(L, K, orc, cl)
        run_pac("ds_unc_audits+pac", len(cl), p2.argmax(1), unc(p2), cl)

        res = run_group_repair(L, K, gold, a.B, alpha=a.alpha_repair, ref_share=a.ref_share,
                               seed=ss_pre, min_size=a.min_size)
        glog = res["guarded"][1]["log"]
        for j, g in enumerate(glog):
            certificates.append({"seed": seed, "group_index": j, **g,
                "original_rows": [int(original_rows[i]) for i in g["members"]],
                "audit_original_rows": [int(original_rows[h["item"]]) for h in g["audit_history"]]})
        gstats.append({"seed": seed, "candidates_tested": len(glog),
                       "repaired": sum(g["guarded"] for g in glog),
                       "rejected_or_abstained": sum(not g["guarded"] for g in glog),
                       "group_phase_audits": sum(g["audited"] for g in glog),
                       "median_group_size": float(np.median([g["size"] for g in glog])) if glog else 0,
                       "harmful_accepted": res["guarded"][1]["harmful_group_repairs"],
                       "unguarded_harmful": res["unguarded"][1]["harmful_group_repairs"]})
        aud = res["guarded"][1]["audited_labels"]
        p3 = dawid_skene(L, K, dict(aud))
        run_pac("ds_same_audits+pac", len(aud), p3.argmax(1), unc(p3), aud)
        # Cross labels and ranking scores on the SAME evidence and PAC draws.
        repaired_y, guarded_info = res["guarded"]
        repair_U = unc(guarded_info["post"])
        for i, pa in guarded_info["repair_conf"].items():
            repair_U[i] = 1.0 - pa
        if not a.baseline_only:
            run_pac("repair_labels_ds_U+pac", len(aud), repaired_y, unc(p3), aud)
            run_pac("ds_labels_repair_U+pac", len(aud), p3.argmax(1), repair_U, aud)
        for k in ("guarded", "unguarded"):
            y, info = res[k]
            U = unc(info["post"])
            for i, pa in info["repair_conf"].items():
                U[i] = 1.0 - pa
            run_pac(f"{k}+pac", len(aud), y, U, aud)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    if a.sim:
        unit = "simulated datasets"
    else:
        unit = "preprocessing realizations of ONE fixed annotation file"
        print(f"NOTE: --labels mode. The {a.seeds} clusters below re-use the same annotations "
              f"({a.labels}) with different item orders and audit randomness. They measure "
              f"variability across preprocessing realizations on this dataset, not generalization "
              f"across datasets or source configurations.")
    report(df, a.boot, a.root_seed, unit)
    out = Path(a.out)
    gs = pd.DataFrame(gstats); gs.to_csv(out.with_name(out.stem + "_groups.csv"), index=False)
    with out.with_name(out.stem + "_certificates.jsonl").open("w") as f:
        for cert in certificates:
            f.write(json.dumps(cert) + "\n")
    print(f"\nRepair stage (alpha_repair = {a.alpha_repair}; joint claim level 1 - "
          f"{a.alpha_repair + a.alpha:.2f}). Per cluster means; harmful = accepted repair whose "
          f"unaudited remainder gain <= 0 (diagnostic):")
    print(gs.drop(columns="seed").agg(["mean", "sum"]).round(2).to_string())


def report(df, boot=2000, root_seed=0, unit="datasets"):
    """Diagnostics for a nested design (datasets x PAC repetitions).

    Validity of PAC is a theorem under its sampling assumptions; the failure counts below are
    diagnostics. Uncertainty is quantified by a cluster bootstrap that resamples DATASETS
    (repetitions stay together), never by pooling runs as if independent.
    """
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, 999]))
    seeds = np.array(sorted(df.seed.unique()))
    per = df.groupby(["method", "seed"]).agg(cost=("expert_total", "mean"),
                                             fail=("violated", "sum"), reps=("violated", "size"))
    methods = df.method.unique()

    def boot_ci(stat):  # stat: array over datasets -> scalar
        vals = [stat(rng.choice(len(seeds), len(seeds))) for _ in range(boot)]
        return np.percentile(vals, [2.5, 97.5])

    print(f"\nPer-method summary (cost = expert labels; CI = 95% cluster bootstrap over {unit})")
    print("Failure counts are diagnostics; validity of the final labels rests on the PAC argument.")
    for m in methods:
        c = per.loc[m].reindex(seeds)
        lo, hi = boot_ci(lambda ix: c.cost.to_numpy()[ix].mean())
        nf, nr = int(c.fail.sum()), int(c.reps.sum())
        if nf == 0:
            # resampling cannot create an event that never occurred: do not print [0, 0]
            fail_txt = (f"0/{nr} observed failures across {len(c)} clusters; bootstrap interval "
                        f"degenerate, not informative about unseen failures")
        else:
            flo, fhi = boot_ci(lambda ix: c.fail.to_numpy()[ix].sum() / c.reps.to_numpy()[ix].sum())
            fail_txt = (f"{nf}/{nr} observed failures ({nf / nr:.3f}) [{flo:.3f}, {fhi:.3f}]; "
                        f"clusters with >=1 failure {(c.fail > 0).sum()}/{len(c)}; "
                        f"max in one cluster {int(c.fail.max())}/{int(c.reps.max())}")
        print(f"  {m:22s} cost {c.cost.mean():7.0f} [{lo:6.0f}, {hi:6.0f}]   {fail_txt}")

    # paired cost comparison of guarded+pac against each DS arm and against the better DS arm per run
    piv = df.pivot_table(index=["seed", "rep"], columns="method", values="expert_total")
    if "guarded+pac" in piv:
        ds_arms = [m for m in ("ds_unc_audits+pac", "ds_same_audits+pac") if m in piv]
        piv["best_ds_per_run"] = piv[ds_arms].min(axis=1)
        print("\nPaired comparison, guarded+pac minus comparator (negative = guarded cheaper)")
        for comp in ds_arms + ["best_ds_per_run", "pac_only", "unguarded+pac",
                               "repair_labels_ds_U+pac", "ds_labels_repair_U+pac"]:
            if comp not in piv:
                continue
            d = (piv["guarded+pac"] - piv[comp]).groupby(level="seed").mean().reindex(seeds)
            win = (piv["guarded+pac"] < piv[comp]).mean()
            lo, hi = boot_ci(lambda ix: d.to_numpy()[ix].mean())
            print(f"  vs {comp:20s} mean diff {d.mean():7.0f} [{lo:6.0f}, {hi:6.0f}]   "
                  f"cheaper in {win:.0%} of runs")


if __name__ == "__main__":
    main()
