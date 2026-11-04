"""Monte Carlo study of the repair safeguard (CPU; synthetic finite populations).

A candidate group has N items; each audited item reveals a gain d in {-1, 0, +1} (x = (d+1)/2).
Populations are fixed exactly (not sampled) with a share z of zero-gain items and the rest split to
give a chosen mean gain. Items are audited in random order under the same stopping rules as
run_group_repair (futility after 10 audits if mean x < 0.45; stop at half the group or 80 audits).
Three acceptance rules are compared on identical audit sequences:
  remainder   accept when N(2L_t - 1) - sum d > 0 (the procedure in the paper)
  whole       accept when the lower bound on the WHOLE group's mean x exceeds 1/2 (L_t > 0.5);
              this ignores that audited items keep their expert labels
  unguarded   accept if the audited mean x > 0.5 at the stopping time of the remainder rule
Harmful = accepted while the true gain on the unaudited remainder is <= 0.
Usage: python run_safeguard_validity.py --out outdir [--reps 2000]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit_repair.core import WorLowerCS  # noqa: E402


def population(N, mean_gain, zero_share):
    n0 = int(round(zero_share * N)); rest = N - n0
    npos = int(round((mean_gain * N + rest) / 2)); npos = min(max(npos, 0), rest)
    return np.array([1] * npos + [-1] * (rest - npos) + [0] * n0)


def one_run(d, alpha, rng, k_max=80):
    N = len(d); order = rng.permutation(N)
    cs = WorLowerCS(N=N, alpha=alpha)
    acc = {"remainder": False, "whole": False}; s = 0; t_stop = None
    for t, i in enumerate(order, 1):
        cs.update((d[i] + 1) / 2); s += d[i]
        if not acc["whole"] and t < N and cs.lower > 0.5:
            acc["whole"] = True; t_whole = t
        if t < N and cs.remainder_gain_lb() > 0:
            acc["remainder"] = True; t_stop = t; break
        if (t >= 10 and cs.mean < 0.45) or t >= k_max or t >= 0.5 * N:
            t_stop = t; break
    t_stop = t_stop or N
    res = {}
    rem_gain = d[order[t_stop:]].sum()
    res["remainder"] = (acc["remainder"], acc["remainder"] and rem_gain <= 0)
    res["unguarded"] = (s / t_stop > 0, s / t_stop > 0 and rem_gain <= 0)
    if acc["whole"]:
        rg = d[order[t_whole:]].sum(); res["whole"] = (True, rg <= 0)
    else:
        res["whole"] = (False, False)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261103)
    rows = []
    for N in (50, 100, 300):
        for zero in (0.0, 0.4):
            for g in np.round(np.arange(-0.20, 0.41, 0.05), 2):
                d = population(N, g, zero)
                tally = {k: [0, 0] for k in ("remainder", "whole", "unguarded")}
                for _ in range(a.reps):
                    for k, (acc, harm) in one_run(d, a.alpha, rng).items():
                        tally[k][0] += acc; tally[k][1] += harm
                for k, (acc, harm) in tally.items():
                    rows.append(dict(N=N, zero_share=zero, mean_gain=float(d.mean()), rule=k,
                                     accept_rate=acc / a.reps, harmful_rate=harm / a.reps))
        print("done N", N, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out / "safeguard_validity.csv", index=False)
    worst = df.groupby("rule").harmful_rate.max()
    txt = (f"alpha = {a.alpha}, {a.reps} audit orders per population\nworst harmful-acceptance rate per rule:\n"
           f"{worst.round(4).to_string()}\n")
    (a.out / "report.txt").write_text(txt); print(txt)
    (a.out / "manifest.json").write_text(json.dumps(dict(reps=a.reps, alpha=a.alpha, numpy=np.__version__)))


if __name__ == "__main__":
    main()
