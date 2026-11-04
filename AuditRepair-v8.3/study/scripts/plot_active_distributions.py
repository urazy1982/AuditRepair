"""Show all paired repetitions while keeping preprocessing seeds visible."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "frozen_v8_1/results/E2_E3/rows.csv"
    d = pd.read_csv(path)
    out = ROOT / "results"
    cells = [("ag_news_5000", "G0_uniform", "AG News development  G0  S1"),
             ("ag_news_5000", "G1", "AG News development  G1  S1"),
             ("ag_news_holdout", "G1", "AG News holdout  G1  S1")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True, sharey=True)
    for ax, (pool, generator, title) in zip(axes, cells):
        z = d[(d.pool == pool) & (d.generator == generator) & (d.score == "S1") & (d.role == "repair")].copy()
        assert len(z) == 200 and z.seed.nunique() == 20
        z["saving"] = -z.paired_cost_delta
        jitter = (z.rep.to_numpy() - 4.5) * .045
        values = z.saving.to_numpy()
        colors = np.where(values > 0, "#176b72", np.where(values < 0, "#a94132", "#a7adb2"))
        ax.scatter(z.seed+jitter, values, c=colors, s=12, alpha=.7, linewidths=0, zorder=3)
        means = z.groupby("seed").saving.mean()
        ax.scatter(means.index, means, marker="D", c="black", s=13, linewidths=.3, edgecolors="white", zorder=4)
        margin = .01 * int(z.n.iloc[0])
        ax.axhline(0, c="#666666", lw=.7, zorder=1)
        for v in (margin, -margin):
            ax.axhline(v, c="#999999", ls="--", lw=.65, zorder=1)
        ax.set_title(title, loc="left", fontsize=10, pad=5)
        ax.text(.02, .97, f"{int(np.count_nonzero(values))}/200 changed; mean {values.mean():+.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.2)
        ax.set_ylim(-420, 540)
        ax.set_yticks([-400, -200, 0, 200, 400])
        ax.set_ylabel("Labels saved")
        ax.grid(axis="y", alpha=.12)
    axes[-1].set_xticks(range(0,20))
    axes[-1].set_xlabel("Preprocessing seed   (10 paired calibration repetitions within each seed)")
    axes[-1].set_xlim(-.6, 19.6)
    fig.subplots_adjust(left=.12, right=.98, top=.96, bottom=.105, hspace=.32)
    fig.savefig(out/"active_cost_distributions.png", dpi=220)
    fig.savefig(out/"active_cost_distributions.pdf")
    plt.close(fig)
    (out/"active_cost_distributions_caption.txt").write_text(
        "Every paired repetition in the three cells with nonzero costs is shown. Positive values save unique expert labels relative to the same-audit, same-score baseline; negative values add labels. Small circles are repetitions, black diamonds are preprocessing-seed means, and dashed lines mark +/-1% of pool size (50 labels for development, 26 for holdout). Zero-valued circles overlap. Seeds remain visible because repetitions within a preprocessing state are clustered. These are the three nonzero cells identified in v8.1, not newly selected winners; the companion tables report all 42 cells and all 10 active cells.\n")
    report = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "plotted_repetitions": 600, "all_42_cells_in_companion_tables": True}
    (out/"active_plot_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
