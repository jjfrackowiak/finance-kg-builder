"""Regenerate the section D group-means figure with the grand-SD band.

The point of the figure is that no hyperparameter separates from the noise. Error
bars alone do not show that — they show each level's own spread, which the eye reads
as "these overlap" without a reference for how much overlap is expected. Shading the
band the runs already occupy (grand mean +/- 1 SD) gives that reference directly:
every level's mean sits well inside it.

Writes group_means.png next to this file; swap it into the report with
embed_figure.py.
"""

import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
GROUND, PANEL = "#fbfbfa", "#f4f4f1"
INK, MUTED, RULE, ACCENT = "#14161a", "#878d96", "#e3e4e0", "#2a78d6"

FACTORS = [
    ("lookback_days", "Lookback (days)", 9, None),
    ("steps", "Evolution steps", 12, None),
    ("chain_hops", "Chain hops", 12, None),
    ("evolution_prompt", "Evolution prompt", 12, ["default", "event_driven", "fundamental"]),
]


def main() -> None:
    rows = list(csv.DictReader(open(HERE / "results.csv")))
    y = [float(r["final_auc"]) for r in rows]
    grand, sd = st.mean(y), st.stdev(y)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), facecolor=GROUND)
    fig.suptitle(
        f"Group means by factor — dotted line = grand mean ({grand:.3f}), "
        f"band = ±1 SD across all 36 runs ({sd:.3f})",
        fontsize=16, color=INK, x=0.012, ha="left", y=0.975,
    )

    for ax, (key, title, per_level, order) in zip(axes.flat, FACTORS):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r[key], []).append(float(r["final_auc"]))
        levels = order or sorted(groups, key=lambda k: int(k))
        means = [st.mean(groups[k]) for k in levels]
        sds = [st.stdev(groups[k]) for k in levels]
        xs = range(len(levels))

        ax.set_facecolor(GROUND)
        # The band is the comparison the section is making — draw it under everything.
        ax.axhspan(grand - sd, grand + sd, color=ACCENT, alpha=0.07, zorder=0)
        ax.axhline(grand + sd, color=ACCENT, alpha=0.35, lw=0.9, zorder=1)
        ax.axhline(grand - sd, color=ACCENT, alpha=0.35, lw=0.9, zorder=1)
        ax.axhline(grand, color=INK, ls=":", lw=1.2, zorder=2)

        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#c3c5bf", elinewidth=1.8,
                    capsize=5, capthick=1.8, zorder=3)
        ax.scatter(xs, means, s=110, color=ACCENT, zorder=4)
        for x, m in zip(xs, means):
            # Labels sit on top of the grand-mean line for levels near it; an opaque
            # box masks the line rather than nudging the label somewhere arbitrary.
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(13, -4), fontsize=12, color=INK, family="monospace",
                        zorder=5,
                        bbox=dict(facecolor=GROUND, edgecolor="none", pad=1.5))

        ax.set_title(f"{title}   (n={per_level} per level)", fontsize=15, color=INK,
                     loc="left", pad=12)
        ax.set_ylabel("mean AUC ± SD", fontsize=12, color=MUTED)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(levels, fontsize=12, color=MUTED, family="monospace")
        ax.set_xlim(-0.5, len(levels) - 0.35)
        ax.set_ylim(0.50, 0.72)
        ax.tick_params(axis="y", labelsize=11, colors=MUTED)
        ax.grid(axis="y", color=RULE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = HERE / "group_means.png"
    fig.savefig(out, dpi=110, facecolor=GROUND)
    print(f"grand mean {grand:.4f}  SD {sd:.4f}  band {grand - sd:.4f}–{grand + sd:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
