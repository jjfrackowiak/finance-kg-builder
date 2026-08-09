"""How often each proposed (node, relationship) addition beat the alternative.

Scoring a candidate against the run's best-so-far does not work: the acceptance gate makes
that reference the maximum seen so far, so a candidate's score mostly reflects how lucky
step 1 was in its run rather than how good the addition is.

The two candidates within a step are the fair comparison — same graph, same days, same
point in the run, differing only in which type was added. We report the **win rate**: how
often an addition scored higher than the one proposed alongside it. The AUC gaps
themselves are not worth reading at 42 validation days; which one won is.

Win rate averages 50% by construction, so it ranks additions against each other and says
nothing about whether additions help in absolute terms — that needs the base graph scored
on its own, which the pipeline does not currently do.

Writes node_effects.png; embed with embed_figure.py.
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
GROUND, INK, MUTED, RULE = "#fbfbfa", "#14161a", "#878d96", "#e3e4e0"
POS, NEG = "#0f8f62", "#c2521f"
MIN_N = 8


def head_to_head(path: Path) -> list:
    """Mark each candidate as having won or lost against its own step's alternative."""
    recs = [r for r in csv.DictReader(open(path)) if r["delta_auc"]]
    for r in recs:
        r["auc"] = float(r["auc"])
    steps = defaultdict(list)
    for r in recs:
        steps[(r["run_id"], r["step"])].append(r)
    for group in steps.values():
        top = max(x["auc"] for x in group)
        winners = [x for x in group if x["auc"] == top]
        for x in group:
            # An exact tie (one step in 180) counts as a loss for both.
            x["won"] = x["auc"] == top and len(winners) == 1
    return recs


def main() -> None:
    recs = head_to_head(HERE / "additions.csv")
    groups = defaultdict(list)
    for r in recs:
        groups[(r["node"], r["relationship"])].append(r["won"])

    rows = sorted(
        ((100 * sum(v) / len(v), k, len(v)) for k, v in groups.items() if len(v) >= MIN_N),
        reverse=True,
    )
    covered = sum(n for _, _, n in rows)

    labels = [f"{node}  —{rel}→" for _, (node, rel), _ in rows]
    means = [m for m, _, _ in rows]
    counts = [n for _, _, n in rows]

    fig, ax = plt.subplots(figsize=(13.5, 7.2), facecolor=GROUND)
    ax.set_facecolor(GROUND)
    ys = range(len(rows))
    ax.barh(list(ys), means, color=[POS if m > 50 else NEG for m in means],
            alpha=0.85, height=0.68, zorder=3)
    ax.axvline(50, color=INK, lw=1.1, zorder=4)

    for y, (m, n) in enumerate(zip(means, counts)):
        ax.annotate(f"{m:.0f}%   n={n}", (m, y), va="center", ha="left",
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=11, color=INK, family="monospace")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=11, color=INK, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("share of head-to-heads won against the alternative proposed "
                  "at the same step", fontsize=12, color=MUTED)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=11, colors=MUTED)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.suptitle(
        f"Win rate per proposed addition — {len(rows)} pairs with n ≥ {MIN_N} "
        f"({covered} of {len(recs)} records); the line is 50%",
        fontsize=15, color=INK, x=0.008, ha="left", y=0.975,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "node_effects.png"
    fig.savefig(out, dpi=110, facecolor=GROUND)
    print(f"{len(rows)} pairs, {covered}/{len(recs)} records")
    print("wrote", out)


if __name__ == "__main__":
    main()
