"""Effect of each proposed (node, relationship) addition, measured without reference bias.

The obvious measure — each candidate's AUC minus the run's current best — is unusable:
the acceptance gate makes the reference the maximum seen so far, so almost any candidate
scores negative whatever its quality. That is regression to the maximum, not evidence.

Within a single step the two candidates are built on the *same* graph and scored on the
same days, so comparing them to each other carries no reference bias at all. Each
candidate's effect is its AUC minus the mean AUC of its own step. Positive means it beat
the alternative proposed alongside it. The measure is zero-sum by construction, so it
ranks additions against each other and says nothing about whether additions help in
absolute terms — that needs the base graph scored on its own, which the pipeline does not
currently do.

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


def within_step_effects(path: Path) -> list:
    recs = [r for r in csv.DictReader(open(path)) if r["delta_auc"]]
    for r in recs:
        r["auc"] = float(r["auc"])
    steps = defaultdict(list)
    for r in recs:
        steps[(r["run_id"], r["step"])].append(r)
    for group in steps.values():
        mean = st.mean(x["auc"] for x in group)
        for x in group:
            x["within"] = x["auc"] - mean
    return recs


def main() -> None:
    recs = within_step_effects(HERE / "additions.csv")
    groups = defaultdict(list)
    for r in recs:
        groups[(r["node"], r["relationship"])].append(r["within"])

    rows = sorted(
        ((st.mean(v), k, len(v)) for k, v in groups.items() if len(v) >= MIN_N),
        reverse=True,
    )
    covered = sum(n for _, _, n in rows)

    labels = [f"{node}  —{rel}→" for _, (node, rel), _ in rows]
    means = [m for m, _, _ in rows]
    counts = [n for _, _, n in rows]

    fig, ax = plt.subplots(figsize=(13.5, 7.2), facecolor=GROUND)
    ax.set_facecolor(GROUND)
    ys = range(len(rows))
    ax.barh(list(ys), means, color=[POS if m > 0 else NEG for m in means],
            alpha=0.85, height=0.68, zorder=3)
    ax.axvline(0, color=INK, lw=1.1, zorder=4)

    for y, (m, n) in enumerate(zip(means, counts)):
        off = 0.0016 if m > 0 else -0.0016
        ax.annotate(f"{m:+.3f}  (n={n})", (m, y), va="center",
                    ha="left" if m > 0 else "right",
                    xytext=(4 if m > 0 else -4, 0), textcoords="offset points",
                    fontsize=11, color=INK, family="monospace")
        del off

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=11, color=INK, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("AUC relative to the alternative proposed at the same step",
                  fontsize=12, color=MUTED)
    ax.set_xlim(-0.075, 0.085)
    ax.tick_params(axis="x", labelsize=11, colors=MUTED)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.suptitle(
        f"Effect per proposed addition — {len(rows)} pairs with n ≥ {MIN_N} "
        f"({covered} of {len(recs)} records)",
        fontsize=15, color=INK, x=0.008, ha="left", y=0.975,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "node_effects.png"
    fig.savefig(out, dpi=110, facecolor=GROUND)
    print(f"{len(rows)} pairs, {covered}/{len(recs)} records")
    print("wrote", out)


if __name__ == "__main__":
    main()
