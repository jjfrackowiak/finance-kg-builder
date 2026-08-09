"""Show that scoring against the run's current best manufactures a decline.

Candidates do not get worse as a run progresses — their mean AUC is flat across steps.
What changes is the yardstick: the acceptance gate makes the reference the maximum AUC
seen so far, which only ever ratchets upward. So the same candidate quality scores
progressively worse, and the share of "negative" candidates climbs from 62% to over 90%.

Any conclusion drawn from AUC-minus-current-best is reading this ratchet, not the
additions.

Writes reference_bias.png; embed with embed_figure.py.
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
GROUND, INK, MUTED, RULE, ACCENT = "#fbfbfa", "#14161a", "#878d96", "#e3e4e0", "#2a78d6"
NEG = "#c2521f"


def main() -> None:
    recs = [r for r in csv.DictReader(open(HERE / "additions.csv")) if r["delta_auc"]]
    for r in recs:
        r["auc"], r["step"] = float(r["auc"]), int(r["step"])
        r["delta"] = float(r["delta_auc"])
        r["reference"] = r["auc"] - r["delta"]

    by_step = defaultdict(list)
    for r in recs:
        by_step[r["step"]].append(r)
    steps = sorted(by_step)
    cand = [st.mean(x["auc"] for x in by_step[s]) for s in steps]
    ref = [st.mean(x["reference"] for x in by_step[s]) for s in steps]
    pct_neg = [100 * sum(1 for x in by_step[s] if x["delta"] < 0) / len(by_step[s]) for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=GROUND)
    fig.suptitle(
        "Scoring against the run's current best manufactures a decline that is not there",
        fontsize=15, color=INK, x=0.008, ha="left", y=0.975,
    )

    ax1.set_facecolor(GROUND)
    ax1.fill_between(steps, cand, ref, color=NEG, alpha=0.12, zorder=1)
    ax1.plot(steps, ref, "o-", color=NEG, lw=2, ms=8, zorder=3,
             label="reference: best AUC so far in the run")
    ax1.plot(steps, cand, "o-", color=ACCENT, lw=2, ms=8, zorder=3,
             label="mean AUC of the candidates proposed")
    ax1.set_title("The yardstick climbs; the candidates do not", fontsize=13,
                  color=INK, loc="left", pad=10)
    ax1.set_xlabel("evolution step", fontsize=12, color=MUTED)
    ax1.set_ylabel("AUC", fontsize=12, color=MUTED)
    ax1.set_ylim(0.44, 0.66)
    ax1.legend(fontsize=10.5, loc="center right", framealpha=0.9)

    ax2.set_facecolor(GROUND)
    ax2.bar(steps, pct_neg, color=NEG, alpha=0.85, width=0.62, zorder=3)
    ax2.axhline(50, color=INK, ls=":", lw=1.2, zorder=4)
    for s, v in zip(steps, pct_neg):
        ax2.annotate(f"{v:.0f}%", (s, v), ha="center", va="bottom",
                     xytext=(0, 4), textcoords="offset points",
                     fontsize=11, color=INK, family="monospace")
    ax2.set_title("So almost everything scores negative", fontsize=13,
                  color=INK, loc="left", pad=10)
    ax2.set_xlabel("evolution step", fontsize=12, color=MUTED)
    ax2.set_ylabel("% of candidates scoring below the reference", fontsize=12, color=MUTED)
    ax2.set_ylim(0, 108)

    for ax in (ax1, ax2):
        ax.set_xticks(steps)
        ax.tick_params(labelsize=11, colors=MUTED)
        ax.grid(axis="y", color=RULE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "reference_bias.png"
    fig.savefig(out, dpi=110, facecolor=GROUND)
    print("candidate AUC by step:", [f"{c:.3f}" for c in cand])
    print("reference by step:    ", [f"{r:.3f}" for r in ref])
    print("% negative by step:   ", [f"{p:.0f}" for p in pct_neg])
    print("wrote", out)


if __name__ == "__main__":
    main()
