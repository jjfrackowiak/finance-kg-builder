"""Pairwise-balance matrix for the 72-config sweep design.

For every pair of the five design factors (lookback_days, steps, evolution_prompt,
min_chain_hops, ticker), shows the observed co-occurrence count in every level
combination cell, colored by deviation from the uniform "balanced" expectation
(total_configs / n_cells for that pair). Built directly from the actual
sweep_runs/run_*.json files, not simulated -- reflects the real, known imperfection
(lookback x evolution_prompt has 4 of 16 cells at 0, documented in
planning/EXPERIMENT_STRATEGY.md) alongside the nine pairs that are perfectly balanced.
"""
import json
import glob
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = "/Users/mac/Desktop/Research and Study/Doktorat/downstream-finance-graph/finance-kg-builder"

configs = []
for f in sorted(glob.glob(f"{REPO}/sweep_runs/run_*.json")):
    configs.extend(json.load(open(f)))
assert len(configs) == 72, len(configs)

FACTORS = ["lookback_days", "steps", "evolution_prompt", "min_chain_hops", "ticker"]
LABELS = {
    "lookback_days": "lookback\ndays",
    "steps": "steps",
    "evolution_prompt": "prompt",
    "min_chain_hops": "chain\nhops",
    "ticker": "ticker",
}

PROMPT_ORDER = [
    "default",
    "resources/fundamental_prompt_template.txt",
    "resources/event_driven_prompt_template.txt",
    "resources/macro_context_prompt_template.txt",
]
PROMPT_SHORT = {
    "default": "defa",
    "resources/fundamental_prompt_template.txt": "fund",
    "resources/event_driven_prompt_template.txt": "even",
    "resources/macro_context_prompt_template.txt": "macr",
}

LEVELS = {
    "lookback_days": [3, 8, 10, 20],
    "steps": [3, 5, 7],
    "evolution_prompt": PROMPT_ORDER,
    "min_chain_hops": [3, 5, 6],
    "ticker": ["MSFT", "TSLA"],
}


def tick_label(factor, level):
    if factor == "evolution_prompt":
        return PROMPT_SHORT[level]
    return str(level)


n = len(FACTORS)
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(n, n, hspace=0.5, wspace=0.35)

max_abs_dev = 0.0
cell_data = {}
for i, fa in enumerate(FACTORS):
    for j, fb in enumerate(FACTORS):
        if i == j:
            continue
        la, lb = LEVELS[fa], LEVELS[fb]
        counts = Counter((c[fa], c[fb]) for c in configs)
        grid = np.array([[counts.get((a, b), 0) for b in lb] for a in la], dtype=float)
        expected = grid.sum() / grid.size
        dev = grid - expected
        cell_data[(i, j)] = (grid, dev)
        max_abs_dev = max(max_abs_dev, np.abs(dev).max())

cmap = plt.cm.RdBu_r
vlim = max(max_abs_dev, 0.5)

for i, fa in enumerate(FACTORS):
    for j, fb in enumerate(FACTORS):
        ax = fig.add_subplot(gs[i, j])
        if i == j:
            ax.set_facecolor("#f2f0eb")
            ax.text(0.5, 0.5, LABELS[fa], ha="center", va="center", fontsize=13,
                     color="#888888", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        grid, dev = cell_data[(i, j)]
        la, lb = LEVELS[fa], LEVELS[fb]
        ax.imshow(dev, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = int(grid[r, c])
                color = "white" if abs(dev[r, c]) > vlim * 0.55 else "#333333"
                weight = "bold" if val == 0 else "normal"
                ax.text(c, r, str(val), ha="center", va="center", fontsize=9,
                         color=color, fontweight=weight)
        ax.set_xticks(range(len(lb)))
        ax.set_xticklabels([tick_label(fb, v) for v in lb], fontsize=7.5)
        ax.set_yticks(range(len(la)))
        ax.set_yticklabels([tick_label(fa, v) for v in la], fontsize=7.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

n_missing_pairs = sum(1 for (i, j), (g, d) in cell_data.items() if i < j and (g == 0).any())
max_dev_val = max(np.abs(d).max() for g, d in cell_data.values())
fig.suptitle(
    "Pairwise balance across the 72-config sweep design\n"
    f"cell = observed co-occurrence count; color = deviation from the uniform-balanced expectation "
    f"(max |deviation| = {max_dev_val:.1f}, in the lookback×prompt pair only)",
    fontsize=13, y=0.985,
)

legend_patch = mpatches.Patch(facecolor="#f2f0eb", label="diagonal (factor label, not a pair)")
fig.legend(handles=[legend_patch], loc="lower center", bbox_to_anchor=(0.5, 0.01),
           framealpha=0, fontsize=9)

OUT = f"{REPO}/manuscript/figures/balance_matrix"
fig.savefig(OUT + ".png", bbox_inches="tight", dpi=200)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print(f"Saved {OUT}.png and .pdf")
print(f"Pairs with a missing cell: {n_missing_pairs} (expected: 1, lookback x prompt)")
