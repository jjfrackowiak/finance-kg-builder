"""Regenerate every figure in `analyses/tsla_msft_200days_report.html`.

Every figure is a 1x2 (or 4x2) grid with **TSLA on the left and MSFT on the right**,
sharing y-limits within each row so the two panels can be read against each other by
eye. That side-by-side pairing is the whole point of this report: the interesting
quantity is not either ticker's number but whether the two halves agree.

Writes fig_*.png next to this file. Embed with `python embed_figures.py`.
"""

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
GROUND = "#fbfbfa"
INK, MUTED, RULE = "#14161a", "#878d96", "#e3e4e0"
ACCENT, POS, NEG = "#2a78d6", "#0f8f62", "#c2521f"
TICKERS = ("TSLA", "MSFT")
# TSLA ran on mixed articles, MSFT on MSFT-only. Colour encodes the ticker
# consistently in every figure so the eye can pair panels across sections.
COLOR = {"TSLA": ACCENT, "MSFT": "#8c5bd8"}
SUBTITLE = {"TSLA": "TSLA — mixed articles", "MSFT": "MSFT — MSFT-only articles"}
# Per-ticker between-run SD of final AUC, used as the scatter reference in the
# lift figure. Computed from results.csv in report.ipynb.

FACTORS = [
    ("lookback_days", "Lookback (days)", None),
    ("steps", "Evolution steps", None),
    ("chain_hops", "Chain hops", None),
    ("evolution_prompt", "Evolution prompt", ["default", "event_driven", "fundamental"]),
]


def load():
    rows = list(csv.DictReader(open(HERE / "results.csv")))
    for r in rows:
        for k in ("baseline_auc", "final_auc", "delta", "path_mean_norm", "brier_score"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        for k in ("lookback_days", "steps", "chain_hops"):
            r[k] = int(r[k])
    adds = list(csv.DictReader(open(HERE / "additions.csv")))
    for a in adds:
        a["auc"] = float(a["auc"])
        a["step"] = int(a["step"])
        a["delta_auc"] = float(a["delta_auc"]) if a["delta_auc"] not in ("", None) else None
        a["won"] = a["won"] == "True"
    return rows, adds


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(GROUND)
    if title:
        ax.set_title(title, fontsize=14, color=INK, loc="left", pad=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.5, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5, color=MUTED)
    ax.tick_params(labelsize=10.5, colors=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)


def suptitle(fig, text):
    fig.suptitle(text, fontsize=15.5, color=INK, x=0.011, ha="left", y=0.982)


def save(fig, name, rect=(0, 0, 1, 0.93)):
    fig.tight_layout(rect=rect)
    out = HERE / name
    fig.savefig(out, dpi=110, facecolor=GROUND)
    plt.close(fig)
    print("wrote", out.name)


# ---------------------------------------------------------------- figure 1
def fig_distribution(rows):
    """Final vs baseline AUC per run — the headline contrast, ticker by ticker."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), facecolor=GROUND, sharey=True)
    suptitle(fig, "Final structural AUC vs each run's own text baseline — 36 runs per ticker, sorted")
    for ax, t in zip(axes, TICKERS):
        sub = sorted([r for r in rows if r["ticker"] == t], key=lambda r: r["final_auc"])
        xs = np.arange(len(sub))
        fin = [r["final_auc"] for r in sub]
        base = [r["baseline_auc"] for r in sub]
        for x, f, b in zip(xs, fin, base):
            ax.plot([x, x], [b, f], color=POS if f > b else NEG, lw=1.6, alpha=0.55, zorder=1)
        ax.scatter(xs, base, s=26, color=MUTED, zorder=2, label="text baseline")
        ax.scatter(xs, fin, s=34, color=COLOR[t], zorder=3, label="final structural")
        ax.axhline(0.5, color=INK, ls=":", lw=1.1, zorder=0)
        m = st.mean(fin)
        ax.axhline(m, color=COLOR[t], ls="--", lw=1.1, alpha=0.7, zorder=0)
        ax.annotate(f"mean {m:.3f}", (0.4, m), fontsize=11, color=COLOR[t],
                    va="bottom", family="monospace",
                    bbox=dict(facecolor=GROUND, edgecolor="none", pad=1.5))
        beat = sum(1 for r in sub if r["delta"] > 0)
        style(ax, f"{SUBTITLE[t]}   ({beat}/{len(sub)} beat baseline)", "run (sorted by final AUC)",
              "AUC" if t == "TSLA" else None)
        ax.set_ylim(0.42, 0.82)
        ax.legend(fontsize=10.5, frameon=False, loc="upper left", labelcolor=MUTED)
    save(fig, "fig_distribution.png")


# ---------------------------------------------------------------- figure 2
def fig_dose_response(rows):
    """Realised path signal against final AUC. Flat in both halves."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), facecolor=GROUND, sharey=True)
    suptitle(fig, "Path-chain signal against final AUC — no dose-response in either ticker")
    for ax, t in zip(axes, TICKERS):
        sub = [r for r in rows if r["ticker"] == t]
        x = np.array([r["path_mean_norm"] for r in sub])
        y = np.array([r["final_auc"] for r in sub])
        for st_, mk in ((3, "o"), (5, "s"), (7, "^")):
            m = [r["steps"] == st_ for r in sub]
            ax.scatter(x[m], y[m], s=62, marker=mk, color=COLOR[t], alpha=0.8,
                       edgecolor="none", label=f"steps={st_}")
        slope, intercept = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 50)
        ax.plot(xr, slope * xr + intercept, color=INK, ls="--", lw=1.3, alpha=0.75)
        empty = int((x == 0).sum())
        style(ax, f"{SUBTITLE[t]}   slope {slope:+.3f}   ({empty} runs with an empty path block)",
              "path_mean_norm  (mean Euclidean length of the day's path vector)",
              "final AUC" if t == "TSLA" else None)
        ax.set_ylim(0.42, 0.82)
        ax.legend(fontsize=10.5, frameon=False, loc="upper right", labelcolor=MUTED)
    save(fig, "fig_dose_response.png")


# ---------------------------------------------------------------- figure 3
def fig_group_means(rows):
    """Four factors x two tickers, each against its own ticker's grand-mean band."""
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 17.5), facecolor=GROUND)
    grand = {t: st.mean(r["final_auc"] for r in rows if r["ticker"] == t) for t in TICKERS}
    sd = {t: st.stdev([r["final_auc"] for r in rows if r["ticker"] == t]) for t in TICKERS}
    suptitle(
        fig,
        f"Group means by factor — dotted line = that ticker's grand mean, band = ±1 SD across its 36 runs\n"
        f"TSLA {grand['TSLA']:.3f} ± {sd['TSLA']:.3f}        MSFT {grand['MSFT']:.3f} ± {sd['MSFT']:.3f}",
    )
    for row, (key, title, order) in enumerate(FACTORS):
        for col, t in enumerate(TICKERS):
            ax = axes[row][col]
            sub = [r for r in rows if r["ticker"] == t]
            groups = defaultdict(list)
            for r in sub:
                groups[r[key]].append(r["final_auc"])
            levels = order or sorted(groups, key=int)
            means = [st.mean(groups[k]) for k in levels]
            sds = [st.stdev(groups[k]) for k in levels]
            xs = range(len(levels))

            g, s = grand[t], sd[t]
            ax.set_facecolor(GROUND)
            ax.axhspan(g - s, g + s, color=COLOR[t], alpha=0.07, zorder=0)
            ax.axhline(g + s, color=COLOR[t], alpha=0.35, lw=0.9, zorder=1)
            ax.axhline(g - s, color=COLOR[t], alpha=0.35, lw=0.9, zorder=1)
            ax.axhline(g, color=INK, ls=":", lw=1.2, zorder=2)
            ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#c3c5bf", elinewidth=1.7,
                        capsize=5, capthick=1.7, zorder=3)
            ax.scatter(xs, means, s=105, color=COLOR[t], zorder=4)
            for x, m in zip(xs, means):
                ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points", xytext=(13, -4),
                            fontsize=11.5, color=INK, family="monospace", zorder=5,
                            bbox=dict(facecolor=GROUND, edgecolor="none", pad=1.5))
            n_per = len(sub) // len(levels)
            style(ax, f"{title} — {t}   (n={n_per} per level)", None,
                  "mean AUC ± SD" if col == 0 else None)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(levels, fontsize=11.5, color=MUTED, family="monospace")
            ax.set_xlim(-0.5, len(levels) - 0.35)
            ax.set_ylim(0.48, 0.74)
    save(fig, "fig_group_means.png", rect=(0, 0, 1, 0.962))


# ---------------------------------------------------------------- figure 4
def fig_reference_bias(adds):
    """The acceptance gate makes the reference the running maximum."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), facecolor=GROUND, sharey=True)
    suptitle(fig, "The reference climbs; the candidates do not — why AUC-minus-current-best cannot score an addition")
    for ax, t in zip(axes, TICKERS):
        sub = [a for a in adds if a["ticker"] == t]
        steps = sorted({a["step"] for a in sub})
        cand, ref, negshare = [], [], []
        for s in steps:
            g = [a for a in sub if a["step"] == s]
            cand.append(st.mean(a["auc"] for a in g))
            withref = [a for a in g if a["delta_auc"] is not None]
            ref.append(st.mean(a["auc"] - a["delta_auc"] for a in withref) if withref else float("nan"))
            negshare.append(100 * sum(1 for a in withref if a["delta_auc"] < 0) / len(withref) if withref else float("nan"))
        ax.plot(steps, cand, "o-", color=COLOR[t], lw=2, ms=8, label="mean candidate AUC")
        ax.plot(steps, ref, "s--", color=NEG, lw=2, ms=7, label="mean reference (best so far)")
        style(ax, SUBTITLE[t], "evolution step", "AUC" if t == "TSLA" else None)
        ax.set_ylim(0.44, 0.70)
        ax.set_xticks(steps)
        ax2 = ax.twinx()
        ax2.plot(steps, negshare, ":", color=MUTED, lw=1.8)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("% scored negative" if t == "MSFT" else None, fontsize=11.5, color=MUTED)
        ax2.tick_params(labelsize=10.5, colors=MUTED)
        for s in ("top", "left", "bottom"):
            ax2.spines[s].set_visible(False)
        ax2.spines["right"].set_color(RULE)
        ax.legend(fontsize=10.5, frameon=False, loc="lower left", labelcolor=MUTED)
    save(fig, "fig_reference_bias.png")


# ---------------------------------------------------------------- figure 5
def fig_win_rates(adds):
    """Win rate per (node, relationship), the reference-free ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4), facecolor=GROUND)
    suptitle(fig, "Win rate per proposed addition — how often it beat the alternative in the same step (line = 50%)")
    for ax, t in zip(axes, TICKERS):
        by_pair = defaultdict(list)
        for a in adds:
            if a["ticker"] == t and a["delta_auc"] is not None:
                by_pair[(a["node"], a["relationship"])].append(a["won"])
        stats = sorted(
            ((100 * sum(v) / len(v), k, len(v)) for k, v in by_pair.items() if len(v) >= 8),
            reverse=True,
        )
        # Relationship names run to 32 characters; left unabbreviated they push the
        # axes so far right that the neighbouring panel's title is clipped.
        labels = [f"{n} · {r[:24] + '…' if len(r) > 25 else r}" for _, (n, r), _ in stats]
        rates = [s for s, _, _ in stats]
        ns = [n for _, _, n in stats]
        ys = np.arange(len(stats))[::-1]
        ax.barh(ys, rates, color=[POS if r > 50 else NEG if r < 50 else MUTED for r in rates],
                alpha=0.85, height=0.68)
        ax.axvline(50, color=INK, ls="--", lw=1.2)
        for y, r, n in zip(ys, rates, ns):
            ax.annotate(f"{r:.0f}%  (n={n})", (r, y), textcoords="offset points", xytext=(6, -4),
                        fontsize=10, color=INK, family="monospace")
        style(ax, SUBTITLE[t], f"win rate (%)   —   {len(stats)} pairs proposed ≥8 times")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9.5, color=MUTED)
        ax.set_xlim(0, 128)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", lw=0)
        ax.grid(axis="x", color=RULE, lw=0.8)
    save(fig, "fig_win_rates.png")


# ---------------------------------------------------------------- figure 6
def fig_acceptance(adds):
    """Acceptance rate by step — the ontology saturates after step 1."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), facecolor=GROUND, sharey=True)
    suptitle(fig, "Acceptance rate by evolution step — the ontology saturates after step 1")
    for ax, t in zip(axes, TICKERS):
        sub = [a for a in adds if a["ticker"] == t]
        steps = sorted({a["step"] for a in sub})
        rates, ns = [], []
        for s in steps:
            g = [a for a in sub if a["step"] == s]
            rates.append(100 * sum(1 for a in g if a["status"] == "accepted") / len(g))
            ns.append(len(g))
        ax.bar(steps, rates, color=COLOR[t], alpha=0.85, width=0.62)
        for s, r, n in zip(steps, rates, ns):
            ax.annotate(f"{r:.0f}%\nn={n}", (s, r), textcoords="offset points", xytext=(0, 5),
                        fontsize=10.5, color=INK, ha="center", family="monospace")
        style(ax, SUBTITLE[t], "evolution step", "accepted (%)" if t == "TSLA" else None)
        ax.set_xticks(steps)
        ax.set_ylim(0, 62)
    save(fig, "fig_acceptance.png")


# ---------------------------------------------------------------- figure 7
def fig_delta_vs_noise(rows):
    """Every run's lift over its own text baseline — the report's one positive result.

    Filename kept for continuity with earlier drafts; the band it once carried (the
    sampling SE of a single AUC) was the wrong reference for this comparison and has
    been dropped in favour of the sign test, which uses the runs as replicates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), facecolor=GROUND, sharey=True)
    sds = {t: st.stdev([r["final_auc"] for r in rows if r["ticker"] == t]) for t in TICKERS}
    suptitle(fig, "Per-run lift over each run's own text baseline — shaded band is ±1 SD of that "
                  f"ticker's 36 final AUCs (TSLA {sds['TSLA']:.3f}, MSFT {sds['MSFT']:.3f})")
    for ax, t in zip(axes, TICKERS):
        sub = sorted([r for r in rows if r["ticker"] == t], key=lambda r: r["delta"])
        d = [r["delta"] for r in sub]
        sd = st.stdev([r["final_auc"] for r in rows if r["ticker"] == t])
        xs = np.arange(len(d))
        # Shade +/- the between-run SD of AUC: the scatter any single run sits within.
        ax.axhspan(-sd, sd, color=MUTED, alpha=0.14, zorder=0)
        ax.bar(xs, d, color=[POS if v > 0 else NEG for v in d], alpha=0.85, width=0.72, zorder=2)
        ax.axhline(0, color=INK, lw=1.2, zorder=3)
        m = st.mean(d)
        ax.axhline(m, color=COLOR[t], ls="--", lw=1.5, zorder=4)
        ax.annotate(f"mean {m:+.3f}", (0.3, m), fontsize=11, color=COLOR[t], va="bottom",
                    family="monospace", bbox=dict(facecolor=GROUND, edgecolor="none", pad=1.5))
        won = sum(1 for v in d if v > 0)
        style(ax, f"{SUBTITLE[t]}   {won}/{len(d)} beat text   mean {m:+.3f}",
              "run (sorted by lift)", "final AUC − text AUC" if t == "TSLA" else None)
        ax.set_ylim(-0.20, 0.30)
    save(fig, "fig_delta_vs_noise.png")


# ---------------------------------------------------------------- figure 8
def fig_step_curve(adds):
    """Running best AUC per step, with the per-step candidate mean alongside it.

    The running best climbs by construction — it is a maximum over an accumulating
    pool. The per-step mean does not accumulate, so it rises only if the proposals
    themselves are getting better. Plotting both makes the difference visible
    without needing a null.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), facecolor=GROUND, sharey=True)
    suptitle(fig, "Best ontology AUC so far, and the mean candidate proposed at each step")
    for ax, t in zip(axes, TICKERS):
        recs = [a for a in adds if a["ticker"] == t]
        steps = sorted({a["step"] for a in recs})

        # mean AUC of candidates proposed AT that step — not cumulative
        per_step = [st.mean([a["auc"] for a in recs if a["step"] == s_]) for s_ in steps]

        # running best within each run, then averaged across runs
        by_run = defaultdict(list)
        for a in recs:
            by_run[a["run_id"]].append(a)
        acc = defaultdict(list)
        for rows in by_run.values():
            cur = float("-inf")
            for s_ in sorted({r["step"] for r in rows}):
                cur = max(cur, max(r["auc"] for r in rows if r["step"] == s_))
                acc[s_].append(cur)
        running = [st.mean(acc[s_]) for s_ in steps]
        n_by_step = [len(acc[s_]) for s_ in steps]

        ax.plot(steps, running, "o-", color=COLOR[t], lw=2.2, ms=7,
                label="best found so far (accumulates)")
        ax.plot(steps, per_step, "s--", color=NEG, lw=2, ms=6,
                label="mean candidate at this step")
        ax.axhline(0.5, color=INK, ls=":", lw=1.1, zorder=0)
        for s_, n in zip(steps, n_by_step):
            ax.annotate(f"n={n}", (s_, 0.662), fontsize=9, color=MUTED,
                        ha="center", family="monospace")
        style(ax, SUBTITLE[t], "evolution step", "AUC" if t == "TSLA" else None)
        ax.set_xticks(steps)
        ax.set_ylim(0.45, 0.672)
        ax.legend(fontsize=10, frameon=False, loc="lower left", labelcolor=MUTED)
    save(fig, "fig_step_curve.png")


def main():
    rows, adds = load()
    fig_distribution(rows)
    fig_dose_response(rows)
    fig_group_means(rows)
    fig_reference_bias(adds)
    fig_win_rates(adds)
    fig_acceptance(adds)
    fig_delta_vs_noise(rows)
    fig_step_curve(adds)


if __name__ == "__main__":
    main()
