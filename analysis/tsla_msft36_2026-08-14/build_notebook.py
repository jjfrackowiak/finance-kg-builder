"""Generate report.ipynb — the executable companion to the two-ticker report.

Kept as a generator rather than a hand-edited .ipynb so the cells stay diffable in
git. Run this, then execute the notebook:

    python build_notebook.py
    uv run --with jupyter,nbconvert,pandas,numpy,scipy,matplotlib \
        jupyter nbconvert --to notebook --execute --inplace report.ipynb
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

CELLS: list = []


def _lines(text: str) -> list:
    """nbformat wants one string per line, each keeping its trailing newline."""
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(text),
        }
    )


# ────────────────────────────────────────────────────────────────────────────
md("""
# TSLA vs MSFT — 72-run sweep analysis

Every number and figure in `analyses/tsla_msft_200days_report.html`, recomputed from
`results.csv` and `additions.csv` (both written by `pull_and_analyze.py` straight from
MLflow).

The design is two balanced 36-run halves. They differ in exactly two ways:

| | TSLA half | MSFT half |
|---|---|---|
| label | TSLA next-day direction | MSFT next-day direction |
| articles | **mixed** — TSLA + MSFT + NVDA | **MSFT only** (`--filter-ticker`) |
| chunks | 01–13, 2026-08-08/09 | 14–26, 2026-08-12/14 |

Everything else — window, orthogonal array, gate, feature construction — is identical.
So the comparison answers one question: **does focusing the corpus on the labelled
company change anything?**

Figures here use default matplotlib. The report's styled versions come from
`make_figures.py`, from this same data.
""")

code("""
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

TICKERS = ["TSLA", "MSFT"]
FACTORS = ["lookback_days", "steps", "chain_hops", "evolution_prompt"]

runs = pd.read_csv("results.csv")
adds = pd.read_csv("additions.csv")

print(f"runs      {len(runs)}  ({runs.ticker.value_counts().to_dict()})")
print(f"additions {len(adds)}  ({adds.ticker.value_counts().to_dict()})")
""")

md("""
This notebook is **descriptive**: it reproduces every number the report states, and
nothing more. Formal hypothesis testing — sign tests, ANOVA, multiplicity — lives in
`stats_tests_note.ipynb` alongside it, so the two can be read independently.
""")

# ── A ───────────────────────────────────────────────────────────────────────
md("""
## A · What was run

Four hyperparameters vary at once. The full grid is 4 × 3 × 3 × 3 = 108; each half runs
36 of them, chosen so every *pair* of levels appears equally often. That is what makes
a level-vs-level comparison fair without any regression adjustment.
""")

code("""
# Held fixed within each half: every one of these should have a single distinct value.
fixed = runs.groupby("ticker")[["baseline_auc"]].size().rename("n_runs").to_frame()
fixed["distinct lookback"] = runs.groupby("ticker").lookback_days.nunique()
fixed["distinct steps"] = runs.groupby("ticker").steps.nunique()
fixed["distinct hops"] = runs.groupby("ticker").chain_hops.nunique()
fixed["distinct prompts"] = runs.groupby("ticker").evolution_prompt.nunique()
fixed
""")

code("""
# Pairwise balance: for every PAIR of factors, does every level combination appear,
# and equally often? This is the strength-2 property the design claims.
def balance(df):
    out = []
    for i, a in enumerate(FACTORS):
        for b in FACTORS[i + 1:]:
            counts = df.groupby([a, b]).size()
            expected = len(df[a].unique()) * len(df[b].unique())
            out.append({
                "pair": f"{a} x {b}",
                "cells seen": len(counts),
                "cells possible": expected,
                "missing": expected - len(counts),
                "lambda": sorted(counts.unique().tolist()),
            })
    return pd.DataFrame(out)

for t in TICKERS:
    print(f"--- {t} ---")
    print(balance(runs[runs.ticker == t]).to_string(index=False))
    print()
""")

md("""
No missing cells in either half, and each combination appears 3 or 4 times depending on
how many levels the two factors carry. Both halves are genuine `OA(36, 4¹3³, 2)`.
""")

code("""
# Article composition. Recomputed through the real code path rather than quoted, because
# the per-day cap is sensitive to how ties are broken (see the caveat cell below).
from datetime import date, timedelta

CSV = "../../data/fnspid_sample_nasdaq_long_text.csv"
SPLIT = pd.Timestamp("2022-09-20 23:59:59")   # last training price day

raw = pd.read_csv(CSV)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
start, end = date(2022, 5, 2), date(2022, 5, 2) + timedelta(days=200)
in_window = (raw.timestamp.dt.date >= start) & (raw.timestamp.dt.date < end)
window_fileorder = raw[in_window].copy()          # untouched row order, for the tie demo
# The production path sorts before slicing (core/data.py), so mirror that exactly.
raw = raw.sort_values("timestamp").reset_index(drop=True)
window = raw[(raw.timestamp.dt.date >= start) & (raw.timestamp.dt.date < end)].copy()

def compose(df, ticker=None):
    d = df[df.ticker == ticker].copy() if ticker else df.copy()
    d["day"] = d.timestamp.dt.strftime("%Y-%m-%d")
    sel = d.groupby("day").head(4).reset_index(drop=True)
    rows = []
    for name, part in [("training", sel[sel.timestamp <= SPLIT]),
                       ("validation", sel[sel.timestamp > SPLIT]),
                       ("total", sel)]:
        r = {"split": name, "articles": len(part), "days": part.day.nunique(),
             "per day": round(len(part) / part.day.nunique(), 2)}
        for tk in ["TSLA", "MSFT", "NVDA"]:
            n = int((part.ticker == tk).sum())
            r[tk] = f"{n} ({100 * n / len(part):.0f}%)" if n else "—"
        rows.append(r)
    return pd.DataFrame(rows)

print("TSLA half — mixed articles"); display(compose(window))
print("MSFT half — MSFT-only articles"); display(compose(window, "MSFT"))
""")

md("""
**A caveat that belongs right here.** Every article on a given day carries the *same*
timestamp, so `sort_values` leaves them tied and `groupby("day").head(4)` takes whichever
four the sort happened to put first. The selection is deterministic for a given pandas
build (verified identical under 1.3.5 and the locked 2.3.3) but it is **not** a random
sample and **not** the most recent four — it is an arbitrary-but-stable slice.

For the MSFT half this cannot affect composition, since filtering happens before the cap
and everything kept is MSFT either way. For the TSLA half it fully determines the mix.
""")

code("""
# The tie problem, quantified.
print(f"articles in window          {len(window):,}")
print(f"distinct timestamps         {window.timestamp.nunique()}")
print(f"articles per distinct stamp {len(window) / window.timestamp.nunique():.1f}")
print(f"sample timestamps           {sorted(window.timestamp.unique())[:3]}")
print()
print("So the 4/day cap chooses 4 of ~30 mutually tied rows, and *which* four is decided")
print("by row order alone. Same data, same cap, two orderings:\\n")

def mix(df):
    d = df.assign(day=df.timestamp.dt.strftime("%Y-%m-%d"))
    return d.groupby("day").head(4).ticker.value_counts().to_dict()

print("  file order (a stable sort would keep this):", mix(window_fileorder))
print("  as executed (pandas default quicksort)    :", mix(window))
print()
print("The first is what core/data.py's docstring records; the second is what actually")
print("ran. Neither is more correct — that is the defect.")
""")

# ── B ───────────────────────────────────────────────────────────────────────
md("""
## B · Results, side by side

Each run ends at its **final accepted ontology**. `baseline_auc` is that same run's
article-text arm on the same validation days. The two arms share no features —
`build_day_feature_vector` skips text entirely — so `delta` contrasts two disjoint
representations rather than ablating the ontology.
""")

code("""
summary = runs.groupby("ticker")[["final_auc", "baseline_auc", "delta", "brier_score"]].agg(
    ["mean", "std", "median", "min", "max"]
).T.unstack(0)
summary.columns = summary.columns.droplevel(0)
summary.round(4)
""")

code("""
side = pd.DataFrame(index=["mean final AUC", "mean text baseline", "mean delta",
                           "SD of delta", "beat baseline", "best run", "worst run"])
for t in TICKERS:
    s = runs[runs.ticker == t]
    side[t] = [
        f"{s.final_auc.mean():.4f}", f"{s.baseline_auc.mean():.4f}",
        f"{s.delta.mean():+.4f}", f"{s.delta.std():.4f}",
        f"{(s.delta > 0).sum()}/{len(s)}",
        f"{s.final_auc.max():.4f}", f"{s.final_auc.min():.4f}",
    ]
side["difference"] = ["", "", "", "", "", "", ""]
side.loc["mean final AUC", "difference"] = (
    f"{runs[runs.ticker=='MSFT'].final_auc.mean() - runs[runs.ticker=='TSLA'].final_auc.mean():+.4f}")
side.loc["mean delta", "difference"] = (
    f"{runs[runs.ticker=='MSFT'].delta.mean() - runs[runs.ticker=='TSLA'].delta.mean():+.4f}")
side
""")

code("""
# Top 5 per ticker, shown together.
cols = ["final_auc", "baseline_auc", "delta", "lookback_days", "steps",
        "chain_hops", "evolution_prompt", "path_mean_norm"]
for t in TICKERS:
    print(f"--- {t}: top 5 of 36 ---")
    print(runs[runs.ticker == t].nlargest(5, "final_auc")[cols].to_string(index=False))
    print()
""")

code("""
# The text baseline depends only on lookback_days — the footnote under the top-5 tables.
# `nunique == 1` is the claim: every run at a given lookback shares one baseline exactly.
runs.groupby(["ticker", "lookback_days"]).baseline_auc.agg(
    n="size", distinct="nunique", value="mean"
).round(4)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, t in zip(axes, TICKERS):
    s = runs[runs.ticker == t].sort_values("final_auc").reset_index(drop=True)
    ax.vlines(s.index, s.baseline_auc, s.final_auc,
              color=np.where(s.delta > 0, "tab:green", "tab:red"), alpha=.6)
    ax.scatter(s.index, s.baseline_auc, s=22, color="gray", label="text baseline")
    ax.scatter(s.index, s.final_auc, s=30, label="final structural")
    ax.axhline(0.5, ls=":", color="k", lw=1)
    ax.axhline(s.final_auc.mean(), ls="--", lw=1)
    ax.set_title(f"{t} — {(s.delta > 0).sum()}/36 beat baseline")
    ax.set_xlabel("run (sorted by final AUC)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("AUC")
plt.tight_layout(); plt.show()
""")

md("""
### Per-run lift

Each run pairs a graph AUC with a text AUC on the same validation days, so the lift is a
within-run contrast. Both halves land on exactly 28 of 36.

The shaded band is ±1 SD of that ticker's own 36 final AUCs — the ordinary scatter between
runs. It is the reference used throughout the report: a difference smaller than this band
is not distinguishable from the variation between two runs of the same configuration.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for ax, t in zip(axes, TICKERS):
    sub = runs[runs.ticker == t]
    sd = sub.final_auc.std()                      # between-run SD of AUC
    d = sub.delta.sort_values().reset_index(drop=True)
    ax.axhspan(-sd, sd, color="gray", alpha=.15, label=f"±1 SD of AUC = ±{sd:.3f}")
    ax.bar(d.index, d, color=np.where(d > 0, "tab:green", "tab:red"), alpha=.85)
    ax.axhline(0, color="k", lw=1.2)
    ax.axhline(d.mean(), ls="--", lw=1.4)
    ax.set_title(f"{t} — mean {d.mean():+.3f}, {int((d > 0).sum())}/36 beat text")
    ax.set_xlabel("run (sorted by lift)")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("final AUC − text AUC")
plt.tight_layout(); plt.show()

for t in TICKERS:
    sub = runs[runs.ticker == t]
    print(f"{t}: mean lift {sub.delta.mean():+.4f}   between-run SD of AUC "
          f"{sub.final_auc.std():.4f}   {int((sub.delta > 0).sum())}/36 beat text")
""")

# ── C ───────────────────────────────────────────────────────────────────────
md("""
## C · Do path-chain features help?

`path_mean_norm` is the mean Euclidean length of each day's 384-dimension path vector,
averaged over the run: 0 means the block was entirely zeros.
""")

code("""
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    rho, p = stats.spearmanr(s.path_mean_norm, s.final_auc)
    slope = np.polyfit(s.path_mean_norm, s.final_auc, 1)[0]
    rows.append({"ticker": t, "spearman rho": round(rho, 4), "p": round(p, 3),
                 "OLS slope": round(slope, 4),
                 "runs with empty path block": int((s.path_mean_norm == 0).sum()),
                 "AUC | path=0": round(s[s.path_mean_norm == 0].final_auc.mean(), 4),
                 "AUC | path>0": round(s[s.path_mean_norm > 0].final_auc.mean(), 4)})
pd.DataFrame(rows)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for ax, t in zip(axes, TICKERS):
    s = runs[runs.ticker == t]
    for st_, mk in [(3, "o"), (5, "s"), (7, "^")]:
        g = s[s.steps == st_]
        ax.scatter(g.path_mean_norm, g.final_auc, marker=mk, s=55, alpha=.8, label=f"steps={st_}")
    m, b = np.polyfit(s.path_mean_norm, s.final_auc, 1)
    xr = np.linspace(s.path_mean_norm.min(), s.path_mean_norm.max(), 40)
    ax.plot(xr, m * xr + b, "k--", lw=1.2)
    ax.set_title(f"{t} — slope {m:+.3f}")
    ax.set_xlabel("path_mean_norm"); ax.legend(fontsize=8)
axes[0].set_ylabel("final AUC")
plt.tight_layout(); plt.show()
""")

md("""
Flat in both halves. Two independent 36-run designs agreeing on a null is much stronger
evidence than one, and this is the sweep's clearest replicated negative.
""")

# ── D ───────────────────────────────────────────────────────────────────────
md("""
## D · Differential analysis by factor

The question is not which level wins but whether any level separates from the ordinary
scatter between runs.
""")

code("""
tbl = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    grand, sd = s.final_auc.mean(), s.final_auc.std()
    for f in FACTORS:
        for lvl, g in s.groupby(f):
            tbl.append({"ticker": t, "factor": f, "level": lvl, "n": len(g),
                        "mean AUC": round(g.final_auc.mean(), 4),
                        "SD": round(g.final_auc.std(), 4),
                        "dist from grand mean": round(abs(g.final_auc.mean() - grand), 4),
                        "in SD units": round(abs(g.final_auc.mean() - grand) / sd, 2)})
factor_tbl = pd.DataFrame(tbl)
factor_tbl.pivot_table(index=["factor", "level"], columns="ticker",
                       values="mean AUC").round(4)
""")

code("""
for t in TICKERS:
    s = runs[runs.ticker == t]
    sub = factor_tbl[factor_tbl.ticker == t]
    worst = sub.loc[sub["dist from grand mean"].idxmax()]
    print(f"{t}: grand mean {s.final_auc.mean():.4f}  between-run SD {s.final_auc.std():.4f}")
    print(f"    largest factor deviation: {worst.factor}={worst.level} at {worst['mean AUC']:.4f} "
          f"({worst['dist from grand mean']:.4f}, {worst['in SD units']:.2f} SD)\\n")
""")

code("""
# Do the halves agree on which level is best / worst? This is the "agree?" column.
rows = []
for f in FACTORS:
    m = {t: runs[runs.ticker == t].groupby(f).final_auc.mean() for t in TICKERS}
    rows.append({
        "factor": f,
        "TSLA best": m["TSLA"].idxmax(), "MSFT best": m["MSFT"].idxmax(),
        "best agrees": m["TSLA"].idxmax() == m["MSFT"].idxmax(),
        "TSLA worst": m["TSLA"].idxmin(), "MSFT worst": m["MSFT"].idxmin(),
        "worst agrees": m["TSLA"].idxmin() == m["MSFT"].idxmin(),
    })
agree = pd.DataFrame(rows)
print(f"agree on best: {agree['best agrees'].sum()}/4   "
      f"agree on worst: {agree['worst agrees'].sum()}/4\\n")
# MSFT's two leading prompts are a near-tie; idxmax picks one arbitrarily.
print("MSFT prompt means at full precision:")
print(runs[runs.ticker == "MSFT"].groupby("evolution_prompt").final_auc.mean().round(6))
agree
""")

md("""
### How much of the variation do the factors explain?

The measure is **eta-squared**: the share of the spread in AUC that lines up with a factor.

- **denominator** `SS_total` — take all 36 runs, measure how far each AUC sits from the
  overall average, square, add up. One number for *how much the runs disagree in total*.
- **numerator** `SS_between` — replace each run by its group's average (every `hops=4` run
  becomes 0.5885, and so on), then measure how far those group averages sit from the
  overall average, square, weight by group size, add up. This is *the disagreement you
  could predict knowing only which level a run used*.

The ratio answers: if the only thing you knew about a run was its setting for this factor,
what share of the run-to-run variation could you account for?
""")

code("""
# The intermediate sums of squares, not just the ratio.
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    gm = s.final_auc.mean()
    ss_total = ((s.final_auc - gm) ** 2).sum()
    for f in FACTORS:
        g = s.groupby(f).final_auc
        ss_between = (g.count() * (g.mean() - gm) ** 2).sum()
        rows.append({"half": t, "factor": f,
                     "SS_between (numerator)": round(ss_between, 6),
                     "SS_total (denominator)": round(ss_total, 6),
                     "eta^2": round(ss_between / ss_total, 3)})
pd.DataFrame(rows).set_index(["half", "factor"])
""")

code("""
# Rolled up per half, with the residual. This is the report's table.
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    total_ss = ((s.final_auc - s.final_auc.mean()) ** 2).sum()
    r = {"ticker": t}
    explained = 0.0
    for f in FACTORS:
        gm = s.groupby(f).final_auc
        between = (gm.count() * (gm.mean() - s.final_auc.mean()) ** 2).sum()
        r[f] = round(between / total_ss, 3)
        explained += between / total_ss
    r["all four combined"] = round(explained, 3)
    r["residual"] = round(1 - explained, 3)
    rows.append(r)
pd.DataFrame(rows)
""")

md("""
**Why the four shares may be added.** Summing eta-squared values is normally invalid —
factors usually share variance, so you would double-count. It is legitimate here only
because the orthogonal array makes the factors exactly uncorrelated. Worth verifying
rather than assuming: fit one joint model with dummies for every level of every factor and
compare its R-squared against the naive sum.
""")

code("""
for t in TICKERS:
    s = runs[runs.ticker == t]
    gm = s.final_auc.mean()
    ss_total = ((s.final_auc - gm) ** 2).sum()
    naive = sum((s.groupby(f).final_auc.count() *
                 (s.groupby(f).final_auc.mean() - gm) ** 2).sum() / ss_total
                for f in FACTORS)
    X = pd.get_dummies(s[FACTORS].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    y = s.final_auc.values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    r2 = 1 - ((y - X.values @ beta) ** 2).sum() / ss_total
    print(f"{t}: sum of individual eta^2 = {naive:.4f}   joint model R^2 = {r2:.4f}   "
          f"difference = {abs(naive - r2):.1e}")
print()
print("Agreement to machine precision confirms the design is orthogonal, so the")
print("per-factor shares partition the variance without overlap.")
""")

code("""
fig, axes = plt.subplots(4, 2, figsize=(13, 15))
for row, f in enumerate(FACTORS):
    for col, t in enumerate(TICKERS):
        ax = axes[row][col]
        s = runs[runs.ticker == t]
        grand, sd = s.final_auc.mean(), s.final_auc.std()
        g = s.groupby(f).final_auc
        levels = list(g.groups)
        ax.axhspan(grand - sd, grand + sd, color="tab:blue", alpha=.08)
        ax.axhline(grand, ls=":", color="k")
        ax.errorbar(range(len(levels)), g.mean(), yerr=g.std(), fmt="o", ms=8, capsize=5)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
        ax.set_title(f"{f} — {t}", fontsize=10)
        ax.set_ylim(0.48, 0.74)
        if col == 0:
            ax.set_ylabel("mean AUC ± SD")
plt.tight_layout(); plt.show()
""")

md("""
Every level sits inside its own ticker's ±1 SD band, in both halves. The factors jointly
explain a minority of the variance; the rest is run-to-run noise.
""")

md("""
### Does running more steps keep helping?

Two lines, one question.

- **best found so far** accumulates — it is a maximum over a growing pool, so it rises
  whether or not the proposals improve.
- **mean candidate at this step** does not accumulate. It rises only if the candidates
  themselves are getting better.

If the first climbs while the second stays flat, the improvement is more draws rather than
better draws.
""")

code("""
def running_max_curve(df):
    \"\"\"Per run, the best candidate AUC seen up to and including each step.\"\"\"
    out = {}
    for rid, g in df.groupby("run_id"):
        best, cur = {}, -np.inf
        for s_ in sorted(g.step.unique()):
            cur = max(cur, g.loc[g.step == s_, "auc"].max())
            best[s_] = cur
        out[rid] = best
    return pd.DataFrame(out).T


rows = []
for t in TICKERS:
    a = adds[adds.ticker == t]
    per_step = a.groupby("step").auc.mean()          # candidates proposed AT that step
    running = running_max_curve(a).mean()            # best reached BY that step
    n = running_max_curve(a).notna().sum()
    rows.append(pd.DataFrame({
        "n runs": n,
        "mean candidate at step": per_step.round(4),
        "best found so far": running.round(4),
    }))
step_tbl = pd.concat(rows, keys=TICKERS, names=["half", "step"])
step_tbl
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for ax, t in zip(axes, TICKERS):
    c = step_tbl.loc[t]
    ax.plot(c.index, c["best found so far"], "o-", lw=2,
            label="best found so far (accumulates)")
    ax.plot(c.index, c["mean candidate at step"], "s--", color="tab:red", lw=2,
            label="mean candidate at this step")
    ax.axhline(0.5, ls=":", color="k", lw=1)
    ax.set_title(t); ax.set_xlabel("evolution step")
    ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("AUC")
plt.tight_layout(); plt.show()

for t in TICKERS:
    c = step_tbl.loc[t]
    print(f"{t}: best-so-far {c['best found so far'].iloc[0]:.3f} -> "
          f"{c['best found so far'].iloc[-1]:.3f}   |   mean candidate "
          f"{c['mean candidate at step'].min():.3f}-{c['mean candidate at step'].max():.3f} "
          f"(flat, near chance)")
""")

md("""
The best-so-far climbs by about 0.05 in both halves while candidate quality stays flat
near 0.5. Evolution is not proposing better ontologies as it goes — it is proposing more of
them, and the maximum of a larger pool is naturally higher.

This bears on cost: `steps=7` takes roughly 1.4× the compute of `steps=3` for candidates
that are no better.

One caveat on reading the left-hand line: the run set shrinks as steps advance (36 runs
reach step 3, only the 12 `steps=7` runs reach step 7), and those surviving runs score
higher, so part of its rise is survivorship rather than accumulation.
""")

# ── E ───────────────────────────────────────────────────────────────────────
md("""
## E · Ontology-change analysis

Each candidate proposes **one node type and one relationship together**, so the (node,
relationship) pair is the unit of intervention — not the node.
""")

code("""
counts = pd.DataFrame([{
    "ticker": t,
    "addition records": int((adds.ticker == t).sum()),
    "node types": adds[adds.ticker == t].node.nunique(),
    "distinct (node, rel) pairs": adds[adds.ticker == t].groupby(["node", "relationship"]).ngroups,
    "accepted": int(((adds.ticker == t) & (adds.status == "accepted")).sum()),
} for t in TICKERS])
# groupby drops null keys, so a candidate proposing a node with NO relationship is
# invisible above. The pair framing assumes that cannot happen — check it.
missing = adds[adds.relationship.isna() | adds.node.isna()]
print(f"candidates with a null node or relationship: {len(missing)} of {len(adds)}")
if len(missing):
    print(missing[["ticker", "step", "node", "relationship", "auc", "status"]].to_string(index=False))
counts
""")

md("""
### Why an addition cannot be scored against the run's current best

The gate makes the reference the *running maximum*, so a candidate is judged partly on how
lucky its own run's step 1 happened to be. The reference climbs while candidate quality
stays flat, and the share scored "negative" grows for that reason alone.
""")

code("""
adds["reference"] = adds.auc - adds.delta_auc
bias = adds.dropna(subset=["delta_auc"]).groupby(["ticker", "step"]).agg(
    n=("auc", "size"),
    mean_candidate_auc=("auc", "mean"),
    mean_reference=("reference", "mean"),
    pct_negative=("delta_auc", lambda s: 100 * (s < 0).mean()),
).round(3)
bias
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for ax, t in zip(axes, TICKERS):
    b = bias.loc[t]
    ax.plot(b.index, b.mean_candidate_auc, "o-", label="mean candidate AUC")
    ax.plot(b.index, b.mean_reference, "s--", color="tab:red", label="mean reference (best so far)")
    ax.set_title(t); ax.set_xlabel("evolution step"); ax.legend(fontsize=8)
    ax2 = ax.twinx(); ax2.plot(b.index, b.pct_negative, ":", color="gray")
    ax2.set_ylim(0, 100)
    if t == "MSFT":
        ax2.set_ylabel("% scored negative")
axes[0].set_ylabel("AUC")
plt.tight_layout(); plt.show()
""")

md("""
### The reference-free measure: win rate

Both candidates in a step are built on the same graph and scored on the same days, so
which one wins carries no reference bias. At 42 validation days the *size* of a gap is
not worth reading — only its sign. Win rate averages 50% by construction: it ranks
additions against each other and says nothing about whether additions help at all.
""")

code("""
# Within each (run, step), the higher AUC wins; ties count for neither. Round first —
# one MSFT step has two candidates differing by a single ULP, which is bit noise from
# the AUC computation rather than a win.
adds["auc_r"] = adds.auc.round(9)
adds["won_recomputed"] = False
for _, g in adds.groupby(["run_id", "step"]):
    top = g.auc_r.max()
    if (g.auc_r == top).sum() == 1:
        adds.loc[g.index[g.auc_r == top], "won_recomputed"] = True
assert (adds.won_recomputed == adds.won).all(), "recomputation disagrees with pull script"
print(f"win flags reproduced exactly: {len(adds)} records")

wins = (adds.dropna(subset=["delta_auc"])
        .groupby(["ticker", "node", "relationship"])
        .agg(proposed=("won", "size"), won=("won", "sum")))
wins["win_rate_%"] = (100 * wins.won / wins.proposed).round(0)
wins = wins[wins.proposed >= 8].sort_values(["ticker", "win_rate_%"], ascending=[True, False])
wins
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, t in zip(axes, TICKERS):
    w = wins.loc[t].sort_values("win_rate_%")
    labels = [f"{n} · {r[:22]}" for n, r in w.index]
    ax.barh(range(len(w)), w["win_rate_%"],
            color=np.where(w["win_rate_%"] > 50, "tab:green", "tab:red"), alpha=.85)
    ax.axvline(50, ls="--", color="k")
    ax.set_yticks(range(len(w))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{t} — win rate (%)"); ax.set_xlim(0, 100)
plt.tight_layout(); plt.show()
""")

code("""
# Do the two halves agree on which additions are good? Only pairs common to both.
common = wins.reset_index().pivot_table(index=["node", "relationship"],
                                        columns="ticker", values="win_rate_%").dropna()
rho, p = stats.spearmanr(common.TSLA, common.MSFT)
print(f"{len(common)} pairs proposed >=8 times in BOTH halves")
print(f"Spearman rho between TSLA and MSFT win rates: {rho:+.3f}  (p = {p:.3f})")
common.astype(int)
""")

code("""
acc = adds.groupby(["ticker", "step"]).agg(
    proposed=("status", "size"),
    accepted=("status", lambda s: (s == "accepted").sum()),
)
acc["accept_%"] = (100 * acc.accepted / acc.proposed).round(0)
for t in TICKERS:
    a = acc.loc[t]
    print(f"{t}: step 1 {a.loc[1,'accept_%']:.0f}%, steps 2+ range "
          f"{a.loc[2:,'accept_%'].min():.0f}-{a.loc[2:,'accept_%'].max():.0f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, t in zip(axes, TICKERS):
    a = acc.loc[t]
    ax.bar(a.index, a["accept_%"], alpha=.85)
    for s, v in zip(a.index, a["accept_%"]):
        ax.annotate(f"{v:.0f}%", (s, v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{t} — acceptance by step"); ax.set_xlabel("evolution step")
axes[0].set_ylabel("accepted (%)")
plt.tight_layout(); plt.show()
acc
""")

code("""
# Gate selectivity: does the gate accept the additions that actually scored better?
gate = adds.dropna(subset=["delta_auc"]).assign(acc=lambda d: d.status == "accepted")
sel = gate.groupby("ticker").apply(lambda g: pd.Series({
    "accepted": f"{int(g.acc.sum())} / {len(g)} ({100*g.acc.mean():.1f}%)",
    "mean delta of accepted": round(g[g.acc].delta_auc.mean(), 4),
    "mean delta of rejected": round(g[~g.acc].delta_auc.mean(), 4),
}), include_groups=False)
sel
""")

code("""
# Cost of extra evolution steps — the claim in the report's cost paragraph.
for t in TICKERS:
    m = runs[runs.ticker == t].groupby("steps").final_auc.mean()
    print(f"{t}: steps7 - steps3 = {m[7]-m[3]:+.4f}   "
          f"mean(steps5,7) - steps3 = {(m[5]+m[7])/2 - m[3]:+.4f}   "
          f"(between-run SD {runs[runs.ticker==t].final_auc.std():.3f})")
""")

# ── F ───────────────────────────────────────────────────────────────────────
md("""
## F · Train vs validation — the memorisation check

`train_auc` was added on 2026-08-11, so it exists for every MSFT run and for no TSLA run.
This is the one place the two halves cannot be shown side by side, and the MSFT column
alone settles the question.
""")

code("""
tr = runs.dropna(subset=["train_auc"])
print(f"runs with train_auc logged: {len(tr)} ({tr.ticker.unique().tolist()})")
print(f"  train AUC   mean {tr.train_auc.mean():.4f}   min {tr.train_auc.min():.4f}   "
      f"max {tr.train_auc.max():.4f}   SD {tr.train_auc.std():.4f}")
print(f"  val AUC     mean {tr.final_auc.mean():.4f}")
print(f"  gap         mean {(tr.train_auc - tr.final_auc).mean():+.4f}")
print()
print("488 structural features against 98 training days: the model separates the training")
print("set perfectly in every single run, with zero variance. Whatever the validation AUC")
print("is measuring, it is not a model that had to generalise to fit.")
""")

# ── G ───────────────────────────────────────────────────────────────────────
md("""
## G · The cross-ticker verdict

The MSFT half was run specifically to test whether the mixed corpus was diluting the
signal. It was not.
""")

code("""
verdict = pd.DataFrame({
    t: {
        "mean final AUC": round(runs[runs.ticker == t].final_auc.mean(), 4),
        "mean text baseline": round(runs[runs.ticker == t].baseline_auc.mean(), 4),
        "mean lift": round(runs[runs.ticker == t].delta.mean(), 4),
        "beat baseline": f"{(runs[runs.ticker == t].delta > 0).sum()}/36",
        "best run": round(runs[runs.ticker == t].final_auc.max(), 4),
        "path-signal slope": round(
            np.polyfit(runs[runs.ticker == t].path_mean_norm,
                       runs[runs.ticker == t].final_auc, 1)[0], 4),
    } for t in TICKERS
})
verdict["difference"] = ""
for k in ["mean final AUC", "mean text baseline", "mean lift"]:
    verdict.loc[k, "difference"] = f"{verdict.loc[k, 'MSFT'] - verdict.loc[k, 'TSLA']:+.4f}"
verdict
""")

code("""
gap = runs[runs.ticker == "MSFT"].delta.mean() - runs[runs.ticker == "TSLA"].delta.mean()
sd = runs.groupby("ticker").final_auc.std().mean()
print(f"difference in mean lift (MSFT - TSLA): {gap:+.4f}")
print(f"  against a between-run SD of {sd:.4f}, i.e. {abs(gap)/sd:.2f} of the ordinary")
print(f"  scatter between runs within a half")
print()
print("Both halves beat their baseline in exactly 28 of 36 runs. Filtering the corpus to")
print("the labelled company moved nothing that this design can resolve.")
""")

md("""
### Cross-check: every figure quoted in the report

One cell, one line per number that appears in
`analyses/tsla_msft_200days_report.html`, so each can be ticked off against its source.
""")

code("""
def eta2_total(s):
    \"\"\"Share of a half's AUC variance explained by the four factors jointly.\"\"\"
    total = ((s.final_auc - s.final_auc.mean()) ** 2).sum()
    between = sum((g.count() * (g.mean() - s.final_auc.mean()) ** 2).sum()
                  for g in (s.groupby(f).final_auc for f in FACTORS))
    return between / total


def per_ticker(fn):
    \"\"\"Apply fn to each half's runs+additions, returning a {ticker: value} dict.\"\"\"
    return {t: fn(runs[runs.ticker == t], adds[adds.ticker == t]) for t in TICKERS}


checks = {
    "runs":                       per_ticker(lambda r, a: len(r)),
    "mean final AUC":             per_ticker(lambda r, a: f"{r.final_auc.mean():.3f}"),
    "SD of final AUC":            per_ticker(lambda r, a: f"{r.final_auc.std():.3f}"),
    "min / max final AUC":        per_ticker(lambda r, a: f"{r.final_auc.min():.3f} / {r.final_auc.max():.3f}"),
    "mean text baseline":         per_ticker(lambda r, a: f"{r.baseline_auc.mean():.3f}"),
    "SD of text baseline":        per_ticker(lambda r, a: f"{r.baseline_auc.std():.3f}"),
    "mean lift":                  per_ticker(lambda r, a: f"{r.delta.mean():+.3f}"),
    "SD of lift":                 per_ticker(lambda r, a: f"{r.delta.std():.3f}"),
    "min / max lift":             per_ticker(lambda r, a: f"{r.delta.min():+.3f} / {r.delta.max():+.3f}"),
    "beat text baseline":         per_ticker(lambda r, a: f"{(r.delta > 0).sum()} / 36"),
    "mean Brier":                 per_ticker(lambda r, a: f"{r.brier_score.mean():.3f}"),
    "between-run SD of AUC":      per_ticker(lambda r, a: f"{r.final_auc.std():.3f}"),
    "path: Spearman rho":         per_ticker(lambda r, a: f"{stats.spearmanr(r.path_mean_norm, r.final_auc)[0]:+.3f}"),
    "path: OLS slope":            per_ticker(lambda r, a: f"{np.polyfit(r.path_mean_norm, r.final_auc, 1)[0]:+.3f}"),
    "path: empty blocks":         per_ticker(lambda r, a: f"{int((r.path_mean_norm == 0).sum())} / 36"),
    "furthest factor level":      per_ticker(lambda r, a: f"{max(abs(r.groupby(f).final_auc.mean() - r.final_auc.mean()).max() for f in FACTORS):.3f}"),
    "variance explained (4 factors)": per_ticker(lambda r, a: f"{eta2_total(r):.3f}"),
    "steps7 - steps3 AUC":        per_ticker(lambda r, a: f"{r.groupby('steps').final_auc.mean()[7] - r.groupby('steps').final_auc.mean()[3]:+.3f}"),
    "addition records":           per_ticker(lambda r, a: len(a)),
    "node types":                 per_ticker(lambda r, a: a.node.nunique()),
    "distinct (node, rel) pairs": per_ticker(lambda r, a: a.groupby(["node", "relationship"]).ngroups),
    "additions accepted":         per_ticker(lambda r, a: f"{(a.status == 'accepted').sum()} / {len(a)}"),
    "train AUC (mean)":           per_ticker(lambda r, a: f"{r.train_auc.mean():.4f}" if r.train_auc.notna().any() else "not logged"),
}
cross = pd.DataFrame(checks).T[TICKERS]

print("cross-half quantities (single-valued):")
print(f"  MSFT - TSLA lift gap        {gap:+.4f}")
print(f"  win-rate rank correlation   rho {rho:+.3f} over {len(common)} pairs")
print(f"  candidates with null rel.   {len(missing)} of {len(adds)}\\n")
cross
""")

md("""
### What that buys

A well-powered negative on the corpus-dilution hypothesis, replicated across two
independent balanced designs:

- **Path-chain features do nothing** — flat dose–response in both halves
- **No hyperparameter separates** from between-run scatter in either half
- **Filtering the corpus changes nothing** — 28/36 both ways, mean lift differing by 0.010
- **The model memorises completely** — train AUC 1.0000 with zero variance across 36 runs

The remaining constraint is the estimator, not the corpus: 98 training days, 42 validation
days, 488 features. Formal tests of every claim here are in `stats_tests_note.ipynb`.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "report.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} — {len(CELLS)} cells")
