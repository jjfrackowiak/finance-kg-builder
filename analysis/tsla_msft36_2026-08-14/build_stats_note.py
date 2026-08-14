"""Generate stats_tests_note.ipynb — formal hypothesis testing, kept out of the report.

`report.ipynb` is descriptive: it reproduces the numbers the report states. This one is
inferential: it asks which of those differences survive a test. Split deliberately, so
the report and its companion stay readable without a statistics detour, and so the
testing can be revisited on its own terms later.

    python build_stats_note.py
    uv run --with jupyter,nbconvert,pandas==2.3.3,numpy,scipy,matplotlib \
        jupyter nbconvert --to notebook --execute --inplace stats_tests_note.ipynb
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
CELLS: list = []


def _lines(text: str) -> list:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(text)})


md("""
# Statistical tests — side note to the TSLA/MSFT sweep report

`analyses/tsla_msft_200days_report.html` and its companion `report.ipynb` are
deliberately descriptive: means, spreads, counts, rankings. This note asks the separate
question of **which differences survive a formal test**, and is kept apart so neither
document has to carry the other's baggage.

## The design point that makes testing easy

Each half contains **36 runs**, so the runs themselves are the replicates. Every test
below uses the observed run-to-run scatter as its error term — nothing depends on a
theoretical model of how AUC is distributed.

That matters, because the obvious alternative is wrong. One might compare a factor effect
against the sampling error of a single AUC (about 0.09 on a 42-day validation window,
from the Mann–Whitney variance). But a factor level's mean averages 9–12 runs, so that is
the wrong reference: it compares a mean to the SE of one observation. The 0.09 figure
describes the *resolution of a single measurement*, which is a real limit worth knowing,
but it is not the yardstick for a group difference. ANOVA measures the right error term
directly.

## Summary of what follows

| Question | Test | Result |
|---|---|---|
| Does the graph arm beat text? | sign / Wilcoxon / paired t | **significant, and replicated** |
| Does any hyperparameter matter? | one-way ANOVA × 8 | no (min p = 0.055) |
| Do path features help? | Spearman; empty vs non-empty | no |
| Did filtering the corpus matter? | Welch t between halves | no |
| Are some ontology additions better? | binomial vs 50%, × 28 | 1 survives, but flips sign in the other half |
| Does the gate discriminate? | Welch t on accepted vs rejected | yes, but **circular** |
""")

code("""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 150)

TICKERS = ["TSLA", "MSFT"]
FACTORS = ["lookback_days", "steps", "chain_hops", "evolution_prompt"]

runs = pd.read_csv("results.csv")
adds = pd.read_csv("additions.csv")
print(f"{len(runs)} runs, {len(adds)} addition records")
""")

# ── 1 ───────────────────────────────────────────────────────────────────────
md("""
## 1 · Does the graph arm beat the text baseline?

Each run pairs a graph AUC with a text AUC scored on the same validation days, so this is
a paired comparison. Three tests, in increasing order of what they assume:

- **sign test** — counts wins, ignores their size entirely
- **Wilcoxon signed-rank** — uses the rank of the differences
- **paired t** — uses the differences themselves, assuming approximate normality
""")

code("""
rows = []
for label, d in [(t, runs[runs.ticker == t].delta) for t in TICKERS] + [("pooled", runs.delta)]:
    wins, n = int((d > 0).sum()), len(d)
    rows.append({
        "half": label,
        "wins": f"{wins}/{n}",
        "mean lift": round(d.mean(), 4),
        "sign test p": stats.binomtest(wins, n, 0.5).pvalue,
        "Wilcoxon p": stats.wilcoxon(d).pvalue,
        "paired t": round(stats.ttest_1samp(d, 0).statistic, 2),
        "t-test p": stats.ttest_1samp(d, 0).pvalue,
    })
pd.DataFrame(rows).set_index("half")
""")

md("""
**Significant on every test, in both halves independently.**

The more persuasive fact is not any single p-value but the **replication**: two halves
with different labels, different corpora and independently drawn ontologies both land on
exactly 28 of 36.

### The caveat that matters

The 36 runs within a half are *not* independent. They share the same validation days, and
all 9 runs at a given lookback share one identical text baseline — there are only four
distinct baselines per half. Dependence like that inflates within-half significance, so
the p-values above should be read as optimistic.

What does not depend on any independence assumption is the cross-half replication. Treating
each half as one observation, two out of two agreeing in direction is weak evidence on its
own — but combined with 28/36 in each, the result is about as solid as this design can make
it.

### What it does *not* establish

The graph arm and the text arm share no features: `build_day_feature_vector` skips text
entirely. So this tests *which of two disjoint representations wins alone*, not whether
the graph **adds** anything to article text. That question needs a text + graph arm, which
has never been run.
""")

# ── 2 ───────────────────────────────────────────────────────────────────────
md("""
## 2 · Does any hyperparameter matter?

One-way ANOVA per factor. The design is a balanced strength-2 orthogonal array, so within
any one factor the other three are distributed identically across levels — a one-way test
is valid with no adjustment.
""")

code("""
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    for f in FACTORS:
        groups = [g.final_auc.values for _, g in s.groupby(f)]
        F, p = stats.f_oneway(*groups)
        m = s.groupby(f).final_auc.mean()
        rows.append({"half": t, "factor": f, "levels": len(groups),
                     "F": round(F, 2), "p": round(p, 3),
                     "best": m.idxmax(), "worst": m.idxmin(),
                     "best-worst gap": round(m.max() - m.min(), 3)})
anova = pd.DataFrame(rows)
print(f"significant at 0.05: {(anova.p < 0.05).sum()} of {len(anova)}")
print(f"Bonferroni threshold for {len(anova)} tests: {0.05/len(anova):.4f}")
anova.set_index(["half", "factor"])
""")

md("""
**Nothing is significant.** The smallest p is 0.055 — `chain_hops` on MSFT — which fails
even before correcting for the eight tests performed.

And that near-miss argues against itself. MSFT's `chain_hops` favours **h3**; TSLA's
(p = 0.174) favours **h5**. The one factor that comes closest to significance points in
opposite directions in the two halves, which is the signature of noise rather than effect.
""")

code("""
# Same conclusion without the normality assumption: Kruskal-Wallis on ranks.
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    for f in FACTORS:
        H, p = stats.kruskal(*[g.final_auc.values for _, g in s.groupby(f)])
        rows.append({"half": t, "factor": f, "H": round(H, 2), "p": round(p, 3)})
kw = pd.DataFrame(rows)
print(f"significant at 0.05: {(kw.p < 0.05).sum()} of {len(kw)}   min p = {kw.p.min():.3f}")
kw.set_index(["half", "factor"])
""")

# ── 3 ───────────────────────────────────────────────────────────────────────
md("""
## 3 · Do path-chain features help?

Two framings: a correlation between realised path signal and AUC, and a group comparison
between runs whose path block was entirely empty and runs where it carried signal.
""")

code("""
rows = []
for t in TICKERS:
    s = runs[runs.ticker == t]
    rho, p_rho = stats.spearmanr(s.path_mean_norm, s.final_auc)
    empty, full = s[s.path_mean_norm == 0].final_auc, s[s.path_mean_norm > 0].final_auc
    tt, p_t = stats.ttest_ind(empty, full, equal_var=False)
    rows.append({"half": t, "spearman rho": round(rho, 3), "rho p": round(p_rho, 3),
                 "n empty": len(empty), "mean AUC empty": round(empty.mean(), 3),
                 "mean AUC non-empty": round(full.mean(), 3),
                 "Welch t": round(tt, 2), "t p": round(p_t, 3)})
pd.DataFrame(rows).set_index("half")
""")

md("""
Nothing significant, and the point estimates lean the *wrong* way: in both halves runs
with an entirely empty path block averaged slightly **higher** AUC than runs carrying real
chain embeddings. The two halves also disagree on the sign of the correlation.

These features cost real compute to construct, which makes a null here consequential
rather than merely uninteresting.
""")

# ── 4 ───────────────────────────────────────────────────────────────────────
md("""
## 4 · Did filtering the corpus change anything?

The MSFT half was run specifically to test whether the mixed article set was diluting the
signal. Between-half comparison, so an unpaired Welch t.
""")

code("""
T, M = runs[runs.ticker == "TSLA"], runs[runs.ticker == "MSFT"]
rows = []
for label, a, b in [("mean lift", M.delta, T.delta),
                    ("mean final AUC", M.final_auc, T.final_auc),
                    ("mean text baseline", M.baseline_auc, T.baseline_auc)]:
    tt, p = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b)
    rows.append({"quantity": label, "MSFT": round(a.mean(), 4), "TSLA": round(b.mean(), 4),
                 "difference": round(a.mean() - b.mean(), 4),
                 "Welch t": round(tt, 2), "p": round(p, 3), "Mann-Whitney p": round(p_u, 3)})
pd.DataFrame(rows).set_index("quantity")
""")

md("""
No difference on any measure. Both halves also beat their baseline in exactly 28 of 36
runs — an identical win count.

This is a **well-powered null on a pre-specified question**, which is worth more than the
p-value suggests: it closes off the most obvious alternative explanation for the sweep's
results rather than leaving it open.
""")

# ── 5 ───────────────────────────────────────────────────────────────────────
md("""
## 5 · Are some ontology additions genuinely better than others?

Within a step, two candidates are built on the same graph and scored on the same days, so
which one wins is a clean paired comparison free of reference bias. Test each addition's
win count against 50%.
""")

code("""
w = (adds.dropna(subset=["delta_auc"])
         .groupby(["ticker", "node", "relationship"])
         .agg(n=("won", "size"), wins=("won", "sum")))
w = w[w.n >= 8].copy()
w["win_rate_%"] = (100 * w.wins / w.n).round(0)
w["p"] = [stats.binomtest(int(r.wins), int(r.n), 0.5).pvalue for _, r in w.iterrows()]
bonf = 0.05 / len(w)
print(f"{len(w)} tests   raw p<0.05: {(w.p < 0.05).sum()}   "
      f"Bonferroni {bonf:.4f}: {(w.p < bonf).sum()} survive")
w.sort_values("p").head(8).round(4)
""")

code("""
# Does the ranking replicate across halves? Only pairs proposed >=8 times in BOTH.
common = (w.reset_index()
            .pivot_table(index=["node", "relationship"], columns="ticker", values="win_rate_%")
            .dropna())
# Distinct names: `p` gets reused by later cells, and this value is needed at the end.
rho_replic, p_replic = stats.spearmanr(common.TSLA, common.MSFT)
print(f"{len(common)} pairs common to both halves")
print(f"Spearman rho between the two rankings: {rho_replic:+.3f}  (p = {p_replic:.3f})\\n")
common.assign(swing=(common.MSFT - common.TSLA).astype(int)).astype(int).sort_values("TSLA", ascending=False)
""")

md("""
One pair clears Bonferroni — `EconomicIndicator / REFERENCES` on TSLA, winning 8 of 36
(22%). But in the MSFT half the same pair wins 47%, indistinguishable from chance. The
second-strongest, `RegulatoryBody / MENTIONS_REGULATORY_BODY` at 78% on TSLA, drops to 54%.

Across the 13 pairs testable in both halves the rank correlation is **ρ = −0.03**: no
relationship whatsoever. Whatever the within-half tests are picking up does not survive a
change of ticker, so it should not be read as "these concepts matter for prediction".
""")

# ── 6 ───────────────────────────────────────────────────────────────────────
md("""
## 6 · Does the acceptance gate discriminate?

Included for completeness, and as a warning about what a small p-value is worth.
""")

code("""
rows = []
for t in TICKERS:
    a = adds[adds.ticker == t].dropna(subset=["delta_auc"])
    acc, rej = a[a.status == "accepted"].delta_auc, a[a.status != "accepted"].delta_auc
    tt, p = stats.ttest_ind(acc, rej, equal_var=False)
    rows.append({"half": t, "n accepted": len(acc), "n rejected": len(rej),
                 "mean delta accepted": round(acc.mean(), 3),
                 "mean delta rejected": round(rej.mean(), 3),
                 "Welch t": round(tt, 1), "p": f"{p:.1e}"})
pd.DataFrame(rows).set_index("half")
""")

md("""
p on the order of 10⁻²⁶ — and **meaningless**. The gate accepts a candidate precisely when
its `delta_auc` clears a threshold, so testing whether accepted candidates have higher
`delta_auc` tests the definition of the gate, not any property of the ontology. It is
included here as the clearest example in this dataset of a significant result that
establishes nothing.
""")

# ── 7 ───────────────────────────────────────────────────────────────────────
md("""
## 7 · What the single-AUC resolution limit is, and why it is not used above

For completeness, since it is a natural thing to reach for and was used in an earlier
draft of the report.

AUC is the Mann–Whitney U statistic rescaled, `AUC = U/(n1*n0)`. Under the null of no
discrimination U has variance `n1*n0*(n1+n0+1)/12`, giving a distribution-free standard
error that depends only on the class counts:

$$\\mathrm{SE}(\\mathrm{AUC}) = \\sqrt{\\frac{n_1+n_0+1}{12\\,n_1 n_0}}$$
""")

code("""
def auc_se(n1, n0):
    return np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))

# Realised validation splits, reconstructed from price data and cross-checked against the
# class_balance metric MLflow logged over all 140 modelled days (numerators matched exactly).
splits = {"TSLA": (19, 22), "MSFT": (20, 21)}
for t, (n1, n0) in splits.items():
    print(f"  {t}: {n1} up / {n0} down  ->  SE = {auc_se(n1, n0):.4f}")
print(f"\\n  balanced 21/21 reference: {auc_se(21, 21):.4f}")
print(f"  across every split from 1/3 to 2/3 balance at n=41-42: "
      f"{min(auc_se(k, n-k) for n in (41,42) for k in range(14, n-13)):.4f}"
      f" to {max(auc_se(k, n-k) for n in (41,42) for k in range(14, n-13)):.4f}")
""")

md("""
So a single run's AUC carries roughly **±0.09** of sampling noise. That is a genuine and
useful fact — it is why no individual run's number, including the 0.787 best result, should
be read as precise.

**But it is the wrong reference for the comparisons in sections 2–4.** Those compare group
means of 9–12 runs, and the correct error term for a group mean is not the SE of a single
observation. ANOVA estimates that error term from the data, which is why it is used above.

The two approaches happen to agree here — the between-run SD is 0.065 (TSLA) and 0.064
(MSFT), the largest best-worst factor gap is 0.057, and no ANOVA reaches significance — but
they agree by coincidence of magnitude, not because the single-AUC SE was ever the right
test.
""")

# ── 8 ───────────────────────────────────────────────────────────────────────
md("""
## 8 · Everything in one table
""")

code("""
summary = pd.DataFrame([
    ("graph beats text — TSLA", "sign test, 28/36", f"{stats.binomtest(28,36,0.5).pvalue:.4f}", "YES"),
    ("graph beats text — MSFT", "sign test, 28/36", f"{stats.binomtest(28,36,0.5).pvalue:.4f}", "YES"),
    ("graph beats text — pooled", "sign test, 56/72", f"{stats.binomtest(56,72,0.5).pvalue:.1e}", "YES"),
    ("hyperparameters (8 tests)", "one-way ANOVA", f"min {anova.p.min():.3f}", "no"),
    ("hyperparameters (8 tests)", "Kruskal-Wallis", f"min {kw.p.min():.3f}", "no"),
    ("path features", "Spearman / Welch", "0.20 - 0.96", "no"),
    ("corpus filtering", "Welch t", "0.61", "no"),
    ("addition ranking replicates", "Spearman across halves", f"{p_replic:.2f}", "no"),
    ("gate discriminates", "Welch t", "1e-26", "yes, but circular"),
], columns=["question", "test", "p", "significant?"])
summary.set_index("question")
""")

md("""
### The bottom line

One real positive result, replicated across two independent halves: **the evolved graph
representation beats a text-only baseline**, 28 of 36 runs each time.

Everything else is null — and these are reasonably well-powered nulls rather than
uninformative ones, because each is backed by two balanced 36-run designs that agree.
The nulls are the more useful contribution: they rule out corpus dilution, path-chain
features, and every hyperparameter under study as explanations for anything.
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
out = HERE / "stats_tests_note.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} — {len(CELLS)} cells")
