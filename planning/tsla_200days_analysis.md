# TSLA 200-day sweep — analysis plan

Planning document for the supervisor meeting. Covers what was run, what data exists,
and the analyses to produce from it.

Status at time of writing: chunks 01–12 complete, chunk 13 running. All 36 TSLA
configs available once 13 lands.

---

## 1. What was run

**Design.** Per-ticker orthogonal array `OA(36, 4^1 3^3, 2)` — 36 TSLA configs, four
factors, all six factor pairs balanced (λ = 3 or 4). Balance verified directly against
the 26 config files, not assumed.

| Factor | Levels | Runs per level |
|---|---|---|
| `lookback_days` | 3, 8, 10, 20 | 9 |
| `steps` | 3, 5, 7 | 12 |
| `evolution_prompt` | default, event_driven, fundamental | 12 |
| `min_chain_hops` | 3, 4, 5 | 12 |

**Fixed across all 36 runs:** ticker TSLA, window 2022-05-02 → 2022-11-18 (201 days),
4 articles/day, `candidates=2`, `feature_mode=hybrid`, `single_addition=true`,
`auc_drop_tolerance=0`, `train_ratio` default (98 train days), sidecar Neo4j.

Because the design is balanced within ticker, every comparison is a **group-means
contrast with the other factors distributed identically across the compared groups**.
No regression adjustment is needed and none should be used. Ticker is not a factor —
MSFT is an independent replication of the same design, not a missing half.

**Infrastructure.** 8 × g5.xlarge (A10G) vLLM workers, 6 embedding replicas,
5 × t3.xlarge CPU nodes, ephemeral Neo4j sidecar per job, on EKS. Chunks dispatched by
a self-chaining GitHub Actions workflow that marks progress in `sweep_runs/MANIFEST.md`.

**Timing and cost (measured, chunks 01–12):**

| Group | Chunks | Configs | Mean per chunk | Subtotal |
|---|---|---|---|---|
| `steps=3` | 01–03 | 12 | 83 min | 4.2 h |
| `steps=5` | 04–07 | 12 | 114 min | 7.6 h |
| `steps=7` | 08–13 | 12 | 114 min | 11.4 h (est. incl. 13) |
| **Total** | **13** | **36** | — | **~23 h** |

Cost at eu-central-1 on-demand ($1.258/h g5.xlarge, $0.192/h t3.xlarge):
**≈ $255** for the TSLA half. Add ~$27 of debugging and validation runs on 2026-08-07.

About 22% of each chunk is overhead — node drain/provision plus vLLM model load — because
the workflow tears the nodegroup down and rebuilds it between chunks. Recoverable
(~4.5 h across 13 chunks) but not attempted mid-sweep.

### 1.1 A defect was found and fixed before this sweep

Worth stating explicitly to the supervisor, because it invalidates earlier results.

`embed_relationship_chains` branched on `local` vs else-OpenAI, with no `remote` branch.
Every EKS run sets `EMBEDDING_TYPE=remote`, so chains were extracted correctly and then
**discarded at the embedding step** — the function returned `[]`, the 384-dim path block
was all zeros, and `max_hops` stayed 0. Any sweep run before 2026-08-08 has a dead path
feature block and must not be pooled with these results.

Fixed in `36f4f88`. Per-block coverage metrics (`feat/path_*`, `feat/subgraph_*`,
`feat/topo_*`) were added at the same time so a silently-dead feature block can never
again pass unnoticed.

---

## 2. Data inventory

**Per-run metrics** (MLflow parent run, logged per evolution step):

- Downstream: `auc`, `f1`, `precision`, `recall`, `brier_score`, `class_balance`
- Best-of-run: `best_auc`, `best_f1`, tag `best_candidate`
- Graph: `graph/n_nodes`, `graph/n_edges`
- Chains: `max_hops_train`, `max_hops_val`
- Split: `n_train_days`, `n_val_days`
- **Feature-block coverage:** `feat/{path,subgraph,topo}_{dims,nonzero_frac,mean_norm}`

**Per-candidate child runs:** one per (step, candidate) — 14 per `steps=7` run — with
tags `winner`, `step_winner`, `step_accepted`, and `delta_auc` against the baseline.

**Artifacts per run:**

- `ontologies/addition_history.json` — one record per (step, candidate)
- `ontologies/ontology_summary.json` — final ontology state
- `ontologies/step_N_candidate_M.json` — every candidate ontology
- `additions/addition_history_*.json`

**Addition record schema** (this is the backbone of the ontology analysis):

```json
{
  "step": 1, "variant": 0, "candidate_tag": "step_1_candidate_0",
  "node": "EconomicIndicator", "relationship": "REFERENCES",
  "patterns": [["Article", "MENTIONS", "EconomicIndicator"]],
  "auc": 0.4394, "delta_auc": -0.2105,
  "delta_reference": "baseline_article_embedding",
  "status": "rejected"
}
```

**360 addition records** across the 36 TSLA runs (12 runs × 3 steps × 2 candidates +
12 × 5 × 2 + 12 × 7 × 2).

### 2.1 What `path_mean_norm` measures

The headline feature-health metric, and the x-axis of the dose-response analysis (D2),
so it is worth defining precisely.

The path block is 384 numbers per day — the aggregated embedding of the relationship
chains extracted for that day. `feat/path_mean_norm` is:

```python
np.mean(np.linalg.norm(block, axis=1))
```

For each day, take the Euclidean length of its 384-number path vector — how "large" that
day's chain embedding is — then average over all days in the run. Interpretation:

| Value | Meaning |
|---|---|
| 0.000 | every day's path vector is all zeros — no path features at all |
| ~0.09 | a handful of days have chains, most are empty |
| ~0.79 | most days carry real chain embeddings |

Observed in this sweep: **≈0.55–0.79 at h3, ≈0.09–0.23 at h4, exactly 0 at h5.**

**Why not `path_nonzero_frac`.** That metric counts how many of the 384 columns hold any
non-zero value anywhere. Because embeddings are dense, a *single* day with one chain
lights up all 384 columns and yields `nonzero_frac = 1.0` — indistinguishable from a
fully populated block. One h5 config showed exactly that: `nonzero_frac = 1.0` with
`mean_norm = 0.014`, i.e. essentially empty. So `nonzero_frac` answers *"is anything
there at all"* and is the right alarm for a dead block; `path_mean_norm` answers *"how
much"* and is the right quantity to analyse and report.

---

## 3. Analysis A — headline / top-ranking metrics

Purpose: the "what did we get" slide.

1. **Leaderboard of all 36 configs** by final `auc`, showing the four factor levels
   plus `path_mean_norm` and `max_hops`. Also ranked by `best_auc` — note the two
   differ, since `best_auc` is the best step, not the final one.
2. **Baseline contrast.** Every run logs `baseline_article_embedding` as a child. Report
   final AUC minus baseline AUC per run; that is the quantity the whole pipeline is
   supposed to move.
3. **Descriptives across the 36 runs:** mean, SD, min, max, quartiles for AUC, F1,
   Brier. Establishes the noise floor before any factor comparison is believed.

Report Brier alongside AUC. AUC on ~42 validation days is coarse; a calibration metric
guards against reading rank noise as signal.

---

## 4. Analysis B — differential analysis by hyperparameter

This is the core of the balanced design, and the section the design was built for.

**B1. Group means with dispersion.** For each factor level: n, mean AUC, SD, SEM.
Present as a table plus a point-and-error-bar plot per factor. Because the design is
balanced, the difference between two level means is attributable to that factor alone.

**B2. Distributions, not just means** (your idea, and the right instinct). Per factor
level, plot the full distribution of AUC across runs — overlaid densities or
box/violin with individual points shown. With n = 9–12 per level, show the raw points;
a box plot alone hides too much at this sample size.

**B3. Factor-vs-factor interaction views.** Heatmap of mean AUC over pairs of factors
(e.g. `steps` × `min_chain_hops`, `lookback` × `prompt`). Every cell is populated by
construction (λ = 3 or 4), so no cell is empty — the balance pays off directly here.

**B4. Variance decomposition.** How much of the AUC spread is attributable to each
factor versus residual? Even a simple between/within-group variance ratio per factor
communicates "does this hyperparameter matter at all" more honestly than a table of
means. Expect the residual to dominate; that is itself a finding.

**Discipline:** four factors × several metrics invites false positives. Fix the primary
metric (AUC) and the primary factor comparisons **before** looking, treat everything
else as exploratory, and say which is which on the slide.

---

## 5. Analysis C — ontology-change analysis

Purpose: average effect and SD per introduced ontology change, from the 360 addition
records.

**C1. Effect per introduced node type.** Group all additions by `node`, report n,
mean `delta_auc`, SD, and acceptance rate. Same for `relationship`. This answers "which
ontology extensions actually help."

**C2. Acceptance dynamics.** Fraction of candidates accepted, broken down by `step`
index and by `evolution_prompt`. Does acceptance decay as the ontology saturates? Do the
three prompts propose systematically different things?

**C3. Effect by step position.** Mean `delta_auc` by step number. Tests whether later
evolution steps still add value — directly relevant to whether `steps=7` is worth 
2.7× the compute of `steps=3`.

**C4. Pattern-level analysis.** The `patterns` field holds the metapath introduced
(e.g. `Article -MENTIONS-> EconomicIndicator`). Aggregate effect by pattern shape —
e.g. do entity→entity patterns outperform article→entity ones?

**C5. Proposal diversity by prompt.** Count distinct `(node, relationship)` pairs each
prompt proposes across its 12 runs. A prompt that proposes the same three things every
time is a different object from one that explores.

Caveat to state: `delta_auc` is measured against the baseline on the same short
validation window, so per-addition effects are noisy. Aggregate over many records and
report SD; do not rank individual additions.

---

## 6. Analysis D — proposals

**D1. Path-feature ablation (highest value).** The `h5` arm produced **no path features
at all** — `path_mean_norm = 0`, `max_hops = 0` — while `h3` and `h4` did. That gives a
ready-made controlled ablation: **12 runs without path features vs 24 with.**

*The groups are exactly balanced — verified against the config files, not assumed:*

| Factor | path-free (h5, n=12) | path-present (h3+h4, n=24) |
|---|---|---|
| lookback 3 / 8 / 10 / 20 | 3 / 3 / 3 / 3 | 6 / 6 / 6 / 6 |
| steps 3 / 5 / 7 | 4 / 4 / 4 | 8 / 8 / 8 |
| prompt default / event / fundamental | 4 / 4 / 4 | 8 / 8 / 8 |

Every other factor appears in identical proportions (exact 1:2). So the difference in
mean AUC between the groups is attributable to the presence of path features alone — no
adjustment, no covariates, no assumptions beyond the design itself. This is the same
group-means logic as B1, applied to a contrast we did not plan but did get.

Report: n, mean AUC, SD, and the difference with its standard error. Also run it on
`best_auc` and on AUC-minus-baseline, so the conclusion does not hinge on one endpoint.

In the chunks inspected so far the path-free configs scored **higher** AUC. At n = 12 vs
24 this becomes answerable rather than anecdotal. If it holds it is the single most
important result in the sweep, and it belongs early in the meeting, framed honestly.

*Limitation to state:* h5 is path-free *because* 5-hop chains are unreachable at this
graph density, so "no path features" is confounded with "deepest hop setting." The
groups are balanced on the three other design factors but not on hop depth itself —
which is unavoidable, since hop depth is what causes the emptiness. D2 is the check that
does not have this problem.

**D2. Dose-response on path signal (the stronger test).** Instead of treating hops as a
category, use `path_mean_norm` as a **continuous** measure of how much path signal a run
actually carried (see §2.1) and plot AUC against it across all 36 runs.

- x-axis: `path_mean_norm`, 0 → ~0.8
- y-axis: final `auc` (repeat with `best_auc` and AUC-minus-baseline)
- colour or facet by `steps`, since more steps build denser graphs and so shift x
- overlay a fitted line with a confidence band, and report Spearman ρ alongside it —
  rank correlation is more honest than a slope at n = 36 with this much scatter

**Why this is better than D1.** It avoids the hop-depth confound entirely: runs at the
same hop level still vary widely in realised path signal (h4 ranged 0.04 → 0.23), so the
x-axis is not a proxy for the hop setting. It uses all 36 runs instead of splitting
12/24, and it asks the graded question — *does more path signal track better prediction*
— rather than the binary one.

**How to read it:** a positive slope means path features earn their place, and more of
them is better. A flat line means they are inert. A negative slope would mean they are
actively hurting, which given the earlier chunks is a live possibility and should not be
explained away if it appears. Expect wide scatter; the honest read at n = 36 is
direction and rank correlation, not a precise effect size.

**D3. Graph growth vs performance.** `graph/n_edges` ranges roughly 12k–24k. Does a
larger evolved graph predict better? Bears on whether ontology evolution is adding
signal or just mass.

**D4. Feature-block coverage as a QA panel.** `feat/*_nonzero_frac` per run, all 36 in
one figure. Demonstrates to the supervisor that every block was verified live, and makes
the h5 path-emptiness visible as a designed observation rather than an anomaly.

**D5. Step trajectory.** `addition_history` carries `auc` per (step, candidate), so the
within-run AUC trajectory can be reconstructed. Does AUC rise monotonically with
evolution steps, plateau, or oscillate? Speaks to whether the feedback loop converges.

---

## 7. Caveats to state up front

1. **The h5 arm has no path features.** Expected and documented — a property of graph
   density and the expansion filters (`labelFilter -Article|-Day`, five excluded
   relation types, `candidate_tags`, `first_seen`), not a defect. Phrase findings as
   "at this graph density, under these filters," not "5-hop paths do not matter."
2. **Validation windows are short** (~42 days). AUC is noisy; observed spread across
   configs differing only in lookback has been 0.47–0.79.
3. **n = 9–12 per factor level.** Adequate for large effects only.
4. **Pre-2026-08-08 runs are invalid** (dead path block) and must not be pooled.
5. **LLM-driven evolution is stochastic.** Two runs at identical settings produced
   `path_mean_norm` differing ~6×. Run-to-run variance may exceed factor effects — D2
   and B4 are the checks for this.
6. **Single ticker.** All conclusions are TSLA-specific until MSFT replicates.

---

## 8. Deliverables for the meeting

| # | Artifact | Source |
|---|---|---|
| 1 | Run summary table — design, timing, cost | §1 |
| 2 | 36-config leaderboard + baseline deltas | A1, A2 |
| 3 | Group means ± SD per factor (4 plots) | B1 |
| 4 | AUC distributions per factor level | B2 |
| 5 | Factor × factor heatmaps | B3 |
| 6 | Path ablation: 12 vs 24 runs, with balance table | D1 |
| 7 | **Dose-response: AUC vs `path_mean_norm`, all 36 runs** | D2 |
| 8 | Ontology effect table by node / relationship | C1 |
| 9 | Acceptance rate by step and prompt | C2, C3 |
| 10 | Feature-block coverage QA panel | D4 |

**Suggested order for a 30-minute meeting:** 1 → 2 → 6 → 3/4 → 8/9, with the rest held
in reserve. Lead with what was run, then the headline numbers, then the ablation —
because if D1 says path features are not helping, that reframes everything after it and
should not be buried at the end.

---

## 9. Open questions for discussion

- If path features do not improve AUC, is the right response to change the feature
  construction, the downstream task, or the claim?
- Is next-day direction the right target? It is close to a coin flip and may not have
  enough signal to distinguish KG variants at this sample size.
- Should MSFT run as-is for replication, or should the design change first based on
  what TSLA shows?
- Is `steps=7` justified? C3 answers whether late steps still contribute.
