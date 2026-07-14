# Experiment Strategy

## Evaluation Windows

| Period | Start | End | Business days |
|--------|-------|-----|---------------|
| **Training** (hparam sweep) | 2022-05-02 | 2023-08-16 | 338 |
| **Out-of-sample** (one-shot test) | 2023-08-16 | 2023-12-16 | 88 |

```
--day-start 2022-05-02 --day-end 2023-08-16   # training / sweep
--day-start 2023-08-16 --day-end 2023-12-16   # OOS test
```

Identical dates for all tickers (see manuscript figure for window visualisation). Start date 2022-05-02 is the earliest common coverage across all three tickers; the 338-day training window provides sufficient article volume, and the 4-month OOS hold-out gives 88 business days for a stable final evaluation.

---

OOS jobs use `--fixed-ontology <path>` to load the evolved schema from Phase 1.

---

## Ticker Selection

**NVDA is excluded as a primary ticker.** During the training window (2022-05-02 → 2023-08-16) NVDA undergoes a near-4× rally driven by the AI macro narrative — direction is dominated by fundamental sector demand, not article-level graph signal. This makes the prediction either trivially easy or impossible, neither of which is an interesting result.

**Primary tickers: MSFT and TSLA.**
- **MSFT**: stable large-cap, mixed correction + recovery regime, credible finance benchmark, AI-adjacent without being the pure play
- **TSLA**: volatile, Elon/sentiment-driven, exactly the case where entity-relationship graph structure should add value over bag-of-words

NVDA is kept in the dataset table and figure for completeness but is not used in the main experiments.

---

## Research Hypotheses

The sweep is organised around the following testable claims. Each hypothesis isolates one hyperparameter dimension; the **Symmetry Principle** (below) ensures all other free dimensions co-vary uniformly across every group.

**H1 — Market memory (lookback window)**  
`--lookback-days` ∈ {1, 3, 5, 7}  
AUC peaks at 3–5 days then drops. News older than ~1 week is too stale to carry directional signal in the graph.

**H2 — Evolution depth is the core claim**  
`--steps` ∈ {0, 3, 4, 5}  
AUC increases with steps up to a convergence point. steps=0 is the no-evolution baseline the method must beat.

**H3 — Multi-hop paths carry more signal than shallow ones**  
`--min-chain-hops` = `--max-chain-hops` ∈ {4, 5, 6}  
Deeper chains outperform shallower ones. Indirect entity chains encode signal not visible in direct mentions.

**H4 — Exploration breadth (candidates per step)**  
`--candidates` ∈ {1, 2, 3}  
More candidates → better ontology selected per step → better AUC. Primarily a cost/quality tradeoff.

**H6 — News volume per day**  
`--articles-per-day` ∈ {3, 5, 10, 20}  
AUC improves up to ~10 articles/day then saturates. Data efficiency: how much news does the graph need?

**H7a — Which entity categories drive performance (post-hoc)**  
Extracted from evolved ontologies after H1–H6 runs complete.  
XGBoost feature importances aggregated by semantic category (corporate-fundamental, event-driven, macro-structural) reveal which ontology additions are responsible for AUC gains. Requires adding `feature/importances` artifact logging.

**H7b — Evolution prompt strategy**  
`--evolution-prompt` ∈ {default, fundamental, event-driven, macro-context}  
Biasing ontology evolution toward corporate-fundamental entities outperforms the unstructured default. Each prompt uses the same AUC-gated threshold logic but guides the LLM toward a different semantic category cluster when proposing new types. Embedded as a co-varying dimension across **all** hypothesis groups (see Symmetry Principle below).

---

## Symmetry Principle

Each hypothesis group varies **exactly one dimension** at a time. All other free dimensions are held at their reference value.

**Evolution prompt (H7b) is not an isolated group — it co-varies with every other group.** Every config in every hypothesis group is replicated across all four prompt variants. This guarantees:

1. The effect of any single hparam is not confounded with prompt choice
2. H7b can be answered from the data of every hypothesis group, not a dedicated subset
3. The full sweep is a clean factorial: (hypothesis dimension) × prompt × ticker

**Reference values** (fixed except when explicitly varied):

| Dimension | Reference |
|-----------|-----------|
| `steps` | 3 |
| `lookback_days` | 3 |
| `min_chain_hops` / `max_chain_hops` | 5 / 5 |
| `candidates` | 2 |
| `articles_per_day` | 5 |
| `feature_mode` | hybrid (histogram + path + topology) |

**Dimensions always crossed with every group:**

| Dimension | Values |
|-----------|--------|
| `evolution_prompt` | default, fundamental, event-driven, macro-context |
| `ticker` | MSFT, TSLA |

**Exception — steps=0 baseline:** the evolution prompt is never called at steps=0, so all four prompt variants would produce identical runs. Run 1 job per ticker (2 jobs total) as the baseline rather than 4 × 2 = 8 redundant jobs.

**Job count per hypothesis group:**

| Hypothesis | Varied dim | Values | Formula | Jobs |
|------------|-----------|--------|---------|------|
| H1 | lookback_days | {1, 3, 5, 7} | 4 × 4 prompts × 2 tickers | 32 |
| H2 | steps | {0, 3, 4, 5} | 2 (baseline) + 3 × 4 × 2 | 26 |
| H3 | chain hops (min=max) | {4, 5, 6} | 3 × 4 × 2 | 24 |
| H4 | candidates | {1, 2, 3} | 3 × 4 × 2 | 24 |
| H6 | articles_per_day | {3, 5, 10, 20} | 4 × 4 × 2 | 32 |
| **Total** | | | | **138 jobs** |

---

## Open Design Questions

**Subgroup averaging instead of a single reference run.** The Symmetry Principle currently
compares every swept value against one fixed pseudo-default configuration (`steps=3,
lookback_days=3, chain hops=5, candidates=2, articles_per_day=5`), so each ablation's effect is
read off relative to a single arbitrary point rather than the spread of outcomes when other
dimensions vary too. Since `evolution_prompt` and `ticker` are already crossed with every group,
we could instead compare a swept value's mean AUC across its 8 replicate runs (4 prompts × 2
tickers) against the mean AUC of other subgroups that are otherwise identical, rather than
collapsing each group to the single reference config's AUC — this is less sensitive to the
reference point being an unrepresentative pick.

**One node+relationship pair per evolution step.** Extend the single-addition constraint (see
`planning/POST_HOC_SEMANTIC_COMPARISON.md` Idea 1) so each step proposes exactly one new node
type *and* the one relationship type that connects it, as a single coupled addition, rather than
a node type or a relationship type alone. A relationship type with no new node to reach isn't
independently attributable, and a new node type with no new relationship reaching it is inert —
pairing them keeps each step's causal unit clean and directly attributable to one ΔAUC.

**Drop non-improving winners for monotonic AUC.** Each step currently advances to the candidate
with the highest validation AUC even when that AUC is below the current best-so-far, so
step-by-step AUC plots can be non-monotonic and muddy the "evolution helps" narrative. Add a
gating rule: only adopt a step's winning candidate as the new base ontology if its AUC exceeds
the current best; otherwise keep the ontology unchanged for that step. This makes the AUC-vs-step
curve monotonic by construction, at the cost of some steps producing no schema change.

---

## Paper Figures & Tables

### Table 1 — Main ablation (from Phase 1)

| Method | AUC | F1 | Precision | Recall |
|--------|-----|-----|-----------|--------|
| Article text baseline | | | | |
| KG base (step 0) — path | | | | |
| KG base (step 0) — subgraph | | | | |
| KG evolved (step 3) — path | | | | |
| KG evolved (step 3) — subgraph | | | | |
| KG evolved (step 3) — hybrid | | | | |

### Figure 1 — AUC across ontology evolution steps

Line plot: x=step (0→3), y=AUC, one line per feature mode.  
Source: MLflow `auc` metric logged at `step=N`.

### Figure 2 — KG growth across evolution steps

Stacked bar: x=step, y=node count, colored by node type.  
Source: `graph/n_{NodeType}` metrics.

### Figure 3 — Feature importance

Horizontal bar chart of top-20 XGBoost feature importances for the best model.  
Source: `feature/importances` artifact.

### Table 2 — Cross-ticker generalization

| Ticker | Window | AUC | F1 |
|--------|--------|-----|-----|
| MSFT   | 2022-05-02 – 2023-08-16 | | |
| TSLA   | 2022-05-02 – 2023-08-16 | | |

---

## Post-hoc Semantic Attribution

**Per-candidate category importance.** Label each hash bucket in the subgraph feature vector with
the entity/relationship type(s) that map to it by forward-hashing the ontology's known type
vocabulary, then aggregate the trained XGBoost `feature_importances_` by semantic category
(corporate fundamentals, macro, supply chain, etc.) for a single evaluated candidate. Needs a
one-line addition to persist `feature_importances_` and a bucket→type labelling utility.

---

