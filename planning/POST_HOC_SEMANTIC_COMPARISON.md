# Post-hoc Semantic Comparison

The core claim: the system autonomously discovers which semantic categories of information are predictive by evolving the ontology schema — without manual feature engineering.

> "Task-aware ontology evolution discovers that [Category X] entities predict stock direction, while [Category Y] do not."

This requires: (1) constrained evolution so each step adds one identifiable thing, (2) feature attribution by semantic category, and (3) structured LLM output encoding a hypothesis per candidate.

---

## Idea 1 — Ontology Delta → AUC Delta (evolution trace)

**What it shows:** Which specific entity/relationship additions caused AUC to improve or regress.

**Problem:** The LLM currently proposes a full ontology variant (many changes at once). If three types are added simultaneously and AUC goes up, you cannot attribute the gain to any single one.

**Fix — constrain the evolution prompt to one addition per candidate:**

Add to `resources/evolution_prompt_template.txt`:
```
CONSTRAINT: Propose adding exactly ONE new node type OR ONE new relationship type per candidate.
Do not modify, rename, or remove existing types. If you propose a relationship type,
specify exactly which existing node types it connects.
```

**Structured output schema:**
```json
{
  "proposed_addition": {
    "type": "node_type",
    "name": "CreditRating",
    "semantic_category": "corporate_fundamentals",
    "hypothesis": "Rating changes and outlooks lead equity price moves by 1–3 days",
    "connects_to": ["Company"],
    "extraction_template": "Extract: agency name, rating action (upgrade/downgrade/affirm), outlook, affected entity"
  },
  "full_schema": { ... }
}
```

**Files to modify:**
- `resources/evolution_prompt_template.txt` — add constraint + structured output spec
- `kg_builder_llm/pipeline/ontology_evolution.py` — parse `proposed_addition`; store on `OntologyCandidate`
- `kg_builder_llm/core/ontology_io.py` — persist `proposed_addition` in saved ontology JSON
- `kg_builder_llm/pipeline/orchestrator.py` — log `semantic_category`, `hypothesis`, `addition_name`, `addition_type` as MLflow params on nested run

**Paper artefact — evolution trace table:**

| Step | Addition | Category | Hypothesis | ΔAUC |
|------|----------|----------|------------|------|
| 1 | CreditRating | Corporate | Rating changes lead price | +0.031 |
| 2 | MacroEvent | Macro | Fed decisions move sectors | −0.008 |
| 3 | SupplyChainActor | Micro | Supply disruptions → inventory signal | +0.019 |

---

## Idea 2 — Feature Importance × Semantic Category

**What it shows:** Which semantic categories of KG features drive XGBoost predictions.

**Feature vector layout** (`ml/subgraph_features.py`):
```
dims  0–15   node type histogram      (16 hash buckets)
dims 16–39   rel type histogram       (24 hash buckets)
dims 40–87   metapath counts 2–3 hop  (48 hash buckets)
dims 88–95   temporal novelty stats   (8 scalars)
```

Hash bucketing maps entity type strings → bucket index deterministically, so we can assign bucket → type name and aggregate XGBoost `feature_importances_` by semantic category.

**Semantic categories:**

| Category | Entity / Rel types |
|----------|--------------------|
| `corporate_fundamentals` | Company, CreditRating, Sector |
| `macro_systematic` | MacroEvent, RegulatoryBody, Index |
| `supply_chain` | SupplyChainActor |
| `text_surface` | Article, Author, text embeddings |
| `temporal` | Day, price labels, temporal novelty stats |

**Files to modify:**
- `kg_builder_llm/ml/subgraph_features.py` — export `SEMANTIC_CATEGORY_MAP`, `bucket_to_type()`, `feature_names()`
- `kg_builder_llm/ml/modeling.py` — after `model.fit()`, log `feature_importances.json` + `category_importances.json` as MLflow artifacts
- `kg_builder_llm/pipeline/orchestrator.py` — pass `feature_names` through to training

**Paper artefact:** horizontal bar chart, semantic category on y-axis, mean XGBoost importance on x-axis, error bars across candidates.

---

## Idea 3 — Semantic Contribution Network (diagram)

**What it shows:** KG schema as a graph, nodes coloured by AUC contribution.

**How:** For each entity type T, mask its features (zero out histogram buckets), re-evaluate, compute `ΔAUC = AUC_full − AUC_masked`. Node size = frequency; node colour = ΔAUC (green positive, grey neutral, red negative).

**Files to modify / add:**
- `kg_builder_llm/pipeline/evaluator.py` — add `evaluate_candidate_ablation(driver, ..., mask_types) -> dict[str, float]`
- `kg_builder_llm/pipeline/orchestrator.py` — call ablation after final selection; log `ablation/{type_name}` as MLflow metrics
- `kg_builder_llm/scripts/plot_semantic_network.py` (new) — reads MLflow run, builds networkx graph from ontology JSON + ablation metrics, renders with matplotlib/pyvis

**Paper artefact:** base ontology vs. best evolved ontology network diagram showing which types were added and whether they are helpful.

---

## Idea 4 — Structured Hypothesis Testing Experiment

Each sweep job tests one semantic category by biasing the evolution prompt:

```json
[
  {"steps":1,"candidates":1,"feature_mode":"hybrid","day_start":"2022-05-02","day_end":"2023-08-16","evolution_prompt":"resources/hypothesis_corporate.txt"},
  {"steps":1,"candidates":1,"feature_mode":"hybrid","day_start":"2022-05-02","day_end":"2023-08-16","evolution_prompt":"resources/hypothesis_macro.txt"},
  {"steps":1,"candidates":1,"feature_mode":"hybrid","day_start":"2022-05-02","day_end":"2023-08-16","evolution_prompt":"resources/hypothesis_supply_chain.txt"}
]
```

**Files to add:**
- `resources/hypothesis_corporate.txt`, `resources/hypothesis_macro.txt`, `resources/hypothesis_supply_chain.txt`

---

## Idea 5 — Population-Level Category–AUC Correlation (no schema changes required)

**What it shows:** Whether candidates whose model relies more heavily on a given semantic
category of KG features tend to achieve higher AUC — using only signals every sweep run already
produces, or can produce with a one-line addition: final validation AUC, the trained classifier's
`feature_importances_`, and the candidate's persisted ontology definition. Unlike Ideas 1, 3, and
4, this requires no evolution-prompt constraint, no structured `hypothesis`/`semantic_category`
LLM output, and no masked-feature ablation re-evaluation — it is purely observational, computed
after the sweep has already run.

**Mechanism:**
1. *Offline bucket labelling, no runtime changes.* For each completed candidate, load its stored
   ontology JSON (`T_N ∪ T_R` types) and forward-hash every type string through the same
   `_hash_bucket` function already used in `ml/subgraph_features.py`, producing a
   `bucket → {type_name, ...}` map specific to that candidate. This is possible without any new
   instrumentation because the full type vocabulary is already persisted per candidate
   (`core/ontology_io.py`) — nothing needs to be logged that isn't logged today. Reuse the
   `SEMANTIC_CATEGORY_MAP` from Idea 2 to fold buckets into coarse categories.
2. *Feature importances, one-line addition.* Extract `model.feature_importances_` right after
   `model.fit()` in `ml/modeling.py` — no architecture or LLM-output change, just persisting an
   array that already exists in memory at evaluation time.
3. *Per-candidate category score.* Sum `feature_importances_` over the buckets attributed to each
   category (step 1), giving one importance share per category per candidate.
4. *Population-level correlation.* Across all completed sweep candidates (pooled, not a single
   step-by-step trace), correlate each category's importance share with candidate AUC — Spearman
   correlation or a simple OLS per category is enough; no need to fit anything more complex.

**Caveats to state honestly:**
- Only meaningful for `subgraph`/`hybrid` feature-mode runs; the `path` mode's 384-dim embedding
  vector isn't hash-bucketed to types and can't be labelled this way.
- Node-type (16 buckets) and relation-type (24 buckets) histograms have low collision risk for
  typical ontology sizes; metapath buckets (48) are far more collision-prone given the
  combinatorial number of possible typed 2-/3-hop paths — treat metapath-bucket attributions as
  noisier than node/relation-type ones, not as ground truth.
- This is correlational across the candidate population, not a causal per-type attribution (that
  is what Idea 3's ablation is for) — report as "associated with," not "causes."

**Files to modify:**
- `kg_builder_llm/ml/subgraph_features.py` — add `bucket_to_type(ontology_dict, buckets) ->
  dict[int, list[str]]`, forward-hashing every type string from a stored ontology definition
  (reuses `_hash_bucket`); reuse/extend `SEMANTIC_CATEGORY_MAP` from Idea 2.
- `kg_builder_llm/ml/modeling.py` — persist `feature_importances_` alongside AUC/F1 after
  `model.fit()`.
- new `kg_builder_llm/scripts/semantic_category_auc_correlation.py` — post-hoc script: pull AUC +
  feature importances + ontology JSON per candidate from MLflow/Neo4j, apply `bucket_to_type`,
  aggregate by category, compute correlation with AUC across the full sweep.

**Paper artefact:** bar chart, semantic category on the x-axis, correlation with candidate AUC on
the y-axis, computed across the full sweep (MSFT and TSLA shown separately or pooled).

---

## Implementation Priority

| Priority | Work item | Effort | Files |
|----------|-----------|--------|-------|
| **P0** | Constrained single-addition prompt | Low | `resources/evolution_prompt_template.txt` |
| **P0** | Structured `proposed_addition` JSON in LLM output | Medium | `ontology_evolution.py`, `orchestrator.py` |
| **P1** | Log `semantic_category` + `hypothesis` to MLflow | Low | `orchestrator.py` |
| **P1** | Feature dimension labelling by semantic type | Medium | `subgraph_features.py` |
| **P1** | XGBoost importance → category aggregation + MLflow artifact | Medium | `modeling.py`, `orchestrator.py` |
| **P2** | Ablation evaluator (mask per entity type) | High | `evaluator.py`, `orchestrator.py` |
| **P2** | Semantic network plot script | Medium | new `scripts/plot_semantic_network.py` |
| **P3** | Per-category hypothesis prompt variants | Low | `resources/hypothesis_*.txt` |
| **P0'** | Bucket→type labelling + feature-importance logging (Idea 5) | Low | `subgraph_features.py`, `modeling.py`, new `scripts/semantic_category_auc_correlation.py` |

**Sequence:** P0 first (shapes data collected from next sweep), then P1 (adds logging only), then
P2/P3 for paper figures. P0' (Idea 5) needs no prompt/schema changes and works retroactively on
any already-completed sweep — it can run in parallel with P0, ahead of Ideas 1–4, as a cheap first
result to validate before investing in the heavier LLM-output changes.
