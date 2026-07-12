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

**Sequence:** P0 first (shapes data collected from next sweep), then P1 (adds logging only), then P2/P3 for paper figures.
