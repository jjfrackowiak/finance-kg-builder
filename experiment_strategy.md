# Experiment Strategy

## Dataset

**Source:** FNSPID sample — NASDAQ long-text articles  
**File:** `s3://kg-experiments-data-039293892587/data/fnspid_sample_nasdaq_long_text.csv`  
**Coverage:** 2021-08-17 → 2023-12-16 (26,165 articles, 3 tickers)

| Ticker | First date | Last date | Articles | Days |
|--------|-----------|-----------|----------|------|
| NVDA   | 2021-08-17 | 2023-12-16 | 8,716 | 811 |
| MSFT   | 2022-04-26 | 2023-12-16 | 8,737 | 594 |
| TSLA   | 2022-05-02 | 2023-12-16 | 8,712 | 588 |

---

## Evaluation Windows

Use `--day-start` + `--day-end` for exact reproducibility. Avoid `--time-window-days` alone in final experiments.

### Final window selection

All three tickers use **identical dates** — chosen to maximise common coverage (intersection of all three coverages) with a 4-month out-of-sample hold-out.

| Period | Start | End | Business days |
|--------|-------|-----|---------------|
| **Training** (hparam sweep) | 2022-05-02 | 2023-08-16 | 338 |
| **Out-of-sample** (one-shot test) | 2023-08-16 | 2023-12-16 | 88 |

```
--day-start 2022-05-02 --day-end 2023-08-16   # training / sweep
--day-start 2023-08-16 --day-end 2023-12-16   # OOS test (one shot, after config is locked)
```

Rationale: 2022-05-02 is the earliest date all three tickers have coverage (TSLA starts 2022-05-02). 2023-12-16 is the end of the dataset. 4 months OOS gives 88 business days — enough for a stable AUC estimate while keeping 338 days for the sweep.

Within the training window, XGBoost uses the default 75/25 train/val split — approximately 254 train days / 84 val days per run.

---

### Out-of-sample test step

After the hyperparameter sweep selects the best config and its evolved ontology:

- The **ontology schema is frozen** — no further evolution
- The **graph is rebuilt from scratch** from 2023-08-16→2023-12-16 articles using that fixed schema
- XGBoost is trained and evaluated on those 88 OOS trading days
- The same is done for the `steps=0` baseline on the identical OOS window
- Both AUCs reported as the final baseline-vs-evolved comparison in the paper

This is a single one-shot run — no tuning touches OOS data.

**Implementation note:** requires a `--fixed-ontology <path>` argument in `main.py` so the OOS job loads the evolved schema without triggering the ontology LLM.

---

## Ticker Selection

**NVDA is excluded as a primary ticker.** During the training window (2022-05-02 → 2023-08-16) NVDA undergoes a near-4× rally driven by the AI macro narrative — direction is dominated by fundamental sector demand, not article-level graph signal. This makes the prediction either trivially easy or impossible, neither of which is an interesting result.

**Primary tickers: MSFT and TSLA.**
- **MSFT**: stable large-cap, mixed correction + recovery regime, credible finance benchmark, AI-adjacent without being the pure play
- **TSLA**: volatile, Elon/sentiment-driven, exactly the case where entity-relationship graph structure should add value over bag-of-words

NVDA is kept in the dataset table and figure for completeness but is not used in the main experiments.

**Implementation:** ticker is set via `TARGET_TICKER` env var (default `TSLA` in `config.py`). Add one line to `sweep.py` `build_job_manifest()` to override it per job:
```python
if "ticker" in cfg:
    kg_builder_env.append(client.V1EnvVar(name="TARGET_TICKER", value=cfg["ticker"]))
```

---

## Sweep Configs

All configs use the finalised common window. `candidates=2` throughout Phase 1.

### Phase 1 — Ablation: feature mode × ontology depth (24 jobs)

Both tickers × all steps × all feature modes run in parallel.

```json
[
  {"ticker":"MSFT","steps":0,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":0,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":0,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":1,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":1,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":1,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":2,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":2,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":2,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":3,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":3,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"MSFT","steps":3,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":0,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":0,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":0,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":1,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":1,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":1,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":2,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":2,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":2,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":3,"candidates":2,"feature_mode":"path",    "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":3,"candidates":2,"feature_mode":"subgraph","day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10},
  {"ticker":"TSLA","steps":3,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-05-02","day_end":"2023-08-16","articles_per_day":10}
]
```

### Phase 2 — OOS test (after Phase 1 config is locked, 4 jobs)

Best `steps` + `feature_mode` from Phase 1, both tickers, evolved vs. baseline:

```json
[
  {"ticker":"MSFT","steps":0,        "candidates":2,"feature_mode":"<best>","day_start":"2023-08-16","day_end":"2023-12-16","articles_per_day":10},
  {"ticker":"MSFT","steps":"<best>", "candidates":2,"feature_mode":"<best>","day_start":"2023-08-16","day_end":"2023-12-16","articles_per_day":10},
  {"ticker":"TSLA","steps":0,        "candidates":2,"feature_mode":"<best>","day_start":"2023-08-16","day_end":"2023-12-16","articles_per_day":10},
  {"ticker":"TSLA","steps":"<best>", "candidates":2,"feature_mode":"<best>","day_start":"2023-08-16","day_end":"2023-12-16","articles_per_day":10}
]
```

The evolved runs use `--fixed-ontology <path>` to load the schema from Phase 1 without re-running the LLM.

**Total: 3 jobs.**

### Phase 4 — Cross-ticker (best full config)

Run the winning config from Phase 3 on MSFT and TSLA windows.  
**Total: 2 jobs.**

---

## MLflow Logging Plan

### Currently logged (per nested run)
- `params`: `steps`, `candidates`, `feature_mode`, `lookback_days`, `node_types`, `rel_types`, `n_node_types`, `n_rel_types`, all CLI args
- `metrics`: `auc`, `f1`, `max_hops_train`, `max_hops_val`
- `tags`: `winner`, `step_winner`
- `artifacts`: ontology JSON files

### To add (see TODO below)

| Metric key | When | Why |
|-----------|------|-----|
| `graph/n_nodes` | after each extraction | KG growth story |
| `graph/n_edges` | after each extraction | KG growth story |
| `graph/n_{NodeType}` | after each extraction | which types appear |
| `graph/articles_with_entities` | after each extraction | coverage |
| `timing/extraction_s` | per step | cost vs gain |
| `timing/evolution_s` | per step | LLM call cost |
| `timing/evaluation_s` | per step | eval overhead |
| `eval/precision` | per candidate | paper Table 1 |
| `eval/recall` | per candidate | paper Table 1 |
| `eval/brier_score` | per candidate | calibration |
| `eval/n_train_days` | per candidate | reproducibility |
| `eval/n_val_days` | per candidate | reproducibility |
| `eval/class_balance` | per candidate | dataset sanity |
| `ontology/n_node_types_added` | per step | evolution delta |
| `ontology/n_rel_types_added` | per step | evolution delta |
| `feature/importances` | per candidate | artifact JSON |

**Log metrics with `step=N`** (the evolution step number) so DagsHub renders the AUC-vs-step line chart automatically.

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

### Table 2 — Cross-ticker generalization (Phase 4)

| Ticker | Window | AUC | F1 |
|--------|--------|-----|-----|
| NVDA   | 2022   | | |
| MSFT   | H2 2022–H1 2023 | | |
| TSLA   | H2 2022–H1 2023 | | |

---

## Semantic–Metric Comparison (Research Idea)

The core research contribution is that the system discovers *which semantic categories of information are predictive* by evolving the ontology schema. The ideas below describe how to make that claim testable and visualisable for the paper.

### The central claim

> "Task-aware ontology evolution autonomously discovers that [Category X] entities predict stock direction, while [Category Y] do not — without manual feature engineering."

This requires three things: (1) constrained evolution so each step adds one identifiable thing, (2) feature attribution by semantic category, and (3) structured LLM output that encodes a hypothesis per candidate.

---

### Idea 1 — Ontology Delta → AUC Delta (evolution trace)

**What it shows:** Which specific entity/relationship additions caused AUC to improve or regress.

**Problem with current setup:** The LLM proposes a full ontology variant (many changes at once). If three types are added simultaneously and AUC goes up, you cannot attribute the gain to any single one.

**Fix — constrain the evolution prompt to one addition per candidate:**

Add a constraint block to `resources/evolution_prompt_template.txt`:

```
CONSTRAINT: Propose adding exactly ONE new node type OR ONE new relationship type per candidate.
Do not modify, rename, or remove existing types. If you propose a relationship type,
specify exactly which existing node types it connects.
```

**Structured output to require** (modify the prompt to enforce this JSON schema):

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
- `kg_builder_llm/pipeline/ontology_evolution.py` — parse `proposed_addition` from LLM response; store on `OntologyCandidate`
- `kg_builder_llm/core/ontology_io.py` — persist `proposed_addition` in saved ontology JSON
- `kg_builder_llm/pipeline/orchestrator.py` — `_log_candidate_to_mlflow`: log `semantic_category`, `hypothesis`, `addition_name`, `addition_type` as params on the nested run

**Resulting paper artefact — Table (evolution trace):**

| Step | Addition | Category | Hypothesis | ΔAUC |
|------|----------|----------|------------|------|
| 1 | CreditRating | Corporate | Rating changes lead price | +0.031 |
| 2 | MacroEvent | Macro | Fed decisions move sectors | −0.008 |
| 3 | SupplyChainActor | Micro | Supply disruptions → inventory signal | +0.019 |

---

### Idea 2 — Feature Importance × Semantic Category

**What it shows:** Which semantic categories of KG features drive XGBoost predictions.

**How it works:** The subgraph feature vector has a fixed layout (`ml/subgraph_features.py`):

```
dims  0–15   node type histogram      (16 hash buckets)
dims 16–39   rel type histogram       (24 hash buckets)
dims 40–87   metapath counts 2–3 hop  (48 hash buckets)
dims 88–95   temporal novelty stats   (8 scalars)
```

The hash bucketing maps entity type strings → bucket index deterministically. Since each entity type has a known string, we can compute `bucket = hash(type_name) % n_buckets` and assign bucket → type name. Then we aggregate XGBoost `feature_importances_` by semantic category.

**Pre-defined semantic categories:**

| Category | Entity / Rel types |
|----------|--------------------|
| `corporate_fundamentals` | Company, CreditRating, Sector |
| `macro_systematic` | MacroEvent, RegulatoryBody, Index |
| `supply_chain` | SupplyChainActor |
| `text_surface` | Article, Author, text embeddings |
| `temporal` | Day, price labels, temporal novelty stats |

**Files to modify:**
- `kg_builder_llm/ml/subgraph_features.py` — export `SEMANTIC_CATEGORY_MAP: dict[str, str]` and `bucket_to_type(bucket_idx, n_buckets) -> str`; add `feature_names() -> list[str]` that labels every dimension
- `kg_builder_llm/ml/modeling.py` — after `model.fit()`, extract `model.feature_importances_`, build a dict `{feature_name: importance}`, log as MLflow artifact `feature_importances.json`; also aggregate by category and log as `category_importances.json`
- `kg_builder_llm/pipeline/orchestrator.py` — pass `feature_names` through to the training call

**Resulting paper artefact — Figure (importance by category):**

Horizontal bar chart: semantic category on y-axis, mean XGBoost importance on x-axis. Error bars across candidates. Shows e.g. *"corporate_fundamentals accounts for 41% of predictive signal; macro_systematic for 8%."*

---

### Idea 3 — Semantic Contribution Network (diagram)

**What it shows:** The KG schema as a graph, nodes coloured by AUC contribution — a visual proof that the evolved schema is shaped by task performance.

**How it works:**
- Run ablation: for each entity type T, mask its features (zero out its histogram buckets), re-evaluate, compute `ΔAUC = AUC_full − AUC_masked`
- Node size = frequency in KG; node colour = ΔAUC (green positive, grey neutral, red negative)
- Edges = relationship types between entity types

**Files to modify / add:**
- `kg_builder_llm/pipeline/evaluator.py` — add `evaluate_candidate_ablation(driver, ..., mask_types: list[str]) -> ModelMetrics`; loops over entity types, zeros their feature buckets, returns per-type ΔAUC dict
- `kg_builder_llm/pipeline/orchestrator.py` — after final candidate selection, call ablation evaluator; log `ablation/{type_name}` as MLflow metrics
- New script `kg_builder_llm/scripts/plot_semantic_network.py` — reads MLflow run, builds networkx graph from ontology JSON + ablation metrics, renders with matplotlib/pyvis

**Resulting paper artefact — Figure (semantic network):**

Network diagram: entity types as nodes, relationship types as labelled edges, node colour = AUC contribution. One diagram for base ontology vs. one for best evolved ontology — shows visually which types the evolution added and whether they are green (helpful) or grey.

---

### Idea 4 — Structured Hypothesis Testing Experiment

An explicit hypothesis-testing sweep where each job tests one semantic category:

```json
[
  {"steps":1,"candidates":1,"feature_mode":"subgraph",
   "day_start":"2022-01-03","day_end":"2023-01-03",
   "evolution_prompt":"prompts/hypothesis_corporate.txt"},
  {"steps":1,"candidates":1,"feature_mode":"subgraph",
   "day_start":"2022-01-03","day_end":"2023-01-03",
   "evolution_prompt":"prompts/hypothesis_macro.txt"},
  {"steps":1,"candidates":1,"feature_mode":"subgraph",
   "day_start":"2022-01-03","day_end":"2023-01-03",
   "evolution_prompt":"prompts/hypothesis_supply_chain.txt"}
]
```

Each prompt biases the LLM toward proposing types in that category. Controlled comparison of AUC with each category added = direct hypothesis test.

**Files to modify:**
- `resources/` — add `hypothesis_corporate.txt`, `hypothesis_macro.txt`, `hypothesis_supply_chain.txt` prompt variants
- `kg_builder_llm/scripts/sweep.py` — pass `evolution_prompt` from cfg dict through to `main_cmd` as `--evolution-prompt {cfg['evolution_prompt']}`
- `kg_builder_llm/main.py` — `--evolution-prompt` already exists; just ensure it's threaded into the sweep cfg

---

### Implementation Priority

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
| **P3** | Pass `evolution_prompt` through sweep config | Low | `sweep.py` |

**Recommended sequence:** P0 first (changes how data is collected from the next sweep onward), then P1 (adds logging without changing the experiment), then P2/P3 for the paper figures.

---

## Infrastructure Notes

- **Neo4j mode**: use `sidecar` (ephemeral per job) for all sweep phases
- **vLLM replicas** (`n_workers`): 1 for ≤6 parallel jobs; scale to 3–5 for Phase 1 full 12-job run
- **GPU nodes**: 1 per vLLM replica (g5.xlarge, auto-scaled)
- **Estimated runtime per job**: ~25–40 min at `--articles-per-day 10` over 365 days
- **Phase 1 total wall-clock**: ~40 min (12 jobs parallel on 2–3 GPU nodes)
- **MLflow experiment name**: `kg-sweep` (already created on DagsHub)
