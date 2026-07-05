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

---

### The validation leakage problem — and whether we need a test set

#### The double-role problem

The current design uses one internal train/val split per run. The **validation AUC serves two roles simultaneously**:

1. **Within-run oracle** — at each evolution step the system picks the candidate ontology that scores highest on val. The winning ontology was literally selected *because* it scored well on val → val AUC for the winner is inflated.
2. **Cross-run comparison metric** — when comparing hyperparameter configs across the sweep (steps=1 vs steps=2, path vs subgraph), we compare val AUCs. But each run's val AUC is already inflated by role 1, and we then pick the best-looking run → a second layer of optimism.

This is the winner's curse stacked twice: ontology selection inflates val AUC, and then hyperparameter selection inflates it again.

#### Do we actually need a test set?

It depends on the claim:

| Claim type | Test set needed? | Why |
|------------|-----------------|-----|
| **Methodological**: "task-aware evolution finds semantically meaningful entity types" | No | The claim is about *what* the system discovered, not *how well* it predicts. The evolved schema's structure is valid regardless of AUC optimism. |
| **Performance**: "our method achieves AUC X on stock direction" | Yes | Any AUC you report is optimistic. Without a held-out set, a reader cannot interpret the number. |
| **Comparative**: "steps=2 beats steps=0 (baseline)" | Partially | If baseline (steps=0) uses the same val split but the split doesn't drive its selection (there's nothing to select — it's always the same ontology), the baseline AUC is honest and the evolved AUC is inflated → comparison is biased in favour of evolution. A test set fixes this. |

**Recommendation for this paper:** the primary contribution is methodological — the *process* of linking ontology evolution to task performance, and the semantic analysis of what the system discovers. A leaderboard-style performance number is not the point. Therefore:

- Report val AUC as an **internal evolution signal**, not as a claimed generalisation metric
- Use the 2023-H1 temporal hold-out as a **single one-shot test** for the final chosen configuration, looked at exactly once, after all hyperparameter choices are locked
- Be explicit in the paper: *"validation AUC guides ontology selection and is therefore optimistic; we report out-of-sample AUC on a held-out period to bound the generalisation error"*

#### Design options

**Option A — Two-stage temporal split (current design, repaired)**

```
|── 2022-01-03 ──────── 2023-01-03 ──|── 2023-01-03 ── 2023-07-01 ──|
         train+val (evolution signal)          test (one shot)
```

- Sweep runs freely over the train+val window to tune everything
- After the best config is chosen, run once on 2023 window and report that AUC
- **Problem**: the 2023 window has a very different regime (AI bull rally) from 2022 (correction) — the gap may be confounded by regime shift, not model quality

**Option B — Three-way split within each run**

Split each run window into train (60%) / val (20%) / test (20%). Val drives ontology selection; test reports the honest run-level AUC.

```
190 train days / 42 val days / 20 test days   (from 365 calendar days)
```

- 20 test trading days → AUC CI ≈ ±0.10 — **too few for reliable reporting**
- Not viable at our data size

**Option C — Walk-forward cross-validation (honest but expensive)**

Roll a 6-month train window forward month by month, evaluate on the next month. Aggregate AUC over 6 folds → less variance, no snooping on any single val set.

- Each fold requires a full pipeline run (LLM calls, graph build, train) — ~6× cost
- Works for baseline comparison (steps=0 folds = clean AUC); evolved folds still have within-run ontology snooping, just spread across time
- Best option if performance claim is central to the paper

**Option D — Reframe: no test set, embrace methodological framing**

Do not report a generalisation AUC at all. Instead:

- Report val AUC *trends* (does evolution improve over baseline consistently across tickers?)
- Report the semantic analysis (which categories contribute, evolution trace)
- Note the leakage explicitly as a limitation and motivate Option C as future work

This is defensible: many KG + finance papers (including ding2015stock, ding2016kgeventstock) report validation-era performance and frame the result as *"the model learned useful structure"*, not *"use this in production"*.

#### Recommended approach for the paper

1. **Primary evaluation frame**: methodological and structural — what did the ontology evolve to, which semantic categories carry signal (Semantic-Metric Analysis section above)
2. **Secondary evaluation frame**: val AUC trends across configs, explicitly labelled as "optimistic upper bound" in the paper
3. **One honest number**: run the final winning config on the 2023-H1 hold-out, report it as *"out-of-sample AUC"* with the caveat that the window has a different regime
4. **Limitation**: acknowledge the within-run val leakage; propose walk-forward CV as future work

This avoids needing a formal test set while being honest about what the numbers mean.

---

### How to pick the best days

Choosing dates is not arbitrary — bad choices produce uninterpretable AUC and inflate or mask the KG's contribution. The criteria below guide selection.

#### Criterion 1 — Sufficient validation sample size

The XGBoost model is evaluated on held-out trading days. AUC on fewer than ~50 days has high variance (±0.05 swings from 10-day differences are common). Minimum thresholds:

| Val days | Minimum total window | Notes |
|----------|---------------------|-------|
| 50 | ~200 calendar days | Lower bound; AUC CI ≈ ±0.07 |
| 62 | ~365 calendar days | Safe for reporting; 75/25 split of ~252 trading days |
| 100+ | ~550 calendar days | For tight CIs or significance testing |

Our 75/25 split means: `val_trading_days ≈ window_calendar_days × (252/365) × 0.25`

#### Criterion 2 — Regime homogeneity

Mixing bull-market and bear-market regimes in a single window degrades the signal — the model must learn two different regimes simultaneously. Prefer windows where the price trend is roughly monotone (sustained up or sustained down), not a V-shape.

**Check:** Plot the daily closing price for the ticker over the candidate window before committing. Look for:
- Single dominant trend (avoid windows that cross a major inflection by more than 20% of the range)
- No data gaps > 5 trading days (earnings blackouts, halts)

For NVDA specifically:
- **2021-08** → **2022-01**: strong bull (avoid — short, ends at peak)
- **2022-01** → **2023-01**: sustained correction then partial recovery — **heterogeneous but high volume**; acceptable because the vol itself is informative
- **2023-01** → **2023-07**: AI-driven bull rally — high information content, directional
- **2022-07** → **2023-07**: straddles inflection — best for generalization tests (forces robustness)

#### Criterion 3 — News volume per day

The KG is only as good as the extraction input. Thin news days produce sparse graphs with no signal.

**Check** (run locally before committing a window):
```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/fnspid_sample_nasdaq_long_text.csv", parse_dates=["Date"])
df = df[df["Ticker"] == "NVDA"]
df = df[(df["Date"] >= "2022-01-03") & (df["Date"] < "2023-01-03")]
print(df.groupby("Date").size().describe())
EOF
```
Accept windows where **p25 ≥ 3 articles/day** and **mean ≥ 6 articles/day**. Below p25 < 2, many training days produce no graph and XGBoost sees NaN rows.

#### Criterion 4 — Avoid earnings confounders at window boundaries

Starting or ending a window immediately before/after an earnings release creates a distributional shift at the boundary (spike in articles + discontinuous price move). The model may learn the spike, not the KG. Leave ≥ 3 weeks buffer from major earnings dates.

NVDA earnings dates (approximate): Feb 16, May 24, Aug 23, Nov 16 (repeat annually).

#### Criterion 5 — Cross-ticker alignment

When comparing NVDA, MSFT, TSLA, use the **same calendar window** for all three. Different windows → different regimes → incomparable AUC. The cross-ticker window must be contained within each ticker's coverage.

All three tickers have data from 2022-07-01 onward → use that as the universal start for comparisons.

---

### Recommended windows (final)

#### Development / smoke-test
```
--day-start 2021-08-17 --day-end 2021-09-17   # 30 days, NVDA only, ~70 articles
```
Purpose: fast iteration, CI validation. **Not used in paper.**

#### Main experiment window (NVDA)
```
--day-start 2022-01-03 --day-end 2023-01-03   # calendar year 2022, ~365 days
```
- ~3,000 articles (up to 10/day via `--articles-per-day 10`)
- ~252 trading days → ~190 train / ~62 val at 75/25 split
- Criterion check: vol + correction regime ✓; p25 news volume ✓; no major boundary confounders ✓

#### Out-of-sample hold-out (NVDA)
```
--day-start 2023-01-03 --day-end 2023-07-01   # H1 2023, ~180 days
```
- Used for final model evaluation only — never tuned on this.
- AI-driven bull regime: tests whether KG learned structural signal, not regime-specific bias.

#### Cross-ticker validation
```
MSFT: --day-start 2022-07-01 --day-end 2023-07-01
TSLA: --day-start 2022-07-01 --day-end 2023-07-01
NVDA: --day-start 2022-07-01 --day-end 2023-07-01  # (same window for comparability)
```
- Tests generalization of best ontology/feature config to other tickers.
- All three windows are identical calendar → AUC is directly comparable across tickers.

#### How to validate a candidate window before running a sweep

```python
import pandas as pd
import yfinance as yf  # or use pre-downloaded price CSV

df = pd.read_csv("data/fnspid_sample_nasdaq_long_text.csv", parse_dates=["Date"])
ticker = "NVDA"; start = "2022-01-03"; end = "2023-01-03"

articles = df[(df.Ticker == ticker) & (df.Date >= start) & (df.Date < end)]
daily = articles.groupby("Date").size()
print("Articles/day:", daily.describe())

# Price check
px = yf.download(ticker, start=start, end=end)["Close"]
print("Price range:", px.min().item(), "–", px.max().item())
print("Max drawdown:", ((px / px.cummax()) - 1).min().item())
```

Accept the window if:
- `daily.quantile(0.25) >= 3`
- `daily.mean() >= 6`
- `len(px) >= 180` (trading days)
- Inspect the price plot for obvious V-shapes spanning >50% of window

---

## Hyperparameter Grid

### Phase 1 — Feature mode & ontology depth (main ablation)

Fixed: `--day-start 2022-01-03 --day-end 2023-01-03 --candidates 2 --articles-per-day 10`

| `steps` | `feature_mode` | Configs |
|---------|---------------|---------|
| 0       | path, subgraph, hybrid | 3 |
| 1       | path, subgraph, hybrid | 3 |
| 2       | path, subgraph, hybrid | 3 |
| 3       | path, subgraph, hybrid | 3 |

**Total: 12 jobs.** This is the core ablation: does adding evolution steps help, and which feature representation benefits most?

Sweep config JSON:
```json
[
  {"steps":0,"candidates":2,"feature_mode":"path",    "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":0,"candidates":2,"feature_mode":"subgraph","day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":0,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":1,"candidates":2,"feature_mode":"path",    "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":1,"candidates":2,"feature_mode":"subgraph","day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":1,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":2,"candidates":2,"feature_mode":"path",    "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":2,"candidates":2,"feature_mode":"subgraph","day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":2,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":3,"candidates":2,"feature_mode":"path",    "day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":3,"candidates":2,"feature_mode":"subgraph","day_start":"2022-01-03","day_end":"2023-01-03"},
  {"steps":3,"candidates":2,"feature_mode":"hybrid",  "day_start":"2022-01-03","day_end":"2023-01-03"}
]
```

### Phase 2 — Feature engineering hyperparameters (best config from Phase 1)

Fixed: best `steps` + `feature_mode` from Phase 1.

| `lookback_days` | `min_chain_hops` | `max_chain_hops` |
|----------------|-----------------|-----------------|
| 1              | 3               | 5               |
| 2              | 4               | 6               |
| 3              | 5               | 7               |

**Total: 9 jobs** (3×3 grid, other dim fixed at default).

### Phase 3 — Candidates per step (best config from Phase 2)

| `candidates` | Note |
|-------------|------|
| 1           | greedy (current) |
| 2           | current default |
| 3           | more exploration |

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
