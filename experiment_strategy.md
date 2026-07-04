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

### Development / smoke-test window
```
--day-start 2021-08-17 --day-end 2021-09-17   # 30 days, NVDA only, ~70 articles
```
Purpose: fast iteration, CI validation. **Not used in paper.**

### Main experiment window (NVDA)
```
--day-start 2022-01-03 --day-end 2023-01-03   # calendar year 2022, ~365 days
```
- ~3,000 articles (up to 10/day via `--articles-per-day 10`)
- ~252 trading days → ~190 train / ~62 val at 75/25 split
- Rationale: post-chip-shortage period, high NVDA news volume, volatile price action

### Out-of-sample hold-out (NVDA)
```
--day-start 2023-01-03 --day-end 2023-07-01   # H1 2023, ~180 days
```
- Used for final model evaluation only — not tuned on this.

### Cross-ticker validation
```
MSFT: --day-start 2022-07-01 --day-end 2023-07-01
TSLA: --day-start 2022-07-01 --day-end 2023-07-01
```
- Tests generalization of best ontology/feature config to other tickers.

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

## Infrastructure Notes

- **Neo4j mode**: use `sidecar` (ephemeral per job) for all sweep phases
- **vLLM replicas** (`n_workers`): 1 for ≤6 parallel jobs; scale to 3–5 for Phase 1 full 12-job run
- **GPU nodes**: 1 per vLLM replica (g5.xlarge, auto-scaled)
- **Estimated runtime per job**: ~25–40 min at `--articles-per-day 10` over 365 days
- **Phase 1 total wall-clock**: ~40 min (12 jobs parallel on 2–3 GPU nodes)
- **MLflow experiment name**: `kg-sweep` (already created on DagsHub)
