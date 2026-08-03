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

The sweep is organised around the following testable claims. Each hypothesis isolates one hyperparameter dimension; the **Balanced Factorial Design** (below) ensures every other free dimension is *equally distributed* across the levels of the dimension under test, so each hparam's effect is estimated over a shared, representative spread of the others rather than at a single arbitrary reference point.

**H1 — Market memory (lookback window)**  
`--lookback-days` ∈ {3, 8, 10, 20}  
AUC peaks at a moderate window then drops as older news goes stale. Tested factor.

**H2 — Evolution depth is the core claim**  
`--steps` ∈ {3, 5, 7}  *(reduced from {3,5,7,9}; `steps=9` dropped — see Reduced Design below)*  
AUC increases with steps up to a convergence point. Tested factor. (The no-evolution comparison is provided per-run by the step-0 base-structure and article-text baselines that every job already logs, so steps=0 is not run as a separate config.)

**H3 — Multi-hop paths carry more signal than shallow ones**  
`--min-chain-hops` = `--max-chain-hops` ∈ {3, 5, 6}  
Deeper chains outperform shallower ones. Indirect entity chains encode signal not visible in direct mentions. Tested factor.

**H4 — Exploration breadth (candidates per step) — HELD FIXED this sweep**  
`--candidates` fixed at 3. Not a tested factor in this design; kept constant so it does not spend runs. Test separately later if needed.

**H6 — News volume per day — HELD FIXED this sweep**  
`--articles-per-day` fixed at 10. Not a tested factor in this design; kept constant. Test separately later if needed.

**H7a — Which entity categories drive performance (post-hoc)**  
Extracted from evolved ontologies after the sweep completes.  
XGBoost feature importances aggregated by semantic category (corporate-fundamental, event-driven, macro-structural) reveal which ontology additions are responsible for AUC gains. `feature_importances` artifact logging is in place.

**H7b — Evolution prompt strategy**  
`--evolution-prompt` ∈ {default, fundamental, event-driven, macro-context}  
Biasing ontology evolution toward corporate-fundamental entities outperforms the unstructured default. Each prompt uses the same AUC-gated threshold logic but guides the LLM toward a different semantic category cluster when proposing new types. Prompt is a full factor in the design, so H7b is answered from the same 36 runs as everything else.

---

## Balanced Factorial Design

**Why not one-factor-at-a-time (OFAT).** The tempting design is: to test one hparam, hold every other hparam at a fixed "reference" value and sweep only the one. That produces a clean comparison — but *only at that single reference operating point*. If, say, the best `lookback_days` is different when `articles_per_day` is high, OFAT never sees it: the other knobs never move. Each hparam's effect is then measured at one arbitrary corner of the space, and interactions are invisible.

**What we do instead — a strength-2 (pairwise-balanced) orthogonal array.** All five *tested* factors vary simultaneously, but the runs are chosen so that **for every pair of factors, every combination of their levels appears equally often**. The consequence is the property we actually want:

> For each hparam value, the *distribution of every other hparam is identical*. So `mean(AUC | lookback=8)` and `mean(AUC | lookback=3)` are each averaged over the **same** balanced spread of steps, hops, prompt, and ticker. The difference between those means is therefore attributable to `lookback` alone — a fair marginal effect, not an artifact of the reference point.

**Tested factors and levels:**

| Factor | Levels | n |
|--------|--------|---|
| `lookback_days` | 3, 8, 10, 20 | 4 |
| `steps` | 3, 5, 7 | 3 |
| `evolution_prompt` | default, fundamental, event-driven, macro-context | 4 |
| `chain_hops` (min = max) | 3, 5, 6 | 3 |
| `ticker` | MSFT, TSLA | 2 |

**Held fixed** (not under test): `candidates = 3`, `feature_mode = hybrid`, `single_addition = true`, monotonic AUC gate on (`keep_regressing_steps` off), `auc_drop_tolerance = 0.0`.

**`articles_per_day` and the date window were cut for execution reasons — see "Data-volume cut" below.** The values in this section (`articles_per_day = 10`, window `2022-05-02 → 2023-08-16`) describe the *original, intended* design; the currently-dispatching pilot uses reduced values and is explicitly flagged as such in `sweep_runs/MANIFEST.md`.

**Original construction (48 runs).** The clean balanced set was 48 runs, with `steps` as a *4-level* factor {3,5,7,9}. Perfect pairwise balance requires the run count to be divisible by every pairwise level-product (16, 12, 8, 6) — LCM **48**. Construction: a 16-run orthogonal block for the three 4-level factors (`lookback`, `steps`, `prompt`) — `L=i`, `S=j` (full 4×4), `P=(i+j) mod 4` — replicated three times using the copy index as the 3-level `chain_hops` factor; `ticker` assigned by a balancing search; verified **zero** deviation on all 10 factor pairs. That grid is `sweep_grid.xlsx` (`configs` = the 48 jobs).

**Reduced Design — run count = 36 (current).** `steps=9` was dropped to remove the 12 heaviest jobs (each `steps=9` job = 27 candidate-builds), cutting sweep cost by ~⅓. This leaves **36 runs** with `steps ∈ {3,5,7}`. Balance consequence: a *perfectly* balanced design for the new level set {4,4,3,3,2} would need **144 runs** — two 3-level factors (`steps`, `chain_hops`) force divisibility by 9, which collides with the 16 from the two 4-level factors (`lookback`, `prompt`); LCM(16,9)=144. At 36 runs the design is therefore **not perfectly balanced**, but the damage is minimal and characterised:
- All five **marginals stay uniform** (lookback 9 each, steps 12, prompt 9, chain_hops 12, ticker 18), and every factor pair among {lookback, steps, prompt, chain_hops} **remains balanced**.
- Only the three **`ticker` pairings** (`lookback×ticker`, `prompt×ticker`, `chain_hops×ticker`) are **4-vs-5** instead of even. This is *unavoidable and minimal*: with 9 configs per 4-level level, a 2-level factor cannot split evenly (9 is odd).
- **The core `steps` effect (H2) stays perfectly clean.** Only ticker-related comparisons carry a ≤1-config confound.

**Data-volume cut (2026-07-26 — 100-day pilot, superseded).** At the original `articles_per_day=10` over the full 471-day window (~3,730 articles/build), a single `steps=7` job's **step 1 alone took 90+ minutes and never finished** in live testing (see `planning/SWEEP_EXECUTION_FINDINGS.md`) — GPU-count tuning alone (tried 6, 8, 10) could not fix this; it is a workload-volume problem, not an infra one. The window was cut to **100 calendar days (2022-05-02 → 2022-08-10)** and `articles_per_day` cut **10→4** (measured ~355 articles/build, a ~10.5x reduction). Validated: 2 chunks (both `steps=3`) completed cleanly at ~70-71 min each, ~17-19 min/evolution-step — confirming the pipeline finishes end-to-end at this volume.

**Window increased to 200 days (2026-08-03 — current).** GitHub-hosted runners have a **hard, non-configurable 6h job execution cap** (confirmed against GitHub's docs — `timeout-minutes` cannot exceed it; only self-hosted runners, up to 5 days, escape it). Rather than set up a self-hosted runner, the window was increased from 100→**200 calendar days (2022-05-02 → 2022-11-18)**, keeping `articles_per_day=4` (measured ~720 articles/build, ~2.03x the 100-day pilot). Projected per-step time (~37 min, scaled from the measured ~18 min/step): `steps=3` ≈ 2.2h, `steps=5` ≈ 3.4h, `steps=7` ≈ 4.6h — all under the 6h cap, with `steps=7` the tightest at ~1.4h margin. **This supersedes the 100-day pilot's results** — the 2 previously-completed `steps=3` chunks used a different window and must be re-run for comparability; all 13 chunks restart from scratch under the 200-day design (see `sweep_runs/MANIFEST.md`).

**Execution — 13 runs to fit the 6h GitHub Actions cap, grouped by `steps`.** Runs are split into **13 `sweep.yml` dispatches** (`sweep_runs/run_01.json … run_13.json`; tracked in `sweep_runs/MANIFEST.md`), one steps-level per chunk (not mixed): 3 chunks of 4 configs at `steps=3` (~74min), 4 chunks of 3 configs at `steps=5` (~110min), 6 chunks of 2 configs at `steps=7` (~146min). Chunks were originally mixed by steps level (LPT bin-packed by total call volume), which made wall-time unpredictable — a light job finishes and frees GPU capacity while a heavy one is still running, so the chunk's wall-clock was governed by whichever config had the most steps, and packing by *total calls* doesn't balance *wall-clock*, which is step-sequential-bound. Grouping by steps makes each chunk's wall-time uniform and predictable, at the cost that same-steps configs stay synchronized through every step (no early tapering) — chunk size is shrunk as steps grows to compensate. 8 vLLM GPU workers (g5.xlarge / A10G), 5 CPU nodes, 6 embedding workers, sidecar Neo4j, semaphore 220/293/440 (scaled to configs-per-chunk). **All 13 must complete before analysis** — the balanced marginals only exist over the full 36-config set; partial runs are not interpretable for any factor. Splits are by *runtime*, never by factor (splitting by factor would be OFAT and destroy the balance).

**Calibration (single-job measurement, 2026-07-25, original full-volume window).** ~**1.7** completed extractions/s per A10G (Qwen2.5-7B-AWQ), ~**140** concurrent-sequence KV ceiling per GPU, ~**3,730** articles/build (10/day cap over the 466-day training window). Full-48 sweep ≈ 3.2M extraction calls; reduced-36 ≈ 2.0M — both figures describe the *original* volume, since abandoned per the data-volume cut above.

**Reading effects off the results.** After all 9 runs complete, pull the 36 parent runs (`best_auc` logged as a flat metric) and, for any hparam, `groupby(value).best_auc.agg(['mean','std'])`. Group means are directly comparable for lookback / steps / prompt / chain_hops (still pairwise-balanced); ticker contrasts (and any ticker-confounded comparison) should note the 4-vs-5 imbalance. Because the design is only strength-2, a paired/blocked test may block on any **single** other factor (whose margins match) — but **not** on the full joint cell of all remaining factors, which would require a full-factorial or higher-strength design.

---

## Open Design Questions

**Subgroup averaging instead of a single reference run. — RESOLVED** by the Balanced Factorial
Design above. Effects are now read off as marginal means over a balanced spread of all other
factors (`groupby(value).best_auc.mean()`), not against one fixed pseudo-default configuration.
The old OFAT/Symmetry-Principle design (each swept value compared at a single reference point) is
superseded.

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

