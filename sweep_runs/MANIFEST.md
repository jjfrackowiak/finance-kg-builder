# Sweep Run Manifest — fully balanced 72-config design, 200-day window (2026-08-07)

**Design rebuilt 2026-08-07 to be a genuine strength-2 mixed orthogonal array.** The previous
design used four evolution-prompt variants, which made exact pairwise balance impossible at this
run count: lookback (4 levels) against prompt (4 levels) needs the run count divisible by 16,
while steps against chain-hops needs divisibility by 9, so full balance would have required 144
runs per ticker rather than 36. Dropping one prompt variant (`macro_context`) to three levels
makes the requirement exactly 36 per ticker — the budget already in place.

The design is now, verified directly against these files:
- per ticker: `OA(36, 4^1 3^3, 2)` — all 6 factor pairs balanced (lambda = 3 or 4)
- pooled:     `OA(72, 4^1 3^3 2^1, 2)` — all 10 factor pairs balanced (lambda = 6, 8, 9 or 12)

Construction: the standard 9-run orthogonal array on (prompt, steps, chain_hops), each a 3-level
factor, crossed with all 4 lookback levels; repeated per ticker. 36 runs per ticker is one third
of the 108-run full grid.

Practical consequence: comparing any two levels of any hyperparameter, the other hyperparameters
are distributed identically across the two groups, so the difference is attributable to that
hyperparameter alone. No regression adjustment and no caveats are needed for any dimension.

**Also note the APOC fix (see `planning/SWEEP_EXECUTION_FINDINGS.md`).** All runs prior to
2026-08-07 are invalid: the Neo4j sidecar lacked the APOC plugin, so path-embedding features were
silently always the zero vector and entity deduplication never ran. Fixed and verified against a
live test pod before this restart.

**Execution order: TSLA first.** Chunks 01-13 are all TSLA (36 configs), chunks 14-26 all MSFT.
A complete, self-contained TSLA analysis is therefore possible once chunk 13 finishes, without
waiting on MSFT. Within each ticker, chunks run light-to-heavy (steps 3 -> 5 -> 7).

8 vLLM GPU workers (g5.xlarge / A10G), 5 CPU nodes, 6 embedding workers, sidecar Neo4j.
2 candidates per evolution step (fixed across all 72 configs; not a design factor).

**Expected: the h5 arm carries no path features.** Measured chain yield falls sharply with
depth -- chunk 01 gave path_mean_norm 0.79 and 0.55 at h3, 0.088 at h4, and exactly 0 at h5
(max_hops 0 in both train and validation). Two earlier probes agreed on the same ordering.
This is a property of the graph density and the expansion filters (labelFilter -Article|-Day,
excluded rels PUBLISHED_ON/WRITTEN_BY/MENTIONS/MENTIONED_IN/REFERENCES, candidate_tags and
first_seen predicates), not a defect -- do not 'fix' it. It makes the 24 h5 runs an implicit
subgraph+topology ablation, balanced by construction against the h3 and h4 arms.
Chunk sizing: steps=3 -> 4 configs/chunk (sem 220); steps=5 -> 3/chunk (sem 293); steps=7 -> 2/chunk (sem 440).

**The MSFT half (14-26) runs ticker-filtered; the TSLA half (01-13) did not.** Chunks 14-26 carry
`"filter_ticker": true`, so `--filter-ticker` reaches main.py and the 4-articles/day cap draws only
from MSFT: 712 articles over 196 of the 200 days, 100% MSFT, verified against
data/fnspid_sample_nasdaq_long_text.csv. Unfiltered the same window gives a mixed 794 (TSLA 310 /
MSFT 309 / NVDA 175), which is what chunks 01-13 used and what the TSLA report's caveat records.
The two halves are therefore no longer directly comparable on article composition -- that is
deliberate, the MSFT half is the focused-corpus condition. An unfiltered MSFT attempt on
2026-08-11 (GHA 31482545723, chunks 14-16 only) is superseded by this re-run and must not be
pooled with it.

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Ticker | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|----|
| 01 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb3·s3·default·h3<br>TSLA·lb3·s3·event_dri·h5<br>TSLA·lb3·s3·fundament·h4<br>TSLA·lb8·s3·default·h3 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31259971075) |
| 02 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb8·s3·event_dri·h5<br>TSLA·lb8·s3·fundament·h4<br>TSLA·lb10·s3·default·h3<br>TSLA·lb10·s3·event_dri·h5 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31263190583) |
| 03 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb10·s3·fundament·h4<br>TSLA·lb20·s3·default·h3<br>TSLA·lb20·s3·event_dri·h5<br>TSLA·lb20·s3·fundament·h4 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31266647013) |
| 04 | TSLA | 5 | ✅ DONE (auto) | TSLA·lb3·s5·default·h4<br>TSLA·lb3·s5·event_dri·h3<br>TSLA·lb3·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31270281084) |
| 05 | TSLA | 5 | ✅ DONE (auto) | TSLA·lb8·s5·default·h4<br>TSLA·lb8·s5·event_dri·h3<br>TSLA·lb8·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31274907742) |
| 06 | TSLA | 5 | ✅ DONE (auto) | TSLA·lb10·s5·default·h4<br>TSLA·lb10·s5·event_dri·h3<br>TSLA·lb10·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31279579742) |
| 07 | TSLA | 5 | ✅ DONE (auto) | TSLA·lb20·s5·default·h4<br>TSLA·lb20·s5·event_dri·h3<br>TSLA·lb20·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31283796653) |
| 08 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb3·s7·default·h5<br>TSLA·lb3·s7·event_dri·h4 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31288088287) |
| 09 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb3·s7·fundament·h3<br>TSLA·lb8·s7·default·h5 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31292184354) |
| 10 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb8·s7·event_dri·h4<br>TSLA·lb8·s7·fundament·h3 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31296288867) |
| 11 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb10·s7·default·h5<br>TSLA·lb10·s7·event_dri·h4 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31300437844) |
| 12 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb10·s7·fundament·h3<br>TSLA·lb20·s7·default·h5 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31304754819) |
| 13 | TSLA | 7 | ✅ DONE (auto) | TSLA·lb20·s7·event_dri·h4<br>TSLA·lb20·s7·fundament·h3 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31309446284) |
| 14 | MSFT | 3 | ✅ DONE (auto) | MSFT·lb3·s3·default·h3<br>MSFT·lb3·s3·event_dri·h5<br>MSFT·lb3·s3·fundament·h4<br>MSFT·lb8·s3·default·h3 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31596379653) |
| 15 | MSFT | 3 | ✅ DONE (auto) | MSFT·lb8·s3·event_dri·h5<br>MSFT·lb8·s3·fundament·h4<br>MSFT·lb10·s3·default·h3<br>MSFT·lb10·s3·event_dri·h5 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31604338667) |
| 16 | MSFT | 3 | ✅ DONE (auto) | MSFT·lb10·s3·fundament·h4<br>MSFT·lb20·s3·default·h3<br>MSFT·lb20·s3·event_dri·h5<br>MSFT·lb20·s3·fundament·h4 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31612426094) |
| 17 | MSFT | 5 | ✅ DONE (auto) | MSFT·lb3·s5·default·h4<br>MSFT·lb3·s5·event_dri·h3<br>MSFT·lb3·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31619889444) |
| 18 | MSFT | 5 | ✅ DONE (auto) | MSFT·lb8·s5·default·h4<br>MSFT·lb8·s5·event_dri·h3<br>MSFT·lb8·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31628582776) |
| 19 | MSFT | 5 | ✅ DONE (auto) | MSFT·lb10·s5·default·h4<br>MSFT·lb10·s5·event_dri·h3<br>MSFT·lb10·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31638070112) |
| 20 | MSFT | 5 | ✅ DONE (auto) | MSFT·lb20·s5·default·h4<br>MSFT·lb20·s5·event_dri·h3<br>MSFT·lb20·s5·fundament·h5 | 293 | ~135min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31646600648) |
| 21 | MSFT | 7 | ✅ DONE (auto) | MSFT·lb3·s7·default·h5<br>MSFT·lb3·s7·event_dri·h4 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31653622832) |
| 22 | MSFT | 7 | ✅ DONE (auto) | MSFT·lb3·s7·fundament·h3<br>MSFT·lb8·s7·default·h5 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31659746247) |
| 23 | MSFT | 7 | ✅ DONE (auto) | MSFT·lb8·s7·event_dri·h4<br>MSFT·lb8·s7·fundament·h3 | 440 | ~185min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31665427548) |
| 24 | MSFT | 7 | ⏳ DISPATCHED (auto) | MSFT·lb10·s7·default·h5<br>MSFT·lb10·s7·event_dri·h4 | 440 | ~185min | — |
| 25 | MSFT | 7 | · pending | MSFT·lb10·s7·fundament·h3<br>MSFT·lb20·s7·default·h5 | 440 | ~185min | — |
| 26 | MSFT | 7 | · pending | MSFT·lb20·s7·event_dri·h4<br>MSFT·lb20·s7·fundament·h3 | 440 | ~185min | — |

---

## Out-of-sample runs (Phase 2)

Not part of the balanced design. Each OOS run takes a schema an earlier sweep already
selected, holds it fixed (`--fixed-ontology`), and builds a graph on a window the sweep
never saw. Nothing is fitted to the OOS window except the downstream classifier, so the
number it produces tests the *selection*, not the search.

**One config per dispatch, always.** OOS runs use `neo4j_mode=external` (AuraDB) so the
graph survives the job and can be inspected afterwards. Every job wipes the database on
startup, so two concurrent jobs in external mode would destroy each other's graph;
`sweep.py` now refuses more than one config in external mode.

Dispatch: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/oos_01.json)" -f n_workers=2 -f n_embedding_workers=2 -f gpu_nodes=2 -f cpu_nodes=1 -f neo4j_mode=external -f timeout_minutes=120`

| Run | Ticker | Ontology | Window | Status | GHA |
|----|----|----|----|----|----|
| oos_01 | TSLA | `resources/ontologies/tsla_best_step_1_candidate_1.json` — sweep winner (base + `EarningsReport`/`HAS_EARNINGS`), in-sample AUC 0.787 | 2023-08-16 → 2023-12-16 | ✅ DONE — AUC 0.503 vs baseline 0.719 (−0.216) | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31316716018) |

Feature hyperparameters are carried over from the winning sweep config (lookback 3,
hops 3-3, hybrid, 4 articles/day) — they are part of what was selected, so changing
them would test a different thing.
