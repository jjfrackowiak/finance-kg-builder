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

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Ticker | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|----|
| 01 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb3·s3·default·h3<br>TSLA·lb3·s3·event_dri·h5<br>TSLA·lb3·s3·fundament·h4<br>TSLA·lb8·s3·default·h3 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31259971075) |
| 02 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb8·s3·event_dri·h5<br>TSLA·lb8·s3·fundament·h4<br>TSLA·lb10·s3·default·h3<br>TSLA·lb10·s3·event_dri·h5 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31263190583) |
| 03 | TSLA | 3 | ✅ DONE (auto) | TSLA·lb10·s3·fundament·h4<br>TSLA·lb20·s3·default·h3<br>TSLA·lb20·s3·event_dri·h5<br>TSLA·lb20·s3·fundament·h4 | 220 | ~85min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31266647013) |
| 04 | TSLA | 5 | ⏳ DISPATCHED (auto) | TSLA·lb3·s5·default·h4<br>TSLA·lb3·s5·event_dri·h3<br>TSLA·lb3·s5·fundament·h5 | 293 | ~135min | — |
| 05 | TSLA | 5 | · pending | TSLA·lb8·s5·default·h4<br>TSLA·lb8·s5·event_dri·h3<br>TSLA·lb8·s5·fundament·h5 | 293 | ~135min | — |
| 06 | TSLA | 5 | · pending | TSLA·lb10·s5·default·h4<br>TSLA·lb10·s5·event_dri·h3<br>TSLA·lb10·s5·fundament·h5 | 293 | ~135min | — |
| 07 | TSLA | 5 | · pending | TSLA·lb20·s5·default·h4<br>TSLA·lb20·s5·event_dri·h3<br>TSLA·lb20·s5·fundament·h5 | 293 | ~135min | — |
| 08 | TSLA | 7 | · pending | TSLA·lb3·s7·default·h5<br>TSLA·lb3·s7·event_dri·h4 | 440 | ~185min | — |
| 09 | TSLA | 7 | · pending | TSLA·lb3·s7·fundament·h3<br>TSLA·lb8·s7·default·h5 | 440 | ~185min | — |
| 10 | TSLA | 7 | · pending | TSLA·lb8·s7·event_dri·h4<br>TSLA·lb8·s7·fundament·h3 | 440 | ~185min | — |
| 11 | TSLA | 7 | · pending | TSLA·lb10·s7·default·h5<br>TSLA·lb10·s7·event_dri·h4 | 440 | ~185min | — |
| 12 | TSLA | 7 | · pending | TSLA·lb10·s7·fundament·h3<br>TSLA·lb20·s7·default·h5 | 440 | ~185min | — |
| 13 | TSLA | 7 | · pending | TSLA·lb20·s7·event_dri·h4<br>TSLA·lb20·s7·fundament·h3 | 440 | ~185min | — |
| 14 | MSFT | 3 | · pending | MSFT·lb3·s3·default·h3<br>MSFT·lb3·s3·event_dri·h5<br>MSFT·lb3·s3·fundament·h4<br>MSFT·lb8·s3·default·h3 | 220 | ~85min | — |
| 15 | MSFT | 3 | · pending | MSFT·lb8·s3·event_dri·h5<br>MSFT·lb8·s3·fundament·h4<br>MSFT·lb10·s3·default·h3<br>MSFT·lb10·s3·event_dri·h5 | 220 | ~85min | — |
| 16 | MSFT | 3 | · pending | MSFT·lb10·s3·fundament·h4<br>MSFT·lb20·s3·default·h3<br>MSFT·lb20·s3·event_dri·h5<br>MSFT·lb20·s3·fundament·h4 | 220 | ~85min | — |
| 17 | MSFT | 5 | · pending | MSFT·lb3·s5·default·h4<br>MSFT·lb3·s5·event_dri·h3<br>MSFT·lb3·s5·fundament·h5 | 293 | ~135min | — |
| 18 | MSFT | 5 | · pending | MSFT·lb8·s5·default·h4<br>MSFT·lb8·s5·event_dri·h3<br>MSFT·lb8·s5·fundament·h5 | 293 | ~135min | — |
| 19 | MSFT | 5 | · pending | MSFT·lb10·s5·default·h4<br>MSFT·lb10·s5·event_dri·h3<br>MSFT·lb10·s5·fundament·h5 | 293 | ~135min | — |
| 20 | MSFT | 5 | · pending | MSFT·lb20·s5·default·h4<br>MSFT·lb20·s5·event_dri·h3<br>MSFT·lb20·s5·fundament·h5 | 293 | ~135min | — |
| 21 | MSFT | 7 | · pending | MSFT·lb3·s7·default·h5<br>MSFT·lb3·s7·event_dri·h4 | 440 | ~185min | — |
| 22 | MSFT | 7 | · pending | MSFT·lb3·s7·fundament·h3<br>MSFT·lb8·s7·default·h5 | 440 | ~185min | — |
| 23 | MSFT | 7 | · pending | MSFT·lb8·s7·event_dri·h4<br>MSFT·lb8·s7·fundament·h3 | 440 | ~185min | — |
| 24 | MSFT | 7 | · pending | MSFT·lb10·s7·default·h5<br>MSFT·lb10·s7·event_dri·h4 | 440 | ~185min | — |
| 25 | MSFT | 7 | · pending | MSFT·lb10·s7·fundament·h3<br>MSFT·lb20·s7·default·h5 | 440 | ~185min | — |
| 26 | MSFT | 7 | · pending | MSFT·lb20·s7·event_dri·h4<br>MSFT·lb20·s7·fundament·h3 | 440 | ~185min | — |
