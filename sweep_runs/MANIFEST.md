# Sweep Run Manifest — 72-config design (36 MSFT + 36 TSLA, each independently balanced), 200-day window (2026-08-07)

**FULL RESTART (2026-08-07).** All prior runs (chunks 1-14 of the previous ordering, ~44 completed
configs) are invalidated and superseded. Root cause: the Neo4j sidecar was missing the APOC plugin
entirely (`apoc.path.expandConfig` / `apoc.refactor.mergeNodes` both raised
`Neo.ClientError.Procedure.ProcedureNotFound` on every call). Two consequences, both silent because
the calling code catches the exception and logs a warning instead of failing the job:

1. **Path-embedding features were always the zero-fallback vector.** Every completed run's
   `feature_importances` showed exactly zero importance across the entire path-embedding block —
   confirmed across every run checked. Only the subgraph-histogram and topology blocks (which don't
   use APOC) carried real signal.
2. **Entity deduplication never ran.** `apoc.refactor.mergeNodes` failed the same way, so duplicate
   entities (e.g. the same company mentioned with slightly different name casing across articles)
   were never merged — graphs are more fragmented than intended, diluting even the
   subgraph/topology signal that did work.

**Fix**: `NEO4J_PLUGINS=["apoc"]` (+ unrestrict/allowlist settings) added to the sidecar container
spec in `kg_builder_llm/scripts/sweep.py`. Verified empirically against a live, isolated test pod
running the exact fixed config before restarting: `apoc.path.expandConfig` found a real 2-hop path,
`apoc.refactor.mergeNodes` successfully merged a duplicate pair down to one node. Two smaller,
related bugs also fixed: a missing `first_seen IS NULL OR` null-guard in the chain-extraction query
(`relationship_chains.py`, harmless in practice since Article/Day nodes are excluded from paths
anyway, but inconsistent with every other query in the codebase), and `get_embedding_dim("remote",
...)` hardcoding 1536 instead of the real 384 (now probes the live service once and caches the
result).

**Reordered execution (2026-08-07): TSLA-first, complete by the halfway point.** Previously chunks
mixed/alternated tickers throughout the run, so a partial run left both tickers incomplete. Now
chunks 01-13 are 100% TSLA (all 36 TSLA configs) and chunks 14-26 are 100% MSFT (all 36 MSFT
configs) — same 72-config balanced design, same per-ticker Latin-square construction, just
reordered so a full TSLA-only analysis is possible as soon as chunk 13 completes, without waiting
on MSFT. Within each ticker block, chunks still progress light-to-heavy (`steps=3` → `5` → `7`) to
fail fast and cheap if anything is still wrong.

8 vLLM GPU workers (g5.xlarge / A10G), 5 CPU nodes, 6 embedding workers, sidecar Neo4j. Chunk sizing
unchanged (steps=3: 4 configs/chunk sem=220; steps=5: 3/chunk sem=293; steps=7: 2/chunk sem=440).

**All 26 chunks must complete before pooled analysis** — the balanced marginals only exist over the
full 72-config set (36 per ticker). A TSLA-only analysis is valid as soon as chunks 01-13 complete
(36/36 TSLA, independently balanced); an MSFT-only analysis needs chunks 14-26.

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Ticker | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|----|
| 01 | TSLA | 3 | · pending | TSLA·lb3·s3·default·h3<br>TSLA·lb3·s3·default·h5<br>TSLA·lb3·s3·default·h6<br>TSLA·lb8·s3·fundament·h3 | 220 | ~130min | — |
| 02 | TSLA | 3 | · pending | TSLA·lb8·s3·fundament·h5<br>TSLA·lb8·s3·fundament·h6<br>TSLA·lb10·s3·event_dri·h3<br>TSLA·lb10·s3·event_dri·h5 | 220 | ~130min | — |
| 03 | TSLA | 3 | · pending | TSLA·lb10·s3·event_dri·h6<br>TSLA·lb20·s3·macro_con·h3<br>TSLA·lb20·s3·macro_con·h5<br>TSLA·lb20·s3·macro_con·h6 | 220 | ~130min | — |
| 04 | TSLA | 5 | · pending | TSLA·lb3·s5·fundament·h3<br>TSLA·lb3·s5·fundament·h5<br>TSLA·lb3·s5·fundament·h6 | 293 | ~203min | — |
| 05 | TSLA | 5 | · pending | TSLA·lb8·s5·event_dri·h3<br>TSLA·lb8·s5·event_dri·h5<br>TSLA·lb8·s5·event_dri·h6 | 293 | ~203min | — |
| 06 | TSLA | 5 | · pending | TSLA·lb10·s5·macro_con·h3<br>TSLA·lb10·s5·macro_con·h5<br>TSLA·lb10·s5·macro_con·h6 | 293 | ~203min | — |
| 07 | TSLA | 5 | · pending | TSLA·lb20·s5·default·h3<br>TSLA·lb20·s5·default·h5<br>TSLA·lb20·s5·default·h6 | 293 | ~203min | — |
| 08 | TSLA | 7 | · pending | TSLA·lb3·s7·event_dri·h3<br>TSLA·lb3·s7·event_dri·h5 | 440 | ~276min | — |
| 09 | TSLA | 7 | · pending | TSLA·lb3·s7·event_dri·h6<br>TSLA·lb8·s7·macro_con·h3 | 440 | ~276min | — |
| 10 | TSLA | 7 | · pending | TSLA·lb8·s7·macro_con·h5<br>TSLA·lb8·s7·macro_con·h6 | 440 | ~276min | — |
| 11 | TSLA | 7 | · pending | TSLA·lb10·s7·default·h3<br>TSLA·lb10·s7·default·h5 | 440 | ~276min | — |
| 12 | TSLA | 7 | · pending | TSLA·lb10·s7·default·h6<br>TSLA·lb20·s7·fundament·h3 | 440 | ~276min | — |
| 13 | TSLA | 7 | · pending | TSLA·lb20·s7·fundament·h5<br>TSLA·lb20·s7·fundament·h6 | 440 | ~276min | — |
| 14 | MSFT | 3 | · pending | MSFT·lb3·s3·default·h3<br>MSFT·lb3·s3·default·h5<br>MSFT·lb3·s3·default·h6<br>MSFT·lb8·s3·fundament·h3 | 220 | ~130min | — |
| 15 | MSFT | 3 | · pending | MSFT·lb8·s3·fundament·h5<br>MSFT·lb8·s3·fundament·h6<br>MSFT·lb10·s3·event_dri·h3<br>MSFT·lb10·s3·event_dri·h5 | 220 | ~130min | — |
| 16 | MSFT | 3 | · pending | MSFT·lb10·s3·event_dri·h6<br>MSFT·lb20·s3·macro_con·h3<br>MSFT·lb20·s3·macro_con·h5<br>MSFT·lb20·s3·macro_con·h6 | 220 | ~130min | — |
| 17 | MSFT | 5 | · pending | MSFT·lb3·s5·fundament·h3<br>MSFT·lb3·s5·fundament·h5<br>MSFT·lb3·s5·fundament·h6 | 293 | ~203min | — |
| 18 | MSFT | 5 | · pending | MSFT·lb8·s5·event_dri·h3<br>MSFT·lb8·s5·event_dri·h5<br>MSFT·lb8·s5·event_dri·h6 | 293 | ~203min | — |
| 19 | MSFT | 5 | · pending | MSFT·lb10·s5·macro_con·h3<br>MSFT·lb10·s5·macro_con·h5<br>MSFT·lb10·s5·macro_con·h6 | 293 | ~203min | — |
| 20 | MSFT | 5 | · pending | MSFT·lb20·s5·default·h3<br>MSFT·lb20·s5·default·h5<br>MSFT·lb20·s5·default·h6 | 293 | ~203min | — |
| 21 | MSFT | 7 | · pending | MSFT·lb3·s7·event_dri·h3<br>MSFT·lb3·s7·event_dri·h5 | 440 | ~276min | — |
| 22 | MSFT | 7 | · pending | MSFT·lb3·s7·event_dri·h6<br>MSFT·lb8·s7·macro_con·h3 | 440 | ~276min | — |
| 23 | MSFT | 7 | · pending | MSFT·lb8·s7·macro_con·h5<br>MSFT·lb8·s7·macro_con·h6 | 440 | ~276min | — |
| 24 | MSFT | 7 | · pending | MSFT·lb10·s7·default·h3<br>MSFT·lb10·s7·default·h5 | 440 | ~276min | — |
| 25 | MSFT | 7 | · pending | MSFT·lb10·s7·default·h6<br>MSFT·lb20·s7·fundament·h3 | 440 | ~276min | — |
| 26 | MSFT | 7 | · pending | MSFT·lb20·s7·fundament·h5<br>MSFT·lb20·s7·fundament·h6 | 440 | ~276min | — |
