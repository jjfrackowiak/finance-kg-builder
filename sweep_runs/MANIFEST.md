# Sweep Run Manifest — reduced design, reduced data window, grouped by steps (2026-07-26)

**Data window: 100 calendar days (2022-05-02 → 2022-08-10), `articles_per_day`=4** (measured ~355 articles/build — see `planning/SWEEP_EXECUTION_FINDINGS.md` for why the full 471-day/10-per-day window was abandoned: a single `steps=7` job's step 1 alone took 90+ min and never finished).

Still the 36-config reduced design (`steps ∈ {3,5,7}`, `steps=9` dropped — see `planning/EXPERIMENT_STRATEGY.md` "Reduced Design"). **Chunks are now grouped by `steps` level** (previously mixed steps=3/5/7 within a chunk, which made wall-time unpredictable — a light job finishes and frees GPU capacity while a heavy one is still running, so the chunk's wall-clock was governed by whichever config had the most steps). Grouping by steps makes each chunk's wall-time roughly uniform and predictable, at a cost: same-steps configs stay synchronized through every step (no early tapering), so **chunk size shrinks as steps grows** to keep sustained peak GPU demand in check:

| steps | configs/chunk | chunks | semaphore | ~real wall-time (measured ~18min/step + ~20min warmup) |
|---|---|---|---|---|
| 3 | 4 | 3 | 220 | ~74 min |
| 5 | 3 | 4 | 293 | ~110 min |
| 7 | 2 | 6 | 440 | ~146 min |

8 GPUs / 6 embedding workers / 5 CPU nodes, unchanged. All comfortably under the 6h GHA cap with real margin, even for steps=7.

**DagsHub quota (100-run cap on private repos) was blocking all dispatch — resolved 2026-08-03 by making the DagsHub repo public** (free plan: unlimited runs on public repos). Verified via a live create-run probe.

**sweep.py hardened (commit `57dcc23`):** failed-job pod logs are now dumped before the 1h job TTL wipes them, and `main()` exits non-zero on any job failure (previously always exited 0, so GHA showed green on a silent partial failure — this is exactly what happened to the `TSLA/steps=7` config in the first 100-day pilot chunk, cause unrecoverable at the time).

**All 13 runs must complete before analysis** — the balanced marginals only exist over the full 36-config set.

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|
| 01 | 3 | ✅ DONE (4/4, ~71min) | MSFT·lb3·s3·default·h3<br>TSLA·lb8·s3·fundament·h3<br>MSFT·lb10·s3·event_dri·h3<br>MSFT·lb20·s3·macro_con·h3 | 220 | ~74min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30801110320) |
| 02 | 3 | · pending | TSLA·lb3·s3·default·h5<br>MSFT·lb8·s3·fundament·h5<br>TSLA·lb10·s3·event_dri·h5<br>MSFT·lb20·s3·macro_con·h5 | 220 | ~74min | — |
| 03 | 3 | · pending | TSLA·lb3·s3·default·h6<br>TSLA·lb8·s3·fundament·h6<br>MSFT·lb10·s3·event_dri·h6<br>TSLA·lb20·s3·macro_con·h6 | 220 | ~74min | — |
| 04 | 5 | · pending | MSFT·lb3·s5·fundament·h3<br>TSLA·lb8·s5·event_dri·h3<br>TSLA·lb10·s5·macro_con·h3 | 293 | ~110min | — |
| 05 | 5 | · pending | TSLA·lb20·s5·default·h3<br>MSFT·lb3·s5·fundament·h5<br>TSLA·lb8·s5·event_dri·h5 | 293 | ~110min | — |
| 06 | 5 | · pending | TSLA·lb10·s5·macro_con·h5<br>TSLA·lb20·s5·default·h5<br>MSFT·lb3·s5·fundament·h6 | 293 | ~110min | — |
| 07 | 5 | · pending | MSFT·lb8·s5·event_dri·h6<br>MSFT·lb10·s5·macro_con·h6<br>MSFT·lb20·s5·default·h6 | 293 | ~110min | — |
| 08 | 7 | · pending | TSLA·lb3·s7·event_dri·h3<br>TSLA·lb8·s7·macro_con·h3 | 440 | ~146min | — |
| 09 | 7 | · pending | MSFT·lb10·s7·default·h3<br>TSLA·lb20·s7·fundament·h3 | 440 | ~146min | — |
| 10 | 7 | · pending | MSFT·lb3·s7·event_dri·h5<br>MSFT·lb8·s7·macro_con·h5 | 440 | ~146min | — |
| 11 | 7 | · pending | TSLA·lb10·s7·default·h5<br>MSFT·lb20·s7·fundament·h5 | 440 | ~146min | — |
| 12 | 7 | · pending | TSLA·lb3·s7·event_dri·h6<br>MSFT·lb8·s7·macro_con·h6 | 440 | ~146min | — |
| 13 | 7 | · pending | MSFT·lb10·s7·default·h6<br>TSLA·lb20·s7·fundament·h6 | 440 | ~146min | — |
