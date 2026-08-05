# Sweep Run Manifest — 200-day window, grouped by steps (2026-08-03)

**Data window: 200 calendar days (2022-05-02 → 2022-11-18), `articles_per_day`=4** (measured ~720 articles/build, ~2.03x the previous 100-day pilot's ~355). Chosen as a middle ground: no self-hosted runner needed (GitHub-hosted 6h cap confirmed as a hard platform limit, not configurable — see conversation), and the tightest single-job projection (`steps=7`, ~4.6h) still has ~1.4h margin under 6h.

**⚠️ Supersedes the 100-day pilot.** The 100-day/4-per-day pilot's 2 completed chunks (runs 1-2, both `steps=3`, both 4/4 succeeded) used a *different* window and are **not comparable** to this design — they must be re-run under the 200-day window for the balanced marginals to be valid. All 13 chunks below are freshly generated for 200 days and start from scratch.

Still the 36-config reduced design (`steps ∈ {3,5,7}`, `steps=9` dropped). Chunk structure (steps-grouped, sizes 4/3/2 for steps 3/5/7) is unchanged from the 100-day pilot — concurrency shape, not article volume, drives chunk sizing. Per-step time scaled by the measured article-volume ratio (~2.03x):

| steps | configs/chunk | chunks | semaphore | ~projected wall-time |
|---|---|---|---|---|
| 3 | 4 | 3 | 220 | ~130 min (2.2h) |
| 5 | 3 | 4 | 293 | ~203 min (3.4h) |
| 7 | 2 | 6 | 440 | ~276 min (4.6h) — tightest margin, recommend validating before trusting |

8 GPUs / 6 embedding workers / 5 CPU nodes, unchanged.

**All 13 runs must complete before analysis** — the balanced marginals only exist over the full 36-config set.

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|
| 01 | 3 | ✅ DONE (4/4, ~104min) | MSFT·lb3·s3·default·h3<br>TSLA·lb8·s3·fundament·h3<br>MSFT·lb10·s3·event_dri·h3<br>MSFT·lb20·s3·macro_con·h3 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30899967068) |
| 02 | 3 | ✅ DONE (4/4, ~103min) | TSLA·lb3·s3·default·h5<br>MSFT·lb8·s3·fundament·h5<br>TSLA·lb10·s3·event_dri·h5<br>MSFT·lb20·s3·macro_con·h5 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30938173506) |
| 03 | 3 | ✅ DONE (4/4, ~102min) | TSLA·lb3·s3·default·h6<br>TSLA·lb8·s3·fundament·h6<br>MSFT·lb10·s3·event_dri·h6<br>TSLA·lb20·s3·macro_con·h6 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30949749397) |
| 04 | 5 | · pending | MSFT·lb3·s5·fundament·h3<br>TSLA·lb8·s5·event_dri·h3<br>TSLA·lb10·s5·macro_con·h3 | 293 | ~203min | — |
| 05 | 5 | · pending | TSLA·lb20·s5·default·h3<br>MSFT·lb3·s5·fundament·h5<br>TSLA·lb8·s5·event_dri·h5 | 293 | ~203min | — |
| 06 | 5 | · pending | TSLA·lb10·s5·macro_con·h5<br>TSLA·lb20·s5·default·h5<br>MSFT·lb3·s5·fundament·h6 | 293 | ~203min | — |
| 07 | 5 | · pending | MSFT·lb8·s5·event_dri·h6<br>MSFT·lb10·s5·macro_con·h6<br>MSFT·lb20·s5·default·h6 | 293 | ~203min | — |
| 08 | 7 | · pending | TSLA·lb3·s7·event_dri·h3<br>TSLA·lb8·s7·macro_con·h3 | 440 | ~276min | — |
| 09 | 7 | · pending | MSFT·lb10·s7·default·h3<br>TSLA·lb20·s7·fundament·h3 | 440 | ~276min | — |
| 10 | 7 | · pending | MSFT·lb3·s7·event_dri·h5<br>MSFT·lb8·s7·macro_con·h5 | 440 | ~276min | — |
| 11 | 7 | · pending | TSLA·lb10·s7·default·h5<br>MSFT·lb20·s7·fundament·h5 | 440 | ~276min | — |
| 12 | 7 | · pending | TSLA·lb3·s7·event_dri·h6<br>MSFT·lb8·s7·macro_con·h6 | 440 | ~276min | — |
| 13 | 7 | · pending | MSFT·lb10·s7·default·h6<br>TSLA·lb20·s7·fundament·h6 | 440 | ~276min | — |
