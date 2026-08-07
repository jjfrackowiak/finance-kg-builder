# Sweep Run Manifest — 72-config design (36 MSFT + 36 TSLA, each independently balanced), 200-day window (2026-08-05)

**Extended from 36 to 72 configs.** Each ticker now gets its own complete, independently-balanced 36-config Latin-square design (steps ∈ {3,5,7}, steps=9 dropped) — not a 36-config array with ticker as a 5th factor. This achieves **perfect balance on every pair** within each ticker (verified: 0 missing cells for lookback×steps/hops, steps×prompt/hops, prompt×hops in each ticker's own 36; the one remaining structural feature — `lookback×prompt` has 4 of 16 cells missing, a side effect of the Latin-square-minus-one-row construction — is present in both tickers' 36 identically, and is fully resolved by using **regression (OLS with all 4 factors as covariates), not naive `groupby().mean()`**, to read off `lookback` and `steps` effects specifically. Verified empirically (zero-noise synthetic test): naive group means are biased for lookback/prompt; regression recovers true effects exactly. `steps` (H2, the core claim) and `chain_hops` (H3) have zero missing cells with anything and are unaffected either way.

**Also documents the resolved cross-ticker construct-validity question** (see `EXPERIMENT_STRATEGY.md`): raw AUC must not be compared directly between MSFT and TSLA (different underlying task difficulty), but pooled marginal-mean *differences* remain valid since ticker is now perfectly balanced against every other factor within each ticker's own design. With 72 configs, MSFT-only and TSLA-only analyses are now ALSO independently valid (not just the pooled analysis) — this was the actual motivation for the extension.

**Reuse from the prior 36-config (2-ticker) plan**: all 12 completed configs (chunks 1-3) and the 3 in-flight configs (chunk 4) are part of the 72-config target as-is (same lookback/steps/prompt/hops combos, now needed for both tickers instead of a ticker-split). 15/72 covered once chunk 4 finishes; **57 new configs / 22 new chunks** (run_05..run_26) needed.

8 vLLM GPU workers (g5.xlarge / A10G), 5 CPU nodes, 6 embedding workers, sidecar Neo4j. Chunk sizing unchanged (steps=3: 4 configs/chunk sem=220; steps=5: 3/chunk sem=293; steps=7: 2/chunk sem=440).

**All 26 chunks must complete before analysis** — the balanced marginals only exist over the full 72-config set (36 per ticker).

Dispatch one run: `gh workflow run sweep.yml --ref dev -f configs="$(cat sweep_runs/run_NN.json)" -f n_workers=8 -f n_embedding_workers=6 -f gpu_nodes=8 -f cpu_nodes=5 -f neo4j_mode=sidecar`

| Run | Steps | Status | Configs (ticker·lookback·steps·prompt·hops) | Sem | ~Time | GHA |
|----|----|----|----|----|----|----|
| 01 | 3 | ✅ DONE (4/4, ~104min) | MSFT·lb3·s3·default·h3<br>TSLA·lb8·s3·fundament·h3<br>MSFT·lb10·s3·event_dri·h3<br>MSFT·lb20·s3·macro_con·h3 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30899967068) |
| 02 | 3 | ✅ DONE (4/4, ~103min) | TSLA·lb3·s3·default·h5<br>MSFT·lb8·s3·fundament·h5<br>TSLA·lb10·s3·event_dri·h5<br>MSFT·lb20·s3·macro_con·h5 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30938173506) |
| 03 | 3 | ✅ DONE (4/4, ~102min) | TSLA·lb3·s3·default·h6<br>TSLA·lb8·s3·fundament·h6<br>MSFT·lb10·s3·event_dri·h6<br>TSLA·lb20·s3·macro_con·h6 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30949749397) |
| 04 | 5 | ✅ DONE (3/3, ~140min) | MSFT·lb3·s5·fundament·h3<br>TSLA·lb8·s5·event_dri·h3<br>TSLA·lb10·s5·macro_con·h3 | 293 | ~203min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/30994206421) |
| 05 | 3 | ✅ DONE (4/4, ~104min) | MSFT·lb3·s3·default·h5<br>MSFT·lb3·s3·default·h6<br>MSFT·lb8·s3·fundament·h3<br>MSFT·lb8·s3·fundament·h6 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31004172132) |
| 06 | 3 | ✅ DONE (auto) | MSFT·lb10·s3·event_dri·h5<br>MSFT·lb20·s3·macro_con·h6<br>TSLA·lb3·s3·default·h3<br>TSLA·lb8·s3·fundament·h5 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31039344798) |
| 07 | 3 | ✅ DONE (auto) | TSLA·lb10·s3·event_dri·h3<br>TSLA·lb10·s3·event_dri·h6<br>TSLA·lb20·s3·macro_con·h3<br>TSLA·lb20·s3·macro_con·h5 | 220 | ~130min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31047522102) |
| 08 | 5 | ✅ DONE (auto) | MSFT·lb3·s5·fundament·h5<br>MSFT·lb3·s5·fundament·h6<br>MSFT·lb8·s5·event_dri·h3 | 293 | ~203min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31055108809) |
| 09 | 5 | ✅ DONE (auto) | MSFT·lb8·s5·event_dri·h5<br>MSFT·lb8·s5·event_dri·h6<br>MSFT·lb10·s5·macro_con·h3 | 293 | ~203min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31062557589) |
| 10 | 5 | ✅ DONE (auto) | MSFT·lb10·s5·macro_con·h5<br>MSFT·lb10·s5·macro_con·h6<br>MSFT·lb20·s5·default·h3 | 293 | ~203min | [link](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31069233385) |
| 11 | 5 | ⏳ DISPATCHED (attempt 5, GH outage resolved) | MSFT·lb20·s5·default·h5 ✅ (done, attempt 3)<br>MSFT·lb20·s5·default·h6<br>TSLA·lb3·s5·fundament·h3 | 293 | ~203min | [attempt 1](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31076389173) (vllm timeout) · [attempt 2](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31112684227) (my bug) · [attempt 3](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31113031688) (cancelled, GH outage) · [attempt 4](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31121807408) (no runner, GH outage) · [attempt 5](https://github.com/jjfrackowiak/finance-kg-builder/actions/runs/31164811808) |
| 12 | 5 | · pending | TSLA·lb3·s5·fundament·h5<br>TSLA·lb3·s5·fundament·h6<br>TSLA·lb8·s5·event_dri·h5 | 293 | ~203min | — |
| 13 | 5 | · pending | TSLA·lb8·s5·event_dri·h6<br>TSLA·lb10·s5·macro_con·h5<br>TSLA·lb10·s5·macro_con·h6 | 293 | ~203min | — |
| 14 | 5 | · pending | TSLA·lb20·s5·default·h3<br>TSLA·lb20·s5·default·h5<br>TSLA·lb20·s5·default·h6 | 293 | ~203min | — |
| 15 | 7 | · pending | MSFT·lb3·s7·event_dri·h3<br>MSFT·lb3·s7·event_dri·h5 | 440 | ~276min | — |
| 16 | 7 | · pending | MSFT·lb3·s7·event_dri·h6<br>MSFT·lb8·s7·macro_con·h3 | 440 | ~276min | — |
| 17 | 7 | · pending | MSFT·lb8·s7·macro_con·h5<br>MSFT·lb8·s7·macro_con·h6 | 440 | ~276min | — |
| 18 | 7 | · pending | MSFT·lb10·s7·default·h3<br>MSFT·lb10·s7·default·h5 | 440 | ~276min | — |
| 19 | 7 | · pending | MSFT·lb10·s7·default·h6<br>MSFT·lb20·s7·fundament·h3 | 440 | ~276min | — |
| 20 | 7 | · pending | MSFT·lb20·s7·fundament·h5<br>MSFT·lb20·s7·fundament·h6 | 440 | ~276min | — |
| 21 | 7 | · pending | TSLA·lb3·s7·event_dri·h3<br>TSLA·lb3·s7·event_dri·h5 | 440 | ~276min | — |
| 22 | 7 | · pending | TSLA·lb3·s7·event_dri·h6<br>TSLA·lb8·s7·macro_con·h3 | 440 | ~276min | — |
| 23 | 7 | · pending | TSLA·lb8·s7·macro_con·h5<br>TSLA·lb8·s7·macro_con·h6 | 440 | ~276min | — |
| 24 | 7 | · pending | TSLA·lb10·s7·default·h3<br>TSLA·lb10·s7·default·h5 | 440 | ~276min | — |
| 25 | 7 | · pending | TSLA·lb10·s7·default·h6<br>TSLA·lb20·s7·fundament·h3 | 440 | ~276min | — |
| 26 | 7 | · pending | TSLA·lb20·s7·fundament·h5<br>TSLA·lb20·s7·fundament·h6 | 440 | ~276min | — |
