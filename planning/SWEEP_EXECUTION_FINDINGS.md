# Sweep Execution Findings (2026-07-25)

Log of what was learned trying to actually run the reduced 36-config sweep
(`sweep_runs/`) on EKS. Three dispatch attempts, three different failures,
one still-open workload-volume problem. Kept here so the next attempt
doesn't re-discover the same things.

## Bugs found and fixed (all merged to `dev`)

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | `sweep.py` crashed before any job ran: `TimeoutError: vllm not ready after 600s` | 10 fresh GPU nodes pulling the image + loading Qwen2.5-7B-AWQ from the shared EFS cache can take >600s | Raised `wait_for_deployment_ready` timeout 600s→1200s (embeddings 300s→600s) | `fbb970b` |
| 2 | AWS temp credentials (`AWS_ACCESS_KEY_ID`/`SECRET`/`SESSION_TOKEN`) printed in plaintext in every Actions log | `$GITHUB_ENV` values aren't auto-masked; the assume-role step never called `::add-mask::` (unlike the MLflow token) | Added `::add-mask::` before export | `fbb970b` |
| 3 | Sweep crashed ~45-63min in: `AccessDenied`/exit 254 on `aws sts assume-role` inside `CredentialRefresher.maybe_refresh()` | `terraform_deployment` IAM role's trust policy only trusted the GitHub OIDC role + local IAM user — never itself. The refresher re-assumes the *same* role every 45min (chained sessions cap at 1h regardless of `max_session_duration`), and that self-chained assumption was denied | Added a 3rd trust-policy statement: self-trust (ARN constructed manually — Terraform disallows a resource referencing its own attribute, "Self-referential block" error) | `4f79127` (applied via `terraform apply` in `infra/configs/backend/dev`, profile `wne-uw` — separate bootstrap root, NOT the `infra/` root that auto-applies via `terraform_deploy.yml`) |
| 4 | Sweep crashed again right after a *successful* credential refresh: `kubernetes.client.exceptions.UnauthorizedException: (401)` on the very next `read_namespaced_job` call | Freshly-issued STS credentials can take a few seconds to propagate to the EKS auth webhook; `wait_for_job` had no retry, so one transient 401 killed the whole process | Added a `time.sleep(5)` after refresh + retry-with-backoff around `read_namespaced_job` specifically for 401 | `b765572` |

**Residual risk, not yet exercised:** the 401-retry only catches `ApiException` with `status==401`. A 403, a 500, or a raw network exception (connection reset, DNS blip) would still crash the sweep uncaught. Only one specific transient-auth failure mode has actually been observed and fixed — there may be others.

## GPU sizing attempts — the actual headroom data

Three dispatches of the same 4-config chunk (`run_01.json`: 2× `steps=7` + 2×
`steps=3`, mixed lookback/prompt/hops), same code, different GPU counts:

| Attempt | GPUs | Semaphore | Outcome | KV utilization observed |
|---|---|---|---|---|
| v3 | 10 | 275 | Crashed (bug #3, self-trust) before any headroom reading was clean | early ramp only: ~19% avg, peak Running=417 (≈3 GPUs' worth) |
| v4 | 6 | 165 | Crashed (bug #4, transient 401) at ~63min, still on step 1 | climbed to 95-99% KV with real queuing (Waiting 150-204) — **6 GPUs undersized** |
| v5 | 8 | 220 | Killed manually at ~115min, still on step 1, no crash | bursty: KV oscillated 20-48%, Waiting spiked 94-165 repeatedly — **better than 6 but still not smooth/saturated** |

**Conclusion on GPU count:** demand is bursty and grows within a single step
as all 3 candidates ramp up together (not primarily across steps — we never
observed step 2 in any attempt, so step-depth effects on demand are still
unmeasured). 8 GPUs handles it better than 6 but still shows real queuing.
Naive throughput-based sizing (calls / (GPU × rate)) has now been wrong twice
in a row — the pipeline's demand shape (bursty, gap-bound between LLM waves
due to Neo4j writes + candidate eval) doesn't match a simple queueing model.

## The dominant problem: workload volume, not infrastructure

This is the real finding. In the v5 attempt (8 GPUs, no crash, ran cleanly
for ~115 minutes):

- Step 1 extraction began ~19:04-19:06 UTC (confirmed directly from job logs).
- At 20:38 UTC (**~90+ minutes later**), all 4 jobs were *still on step 1* —
  no job had logged a step-2 transition or a "winner selected" event.
- Two of this chunk's 4 configs need **7 steps total**. At ~90min/step (or
  worse, since step 1 hadn't even concluded), a `steps=7` job cannot
  possibly finish within the 6h GitHub Actions hard cap — it would need
  ~10.5h minimum just for its own steps, before any queuing overhead.
- This is **compute volume**, not a bug: each step re-runs 3 full candidate
  KG builds over ~3,730 articles each (see calibration below). More GPUs
  reduce *queuing* overhead but do not reduce the *sequential* per-job
  critical path (steps run sequentially within a job by design — step N+1
  needs step N's accepted ontology).

**This means the current per-chunk sizing (4 configs, articles_per_day=10,
candidates=3) cannot fit a `steps=7` job inside a 6h GHA run, regardless of
GPU count.** Confirmed by direct observation, not just modeling.

## Calibration numbers (still believed accurate)

- ~1.7 completed extractions/s per A10G (Qwen2.5-7B-AWQ), measured on a
  single-GPU calibration job (2026-07-25 AM).
- ~140 concurrent-sequence KV cache ceiling per GPU.
- ~3,730 articles per KG build (10/day cap over the 466-day training window,
  averaged across MSFT/TSLA).
- Each job = `candidates(3) × steps` full builds, ~3,730 articles each.

## What this points to: reduce the data, not just the infra

Given steps=7 alone didn't finish in ~115min (let alone 3-9 steps × 3
candidates), the fix is very likely **cutting per-build article volume**
(`articles_per_day` 10→3 or 5) and/or **candidates** 3→2, not further GPU
tuning. This was flagged as the dominant lever back when the sweep was
first calibrated (see `EXPERIMENT_STRATEGY.md` "Reduced Design" section) —
this execution attempt is direct empirical confirmation that infra tuning
alone (GPU count, semaphore, embedding workers) cannot make the current
`articles_per_day=10, candidates=3` configuration fit inside a 6h GHA run
for the higher-`steps` configs.

**Not yet decided:** how much to cut, and whether to cut per-chunk (some
runs at full fidelity, heavy ones reduced) or uniformly across all 36
configs (cleaner, keeps balance, but changes the design further beyond the
steps=9 removal already made).

## Update 2026-07-26: data-volume cut validated, but partial failures are invisible

After the findings above, the window was cut to **100 calendar days
(2022-05-02 → 2022-08-10)** and `articles_per_day` cut **10→4**
(measured ~355 articles/build, a ~10.5x cut from ~3,730). Redispatched as
**14 chunks** (2-3 configs each, up from 9) with semaphore raised to
**400** (safe — per-job demand is far lower now). Same 8 GPUs / 6
embedding workers / 5 CPU nodes.

**Chunk 1 result: validates the fix.** Dispatched 21:38 UTC, completed
23:00 UTC — **~82 minutes total**, vs. the previous attempt where a
single job hadn't even finished step 1 after 115 minutes. Step 1 itself
completed in ~17-19 minutes (previously: never finished in 90+ min).

**But 2 of 3 configs succeeded, 1 failed — with zero diagnostic info.**
The failed config was `TSLA / steps=7` (the heaviest in the chunk).
Root cause unrecoverable: by the time this was investigated (~10h after
completion, due to real elapsed session time), the failed job's pod logs
and cluster events had both expired (job `ttl_seconds_after_finished` is
3600s = 1h). MLflow's parent run confirmed `succeeded=2, failed=1` but no
child run was ever created for the failed job (it likely crashed before
`main.py` reached `mlflow.start_run`).

**Two process gaps this surfaced, independent of the specific failure:**

1. **`sweep.py` doesn't propagate partial failures to its exit code.**
   `main()` prints `"Sweep complete. N/M succeeded"` and exits 0
   regardless — so GitHub Actions shows green `success` on a run that
   silently dropped a config. Nothing in the UI signals it; only manually
   grepping the "Run sweep" step log reveals `"kg-builder-X failed"`.
2. **No failure diagnostics are captured before cleanup.** Nothing fetches
   the failing pod's container logs when `job.status.failed` is detected.
   Combined with the 1h job TTL, any failure not caught within that
   window (e.g. because nobody was watching in real time) is permanently
   unrecoverable, exactly what happened here.

**Not yet fixed — proposed for before dispatching the remaining chunks:**
- `wait_for_job` (or the caller) should fetch and print/log the failing
  pod's container logs (both `kg-builder` and `neo4j`) the moment
  `job.status.failed` is true, before it's ever deleted.
- `main()` should exit non-zero if `failed_count > 0`, so GHA accurately
  reflects partial failures instead of reporting green.
- The failed `TSLA/steps=7` config (from `run_01.json`, this window) needs
  to be re-run once diagnostics are in place, ideally alone so any repeat
  failure's logs are easy to find in the "Run sweep" step output directly.

## Cluster state at time of writing

Killed manually after the v5 finding. GPU nodegroup `desiredSize=0` (nodes
draining, no new billing). All 4 v5 job objects deleted. `sweep_runs/`
config files still reflect the full-fidelity (`articles_per_day=10,
candidates=3`) design and have NOT been changed — no dispatch has
succeeded yet; run 1 remains "not completed" in `sweep_runs/MANIFEST.md`.

## Update 2026-08-03/04: 100-day pilot succeeded (2 chunks), then window raised to 200 days

The 100-day/4-articles-per-day pilot (steps-grouped, 13 chunks) ran cleanly:
DagsHub's 100-run quota cap (private-repo free-tier limit) was hit and blocked
all dispatch — fixed by making the DagsHub repo public (free plan: unlimited
runs on public repos). `sweep.py` was hardened to dump failed-job pod logs
before the 1h TTL wipes them and to exit non-zero on partial failure (both
gaps found the hard way on an earlier run whose cause is permanently
unrecoverable). With both fixes live, **chunks 1 and 2 (`steps=3`) each
completed 4/4 cleanly in ~70-71 min**, matching the ~74min projection — first
fully clean chunks all session.

**Window then raised 100→200 days** (2022-05-02 → 2022-11-18) to increase
fidelity, still at `articles_per_day=4` (~720 articles/build, 2.03x). This
required confirming GitHub-hosted runners have a **hard, non-configurable 6h
job cap** (verified against GitHub's own docs — `timeout-minutes` cannot
exceed it; only self-hosted runners, up to 5 days, escape it) — chosen over
setting up a self-hosted runner since 200 days' projected worst case
(`steps=7`, ~4.6h) still fits with ~1.4h margin. This **supersedes** the
100-day pilot's 2 completed chunks (different window, not comparable); all
13 chunks were regenerated and restart from scratch.

**Chunk 1 (200-day) crashed at 91min**, well into step 2, with
`neo4j.exceptions.ServiceUnavailable: Connection refused` on `localhost:7687`
— multiple steps/candidates had already succeeded (confirmed via MLflow run
links in the captured logs), so this was **Neo4j dying mid-run**, not a
startup issue. Root cause: the Neo4j sidecar's memory limit (768Mi, 512m JVM
heap) was tuned for the 100-day pilot's smaller graph; at 200 days (~2x the
data) the accumulated graph across evolution steps most likely exceeded it,
triggering an OOM-kill. Critically, the job pod's `restartPolicy=Never`
means a crashed sidecar **never recovers** — every later Neo4j query in that
job fails permanently, which is exactly the "several steps succeed, then
total failure" pattern observed. **Fixed** (commit `5b41ea6`): heap max
512m→1g, pagecache 64m→128m, container memory limit 768Mi→1.5Gi, request
512Mi→1Gi. Chunk 1 re-dispatched at 08:08 UTC 2026-08-04 to validate.

**Also validated the failed-job log capture fix from the 100-day pilot**: this
was the first *real* failure since that hardening landed, and it worked
exactly as intended — the full pod traceback (including the underlying
`neo4j.exceptions.ServiceUnavailable`) was captured and readable in the GHA
log well after the 1h job TTL would otherwise have erased it.
