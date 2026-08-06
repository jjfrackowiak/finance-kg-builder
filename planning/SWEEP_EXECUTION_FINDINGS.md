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

**Chunk 1 retry (v2, neo4j mem fix) crashed again at 65m53s** — same
`subprocess.CalledProcessError` / exit-254 symptom as the original self-trust
bug, but the self-trust IAM policy was verified still in place (re-checked
live). Different root cause: `CredentialRefresher.last_refresh` was seeded
with `time.time()` at **Python object instantiation**, which happens only
after GPU node provisioning (up to 20min with the 1200s timeout) and job
submission — 20-25min after the *actual* AWS session was created in the
workflow's "Assume deployment role" bash step. So the 45-minute countdown
started ~20-25min late relative to true session age, meaning the first
refresh could fire *after* AWS's 1h chained-session hard cap had already
passed (65m53s observed > 60min cap). **Fixed** (commit `24498be`): the
workflow now exports `ROLE_ASSUMED_AT_EPOCH` right after the assume-role
call; `CredentialRefresher` anchors its countdown to that real timestamp
instead of its own construction time. Chunk 1 re-dispatched (3rd attempt,
run `30899967068`) with both this and the neo4j memory fix live.

**Open question, not yet actioned:** candidate pruning (`_prune_candidate_tags`,
`orchestrator.py:711`) only strips a losing candidate's tag from nodes/
relationships — it never deletes them (confirmed via a full-package search:
the only `DETACH DELETE` calls are the start-of-run full wipe). The code
comment at `orchestrator.py:459` confirms this is intentional design, not an
oversight. This means graph size grows monotonically for a job's entire
lifetime (every candidate at every step, win or lose, adds permanent data),
which is almost certainly the underlying driver behind the Neo4j OOM above —
the memory bump treats the symptom, not the cause. Real fix would be
deleting nodes/relationships whose `candidate_tags` becomes empty after
untagging (i.e. truly orphaned, not shared with `base_structure` or an
accepted lineage) — a genuine pipeline change, not yet implemented. Risk is
highest for the 6 remaining `steps=7` chunks (most accumulated candidate
history before a job ends).

## Update 2026-08-05: extended to 72 configs, and a Counter-based balance check flaw found

After 12/36 configs completed cleanly (chunks 1-3, all `steps=3`, ~102-104min
each, both the neo4j memory fix and credential-refresh timing fix holding)
plus chunk 4 (first `steps=5`) in flight, discussion turned to whether MSFT
and TSLA results could be meaningfully compared. Conclusion: raw AUC can't be
compared across tickers directly (different underlying task difficulty), but
pooled marginal-mean *differences* remain valid (ticker-difficulty cancels
when subtracted, same logic as any balanced-design confounder control) —
*however* this doesn't support independently analyzing MSFT-only or
TSLA-only, since filtering the pooled 36-design by ticker breaks balance on
6 of 6 remaining pairs (verified empirically, not assumed).

**Extended to 72 configs** (36 MSFT + 36 TSLA, each its own independently-
balanced `steps=9`-dropped design) to support per-ticker analysis without
the earlier-declined 288-config full rebuild — the reduced 36-per-ticker
version turns out to need no more configs than doing it once for a full
4-factor design would (144/2=72... actually simpler: same reduction ratio
applies per-ticker as it did pooled). Reuses all 12 completed + 3 in-flight
configs (same hparam combos, now needed under both tickers instead of a
ticker split); 22 new chunks (`run_05..run_26`) generated for the remaining
57 configs — see `sweep_runs/MANIFEST.md`.

**Real methodology bug caught while verifying the single-ticker design's
balance**: checking pairwise balance via `Counter` of *observed* factor
combinations — the exact method used everywhere earlier in this document
and in `EXPERIMENT_STRATEGY.md` — silently hides **missing** cells, since a
0-count combination never appears as a Counter key. Explicit full-grid
verification (checking every possible cell, not just observed ones) found
`lookback×prompt` has 4 of 16 cells genuinely absent in the `steps=9`-dropped
construction (each `lookback` level never co-occurs with one specific
`prompt` value — an artifact of the `P=(i+j) mod 4` formula losing one row).
Only `lookback` and `prompt` are affected, only with each other; `steps`
(the core H2 claim) and `chain_hops` have zero missing cells with anything.

**The fix needed no new data** — confirmed via an empirical zero-noise
synthetic-data test that the design matrix is full rank (the effects ARE
identifiable) and that naive `groupby().mean()` is measurably biased for
`lookback`/`prompt` while **OLS regression with all 4 factors as covariates
recovers the true effects exactly**. One further self-correction along the
way: an initial "regression also fails" result was itself a bug (pandas'
`get_dummies` sorted `lookback`'s string-cast values alphabetically,
silently using `"10"` as the reference category instead of `"3"`) — fixed by
specifying explicit `pd.Categorical` reference categories, after which
regression matched ground truth to full floating-point precision.

**Analysis implication going forward**: use OLS regression, not naive
`groupby(value).mean()`, when reading off `lookback` and `evolution_prompt`
effects specifically (from either the pooled 72 or either ticker's 36
alone). `steps` and `chain_hops` are safe with either method.

**Minor, non-blocking finding (chunk 4, still in flight): occasional
extraction JSON truncation.** ~56 `pydantic_core.ValidationError: Invalid
JSON: EOF while parsing...` occurrences seen by step 5 (the deepest step
reached this session) — `incremental_kg_mutator.py:245` fails to parse the
LLM's JSON response because it's truncated mid-string, most likely a
`max_tokens` ceiling being hit on longer/more-complex extractions as the
evolved ontology grows. The immediate handler (`:252`) logs and re-raises,
but the job pod stays `Running` with 0 restarts through 106+ minutes and
multiple steps, so something upstream catches it per-article and skips
rather than crashing the job. **Net effect: not fatal, but silently drops
that one article's contribution** — worth investigating (likely just
raising the extraction call's `max_tokens`) before or alongside the
`steps=7` chunks, where deeper ontologies make longer completions more
likely. Not yet actioned.

## Update 2026-08-06: chunk 11 GPU billing leak + root cause (slow node provisioning)

Chunk 11 (`run_id=31076389173`, dispatched 06:10:09Z) failed with
`TimeoutError: vllm not ready after 1800s`. Investigating cleanup afterward
found the GPU nodegroup still at `desiredSize=8` — 8 `g5.xlarge` nodes had
been running since the chunk's own scale-up, undetected, for **~7-8 hours**
by the time it was caught (node creation timestamps 06:15-07:51Z, caught at
14:42Z).

**Two compounding bugs, both fixed in `2e2d21b`:**

1. **Cleanup silently skipped the scale-down call.** The "🧹 Scale down"
   step's `wait_active` helper had a 300s timeout waiting for the nodegroup
   to leave `UPDATING`. On timeout it `return`ed non-zero, and the caller
   chained the actual `aws eks update-nodegroup-config --desiredSize=0` call
   with `&&`, so a slow-to-settle nodegroup meant the scale-down command
   *never ran at all* — no error, no warning, nothing in the logs to
   indicate the leak. Fixed: `wait_active` timeout raised to 600s, and the
   scale-down call is now always attempted regardless of `wait_active`'s
   outcome (`||` fallback logs a warning instead of skipping); a
   post-scale-down check re-reads `desiredSize` and emits a loud
   `::error::` (non-blocking, so it doesn't break the later auto-chain
   `success()` check) if it's not actually `0`.

2. **Root cause of the original vllm timeout: scale-up only waited for 1/8
   GPU nodes.** The "⚡ Scale up" step's readiness wait
   (`until kubectl get nodes -l node-role=gpu ... grep -qE Ready`) returned
   as soon as a *single* GPU node was `Ready`, then immediately proceeded to
   `sweep.py`, which starts polling vllm's own 1800s readiness timeout.
   That day, GPU node provisioning was unusually slow — 3 of 8 nodes Ready
   by 06:16, but the last 5 not until 07:20-07:51, over an hour after
   dispatch. vllm's 8-replica deployment could never reach `Ready` within
   1800s while most of its nodes were still `Pending`. Fixed: scale-up now
   waits for **all** requested GPU nodes to be `Ready` (40m timeout, logs
   progress every 60s, proceeds with a `::warning::` only if at least 1 node
   is ready after the deadline) before moving on to `sweep.py`.

**Manual remediation performed immediately upon discovery** (before the
code fix, to stop the leak): `aws eks update-nodegroup-config
--nodegroup-name kg-experiments-dev-gpu --scaling-config desiredSize=0` +
`kubectl -n kg-experiments scale deployment vllm embeddings --replicas=0`.
Confirmed drained to 0 nodes before redispatching chunk 11.

**Cost impact**: ~8 `g5.xlarge` (A10G) nodes ran ~7-8h instead of the
budgeted ~30-45min for one chunk attempt — roughly an order of magnitude
more GPU-hours than expected for this single chunk. Not independently
verified against AWS Cost Explorer; flagged here for awareness rather than
quantified precisely.

**Process takeaway**: this is the second time this session a cleanup-path
edge case caused a real, undetected AWS cost (the first being the
credential-masking gap, bug #2 above, which was a security leak rather
than a billing one). Cleanup/teardown code needs the same scrutiny as the
main happy path, not less — it runs unconditionally (`if: always()`) but
was written and tested less rigorously than the scale-up path.

**Immediate follow-up regression (same day): the fix in (2) above broke the
scale-up step entirely.** First retry (`run 31112684227`) failed within
~30s, before any GPU node could possibly be ready. Cause: the new
`READY_NOW="$(kubectl get nodes ... | grep -c ' Ready ')"` line is a bare
variable assignment inside the loop *body* (not the `until` condition,
which is `-e`-exempt). `grep -c` exits 1 when the count is 0, and under
this step's `bash -e` shell a failing `var=$(cmd)` assignment propagates
that exit and kills the whole step immediately. Fixed in `e79df82` by
wrapping the count read in a `count_ready_gpu() { ... | grep -c ' Ready '
|| true; }` helper. Verified locally under `bash -e` with a mocked
zero-match `kubectl` before pushing. **Lesson**: `grep -c` inside any bare
assignment under `set -e` is a landmine — the loop *condition* itself is
safe (POSIX exempts `while`/`until` conditions from `-e`), but any count
read inside the loop *body* is not.

**Third attempt (`run 31113031688`): unexplained GitHub-side cancellation,
one config completed first.** After both bugs above were fixed, GPU
provisioning worked correctly (8/8 nodes Ready in ~11min) and the sweep
started normally. 76 minutes into the `🧪 Run sweep` step, GitHub Actions
cancelled the job outright (`##[error]The operation was canceled.`, no
further detail). Investigated and ruled out: no `concurrency:` block in
the workflow: no overlapping dispatch existed; repo is public so Actions
minutes/spending limits don't apply; the user confirmed they did not
cancel it manually. No definitive cause was found — the leading
hypothesis is a transient GitHub-hosted-runner-side interruption (rare
but known to happen on hosted `ubuntu-latest` runners for single steps
that block for 1h+ polling an external system), not anything in our
config or code. Flagging as unresolved rather than guessing further.

The silver lining: the underlying k8s job for the first of the two
configs (`MSFT lb20 h5 default`) had already completed successfully
(exit 0, full MLflow run logged) before the cancellation landed, and
cleanup correctly scaled GPU nodes back to 0 despite the cancel. Rather
than re-run all 3 configs, attempt 4 (`run 31121807408`) was dispatched
with only the 2 still-outstanding configs (`MSFT lb20 h6 default`,
`TSLA lb3 h3 fundamental`) built from `run_11.json[1:]` — `run_11.json`
itself was left unmodified as the canonical chunk-11 definition.
`run_number` was deliberately omitted from this dispatch (no auto-chain)
so a further failure here doesn't cascade into a premature chunk 12
dispatch; auto-chain will be re-armed on the next full chunk once 11
actually completes.
