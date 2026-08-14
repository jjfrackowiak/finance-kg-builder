"""Pull and analyse both halves of the balanced 72-config design.

This supersedes `analysis/tsla36_2026-08-09/`, which covered the TSLA half alone.
Same selection logic and same "final accepted AUC" definition; the only additions
are a `ticker` column and the MSFT-specific `filter_ticker` condition.

Run selection — each condition is necessary, and together they yield exactly 36 per
ticker:

  experiment_id "0"          — kg-ontology-evolution
  no mlflow.parentRunId      — top-level run, one per config
  status FINISHED
  param time_window_days=200 — the current window design (2022-05-02 → 2022-11-18)
  param candidates=2         — the 36-config design; earlier dispatches used 3
  start_time >= 2026-08-08T12:00Z — after the path-embedding fix (36f4f88)

  TSLA: param ticker=TSLA, filter_ticker absent or "False"
  MSFT: param ticker=MSFT, filter_ticker == "True"

The `filter_ticker` condition on MSFT is load-bearing. Chunks 14-16 were dispatched
on 2026-08-11 WITHOUT the filter (mixed articles) and 12 such runs are in MLflow;
they are superseded by the 2026-08-12/13 re-run and must not be pooled with it. See
sweep_runs/MANIFEST.md. TSLA carries no filter_ticker param at all, because the flag
did not exist when chunks 01-13 ran — that is the asymmetry this whole report is
about, and it is stated in section A rather than hidden here.

"Final AUC" is the AUC of the **final accepted ontology**: walk the step winners in
order and keep the last one whose step was accepted; if no step was accepted, the run
ends at its article-text baseline. This is deliberately *not* the `best_auc` metric
the pipeline logs, which is the max over every candidate ever evaluated including
rejected ones and the baseline — that overstates what a run actually ended up with.

Note on what the baseline is (this matters for reading `delta`): the two arms share
no features. `build_day_feature_vector` skips text entirely (feature_engineering.py,
"using structural signals only"), so the candidate arm is path+subgraph+topology and
the baseline arm is mean-pooled article-text embeddings. `delta` is therefore a
contrast between two disjoint representations, not an ablation of the ontology.

Usage:  set MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_TOKEN, then `python pull_and_analyze.py`
"""

import csv
import datetime as dt
import json
import math
import os
import statistics as st
import urllib.parse
from collections import defaultdict
from pathlib import Path

import requests

BASE = "https://dagshub.com/jjfrackowiak/finance-kg-builder.mlflow"
API = BASE + "/api/2.0/mlflow"
HERE = Path(__file__).parent
AUTH = (
    os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
    os.environ.get("MLFLOW_TRACKING_TOKEN") or os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
)
CUTOFF_MS = dt.datetime(2026, 8, 8, 12, 0, tzinfo=dt.timezone.utc).timestamp() * 1000
TICKERS = ("TSLA", "MSFT")

tags = lambda r: {t["key"]: t["value"] for t in r["data"].get("tags", [])}  # noqa: E731
params = lambda r: {p["key"]: p["value"] for p in r["data"].get("params", [])}  # noqa: E731
metrics = lambda r: {m["key"]: m["value"] for m in r["data"].get("metrics", [])}  # noqa: E731


def fetch_all_runs() -> list:
    """Page through every run in experiment 0."""
    runs, token = [], None
    while True:
        body = {"experiment_ids": ["0"], "max_results": 1000, "run_view_type": "ACTIVE_ONLY"}
        if token:
            body["page_token"] = token
        resp = requests.post(f"{API}/runs/search", json=body, auth=AUTH, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        runs += payload.get("runs", [])
        token = payload.get("next_page_token")
        if not token:
            return runs


def select_configs(runs: list, ticker: str) -> list:
    """Top-level runs for one ticker's 36-config half."""
    out = []
    for r in runs:
        p = params(r)
        if (
            "mlflow.parentRunId" in tags(r)
            or r["info"]["status"] != "FINISHED"
            or p.get("time_window_days") != "200"
            or p.get("ticker") != ticker
            or p.get("candidates") != "2"
            or r["info"]["start_time"] < CUTOFF_MS
        ):
            continue
        filtered = p.get("filter_ticker") == "True"
        # MSFT keeps only the filtered re-run; TSLA only ever ran unfiltered.
        if (ticker == "MSFT") != filtered:
            continue
        out.append(r)
    return out


def final_accepted_auc(children: list, baseline: float) -> tuple:
    """AUC of the last accepted step winner, falling back to the baseline."""
    auc, tag = baseline, "(no step accepted)"
    for c in sorted(children, key=lambda c: tags(c).get("mlflow.runName", "")):
        t = tags(c)
        if t.get("winner") == "true" and t.get("step_accepted") == "true":
            auc, tag = metrics(c)["auc"], t["mlflow.runName"]
    return auc, tag


def final_accepted_train_auc(children: list) -> float:
    """train_auc of the same winner `final_accepted_auc` returns, or None.

    Logged from 2026-08-11 onward, so present for every MSFT run and absent for
    every TSLA run. The overfitting section reports it for MSFT only and says so.
    """
    train = None
    for c in sorted(children, key=lambda c: tags(c).get("mlflow.runName", "")):
        t = tags(c)
        if t.get("winner") == "true" and t.get("step_accepted") == "true":
            train = metrics(c).get("train_auc")
    return train


def fetch_addition_history(run_id: str) -> list:
    path = urllib.parse.quote("ontologies/addition_history.json", safe="")
    resp = requests.get(f"{BASE}/get-artifact?path={path}&run_uuid={run_id}", auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()


def describe(label: str, xs: list) -> str:
    return (
        f"{label:<26} mean={st.mean(xs):.4f} sd={st.stdev(xs):.4f} "
        f"median={st.median(xs):.4f} min={min(xs):.4f} max={max(xs):.4f}"
    )


def build_rows(configs: list, children: dict, ticker: str) -> list:
    rows = []
    for r in configs:
        rid = r["info"]["run_id"]
        p, m, ch = params(r), metrics(r), children[rid]
        baseline = next(
            metrics(c)["auc"] for c in ch if tags(c).get("mlflow.runName") == "baseline_article_embedding"
        )
        final, final_tag = final_accepted_auc(ch, baseline)
        rows.append(
            {
                "ticker": ticker,
                "run_id": rid,
                "lookback_days": int(p["lookback_days"]),
                "steps": int(p["steps"]),
                "chain_hops": int(p["min_chain_hops"]),
                "evolution_prompt": p.get("evolution_prompt", "default")
                .split("/")[-1]
                .replace("_prompt_template.txt", ""),
                "baseline_auc": round(baseline, 6),
                "final_auc": round(final, 6),
                "delta": round(final - baseline, 6),
                "train_auc": final_accepted_train_auc(ch),
                "final_candidate": final_tag,
                "path_mean_norm": m.get("feat/path_mean_norm"),
                "brier_score": m.get("brier_score"),
                "f1": m.get("f1"),
                "n_nodes": m.get("graph/n_nodes"),
                "n_edges": m.get("graph/n_edges"),
                "max_hops_val": m.get("max_hops_val"),
            }
        )
    rows.sort(key=lambda r: -r["final_auc"])
    return rows


def summarise(rows: list, additions: list, ticker: str) -> list:
    """The per-ticker block of headline_numbers.txt."""
    finals = [r["final_auc"] for r in rows]
    bases = [r["baseline_auc"] for r in rows]
    deltas = [r["delta"] for r in rows]

    out = [
        "=" * 72,
        f"{ticker}  —  {len(rows)} configs, {len(additions)} addition records",
        "=" * 72,
        "",
        describe("final AUC", finals),
        describe("baseline AUC", bases),
        describe("delta", deltas),
        f"beat baseline: {sum(1 for d in deltas if d > 0)} | below: {sum(1 for d in deltas if d < 0)}"
        f" | equal: {sum(1 for d in deltas if d == 0)}",
    ]
    trains = [r["train_auc"] for r in rows if r["train_auc"] is not None]
    if trains:
        out.append(
            f"train AUC (accepted winner): mean={st.mean(trains):.4f} "
            f"min={min(trains):.4f} max={max(trains):.4f}  n={len(trains)}"
        )
    else:
        out.append("train AUC: not logged for this half (predates the 2026-08-11 change)")

    out += ["", "FACTOR MEANS (best first)"]
    for factor in ("lookback_days", "steps", "chain_hops", "evolution_prompt"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[factor]].append(r["final_auc"])
        out.append(f"  {factor}:")
        for level, vals in sorted(groups.items(), key=lambda kv: -st.mean(kv[1])):
            out.append(f"    {str(level):<14} n={len(vals):<3} mean={st.mean(vals):.4f} sd={st.stdev(vals):.4f}")

    empty = sum(1 for r in rows if r["path_mean_norm"] == 0)
    out += ["", f"PATH: empty blocks (path_mean_norm == 0): {empty} of {len(rows)}"]

    steps_groups = defaultdict(list)
    for a in additions:
        steps_groups[(a["run_id"], a["step"])].append(a)
    for group in steps_groups.values():
        # Round before comparing. One MSFT step has two candidates differing by a
        # single ULP (0.40681818181818186 vs 0.4068181818181818) — bit noise from the
        # AUC computation, not a win. Comparing raw floats would award it to whichever
        # happened to land higher, and pandas' CSV reader collapses the two to one
        # value anyway, so the notebook's recomputation would disagree.
        aucs = [round(a["auc"], 9) for a in group]
        top = max(aucs)
        winners = [i for i, v in enumerate(aucs) if v == top]
        for i, a in enumerate(group):
            a["won"] = len(winners) == 1 and i == winners[0]

    by_pair = defaultdict(list)
    for a in additions:
        if a.get("delta_auc") is not None:
            by_pair[(a["node"], a["relationship"])].append(a["won"])
    pair_stats = [(100 * sum(v) / len(v), k, len(v), sum(v)) for k, v in by_pair.items() if len(v) >= 8]
    out += ["", "ADDITION WIN RATES (best first, n >= 8)"]
    for rate, (node, rel), n, wins in sorted(pair_stats, reverse=True):
        out.append(f"  {node:<24} {rel:<34} n={n:<3} won {wins:<3} = {rate:.0f}%")
    out.append(
        f"  ({len(pair_stats)} pairs clear n>=8, covering "
        f"{sum(n for _, _, n, _ in pair_stats)} of {len(additions)} records; "
        f"{len({a['node'] for a in additions})} node types across {len(by_pair)} distinct pairs)"
    )

    by_step = defaultdict(list)
    for a in additions:
        by_step[a["step"]].append(a["status"] == "accepted")
    out += ["", "ACCEPTANCE BY STEP"]
    for step in sorted(by_step):
        out.append(f"  step {step}: {100 * sum(by_step[step]) / len(by_step[step]):.0f}% of {len(by_step[step])}")
    out.append("")
    return out


def main() -> None:
    all_runs = fetch_all_runs()

    children = defaultdict(list)
    for r in all_runs:
        parent = tags(r).get("mlflow.parentRunId")
        if parent:
            children[parent].append(r)

    all_rows, all_additions, out = [], [], []
    for ticker in TICKERS:
        configs = select_configs(all_runs, ticker)
        if len(configs) != 36:
            print(f"WARNING: {ticker} selected {len(configs)} configs, expected 36")
        rows = build_rows(configs, children, ticker)

        additions = []
        for r in rows:
            for a in fetch_addition_history(r["run_id"]):
                a["ticker"] = ticker
                a["run_id"] = r["run_id"]
                a["evolution_prompt"] = r["evolution_prompt"]
                a["patterns"] = json.dumps(a.get("patterns", []))
                additions.append(a)

        out += summarise(rows, additions, ticker)
        all_rows += rows
        all_additions += additions

    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    fields = sorted({k for a in all_additions for k in a})
    with open(HERE / "additions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_additions)

    # Cross-ticker contrast: the point of the whole report.
    out += ["=" * 72, "TSLA vs MSFT", "=" * 72, ""]
    for label, key in (("final AUC", "final_auc"), ("baseline AUC", "baseline_auc"), ("delta", "delta")):
        vals = {t: [r[key] for r in all_rows if r["ticker"] == t] for t in TICKERS}
        out.append(
            f"  {label:<14} TSLA {st.mean(vals['TSLA']):.4f} (sd {st.stdev(vals['TSLA']):.4f})   "
            f"MSFT {st.mean(vals['MSFT']):.4f} (sd {st.stdev(vals['MSFT']):.4f})   "
            f"diff {st.mean(vals['MSFT']) - st.mean(vals['TSLA']):+.4f}"
        )
    for t in TICKERS:
        d = [r["delta"] for r in all_rows if r["ticker"] == t]
        out.append(f"  beat baseline  {t}: {sum(1 for x in d if x > 0)}/{len(d)}")
    # A single AUC on ~42 balanced validation days carries this much noise under H0.
    n1 = 21
    se = math.sqrt((2 * n1 + 1) / (12 * n1 * n1))
    out += [
        "",
        f"  noise floor: SE(AUC) on 42 balanced days under H0 = {se:.4f}",
        "  every effect above is smaller than this, which is the report's main point",
        "",
    ]

    text = "\n".join(out)
    (HERE / "headline_numbers.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
