"""Pull and analyse the 36 valid TSLA sweep runs (200-day window, post-36f4f88).

This supersedes `analysis/interim_2026-08-07*`, which pooled MSFT+TSLA, included
`candidates=3` runs from the abandoned design, and — decisively — included runs
from before the `embed_relationship_chains` fix, whose path feature block was
silently all zeros. Those runs must not be pooled with these.

Run selection (each condition is necessary; together they yield exactly 36):

  experiment_id "0"          — kg-ontology-evolution
  no mlflow.parentRunId      — top-level run, one per config
  status FINISHED
  param time_window_days=200 — the current window design (2022-05-02 → 2022-11-18)
  param ticker=TSLA          — MSFT half of the design is unrun
  param candidates=2         — the 36-config design; earlier dispatches used 3
  start_time >= 2026-08-08T12:00Z — after the path-embedding fix (36f4f88)

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


def select_configs(runs: list) -> list:
    return [
        r
        for r in runs
        if "mlflow.parentRunId" not in tags(r)
        and r["info"]["status"] == "FINISHED"
        and params(r).get("time_window_days") == "200"
        and params(r).get("ticker") == "TSLA"
        and params(r).get("candidates") == "2"
        and r["info"]["start_time"] >= CUTOFF_MS
    ]


def final_accepted_auc(children: list, baseline: float) -> tuple:
    """AUC of the last accepted step winner, falling back to the baseline."""
    auc, tag = baseline, "(no step accepted)"
    for c in sorted(children, key=lambda c: tags(c).get("mlflow.runName", "")):
        t = tags(c)
        if t.get("winner") == "true" and t.get("step_accepted") == "true":
            auc, tag = metrics(c)["auc"], t["mlflow.runName"]
    return auc, tag


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


def main() -> None:
    all_runs = fetch_all_runs()
    configs = select_configs(all_runs)

    children = defaultdict(list)
    for r in all_runs:
        parent = tags(r).get("mlflow.parentRunId")
        if parent:
            children[parent].append(r)

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

    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    additions = []
    for r in rows:
        for a in fetch_addition_history(r["run_id"]):
            a["run_id"] = r["run_id"]
            a["evolution_prompt"] = r["evolution_prompt"]
            a["patterns"] = json.dumps(a.get("patterns", []))
            additions.append(a)
    with open(HERE / "additions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(additions[0]))
        w.writeheader()
        w.writerows(additions)

    finals = [r["final_auc"] for r in rows]
    bases = [r["baseline_auc"] for r in rows]
    deltas = [r["delta"] for r in rows]
    se = st.stdev(deltas) / math.sqrt(len(deltas))

    out = [f"n = {len(rows)} configs, {len(additions)} addition records", ""]
    out += [describe("final AUC", finals), describe("baseline AUC", bases), describe("delta", deltas)]
    out += [
        f"paired t = {st.mean(deltas) / se:+.2f} (SE {se:.4f})",
        f"beat baseline: {sum(1 for d in deltas if d > 0)} | below: {sum(1 for d in deltas if d < 0)}"
        f" | equal: {sum(1 for d in deltas if d == 0)}",
        "",
        "FACTOR MEANS (best first)",
    ]
    for factor in ("lookback_days", "steps", "chain_hops", "evolution_prompt"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[factor]].append(r["final_auc"])
        out.append(f"  {factor}:")
        for level, vals in sorted(groups.items(), key=lambda kv: -st.mean(kv[1])):
            out.append(f"    {str(level):<14} n={len(vals):<3} mean={st.mean(vals):.4f} sd={st.stdev(vals):.4f}")

    h5 = [r["final_auc"] for r in rows if r["chain_hops"] == 5]
    rest = [r["final_auc"] for r in rows if r["chain_hops"] in (3, 4)]
    ab_se = math.sqrt(st.variance(h5) / len(h5) + st.variance(rest) / len(rest))
    empty = sum(1 for r in rows if r["path_mean_norm"] == 0)
    out += [
        "",
        "PATH ABLATION",
        f"  h5   n={len(h5)} mean={st.mean(h5):.4f} sd={st.stdev(h5):.4f}",
        f"  h3+4 n={len(rest)} mean={st.mean(rest):.4f} sd={st.stdev(rest):.4f}",
        f"  difference {st.mean(h5) - st.mean(rest):+.4f} SE {ab_se:.4f} t={(st.mean(h5) - st.mean(rest)) / ab_se:+.2f}",
        f"  empty path blocks (path_mean_norm == 0): {empty} of {len(rows)}",
        "",
        "NODE EFFECTS (best first, n >= 8)",
    ]
    by_node = defaultdict(list)
    for a in additions:
        if a.get("delta_auc") is not None:
            by_node[a["node"]].append((a["delta_auc"], a["status"]))
    node_stats = [
        (st.mean([d for d, _ in v]), k, len(v), st.stdev([d for d, _ in v]),
         100 * sum(1 for _, s in v if s == "accepted") / len(v))
        for k, v in by_node.items()
        if len(v) >= 8
    ]
    for mean, node, n, sd, acc in sorted(node_stats, reverse=True):
        out.append(f"  {node:<26} n={n:<3} mean={mean:+.4f} sd={sd:.3f} accepted={acc:.0f}%")

    by_step = defaultdict(list)
    for a in additions:
        by_step[a["step"]].append(a["status"] == "accepted")
    out += ["", "ACCEPTANCE BY STEP"]
    for step in sorted(by_step):
        out.append(f"  step {step}: {100 * sum(by_step[step]) / len(by_step[step]):.0f}% of {len(by_step[step])}")

    text = "\n".join(out)
    (HERE / "headline_numbers.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
