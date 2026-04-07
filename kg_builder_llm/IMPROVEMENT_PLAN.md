# KG Builder LLM — Improvement Plan

Findings from holistic code review + live graph inspection (2026-04-07).
Issues are ordered by root-cause priority, not surface-level symptom severity.

---

## Current State Summary

The pipeline runs but produces a semantically broken graph:

- Same real-world entity exists as 3–5 separate nodes (Nvidia as `NVDA`, `NVIDIA`, `nvidia`; S&P 500 as `sp500`, `sp_500`, `SPX`, `GSPC`).
- Only 6 non-structural relationship types exist, with `RELATES_TO` covering ~30% of all edges as a catch-all.
- The ontology evolution prompt recommends entity types that are already in the graph.
- Candidate selection has no performance floor — a degrading candidate is still advanced.
- Path-based features are dominated by Nvidia (degree 180 in a 181-node company graph) so embeddings converge regardless of which candidate is being evaluated.

---

## ~~Fix 1 — Entity Deduplication~~ ✅ DONE (2026-04-07)

**Implemented in commit `dd0a283`.**

**Problem:** `normalize_key()` in `core/entity_resolution.py` is never called during the pipeline write path. The LLM generates different keys for the same entity across articles (ticker vs. slug vs. full name). `MERGE_DUPLICATES_CYPHER` is defined but never executed.

**Evidence from live graph:**
```
Nvidia:     NVIDIA, NVDA
Intel:      INTEL, INTC
S&P 500:    sp500, sp_500, SPX, GSPC
Technology: tech, technology (as both Sector and Market)
```

**Fix A — Post-extraction key normalization (in mutator, before Neo4j write):**

In `mutations/incremental_kg_mutator.py`, after parsing the LLM JSON and before calling `_apply_mutation`, run every extracted key through `normalize_key(key, label)`. This ensures keys written to Neo4j are consistent from the start.

```python
# After parsing LLM JSON:
for node in extraction.nodes:
    node["key"] = normalize_key(node["key"], node["label"])
for rel in extraction.relationships:
    rel["from_key"] = normalize_key(rel["from_key"])
    rel["to_key"] = normalize_key(rel["to_key"])
```

**Fix B — Unify ticker vs. name keys:**

The current logic in `normalize_key` uppercases anything ≤5 chars or all-caps, producing `NVDA`, then lowercases long names, producing `nvidia`. These never merge. Pick one strategy for Company nodes: **always use the ticker symbol (uppercase, max 5 chars) as canonical key** when extractable, otherwise use the lowercased slug. Add a known-ticker lookup dict for the core NASDAQ/NYSE symbols in the dataset.

**Fix C — Run periodic deduplication after each step:**

After `build_kg_incremental_candidate()` completes for a step, execute the merge query via `driver.run_query(MERGE_DUPLICATES_CYPHER)`. This catches cross-article drift within a step.

**Fix D — Merge cross-label duplicates (Sector vs Market):**

`Technology` exists as both `Sector` and `Market`. These arise because the ontology has both types and the LLM freely chooses. Either collapse the two into one label during base ontology setup, or add a dedup query that merges same-name nodes across `Market`/`Sector` labels into the one with more connections.

---

## Fix 2 — Extraction Prompt: Key Consistency

**Problem:** The extraction prompt gives contradictory key examples. It says:
- "For person/entity names: Use lowercase with underscores (e.g., `nvidia`, `apple`)"
- But the example JSON shows `{"label": "Company", "key": "NVDA"}` — uppercase ticker

The LLM gets inconsistent guidance and alternates between strategies across articles.

Also: typo in prompt — `"nvidia"` is written as `"nvdia"` in the consistency note.

**Fix — Rewrite the key normalization guide in the prompt:**

Replace the contradictory list with a single rule per entity type:

```
KEY RULES (follow exactly):
- Company: Use the stock ticker in UPPERCASE (e.g., "NVDA", "AAPL", "INTC", "TSLA")
  If ticker unknown, use lowercase slug: "taiwan_semiconductor"
- Sector / Market / Index: Use lowercase slug (e.g., "tech", "sp500", "semiconductors")
- Person (Author, Insider, Executive): Use lowercase slug of full name (e.g., "jensen_huang")
- Deal / Event / Regulation: Use a short lowercase slug describing the event (e.g., "arm_acquisition_2023")
- Never generate keys like "company_1", "deal_1", "market_1" — these are not informative
```

Remove the contradictory examples. Fix the `"nvdia"` typo.

---

## Fix 3 — Ontology Evolution Prompt: Stop Proposing Existing Types

**Problem:** The evolution prompt includes a static suggestion list:
```
Market, Sector, Index, Deal, Contract, Regulation, Product, Event, Quarter,
Competitor, Insider, Executive, Fund, Portfolio, Sentiment, PriceDecrease, PriceIncrease
```

By step 2 most of these already exist in the graph. The LLM adds them again under slightly different names (e.g., `Competitor` instead of `Company`, `PriceEvent` instead of `Event`), inflating the ontology with near-duplicate types.

**Fix A — Populate the "do not add" list dynamically:**

The prompt already builds `existing_nodes` from the schema. Use it to explicitly subtract from the suggestion list:

```python
already_exists = set(existing_nodes)
suggestions = [t for t in ALL_CANDIDATE_TYPES if t not in already_exists]
```

Only pass remaining suggestions to the prompt.

**Fix B — Shift evolution focus to relationship specialization:**

After 1–2 steps the node vocabulary is saturated. The prompt should redirect the LLM to *specialize* existing generic relationship types rather than add more node types:

```
RELATIONSHIP SPECIALIZATION TASK (steps 2+):
The current ontology has generic relationships (RELATES_TO, INVOLVES, PART_OF).
Your task is to REPLACE these with semantically specific relationships, for example:
  - RELATES_TO between Company nodes → COMPETES_WITH, SUPPLIES_TO, ACQUIRES, PARTNERS_WITH
  - INVOLVES between Company and Deal → ACQUIRES, MERGES_WITH, INVESTS_IN
  - PART_OF between Company and Sector → OPERATES_IN, LISTED_IN, HEADQUARTERED_IN
Do NOT add new node types unless none of the above apply.
```

**Fix C — Remove hardcoded AUC thresholds:**

The three branches (`auc < 0.55`, `0.55–0.60`, `> 0.60`) are arbitrary and poorly calibrated. Replace with a relative improvement signal:

```python
improvement = metrics.auc - previous_metrics.auc
# Pass `improvement` to prompt instead of raw AUC
```

Then the prompt branches on "did the last step improve significantly?" rather than absolute AUC bands.

---

## Fix 4 — Candidate Selection Floor

**Problem:** `_select_best_candidate_from_current_step` picks the highest AUC within the current step only, with no comparison to the previous step's winner. If all candidates in step N score AUC 0.47, the system advances a worse-than-random candidate.

**Fix — Add a performance floor:**

```python
def _select_best_candidate_from_current_step(self, step: int, previous_best_auc: float) -> OntologyCandidate:
    step_candidates = self.candidates_per_step[step]
    best = max(step_candidates, key=lambda c: self.results.get(c.candidate_tag, ModelMetrics(0,0,0,0)).auc)
    best_auc = self.results.get(best.candidate_tag, ModelMetrics(0,0,0,0)).auc

    if best_auc <= previous_best_auc:
        logger.warning(
            "Step %d best AUC %.4f does not beat previous %.4f — falling back to previous winner",
            step, best_auc, previous_best_auc,
        )
        return self._previous_winner  # keep previous step's schema

    return best
```

Track `previous_best_auc` across steps in the orchestrator.

---

## Fix 5 — Path Feature Diversity (Structural)

**Problem:** Path extraction in `relationship_chains.py` produces low-diversity embeddings because:
1. Default `min_chain_hops = max_chain_hops = 5` — only 5-hop paths, skipping direct relationships.
2. Paths are ordered `ORDER BY hop_count DESC` then capped at 10 — always the longest, least local paths.
3. Hub nodes (Nvidia degree 180) appear in almost every path; mean-pooled embeddings converge.
4. Chain text format `"node1 -[REL]-> node2"` omits node type labels.

**Fix A — Include node type in chain text:**

```python
# Current:
chain_text = "Tesla -[PRODUCES]-> EV"
# Better:
chain_text = "Company:Tesla -[PRODUCES]-> Product:EV"
```

One line change in `relationship_chains.py` — pass `labels(n)[0]` alongside `n.name` in the Cypher `nodes` projection, then prefix in the Python chain builder.

**Fix B — Deduplicate by rel-type signature before embedding:**

After extracting paths, group by the relationship-type sequence (the metapath). Keep at most 2 paths per unique metapath sequence. This guarantees structural diversity without changing the extraction query:

```python
from collections import defaultdict

def deduplicate_by_metapath(chains: list[dict], max_per_metapath: int = 2) -> list[dict]:
    buckets = defaultdict(list)
    for chain in chains:
        sig = tuple(chain["rel_types"])
        buckets[sig].append(chain)
    return [c for bucket in buckets.values() for c in bucket[:max_per_metapath]]
```

Call this in `build_day_feature_vector` after `extract_chains_batch` and before `embed_relationship_chains`.

**Fix C — Variable hop lengths with stratum sampling:**

Change defaults to `min_chain_hops=1, max_chain_hops=5`. Then when collecting chains per article, ensure at least one path per hop-length stratum (1-hop, 2-hop, …) before filling remaining budget with longer paths. Prevents the 5-hop-only collapse.

---

## Fix 6 — Subgraph Features: Replace Hash Buckets with Vocabulary Counts

**Problem:** `subgraph_features.py` uses MD5 hash bucketing (16 node buckets, 24 rel buckets, 48 metapath buckets). Different entity types collide into the same bucket, destroying signal. With an ontology of ~12 node types and ~6 rel types, the vocabulary is small enough to use exact counts.

**Fix — Build vocabulary once, use count vectors:**

```python
# Known node types (from base + expected candidates):
NODE_VOCAB = ["Company", "Sector", "Market", "Index", "Deal", "Event",
              "Regulation", "Insider", "Author", "Contract", "Day", "Article"]

REL_VOCAB = ["PART_OF", "RELATES_TO", "INVOLVES", "PARTICIPATES_IN",
             "GOVERNED_BY", "FILES", "MENTIONS", "PUBLISHED_ON"]
```

Replace `_hashed_histogram` with `_vocab_histogram` that directly maps token to index via a dict lookup. Zero-pad unknown types. This is a ~20-line change in `subgraph_features.py` that eliminates all collision noise and makes XGBoost feature importances interpretable (you know which node type / rel type is important).

Metapaths are too numerous for a fixed vocab; keep hash-bucketing only for those, but increase bucket count to 128.

---

## ~~Fix 7 — Minor Bugs~~ ✅ DONE (2026-04-07)

| Bug | Fix |
|-----|-----|
| `"nvdia"` typo in extraction prompt | Fixed |
| `normalize_key()` never called on write path | Replaced by `canonical_key()` |
| `MERGE_DUPLICATES_CYPHER` defined but unused | Now executed via `_deduplicate_graph()` after each step |
| Price fetch uses Stooq (rate-limited) | Switched to yfinance |

---

## Suggested Implementation Order

```
Fix 1A  →  Fix 2  →  Fix 1C  →  Fix 3  →  Fix 4  →  Fix 5B  →  Fix 5A  →  Fix 6
 (key     (prompt   (dedup    (prompt   (floor)   (metapath  (node    (vocab
  norm)    fixes)   query)    evol.)             dedup)     labels)  counts)
```

Fixes 1A and 2 together should visibly reduce the duplicate node count in a single pipeline run. Fix 3 addresses the feedback loop quality. Fixes 5 and 6 are ML improvements that only matter after the graph is structurally sound.

---

## Validation Checklist

After each fix, verify:

- [ ] `MATCH (a),(b) WHERE labels(a)[0]=labels(b)[0] AND a.name=b.name AND elementId(a)<>elementId(b) RETURN count(*)` → should decrease toward 0
- [ ] Distinct relationship types in graph → should increase from 6 toward 10+
- [ ] AUC variance across candidates in the same step → should become non-trivial (candidates should differ meaningfully)
- [ ] Path texts in logs → should show node type prefixes and structural variety
- [ ] Step winner AUC should be monotonically non-decreasing across steps
