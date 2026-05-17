# kg_builder_temporal

Temporal knowledge graph builder for finance research. Evolves a Neo4j knowledge graph incrementally from financial news articles and evaluates each ontology variant's predictive power for next-day stock price direction (binary classification).

## Architecture

The system runs as a multi-step experiment loop:

1. **Base structure** — loads articles from CSV, creates `Article` and `Day` nodes in Neo4j, fetches OHLC prices from Stooq, writes `direction` / `return_next_day` labels onto `Day` nodes.
2. **Ontology evolution** — `OntologyEvolutionAgent` uses an LLM (GPT-4o-mini by default) to propose evolved entity/relationship schemas at each step.
3. **Incremental KG mutation** — `IncrementalArticleKGMutator` extracts entities/relationships from article text using the proposed ontology and writes them to Neo4j, tagged with a `candidate_tag` on every node and relationship for isolation.
4. **Evaluation** — `evaluate_candidate` computes rolling 5-day HOPE embeddings (HOPE via `karateclub`) over temporal graph snapshots and trains an XGBoost classifier. Reports AUC and F1.
5. **Selection** — best candidate by AUC is carried forward as the parent for the next evolution step; losing candidates' tags are pruned from the graph (nodes/rels kept, tags removed).

### Module map

```
kg_builder_temporal/
├── main.py                        # CLI entry point and arg parsing
├── config.py                      # Neo4jConfig, OpenAIConfig, ExperimentConfig (env-driven)
├── core/
│   ├── data.py                    # CSV loading, date windowing, per-day article limits
│   ├── graph.py                   # GraphDriver wrapper around neo4j.Driver
│   ├── neo4j_io.py                # Write price labels to Day nodes
│   ├── ontology.py                # OntologyCandidate dataclass, create_base_ontology()
│   ├── ontology_io.py             # Save ontology candidates and summary JSON to disk
│   ├── article_linking.py         # Create Day nodes and PUBLISHED_ON relationships
│   ├── entity_resolution.py       # normalize_key() for deduplication
│   └── tagging.py                 # Tag nodes/rels with candidate_tag in bulk
├── mutations/
│   ├── base.py                    # build_kg_incremental_candidate() — top-level build fn
│   └── incremental_kg_mutator.py  # IncrementalArticleKGMutator — LLM extraction + Neo4j write
├── pipeline/
│   ├── orchestrator.py            # Orchestrator — drives the full experiment loop
│   ├── evaluator.py               # evaluate_candidate() — HOPE embeddings + XGBoost
│   └── ontology_evolution.py      # OntologyEvolutionAgent — LLM-based schema proposals
└── ml/
    ├── temporal_hope_embeddings.py # compute_rolling_hope_embeddings() — per-day HOPE
    ├── modeling.py                 # train_classifier_on_embeddings(), ModelMetrics, temporal split
    ├── topology_features.py        # compute_topology_features() — degree/mention features
    ├── feature_engineering.py      # Additional feature transforms
    ├── relationship_chains.py      # Relationship chain extraction helpers
    └── embeddings.py              # Embedding utilities
```

## Running

```bash
# Minimal run (local embeddings, 3 steps, 2 candidates/step)
python -m kg_builder_temporal.main \
  --data data/fnspid_sample_nasdaq_long_text.csv \
  --steps 3 \
  --candidates 2

# Full options
python -m kg_builder_temporal.main \
  --data data/fnspid_sample_nasdaq_long_text.csv \
  --time-window-days 150 \
  --articles-per-day 3 \
  --steps 3 \
  --candidates 2 \
  --embedding-type local \
  --local-model all-MiniLM-L6-v2 \
  --semaphore-limit 50 \
  --output results/ontology_experiment_results.json \
  --log-level INFO
```

## Environment variables

Copy `.env` into the repo root (three levels up from this package). Required:

| Variable | Default | Notes |
|---|---|---|
| `NEO4J_URI` | `neo4j+s://localhost:7687` | |
| `NEO4J_USERNAME` | `neo4j` | |
| `NEO4J_PASSWORD` | _(required)_ | |
| `NEO4J_DATABASE` | `neo4j` | |
| `OPENAI_API_KEY` | _(required for openai embedding type)_ | |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for ontology evolution and entity extraction |
| `EMBEDDING_TYPE` | `local` | `local` (sentence-transformers, free) or `openai` |
| `LOCAL_MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers model name |

## Key design decisions

- **Candidate tagging, not graph copies** — All nodes and relationships carry a `candidate_tags: [str]` property. Multiple ontology candidates coexist in the same graph; evaluation filters by tag. Losing candidates are pruned (tag removed) rather than deleted.
- **Incremental mutation, not full rebuild** — Each evolution step adds only the new entity types / relationship types on top of the accepted base, avoiding redundant LLM calls on unchanged articles.
- **Rolling HOPE embeddings** — For each day `t`, a 5-day temporal subgraph `[t-4, t]` is materialized, HOPE embeddings computed, and entity embeddings aggregated into a single day-level vector used as XGBoost features.
- **Stooq for price labels** — Real OHLC data is fetched at runtime from Stooq (`https://stooq.com`) based on the ticker column in the article CSV.

## Dev setup

```bash
uv sync
uv run python -m kg_builder_temporal.main --help
```

Tests (if present) live in `tests/` at the repo root:

```bash
uv run pytest
```

Dependencies are managed via `pyproject.toml`. Python ≥ 3.11 required. Notable heavy deps: `torch`, `sentence-transformers`, `karateclub`, `xgboost`, `neo4j-graphrag`.
