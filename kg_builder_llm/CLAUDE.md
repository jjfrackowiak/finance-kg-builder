# kg_builder_llm

LLM-driven finance knowledge graph pipeline.

## Project Layout

- `main.py`, `cli.py` — CLI entry points
- `config.py`, `logger.py`, `constants.py` — runtime config and logging
- `core/` — data loading, ontology models, Neo4j I/O, tagging, entity resolution
- `pipeline/` — orchestration, ontology evolution, candidate evaluation
- `ml/` — feature engineering, embeddings, relationship-chain extraction, model training
- `mutations/` — incremental graph mutation logic
- `scripts/` — one-off maintenance utilities

Project-level metadata and tests live in the repository root.

## Common Commands

Run from the repository root:

```bash
python -m kg_builder_llm.main --help
python -m kg_builder_llm.main --data data/articles.csv --steps 3
pytest
pytest tests/test_pipeline.py
black . && ruff check . && mypy kg_builder_llm
```

```bash
uv run python -m kg_builder_llm.main \
  --data data/fnspid_sample_nasdaq_long_text.csv \
  --time-window-days 100 \
  --articles-per-day 3 \
  --steps 3 \
  --candidates 2 \
  --embedding-type local \
  --local-model all-MiniLM-L6-v2 \
  --lookback-days 3 \
  --min-chain-hops 5 \
  --max-chain-hops 6 \
  --path-uniqueness NODE_PATH \
  --train-ratio 0.75 \
  --output results/experiment_100days_unique_paths.json \
  --log-level INFO
```
## Environment Variables

Required: `NEO4J_URI`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`. Never commit `.env` files or credentials.

## Code Style

- Python 3.11, 4-space indentation, type hints on all public functions
- Line length: 100 characters (`black` + `ruff`)
- `snake_case` for modules and functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Keep functions narrow; add explicit logging around graph mutations, LLM calls, and evaluation steps

## Testing

- `pytest` with `pytest-asyncio` and coverage enabled
- Test files match `tests/test_*.py`
- Add or update tests for changes in `core/`, `pipeline/`, `ml/`, or `mutations/`

## Commits & PRs

Short imperative commit messages (e.g. `add batch node path extraction`). PRs should include:
- Summary of the behavior change
- Any env/Neo4j setup notes
- Test coverage or manual verification steps
- Sample CLI invocation if experiment flow or outputs change

## ML Feature Engineering

`build_day_feature_vector` in `ml/feature_engineering.py` supports three modes via `--feature-mode`: `path`, `subgraph`, `hybrid`.

The subgraph block (`ml/subgraph_features.py`) concatenates four fixed-size components:
- **Node type histogram** — 16 hash buckets
- **Relationship type histogram** — 24 hash buckets (`compute_relation_type_counts`) — this is the "rel type frequency vector" signal; implemented but uses hash bucketing, so XGBoost importances map to buckets, not named rel types
- **Metapath counts** — 48 hash buckets (2-hop and 3-hop typed paths)
- **Temporal novelty stats** — 8 scalars (article count, entity novelty ratio, etc.)

Topology features (degree, mention counts) are computed separately in `ml/topology_features.py` and always appended last.

## Neo4j / AuraDB

The AuraDB instance is accessible via the Neo4j MCP server configured for this project (`mcp-neo4j-cypher` via `uvx`). Available MCP tools: `get-neo4j-schema`, `read-neo4j-cypher`. Credentials are in `.env` (never commit).

## Caution

The orchestrator **wipes the target Neo4j database** at the start of each run. Handle graph-clearing logic carefully.
