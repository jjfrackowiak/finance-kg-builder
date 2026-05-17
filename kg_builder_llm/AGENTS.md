# Repository Guidelines

## Project Structure & Module Organization

This directory contains the `kg_builder_llm` package, an LLM-driven finance knowledge graph pipeline. Core areas:

- `main.py`, `cli.py`: CLI entry points for running experiments.
- `config.py`, `logger.py`, `constants.py`: runtime configuration and logging.
- `core/`: data loading, ontology models, Neo4j I/O, tagging, entity resolution.
- `pipeline/`: orchestration, ontology evolution, and candidate evaluation.
- `ml/`: feature engineering, embeddings, relationship-chain extraction, model training.
- `mutations/`: incremental graph mutation logic.
- `scripts/`: one-off maintenance utilities.

Project-level metadata and tests are defined in the repository root, not in this package subdirectory.

## Build, Test, and Development Commands

Run commands from the repository root when possible:

- `python -m kg_builder_llm.main --help`: show CLI options.
- `python -m kg_builder_llm.main --data data/articles.csv --steps 3`: run an experiment locally.
- `pytest`: run the test suite from `tests/`.
- `pytest tests/test_pipeline.py`: run a focused test file.
- `black . && ruff check . && mypy kg_builder_llm`: format and validate Python code.

The code expects Neo4j and environment variables such as `NEO4J_URI`, `NEO4J_PASSWORD`, and `OPENAI_API_KEY`.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, and type hints for public functions. Follow the repo’s configured style:

- `black` and `ruff` use a 100-character line length.
- modules and functions: `snake_case`
- classes and dataclasses: `PascalCase`
- constants: `UPPER_SNAKE_CASE`

Keep functions narrow and favor explicit logging around graph mutations, LLM calls, and evaluation steps.

## Testing Guidelines

Tests use `pytest` with `pytest-asyncio` and coverage enabled. Test files follow the `tests/test_*.py` pattern. Add or update tests for changes in `core/`, `pipeline/`, `ml/`, or `mutations/`, especially for behavior that affects Neo4j queries, ontology evolution, or feature generation.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit messages such as `add batch node path extraction and batch embedding logic`. Keep commits focused and descriptive.

For pull requests, include:

- a short summary of the behavior change
- any required env or Neo4j setup notes
- test coverage or manual verification steps
- sample CLI invocation when changing experiment flow or outputs

## Security & Configuration Tips

Do not commit `.env` files, API keys, or Neo4j credentials. Treat graph-clearing logic carefully: the orchestrator currently wipes the target database at the start of a run.
