FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-extra local --no-extra sweep

COPY kg_builder_llm/ ./kg_builder_llm/
COPY resources/ ./resources/

ENV PATH="/app/.venv/bin:$PATH"

# Smoke-test: catch missing imports at build time, not at runtime in prod
RUN python -c "from kg_builder_llm.pipeline.orchestrator import Orchestrator; from kg_builder_llm.mutations.base import build_kg_incremental_candidate; from kg_builder_llm.main import main; print('imports OK')"

ENTRYPOINT ["python", "-m", "kg_builder_llm.main"]
