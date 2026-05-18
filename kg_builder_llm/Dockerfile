FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY kg_builder_llm/ ./kg_builder_llm/

ENTRYPOINT ["uv", "run", "python", "-m", "kg_builder_llm.main"]
