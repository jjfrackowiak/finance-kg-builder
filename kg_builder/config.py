"""Configuration management for kg_builder."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load from environment variables."""
        uri = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        return cls(uri=uri, user=user, password=password, database=database)


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""

    api_key: str
    model_name: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        return cls(api_key=api_key, model_name=model_name)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    target_ticker: str = "TSLA"
    sample_size: int = 10
    max_candidates_per_step: int = 2
    num_steps: int = 2

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Load from environment variables."""
        return cls(
            target_ticker=os.getenv("TARGET_TICKER", "TSLA"),
            sample_size=int(os.getenv("SAMPLE_SIZE", "10")),
            max_candidates_per_step=int(os.getenv("MAX_CANDIDATES", "2")),
            num_steps=int(os.getenv("NUM_STEPS", "2")),
        )


@dataclass
class Config:
    """Main configuration container."""

    neo4j: Neo4jConfig
    openai: OpenAIConfig
    experiment: ExperimentConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load all configurations from environment."""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            experiment=ExperimentConfig.from_env(),
        )
