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
    semaphore_limit: int = 50
    embedding_type: str = "local"  # "local" or "openai"
    local_model_name: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    
    # Feature engineering hyperparameters
    lookback_days: int = 2  # Days to look back for article/feature extraction
    min_chain_hops: int = 5  # Minimum path length for relationship chains
    max_chain_hops: int = 5  # Maximum path length for relationship chains
    path_uniqueness: str = "NODE_PATH"  # APOC path uniqueness: NODE_PATH, NODE_GLOBAL, RELATIONSHIP_PATH, RELATIONSHIP_GLOBAL
    feature_mode: str = "path"  # "path", "subgraph", or "hybrid"
    max_metapath_hops: int = 2  # Maximum hop count for typed metapath features
    
    # Model training hyperparameters
    train_ratio: float = 0.7  # Fraction of data for training (0.7 = 70/30 split)
    
    # Ontology evolution
    evolution_prompt_template: str = "default"  # Path to custom prompt template or "default"

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Load from environment variables."""
        return cls(
            target_ticker=os.getenv("TARGET_TICKER", "TSLA"),
            sample_size=int(os.getenv("SAMPLE_SIZE", "10")),
            max_candidates_per_step=int(os.getenv("MAX_CANDIDATES", "2")),
            num_steps=int(os.getenv("NUM_STEPS", "2")),
            semaphore_limit=int(os.getenv("SEMAPHORE_LIMIT", "50")),
            embedding_type=os.getenv("EMBEDDING_TYPE", "local"),
            local_model_name=os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2"),
            lookback_days=int(os.getenv("LOOKBACK_DAYS", "2")),
            min_chain_hops=int(os.getenv("MIN_CHAIN_HOPS", "5")),
            max_chain_hops=int(os.getenv("MAX_CHAIN_HOPS", "5")),
            path_uniqueness=os.getenv("PATH_UNIQUENESS", "NODE_PATH"),
            feature_mode=os.getenv("FEATURE_MODE", "path"),
            max_metapath_hops=int(os.getenv("MAX_METAPATH_HOPS", "2")),
            train_ratio=float(os.getenv("TRAIN_RATIO", "0.7")),
            evolution_prompt_template=os.getenv("EVOLUTION_PROMPT_TEMPLATE", "default"),
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
