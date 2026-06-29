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
    base_url: str = ""

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Load from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "fake"),
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL", ""),
        )


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    tracking_uri: str = ""
    experiment_name: str = "kg-ontology-evolution"

    @classmethod
    def from_env(cls) -> "MLflowConfig":
        """Load from environment variables."""
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "kg-ontology-evolution"),
        )


@dataclass
class FeatureConfig:
    """Feature engineering hyperparameters — passed as one object instead of 8 individual args."""

    lookback_days: int = 2
    embedding_type: str = "local"
    local_model_name: str = "all-MiniLM-L6-v2"
    min_chain_hops: int = 5
    max_chain_hops: int = 5
    path_uniqueness: str = "NODE_PATH"
    feature_mode: str = "path"
    max_metapath_hops: int = 2
    train_ratio: float = 0.7

    @classmethod
    def from_env(cls) -> "FeatureConfig":
        return cls(
            lookback_days=int(os.getenv("LOOKBACK_DAYS", "2")),
            embedding_type=os.getenv("EMBEDDING_TYPE", "local"),
            local_model_name=os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2"),
            min_chain_hops=int(os.getenv("MIN_CHAIN_HOPS", "5")),
            max_chain_hops=int(os.getenv("MAX_CHAIN_HOPS", "5")),
            path_uniqueness=os.getenv("PATH_UNIQUENESS", "NODE_PATH"),
            feature_mode=os.getenv("FEATURE_MODE", "path"),
            max_metapath_hops=int(os.getenv("MAX_METAPATH_HOPS", "2")),
            train_ratio=float(os.getenv("TRAIN_RATIO", "0.7")),
        )


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    target_ticker: str = "TSLA"
    sample_size: int = 10
    max_candidates_per_step: int = 2
    num_steps: int = 2
    semaphore_limit: int = 50
    evolution_prompt_template: str = "default"
    feature: FeatureConfig = None

    def __post_init__(self) -> None:
        if self.feature is None:
            self.feature = FeatureConfig()

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Load from environment variables."""
        return cls(
            target_ticker=os.getenv("TARGET_TICKER", "TSLA"),
            sample_size=int(os.getenv("SAMPLE_SIZE", "10")),
            max_candidates_per_step=int(os.getenv("MAX_CANDIDATES", "2")),
            num_steps=int(os.getenv("NUM_STEPS", "2")),
            semaphore_limit=int(os.getenv("SEMAPHORE_LIMIT", "50")),
            evolution_prompt_template=os.getenv("EVOLUTION_PROMPT_TEMPLATE", "default"),
            feature=FeatureConfig.from_env(),
        )


@dataclass
class BedrockLLMConfig:
    """Bedrock LLM config for reasoning-heavy calls (ontology evolution)."""

    model_id: str = "qwen.qwen3-32b-v1:0"
    region: str = "eu-central-1"

    @classmethod
    def from_env(cls) -> "BedrockLLMConfig":
        return cls(
            model_id=os.getenv("ONTOLOGY_LLM_MODEL", "qwen.qwen3-32b-v1:0"),
            region=os.getenv("AWS_REGION", "eu-central-1"),
        )

    def base_url(self) -> str:
        return f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/converse"


@dataclass
class Config:
    """Main configuration container."""

    neo4j: Neo4jConfig
    openai: OpenAIConfig
    experiment: ExperimentConfig
    mlflow: MLflowConfig = None
    bedrock: BedrockLLMConfig = None

    def __post_init__(self) -> None:
        if self.mlflow is None:
            self.mlflow = MLflowConfig()
        if self.bedrock is None:
            self.bedrock = BedrockLLMConfig()

    @classmethod
    def from_env(cls) -> "Config":
        """Load all configurations from environment."""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            experiment=ExperimentConfig.from_env(),
            mlflow=MLflowConfig.from_env(),
            bedrock=BedrockLLMConfig.from_env(),
        )
