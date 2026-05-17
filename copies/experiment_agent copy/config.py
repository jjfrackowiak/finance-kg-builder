
# file: finance_kg_experiment/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

@dataclass
class ExperimentConfig:
    # Data
    csv_path: str = "fnspid_sample_nasdaq.csv"
    target_ticker: str = "TSLA"
    n_days: int = 20
    max_articles_per_day: int = 2

    # Ontology search
    n_steps: int = 1               # number of ontology augmentation steps
    candidates_per_step: int = 3   # ontology candidates per step

    # Embeddings
    embedding_dim: int = 128

    # Modeling
    train_fraction: float = 0.7    # temporal split: first 70% days train, rest val
    random_state: int = 42

    # Price source
    use_stooq: bool = True         # use Stooq for TSLA prices

@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
