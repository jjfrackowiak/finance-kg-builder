# file: finance_kg_experiment/cli.py
from dotenv import load_dotenv
load_dotenv()
import logging
from logging_config import setup_logging

# Setup logging to silence verbose libraries
setup_logging(level=logging.INFO)

import os
from config import Neo4jConfig, ExperimentConfig, OpenAIConfig
from pipeline import run_ontology_experiment
import asyncio

async def main():
    neo4j_cfg = Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )

    exp_cfg = ExperimentConfig(
        csv_path="data/fnspid_sample_nasdaq_long_text.csv",
        target_ticker=os.getenv("TARGET_TICKER", "TSLA"),
        n_days=10,
        max_articles_per_day=1,
        n_steps=2,
        candidates_per_step=2,
        embedding_dim=128,
        train_fraction=0.7,
    )

    openai_cfg = OpenAIConfig(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    report = await run_ontology_experiment(neo4j_cfg, exp_cfg, openai_cfg)
    report.pretty_print()

    output_path = "ontology_experiment_results.json"
    report.save_json(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())  # Use asyncio.run to execute the async main function

# Latest Run 3 candidates, 1 step

# === Ontology Experiment Report ===
# Total candidates tested: 3

# Step 0 candidate 0: base_v1
#   Desc: Base ontology with Company, Article, Day and MENTIONS/PUBLISHED_ON.
#   AUC: 0.3263 | F1: 0.2500
#   Train N: 65 | Val N: 67

# Step 0 candidate 1: base_v1_variant1
#   Desc: Augmented ontology variant 1: Adds Event and Action nodes for event-based semantic extraction.
#   AUC: 0.6875 | F1: 0.4262
#   Train N: 131 | Val N: 119

# Step 0 candidate 2: base_v1_variant2
#   Desc: Augmented ontology variant 2: Adds Sentiment and Emotion nodes with relationships to Articles.
#   AUC: 0.6359 | F1: 0.3564
#   Train N: 186 | Val N: 174

# Best candidate by AUC:
#   base_v1_variant1 (step 0, idx 1)
#   AUC: 0.6875 | F1: 0.4262