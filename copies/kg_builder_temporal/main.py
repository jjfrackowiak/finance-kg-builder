"""Main CLI entry point."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from copies.kg_builder_temporal.config import Config
from copies.kg_builder_temporal.core.data import filter_articles_by_date_window, load_articles, prepare_articles
from copies.kg_builder_temporal.core.graph import GraphDriver
from copies.kg_builder_temporal.logger import get_logger, setup_logging
from copies.kg_builder_temporal.pipeline.orchestrator import Orchestrator

logger = get_logger(__name__)


# Load .env from parent directory
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KG Builder - Knowledge Graph Evolution with LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kg_builder.main --data data/articles.csv --steps 4
  python -m kg_builder.main --data data/articles.csv --limit 100
  python -m kg_builder.main --help
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/fnspid_sample_nasdaq_long_text.csv",
        help="Path to articles CSV file (default: data/fnspid_sample_nasdaq_long_text.csv)",
    )

    parser.add_argument(
        "--time-window-days",
        type=int,
        default=150,
        help="Number of sequential days to use for training (default: 150)",
    )

    parser.add_argument(
        "--articles-per-day",
        type=int,
        default=3,
        help="Max articles per day (default: all articles in time window)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(DEPRECATED) Use --days instead. Limit number of articles to process",
    )

    parser.add_argument(
        "--steps", type=int, default=3, help="Number of incremental evolution steps after base structure (0=base only, 1=base+1 evolution step, default: 2)"
    )

    parser.add_argument(
        "--candidates", type=int, default=2, help="Number of candidates per step (default: 3)"
    )

    parser.add_argument(
        "--semaphore-limit",
        type=int,
        default=50,
        help="Max concurrent article processing tasks (default: 50)",
    )

    parser.add_argument(
        "--embedding-type",
        type=str,
        default="local",
        choices=["local", "openai"],
        help="Embedding type: 'local' (sentence-transformers, free) or 'openai' (API, paid) (default: local)",
    )

    parser.add_argument(
        "--local-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Local embedding model name for sentence-transformers (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/ontology_experiment_results.json",
        help="Output path for results (default: results/ontology_experiment_results.json)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def fetch_stooq_prices(symbol: str = "TSLA") -> pd.DataFrame:
    """Fetch daily OHLC data from Stooq.

    Args:
        symbol: Stock ticker symbol (default: TSLA)

    Returns:
        DataFrame with columns: ['day', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    logger.info(f"Fetching price data for {symbol} from Stooq...")
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"

    try:
        df = pd.read_csv(url)
        df = df.rename(
            columns={
                "Date": "day",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
        )
        df = df.dropna()
        logger.info(f"✓ Fetched {len(df)} days of price data for {symbol}")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to fetch price data: {e}")
        raise


def build_return_labels(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute next-day returns and binary direction label.

    Args:
        price_df: Price data from fetch_stooq_prices

    Returns:
        DataFrame indexed by 'day' with columns: ['return_next_day', 'direction']
    """
    logger.info("Computing return labels...")

    price_df = price_df.copy()
    price_df["next_close"] = price_df["Close"].shift(-1)
    price_df["return_next_day"] = (price_df["next_close"] - price_df["Close"]) / price_df["Close"]

    price_df = price_df.dropna(subset=["return_next_day"])

    # direction label: 1 = up, 0 = down or flat
    price_df["direction"] = (price_df["return_next_day"] > 0.0).astype(int)

    df_labels = price_df[["day", "return_next_day", "direction"]].copy()
    df_labels.set_index("day", inplace=True)
    logger.info(f"✓ Gathered labels for {len(df_labels)} days")
    return df_labels


async def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if args is None:
        args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    logger.info("=" * 80)
    logger.info("KG Builder - Knowledge Graph Evolution with LLM")
    logger.info("=" * 80)

    try:
        # Load configuration
        logger.info("Loading configuration from environment...")
        config = Config.from_env()

        # Override with command line args
        config.experiment.num_steps = args.steps
        config.experiment.max_candidates_per_step = args.candidates
        config.experiment.semaphore_limit = args.semaphore_limit
        config.experiment.embedding_type = args.embedding_type
        config.experiment.local_model_name = args.local_model

        logger.info("✓ Configuration loaded:")
        logger.info(f"  - NEO4J_URI={config.neo4j.uri}")
        logger.info(f"  - NEO4J_USERNAME={config.neo4j.user}")
        logger.info(f"  - NEO4J_DATABASE={config.neo4j.database}")
        logger.info(f"  - OPENAI_MODEL={config.openai.model_name}")
        logger.info(f"  - EMBEDDING_TYPE={config.experiment.embedding_type}")
        if config.experiment.embedding_type == "local":
            logger.info(f"  - LOCAL_MODEL={config.experiment.local_model_name}")

        # Validate required environment variables
        if config.experiment.embedding_type == "openai" and not config.openai.api_key:
            logger.error("✗ OPENAI_API_KEY not set in environment (required for openai embedding type)")
            return 1

        if not config.neo4j.password:
            logger.error("✗ NEO4J_PASSWORD not set in environment")
            return 1

        # Initialize Neo4j driver
        logger.info("Connecting to Neo4j...")
        driver = GraphDriver(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            database=config.neo4j.database,
        )

        try:
            driver.run_query("RETURN 1 as test")
            logger.info("✓ Connected to Neo4j")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            return 1

        # Load articles
        logger.info(f"Loading articles from {args.data}...")
        if not Path(args.data).exists():
            logger.error(f"✗ Data file not found: {args.data}")
            return 1

        articles_df = load_articles(args.data, limit=args.limit)
        logger.info(f"✓ Loaded {len(articles_df)} articles")

        # Prepare articles (parse timestamps, add day column)
        logger.info("Preparing articles...")
        articles_df = prepare_articles(articles_df)
        logger.info("✓ Prepared articles")

        # Filter to date window
        logger.info(
            f"Filtering to {args.time_window_days} days with max {args.articles_per_day or 'unlimited'} articles/day..."
        )
        articles_df = filter_articles_by_date_window(
            articles_df,
            num_days=args.time_window_days,
            articles_per_day=args.articles_per_day,
        )
        logger.info(
            f"✓ Filtered to {len(articles_df)} articles in {args.time_window_days}-day window"
        )

        # Fetch real price data from Stooq
        ticker = articles_df["ticker"].iloc[0] if "ticker" in articles_df.columns else "TSLA"
        price_raw = fetch_stooq_prices(ticker)
        price_df = build_return_labels(price_raw)

        # Filter price data to match article date window
        article_start = articles_df["timestamp"].min().date()
        article_end = articles_df["timestamp"].max().date()
        logger.info(f"Filtering price data to article window: {article_start} to {article_end}")

        # Reset index to make 'day' a column for filtering
        price_df = price_df.reset_index()
        price_df["date"] = pd.to_datetime(price_df["day"]).dt.date
        mask = (price_df["date"] >= article_start) & (price_df["date"] <= article_end)
        price_df = price_df[mask].copy()
        logger.info(f"✓ Filtered price data to {len(price_df)} trading days")

        # Initialize LLM and embedder
        logger.info("Initializing LLM and embedder...")
        llm = OpenAILLM(
            api_key=config.openai.api_key,
            model_name=config.openai.model_name,
        )
        embedder = OpenAIEmbeddings(
            api_key=config.openai.api_key,
            model="text-embedding-3-small",
        )
        logger.info("✓ LLM and embedder initialized")

        # Create orchestrator
        logger.info("Creating orchestrator...")
        orchestrator = Orchestrator(config, driver, llm, embedder)
        logger.info("✓ Orchestrator created")

        # Run experiment
        logger.info("=" * 80)
        logger.info(f"Starting experiment: {args.steps} steps, {args.candidates} candidates/step")
        logger.info("=" * 80)

        results = await orchestrator.run(articles_df, price_df)

        logger.info("=" * 80)
        logger.info("✓ Experiment completed!")
        logger.info("=" * 80)

        # Display results
        logger.info("\nResults by candidate:")
        for candidate_tag in sorted(results.keys()):
            metrics = results[candidate_tag]
            logger.info(
                f"  {candidate_tag}: AUC={metrics.auc:.4f}, F1={metrics.f1:.4f}, "
                f"max_hops_train={metrics.max_hops_train}, max_hops_val={metrics.max_hops_val}"
            )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            tag: {
                "auc": metrics.auc,
                "f1": metrics.f1,
                "max_hops_train": metrics.max_hops_train,
                "max_hops_val": metrics.max_hops_val,
            }
            for tag, metrics in results.items()
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"\n✓ Results saved to {output_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n✗ Pipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        return 1

    finally:
        driver.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# 2026-01-01 15:45:17 [INFO] __main__: ✓ Experiment completed!
# 2026-01-01 15:45:17 [INFO] __main__: ================================================================================
# 2026-01-01 15:45:17 [INFO] __main__: 
# Results by candidate:
# 2026-01-01 15:45:17 [INFO] __main__:   step_1_candidate_0: AUC=0.8889, F1=0.6667, n_train=215, n_val=215
# 2026-01-01 15:45:17 [INFO] __main__:   step_1_candidate_1: AUC=0.6667, F1=0.7500, n_train=250, n_val=250
# 2026-01-01 15:45:17 [INFO] __main__:   step_2_candidate_0: AUC=0.5556, F1=0.5714, n_train=301, n_val=301
# 2026-01-01 15:45:17 [INFO] __main__:   step_2_candidate_1: AUC=0.8889, F1=0.8000, n_train=334, n_val=334
# 2026-01-01 15:45:17 [INFO] __main__:   step_3_candidate_0: AUC=0.8889, F1=0.8571, n_train=388, n_val=388
# 2026-01-01 15:45:17 [INFO] __main__:   step_3_candidate_1: AUC=0.8889, F1=0.8000, n_train=414, n_val=414
# 2026-01-01 15:45:17 [INFO] __main__: 
# ✓ Results saved to results/ontology_experiment_results.json