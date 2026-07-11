"""Main CLI entry point."""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import yfinance as yf

import pandas as pd
from dotenv import load_dotenv
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from kg_builder_llm.config import Config
from kg_builder_llm.core.data import filter_articles_by_date_window, load_articles, prepare_articles
from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.logger import get_logger, setup_logging
from kg_builder_llm.pipeline.orchestrator import Orchestrator

logger = get_logger(__name__)


# ============================================================================
# Environment Setup
# ============================================================================

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)


# ============================================================================
# Helper Functions - Data
# ============================================================================

def fetch_stooq_prices(symbol: str = "TSLA") -> pd.DataFrame:
    """Fetch daily OHLC data via yfinance.

    Args:
        symbol: Stock ticker symbol (default: TSLA)

    Returns:
        DataFrame with columns: ['day', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    logger.info(f"Fetching price data for {symbol} via yfinance...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {symbol}")
        df = df.reset_index()
        # yfinance returns 'Date' as a tz-aware datetime column
        df["day"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
        df = df[["day", "Open", "High", "Low", "Close", "Volume"]].dropna()
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


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KG Builder - Knowledge Graph Evolution with LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kg_builder_llm.main --data data/articles.csv --steps 4 --lookback-days 3
  python -m kg_builder_llm.main --data data/articles.csv --train-ratio 0.8
  python -m kg_builder_llm.main --help
        """,
    )

    # === Data Arguments ===
    data_group = parser.add_argument_group('data arguments', 'Input data configuration')
    data_group.add_argument(
        "--data",
        type=str,
        default=os.environ.get("DATA_URI", "data/fnspid_sample_nasdaq_long_text.csv"),
        help="Path to articles CSV file or s3:// URI (default: DATA_URI env var)",
    )
    data_group.add_argument(
        "--time-window-days",
        type=int,
        default=150,
        help="Number of sequential days to use for training (default: 150)",
    )
    data_group.add_argument(
        "--articles-per-day",
        type=int,
        default=3,
        help="Max articles per day (default: all articles in time window)",
    )
    data_group.add_argument(
        "--day-start",
        type=str,
        default=None,
        help="Window start date in YYYY-MM-DD format (default: earliest date in dataset)",
    )
    data_group.add_argument(
        "--day-end",
        type=str,
        default=None,
        help="Window end date in YYYY-MM-DD format (exclusive). Overrides --time-window-days.",
    )
    data_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(DEPRECATED) Use --time-window-days instead. Limit number of articles to process",
    )

    # === Experiment Arguments ===
    exp_group = parser.add_argument_group('experiment arguments', 'Ontology evolution configuration')
    exp_group.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of incremental evolution steps after base structure (0=base only, default: 3)",
    )
    exp_group.add_argument(
        "--candidates",
        type=int,
        default=2,
        help="Number of candidates per step (default: 2)",
    )

    exp_group.add_argument(
        "--semaphore-limit",
        type=int,
        default=50,
        help="Max concurrent article processing tasks (default: 50)",
    )
    exp_group.add_argument(
        "--evolution-prompt",
        type=str,
        default="default",
        help="Path to custom ontology evolution prompt template file or 'default' (default: default)",
    )

    # === Embedding Arguments ===
    embed_group = parser.add_argument_group('embedding arguments', 'Text embedding configuration')
    embed_group.add_argument(
        "--embedding-type",
        type=str,
        default=os.environ.get("EMBEDDING_TYPE", "local"),
        choices=["local", "openai", "remote"],
        help="Embedding type: 'local', 'openai', or 'remote' (text-embeddings-inference sidecar) (default: EMBEDDING_TYPE env var)",
    )
    embed_group.add_argument(
        "--local-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Local embedding model name for sentence-transformers (default: all-MiniLM-L6-v2)",
    )

    # === Feature Engineering Hyperparameters ===
    feat_group = parser.add_argument_group('feature engineering', 'Feature extraction hyperparameters')
    feat_group.add_argument(
        "--lookback-days",
        type=int,
        default=2,
        help="Number of days to look back for article/feature extraction (default: 2)",
    )
    feat_group.add_argument(
        "--min-chain-hops",
        type=int,
        default=5,
        help="Minimum path length for relationship chain extraction (default: 5)",
    )
    feat_group.add_argument(
        "--max-chain-hops",
        type=int,
        default=5,
        help="Maximum path length for relationship chain extraction (default: 5)",
    )
    feat_group.add_argument(
        "--path-uniqueness",
        type=str,
        default="NODE_PATH",
        choices=["NODE_PATH", "NODE_GLOBAL", "RELATIONSHIP_PATH", "RELATIONSHIP_GLOBAL", "NODE_LEVEL", "NONE"],
        help="APOC path uniqueness mode for chain extraction (default: NODE_PATH)",
    )
    feat_group.add_argument(
        "--feature-mode",
        type=str,
        default="path",
        choices=["path", "subgraph", "hybrid"],
        help="Day-level feature representation: path, subgraph, or hybrid (default: path)",
    )
    feat_group.add_argument(
        "--max-metapath-hops",
        type=int,
        default=2,
        choices=[2, 3],
        help="Maximum hop count for temporal subgraph metapath features (default: 2)",
    )
    # === Model Training Hyperparameters ===
    model_group = parser.add_argument_group('model training', 'Model training hyperparameters')
    model_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training in train/val split (default: 0.7 = 70/30 split)",
    )

    # === Output Arguments ===
    output_group = parser.add_argument_group('output arguments', 'Results and logging')
    output_group.add_argument(
        "--output",
        type=str,
        default="results/ontology_experiment_results.json",
        help="Output path for results (default: results/ontology_experiment_results.json)",
    )
    output_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


# ============================================================================
# Helper Functions - Configuration
# ============================================================================

def setup_config(args: argparse.Namespace) -> Config:
    """Load and configure the application configuration.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured Config object
    """
    logger.info("Loading configuration from environment...")
    config = Config.from_env()

    # Override with command line args
    config.experiment.num_steps = args.steps
    config.experiment.max_candidates_per_step = args.candidates
    config.experiment.semaphore_limit = args.semaphore_limit
    config.experiment.evolution_prompt_template = args.evolution_prompt
    config.experiment.feature.embedding_type = args.embedding_type
    config.experiment.feature.local_model_name = args.local_model
    config.experiment.feature.lookback_days = args.lookback_days
    config.experiment.feature.min_chain_hops = args.min_chain_hops
    config.experiment.feature.max_chain_hops = args.max_chain_hops
    config.experiment.feature.path_uniqueness = args.path_uniqueness
    config.experiment.feature.feature_mode = args.feature_mode
    config.experiment.feature.max_metapath_hops = args.max_metapath_hops
    config.experiment.feature.train_ratio = args.train_ratio

    logger.info("✓ Configuration loaded:")
    logger.info(f"  - NEO4J_URI={config.neo4j.uri}")
    logger.info(f"  - NEO4J_USERNAME={config.neo4j.user}")
    logger.info(f"  - NEO4J_DATABASE={config.neo4j.database}")
    logger.info(f"  - OPENAI_MODEL={config.openai.model_name}")
    logger.info(f"  - EMBEDDING_TYPE={config.experiment.feature.embedding_type}")
    if config.experiment.feature.embedding_type == "local":
        logger.info(f"  - LOCAL_MODEL={config.experiment.feature.local_model_name}")
    logger.info(f"  - LOOKBACK_DAYS={config.experiment.feature.lookback_days}")
    logger.info(f"  - CHAIN_HOPS={config.experiment.feature.min_chain_hops}-{config.experiment.feature.max_chain_hops}")
    logger.info(f"  - PATH_UNIQUENESS={config.experiment.feature.path_uniqueness}")
    logger.info(f"  - FEATURE_MODE={config.experiment.feature.feature_mode}")
    logger.info(f"  - MAX_METAPATH_HOPS={config.experiment.feature.max_metapath_hops}")
    logger.info(f"  - TRAIN_RATIO={config.experiment.feature.train_ratio:.2f}")
    logger.info(f"  - EVOLUTION_PROMPT={config.experiment.evolution_prompt_template}")
    
    return config


def validate_config(config: Config) -> None:
    """Validate configuration has required values.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If required configuration is missing
    """
    if config.experiment.feature.embedding_type == "openai" and not config.openai.api_key:
        raise ValueError("OPENAI_API_KEY not set in environment (required for openai embedding type)")
    
    if not config.neo4j.password:
        raise ValueError("NEO4J_PASSWORD not set in environment")


def initialize_driver(config: Config) -> GraphDriver:
    """Initialize and test Neo4j driver connection.
    
    Args:
        config: Configuration with Neo4j settings
        
    Returns:
        Connected GraphDriver instance
        
    Raises:
        ConnectionError: If connection to Neo4j fails
    """
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
        return driver
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Neo4j: {e}")


def _parse_day_start(day_start_str: str | None) -> "datetime.date | None":
    """Parse --day-start string to date, or return None (use dataset earliest)."""
    if day_start_str is None:
        return None
    try:
        return datetime.date.fromisoformat(day_start_str)
    except ValueError:
        raise ValueError(f"--day-start must be YYYY-MM-DD, got: {day_start_str!r}")


def _resolve_time_window(args: argparse.Namespace) -> int:
    """Compute effective time_window_days from --day-start / --day-end / --time-window-days.

    Priority: if --day-end is given, derive window from (day_end - day_start).
    Falls back to --time-window-days otherwise.
    """
    if args.day_end is None:
        return args.time_window_days
    end = datetime.date.fromisoformat(args.day_end)
    start = (
        datetime.date.fromisoformat(args.day_start)
        if args.day_start
        else None
    )
    if start is None:
        raise ValueError("--day-end requires --day-start to be set")
    delta = (end - start).days
    if delta <= 0:
        raise ValueError(f"--day-end ({args.day_end}) must be after --day-start ({args.day_start})")
    return delta


def load_and_prepare_data(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare articles and price data.
    
    Args:
        args: Command line arguments with data paths
        
    Returns:
        Tuple of (articles_df, price_df)
        
    Raises:
        FileNotFoundError: If data file not found
    """
    # Load articles
    logger.info(f"Loading articles from {args.data}...")
    if not args.data.startswith("s3://") and not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    articles_df = load_articles(args.data, limit=args.limit)
    logger.info(f"✓ Loaded {len(articles_df)} articles")

    # Prepare articles (parse timestamps, add day column)
    logger.info("Preparing articles...")
    articles_df = prepare_articles(articles_df)
    logger.info("✓ Prepared articles")

    # Filter to date window
    effective_days = _resolve_time_window(args)
    logger.info(
        f"Filtering to {effective_days} days with max {args.articles_per_day or 'unlimited'} articles/day..."
    )
    articles_df = filter_articles_by_date_window(
        articles_df,
        num_days=effective_days,
        articles_per_day=args.articles_per_day,
        start_date=_parse_day_start(args.day_start),
    )
    logger.info(
        f"✓ Filtered to {len(articles_df)} articles in {effective_days}-day window"
    )

    # Fetch price data
    ticker = articles_df["ticker"].iloc[0] if "ticker" in articles_df.columns else "TSLA"
    price_raw = fetch_stooq_prices(ticker)
    price_df = build_return_labels(price_raw)

    # Filter price data to match article date window
    article_start = articles_df["timestamp"].min().date()
    article_end = articles_df["timestamp"].max().date()
    logger.info(f"Filtering price data to article window: {article_start} to {article_end}")

    price_df = price_df.reset_index()
    price_df["date"] = pd.to_datetime(price_df["day"]).dt.date
    mask = (price_df["date"] >= article_start) & (price_df["date"] <= article_end)
    price_df = price_df[mask].copy()
    logger.info(f"✓ Filtered price data to {len(price_df)} trading days")
    
    return articles_df, price_df


def save_results(results: dict, output_path: Path) -> None:
    """Save experiment results to JSON file.
    
    Args:
        results: Dictionary of results by candidate tag
        output_path: Path to save results
    """
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

    logger.info(f"✓ Results saved to {output_path}")


# ============================================================================
# MLflow Helpers
# ============================================================================

def _log_data_stats(
    articles_df: pd.DataFrame,
    price_df: pd.DataFrame,
    train_ratio: float,
) -> None:
    """Log runtime data statistics to the active MLflow run."""
    n_articles = len(articles_df)
    n_days = articles_df["day"].nunique() if "day" in articles_df.columns else 0
    date_start = str(articles_df["timestamp"].min().date())
    date_end = str(articles_df["timestamp"].max().date())

    n_price_days = len(price_df)
    n_train_days = round(n_price_days * train_ratio)
    n_val_days = n_price_days - n_train_days
    class_balance = float(price_df["direction"].mean()) if "direction" in price_df.columns else 0.0

    mlflow.log_params({
        "n_articles_actual": n_articles,
        "n_days_actual": n_days,
        "date_start_actual": date_start,
        "date_end_actual": date_end,
        "n_price_days": n_price_days,
        "n_train_days": n_train_days,
        "n_val_days": n_val_days,
    })
    mlflow.log_metric("class_balance", class_balance)
    logger.info(
        "Data: %d articles, %d days (%s → %s), %d price days, train=%d val=%d, up_days=%.1f%%",
        n_articles, n_days, date_start, date_end, n_price_days, n_train_days, n_val_days,
        class_balance * 100,
    )


def _setup_mlflow(config: Config) -> None:
    """Configure MLflow tracking URI and experiment."""
    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        logger.info("MLflow tracking URI: %s", config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)


def _make_run_name(args: argparse.Namespace, config: Config) -> str:
    """Build a human-readable run name from key CLI args."""
    ticker = config.experiment.target_ticker
    day_start = args.day_start or f"w{_resolve_time_window(args)}d"
    prompt = args.evolution_prompt
    if prompt and prompt != "default":
        from pathlib import Path as _Path
        prompt_label = _Path(prompt).stem.replace("_prompt_template", "")
    else:
        prompt_label = "default"
    return (
        f"{ticker}-steps{args.steps}-cands{args.candidates}"
        f"-{args.feature_mode}-{prompt_label}-{day_start}"
    )


def _log_params(args: argparse.Namespace, config: Config) -> None:
    """Log all experiment hyperparameters to the active MLflow run."""
    mlflow.log_params({
        # --- data ---
        "data_file": args.data,
        "day_start": args.day_start or "earliest",
        "day_end": args.day_end or "derived",
        "time_window_days": _resolve_time_window(args),
        "articles_per_day": args.articles_per_day,
        # --- experiment ---
        "ticker": config.experiment.target_ticker,
        "steps": args.steps,
        "candidates": args.candidates,
        "semaphore_limit": args.semaphore_limit,
        "evolution_prompt": args.evolution_prompt,
        # --- embedding ---
        "embedding_type": args.embedding_type,
        "local_model": args.local_model,
        # --- feature engineering ---
        "lookback_days": args.lookback_days,
        "feature_mode": args.feature_mode,
        "min_chain_hops": args.min_chain_hops,
        "max_chain_hops": args.max_chain_hops,
        "path_uniqueness": args.path_uniqueness,
        "max_metapath_hops": args.max_metapath_hops,
        # --- model training ---
        "train_ratio": args.train_ratio,
        # --- output ---
        "output": args.output,
        "log_level": args.log_level,
        # --- infrastructure (no credentials) ---
        "llm_model": config.openai.model_name,
        "llm_base_url": config.openai.base_url or "openai",
        "neo4j_uri": config.neo4j.uri,
        "neo4j_user": config.neo4j.user,
        "neo4j_database": config.neo4j.database,
    })


# ============================================================================
# Main Entry Point
# ============================================================================


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

    driver = None
    try:
        # 1. Load and validate configuration
        config = setup_config(args)
        validate_config(config)

        # 2. Initialize Neo4j driver
        driver = initialize_driver(config)

        # 3. Load and prepare data
        articles_df, price_df = load_and_prepare_data(args)

        # 4. Initialize LLM and embedder
        logger.info("Initializing LLM and embedder...")
        llm_kwargs = {"api_key": config.openai.api_key, "model_name": config.openai.model_name}
        if config.openai.base_url:
            llm_kwargs["base_url"] = config.openai.base_url
            logger.info(f"  - LLM_BASE_URL={config.openai.base_url}")
        llm = OpenAILLM(**llm_kwargs)
        embedder = OpenAIEmbeddings(
            api_key=config.openai.api_key,
            model="text-embedding-3-small",
        )
        logger.info("✓ LLM and embedder initialized")

        # 5. Setup MLflow
        _setup_mlflow(config)

        # 6. Create orchestrator and run experiment
        logger.info("Creating orchestrator...")
        orchestrator = Orchestrator(config, driver, llm, embedder)
        logger.info("✓ Orchestrator created")

        logger.info("=" * 80)
        logger.info(f"Starting experiment: {args.steps} steps, {args.candidates} candidates/step")
        logger.info("=" * 80)

        with mlflow.start_run(run_name=_make_run_name(args, config)) as run:
            _log_params(args, config)
            _log_data_stats(articles_df, price_df, config.experiment.feature.train_ratio)
            results = await orchestrator.run(articles_df, price_df)
            logger.info("MLflow run: %s", run.info.run_id)

        logger.info("=" * 80)
        logger.info("✓ Experiment completed!")
        logger.info("=" * 80)

        # 7. Display and save results
        logger.info("\nResults by candidate:")
        for candidate_tag in sorted(results.keys()):
            metrics = results[candidate_tag]
            logger.info(
                f"  {candidate_tag}: AUC={metrics.auc:.4f}, F1={metrics.f1:.4f}, "
                f"max_hops_train={metrics.max_hops_train}, max_hops_val={metrics.max_hops_val}"
            )

        save_results(results, Path(args.output))

        return 0

    except KeyboardInterrupt:
        logger.warning("\n✗ Pipeline interrupted by user")
        return 130

    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        return 1

    except ConnectionError as e:
        logger.error(f"✗ Connection error: {e}")
        return 1

    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        return 1

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# 2026-02-22 22:40:44 [INFO] __main__: ================================================================================
# 2026-02-22 22:40:44 [INFO] __main__: 
# Results by candidate:
# 2026-02-22 22:40:44 [INFO] __main__:   step_1_candidate_0: AUC=0.6019, F1=0.5714, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__:   step_1_candidate_1: AUC=0.4815, F1=0.5714, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__:   step_2_candidate_0: AUC=0.5648, F1=0.5714, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__:   step_2_candidate_1: AUC=0.6204, F1=0.4286, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__:   step_3_candidate_0: AUC=0.5556, F1=0.5714, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__:   step_3_candidate_1: AUC=0.6852, F1=0.5714, max_hops_train=6, max_hops_val=6
# 2026-02-22 22:40:44 [INFO] __main__: ✓ Results saved to results/experiment_100days_unique_paths.json
