"""Data loading and preprocessing."""

import io
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_articles(
    csv_path: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load articles from a local path or s3:// URI."""
    logger.info("Loading articles from %s", csv_path)

    if csv_path.startswith("s3://"):
        import boto3
        parts = csv_path[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    else:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    logger.info("Loaded %d articles", len(df))
    return df


def filter_articles_by_date_window(
    df: pd.DataFrame,
    num_days: int,
    articles_per_day: Optional[int] = None,
    start_date: Optional[date] = None,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Filter articles to a consecutive date window.

    Args:
        df: DataFrame with 'timestamp' column (must be datetime)
        num_days: Number of consecutive days to use
        articles_per_day: Max articles per day (None = no limit)
        start_date: Window start date (default: earliest date in dataset)
        ticker: Keep only articles for this ticker (None = every ticker in the file).
            Applied BEFORE the per-day cap: capping first would fill the quota with
            whichever tickers happen to lead the file and leave almost nothing after
            filtering. The 2026-08 TSLA sweep hit exactly this — with no filter the
            794 selected articles were 712 MSFT, 51 NVDA and 31 TSLA, while 2,607
            TSLA articles in the same window went unselected.

    Returns:
        Filtered DataFrame spanning num_days
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Use provided start_date or fall back to earliest date in dataset
    if start_date is None:
        start_date = df["timestamp"].min().date()
    end_date = start_date + timedelta(days=num_days)

    # Filter to date window
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date < end_date)
    df_window = df[mask].copy()

    logger.info(
        "Filtered to date window: %s to %s (%d articles)", start_date, end_date, len(df_window)
    )

    # Keep only the target ticker, before the per-day cap (see the `ticker` docstring).
    if ticker and "ticker" in df_window.columns:
        before = len(df_window)
        df_window = df_window[df_window["ticker"] == ticker].copy()
        logger.info(
            "Filtered to ticker %s: %d of %d articles kept", ticker, len(df_window), before
        )
        if df_window.empty:
            raise ValueError(
                f"No articles for ticker {ticker!r} in {start_date}..{end_date}. "
                f"Tickers present in the window: "
                f"{sorted(df[mask]['ticker'].dropna().unique().tolist())[:10]}"
            )
    elif ticker:
        logger.warning(
            "ticker=%s requested but the data has no 'ticker' column — no filter applied", ticker
        )

    # Further limit articles per day if requested
    if articles_per_day:
        df_window["day"] = df_window["timestamp"].dt.strftime("%Y-%m-%d")
        df_window = df_window.groupby("day").head(articles_per_day).reset_index(drop=True)
        logger.info(
            "Limited to %d articles/day (%d articles remaining)", articles_per_day, len(df_window)
        )

    return df_window


def prepare_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare articles for processing.

    Args:
        df: Raw articles DataFrame

    Returns:
        Prepared DataFrame with required columns
    """
    logger.debug("Preparing %d articles", len(df))

    # Ensure required columns exist
    required_columns = ["text", "timestamp", "headline"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Add day column
    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    return df
