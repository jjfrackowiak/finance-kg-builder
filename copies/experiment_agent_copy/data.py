
# file: finance_kg_experiment/data.py
from __future__ import annotations
import pandas as pd
from typing import Tuple
import requests

from copies.experiment_agent_copy.config import ExperimentConfig


def load_article_sample(config: ExperimentConfig) -> pd.DataFrame:
    """
    Load a subsample of articles for a single ticker, periodized by day.
    Returns a DataFrame with columns at least:
      ['timestamp', 'ticker', 'headline', 'text', 'url', 'source', 'day']
    """
    df = pd.read_csv(config.csv_path)
    df = df[df["ticker"] == config.target_ticker].copy()

    # Ensure timestamp and day
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Limit to n_days and at most max_articles_per_day per day
    df = df.sort_values("timestamp")
    unique_days = sorted(df["day"].unique().tolist())[:config.n_days]
    df = df[df["day"].isin(unique_days)].copy()

    df = (
        df.sort_values("timestamp")
          .groupby("day")
          .head(config.max_articles_per_day)
          .reset_index(drop=True)
    )
    return df


def fetch_stooq_prices(symbol: str = "TSLA") -> pd.DataFrame:
    """
    Fetch daily OHLC data from Stooq.
    Returns DataFrame with columns: ['day', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    df = pd.read_csv(url)

    df = df.rename(columns={
        "Date": "day",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })
    df = df.dropna()
    return df


def build_return_labels(config: ExperimentConfig) -> pd.DataFrame:
    """
    Compute next-day returns and a binary direction label for TSLA.
    Returns df_labels indexed by 'day' with:
        ['return_next_day', 'direction']
    """
    price_df = fetch_stooq_prices(config.target_ticker)

    price_df["next_close"] = price_df["Close"].shift(-1)
    price_df["return_next_day"] = (
        price_df["next_close"] - price_df["Close"]
    ) / price_df["Close"]

    price_df = price_df.dropna(subset=["return_next_day"])

    # direction label: 1 = up, 0 = down or flat
    price_df["direction"] = (price_df["return_next_day"] > 0.0).astype(int)

    df_labels = price_df[["day", "return_next_day", "direction"]].copy()
    df_labels.set_index("day", inplace=True)
    return df_labels
