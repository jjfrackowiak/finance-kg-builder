"""Tests for filtering articles to the target ticker.

The 2026-08 TSLA sweep ran without this filter: the 4-articles/day cap took whatever
led the file, so 794 selected articles were 712 MSFT, 51 NVDA and 31 TSLA while 2,607
TSLA articles in the same window went unselected. These tests pin the two properties
that failure needed — that the filter exists, and that it runs before the cap.
"""

from datetime import date

import pandas as pd
import pytest

from kg_builder_llm.core.data import filter_articles_by_date_window


def _articles() -> pd.DataFrame:
    """Three tickers a day for 5 days, MSFT always first — the real file's shape."""
    rows = []
    for day in range(5):
        for tk in ("MSFT", "MSFT", "NVDA", "TSLA"):
            rows.append({"timestamp": pd.Timestamp(f"2022-05-0{day + 1} 12:00"), "ticker": tk,
                         "text": f"{tk} article"})
    return pd.DataFrame(rows)


class TestTickerFilter:
    def test_keeps_only_the_requested_ticker(self):
        out = filter_articles_by_date_window(
            _articles(), num_days=5, start_date=date(2022, 5, 1), ticker="TSLA"
        )
        assert set(out.ticker) == {"TSLA"}
        assert len(out) == 5  # one per day

    def test_filter_runs_before_the_per_day_cap(self):
        """The bug: cap first and the quota fills with MSFT, leaving ~no TSLA."""
        out = filter_articles_by_date_window(
            _articles(), num_days=5, articles_per_day=2, start_date=date(2022, 5, 1),
            ticker="TSLA",
        )
        # Only one TSLA article exists per day, so the cap of 2 is not binding.
        # Had the cap been applied first it would have taken MSFT, MSFT and left nothing.
        assert set(out.ticker) == {"TSLA"}
        assert len(out) == 5

    def test_no_ticker_keeps_every_ticker(self):
        out = filter_articles_by_date_window(
            _articles(), num_days=5, start_date=date(2022, 5, 1)
        )
        assert set(out.ticker) == {"MSFT", "NVDA", "TSLA"}

    def test_unknown_ticker_fails_loudly(self):
        """Silently returning nothing is how this went unnoticed for a whole sweep."""
        with pytest.raises(ValueError, match="No articles for ticker"):
            filter_articles_by_date_window(
                _articles(), num_days=5, start_date=date(2022, 5, 1), ticker="AAPL"
            )

    def test_missing_ticker_column_does_not_crash(self):
        df = _articles().drop(columns=["ticker"])
        out = filter_articles_by_date_window(
            df, num_days=5, start_date=date(2022, 5, 1), ticker="TSLA"
        )
        assert len(out) == 20

    def test_cap_applies_within_the_filtered_ticker(self):
        rows = [{"timestamp": pd.Timestamp("2022-05-01 12:00"), "ticker": "TSLA", "text": f"a{i}"}
                for i in range(10)]
        rows += [{"timestamp": pd.Timestamp("2022-05-01 12:00"), "ticker": "MSFT", "text": "m"}]
        out = filter_articles_by_date_window(
            pd.DataFrame(rows), num_days=1, articles_per_day=4,
            start_date=date(2022, 5, 1), ticker="TSLA",
        )
        assert len(out) == 4
        assert set(out.ticker) == {"TSLA"}
