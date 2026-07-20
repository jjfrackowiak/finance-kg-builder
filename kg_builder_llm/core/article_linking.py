"""Link articles to their publication days in Neo4j."""

import logging

import pandas as pd

from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.core.ids import article_text_id

logger = logging.getLogger(__name__)


def create_and_link_article_days(
    driver: GraphDriver,
    articles_df: pd.DataFrame,
) -> None:
    """Create Day nodes for article publication dates and link articles to them.

    This ensures articles are connected to their actual publication day,
    enabling proper temporal train/val splits.

    Args:
        driver: Graph driver
        articles_df: DataFrame with 'text', 'headline', and 'day' columns
    """
    if "day" not in articles_df.columns:
        raise ValueError("articles_df must have 'day' column")

    if "text" not in articles_df.columns:
        raise ValueError("articles_df must have 'text' column")

    # Get unique days
    unique_days = articles_df["day"].unique()

    logger.info("Creating Day nodes for %d publication dates", len(unique_days))

    # Create Day nodes for each unique day
    day_payload = [{"date": str(day)} for day in unique_days]

    query = """
    UNWIND $rows AS row
    MERGE (d:Day {date: row.date})
    """

    try:
        driver.run_query(query, parameters={"rows": day_payload})
        logger.info("Created/merged %d Day nodes", len(day_payload))
    except Exception as e:
        logger.error("Failed to create Day nodes: %s", str(e))
        raise

    # Now link articles to their publication days
    # This is trickier because we need to identify articles by their content/headline
    # We'll create Article nodes and link them

    logger.info("Linking %d articles to their publication days", len(articles_df))

    article_link_payload = []
    for idx, row in articles_df.iterrows():
        day = str(row["day"])
        headline = str(row.get("headline", ""))[:1024]
        text_hash = article_text_id(str(row.get("text", "")))

        article_link_payload.append(
            {
                "day": day,
                "headline": headline,
                "text_id": text_hash,
            }
        )

    # Link articles to days
    link_query = """
    UNWIND $rows AS row
    MATCH (d:Day {date: row.day})
    MERGE (a:Article {id: row.text_id, headline: row.headline})
    SET a.date = date(row.day)
    MERGE (a)-[r:PUBLISHED_ON]->(d)
    """

    try:
        driver.run_query(link_query, parameters={"rows": article_link_payload})
        logger.info("Linked %d articles to their publication days", len(article_link_payload))
    except Exception as e:
        logger.error("Failed to link articles to days: %s", str(e))
        raise
