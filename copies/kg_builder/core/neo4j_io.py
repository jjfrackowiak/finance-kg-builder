"""Neo4j I/O operations."""

import logging

import pandas as pd

from copies.kg_builder.config import Neo4jConfig
from copies.kg_builder.core.graph import GraphDriver

logger = logging.getLogger(__name__)


def write_price_labels_to_days(
    driver: GraphDriver,
    neo4j_cfg: Neo4jConfig,
    df_labels: pd.DataFrame,
) -> None:
    """Write price labels to Day nodes in Neo4j.

    Args:
        driver: Graph driver
        neo4j_cfg: Neo4j configuration
        df_labels: DataFrame with index=day and columns=['return_next_day', 'direction']
    """
    logger.info("Writing price labels for %d days", len(df_labels))

    # Convert to a list of dicts for UNWIND
    payload = [
        {
            "day": str(day),  # Ensure it's a string
            "ret": float(row["return_next_day"]),
            "direction": int(row["direction"]),
        }
        for day, row in df_labels.iterrows()
    ]

    # Log first few entries for debugging
    if payload:
        logger.debug("First price label entry: %s", payload[0])

    query = """
    UNWIND $rows AS row
    MATCH (d:Day {date: row.day})
    SET d.return_next_day = row.ret,
        d.direction = row.direction
    RETURN COUNT(d) as updated_count
    """

    try:
        results = driver.run_query(query, parameters={"rows": payload})
        if results:
            updated_count = results[0].get("updated_count", 0)
            logger.info(
                "Price labels written for %d days (updated %d Day nodes)",
                len(payload),
                updated_count,
            )
            if updated_count == 0 and len(payload) > 0:
                logger.warning("No Day nodes were updated. Check if date format matches.")
        else:
            logger.warning("Query returned no results")
    except Exception as e:
        logger.error("Failed to write price labels: %s", str(e))
        raise
