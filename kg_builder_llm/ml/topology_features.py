"""Local topology feature extraction for temporal graph analysis."""

import logging
from datetime import date as python_date
from typing import Dict

import numpy as np

from kg_builder_llm.core.graph import GraphDriver

logger = logging.getLogger(__name__)


def compute_topology_features(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int = 2,
) -> Dict[str, np.ndarray]:
    """Compute local topology features for entities mentioned in recent time window.

    Single Cypher query returns all 7 raw stats per entity; Python computes derived features.

    Feature vector (8 dims) per entity:
      [in_degree, out_degree, unique_in_neighbors, unique_out_neighbors,
       rel_type_diversity, mention_count_in_window, days_since_first_mention, total_degree]

    Args:
        driver: GraphDriver instance
        eval_date: Evaluation date (YYYY-MM-DD)
        lookback_days: Number of days before eval_date to include

    Returns:
        Dict mapping node_id → np.ndarray of shape (8,)
    """
    logger.info(
        "Computing topology features for eval_date=%s, lookback_days=%d",
        eval_date,
        lookback_days,
    )

    query = """
    MATCH (article:Article)
    WHERE article.date >= date($eval_date) - duration({days: $lookback_days})
      AND article.date <= date($eval_date)
    MATCH (n)-[:MENTIONED_IN]->(article)
    WITH DISTINCT n

    // Mention count in the lookback window
    OPTIONAL MATCH (n)-[:MENTIONED_IN]->(win_art:Article)
    WHERE win_art.date >= date($eval_date) - duration({days: $lookback_days})
      AND win_art.date <= date($eval_date)
    WITH n, count(DISTINCT win_art) AS mention_count

    // First-ever mention date (entity novelty)
    OPTIONAL MATCH (n)-[:MENTIONED_IN]->(any_art:Article)
    WITH n, mention_count, min(any_art.date) AS first_date

    // In-degree (all relationships in graph — temporal isolation handled by allowed_tags)
    OPTIONAL MATCH (n)<-[r_in]-(src)
    WITH n, mention_count, first_date,
         count(DISTINCT r_in) AS in_degree,
         count(DISTINCT src)  AS unique_in_neighbors

    // Out-degree
    OPTIONAL MATCH (n)-[r_out]->(tgt)
    RETURN
        elementId(n)          AS node_id,
        in_degree,
        count(DISTINCT r_out) AS out_degree,
        unique_in_neighbors,
        count(DISTINCT tgt)   AS unique_out_neighbors,
        mention_count,
        first_date
    """

    try:
        results = driver.run_query(
            query,
            parameters={"eval_date": eval_date, "lookback_days": lookback_days},
        )
    except Exception as e:
        logger.error("Error querying topology features: %s", str(e))
        return {}

    if not results:
        logger.warning("No topology features found for eval_date=%s", eval_date)
        return {}

    eval_dt = python_date.fromisoformat(eval_date)
    features_dict = {}

    for row in results:
        node_id = row.get("node_id")
        in_deg = float(row.get("in_degree") or 0)
        out_deg = float(row.get("out_degree") or 0)
        in_neighbors = float(row.get("unique_in_neighbors") or 0)
        out_neighbors = float(row.get("unique_out_neighbors") or 0)
        mention_count = float(row.get("mention_count") or 0)

        rel_diversity = (in_deg + out_deg) / max(in_neighbors + out_neighbors, 1)

        first_date = row.get("first_date")
        if first_date is not None:
            try:
                first_dt = python_date(first_date.year, first_date.month, first_date.day)
                days_since = float((eval_dt - first_dt).days)
            except Exception:
                days_since = 0.0
        else:
            days_since = 0.0

        features_dict[node_id] = np.array(
            [in_deg, out_deg, in_neighbors, out_neighbors,
             rel_diversity, mention_count, days_since, in_deg + out_deg],
            dtype=np.float32,
        )

    logger.info("Computed topology features for %d entities", len(features_dict))
    return features_dict


def aggregate_topology_features(
    features_dict: Dict[str, np.ndarray],
    agg_method: str = "mean",
) -> np.ndarray:
    """
    Aggregate topology features across all entities.
    
    Args:
        features_dict: Dict mapping node_id → feature_vector
        agg_method: Aggregation method (mean, sum, max)
    
    Returns:
        Aggregated feature vector (shape: (8,))
    """
    if not features_dict:
        logger.warning("No features to aggregate, returning zeros")
        return np.zeros(8, dtype=np.float32)
    
    feature_matrix = np.array(list(features_dict.values()), dtype=np.float32)
    
    if agg_method == "mean":
        agg = np.mean(feature_matrix, axis=0)
    elif agg_method == "sum":
        agg = np.sum(feature_matrix, axis=0)
    elif agg_method == "max":
        agg = np.max(feature_matrix, axis=0)
    else:
        logger.warning("Unknown aggregation method: %s, using mean", agg_method)
        agg = np.mean(feature_matrix, axis=0)
    
    logger.debug("Aggregated topology features: %s", agg)
    return agg
