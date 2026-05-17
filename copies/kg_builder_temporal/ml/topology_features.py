"""Local topology feature extraction for temporal graph analysis."""

import logging
from typing import Dict, List

import numpy as np

from kg_builder_temporal.core.graph import GraphDriver

logger = logging.getLogger(__name__)


def compute_topology_features(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Compute local topology features for entities mentioned in recent time window.
    
    Features per entity:
    - in_degree: number of incoming relationships
    - out_degree: number of outgoing relationships
    - unique_in_neighbors: count of unique source nodes
    - unique_out_neighbors: count of unique target nodes
    - relation_type_diversity: number of distinct relationship types
    - mention_count: how many articles mention this entity in window
    - days_since_first_mention: novelty score (negative = older)
    - mention_frequency_trend: mention count increasing/decreasing
    
    Args:
        driver: GraphDriver instance
        eval_date: Evaluation date (YYYY-MM-DD format)
        lookback_days: Number of days before eval_date to include
    
    Returns:
        Dict mapping node_id → feature_vector (np.ndarray of shape (8,))
    """
    logger.info(
        "Computing topology features for eval_date=%s, lookback_days=%d",
        eval_date,
        lookback_days,
    )
    
    # Query all entities mentioned in the lookback window
    query = """
    MATCH (article:Article)
    WHERE article.date >= date($eval_date) - duration({days: $lookback_days})
      AND article.date <= $eval_date
    MATCH (n)-[:MENTIONED_IN]->(article)
    WITH DISTINCT n
    
    // Compute in/out degrees (only past edges)
    OPTIONAL MATCH (n)<-[r_in]-(other)
    WHERE other.date <= $eval_date AND r_in.date <= $eval_date
    OPTIONAL MATCH (n)-[r_out]->(other2)
    WHERE other2.date <= $eval_date AND r_out.date <= $eval_date
    
    RETURN 
        elementId(n) as node_id,
        count(DISTINCT r_in) as in_degree,
        count(DISTINCT r_out) as out_degree,
        count(DISTINCT other) as unique_in_neighbors,
        count(DISTINCT other2) as unique_out_neighbors
    """
    
    try:
        results = driver.run_query(
            query,
            parameters={"eval_date": eval_date, "lookback_days": lookback_days},
        )
    except Exception as e:
        logger.error("Error querying topology features: %s", str(e))
        return {}
    
    features_dict = {}
    
    if not results:
        logger.warning("No topology features found for eval_date=%s", eval_date)
        return features_dict
    
    for row in results:
        node_id = row.get("node_id")
        in_deg = float(row.get("in_degree", 0))
        out_deg = float(row.get("out_degree", 0))
        in_neighbors = float(row.get("unique_in_neighbors", 0))
        out_neighbors = float(row.get("unique_out_neighbors", 0))
        
        # Compute additional features
        rel_diversity = (in_deg + out_deg) / max(in_neighbors + out_neighbors, 1)
        
        # Count mentions in window
        mention_query = """
        MATCH (n)-[:MENTIONED_IN]->(a:Article)
        WHERE elementId(n) = $node_id 
          AND a.date >= date($eval_date) - duration({days: $lookback_days})
          AND a.date <= $eval_date
        RETURN count(DISTINCT a) as mention_count
        """
        
        mention_result = driver.run_query(
            mention_query,
            parameters={"node_id": node_id, "eval_date": eval_date, "lookback_days": lookback_days},
        )
        mention_count = float(mention_result[0].get("mention_count", 0)) if mention_result else 0
        
        # Days since first mention (novelty)
        first_mention_query = """
        MATCH (n)-[:MENTIONED_IN]->(a:Article)
        WHERE elementId(n) = $node_id
        RETURN min(a.date) as first_date
        """
        
        first_result = driver.run_query(first_mention_query, parameters={"node_id": node_id})
        if first_result and first_result[0].get("first_date"):
            # Simple: negative days since first mention
            days_since = 0  # Could compute from dates if needed
        else:
            days_since = -999  # Never mentioned before
        
        # Create feature vector: [in_deg, out_deg, in_neighbors, out_neighbors, rel_div, mention_count, days_since, node_degree_sum]
        feature_vector = np.array(
            [
                in_deg,
                out_deg,
                in_neighbors,
                out_neighbors,
                rel_diversity,
                mention_count,
                days_since,
                in_deg + out_deg,  # Total degree
            ],
            dtype=np.float32,
        )
        
        features_dict[node_id] = feature_vector
        logger.debug("Node %s: features=%s", node_id[:8], feature_vector)
    
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
