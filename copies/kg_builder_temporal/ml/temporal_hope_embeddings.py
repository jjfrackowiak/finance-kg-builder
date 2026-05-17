"""Rolling temporal HOPE embeddings for time-aware graph analysis.

Strategy:
---------
For each evaluation date t, build a temporal snapshot of the graph using only
edges formed in the recent past [t-4, t], then compute HOPE embeddings on that
snapshot. This captures recent structural patterns while respecting temporal
constraints (no future data leakage).

Key Features:
- Uses n.first_seen attribute to filter entities by when they first appeared
- Infers edge temporality through articles that mention both entities
- Aggregates entity embeddings per day (mean pooling over mentioned entities)
- One embedding vector per day for downstream ML classification

Example:
    For day t = 2024-01-15, window = 5 days:
    1. Get all entities where n.first_seen <= 2024-01-15
    2. Get edges between entities via articles published in [2024-01-11, 2024-01-15]
    3. Compute HOPE embeddings on this 5-day graph snapshot
    4. Aggregate embeddings for entities mentioned on 2024-01-15
    5. Result: One 128-dim vector representing day 2024-01-15
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from kg_builder_temporal.core.graph import GraphDriver
from kg_builder_temporal.ml.embeddings import compute_hope_embeddings

logger = logging.getLogger(__name__)


def fetch_edges_in_temporal_window(
    driver: GraphDriver,
    eval_date: str,
    window_days: int = 5,
    allowed_tags: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Fetch edges for temporal snapshot [eval_date - (window_days-1), eval_date].
    
    Filtering strategy:
    - Nodes: n.first_seen <= eval_date (entity existed by eval_date)
    - Edges: formed through articles with a.date IN [eval_date - window_days + 1, eval_date]
    - Tags: filter by allowed_tags if provided (base + winners + current candidate)
    
    The edge inference works by finding relationships where both entities
    are mentioned in the same article within the temporal window.
    
    Args:
        driver: GraphDriver instance
        eval_date: Evaluation date (YYYY-MM-DD format)
        window_days: Rolling window size (default 5 = [t-4, t])
        allowed_tags: List of tags to filter relationships (optional)
    
    Returns:
        List of (src_id, dst_id) tuples representing directed edges
    """
    logger.debug(
        "Fetching edges for eval_date=%s, window_days=%d, allowed_tags=%s",
        eval_date,
        window_days,
        allowed_tags,
    )
    
    # Query to find edges between entities via co-mentions in articles
    # within the temporal window
    query = """
    // Find articles in temporal window
    MATCH (a:Article)
    WHERE a.date >= date($eval_date) - duration({days: $window_minus_1})
      AND a.date <= date($eval_date)
    
    // Get entities mentioned in these articles
    MATCH (a)-[:MENTIONS]->(s)
    WHERE s.first_seen IS NOT NULL 
      AND s.first_seen <= date($eval_date)
    
    // Find relationships between entities
    MATCH (s)-[r]->(t)
    WHERE t.first_seen IS NOT NULL
      AND t.first_seen <= date($eval_date)
    """
    
    # Add tag filtering if provided
    if allowed_tags:
        query += """
      AND (r.candidate_tags IS NULL 
           OR ANY(tag IN r.candidate_tags WHERE tag IN $allowed_tags))
    """
    
    query += """
    // Ensure the relationship is evidenced by an article in the window
    // (both entities mentioned in same article)
    WITH DISTINCT s, r, t
    MATCH (s)<-[:MENTIONS]-(a2:Article)-[:MENTIONS]->(t)
    WHERE a2.date >= date($eval_date) - duration({days: $window_minus_1})
      AND a2.date <= date($eval_date)
    
    RETURN DISTINCT elementId(s) AS src, elementId(t) AS dst
    """
    
    params = {
        "eval_date": eval_date,
        "window_minus_1": window_days - 1,
    }
    
    if allowed_tags:
        params["allowed_tags"] = allowed_tags
    
    try:
        results = driver.run_query(query, parameters=params)
        edges = [(row["src"], row["dst"]) for row in results] if results else []
        
        logger.debug(
            "Fetched %d edges for eval_date=%s (window=%d days)",
            len(edges),
            eval_date,
            window_days,
        )
        return edges
    except Exception as e:
        logger.error("Error fetching edges for eval_date=%s: %s", eval_date, str(e))
        return []


def fetch_entities_mentioned_on_day(
    driver: GraphDriver,
    day_date: str,
) -> List[str]:
    """
    Get entity IDs mentioned in articles published on specific day.
    
    Uses the Article→PUBLISHED_ON→Day and Article→MENTIONS→Entity schema.
    
    Args:
        driver: GraphDriver instance
        day_date: Day date (YYYY-MM-DD format)
    
    Returns:
        List of entity element IDs mentioned on this day
    """
    logger.debug("Fetching entities mentioned on day=%s", day_date)
    
    query = """
    MATCH (a:Article)-[:PUBLISHED_ON]->(d:Day {date: $day_date})
    MATCH (a)-[:MENTIONS]->(e)
    WHERE e.first_seen IS NOT NULL
    RETURN DISTINCT elementId(e) AS entity_id
    """
    
    try:
        results = driver.run_query(query, parameters={"day_date": day_date})
        entity_ids = [row["entity_id"] for row in results] if results else []
        
        logger.debug(
            "Found %d entities mentioned on day=%s",
            len(entity_ids),
            day_date,
        )
        return entity_ids
    except Exception as e:
        logger.error("Error fetching entities for day=%s: %s", day_date, str(e))
        return []


def compute_rolling_hope_embeddings(
    driver: GraphDriver,
    days: List[str],
    window_days: int = 5,
    embedding_dim: int = 128,
    allowed_tags: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute rolling HOPE embeddings for each day in the dataset.
    
    Algorithm:
    ----------
    For each day t in days:
        1. Extract temporal snapshot: edges in [t - (window_days-1), t]
        2. Compute HOPE embeddings on this snapshot graph
        3. Get entities mentioned in articles published on day t
        4. Aggregate HOPE embeddings for these entities (mean pooling)
        5. Store: day_date → aggregated_embedding_vector
    
    Temporal guarantees:
    - Only entities with n.first_seen <= t are included
    - Only edges evidenced by articles in [t-4, t] are used
    - No future data leakage (entities/edges from t+1 onwards are excluded)
    
    Args:
        driver: GraphDriver instance
        days: List of day dates (YYYY-MM-DD format) to process
        window_days: Rolling window size in days (default 5 = [t-4, t])
        embedding_dim: HOPE embedding dimension (default 128)
        allowed_tags: Optional list of tags to filter relationships
    
    Returns:
        Dict mapping day_date → aggregated_embedding_vector
        Each embedding has shape (embedding_dim,), e.g. (128,)
    
    Notes:
        - Days with no edges get zero vectors
        - Days with no mentioned entities get zero vectors
        - Progress logged every 10 days
    """
    logger.info("=" * 80)
    logger.info("COMPUTING ROLLING HOPE EMBEDDINGS")
    logger.info("=" * 80)
    logger.info("Total days to process: %d", len(days))
    logger.info("Window size: %d days (inclusive)", window_days)
    logger.info("Embedding dimension: %d", embedding_dim)
    logger.info("Allowed tags: %s", allowed_tags or "None (all edges)")
    logger.info("-" * 80)
    
    day_embeddings = {}
    sorted_days = sorted(days)
    
    for i, day in enumerate(sorted_days, 1):
        # 1. Fetch temporal snapshot edges
        edges = fetch_edges_in_temporal_window(
            driver=driver,
            eval_date=day,
            window_days=window_days,
            allowed_tags=allowed_tags,
        )
        
        if not edges:
            logger.debug(
                "Day %s: No edges in temporal window, using zero vector",
                day,
            )
            day_embeddings[day] = np.zeros(embedding_dim, dtype=np.float32)
            continue
        
        # 2. Compute HOPE embeddings on this snapshot
        try:
            entity_embeddings = compute_hope_embeddings(edges, dim=embedding_dim)
        except Exception as e:
            logger.warning(
                "Day %s: Failed to compute HOPE embeddings (%s), using zero vector",
                day,
                str(e),
            )
            day_embeddings[day] = np.zeros(embedding_dim, dtype=np.float32)
            continue
        
        if not entity_embeddings:
            logger.debug(
                "Day %s: No entity embeddings computed, using zero vector",
                day,
            )
            day_embeddings[day] = np.zeros(embedding_dim, dtype=np.float32)
            continue
        
        # 3. Get entities mentioned on this day
        mentioned_entities = fetch_entities_mentioned_on_day(driver, day)
        
        if not mentioned_entities:
            logger.debug(
                "Day %s: No entities mentioned, using zero vector",
                day,
            )
            day_embeddings[day] = np.zeros(embedding_dim, dtype=np.float32)
            continue
        
        # 4. Aggregate embeddings (mean pooling over mentioned entities)
        valid_embeddings = [
            entity_embeddings[eid]
            for eid in mentioned_entities
            if eid in entity_embeddings
        ]
        
        if valid_embeddings:
            aggregated = np.mean(valid_embeddings, axis=0).astype(np.float32)
            day_embeddings[day] = aggregated
            logger.debug(
                "Day %s: Aggregated %d entity embeddings (from %d edges, %d mentioned)",
                day,
                len(valid_embeddings),
                len(edges),
                len(mentioned_entities),
            )
        else:
            logger.debug(
                "Day %s: No valid embeddings for mentioned entities, using zero vector",
                day,
            )
            day_embeddings[day] = np.zeros(embedding_dim, dtype=np.float32)
        
        # Log progress every 10 days
        if i % 10 == 0 or i == len(sorted_days):
            logger.info(
                "Progress: %d/%d days processed (%.1f%%)",
                i,
                len(sorted_days),
                100.0 * i / len(sorted_days),
            )
    
    logger.info("=" * 80)
    logger.info("ROLLING HOPE EMBEDDINGS COMPLETE")
    logger.info("✓ Computed embeddings for %d days", len(day_embeddings))
    logger.info("✓ Embedding dimension: %d", embedding_dim)
    
    # Count non-zero embeddings
    non_zero_count = sum(
        1 for emb in day_embeddings.values() if np.any(emb != 0)
    )
    logger.info("✓ Non-zero embeddings: %d (%.1f%%)", 
                non_zero_count, 
                100.0 * non_zero_count / len(day_embeddings) if day_embeddings else 0)
    logger.info("=" * 80)
    
    return day_embeddings
