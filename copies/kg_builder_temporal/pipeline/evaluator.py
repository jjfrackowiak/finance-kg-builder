"""Candidate evaluation logic."""

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from neo4j_graphrag.embeddings import Embedder

from copies.kg_builder_temporal.config import Neo4jConfig
from copies.kg_builder_temporal.core.graph import GraphDriver
from copies.kg_builder_temporal.ml.modeling import (
    ModelMetrics,
    temporal_train_val_split,
    train_classifier_on_embeddings,
)

logger = logging.getLogger(__name__)



def fetch_edges_for_candidate(
    driver: GraphDriver,
    neo4j_cfg: Optional[Neo4jConfig],
    candidate_tags: List[str],
    allowed_tags: Optional[List[str]] = None,
) -> List[tuple]:
    """Fetch edges from Neo4j for specific tags.

    When allowed_tags is provided, fetches edges that have ANY of those tags.
    This includes: base_structure (articles) + previous winners + current candidate.
    Excludes sibling candidates from the same step.

    Args:
        driver: Graph driver
        neo4j_cfg: Neo4j configuration (unused, for compatibility)
        candidate_tags: Primary tags to fetch (for logging)
        allowed_tags: If provided, only fetch edges with these tags (base + winners + current)

    Returns:
        List of (src_id, dst_id) tuples
    """
    if allowed_tags:
        logger.info("Fetching edges with allowed tags: %s", allowed_tags)
    else:
        logger.info("Fetching all edges from graph")

    try:
        if allowed_tags:
            query = """
            MATCH (n)-[r]->(m)
            WHERE r.candidate_tags IS NULL OR ANY(tag IN r.candidate_tags WHERE tag IN $allowed_tags)
            RETURN elementId(n) AS src, elementId(m) AS dst
            """
            results = driver.run_query(query, parameters={"allowed_tags": allowed_tags})
        else:
            query = """
            MATCH (n)-[r]->(m)
            RETURN elementId(n) AS src, elementId(m) AS dst
            """
            results = driver.run_query(query)

        edges = [(row["src"], row["dst"]) for row in results] if results else []
        logger.info("Fetched %d edges", len(edges))
        return edges
    except Exception as e:
        logger.warning("Error fetching edges: %s", str(e))
        return []


def fetch_day_nodes_with_labels(
    driver: GraphDriver,
    neo4j_cfg: Optional[Neo4jConfig],
) -> pd.DataFrame:
    """Fetch Day nodes with labels from Neo4j.

    Args:
        driver: Graph driver
        neo4j_cfg: Neo4j configuration (unused, for compatibility)

    Returns:
        DataFrame with columns: node_id, day, direction, return_next_day
    """
    logger.info("Fetching Day nodes with labels")

    try:
        query = """
        MATCH (d:Day)
        WHERE d.direction IS NOT NULL
        RETURN elementId(d) AS node_id,
               d.date AS day,
               d.direction AS direction,
               d.return_next_day AS return_next_day
        """

        results = driver.run_query(query)
        if results:
            df = pd.DataFrame(results)
            logger.info("Fetched %d Day nodes with labels", len(df))
            return df
        else:
            logger.warning("No Day nodes with labels found")
            return pd.DataFrame()
    except Exception as e:
        logger.warning("Error fetching day nodes: %s", str(e))
        return pd.DataFrame()


def evaluate_candidate(
    driver: GraphDriver,
    embedder: Embedder,
    candidate_tags: List[str],
    day_labels: Dict[str, int],
    price_df: pd.DataFrame,
    allowed_tags: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
    embedding_dim: int = 128,
) -> ModelMetrics:
    """Evaluate a candidate ontology using ROLLING TEMPORAL HOPE EMBEDDINGS.

    Strategy:
    ---------
    For each day t:
        1. Build 5-day temporal graph snapshot [t-4, t]
        2. Compute HOPE embeddings on this snapshot
        3. Aggregate embeddings for entities mentioned on day t
        4. Result: One embedding vector per day
    
    Then train XGBoost classifier on these day-level embeddings.

    Args:
        driver: Graph driver
        embedder: Embedding model (unused, kept for compatibility)
        candidate_tags: Tags to evaluate
        day_labels: Day-level labels {day_date: direction} for training
        price_df: Price data (unused, kept for compatibility)
        allowed_tags: Tags to filter relationships (base + winners + current candidate)
        api_key: OpenAI API key (unused, kept for compatibility)
        embedding_type: Embedding type (unused, kept for compatibility) 
        local_model: Local model name (unused, kept for compatibility)
        embedding_dim: HOPE embedding dimension (default 128)

    Returns:
        Model metrics
    """
    logger.info("=" * 80)
    logger.info("EVALUATING CANDIDATE WITH ROLLING TEMPORAL HOPE EMBEDDINGS")
    logger.info("=" * 80)
    logger.info("Candidate tags: %s", candidate_tags)
    logger.info("Allowed tags: %s", allowed_tags)
    logger.info("Embedding dimension: %d", embedding_dim)
    logger.info("-" * 80)

    if not day_labels:
        logger.warning("No day labels provided")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    # Fetch Day nodes with labels from Neo4j
    df_days = fetch_day_nodes_with_labels(driver, None)
    
    if df_days.empty:
        logger.error("❌ NO DAY NODES WITH LABELS FOUND")
        logger.error("   → Day nodes exist but 'direction' property not set")
        logger.error("   → This usually means price labels were never written to Neo4j")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    # Extract unique days from Day nodes
    days = sorted(df_days["day"].unique().tolist())
    logger.info("Processing %d days with labels", len(days))
    
    # Compute rolling 5-day temporal HOPE embeddings
    logger.info("Computing rolling HOPE embeddings (5-day windows)...")
    from copies.kg_builder_temporal.ml.temporal_hope_embeddings import compute_rolling_hope_embeddings
    
    try:
        day_embedding_dict = compute_rolling_hope_embeddings(
            driver=driver,
            days=days,
            window_days=5,  # [t-4, t] inclusive = 5 days
            embedding_dim=embedding_dim,
            allowed_tags=allowed_tags,
        )
    except Exception as e:
        logger.error("Failed to compute rolling embeddings: %s", str(e))
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    if not day_embedding_dict:
        logger.warning("No day embeddings computed")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    # Build embedding frame (maps day dates to embeddings + labels)
    from copies.kg_builder_temporal.ml.modeling import build_day_embedding_frame_from_rolling
    
    df_embed = build_day_embedding_frame_from_rolling(df_days, day_embedding_dict)
    
    if df_embed.empty:
        logger.warning("No embeddings found for day nodes")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    logger.info("Built embedding frame with %d days", len(df_embed))
    
    # Train classifier with temporal split (existing function handles this)
    logger.info("Training XGBoost classifier on day-level HOPE embeddings...")
    try:
        metrics = train_classifier_on_embeddings(df_embed)
    except Exception as e:
        logger.error("Failed to train classifier: %s", str(e))
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("✓ AUC: %.4f", metrics.auc)
    logger.info("✓ F1: %.4f", metrics.f1)
    logger.info("✓ Train samples: %d", metrics.max_hops_train)
    logger.info("✓ Val samples: %d", metrics.max_hops_val)
    logger.info("=" * 80)
    
    return metrics

