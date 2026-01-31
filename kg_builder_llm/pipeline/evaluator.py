"""Candidate evaluation logic."""

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from neo4j_graphrag.embeddings import Embedder

from kg_builder_llm.config import Neo4jConfig
from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.ml.feature_engineering import build_day_feature_vector
from kg_builder_llm.ml.modeling import (
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


async def evaluate_candidate(
    driver: GraphDriver,
    embedder: Embedder,
    candidate_tags: List[str],
    day_labels: Dict[str, int],
    price_df: pd.DataFrame,
    allowed_tags: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> ModelMetrics:
    """Evaluate a candidate ontology using text embeddings + relationship chains + topology features.

    Args:
        driver: Graph driver
        embedder: Embedding model (unused, kept for compatibility)
        candidate_tags: Tags to evaluate
        day_labels: Day-level labels {day_date: direction} for training
        price_df: Price data (unused, kept for compatibility)
        allowed_tags: Tags to filter relationships for chain extraction (base + previous winners + current candidate)
        api_key: OpenAI API key for text embeddings
        semaphore: Optional asyncio.Semaphore for rate limiting concurrent API calls

    Returns:
        Model metrics
    """
    logger.info("Evaluating candidate with tags: %s", candidate_tags)

    if not day_labels:
        logger.warning("No day labels provided")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    # Fetch Day nodes and prepare feature vectors
    logger.info("Building feature vectors for %d days (with allowed_tags=%s)", len(day_labels), allowed_tags)
    
    feature_vectors = {}
    max_hop_counts = {}  # Track max hop count per day
    
    # Create tasks for all days
    tasks = []
    for day_date, label in day_labels.items():
        task = build_day_feature_vector(
            driver,
            eval_date=day_date,
            lookback_days=2,
            text_agg_method="mean",
            api_key=api_key,
            allowed_tags=allowed_tags,
            chain_agg_method="mean",
            semaphore=semaphore,
            embedding_type=embedding_type,
            local_model=local_model,
        )
        tasks.append((day_date, label, task))
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    # Process results
    for (day_date, label, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning("Failed to build features for day %s: %s", day_date, str(result))
        else:
            feature_vector, chain_count, max_hops = result
            feature_vectors[day_date] = (feature_vector, label)
            max_hop_counts[day_date] = max_hops
    
    if not feature_vectors:
        logger.warning("No feature vectors built")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    # Create feature matrix and labels array
    feature_matrix = np.array(
        [fv[0] for fv in feature_vectors.values()],
        dtype=np.float32,
    )
    labels = np.array(
        [fv[1] for fv in feature_vectors.values()],
        dtype=np.int32,
    )
    
    logger.info("Feature matrix shape: %s, labels shape: %s", feature_matrix.shape, labels.shape)
    logger.info("Feature vectors keys (days): %s", sorted(list(feature_vectors.keys())))
    
    # Split into train/val by day (temporal split)
    try:
        train_idx, val_idx = temporal_train_val_split(
            list(feature_vectors.keys()),
            train_ratio=0.7,
        )
    except Exception as e:
        logger.error("Failed to split train/val: %s", str(e))
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    if not train_idx or not val_idx:
        logger.warning("Train/val split resulted in empty sets")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)
    
    # Get actual day dates for train/val periods
    day_list = list(feature_vectors.keys())
    train_days = [day_list[i] for i in train_idx]
    val_days = [day_list[i] for i in val_idx]
    
    # Get max hop counts for train/val (longest path per period)
    train_max_hops = max([max_hop_counts.get(day, 0) for day in train_days] + [0])
    val_max_hops = max([max_hop_counts.get(day, 0) for day in val_days] + [0])
    
    logger.info("Longest path in train period (%d days): %d hops", len(train_days), train_max_hops)
    logger.info("Longest path in val period (%d days): %d hops", len(val_days), val_max_hops)
    
    # Create train/val dataframes for modeling
    X_train = feature_matrix[train_idx]
    X_val = feature_matrix[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    
    logger.info(
        "Train/val split: train=%d (X:%s, y:%s), val=%d (X:%s, y:%s)",
        len(train_idx), X_train.shape, y_train.shape,
        len(val_idx), X_val.shape, y_val.shape,
    )
    
    # Train classifier
    df_train = pd.DataFrame({
        'features': list(X_train),
        'direction': y_train,
    })
    df_val = pd.DataFrame({
        'features': list(X_val),
        'direction': y_val,
    })
    
    try:
        metrics = train_classifier_on_embeddings(df_train, df_val)
        # Override max_hops_train/max_hops_val with longest path (max hops) instead of counts
        metrics.max_hops_train = train_max_hops
        metrics.max_hops_val = val_max_hops
    except Exception as e:
        logger.error("Failed to train classifier: %s", str(e))
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=train_max_hops, max_hops_val=val_max_hops)
    
    return metrics

