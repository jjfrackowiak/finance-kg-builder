"""Candidate evaluation logic."""

import logging
from typing import Dict, List, Optional

import pandas as pd
from neo4j_graphrag.embeddings import Embedder

from kg_builder.config import Neo4jConfig
from kg_builder.core.graph import GraphDriver
from kg_builder.ml.embeddings import compute_hope_embeddings
from kg_builder.ml.modeling import (
    ModelMetrics,
    build_day_embedding_frame,
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
    embedding_dim: int = 128,
) -> ModelMetrics:
    """Evaluate a candidate ontology using HOPE embeddings.

    Args:
        driver: Graph driver
        embedder: Embedding model (unused, kept for compatibility)
        candidate_tags: Tags to evaluate
        day_labels: Day-level labels (unused, kept for compatibility)
        price_df: Price data (unused, kept for compatibility)
        allowed_tags: Tags to include in embedding computation (base + winners + current)
        embedding_dim: HOPE embedding dimension (default 128)

    Returns:
        Model metrics
    """
    logger.info("Evaluating candidate with tags: %s", candidate_tags)

    # Count total nodes in graph (for this candidate's context)
    try:
        node_count_results = driver.run_query("MATCH (n) RETURN count(n) as node_count")
        n_nodes = node_count_results[0]["node_count"] if node_count_results else 0
        logger.info("Total nodes in graph for candidate: %d", n_nodes)
    except Exception as e:
        logger.warning("Could not count nodes: %s", str(e))
        n_nodes = 0

    # Fetch edges for this candidate, filtered by allowed tags if provided
    edges = fetch_edges_for_candidate(driver, None, candidate_tags, allowed_tags=allowed_tags)

    if not edges:
        logger.warning("No edges found for candidate tags: %s", candidate_tags)
        return ModelMetrics(auc=0.0, f1=0.0, n_train=n_nodes, n_val=0)

    # Compute HOPE embeddings
    try:
        embedding_dict = compute_hope_embeddings(edges, dim=embedding_dim)
    except Exception as e:
        logger.error("Failed to compute HOPE embeddings: %s", str(e))
        return ModelMetrics(auc=0.0, f1=0.0, n_train=n_nodes, n_val=0)

    if not embedding_dict:
        logger.warning("No embeddings computed for candidate tags: %s", candidate_tags)
        return ModelMetrics(auc=0.0, f1=0.0, n_train=n_nodes, n_val=0)

    # Fetch day nodes with labels
    df_days = fetch_day_nodes_with_labels(driver, None)

    if df_days.empty:
        logger.warning("No day nodes with labels found")
        return ModelMetrics(auc=0.0, f1=0.0, n_train=n_nodes, n_val=0)

    # Build embedding frame
    df_embed = build_day_embedding_frame(df_days, embedding_dict)

    if df_embed.empty:
        logger.warning("No embeddings found for day nodes")
        return ModelMetrics(auc=0.0, f1=0.0, n_train=n_nodes, n_val=0)

    # Train classifier (returns metrics with n_train/n_val from temporal split)
    metrics = train_classifier_on_embeddings(df_embed)

    # Override n_train/n_val to represent total nodes instead of temporal samples
    metrics.n_train = n_nodes
    metrics.n_val = len(embedding_dict)  # Number of nodes that got embeddings

    return metrics
