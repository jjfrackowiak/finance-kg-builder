"""Candidate evaluation logic."""

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from neo4j_graphrag.embeddings import Embedder

from kg_builder_llm.config import Neo4jConfig
from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.ml.feature_engineering import (
    build_day_feature_vector,
    describe_feature_blocks,
)
from kg_builder_llm.ml.modeling import (
    ModelMetrics,
    temporal_train_val_split,
    train_classifier_on_embeddings,
)

logger = logging.getLogger(__name__)


def count_graph_by_tags(
    driver: GraphDriver, allowed_tags: Optional[List[str]]
) -> "tuple[int, int]":
    """Count nodes and relationships belonging to a candidate's effective graph.

    Scopes to entities tagged with any of allowed_tags (base + prior winners +
    current candidate). Returns (n_nodes, n_rels); (0, 0) on any failure so a
    counting error never aborts evaluation. Used for the KG-growth figure.
    """
    try:
        if allowed_tags:
            node_q = (
                "MATCH (n) WHERE n.candidate_tags IS NOT NULL "
                "AND any(t IN n.candidate_tags WHERE t IN $tags) "
                "RETURN count(n) AS c"
            )
            rel_q = (
                "MATCH ()-[r]->() WHERE r.candidate_tags IS NOT NULL "
                "AND any(t IN r.candidate_tags WHERE t IN $tags) "
                "RETURN count(r) AS c"
            )
            params = {"tags": allowed_tags}
        else:
            node_q = "MATCH (n) RETURN count(n) AS c"
            rel_q = "MATCH ()-[r]->() RETURN count(r) AS c"
            params = {}
        n_nodes = driver.run_query(node_q, params)[0]["c"]
        n_rels = driver.run_query(rel_q, params)[0]["c"]
        return int(n_nodes), int(n_rels)
    except Exception as e:
        logger.warning("Failed to count graph size for tags %s: %s", allowed_tags, e)
        return 0, 0


def evaluate_article_text_baseline(
    driver: GraphDriver,
    day_labels: Dict[str, int],
    lookback_days: int = 2,
    train_ratio: float = 0.7,
) -> ModelMetrics:
    """Evaluate a text-only baseline using mean-pooled article text embeddings.

    Uses EXACTLY the same day set and train/val split as ``evaluate_candidate``:
    every day in ``day_labels`` is included; days with no article embeddings in
    their lookback window receive a zero vector (mirroring how candidates handle
    days with no graph chains).

    Args:
        driver: Graph driver
        day_labels: {day_date: direction} — must be the same dict passed to
            evaluate_candidate so the splits are identical.
        lookback_days: Number of calendar days to look back for articles.
        train_ratio: Fraction of days used for training.

    Returns:
        ModelMetrics with AUC/F1 of the text-only classifier (max_hops always 0).
    """
    from datetime import date, timedelta

    logger.info("=== Article-text baseline evaluation (lookback=%d days) ===", lookback_days)

    # Use exactly the same sorted day list as evaluate_candidate would.
    days = sorted(day_labels.keys())

    # Fetch mean-pooled embeddings for days that have articles in their window.
    # d.date is stored as a string in Neo4j — use string comparison.
    day_windows = [
        {
            "eval_date": d,
            "window_start": str(date.fromisoformat(d) - timedelta(days=lookback_days)),
        }
        for d in days
    ]

    query = """
    UNWIND $day_windows AS entry
    MATCH (a:Article)-[:PUBLISHED_ON]->(d:Day)
    WHERE toString(d.date) >= entry.window_start
      AND toString(d.date) <= entry.eval_date
      AND a.text_embedding IS NOT NULL
    WITH entry.eval_date AS eval_date, collect(a.text_embedding) AS embeddings
    WHERE size(embeddings) > 0
    RETURN eval_date,
           [i IN range(0, size(embeddings[0]) - 1) |
               reduce(s = 0.0, e IN embeddings | s + e[i]) / size(embeddings)
           ] AS mean_embedding
    """

    emb_by_day: Dict[str, np.ndarray] = {}
    try:
        rows = driver.run_query(query, parameters={"day_windows": day_windows})
        for row in rows or []:
            day = str(row["eval_date"])
            emb = row.get("mean_embedding")
            if emb and day in day_labels:
                emb_by_day[day] = np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.error("Baseline query failed: %s", e)
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    if not emb_by_day:
        logger.warning("No article embeddings found for baseline — were embeddings written?")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    # Determine embedding dimension from any available vector.
    emb_dim = next(iter(emb_by_day.values())).shape[0]
    zero_vec = np.zeros(emb_dim, dtype=np.float32)

    # Build feature matrix over ALL days — zero vector where no articles found.
    # This guarantees the same day list and split as evaluate_candidate.
    X = np.array([emb_by_day.get(d, zero_vec) for d in days], dtype=np.float32)
    y = np.array([day_labels[d] for d in days], dtype=np.int32)

    logger.info(
        "Baseline: %d / %d days have article embeddings (rest → zero vector)",
        len(emb_by_day), len(days),
    )

    try:
        train_idx, val_idx = temporal_train_val_split(days, train_ratio=train_ratio)
    except Exception as e:
        logger.error("Baseline train/val split failed: %s", e)
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    if not train_idx or not val_idx:
        logger.warning("Baseline train/val split produced empty sets")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    df_train = pd.DataFrame({"features": list(X[train_idx]), "direction": y[train_idx]})
    df_val   = pd.DataFrame({"features": list(X[val_idx]),   "direction": y[val_idx]})

    try:
        metrics = train_classifier_on_embeddings(df_train, df_val)
        metrics.max_hops_train = 0
        metrics.max_hops_val = 0
    except Exception as e:
        logger.error("Baseline classifier failed: %s", e)
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    logger.info("Baseline AUC=%.4f, F1=%.4f", metrics.auc, metrics.f1)
    return metrics


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
    lookback_days: int = 2,
    min_chain_hops: int = 5,
    max_chain_hops: int = 5,
    path_uniqueness: str = "NODE_PATH",
    feature_mode: str = "path",
    max_metapath_hops: int = 2,
    train_ratio: float = 0.7,
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
        embedding_type: Type of embedding to use ("local" or "openai")
        local_model: Local model name for sentence-transformers
        lookback_days: Number of days to look back for article/feature extraction
        min_chain_hops: Minimum path length for relationship chains
        max_chain_hops: Maximum path length for relationship chains
        path_uniqueness: APOC path uniqueness mode (NODE_PATH, NODE_GLOBAL, RELATIONSHIP_PATH, RELATIONSHIP_GLOBAL)
        feature_mode: Feature representation mode ("path", "subgraph", or "hybrid")
        max_metapath_hops: Maximum hop count for metapath features in subgraph modes
        train_ratio: Fraction of data for training (0.7 = 70/30 split)

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
    
    # Process all days synchronously (batch query handles parallelism)
    for day_date, label in day_labels.items():
        try:
            feature_vector, chain_count, max_hops = build_day_feature_vector(
                driver,
                eval_date=day_date,
                lookback_days=lookback_days,
                text_agg_method="mean",
                api_key=api_key,
                allowed_tags=allowed_tags,
                chain_agg_method="mean",
                embedding_type=embedding_type,
                local_model=local_model,
                min_chain_hops=min_chain_hops,
                max_chain_hops=max_chain_hops,
                feature_mode=feature_mode,
                max_metapath_hops=max_metapath_hops,
            )
            feature_vectors[day_date] = (feature_vector, label)
            max_hop_counts[day_date] = max_hops
        except Exception as e:
            logger.warning("Failed to build features for day %s: %s", day_date, str(e))
    
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

    # Per-block coverage. A block can be all zeros while its dimensions are
    # still present, so the shape line above cannot distinguish real features
    # from a dead block -- only these numbers can.
    block_stats = describe_feature_blocks(
        feature_matrix,
        feature_mode=feature_mode,
        embedding_type=embedding_type,
        local_model=local_model,
    )
    for name in ("path", "subgraph", "topo"):
        frac = block_stats.get(f"feat/{name}_nonzero_frac")
        if frac is None:
            continue
        dims = int(block_stats.get(f"feat/{name}_dims", 0))
        norm = block_stats.get(f"feat/{name}_mean_norm", 0.0)
        if dims and frac == 0.0:
            logger.warning(
                "⚠ FEATURE BLOCK '%s' IS ALL ZEROS across %d dims -- it contributes "
                "nothing to the model", name, dims,
            )
        else:
            logger.info(
                "Feature block '%s': dims=%d, live_cols=%.1f%%, mean_norm=%.4f",
                name, dims, 100 * frac, norm,
            )
    
    # Split into train/val by day (temporal split)
    try:
        train_idx, val_idx = temporal_train_val_split(
            list(feature_vectors.keys()),
            train_ratio=train_ratio,
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
        metrics.block_stats = block_stats
    except Exception as e:
        logger.error("Failed to train classifier: %s", str(e))
        failed = ModelMetrics(
            auc=0.0, f1=0.0, max_hops_train=train_max_hops, max_hops_val=val_max_hops
        )
        failed.block_stats = block_stats
        return failed

    # Record graph size for this candidate's effective (allowed-tag) subgraph so
    # KG growth across evolution steps can be plotted from parent-run metrics.
    metrics.n_nodes_total, metrics.n_rels_total = count_graph_by_tags(driver, allowed_tags)
    logger.info(
        "Candidate graph size: %d nodes, %d relationships",
        metrics.n_nodes_total,
        metrics.n_rels_total,
    )

    return metrics
