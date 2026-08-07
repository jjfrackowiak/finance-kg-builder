"""Feature engineering for text embeddings + topology approach."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.ml.embeddings import embed_text_deterministic, get_embedding_dim
from kg_builder_llm.ml.relationship_chains import (
    aggregate_chain_embeddings,
    embed_relationship_chains,
)
from kg_builder_llm.ml.subgraph_features import build_temporal_subgraph_feature_vector
from kg_builder_llm.ml.topology_features import (
    aggregate_topology_features,
    compute_topology_features,
)

logger = logging.getLogger(__name__)


def get_articles_in_date_range(
    driver: GraphDriver,
    end_date: str,
    lookback_days: int = 2,
) -> List[dict]:
    """
    Get all articles published in [end_date - lookback_days, end_date].
    
    Args:
        driver: GraphDriver instance
        end_date: End date (YYYY-MM-DD)
        lookback_days: Number of days to look back
    
    Returns:
        List of article dicts with 'id', 'date', 'text'
    """
    query = """
    MATCH (a:Article)
    WHERE a.date >= date($end_date) - duration({days: $lookback_days})
      AND a.date <= date($end_date)
    RETURN elementId(a) as article_id, a.date as date, a.text_embedding as text_embedding
    ORDER BY a.date DESC
    """
    
    try:
        results = driver.run_query(
            query,
            parameters={"end_date": end_date, "lookback_days": lookback_days},
        )
    except Exception as e:
        logger.error("Error fetching articles in date range: %s", str(e))
        return []
    
    articles = []
    if results:
        for row in results:
            articles.append({
                "id": row.get("article_id"),
                "date": row.get("date"),
                "text": row.get("text"),
            })
    
    logger.info("Found %d articles in date range [%s - %d days, %s]", 
               len(articles), end_date, lookback_days, end_date)
    return articles


def get_entities_mentioned_in_articles(
    driver: GraphDriver,
    article_ids: List[str],
) -> List[dict]:
    """
    Get all entities mentioned in given articles.
    
    Args:
        driver: GraphDriver instance
        article_ids: List of article element IDs
    
    Returns:
        List of entity dicts with 'id', 'name', 'label'
    """
    if not article_ids:
        return []
    
    # Convert article IDs to parameter format
    article_param = "', '".join(article_ids)
    
    query = f"""
    MATCH (a:Article)-[r:MENTIONED_IN|PUBLISHED_ON|REFERENCES]-(n)
    WHERE elementId(a) IN ['{article_param}']
    RETURN DISTINCT elementId(n) as node_id, 
                    n.name as name,
                    labels(n)[0] as label
    """
    
    try:
        results = driver.run_query(query)
    except Exception as e:
        logger.error("Error fetching entities: %s", str(e))
        return []
    
    entities = []
    if results:
        for row in results:
            name = row.get("name")
            if name:  # Only include entities with names
                entities.append({
                    "id": row.get("node_id"),
                    "name": str(name),
                    "label": row.get("label"),
                })
    
    logger.info("Found %d entities mentioned in %d articles", len(entities), len(article_ids))
    return entities


def aggregate_embeddings(
    embeddings: List[np.ndarray],
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate multiple embeddings into single vector.
    
    Args:
        embeddings: List of embedding vectors
        method: Aggregation method (mean, sum, max)
    
    Returns:
        Aggregated embedding vector
    """
    if not embeddings:
        logger.warning("No embeddings to aggregate, returning zeros")
        return np.zeros(1536, dtype=np.float32)  # Assuming text-embedding-3-small (1536 dims)
    
    emb_matrix = np.array(embeddings, dtype=np.float32)
    
    if method == "mean":
        result = np.mean(emb_matrix, axis=0)
    elif method == "sum":
        result = np.sum(emb_matrix, axis=0)
    elif method == "max":
        result = np.max(emb_matrix, axis=0)
    else:
        logger.warning("Unknown aggregation method: %s, using mean", method)
        result = np.mean(emb_matrix, axis=0)
    
    logger.debug("Aggregated %d embeddings using %s", len(embeddings), method)
    return result


def build_day_feature_vector(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int = 2,
    text_agg_method: str = "mean",
    api_key: Optional[str] = None,
    allowed_tags: Optional[List[str]] = None,
    chain_agg_method: str = "mean",
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
    min_chain_hops: int = 5,
    max_chain_hops: int = 5,
    path_uniqueness: str = "NODE_PATH",
    feature_mode: str = "path",
    max_metapath_hops: int = 2,
) -> Tuple[np.ndarray, int, int]:
    """
    Build combined feature vector for a prediction day.
    
    Process:
    1. Get articles from [eval_date - lookback_days, eval_date]
    Modes:
    - path: current chain-embedding representation + topology
    - subgraph: fixed-schema temporal subgraph features + topology
    - hybrid: path features + temporal subgraph features + topology
    
    Args:
        driver: GraphDriver instance
        eval_date: Prediction date (YYYY-MM-DD)
        lookback_days: Days of history to include
        text_agg_method: (unused - kept for compatibility)
        api_key: OpenAI API key (for chain embedding)
        allowed_tags: Tags to filter relationships for chain extraction
        chain_agg_method: How to aggregate chain embeddings ('mean', 'max', 'weighted_mean')
        embedding_type: Type of embedding to use ("local" or "openai")
        local_model: Local model name for sentence-transformers
        min_chain_hops: Minimum path length for relationship chains
        max_chain_hops: Maximum path length for relationship chains
        path_uniqueness: APOC path uniqueness mode (NODE_PATH, NODE_GLOBAL, RELATIONSHIP_PATH, RELATIONSHIP_GLOBAL)
        feature_mode: "path", "subgraph", or "hybrid"
        max_metapath_hops: Maximum hop count for typed metapath features
    
    Returns:
        Tuple of (feature_vector, total_chains, max_hop_count):
        - feature_vector: np.ndarray (shape depends on feature_mode)
        - total_chains: int, total chains extracted for this day
        - max_hop_count: int, maximum hop count among all chains (longest path)
    """
    logger.info("Building feature vector for eval_date=%s", eval_date)
    
    # Step 1: Get articles in window
    articles = get_articles_in_date_range(driver, eval_date, lookback_days)
    
    # Get embedding dimension dynamically
    embed_dim = get_embedding_dim(embedding_type, local_model)
    
    if not articles:
        logger.warning("No articles found in window, returning zero vector")
        use_path = feature_mode in {"path", "hybrid"}
        use_subgraph = feature_mode in {"subgraph", "hybrid"}
        zero_dim = (embed_dim if use_path else 0) + (96 if use_subgraph else 0) + 8
        return np.zeros(zero_dim, dtype=np.float32), 0, 0
    
    # Skip text embedding - current representation uses structural signals only.
    logger.info("Skipping text embeddings - using structural feature blocks")

    chains_embeddings = []
    total_chains_extracted = 0
    total_chains_embedded = 0
    max_hop_count_overall = 0  # Track longest path across all articles

    use_path_features = feature_mode in {"path", "hybrid"}
    use_subgraph_features = feature_mode in {"subgraph", "hybrid"}

    logger.info(
        "Feature mode=%s (use_path=%s, use_subgraph=%s)",
        feature_mode,
        use_path_features,
        use_subgraph_features,
    )

    if use_path_features and allowed_tags:
        logger.info("✓ allowed_tags is truthy, entering chain extraction block")
        logger.info("Processing %d articles for relationship chain extraction...", len(articles))
        
        # STEP 1: Build batch jobs for all articles
        from kg_builder_llm.ml.relationship_chains import extract_chains_batch
        
        # Excluded relationship types (uninformative for chains)
        excluded_rels = ['PUBLISHED_ON', 'WRITTEN_BY', 'MENTIONS', 'MENTIONED_IN', 'REFERENCES']
        
        jobs = []
        article_metadata = []
        for article_idx, article in enumerate(articles, 1):
            article_id = article.get("id", "")
            article_date = article.get("date", "")
            # Convert date object to string format YYYY-MM-DD if needed
            if hasattr(article_date, 'isoformat'):
                article_date = article_date.isoformat()
            elif article_date and not isinstance(article_date, str):
                article_date = str(article_date)
            
            logger.info("Article %d/%d: %s (date=%s) - adding to batch", article_idx, len(articles), article_id[:8], article_date)
            
            # Create job for this article.
            # Use the article's own publication date as the temporal boundary so
            # the chain extractor only sees entities that existed at the time of
            # this article — no leakage from future articles in the same window.
            jobs.append({
                "job_id": article_idx,
                "article_id": article_id,
                "eval_date": article_date,
                "allowed_tags": allowed_tags,
                "excluded_rels": excluded_rels,
                "min_level": min_chain_hops,
                "max_level": max_chain_hops,
                "uniqueness": path_uniqueness
            })
            article_metadata.append((article_idx, article_id))
        
        # STEP 2: Execute batch query - ONE roundtrip to Neo4j for all articles!
        logger.info("Extracting chains from %d articles in batch query...", len(jobs))
        chains_by_job = extract_chains_batch(driver, jobs)
        
        # STEP 3: Collect all chains from all articles
        all_chains = []
        article_chain_counts = []  # Track how many chains per article
        for article_idx, article_id in article_metadata:
            chains = chains_by_job.get(article_idx, [])
            num_chains = len(chains)
            total_chains_extracted += num_chains
            article_chain_counts.append(num_chains)
            
            if num_chains > 0:
                # Tag chains with article index for later grouping
                for chain in chains:
                    chain['_article_idx'] = article_idx
                all_chains.extend(chains)
                logger.info("✓ Article %d (%s): Got %d chains from batch", article_idx, article_id[:8], num_chains)
            else:
                    logger.warning("✗ Article %d (%s): No chains found", article_idx, article_id[:8])
        
        logger.info("Total chains extracted: %d from %d articles", len(all_chains), len(articles))
        
        # STEP 3: Batch embed ALL chains at once (local, fast)
        if all_chains:
            from kg_builder_llm.ml.relationship_chains import embed_relationship_chains
            
            logger.info("Batch embedding %d chains...", len(all_chains))
            embedded_chains = embed_relationship_chains(
                all_chains,
                api_key=api_key,
                embedding_type=embedding_type,
                local_model=local_model,
            )
            total_chains_embedded = len(embedded_chains)
            logger.info("✓ Embedded %d chains in one batch", total_chains_embedded)
            
            # STEP 4: Group embeddings back by article and aggregate per article
            from kg_builder_llm.ml.relationship_chains import aggregate_chain_embeddings
            
            article_embeddings = {}  # article_idx -> list of embedded chains
            chains_without_idx = 0
            for embedded_chain in embedded_chains:
                article_idx = embedded_chain.get('_article_idx')
                if article_idx is None:
                    chains_without_idx += 1
                    logger.debug("Chain missing _article_idx: %s", embedded_chain.get('chain_text', '')[:50])
                if article_idx not in article_embeddings:
                    article_embeddings[article_idx] = []
                article_embeddings[article_idx].append(embedded_chain)
            
            if chains_without_idx > 0:
                logger.warning("WARNING: %d/%d chains missing _article_idx - grouping will fail!", 
                             chains_without_idx, len(embedded_chains))
            
            # Aggregate chains per article (filter out None keys)
            valid_indices = [idx for idx in article_embeddings.keys() if idx is not None]
            none_chains = len(article_embeddings.get(None, []))
            logger.info("Article grouping: %d valid indices, %d chains with None idx. Keys: %s", 
                       len(valid_indices), none_chains, sorted([k for k in article_embeddings.keys() if k is not None]))
            
            for article_idx in sorted(valid_indices):
                article_chains = article_embeddings[article_idx]
                if article_chains:
                    # Aggregate embeddings for this article
                    article_agg = aggregate_chain_embeddings(article_chains, method=chain_agg_method)
                    chains_embeddings.append(article_agg)
                    
                    # Track max hops
                    article_max_hops = max(chain.get("hop_count") or 0 for chain in article_chains)
                    max_hop_count_overall = max(max_hop_count_overall, article_max_hops)
                    # Safe string formatting
                    if article_idx is not None:
                        logger.info("✓ Article %d: Aggregated %d chains (max_hops=%d)", article_idx, len(article_chains), article_max_hops)
        else:
            logger.warning("No chains extracted from any article")
        
        logger.info("=" * 80)
        logger.info("CHAIN EXTRACTION SUMMARY")
        logger.info("=" * 80)
        logger.info("Total articles processed: %d", len(articles))
        logger.info("Total chains extracted: %d", total_chains_extracted)
        logger.info("Total chains embedded: %d", total_chains_embedded)
        logger.info("Articles with chains: %d/%d", len(chains_embeddings), len(articles))
        logger.info("=" * 80)
        
        # Step 5: Aggregate chain embeddings across articles
        if chains_embeddings:
            logger.info("Aggregating %d article embeddings using '%s' method...", len(chains_embeddings), chain_agg_method)
            chains_agg = aggregate_chain_embeddings(
                [{"embedding": emb} for emb in chains_embeddings],
                method=chain_agg_method,
            )
            logger.info(
                "✓ Chain aggregation complete: shape=%s, magnitude=%.4f",
                chains_agg.shape,
                np.linalg.norm(chains_agg),
            )
        else:
            logger.warning("⚠ No chains embedded, using zero vector")
            chains_agg = np.zeros(embed_dim, dtype=np.float32)
    else:
        if use_path_features and not allowed_tags:
            logger.warning("⚠ allowed_tags not provided, skipping chain extraction")
        chains_agg = np.zeros(embed_dim, dtype=np.float32)

    if use_subgraph_features:
        subgraph_agg = build_temporal_subgraph_feature_vector(
            driver,
            eval_date=eval_date,
            lookback_days=lookback_days,
            allowed_tags=allowed_tags,
            max_metapath_hops=max_metapath_hops,
        )
    else:
        subgraph_agg = np.zeros(0, dtype=np.float32)

    # Step 6: Compute topology features
    topo_features_dict = compute_topology_features(driver, eval_date, lookback_days)

    # Step 7: Aggregate topology features
    topo_agg = aggregate_topology_features(topo_features_dict, agg_method="mean")
    logger.debug("Topology aggregate shape: %s", topo_agg.shape)

    feature_blocks = []
    if use_path_features:
        feature_blocks.append(chains_agg)
    if use_subgraph_features:
        feature_blocks.append(subgraph_agg)
    feature_blocks.append(topo_agg)

    # Step 8: Concatenate enabled feature blocks
    feature_vector = np.concatenate(feature_blocks, dtype=np.float32)

    logger.info(
        "Built feature vector: shape=%s, path_dims=%d, subgraph_dims=%d, topo_dims=%d",
        feature_vector.shape,
        len(chains_agg) if use_path_features else 0,
        len(subgraph_agg),
        len(topo_agg),
    )

    return feature_vector, total_chains_extracted, max_hop_count_overall


def _block_stats(name: str, block: np.ndarray) -> Dict[str, float]:
    """Coverage of one feature block: width, share of live columns, mean row norm."""
    if block.size == 0 or block.shape[1] == 0:
        return {
            f"feat/{name}_dims": 0.0,
            f"feat/{name}_nonzero_frac": 0.0,
            f"feat/{name}_mean_norm": 0.0,
        }
    live_cols = float(np.count_nonzero(np.any(block != 0, axis=0)))
    return {
        f"feat/{name}_dims": float(block.shape[1]),
        f"feat/{name}_nonzero_frac": live_cols / block.shape[1],
        f"feat/{name}_mean_norm": float(np.mean(np.linalg.norm(block, axis=1))),
    }


def describe_feature_blocks(
    feature_matrix: np.ndarray,
    feature_mode: str,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, float]:
    """Per-block coverage of an assembled (days x dims) feature matrix.

    Each block can collapse to all zeros while the run still reports success:
    the path block when APOC or the embedding call fails, the subgraph
    histograms whenever they receive no tokens (silently -- _hashed_histogram
    returns zeros with no log line at all), and the topology block on a failed
    query. Shapes stay correct in every case, so only the CONTENT separates
    real features from a dead block. A path block zeroed exactly this way went
    undetected across an entire sweep; these numbers are what would have
    caught it, so they are logged as metrics rather than just printed.

    nonzero_frac == 0.0 for a block means that block contributed nothing.
    """
    from kg_builder_llm.ml.subgraph_features import (
        METAPATH_BUCKETS,
        NODE_TYPE_BUCKETS,
        REL_TYPE_BUCKETS,
        TEMPORAL_STATS_DIM,
    )

    if feature_matrix.ndim != 2 or feature_matrix.size == 0:
        return {}

    subgraph_dim = (
        NODE_TYPE_BUCKETS + REL_TYPE_BUCKETS + METAPATH_BUCKETS + TEMPORAL_STATS_DIM
    )

    leading = []
    if feature_mode in {"path", "hybrid"}:
        leading.append(("path", get_embedding_dim(embedding_type, local_model)))
    if feature_mode in {"subgraph", "hybrid"}:
        leading.append(("subgraph", subgraph_dim))

    stats: Dict[str, float] = {}
    start = 0
    for name, dim in leading:
        stats.update(_block_stats(name, feature_matrix[:, start:start + dim]))
        start += dim
    # Topology is whatever remains rather than a hardcoded 8, so a change to
    # its width can never silently shift the boundaries measured above.
    stats.update(_block_stats("topo", feature_matrix[:, start:]))
    return stats
