"""Feature engineering for text embeddings + topology approach."""

import asyncio
import logging
from typing import List, Optional

import numpy as np

from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.ml.embeddings import embed_text_deterministic, get_embedding_dim
from kg_builder_llm.ml.relationship_chains import (
    aggregate_chain_embeddings,
    embed_relationship_chains,
    extract_and_embed_chains_for_article,
)
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


async def build_day_feature_vector(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int = 2,
    text_agg_method: str = "mean",
    api_key: Optional[str] = None,
    allowed_tags: Optional[List[str]] = None,
    chain_agg_method: str = "mean",
    semaphore: Optional[asyncio.Semaphore] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Build combined feature vector for a prediction day (chains + topology only).
    
    Process:
    1. Get articles from [eval_date - lookback_days, eval_date]
    2. Extract and embed relationship chains per article (with temporal constraint, async)
    3. Aggregate chain embeddings across articles → chains_agg (1536)
    4. Compute topology features for entities mentioned in window
    5. Aggregate topology features → topo_agg (8)
    6. Concatenate: [chains_agg (1536) | topo_agg (8)]
    
    Args:
        driver: GraphDriver instance
        eval_date: Prediction date (YYYY-MM-DD)
        lookback_days: Days of history to include
        text_agg_method: (unused - kept for compatibility)
        api_key: OpenAI API key (for chain embedding)
        allowed_tags: Tags to filter relationships for chain extraction
        chain_agg_method: How to aggregate chain embeddings ('mean', 'max', 'weighted_mean')
        semaphore: Optional asyncio.Semaphore for rate limiting concurrent API calls
    
    Returns:
        Tuple of (feature_vector, total_chains, max_hop_count):
        - feature_vector: np.ndarray (shape depends on embedding type + 8 topology)
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
        return np.zeros(embed_dim + 8, dtype=np.float32), 0, 0  # embed_dim chains + 8 topology (no text), 0 chains, 0 max_hops
    
    # Skip text embedding - we only want chain embeddings
    logger.info("Skipping text embeddings - using only relationship chains")
    text_agg = None
    
    chains_embeddings = []
    total_chains_extracted = 0
    total_chains_embedded = 0
    max_hop_count_overall = 0  # Track longest path across all articles
    
    logger.info("About to check: if allowed_tags=%s", bool(allowed_tags))
    if allowed_tags:
        logger.info("✓ allowed_tags is truthy, entering chain extraction block")
        logger.info("Processing %d articles for relationship chain extraction...", len(articles))
        
        # Create tasks for all articles
        tasks = []
        for article_idx, article in enumerate(articles, 1):
            article_id = article.get("id", "")
            article_date = article.get("date", "")
            # Convert date object to string format YYYY-MM-DD if needed
            if hasattr(article_date, 'isoformat'):
                article_date = article_date.isoformat()
            elif article_date and not isinstance(article_date, str):
                article_date = str(article_date)
            
            logger.info("-" * 80)
            logger.info("Article %d/%d: %s (date=%s)", article_idx, len(articles), article_id[:8], article_date)
            logger.info("-" * 80)
            
            task = extract_and_embed_chains_for_article(
                driver,
                article_id,
                allowed_tags,
                max_hops=9,
                aggregation_method=chain_agg_method,
                api_key=api_key,
                eval_date=article_date,
                semaphore=semaphore,
                embedding_type=embedding_type,
                local_model=local_model,
            )
            tasks.append((article_idx, task))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for (article_idx, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error("✗ Article %d failed: %s", article_idx, str(result))
            else:
                chain_emb, num_chains, max_hops = result
                if num_chains > 0:
                    chains_embeddings.append(chain_emb)
                    total_chains_extracted += num_chains
                    total_chains_embedded += num_chains
                    max_hop_count_overall = max(max_hop_count_overall, max_hops)
                    logger.info("✓ Article %d: Added embedding to pool (max_hops=%d)", article_idx, max_hops)
                else:
                    logger.warning("✗ Article %d: No chains found", article_idx)
        
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
        logger.warning("⚠ allowed_tags not provided, skipping chain extraction")
        chains_agg = np.zeros(embed_dim, dtype=np.float32)
    
    # Step 6: Compute topology features
    topo_features_dict = compute_topology_features(driver, eval_date, lookback_days)
    
    # Step 7: Aggregate topology features
    topo_agg = aggregate_topology_features(topo_features_dict, agg_method="mean")
    logger.debug("Topology aggregate shape: %s", topo_agg.shape)
    
    # Step 8: Concatenate (chains + topology only, no text)
    feature_vector = np.concatenate([chains_agg, topo_agg], dtype=np.float32)
    
    logger.info("Built feature vector: shape=%s, chains_dims=%d, topo_dims=%d",
               feature_vector.shape, len(chains_agg), len(topo_agg))
    
    return feature_vector, total_chains_extracted, max_hop_count_overall
