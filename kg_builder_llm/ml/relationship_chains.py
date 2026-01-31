"""Relationship chain extraction and embedding for candidate evaluation.

Strategy:
---------
For each article, extract multi-hop relationship chains among mentioned entities
that use relationships from the evaluated candidate. Each chain is converted to
a text representation and embedded, creating a rich semantic signal about the
ontology's quality.

Example:
    Article mentions: Tesla, EV, California
    Candidate adds relationships: PRODUCES, HEADQUARTERED_IN
    
    Extracted chains (top 10 by hop count):
    1. "Tesla -[PRODUCES]-> EV -[HAS_TYPE]-> Electric" (2 hops)
    2. "Tesla -[PRODUCES]-> EV" (1 hop)
    3. "Tesla -[HEADQUARTERED_IN]-> California" (1 hop)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.ml.embeddings import embed_text_deterministic, get_embedding_dim

logger = logging.getLogger(__name__)


def get_entities_mentioned_in_article(
    driver: GraphDriver,
    article_id: str,
    allowed_tags: List[str],
    eval_date: Optional[str] = None,
) -> List[Dict]:
    """Extract entities mentioned in article AND connected via allowed_tags relationships.
    
    Filters to only entities that:
    1. Are mentioned in the article (MENTIONED_IN/REFERENCES)
    2. Have connections to other entities via allowed_tags relationships
    3. If eval_date provided: all nodes must be mentioned in articles <= eval_date (prevents data leakage)
    
    Args:
        driver: GraphDriver instance
        article_id: Element ID of the article
        allowed_tags: Tags to filter relationships (e.g., ['base_structure', 'step_1_candidate_0'])
        eval_date: Optional evaluation date (YYYY-MM-DD). If provided, only includes entities mentioned
                  in articles published on or before this date. Prevents future data leakage.
    
    Returns:
        List of dicts with keys: 'id', 'name', 'label'
    """
    tags_param = "', '".join(allowed_tags) if allowed_tags else ""
    
    if eval_date:
        query = f"""
        MATCH (a:Article)-[mention_rel:MENTIONS]-(n)
        WHERE elementId(a) = $article_id AND a.date <= date($eval_date)
        WITH DISTINCT n
        MATCH (n)-[r]-(other_node)
        WHERE (r.candidate_tags IS NULL 
           OR ANY(tag IN r.candidate_tags WHERE tag IN ['{tags_param}']))
          AND ALL(article IN [a] WHERE EXISTS((other_node)-[:MENTIONS]-(a2:Article) WHERE a2.date <= date($eval_date)) OR other_node = n)
        RETURN DISTINCT elementId(n) as node_id, n.name, labels(n)[0] as label
        """
        params = {"article_id": article_id, "eval_date": eval_date}
    else:
        query = f"""
        MATCH (a:Article)-[mention_rel:MENTIONS]-(n)
        WHERE elementId(a) = $article_id
        WITH DISTINCT n
        MATCH (n)-[r]-(other_node)
        WHERE r.candidate_tags IS NULL 
           OR ANY(tag IN r.candidate_tags WHERE tag IN ['{tags_param}'])
        RETURN DISTINCT elementId(n) as node_id, n.name, labels(n)[0] as label
        """
        params = {"article_id": article_id}
    
    try:
        results = driver.run_query(query, parameters=params)
        entities = []
        if results:
            for row in results:
                if row.get("name"):
                    entities.append({
                        "id": row.get("node_id"),
                        "name": str(row.get("name")),
                        "label": row.get("label", "Entity"),
                    })
        logger.debug(
            "Found %d entities in article %s connected via allowed_tags",
            len(entities),
            article_id[:8],
        )
        return entities
    except Exception as e:
        logger.warning("Error fetching entities from article: %s", str(e))
        return []


def extract_top_chains_from_article(
    driver: GraphDriver,
    article_id: str,
    allowed_tags: List[str],
    max_hops: int = 10,
    eval_date: Optional[str] = None,
) -> List[Dict]:
    """Extract top 10 relationship chains from article entities using allowed_tags.
    
    Process:
    1. Get entities mentioned in article (filtered by allowed_tags)
    2. Find all paths between entity pairs using allowed_tags relationships
    3. All nodes in paths must be from articles <= eval_date (if eval_date provided, prevents data leakage)
    4. Sort by hop count descending (longer chains first)
    5. Keep top 10 chains
    6. Convert to text representation
    
    Args:
        driver: GraphDriver instance
        article_id: Element ID of article
        allowed_tags: List of tags to filter relationships
        max_hops: Maximum relationship chain length (default 10)
        eval_date: Optional evaluation date (YYYY-MM-DD). If provided, ensures all nodes in extracted
                  paths are from articles published on or before this date. Prevents future data leakage.
    
    Returns:
        List of max 10 chain dicts with keys:
        - 'path': List of (node_id, node_name) tuples
        - 'rel_types': List of relationship type names
        - 'hop_count': Number of hops in chain
        - 'chain_text': Human-readable string representation
    """
    # Step 1: Query for all paths between entity pairs
    # Entities are fetched and filtered within the query itself
    logger.debug("extract_top_chains_from_article: Starting chain extraction for article_id=%s, eval_date=%s", 
                 article_id[:50], eval_date)
    logger.debug("Allowed tags: %s", allowed_tags)
    
    # Excluded relationship types (uninformative for chains)
    excluded_rels = ['PUBLISHED_ON', 'WRITTEN_BY', 'MENTIONS', 'MENTIONED_IN', 'REFERENCES']
    
    # OPTIMIZED QUERY using APOC path expansion
    # - Samples 10 starting nodes (deterministic via seed)
    # - Uses APOC uniqueness: "NODE_PATH" (no loops)
    # - Directed paths only (relationshipFilter: ">")
    # - Searches paths from max_hops-4 to max_hops+4 for variety
    min_level = max(1, max_hops - 4)
    max_level = max_hops
    
    query = f"""
    WITH
      date($eval_date) AS eval_date,
      $excluded_rels AS excluded_rels,
      'seed-001' AS seed
    
    MATCH (a:Article)-[:MENTIONS]->(s)
    WHERE elementId(a) = $article_id
      AND a.date <= eval_date
      AND (s.candidate_tags IS NULL OR ANY(t IN s.candidate_tags WHERE t IN $allowed_tags))
      AND s.first_seen IS NOT NULL AND s.first_seen <= eval_date
    WITH collect(DISTINCT s) AS starts, excluded_rels, eval_date, seed
    WHERE size(starts) > 0
    
    UNWIND starts AS s
    WITH s, excluded_rels, eval_date,
         apoc.util.md5([seed, elementId(s)]) AS k
    ORDER BY k
    WITH collect(s)[0..10] AS sampled_starts, excluded_rels, eval_date
    
    UNWIND sampled_starts AS s
    CALL apoc.path.expandConfig(s, {{
      minLevel: {min_level},
      maxLevel: {max_level},
      uniqueness: "NODE_PATH",
      labelFilter: "-Article",
      relationshipFilter: ">"
    }}) YIELD path
    
    WHERE NONE(r IN relationships(path) WHERE type(r) IN excluded_rels)
      AND ALL(n IN nodes(path) WHERE
        (n.candidate_tags IS NULL OR ANY(t IN n.candidate_tags WHERE t IN $allowed_tags))
        AND n.first_seen IS NOT NULL AND n.first_seen <= eval_date
      )
    
    WITH path, length(path) AS hop_count
    RETURN
      [n IN nodes(path) | {{id: elementId(n), name: COALESCE(n.name, n.key)}}] AS node_list,
      [r IN relationships(path) | type(r)] AS relationship_types,
      hop_count
    ORDER BY hop_count DESC
    LIMIT 50
    """
    logger.debug("About to execute query:\n%s", query)
    logger.debug("With parameters: article_id=%s, eval_date=%s, allowed_tags=%s, excluded_rels=%s", 
                 article_id[:50], eval_date, allowed_tags, excluded_rels)
    
    try:
        results = driver.run_query(query, parameters={
            "article_id": article_id, 
            "eval_date": eval_date,
            "allowed_tags": allowed_tags,
            "excluded_rels": excluded_rels
        })
        logger.debug("Query executed with article_id=%s, eval_date=%s", article_id[:50], eval_date)
        logger.debug("Query returned %d results", len(results) if results else 0)
    except Exception as e:
        logger.warning("Error querying paths: %s", str(e))
        logger.debug("Failed query: %s", query)
        return []
    
    if not results:
        logger.debug("No paths found between entities for article %s", article_id[:8])
        return []
    
    # Step 3: Convert results to chain dicts and sort
    all_chains = []
    for row in results:
        node_list = row.get("node_list", [])
        rel_types = row.get("relationship_types", [])
        hop_count = row.get("hop_count", 0)
        
        if node_list and rel_types and len(node_list) > 1:
            chain_text = _build_chain_text(node_list, rel_types)
            all_chains.append({
                "path": [(n.get("id"), n.get("name")) for n in node_list],
                "rel_types": rel_types,
                "hop_count": hop_count,
                "chain_text": chain_text,
            })
    
    logger.info("Article %s: Found %d total paths, sorting by hop count", article_id[:8], len(all_chains))
    
    # Step 4: Sort by hop count descending (longer chains first)
    sorted_chains = sorted(all_chains, key=lambda x: x["hop_count"], reverse=True)
    
    # Step 5: Keep top 10
    top_10 = sorted_chains[:10]
    
    # Log top chains
    if top_10:
        logger.info("Article %s: Top 10 chains by hop count:", article_id[:8])
        for i, chain in enumerate(top_10, 1):
            logger.info(
                "  [%d] %d hops: %s",
                i,
                chain["hop_count"],
                chain["chain_text"][:100] + ("..." if len(chain["chain_text"]) > 100 else ""),
            )
    else:
        logger.debug("Article %s: No chains extracted", article_id[:8])
    
    logger.info("Article %s: Extracted and sorted top %d chains", article_id[:8], len(top_10))
    
    return top_10


def _build_chain_text(node_list: List[Dict], relationship_types: List[str]) -> str:
    """Build human-readable text representation of a relationship chain.
    
    Example:
        node_list = [{'name': 'Tesla'}, {'name': 'EV'}, {'name': 'Electric'}]
        rel_types = ['PRODUCES', 'IS_TYPE']
        Result: "Tesla -[PRODUCES]-> EV -[IS_TYPE]-> Electric"
    
    Args:
        node_list: List of node dicts with 'name' key
        relationship_types: List of relationship type names
    
    Returns:
        Chain text string
    """
    if not node_list or len(node_list) < 2:
        return ""
    
    node_names = [str(n.get("name", "Unknown")) for n in node_list]
    
    chain_parts = [node_names[0]]
    for i, rel_type in enumerate(relationship_types):
        chain_parts.append(f"-[{rel_type}]->")
        if i + 1 < len(node_names):
            chain_parts.append(node_names[i + 1])
    
    return " ".join(chain_parts)


async def embed_relationship_chains(
    chains: List[Dict],
    api_key: Optional[str] = None,
    semaphore: Optional[object] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """Embed relationship chain text representations.
    
    For local embeddings: Uses batch encoding (all at once) for speed.
    For OpenAI embeddings: Uses async with semaphore for rate limiting.
    
    Args:
        chains: List of chain dicts from extract_top_chains_from_article()
        api_key: OpenAI API key (required for embedding_type="openai")
        semaphore: Optional asyncio.Semaphore to limit concurrent API calls
        embedding_type: "local" (batch encoding) or "openai" (async)
        local_model: Sentence-transformers model name (for local only)
    
    Returns:
        List of dicts with keys:
        - 'chain_text': Original chain text
        - 'embedding': np.ndarray (384 dims for local, 1536 for openai)
        - 'hop_count': Number of hops
    """
    import asyncio
    from kg_builder_llm.ml.embeddings import embed_text_local, get_local_embedder
    
    embedded_chains = []
    logger.info("Embedding %d chains using %s...", len(chains), embedding_type)
    
    # Filter out empty chains
    valid_chains = [(i, chain) for i, chain in enumerate(chains, 1) 
                    if chain.get("chain_text", "").strip()]
    
    if not valid_chains:
        logger.warning("No valid chains to embed")
        return []
    
    if embedding_type == "local":
        # LOCAL: Use batch encoding for efficiency (much faster than one-by-one)
        try:
            chain_texts = [chain.get("chain_text") for _, chain in valid_chains]
            
            # Batch encode all texts at once
            model = get_local_embedder(local_model)
            logger.debug("Batch encoding %d chains with local model...", len(chain_texts))
            embeddings = model.encode(chain_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
            
            # Build result list
            for (idx, chain), embedding in zip(valid_chains, embeddings):
                embedded_chains.append({
                    "chain_text": chain.get("chain_text"),
                    "embedding": embedding.astype(np.float32),
                    "hop_count": chain.get("hop_count", 0),
                })
                logger.debug(
                    "Chain %d embedded: %d hops, %s",
                    idx,
                    chain.get("hop_count", 0),
                    chain.get("chain_text", "")[:60] + "..." if len(chain.get("chain_text", "")) > 60 else chain.get("chain_text", ""),
                )
            
            logger.info("✓ Batch encoded %d chains with local model", len(embedded_chains))
            
        except Exception as e:
            logger.error("Failed to batch encode chains: %s", str(e))
            return []
    
    else:
        # OPENAI: Use async with semaphore for rate limiting
        async def embed_single_chain(idx: int, chain: Dict):
            chain_text = chain.get("chain_text", "")
            
            if semaphore:
                async with semaphore:
                    try:
                        loop = asyncio.get_event_loop()
                        embedding = await loop.run_in_executor(
                            None, embed_text_deterministic, chain_text, api_key, "text-embedding-3-small", "openai", local_model
                        )
                        logger.debug(
                            "Chain %d/%d embedded: %d hops, %s",
                            idx,
                            len(chains),
                            chain.get("hop_count", 0),
                            chain_text[:60] + "..." if len(chain_text) > 60 else chain_text,
                        )
                        return {
                            "chain_text": chain_text,
                            "embedding": embedding,
                            "hop_count": chain.get("hop_count", 0),
                        }
                    except Exception as e:
                        logger.warning(
                            "Failed to embed chain %d '%s': %s",
                            idx,
                            chain_text[:50],
                            str(e),
                        )
                        return None
            else:
                try:
                    loop = asyncio.get_event_loop()
                    embedding = await loop.run_in_executor(
                        None, embed_text_deterministic, chain_text, api_key, "text-embedding-3-small", "openai", local_model
                    )
                    logger.debug(
                        "Chain %d/%d embedded: %d hops, %s",
                        idx,
                        len(chains),
                        chain.get("hop_count", 0),
                        chain_text[:60] + "..." if len(chain_text) > 60 else chain_text,
                    )
                    return {
                        "chain_text": chain_text,
                        "embedding": embedding,
                        "hop_count": chain.get("hop_count", 0),
                    }
                except Exception as e:
                    logger.warning(
                        "Failed to embed chain %d '%s': %s",
                        idx,
                        chain_text[:50],
                        str(e),
                    )
                    return None
        
        # Embed all chains concurrently
        tasks = [embed_single_chain(idx, chain) for idx, chain in valid_chains]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        embedded_chains = [r for r in results if r is not None]
        
        logger.info("Successfully embedded %d/%d chains with OpenAI", len(embedded_chains), len(chains))
    
    return embedded_chains


def aggregate_chain_embeddings(
    embedded_chains: List[Dict],
    method: str = "mean",
) -> np.ndarray:
    """Aggregate multiple chain embeddings into a single feature vector.
    
    Args:
        embedded_chains: List of dicts with 'embedding' key (from embed_relationship_chains)
        method: Aggregation method: 'mean', 'max', or 'weighted_mean'
    
    Returns:
        Aggregated embedding vector (shape: (1536,))
    """
    if not embedded_chains:
        logger.debug("No chains to aggregate, returning zero vector")
        return np.zeros(1536, dtype=np.float32)
    
    embeddings = np.array([c["embedding"] for c in embedded_chains], dtype=np.float32)
    logger.info("Aggregating %d chain embeddings using '%s' method", len(embeddings), method)
    
    if method == "mean":
        result = np.mean(embeddings, axis=0)
    elif method == "max":
        result = np.max(embeddings, axis=0)
    elif method == "weighted_mean":
        # Weight by hop count (longer chains = higher weight)
        hop_counts = np.array([c.get("hop_count", 1) for c in embedded_chains], dtype=np.float32)
        weights = hop_counts / hop_counts.sum() if hop_counts.sum() > 0 else np.ones_like(hop_counts) / len(hop_counts)
        result = np.average(embeddings, axis=0, weights=weights)
        logger.debug("Weighted by hop counts: %s", hop_counts)
    else:
        logger.warning("Unknown aggregation method: %s, using mean", method)
        result = np.mean(embeddings, axis=0)
    
    logger.info(
        "Aggregated embedding shape: %s, mean magnitude: %.4f",
        result.shape,
        np.linalg.norm(result),
    )
    return result


async def extract_and_embed_chains_for_article(
    driver: GraphDriver,
    article_id: str,
    allowed_tags: List[str],
    max_hops: int = 9,
    aggregation_method: str = "mean",
    api_key: Optional[str] = None,
    eval_date: Optional[str] = None,
    semaphore: Optional[object] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, int]:
    """End-to-end pipeline: extract chains from article → embed → aggregate (async).
    
    Process:
    1. Extract top 10 chains from article entities (filtered by allowed_tags)
    2. Ensure all nodes in chains are from articles <= eval_date (if provided, prevents data leakage)
    3. Embed each chain text using OpenAI embeddings (async with semaphore)
    4. Aggregate embeddings (mean/max/weighted_mean)
    
    Args:
        driver: GraphDriver instance
        article_id: Element ID of article
        allowed_tags: Candidate tags to filter relationships
        max_hops: Maximum relationship chain length (default 10)
        aggregation_method: How to aggregate embeddings ('mean', 'max', 'weighted_mean')
        api_key: OpenAI API key for embeddings
        eval_date: Optional evaluation date (YYYY-MM-DD). If provided, ensures all chain nodes are from
                  articles published on or before this date. Prevents future data leakage.
        semaphore: Optional asyncio.Semaphore for rate limiting concurrent API calls
    
    Returns:
        Tuple of (aggregated_embedding, num_chains_extracted)
        - aggregated_embedding: np.ndarray of shape (1536,)
        - num_chains_extracted: int, number of chains found and embedded
    """
    logger.info("Processing article %s: extracting and embedding chains...", article_id[:8])
    logger.debug("extract_and_embed_chains_for_article called with: article_id=%s, allowed_tags=%s, max_hops=%d, eval_date=%s", 
                 article_id[:50], allowed_tags, max_hops, eval_date)
    
    # Step 1: Extract chains
    logger.debug("About to call extract_top_chains_from_article...")
    chains = extract_top_chains_from_article(driver, article_id, allowed_tags, max_hops=max_hops, eval_date=eval_date)
    logger.debug("extract_top_chains_from_article returned %d chains", len(chains) if chains else 0)
    
    if not chains:
        logger.warning("Article %s: No chains extracted", article_id[:8])
        embed_dim = get_embedding_dim(embedding_type, local_model)
        return np.zeros(embed_dim, dtype=np.float32), 0, 0
    
    logger.info("Article %s: Extracted %d chains, now embedding...", article_id[:8], len(chains))
    
    # Step 2: Embed chains (async)
    embedded_chains = await embed_relationship_chains(
        chains, 
        api_key=api_key, 
        semaphore=semaphore,
        embedding_type=embedding_type,
        local_model=local_model,
    )
    
    if not embedded_chains:
        logger.warning("Article %s: Failed to embed any chains", article_id[:8])
        return np.zeros(embed_dim, dtype=np.float32), 0, 0
    
    logger.info("Article %s: Embedded %d chains, aggregating...", article_id[:8], len(embedded_chains))
    
    # Step 3: Aggregate
    aggregated_emb = aggregate_chain_embeddings(embedded_chains, method=aggregation_method)
    
    # Get max hop count
    max_hop_count = max(chain.get("hop_count", 0) for chain in embedded_chains)
    
    logger.info(
        "Article %s: Complete - %d chains extracted, %d embedded, max_hops=%d, aggregated to shape %s",
        article_id[:8],
        len(chains),
        len(embedded_chains),
        max_hop_count,
        aggregated_emb.shape,
    )
    
    return aggregated_emb, len(embedded_chains), max_hop_count
