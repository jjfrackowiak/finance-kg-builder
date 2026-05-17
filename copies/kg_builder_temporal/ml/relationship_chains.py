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

from kg_builder_temporal.core.graph import GraphDriver
from kg_builder_temporal.ml.embeddings import embed_text_deterministic, get_embedding_dim

logger = logging.getLogger(__name__)


def extract_chains_batch(
    driver: GraphDriver,
    jobs: List[Dict]
) -> Dict[int, List[Dict]]:
    """
    Extract relationship chains for multiple articles in a single batch query.
    
    Much more efficient than individual queries - processes all articles in one Neo4j roundtrip.
    
    Args:
        driver: Neo4j driver instance
        jobs: List of job dictionaries, each containing:
            - job_id: Unique identifier (int)
            - article_id: Article element ID (str)
            - eval_date: ISO date string (str)
            - allowed_tags: List of candidate tags (list)
            - excluded_rels: List of relationship types to exclude (list)
            - min_level: Minimum path length (int)
            - max_level: Maximum path length (int)
    
    Returns:
        Dict mapping job_id -> list of chain dicts, each containing:
        - 'hop_count': Number of hops in chain
        - 'nodes': List of node names
        - 'rels': List of relationship type names
        - 'chain_text': Human-readable string representation
    """
    if not jobs:
        return {}
    
    logger.info("=" * 80)
    logger.info("BATCH CHAIN EXTRACTION: Processing %d articles", len(jobs))
    logger.info("=" * 80)
    
    query = """
    WITH $jobs AS jobs
    
    UNWIND jobs AS job
    WITH job,
         date(job.eval_date) AS eval_date,
         job.allowed_tags AS allowed_tags,
         job.excluded_rels AS excluded_rels
    
    MATCH (a:Article)-[:MENTIONS]->(s)
    WHERE elementId(a) = job.article_id
      AND a.date <= eval_date
    WITH job, eval_date, allowed_tags, excluded_rels,
         collect(DISTINCT s)[0..10] AS sampled_starts
    
    UNWIND sampled_starts AS s
    
    CALL (s, job, eval_date, allowed_tags, excluded_rels) {
      CALL apoc.path.expandConfig(s, {
        minLevel: job.min_level,
        maxLevel: job.max_level,
        uniqueness: "NODE_PATH",
        labelFilter: "-Article|-Day",
        relationshipFilter: ">",
        limit: 30
      }) YIELD path
    
      WITH path, eval_date, allowed_tags, excluded_rels
      LIMIT 200
    
      WHERE NONE(r IN relationships(path) WHERE type(r) IN excluded_rels)
        AND ALL(n IN nodes(path) WHERE
          (n.candidate_tags IS NULL OR ANY(t IN n.candidate_tags WHERE t IN allowed_tags))
          AND n.first_seen <= eval_date
        )
    
      WITH path, length(path) AS hop_count
      ORDER BY hop_count DESC
      LIMIT 20
    
      RETURN path, hop_count
    }
    
    WITH job,
         collect({
           hop_count: hop_count,
           nodes: [n IN nodes(path) | coalesce(n.name, n.key)],
           rels:  [r IN relationships(path) | type(r)]
         })[0..10] AS paths
    
    RETURN job.job_id AS job_id, job.article_id AS article_id, paths
    """
    
    logger.info("Executing batch query with %d jobs...", len(jobs))
    logger.debug("Query:\n%s", query)
    
    try:
        results = driver.run_query(query, parameters={"jobs": jobs})
        logger.info("✓ Batch query executed successfully, got %d results", len(results))
    except Exception as e:
        logger.error("✗ Batch query failed: %s", str(e))
        return {}
    
    # Process results into job_id -> chains mapping
    chains_by_job = {}
    for row in results:
        job_id = row.get("job_id")
        article_id = row.get("article_id", "")
        paths = row.get("paths", [])
        
        # Convert to chain dicts with chain_text
        chains = []
        for path_data in paths:
            nodes = path_data.get("nodes", [])
            rels = path_data.get("rels", [])
            hop_count = path_data.get("hop_count", 0)
            
            if nodes and rels and len(nodes) > 1:
                # Build chain text: "node1 -[REL1]-> node2 -[REL2]-> node3"
                chain_parts = []
                for i, node in enumerate(nodes):
                    chain_parts.append(str(node))
                    if i < len(rels):
                        chain_parts.append(f" -[{rels[i]}]-> ")
                chain_text = "".join(chain_parts)
                
                chains.append({
                    "nodes": nodes,
                    "rel_types": rels,
                    "hop_count": hop_count,
                    "chain_text": chain_text
                })
        
        chains_by_job[job_id] = chains
        logger.info("  Job %d (%s...): Extracted %d chains", job_id, article_id[:8], len(chains))
    
    logger.info("=" * 80)
    logger.info("BATCH EXTRACTION COMPLETE: Processed %d jobs", len(chains_by_job))
    logger.info("=" * 80)
    
    return chains_by_job


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


def embed_relationship_chains(
    chains: List[Dict],
    api_key: Optional[str] = None,
    embedding_type: str = "local",
    local_model: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """Embed relationship chain text representations.
    
    For local embeddings: Uses batch encoding (all at once) for speed.
    For OpenAI embeddings: Sequential processing (async deprecated).
    
    Args:
        chains: List of chain dicts from extract_top_chains_from_article()
        api_key: OpenAI API key (required for embedding_type="openai")
        embedding_type: "local" (batch encoding) or "openai" (sequential)
        local_model: Sentence-transformers model name (for local only)
    
    Returns:
        List of dicts with keys:
        - 'chain_text': Original chain text
        - 'embedding': np.ndarray (384 dims for local, 1536 for openai)
        - 'hop_count': Number of hops
    """
    from kg_builder_temporal.ml.embeddings import embed_text_local, get_local_embedder
    
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
                embedded_chain = {
                    "chain_text": chain.get("chain_text"),
                    "embedding": embedding.astype(np.float32),
                    "hop_count": chain.get("hop_count") or 0,
                }
                # Preserve _article_idx if present (needed for grouping later)
                if "_article_idx" in chain:
                    embedded_chain["_article_idx"] = chain["_article_idx"]
                embedded_chains.append(embedded_chain)
                logger.debug(
                    "Chain %d embedded: %d hops, %s",
                    idx,
                    chain.get("hop_count") or 0,
                    chain.get("chain_text", "")[:60] + "..." if len(chain.get("chain_text", "")) > 60 else chain.get("chain_text", ""),
                )
            
            logger.info("✓ Batch encoded %d chains with local model", len(embedded_chains))
            
        except Exception as e:
            logger.error("Failed to batch encode chains: %s", str(e))
            return []
    
    else:
        # OPENAI: Sequential processing (no longer async)
        for idx, chain in valid_chains:
            chain_text = chain.get("chain_text", "")
            
            try:
                embedding = embed_text_deterministic(
                    chain_text, api_key, "text-embedding-3-small", "openai", local_model
                )
                embedded_chain = {
                    "chain_text": chain_text,
                    "embedding": embedding,
                    "hop_count": chain.get("hop_count") or 0,
                }
                # Preserve _article_idx if present (needed for grouping later)
                if "_article_idx" in chain:
                    embedded_chain["_article_idx"] = chain["_article_idx"]
                embedded_chains.append(embedded_chain)
                logger.debug(
                    "Chain %d/%d embedded: %d hops, %s",
                    idx,
                    len(valid_chains),
                    chain.get("hop_count") or 0,
                    chain_text[:60] + "..." if len(chain_text) > 60 else chain_text,
                )
            except Exception as e:
                logger.warning(
                    "Failed to embed chain %d '%s': %s",
                    idx,
                    chain_text[:50],
                    str(e),
                )
        
        logger.info("✓ Embedded %d chains with OpenAI", len(embedded_chains))
    
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
