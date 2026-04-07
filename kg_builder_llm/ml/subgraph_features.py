"""Stable temporal subgraph feature extraction.

This module builds fixed-size, hash-bucketed representations of the temporal
subgraph around entities mentioned in a lookback window. The goal is to keep
feature semantics stable across days and candidate graphs without fitting an
additional graph embedding model per candidate.
"""

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

from kg_builder_llm.core.graph import GraphDriver

logger = logging.getLogger(__name__)

# Fixed feature dimensions. These stay constant across days and candidates.
NODE_TYPE_BUCKETS = 16
REL_TYPE_BUCKETS = 24
METAPATH_BUCKETS = 48
TEMPORAL_STATS_DIM = 8

EXCLUDED_RELATION_TYPES = ["PUBLISHED_ON", "WRITTEN_BY", "MENTIONS", "MENTIONED_IN", "REFERENCES"]


def _hash_bucket(token: str, buckets: int) -> int:
    """Map a token deterministically into a fixed-size bucket space."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _hashed_histogram(tokens: List[str], buckets: int) -> np.ndarray:
    """Build a normalized hash-bucketed histogram."""
    hist = np.zeros(buckets, dtype=np.float32)
    if not tokens:
        return hist

    for token in tokens:
        hist[_hash_bucket(token, buckets)] += 1.0

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _candidate_filter_clause() -> str:
    """Cypher clause for candidate-tag-aware relationship filtering."""
    return """
      AND (
        $allowed_tags IS NULL
        OR size($allowed_tags) = 0
        OR r.candidate_tags IS NULL
        OR ANY(tag IN r.candidate_tags WHERE tag IN $allowed_tags)
      )
    """


def compute_node_type_counts(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int,
) -> np.ndarray:
    """Count node labels for entities mentioned in the lookback window."""
    query = """
    MATCH (a:Article)-[:MENTIONS]->(n)
    WHERE a.date >= date($eval_date) - duration({days: $lookback_days})
      AND a.date <= date($eval_date)
      AND (n.first_seen IS NULL OR n.first_seen <= date($eval_date))
    RETURN labels(n)[0] AS label, count(DISTINCT n) AS cnt
    """

    results = driver.run_query(
        query,
        parameters={"eval_date": eval_date, "lookback_days": lookback_days},
    )
    tokens = []
    for row in results or []:
        label = row.get("label") or "Unknown"
        cnt = int(row.get("cnt", 0))
        tokens.extend([f"node:{label}"] * cnt)

    return _hashed_histogram(tokens, NODE_TYPE_BUCKETS)


def compute_relation_type_counts(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int,
    allowed_tags: Optional[List[str]] = None,
) -> np.ndarray:
    """Count candidate-filtered relationship types near recent article mentions."""
    query = f"""
    MATCH (a:Article)-[:MENTIONS]->(n)
    WHERE a.date >= date($eval_date) - duration({{days: $lookback_days}})
      AND a.date <= date($eval_date)
      AND (n.first_seen IS NULL OR n.first_seen <= date($eval_date))
    WITH DISTINCT n
    MATCH (n)-[r]-(m)
    WHERE NOT type(r) IN $excluded_rels
      AND (m.first_seen IS NULL OR m.first_seen <= date($eval_date))
      {_candidate_filter_clause()}
    RETURN type(r) AS rel_type, count(*) AS cnt
    """

    results = driver.run_query(
        query,
        parameters={
            "eval_date": eval_date,
            "lookback_days": lookback_days,
            "allowed_tags": allowed_tags or [],
            "excluded_rels": EXCLUDED_RELATION_TYPES,
        },
    )
    tokens = []
    for row in results or []:
        rel_type = row.get("rel_type") or "UNKNOWN_REL"
        cnt = int(row.get("cnt", 0))
        tokens.extend([f"rel:{rel_type}"] * cnt)

    return _hashed_histogram(tokens, REL_TYPE_BUCKETS)


def compute_metapath_counts(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int,
    allowed_tags: Optional[List[str]] = None,
    max_hops: int = 2,
) -> np.ndarray:
    """Count typed metapaths of length 2 or 3 in the temporal subgraph."""
    tokens: List[str] = []

    if max_hops >= 2:
        query_2hop = f"""
        MATCH (a:Article)-[:MENTIONS]->(s)
        WHERE a.date >= date($eval_date) - duration({{days: $lookback_days}})
          AND a.date <= date($eval_date)
          AND (s.first_seen IS NULL OR s.first_seen <= date($eval_date))
        WITH DISTINCT s
        MATCH (s)-[r1]-(m)-[r2]-(t)
        WHERE NOT type(r1) IN $excluded_rels
          AND NOT type(r2) IN $excluded_rels
          AND (m.first_seen IS NULL OR m.first_seen <= date($eval_date))
          AND (t.first_seen IS NULL OR t.first_seen <= date($eval_date))
          {_candidate_filter_clause().replace("r.", "r1.")}
          {_candidate_filter_clause().replace("r.", "r2.")}
        RETURN labels(s)[0] AS s_label,
               type(r1) AS r1_type,
               labels(m)[0] AS m_label,
               type(r2) AS r2_type,
               labels(t)[0] AS t_label
        LIMIT 400
        """
        results = driver.run_query(
            query_2hop,
            parameters={
                "eval_date": eval_date,
                "lookback_days": lookback_days,
                "allowed_tags": allowed_tags or [],
                "excluded_rels": EXCLUDED_RELATION_TYPES,
            },
        )
        for row in results or []:
            tokens.append(
                "meta2:"
                f"{row.get('s_label', 'Unknown')}>"
                f"{row.get('r1_type', 'R1')}>"
                f"{row.get('m_label', 'Unknown')}>"
                f"{row.get('r2_type', 'R2')}>"
                f"{row.get('t_label', 'Unknown')}"
            )

    if max_hops >= 3:
        query_3hop = f"""
        MATCH (a:Article)-[:MENTIONS]->(s)
        WHERE a.date >= date($eval_date) - duration({{days: $lookback_days}})
          AND a.date <= date($eval_date)
          AND (s.first_seen IS NULL OR s.first_seen <= date($eval_date))
        WITH DISTINCT s
        MATCH (s)-[r1]-(m1)-[r2]-(m2)-[r3]-(t)
        WHERE NOT type(r1) IN $excluded_rels
          AND NOT type(r2) IN $excluded_rels
          AND NOT type(r3) IN $excluded_rels
          AND (m1.first_seen IS NULL OR m1.first_seen <= date($eval_date))
          AND (m2.first_seen IS NULL OR m2.first_seen <= date($eval_date))
          AND (t.first_seen IS NULL OR t.first_seen <= date($eval_date))
          {_candidate_filter_clause().replace("r.", "r1.")}
          {_candidate_filter_clause().replace("r.", "r2.")}
          {_candidate_filter_clause().replace("r.", "r3.")}
        RETURN labels(s)[0] AS s_label,
               type(r1) AS r1_type,
               labels(m1)[0] AS m1_label,
               type(r2) AS r2_type,
               labels(m2)[0] AS m2_label,
               type(r3) AS r3_type,
               labels(t)[0] AS t_label
        LIMIT 300
        """
        results = driver.run_query(
            query_3hop,
            parameters={
                "eval_date": eval_date,
                "lookback_days": lookback_days,
                "allowed_tags": allowed_tags or [],
                "excluded_rels": EXCLUDED_RELATION_TYPES,
            },
        )
        for row in results or []:
            tokens.append(
                "meta3:"
                f"{row.get('s_label', 'Unknown')}>"
                f"{row.get('r1_type', 'R1')}>"
                f"{row.get('m1_label', 'Unknown')}>"
                f"{row.get('r2_type', 'R2')}>"
                f"{row.get('m2_label', 'Unknown')}>"
                f"{row.get('r3_type', 'R3')}>"
                f"{row.get('t_label', 'Unknown')}"
            )

    return _hashed_histogram(tokens, METAPATH_BUCKETS)


def compute_temporal_novelty_features(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int,
) -> np.ndarray:
    """Compute stable temporal summary statistics for the lookback window."""
    query = """
    MATCH (a:Article)
    WHERE a.date >= date($eval_date) - duration({days: $lookback_days})
      AND a.date <= date($eval_date)
    OPTIONAL MATCH (a)-[:MENTIONS]->(n)
    WITH collect(DISTINCT a) AS articles, collect(DISTINCT n) AS nodes
    WITH articles,
         [n IN nodes WHERE n IS NOT NULL] AS valid_nodes,
         date($eval_date) - duration({days: $lookback_days}) AS start_date
    RETURN size(articles) AS article_count,
           size(valid_nodes) AS mentioned_entity_count,
           size([n IN valid_nodes WHERE n.first_seen IS NOT NULL AND n.first_seen >= start_date]) AS new_entity_count,
           size([n IN valid_nodes WHERE n.first_seen IS NOT NULL AND n.first_seen < start_date]) AS recurring_entity_count,
           size([n IN valid_nodes WHERE n.first_seen IS NULL]) AS undated_entity_count
    """
    results = driver.run_query(
        query,
        parameters={"eval_date": eval_date, "lookback_days": lookback_days},
    )

    if not results:
        return np.zeros(TEMPORAL_STATS_DIM, dtype=np.float32)

    row = results[0]
    article_count = float(row.get("article_count", 0))
    mentioned_entity_count = float(row.get("mentioned_entity_count", 0))
    new_entity_count = float(row.get("new_entity_count", 0))
    recurring_entity_count = float(row.get("recurring_entity_count", 0))
    undated_entity_count = float(row.get("undated_entity_count", 0))

    novelty_ratio = new_entity_count / mentioned_entity_count if mentioned_entity_count else 0.0
    recurring_ratio = recurring_entity_count / mentioned_entity_count if mentioned_entity_count else 0.0
    entities_per_article = mentioned_entity_count / article_count if article_count else 0.0

    return np.array(
        [
            article_count,
            mentioned_entity_count,
            new_entity_count,
            recurring_entity_count,
            undated_entity_count,
            novelty_ratio,
            recurring_ratio,
            entities_per_article,
        ],
        dtype=np.float32,
    )


def build_temporal_subgraph_feature_vector(
    driver: GraphDriver,
    eval_date: str,
    lookback_days: int = 2,
    allowed_tags: Optional[List[str]] = None,
    max_metapath_hops: int = 2,
) -> np.ndarray:
    """Build the fixed-schema temporal subgraph feature block."""
    node_type_counts = compute_node_type_counts(driver, eval_date, lookback_days)
    relation_type_counts = compute_relation_type_counts(
        driver,
        eval_date,
        lookback_days,
        allowed_tags=allowed_tags,
    )
    metapath_counts = compute_metapath_counts(
        driver,
        eval_date,
        lookback_days,
        allowed_tags=allowed_tags,
        max_hops=max_metapath_hops,
    )
    temporal_stats = compute_temporal_novelty_features(driver, eval_date, lookback_days)

    feature_block = np.concatenate(
        [node_type_counts, relation_type_counts, metapath_counts, temporal_stats],
        dtype=np.float32,
    )
    logger.info(
        "Built temporal subgraph features: shape=%s (node=%d, rel=%d, meta=%d, temporal=%d)",
        feature_block.shape,
        len(node_type_counts),
        len(relation_type_counts),
        len(metapath_counts),
        len(temporal_stats),
    )
    return feature_block
