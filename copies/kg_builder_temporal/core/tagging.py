"""Tagging utilities for candidate tracking in Neo4j."""

import logging
from typing import List, Optional

from copies.kg_builder_temporal.core.graph import GraphDriver

logger = logging.getLogger(__name__)


def tag_candidate_entities(
    driver: GraphDriver,
    candidate_tag: str,
    node_labels: Optional[List[str]] = None,
) -> None:
    """Tag all nodes and edges with candidate_tag for a candidate.

    Args:
        driver: Graph driver
        candidate_tag: Tag to apply (e.g., 'step_0_candidate_0')
        node_labels: Optional list of node labels to tag. If None, tags all nodes.
    """
    logger.info("Tagging entities for candidate: %s", candidate_tag)

    # Tag nodes
    if node_labels:
        # Tag specific node labels
        for label in node_labels:
            query = f"""
            MATCH (n:{label})
            WHERE NOT n.candidate_tags IS NULL OR n.candidate_tags = []
            SET n.candidate_tags = CASE
                WHEN n.candidate_tags IS NULL THEN [$tag]
                ELSE n.candidate_tags + [$tag]
            END
            RETURN COUNT(n) as count
            """
            result = driver.run_query(query, parameters={"tag": candidate_tag})
            if result:
                count = result[0].get("count", 0)
                logger.info("Tagged %d %s nodes", count, label)
    else:
        # Tag all nodes
        query = """
        MATCH (n)
        WHERE NOT (n.candidate_tags IS NOT NULL AND $tag IN n.candidate_tags)
        SET n.candidate_tags = CASE
            WHEN n.candidate_tags IS NULL THEN [$tag]
            ELSE n.candidate_tags + [$tag]
        END
        RETURN COUNT(n) as count
        """
        result = driver.run_query(query, parameters={"tag": candidate_tag})
        if result:
            count = result[0].get("count", 0)
            logger.info("Tagged %d nodes with %s", count, candidate_tag)

    # Tag edges (relationships)
    query = """
    MATCH ()-[r]->()
    WHERE NOT (r.candidate_tags IS NOT NULL AND $tag IN r.candidate_tags)
    SET r.candidate_tags = CASE
        WHEN r.candidate_tags IS NULL THEN [$tag]
        ELSE r.candidate_tags + [$tag]
    END
    RETURN COUNT(r) as count
    """
    result = driver.run_query(query, parameters={"tag": candidate_tag})
    if result:
        count = result[0].get("count", 0)
        logger.info("Tagged %d relationships with %s", count, candidate_tag)


def tag_base_candidate(driver: GraphDriver) -> None:
    """Tag all existing nodes and edges as 'base' candidate.

    Call this after building the base/step 0 candidates.
    """
    logger.info("Tagging base candidate entities")

    tag_candidate_entities(driver, "base")
