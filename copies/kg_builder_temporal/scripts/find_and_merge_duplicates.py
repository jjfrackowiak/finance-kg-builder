#!/usr/bin/env python3
"""Merge duplicate entities in Neo4j graph based on normalized keys."""

import logging
from kg_builder_temporal.config import Config
from kg_builder_temporal.core.graph import GraphDriver
from kg_builder_temporal.core.entity_resolution import normalize_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_duplicates_by_name(driver: GraphDriver, node_label: str = "Entity") -> int:
    """
    Merge duplicate nodes by normalized name.
    
    For each group of nodes with the same normalized name:
    - Keep the node with the longest original name (most complete)
    - Merge all properties and relationships
    
    Args:
        driver: Neo4j GraphDriver
        node_label: Node label to process (default: "Entity")
    
    Returns:
        Number of duplicate groups merged
    """
    
    # Query to find duplicates and merge them
    # Using apoc.refactor.mergeNodes (requires APOC plugin)
    cypher = f"""
    MATCH (n:{node_label})
    WHERE n.name IS NOT NULL
    WITH 
        toLower(trim(n.name)) as normalized_name,
        n.name as orig_name,
        collect(n) as nodes,
        count(*) as cnt
    WHERE cnt > 1
    RETURN normalized_name, orig_name, cnt, nodes
    """
    
    results = driver.run_query(cypher)
    merged_count = 0
    
    if not results:
        logger.info("No duplicates found")
        return 0
    
    for row in results:
        normalized_name = row.get("normalized_name")
        count = row.get("cnt")
        nodes = row.get("nodes")
        
        logger.info(f"Found {count} nodes with normalized name '{normalized_name}'")
        
        # Log the variants
        variants = row.get("orig_name")
        logger.info(f"  Variants: {variants}")
        
        # Try to merge using apoc (if available)
        try:
            merge_cypher = f"""
            MATCH (n:{node_label})
            WHERE toLower(trim(n.name)) = $normalized_name
            WITH collect(n) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {{properties: "combine", relationships: "combine"}})
            YIELD node
            RETURN node
            """
            merge_results = driver.run_query(merge_cypher, parameters={"normalized_name": normalized_name})
            if merge_results:
                merged_count += 1
                logger.info(f"  ✓ Merged {count} nodes into 1")
        except Exception as e:
            logger.warning(f"  ⚠️  Could not merge (apoc.refactor not available): {e}")
            logger.info("  Install APOC plugin for automatic merging: https://neo4j.com/docs/apoc/current/")
    
    return merged_count


def find_duplicates_report(driver: GraphDriver) -> None:
    """Generate a report of all duplicate nodes in the graph."""
    
    # Find all duplicates by key
    cypher = """
    MATCH (n)
    WHERE n.key IS NOT NULL
    WITH labels(n)[0] as label, n.key as key, collect(id(n)) as ids, count(*) as cnt
    WHERE cnt > 1
    RETURN label, key, cnt, ids
    ORDER BY cnt DESC, label
    """
    
    results = driver.run_query(cypher)
    
    if not results:
        logger.info("✓ No duplicate keys found!")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("DUPLICATE NODES REPORT")
    logger.info(f"{'='*80}")
    logger.info(f"{'Label':<20} {'Key':<30} {'Count':<6} {'IDs'}")
    logger.info(f"{'-'*80}")
    
    total_duplicates = 0
    for row in results:
        label = row.get("label", "Unknown")
        key = row.get("key", "NULL")
        count = row.get("cnt", 0)
        ids = row.get("ids", [])
        
        if count > 1:
            total_duplicates += count - 1
            logger.info(f"{label:<20} {str(key):<30} {count:<6} {ids}")
    
    logger.info(f"{'-'*80}")
    logger.info(f"Total duplicate nodes: {total_duplicates}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Load config
    config = Config.from_env()
    driver = GraphDriver(config.neo4j)
    
    try:
        # Generate report
        find_duplicates_report(driver)
        
        # Optionally merge duplicates (if --merge flag provided)
        if len(sys.argv) > 1 and sys.argv[1] == "--merge":
            logger.info("Attempting to merge duplicates...")
            merged = merge_duplicates_by_name(driver)
            logger.info(f"✓ Merged {merged} groups of duplicates")
        else:
            logger.info("Run with --merge flag to attempt merging duplicates")
            logger.info("Note: Requires APOC plugin installed in Neo4j")
    
    finally:
        driver.close()
