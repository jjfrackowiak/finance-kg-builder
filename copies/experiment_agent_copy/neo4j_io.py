# file: finance_kg_experiment/neo4j_io.py
from __future__ import annotations
from typing import List, Dict, Tuple
import logging
from neo4j import GraphDatabase
import pandas as pd
import asyncio

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from copies.experiment_agent_copy.config import Neo4jConfig, ExperimentConfig
from copies.experiment_agent_copy.ontology import OntologyCandidate

logger = logging.getLogger(__name__)


def get_driver(cfg: Neo4jConfig):
    logger.info("Connecting to Neo4j at %s", cfg.uri)
    return GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))


def clear_graph(driver, database: str):
    logger.info("Clearing all nodes and relationships from database '%s'", database)
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.debug("Graph cleared")

async def build_kg_for_candidate(
    driver,
    neo4j_cfg: Neo4jConfig,
    exp_cfg: ExperimentConfig,
    openai_llm: OpenAILLM,
    openai_embedder: OpenAIEmbeddings,
    ontology: OntologyCandidate,
    articles_df: pd.DataFrame,
    candidate_tag: str,
):
    """
    Build or extend the knowledge graph for a specific ontology candidate.
    Attach new entities and relationships to the existing graph.
    """
    logger.info("Building KG for ontology candidate: %s", ontology.name)

    kg_builder = SimpleKGPipeline(
        llm=openai_llm,
        driver=driver,
        schema=ontology.schema,
        from_pdf=False,
        embedder=openai_embedder,
        neo4j_database=neo4j_cfg.database,
        perform_entity_resolution=True,
    )

    # Ensure candidate_tag for base graph
    ensure_candidate_tags(driver, neo4j_cfg, [candidate_tag])

    # Process articles and attach new entities/relationships
    logger.info("Ingesting %d articles for candidate_tag: %s", len(articles_df), candidate_tag)
    articles_df = articles_df.sort_values("timestamp").reset_index(drop=True)

    # Create a semaphore to limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(50)  # Adjust the limit as needed

    async def limited_process_article(*args, **kwargs):
        async with semaphore:
            await process_article(*args, **kwargs)

    # Create tasks for concurrent execution with semaphore
    tasks = []
    for idx, row in articles_df.iterrows():
        raw = row.get("text")
        if not isinstance(raw, str) or not raw.strip():
            logger.debug("Skipping article %d (no valid text)", idx)
            continue

        text = raw.strip()
        headline = str(row.get("headline") or "")[:1024]
        url = str(row.get("url") or "")
        ts_iso = row["timestamp"].isoformat()
        day = row["day"]
        source = str(row.get("source") or "")

        logger.debug("Scheduling KG extraction for article %d on %s", idx, day)

        # Schedule the task with semaphore
        tasks.append(
            asyncio.create_task(
                limited_process_article(
                    kg_builder,
                    driver,
                    neo4j_cfg,
                    exp_cfg,
                    idx,
                    text,
                    headline,
                    url,
                    ts_iso,
                    day,
                    source,
                    candidate_tag,
                )
            )
        )

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    logger.info("KG for candidate %s updated with new entities and relationships", candidate_tag)

    # Log the number of nodes and relationships created
    with driver.session(database=neo4j_cfg.database) as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        logger.info("Ontology candidate '%s' created %d nodes and %d relationships", candidate_tag, node_count, relationship_count)

    # Build NEXT chain between days
    days = sorted(articles_df["day"].unique().tolist())
    logger.info("Days detected for NEXT chain: %d", len(days))

    if len(days) > 1:
        logger.info("Creating NEXT chain between %d Day nodes", len(days))
        with driver.session(database=neo4j_cfg.database) as session:
            session.run(
                """
                WITH $dates AS dates
                UNWIND range(0, size(dates)-2) AS i
                MERGE (d1:Day {date: dates[i]})
                MERGE (d2:Day {date: dates[i+1]})
                MERGE (d1)-[:NEXT {candidate_tag: $candidate_tag}]->(d2)
                """,
                dates=days,
                candidate_tag=candidate_tag,
            )
        logger.debug("NEXT chain created")


async def process_article(
    kg_builder,
    driver,
    neo4j_cfg,
    exp_cfg,
    idx,
    text,
    headline,
    url,
    ts_iso,
    day,
    source,
    candidate_tag,
):
    """
    Process a single article asynchronously.
    """
    await kg_builder.run_async(text=text)

    # Tag all relationships created by SimpleKGPipeline
    with driver.session(database=neo4j_cfg.database) as session:
        session.run(
            """
            MATCH ()-[r]->()
            WHERE r.candidate_tags IS NULL OR NOT $tag IN r.candidate_tags
            SET r.candidate_tags = COALESCE(r.candidate_tags, []) + $tag
            """,
            tag=candidate_tag
        )

    # Write article + link to Day + Company
    with driver.session(database=neo4j_cfg.database) as session:
        session.run(
            """
            MERGE (a:Article {id: $id})
            SET a.headline=$headline,
                a.timestamp=$ts,
                a.url=$url,
                a.source=$source

            MERGE (d:Day {date: $day})
            MERGE (c:Company {symbol: $sym})

            MERGE (a)-[:PUBLISHED_ON {candidate_tag: $candidate_tag}]->(d)
            MERGE (c)-[:ASSOCIATED_WITH {candidate_tag: $candidate_tag}]->(d)
            """,
            id=f"{exp_cfg.target_ticker}:{idx}",
            headline=headline,
            ts=ts_iso,
            url=url,
            source=source,
            day=day,
            sym=exp_cfg.target_ticker,
            candidate_tag=candidate_tag,
        )


def write_price_labels_to_days(
    driver,
    neo4j_cfg: Neo4jConfig,
    df_labels: pd.DataFrame,
):
    logger.info("Writing price labels for %d days (batched)", len(df_labels))

    # Convert to a list of dicts for UNWIND
    payload = [
        {
            "day": day,
            "ret": float(row["return_next_day"]),
            "direction": int(row["direction"]),
        }
        for day, row in df_labels.iterrows()
    ]

    with driver.session(database=neo4j_cfg.database) as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:Day {date: row.day})
            SET d.return_next_day = row.ret,
                d.direction = row.direction
            """,
            rows=payload,
        )

    logger.info("Price labels written (batched)")


def fetch_edges(driver, neo4j_cfg: Neo4jConfig) -> List[Tuple[int, int]]:
    logger.info("Fetching edges from Neo4j")
    with driver.session(database=neo4j_cfg.database) as session:
        result = session.run(
            """
            MATCH (n)-[r]->(m)
            RETURN elementId(n) AS src, elementId(m) AS dst
            """
        )
        edges = [(row["src"], row["dst"]) for row in result]
        logger.info("Fetched %d edges", len(edges))
        return edges


def fetch_day_nodes_with_labels(
    driver,
    neo4j_cfg: Neo4jConfig,
) -> pd.DataFrame:
    logger.info("Fetching Day nodes with labels")
    with driver.session(database=neo4j_cfg.database) as session:
        result = session.run(
            """
            MATCH (d:Day)
            WHERE d.direction IS NOT NULL
            RETURN elementId(d) AS node_id,
                   d.date AS day,
                   d.direction AS direction,
                   d.return_next_day AS return_next_day
            """
        )
        df = pd.DataFrame([dict(row) for row in result])
        logger.info("Fetched %d Day nodes with labels", len(df))
        return df


def fetch_edges_for_candidate(driver, neo4j_cfg: Neo4jConfig, candidate_tag: str) -> List[Tuple[int, int]]:
    """
    Fetch edges from Neo4j for a specific ontology candidate, identified by a unique tag.
    """
    logger.info("Fetching edges for ontology candidate: %s", candidate_tag)
    with driver.session(database=neo4j_cfg.database) as session:
        result = session.run(
            """
            MATCH (n)-[r]->(m)
            WHERE r.candidate_tag = $candidate_tag
            RETURN elementId(n) AS src, elementId(m) AS dst
            """,
            candidate_tag=candidate_tag
        )
        edges = [(row["src"], row["dst"]) for row in result]
        logger.info("Fetched %d edges for candidate: %s", len(edges), candidate_tag)
        return edges


def ensure_candidate_tags(driver, neo4j_cfg: Neo4jConfig, candidate_tags: List[str]):
    """
    Ensure that all nodes and relationships in the base graph have the specified candidate_tags.
    """
    logger.info("Ensuring candidate_tags %s for all nodes and relationships in the base graph", candidate_tags)

    with driver.session(database=neo4j_cfg.database) as session:
        # Add candidate_tags to all nodes
        session.run(
            """
            MATCH (n)
            SET n.candidate_tags = apoc.coll.union(n.candidate_tags, $candidate_tags)
            """,
            candidate_tags=candidate_tags,
        )

        # Add candidate_tags to all relationships
        session.run(
            """
            MATCH ()-[r]->()
            SET r.candidate_tags = apoc.coll.union(r.candidate_tags, $candidate_tags)
            """,
            candidate_tags=candidate_tags,
        )

    logger.info("Candidate_tags %s ensured for all nodes and relationships", candidate_tags)


def delete_candidate_tag(driver, neo4j_cfg, tag: str):
    with driver.session(database=neo4j_cfg.database) as session:
        session.run("""
            MATCH ()-[r]->()
            WHERE r.step_tag = $tag
            DELETE r
        """, tag=tag)

        session.run("""
            MATCH (n)
            WHERE $tag IN n.step_tags AND size(n.step_tags) = 1
            DETACH DELETE n
        """, tag=tag)

def persist_candidate_as_best(driver, neo4j_cfg, old_tag, new_tag):
    with driver.session(database=neo4j_cfg.database) as session:

        # Update node tags
        session.run(
            """
            MATCH (n)
            WHERE $old_tag IN n.candidate_tags
            SET n.candidate_tags = [x IN n.candidate_tags WHERE x <> $old_tag] + $new_tag
            """,
            old_tag=old_tag,
            new_tag=new_tag,
        )

        # Update relationship tags
        session.run(
            """
            MATCH ()-[r]->()
            WHERE $old_tag IN r.candidate_tags
            SET r.candidate_tags = [x IN r.candidate_tags WHERE x <> $old_tag] + $new_tag
            """,
            old_tag=old_tag,
            new_tag=new_tag,
        )


def fetch_edges_for_tags(driver, neo4j_cfg, tags):
    logger.info("Fetching edges for tags: %s", tags)
    with driver.session(database=neo4j_cfg.database) as session:
        result = session.run(
            """
            MATCH (n)-[r]->(m)
            WHERE ANY(tag IN r.candidate_tags WHERE tag IN $tags)
            RETURN elementId(n) AS src, elementId(m) AS dst
            """,
            tags=tags,
        )
        edges = [(row["src"], row["dst"]) for row in result]
        logger.info("Fetched %d edges for tags %s", len(edges), tags)
        return edges

