"""Simple graph mutation using SimpleKGPipeline."""

import asyncio
import logging
from typing import List

import pandas as pd
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from kg_builder.config import ExperimentConfig, Neo4jConfig
from kg_builder.core.graph import GraphDriver
from kg_builder.core.ontology import OntologyCandidate
from kg_builder.mutations.incremental_kg_mutator import IncrementalArticleKGMutator

logger = logging.getLogger(__name__)


async def build_kg_simple(
    driver: GraphDriver,
    neo4j_cfg: Neo4jConfig,
    exp_cfg: ExperimentConfig,
    llm: OpenAILLM,
    embedder: OpenAIEmbeddings,
    ontology: OntologyCandidate,
    articles_df: pd.DataFrame,
    candidate_tag: str,
) -> None:
    """DEPRECATED: Build knowledge graph using SimpleKGPipeline.
    
    This function is deprecated. The pipeline now builds base structure without LLM calls,
    then separately extracts entities using incremental mutation.
    
    Kept for backward compatibility only.

    Args:
        driver: Graph driver
        neo4j_cfg: Neo4j configuration
        exp_cfg: Experiment configuration
        llm: Language model
        embedder: Embedding model
        ontology: Ontology candidate
        articles_df: Articles DataFrame
        candidate_tag: Tag for this candidate
    """
    logger.warning("build_kg_simple() is deprecated. Use build_kg_incremental_candidate() instead.")
    logger.info("Building KG for candidate: %s (using deprecated SimpleKGPipeline)", candidate_tag)
    logger.info("Using semaphore limit: %d", exp_cfg.semaphore_limit)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver.driver,
        schema=ontology.schema,
        from_pdf=False,
        embedder=embedder,
        neo4j_database=neo4j_cfg.database,
        perform_entity_resolution=True,
    )

    # Process each article
    articles_df = articles_df.sort_values("timestamp").reset_index(drop=True)
    semaphore = asyncio.Semaphore(exp_cfg.semaphore_limit)

    async def process_article(text: str, headline: str):
        async with semaphore:
            try:
                await kg_builder.run_async(text=text)
                logger.debug("Processed article: %s", headline[:50])
            except Exception as e:
                logger.warning("Failed to process article %s: %s", headline[:50], str(e))

    # Create tasks
    tasks = []
    valid_articles = []
    for _, row in articles_df.iterrows():
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        headline = str(row.get("headline", ""))[:1024]
        valid_articles.append((text.strip(), headline))
        tasks.append(process_article(text.strip(), headline))

    # Run all tasks with progress bar
    if tasks:
        logger.info("Processing %d articles for candidate: %s", len(tasks), candidate_tag)
        await asyncio.gather(*tasks)

    logger.info("KG built for candidate: %s (deprecated)", candidate_tag)

    # Log stats
    node_count = driver.get_count()
    rel_count = driver.get_relationship_count()
    logger.info("Created %d nodes and %d relationships", node_count, rel_count)


async def build_kg_incremental_candidate(
    driver: GraphDriver,
    neo4j_cfg: Neo4jConfig,
    exp_cfg: ExperimentConfig,
    llm: OpenAILLM,
    ontology: OntologyCandidate,
    articles_df: pd.DataFrame,
    candidate_tag: str,
    accepted_tags: List[str],
) -> None:
    """Build knowledge graph for a candidate using incremental mutation.

    Uses IncrementalArticleKGMutator to extract entities with isolation constraints:
    - New entities can link to accepted_tags (e.g., base_structure)
    - New entities cannot link to other candidates in same step
    - All new entities get tagged with candidate_tag

    Args:
        driver: Graph driver
        neo4j_cfg: Neo4j configuration
        exp_cfg: Experiment configuration
        llm: Language model
        ontology: Ontology candidate
        articles_df: Articles DataFrame
        candidate_tag: Tag for this candidate
        accepted_tags: Tags of entities this candidate can link to (isolation constraint)
    """
    logger.info(
        "Building KG incrementally for candidate: %s with accepted_tags=%s",
        candidate_tag,
        accepted_tags,
    )
    logger.info("Using semaphore limit: %d", exp_cfg.semaphore_limit)
    logger.info(
        "Using ontology schema with %d node types and %d relationship types",
        len(ontology.schema.get("node_types", [])),
        len(ontology.schema.get("relationship_types", [])),
    )
    logger.info("articles_df columns: %s", list(articles_df.columns))
    logger.info("articles_df shape: %s", articles_df.shape)
    if len(articles_df) > 0:
        logger.info(
            "First article text length: %d",
            len(str(articles_df.iloc[0].get("text", ""))) if "text" in articles_df.columns else 0,
        )

    mutator = IncrementalArticleKGMutator(driver.driver, llm)

    # Process each article
    articles_df = articles_df.sort_values("timestamp").reset_index(drop=True)
    semaphore = asyncio.Semaphore(exp_cfg.semaphore_limit)

    async def process_article(article_id: str, text: str, headline: str, day: str):
        async with semaphore:
            try:
                # Extract and add entities using incremental mutator
                await mutator.mutate_article(
                    article_id=article_id,
                    text=text,
                    ontology=ontology.schema,
                    accepted_tags=accepted_tags,
                    candidate_tag=candidate_tag,
                )
                logger.debug("Processed article: %s", headline[:50])
            except Exception as e:
                logger.warning(
                    "Failed to process article %s: %s", headline[:50], str(e), exc_info=True
                )

            # Also ensure Article node exists and is linked to Day
            # (even if mutator extracted nothing, we still need the article-day link)
            with driver.driver.session(database=neo4j_cfg.database) as session:
                text_hash = str(hash(text))
                try:
                    session.run(
                        """
                        MERGE (a:Article {id: $text_id})
                        SET a.headline = $headline
                        WITH a
                        MATCH (d:Day {date: $day})
                        MERGE (a)-[r:PUBLISHED_ON]->(d)
                        """,
                        text_id=text_hash,
                        headline=headline,
                        day=day,
                    )
                except Exception as e:
                    logger.debug("Failed to create Article node: %s", str(e))

    # Create tasks
    tasks = []
    for idx, row in articles_df.iterrows():
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            logger.debug("Skipping article %d: empty or invalid text", idx)
            continue

        article_id = str(row.get("article_id", idx))
        headline = str(row.get("headline", ""))[:1024]
        day = str(row.get("day", ""))
        tasks.append(process_article(article_id, text.strip(), headline, day))

    logger.info("Prepared %d tasks for candidate: %s", len(tasks), candidate_tag)

    # Run all tasks with asyncio.gather
    if tasks:
        logger.info("Processing %d articles for candidate: %s", len(tasks), candidate_tag)
        await asyncio.gather(*tasks)
    else:
        logger.warning(
            "⚠️  NO ARTICLES FOUND TO PROCESS for candidate: %s (articles_df has %d rows)",
            candidate_tag,
            len(articles_df),
        )
        logger.info("All tasks completed for candidate: %s", candidate_tag)

    logger.info("KG built incrementally for candidate: %s", candidate_tag)
