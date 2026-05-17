# file: finance_kg_experiment/pipeline.py
from __future__ import annotations
from typing import List
import logging

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

from config import Neo4jConfig, ExperimentConfig, OpenAIConfig
from data import load_article_sample, build_return_labels
from ontology import base_ontology, augment_ontology, OntologyCandidate
from ontology_agent import OntologyEvolutionAgent
from neo4j_io import (
    get_driver,
    clear_graph,
    build_kg_for_candidate,
    build_kg_for_candidate_incremental,
    write_price_labels_to_days,
    fetch_edges,
    fetch_day_nodes_with_labels,
    fetch_edges_for_tags,
    fetch_edges_for_candidate,
    persist_candidate_as_best,
    delete_candidate_tag,
)
from embeddings import compute_hope_embeddings
from modeling import build_day_embedding_frame, train_classifier_on_embeddings
from report import CandidateResult, ExperimentReport

logger = logging.getLogger(__name__)


async def run_ontology_experiment(
    neo4j_cfg: Neo4jConfig,
    exp_cfg: ExperimentConfig,
    openai_cfg: OpenAIConfig,
) -> ExperimentReport:
    """
    High-level experiment:
      1) Sample n_days × max_articles_per_day news for one ticker
      2) Build first (base) ontology candidate
      3) For each step:
          - generate ontology variants (LLM stub)
          - for each candidate:
              * clear Neo4j
              * build KG with SimpleKGPipeline for that schema
              * write TSLA returns & direction to Day nodes
              * compute HOPE embeddings
              * train XGB classifier on Day embeddings
      4) Return ExperimentReport including metrics for all candidates.
    """

    logger.info("Starting ontology experiment")

    # ----------------------------------------------------------------------
    # 0) Initialize external clients
    # ----------------------------------------------------------------------
    logger.info("Initializing Neo4j driver")
    driver = get_driver(neo4j_cfg)

    logger.info("Initializing LLM and embedder")
    llm = OpenAILLM(model_name=openai_cfg.model_name, api_key=openai_cfg.api_key)
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_cfg.api_key,
    )
    ontology_agent = OntologyEvolutionAgent(
    llm=llm,
    api_key=openai_cfg.api_key,   # for structured outputs
    model=openai_cfg.model_name,  # e.g., gpt-4o-mini
    )


    # -----------------------------------------------------------
    # CLEAR GRAPH BEFORE DOING ANYTHING
    # -----------------------------------------------------------
    logger.info("Clearing Neo4j graph before experiment start")
    clear_graph(driver, neo4j_cfg.database)
    logger.info("Graph cleared")

    # ----------------------------------------------------------------------
    # 1) Sample articles + price labels
    # ----------------------------------------------------------------------
    logger.info(
        "Loading article sample: %d days × max %d articles",
        exp_cfg.n_days,
        exp_cfg.max_articles_per_day,
    )
    articles_df = load_article_sample(exp_cfg)
    logger.info("Loaded %d article records", len(articles_df))

    logger.info("Building return labels")
    df_labels = build_return_labels(exp_cfg)
    logger.info("Return labels created: %d rows", len(df_labels))

    # ----------------------------------------------------------------------
    # 2) Prepare ontology search
    # ----------------------------------------------------------------------
    logger.info("Preparing base ontology")
    base = base_ontology()
    current_parents: List[OntologyCandidate] = [base]
    results: List[CandidateResult] = []

    # ----------------------------------------------------------------------
    # 3) Multi-step ontology search loop
    # ----------------------------------------------------------------------
    for step in range(exp_cfg.n_steps):
        logger.info("=== Step %d/%d ===", step + 1, exp_cfg.n_steps)
        next_parents: List[OntologyCandidate] = []

        for parent_idx, parent in enumerate(current_parents):
            logger.info("Processing parent ontology %d for this step", parent_idx)

            step_candidates_metrics = []

            # Generate candidate ontologies
            candidates: List[OntologyCandidate] = []
            for c_idx in range(exp_cfg.candidates_per_step):
                if step == 0 and c_idx == 0:
                    cand = parent
                    metrics_for_agent = {"auc": 0.0, "acc": 0.0}
                    logger.info("Candidate %d is base ontology", c_idx)
                else:
                    cand = await ontology_agent.propose_new_candidate(
                        previous=parent,
                        metrics=metrics_for_agent,      # feedback from previous run
                        step_index=step,
                        variant_index=c_idx,
                    )
                # else:
                #     cand = augment_ontology(parent, variant_index=c_idx)
                #     logger.info("Generated ontology variant %d", c_idx)
                candidates.append(cand)


            # Tags that represent the current accumulated graph
            # accumulated best tag after earlier steps
            if step == 0:
                active_tags = ["step_0_candidate_0"]
            else:
                active_tags = [current_parents[0].name]  # best ontology from previous step

            for c_idx, cand in enumerate(candidates):

                candidate_tag = f"step_{step}_candidate_{c_idx}"
                cand.tag = candidate_tag
                logger.info("Evaluating candidate %s", candidate_tag)

                # For step 0, use regular builder; for step 1+, use incremental builder
                if step == 0:
                    # Step 0: rebuild completely from scratch
                    await build_kg_for_candidate(
                        driver,
                        neo4j_cfg,
                        exp_cfg,
                        llm,
                        embedder,
                        cand,
                        articles_df,
                        candidate_tag=candidate_tag,
                    )
                else:
                    # Steps 1+: use incremental builder with accepted tags
                    await build_kg_for_candidate_incremental(
                        driver,
                        neo4j_cfg,
                        exp_cfg,
                        llm,
                        embedder,
                        cand,
                        articles_df,
                        candidate_tag=candidate_tag,
                        accepted_tags=active_tags,
                    )

                write_price_labels_to_days(driver, neo4j_cfg, df_labels)

                # Evaluate ONLY edges belonging to: active_tags + this candidate
                tags_for_eval = active_tags + [candidate_tag]

                edges = fetch_edges_for_tags(driver, neo4j_cfg, tags_for_eval)
                embedding_dict = compute_hope_embeddings(edges, dim=exp_cfg.embedding_dim)

                df_days = fetch_day_nodes_with_labels(driver, neo4j_cfg)
                df_embed = build_day_embedding_frame(df_days, embedding_dict)
                metrics = train_classifier_on_embeddings(df_embed, exp_cfg)

                results.append(
                    CandidateResult(
                        candidate=cand,
                        metrics=metrics,
                        step_index=step,
                        candidate_index=c_idx,
                    )
                )

                logger.info("Candidate %s metrics: %s", candidate_tag, metrics)

                step_candidates_metrics.append((cand, metrics, candidate_tag))


        # --- Select best ontology variant ----
        best_cand, best_metrics, best_tag = max(
            step_candidates_metrics,
            key=lambda x: x[1].auc if x[1].auc is not None else -1
        )

        logger.info("Best ontology candidate for step %d: %s", step, best_cand.name)

        # Determine old tag (the tag representing the graph up to now)
        if step == 0:
            old_tag = "step_0_candidate_0"
        else:
            # last winning ontology from previous step
            old_tag = current_parents[0].name

        # new tag is simply the name of the winning ontology
        new_tag = best_cand.name

        # Promote best candidate's subgraph to be the new accumulated graph
        persist_candidate_as_best(driver, neo4j_cfg, old_tag, new_tag)

        # Remove incorrect candidates
        for _, _, cand_tag in step_candidates_metrics:
            if cand_tag != best_tag:
                delete_candidate_tag(driver, neo4j_cfg, cand_tag)

        # Push best ontology into next step
        next_parents.append(best_cand)


    logger.info("Experiment completed — total results: %d", len(results))

    driver.close()
    logger.info("Closed Neo4j driver")

    return ExperimentReport(results=results)


async def construct_base_graph(neo4j_cfg: Neo4jConfig, exp_cfg: ExperimentConfig, openai_cfg: OpenAIConfig):
    """
    Constructs the base graph using the base ontology.
    """
    logger.info("Initializing Neo4j driver for base graph construction")
    driver = get_driver(neo4j_cfg)

    logger.info("Clearing Neo4j graph before base graph construction")
    clear_graph(driver, neo4j_cfg.database)

    logger.info("Loading article sample for base graph")
    articles_df = load_article_sample(exp_cfg)

    logger.info("Preparing base ontology")
    base = base_ontology()

    logger.info("Building base graph with candidate_tag 'base'")
    await build_kg_for_candidate(
        driver,
        neo4j_cfg,
        exp_cfg,
        OpenAILLM(model_name=openai_cfg.model_name, api_key=openai_cfg.api_key),
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_cfg.api_key),
        base,
        articles_df,
        candidate_tag="base",
    )

    logger.info("Base graph construction complete")


async def construct_candidate_graphs(neo4j_cfg: Neo4jConfig, exp_cfg: ExperimentConfig, openai_cfg: OpenAIConfig):
    """
    Constructs graphs for modest and larger candidate ontologies.
    """
    logger.info("Initializing Neo4j driver for candidate graph construction")
    driver = get_driver(neo4j_cfg)

    logger.info("Loading article sample for candidate graphs")
    articles_df = load_article_sample(exp_cfg)

    logger.info("Preparing base ontology")
    base = base_ontology()

    # Modest expansion
    logger.info("Creating modest ontology expansion")
    modest_ontology = augment_ontology(base, variant_index=1)
    logger.info("Building modest candidate graph with candidate_tag 'modest'")
    await build_kg_for_candidate(
        driver,
        neo4j_cfg,
        exp_cfg,
        OpenAILLM(model_name=openai_cfg.model_name, api_key=openai_cfg.api_key),
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_cfg.api_key),
        modest_ontology,
        articles_df,
        candidate_tag="modest",
    )

    # Larger expansion
    logger.info("Creating larger ontology expansion")
    larger_ontology = augment_ontology(base, variant_index=2)
    logger.info("Building larger candidate graph with candidate_tag 'larger'")
    await build_kg_for_candidate(
        driver,
        neo4j_cfg,
        exp_cfg,
        OpenAILLM(model_name=openai_cfg.model_name, api_key=openai_cfg.api_key),
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_cfg.api_key),
        larger_ontology,
        articles_df,
        candidate_tag="larger",
    )

    logger.info("Candidate graph construction complete")


async def process_embeddings_and_ml(neo4j_cfg: Neo4jConfig, exp_cfg: ExperimentConfig):
    """
    Processes embeddings and trains ML models for each candidate ontology.
    """
    logger.info("Initializing Neo4j driver for embedding and ML steps")
    driver = get_driver(neo4j_cfg)

    for candidate_tag in ["base", "modest", "larger"]:
        logger.info("Processing embeddings and ML for candidate_tag: %s", candidate_tag)

        # Fetch edges for the candidate
        logger.info("Fetching edges for candidate_tag: %s", candidate_tag)
        edges = fetch_edges_for_candidate(driver, neo4j_cfg, candidate_tag)
        logger.info("Fetched %d edges for candidate_tag: %s", len(edges), candidate_tag)

        # Compute embeddings
        logger.info("Computing embeddings for candidate_tag: %s", candidate_tag)
        embedding_dict = compute_hope_embeddings(edges, dim=exp_cfg.embedding_dim)
        logger.info("Computed embeddings for %d nodes", len(embedding_dict))

        # Fetch Day nodes with labels
        logger.info("Fetching Day nodes with labels for candidate_tag: %s", candidate_tag)
        df_days = fetch_day_nodes_with_labels(driver, neo4j_cfg)
        logger.info("Fetched %d Day nodes", len(df_days))

        # Build embedding frame
        logger.info("Building embedding frame for candidate_tag: %s", candidate_tag)
        df_embed = build_day_embedding_frame(df_days, embedding_dict)
        logger.info("Built embedding frame for candidate_tag: %s", candidate_tag)

        # Train classifier
        logger.info("Training classifier for candidate_tag: %s", candidate_tag)
        metrics = train_classifier_on_embeddings(df_embed, exp_cfg)
        logger.info("Training complete for candidate_tag: %s. Metrics: %s", candidate_tag, metrics)

    driver.close()
    logger.info("Closed Neo4j driver")


# Ensure articles_df is initialized even when skipping graph construction
    if skip_graph and articles_df is None:
        logger.warning("articles_df is not initialized. Loading articles for later steps.")
        articles_df = load_article_sample(exp_cfg)
        logger.info("Loaded %d article records", len(articles_df))

    # Generate all ontology variants (base + augmented variants)
    logger.info("Generating all ontology variants")
    all_candidates = [
        base,
        augment_ontology(base, variant_index=0),
        augment_ontology(base, variant_index=1),
        augment_ontology(base, variant_index=2),
    ]

    for idx, candidate in enumerate(all_candidates):
        candidate_tag = f"variant_{idx}" if idx > 0 else "base"
        logger.info("Processing candidate: %s", candidate_tag)

        # Clear graph
        logger.info("Clearing Neo4j graph for candidate: %s", candidate_tag)
        clear_graph(driver, neo4j_cfg.database)

        # Build KG
        logger.info("Building KG for candidate: %s", candidate_tag)
        await build_kg_for_candidate(
            driver,
            neo4j_cfg,
            exp_cfg,
            llm,
            embedder,
            candidate,
            articles_df,
            candidate_tag=candidate_tag,
        )

        # Train model
        logger.info("Fetching Day nodes with labels for candidate: %s", candidate_tag)
        df_days = fetch_day_nodes_with_labels(driver, neo4j_cfg)
        logger.info("Fetched %d Day nodes", len(df_days))

        logger.info("Building embedding frame for candidate: %s", candidate_tag)
        edges = fetch_edges(driver, neo4j_cfg)
        embedding_dict = compute_hope_embeddings(edges, dim=exp_cfg.embedding_dim)
        df_embed = build_day_embedding_frame(df_days, embedding_dict)

        logger.info("Training classifier for candidate: %s", candidate_tag)
        metrics = train_classifier_on_embeddings(df_embed, exp_cfg)
        logger.info("Training complete for candidate: %s. Metrics: %s", candidate_tag, metrics)

        # Store results
        results.append(
            CandidateResult(
                candidate=candidate,
                metrics=metrics,
                step_index=0,  # Single-step process for all variants
                candidate_index=idx,
            )
        )

    logger.info("All ontology variants processed. Total results: %d", len(results))
