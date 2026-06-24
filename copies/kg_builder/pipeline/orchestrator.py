"""Experiment orchestration."""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from copies.kg_builder.config import Config
from copies.kg_builder.core.article_linking import create_and_link_article_days
from copies.kg_builder.core.data import prepare_articles
from copies.kg_builder.core.graph import GraphDriver
from copies.kg_builder.core.neo4j_io import write_price_labels_to_days
from copies.kg_builder.core.ontology import OntologyCandidate, create_base_ontology
from copies.kg_builder.core.ontology_io import save_ontology_candidate, save_ontology_summary
from copies.kg_builder.core.tagging import tag_candidate_entities
from copies.kg_builder.ml.modeling import ModelMetrics
from copies.kg_builder.mutations.base import build_kg_incremental_candidate
from copies.kg_builder.pipeline.evaluator import evaluate_candidate
from copies.kg_builder.pipeline.ontology_evolution import OntologyEvolutionAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the KG building experiment."""

    def __init__(
        self,
        config: Config,
        driver: GraphDriver,
        llm: OpenAILLM,
        embedder: OpenAIEmbeddings,
    ):
        """Initialize orchestrator.

        Args:
            config: Configuration
            driver: Graph driver
            llm: Language model
            embedder: Embedding model
        """
        self.config = config
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.evolution_agent = OntologyEvolutionAgent(llm)
        self.results = {}
        self.candidates_per_step: Dict[int, List[OntologyCandidate]] = {}
        self.ontologies_dir = Path("results/ontologies")

    async def run(self, articles_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
        """Run the experiment.

        Args:
            articles_df: Articles data
            price_df: Price data with returns

        Returns:
            Experiment results
        """
        logger.info("Starting experiment with %d articles", len(articles_df))

        # Clear the graph completely for clean state
        logger.info("Clearing Neo4j graph...")
        self.driver.run_query("MATCH (n) DETACH DELETE n")
        logger.info("✓ Graph cleared")

        # Prepare data
        articles_df = prepare_articles(articles_df)

        # Create Day nodes and price labels ONCE at the beginning
        logger.info("Creating Day nodes and writing price labels...")
        create_and_link_article_days(self.driver, articles_df)
        df_labels = self._build_day_labels_df(price_df)
        write_price_labels_to_days(self.driver, self.config.neo4j, df_labels)
        logger.info("✓ Day nodes and price labels ready")

        # Tag Article and Day nodes with "base_structure" so incremental mutator can link to them
        # These are infrastructure nodes needed by all steps
        tag_candidate_entities(self.driver, "base_structure")
        logger.info("✓ Tagged Article and Day nodes as base_structure")

        # Create base ontology
        base_ontology = create_base_ontology()

        # STEP 0: Build base structure (minimal infrastructure - no LLM calls)
        logger.info("=== Step 0 (Base Structure) ===")
        logger.info("Building base structure (articles + days + labels, no LLM calls)...")
        await self._build_base_structure(base_ontology, articles_df)
        logger.info("✓ Base structure ready (step 0 complete)")

        # STEP 1+: Evolve and evaluate ontologies using incremental mutation
        # num_steps=0 means only base, num_steps=1 means base + 1 evolution step, etc.
        best_candidate = base_ontology
        for step in range(1, self.config.experiment.num_steps + 1):
            logger.info(
                "=== Step %d/%d (Incremental Evolution) ===",
                step,
                self.config.experiment.num_steps,
            )

            await self._run_step_incremental_evolution(
                best_candidate=best_candidate,
                articles_df=articles_df,
                price_df=price_df,
                step=step,
            )

            # Select best from THIS step only (not global)
            best_candidate = self._select_best_candidate_from_current_step(step)

        logger.info("Experiment completed")

        # Save all ontologies and summary
        self._save_ontologies()

        return self.results

    async def _build_base_structure(
        self,
        base_ontology: OntologyCandidate,
        articles_df: pd.DataFrame,
    ) -> None:
        """Build base ontology structure - minimal infrastructure without LLM calls.
        
        Creates only:
        - Article nodes with metadata
        - Day nodes with price labels
        - Article PUBLISHED_ON Day relationships
        
        This is everything needed by ML evaluation. Entity extraction happens separately.

        Args:
            base_ontology: Base ontology
            articles_df: Articles data
        """
        base_candidate_tag = "base_structure"
        base_ontology.candidate_tag = base_candidate_tag

        logger.info("Building base structure (articles + days + labels, no LLM)...")
        logger.info("Articles: %d | Days: %d", len(articles_df), articles_df["day"].nunique())
        
        # Infrastructure is already created by orchestrator.run():
        # - Article nodes
        # - Day nodes  
        # - PUBLISHED_ON relationships
        # - Price labels on Day nodes
        # All tagged with "base_structure"
        
        logger.info("✓ Base structure ready for entity extraction")

        # Log base structure size
        base_nodes = self.driver.get_count()
        base_rels = self.driver.get_relationship_count()
        logger.info("  Base structure: %d nodes, %d relationships", base_nodes, base_rels)

    async def _run_step_incremental_evolution(
        self,
        best_candidate: OntologyCandidate,
        articles_df: pd.DataFrame,
        price_df: pd.DataFrame,
        step: int,
    ) -> List[OntologyCandidate]:
        """Run an incremental evolution step.

        Uses incremental mutator to build new candidates based on best from previous step.
        Only evolves and evaluates candidates (no SimpleKGPipeline).

        Args:
            best_candidate: Best candidate from previous step
            articles_df: Articles data
            price_df: Price data
            step: Current step number (1+)

        Returns:
            List of candidates for this step
        """
        logger.info("Evolving ontology from %s", best_candidate.candidate_tag)

        # Get metrics for best candidate
        best_metrics = self.results.get(best_candidate.candidate_tag, ModelMetrics(0, 0, 0, 0))

        # Generate evolved candidates
        candidates = []
        for variant_idx in range(self.config.experiment.max_candidates_per_step):
            evolved = await self.evolution_agent.propose_new_candidate(
                previous=best_candidate,
                metrics=best_metrics,
                step_index=step,
                variant_index=variant_idx,
            )
            candidates.append(evolved)
        self.candidates_per_step[step] = candidates

        for idx, candidate in enumerate(candidates):
            logger.info("Evaluating evolved candidate: %s", candidate.candidate_tag)

            nodes_before = self.driver.get_count()
            rels_before = self.driver.get_relationship_count()

            await build_kg_incremental_candidate(
                self.driver,
                self.config.neo4j,
                self.config.experiment,
                self.llm,
                candidate,
                articles_df,
                candidate.candidate_tag,
                self._get_accepted_tags_for_step(step),
            )

            nodes_after = self.driver.get_count()
            rels_after = self.driver.get_relationship_count()
            nodes_added = nodes_after - nodes_before
            rels_added = rels_after - rels_before
            logger.info(
                "DELTA for %s: +%d nodes, +%d relationships",
                candidate.candidate_tag,
                nodes_added,
                rels_added,
            )

            tag_candidate_entities(self.driver, candidate.candidate_tag)

            # Build allowed tags for evaluation
            # Step 1+: evaluate with base + current + remaining tagged candidates from previous steps
            allowed_tags_for_eval = ["base_structure", candidate.candidate_tag] + [
                c.candidate_tag
                for prev_step in range(step)
                if prev_step in self.candidates_per_step
                for c in self.candidates_per_step[prev_step]
            ]

            logger.info("Evaluating with allowed tags: %s", allowed_tags_for_eval)

            metrics = evaluate_candidate(
                self.driver,
                self.embedder,
                [candidate.candidate_tag],
                {},
                price_df,
                allowed_tags=allowed_tags_for_eval,
            )

            self.results[candidate.candidate_tag] = metrics
            logger.info(
                "Candidate %s: AUC=%.4f, F1=%.4f", candidate.candidate_tag, metrics.auc, metrics.f1
            )

        # Prune tags of losing candidates
        if self.results and self.candidates_per_step[step]:
            best_step = max(
                self.candidates_per_step[step],
                key=lambda c: self.results.get(c.candidate_tag, ModelMetrics(0, 0, 0, 0)).auc,
            )
            logger.info("Winner of step %d: %s", step, best_step.candidate_tag)

            # Prune tags from losing candidates (don't delete nodes/rels, just remove their tags)
            for candidate in self.candidates_per_step[step]:
                if candidate.candidate_tag != best_step.candidate_tag:
                    self._prune_candidate_tags(candidate.candidate_tag)
                    logger.info("Pruned tags from losing candidate: %s", candidate.candidate_tag)

    def _select_best_candidate_from_current_step(self, step: int) -> OntologyCandidate:
        """Select best candidate from the current step only.

        Args:
            step: Current step number

        Returns:
            Best OntologyCandidate from this step
        """
        if step not in self.candidates_per_step or not self.candidates_per_step[step]:
            logger.warning("No candidates for step %d, using base ontology", step)
            return create_base_ontology()

        # Find candidate with highest AUC from THIS STEP ONLY
        step_candidates = self.candidates_per_step[step]
        best_candidate = max(
            step_candidates,
            key=lambda c: self.results.get(c.candidate_tag, ModelMetrics(0, 0, 0, 0)).auc,
        )

        best_metrics = self.results.get(best_candidate.candidate_tag, ModelMetrics(0, 0, 0, 0))
        logger.info(
            "Selected best candidate from step %d: %s (AUC=%.4f)",
            step,
            best_candidate.candidate_tag,
            best_metrics.auc,
        )

        return best_candidate

    def _select_best_candidate(self) -> OntologyCandidate:
        """Select best candidate from current step.

        Returns:
            Best OntologyCandidate
        """
        # Find candidate with highest AUC
        if not self.results:
            logger.warning("No results yet, using base ontology")
            return create_base_ontology()

        best_tag = max(self.results.keys(), key=lambda k: self.results[k].auc)
        best_metrics = self.results[best_tag]

        logger.info("Selected best candidate: %s (AUC=%.4f)", best_tag, best_metrics.auc)

        # Find the candidate object
        for step_candidates in self.candidates_per_step.values():
            if step_candidates is None:
                continue
            for cand in step_candidates:
                if cand.candidate_tag == best_tag:
                    return cand

        # Fallback
        return create_base_ontology()

    def _extract_day_labels(self, price_df: pd.DataFrame) -> Dict[str, int]:
        """Extract day-level labels from price data.

        Args:
            price_df: DataFrame with 'date' and 'return' columns

        Returns:
            Dict mapping date to label (0 or 1)
        """
        labels = {}
        for _, row in price_df.iterrows():
            day = str(row["date"])
            label = 1 if row["return"] > 0 else 0
            labels[day] = label

        return labels

    def _tag_untagged_entities_for_candidate(self, candidate_tag: str) -> None:
        """Tag entities that were created by incremental build but not yet tagged.

        Args:
            candidate_tag: Tag to apply to untagged entities
        """
        query = """
        MATCH (n)-[r]->(m)
        WHERE NOT any(tag IN r.candidate_tags WHERE tag = $tag)
        SET r.candidate_tags = CASE
            WHEN r.candidate_tags IS NULL THEN [$tag]
            ELSE r.candidate_tags + $tag
        END
        WITH r
        LIMIT 10000
        RETURN count(r) as tagged_count
        """
        try:
            result = self.driver.run_query(query, parameters={"tag": candidate_tag})
            if result:
                tagged_count = (
                    result[0].get("tagged_count", 0) if isinstance(result[0], dict) else 0
                )
                if tagged_count > 0:
                    logger.info(
                        "Tagged %d previously untagged relationships with %s",
                        tagged_count,
                        candidate_tag,
                    )
        except Exception as e:
            logger.warning("Could not tag untagged entities: %s", str(e))

    def _build_day_labels_df(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Build day labels DataFrame for writing to Neo4j.

        Args:
            price_df: DataFrame indexed by 'day' with 'return_next_day' and 'direction' columns,
                     OR DataFrame with 'date' and 'return' columns (legacy format)

        Returns:
            DataFrame indexed by date with 'direction' and 'return_next_day' columns
        """
        df_labels = price_df.copy()

        # Check if already in proper format (indexed by day with return_next_day and direction)
        if isinstance(df_labels.index, pd.Index) and df_labels.index.name == "day":
            # Already in proper format from Stooq
            return df_labels[["direction", "return_next_day"]]
        elif "return_next_day" in df_labels.columns and "direction" in df_labels.columns:
            # Already has both columns, just ensure proper indexing
            if "date" in df_labels.columns:
                df_labels = df_labels.set_index("date")
            elif "day" in df_labels.columns:
                df_labels = df_labels.set_index("day")
            return df_labels[["direction", "return_next_day"]]
        else:
            # Legacy format: has 'date'/'day' and 'return' columns
            if "date" in df_labels.columns:
                date_col = "date"
            elif "day" in df_labels.columns:
                date_col = "day"
            else:
                raise ValueError(
                    f"Could not find date column. Available columns: {df_labels.columns.tolist()}"
                )

            df_labels["direction"] = (df_labels["return"] > 0).astype(int)
            df_labels["return_next_day"] = df_labels["return"]
            df_labels = df_labels.set_index(date_col)[["direction", "return_next_day"]]
            return df_labels

    def _extract_day_labels_old(self, price_df: pd.DataFrame) -> Dict[str, int]:
        """Extract day-level labels from price data.

        Args:
            price_df: DataFrame with 'date' and 'return' columns

        Returns:
            Dict mapping date to label (0 or 1)
        """
        labels = {}
        for _, row in price_df.iterrows():
            day = row["date"]
            label = 1 if row["return"] > 0 else 0
            labels[day] = label

        return labels

    def _get_allowed_tags_for_step(self, step: int, current_candidate_tag: str) -> List[str]:
        """Get allowed tags for evaluation: base structure + winners from all previous steps + current candidate.

        Args:
            step: Current step number
            current_candidate_tag: Tag of current candidate being evaluated

        Returns:
            List of allowed tags (base_structure + winners from steps 0..step-1 + current candidate)
        """
        # Always include base structure (has articles)
        allowed = ["base_structure", current_candidate_tag]

        # Add winners from all previous steps
        for prev_step in range(step):
            if prev_step in self.winner_per_step:
                allowed.append(self.winner_per_step[prev_step])
                logger.debug(
                    "Including winner from step %d: %s", prev_step, self.winner_per_step[prev_step]
                )

        return allowed

    def _delete_candidate_from_graph(self, candidate_tag: str) -> None:
        """Delete all entities and relationships tagged only with this candidate.

        Keeps nodes/edges that have other tags (from base or other candidates).

        Args:
            candidate_tag: Tag of candidate to delete
        """
        logger.info("Deleting candidate %s from graph", candidate_tag)

        # Delete relationships tagged only with this candidate
        query_rels = """
        MATCH ()-[r]->()
        WHERE r.candidate_tags IS NOT NULL AND $tag IN r.candidate_tags
        SET r.candidate_tags = [t IN r.candidate_tags WHERE t <> $tag]
        WITH r WHERE r.candidate_tags IS NULL OR size(r.candidate_tags) = 0
        DELETE r
        RETURN count(*) as deleted_rels
        """
        result = self.driver.run_query(query_rels, parameters={"tag": candidate_tag})
        if result:
            deleted_rels = result[0].get("deleted_rels", 0) if isinstance(result[0], dict) else 0
            logger.info("Deleted %d relationships for candidate %s", deleted_rels, candidate_tag)

        # Delete nodes tagged only with this candidate
        query_nodes = """
        MATCH (n)
        WHERE n.candidate_tags IS NOT NULL AND $tag IN n.candidate_tags
        SET n.candidate_tags = [t IN n.candidate_tags WHERE t <> $tag]
        WITH n WHERE n.candidate_tags IS NULL OR size(n.candidate_tags) = 0
        DETACH DELETE n
        RETURN count(*) as deleted_nodes
        """
        result = self.driver.run_query(query_nodes, parameters={"tag": candidate_tag})
        if result:
            deleted_nodes = result[0].get("deleted_nodes", 0) if isinstance(result[0], dict) else 0
            logger.info("Deleted %d nodes for candidate %s", deleted_nodes, candidate_tag)

    def _get_accepted_tags_for_step(self, step: int) -> List[str]:
        """Get accepted tags for building candidates in this step.

        Returns base_structure + ONLY tags that still exist in the graph.
        (Losing candidates already had their tags pruned, so we verify they still exist)

        Args:
            step: Current step number

        Returns:
            List of accepted tags for isolation constraint (only tags that exist in graph)
        """
        # Start with base_structure
        accepted_tags = ["base_structure"]

        # Add candidate tags from previous steps, but ONLY if they still have nodes in the graph
        # (losers are pruned, so we verify they still exist before adding them)
        for prev_step in range(step):
            if prev_step in self.candidates_per_step:
                for candidate in self.candidates_per_step[prev_step]:
                    # Check if this tag still exists in the graph
                    # A tag exists if there's at least one node with it
                    tag_exists = self._tag_exists_in_graph(candidate.candidate_tag)
                    if tag_exists:
                        accepted_tags.append(candidate.candidate_tag)
                    else:
                        logger.debug(
                            "Skipping pruned tag %s (no longer in graph)", candidate.candidate_tag
                        )

        logger.debug("Accepted tags for step %d: %s", step, accepted_tags)
        return accepted_tags

    def _tag_exists_in_graph(self, tag: str) -> bool:
        """Check if a tag still exists in the graph (has at least one node with it).

        Args:
            tag: Tag to check

        Returns:
            True if at least one node or relationship has this tag, False otherwise
        """
        query = """
        MATCH (n)
        WHERE n.candidate_tags IS NOT NULL AND $tag IN n.candidate_tags
        RETURN count(*) as count
        LIMIT 1
        """
        result = self.driver.run_query(query, parameters={"tag": tag})
        if result and len(result) > 0:
            count = result[0].get("count", 0) if isinstance(result[0], dict) else 0
            return count > 0
        return False

    def _prune_candidate_tags(self, candidate_tag: str) -> None:
        """Remove a candidate's tag from all nodes and relationships in the graph.

        This is simpler than deletion: we keep nodes/rels but remove the tag.
        If a node/rel has only this tag, it becomes untagged (can still be linked to).
        If it has multiple tags, we just remove this one.

        Args:
            candidate_tag: Tag to prune
        """
        logger.info("Pruning tags for candidate %s", candidate_tag)

        # Prune relationship tags
        query_rels = """
        MATCH ()-[r]->()
        WHERE r.candidate_tags IS NOT NULL AND $tag IN r.candidate_tags
        SET r.candidate_tags = [t IN r.candidate_tags WHERE t <> $tag]
        RETURN count(*) as pruned_rels
        """
        result = self.driver.run_query(query_rels, parameters={"tag": candidate_tag})
        if result:
            pruned_rels = result[0].get("pruned_rels", 0) if isinstance(result[0], dict) else 0
            logger.info("Pruned %d relationships (removed tag %s)", pruned_rels, candidate_tag)

        # Prune node tags
        query_nodes = """
        MATCH (n)
        WHERE n.candidate_tags IS NOT NULL AND $tag IN n.candidate_tags
        SET n.candidate_tags = [t IN n.candidate_tags WHERE t <> $tag]
        RETURN count(*) as pruned_nodes
        """
        result = self.driver.run_query(query_nodes, parameters={"tag": candidate_tag})
        if result:
            pruned_nodes = result[0].get("pruned_nodes", 0) if isinstance(result[0], dict) else 0
            logger.info("Pruned %d nodes (removed tag %s)", pruned_nodes, candidate_tag)

    def _save_ontologies(self) -> None:
        """Save all ontology candidates and results to files."""
        logger.info("Saving ontology candidates...")

        # Save each candidate individually
        for step, candidates in self.candidates_per_step.items():
            for candidate in candidates:
                metrics = self.results.get(candidate.candidate_tag, ModelMetrics(0, 0, 0, 0))
                save_ontology_candidate(candidate, metrics, self.ontologies_dir)

        # Get best candidate tag
        if not self.results:
            logger.warning("No results to save")
            return

        best_tag = max(self.results.keys(), key=lambda k: self.results[k].auc)

        # Save summary
        save_ontology_summary(
            candidates_by_step=self.candidates_per_step,
            metrics_by_tag=self.results,
            best_candidate_tag=best_tag,
            output_dir=self.ontologies_dir,
        )

        logger.info("✓ Saved ontologies to %s", self.ontologies_dir)
