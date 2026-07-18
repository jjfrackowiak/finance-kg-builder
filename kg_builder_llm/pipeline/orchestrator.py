"""Experiment orchestration."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from kg_builder_llm.config import Config
from kg_builder_llm.core.article_linking import create_and_link_article_days
from kg_builder_llm.core.data import prepare_articles
from kg_builder_llm.core.entity_resolution import (
    MERGE_DUPLICATES_BY_KEY_CYPHER,
    MERGE_DUPLICATES_BY_NAME_CYPHER,
)
from kg_builder_llm.core.graph import GraphDriver
from kg_builder_llm.core.ids import article_text_id
from kg_builder_llm.core.neo4j_io import write_price_labels_to_days
from kg_builder_llm.core.ontology import OntologyCandidate, create_base_ontology
from kg_builder_llm.core.ontology_io import save_ontology_candidate, save_ontology_summary
from kg_builder_llm.core.tagging import tag_candidate_entities
from kg_builder_llm.ml.modeling import ModelMetrics
from kg_builder_llm.mutations.base import build_kg_incremental_candidate
from kg_builder_llm.pipeline.evaluator import evaluate_article_text_baseline, evaluate_candidate
from kg_builder_llm.pipeline.mlflow_logging import MlflowExperimentLogger
from kg_builder_llm.pipeline.ontology_evolution import OntologyEvolutionAgent

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
        ontology_llm = self._build_ontology_llm(config)
        self.evolution_agent = OntologyEvolutionAgent(
            ontology_llm,
            prompt_template_path=config.experiment.evolution_prompt_template,
            single_addition=config.experiment.single_addition,
        )
        self.results = {}
        self.candidates_per_step: Dict[int, List[OntologyCandidate]] = {}
        # Single-addition mode: one record per proposed (node, relationship)
        # addition, updated with ΔAUC and accept/reject outcome as known.
        self.addition_history: List[dict] = []
        self._last_accepted_auc: Optional[float] = None
        self.ontologies_dir = Path("results/ontologies")
        self._mlflow: Optional[MlflowExperimentLogger] = None

    @staticmethod
    def _build_ontology_llm(config: Config):
        """Return an LLM for ontology reasoning.

        Uses Bedrock (Qwen3-32B) when AWS credentials with a session token are
        present (i.e. running on EKS with IRSA), falls back to the same vLLM
        endpoint used for extraction when running locally.
        """
        import boto3

        from kg_builder_llm.core.bedrock_llm import BedrockLLM

        try:
            creds = boto3.session.Session().get_credentials().get_frozen_credentials()
            if not creds or not creds.token:
                raise RuntimeError("no IAM session token")
            logger.info("Using Bedrock %s for ontology evolution", config.bedrock.model_id)
            return BedrockLLM(
                model_id=config.bedrock.model_id,
                region=config.bedrock.region,
            )
        except Exception:
            logger.info("Bedrock unavailable — falling back to OpenAI for ontology LLM")
            kwargs = dict(
                model_name=config.openai.model_name,
                model_params={"temperature": 0.1},
                api_key=config.openai.api_key,
            )
            if config.openai.base_url:
                kwargs["base_url"] = config.openai.base_url
            return OpenAILLM(**kwargs)

    async def run(self, articles_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
        """Run the experiment.

        Args:
            articles_df: Articles data
            price_df: Price data with returns

        Returns:
            Experiment results
        """
        logger.info("Starting experiment with %d articles", len(articles_df))

        # Capture the parent MLflow run now: inside nested child runs the
        # parent is no longer the active run.
        self._mlflow = MlflowExperimentLogger()

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

        # STEP 0: base structure (articles + days + labels) was built above and
        # tagged "base_structure"; the base ontology adopts that tag so results
        # lookups and incremental builds line up.
        logger.info("=== Step 0 (Base Structure) ===")
        base_ontology = create_base_ontology()
        base_ontology.candidate_tag = "base_structure"
        logger.info(
            "✓ Base structure ready: %d nodes, %d relationships",
            self.driver.get_count(),
            self.driver.get_relationship_count(),
        )

        # STEP 0b: Write article text embeddings then run text-only baseline
        logger.info("=== Step 0b (Article-text baseline) ===")
        await self._write_article_embeddings(articles_df)
        baseline_metrics = evaluate_article_text_baseline(
            self.driver,
            day_labels=self._extract_day_labels(price_df),
            lookback_days=self.config.experiment.feature.lookback_days,
            train_ratio=self.config.experiment.feature.train_ratio,
        )
        self.results["baseline_article_embedding"] = baseline_metrics
        logger.info(
            "Baseline (article text only): AUC=%.4f, F1=%.4f",
            baseline_metrics.auc,
            baseline_metrics.f1,
        )
        self._mlflow.log_baseline(baseline_metrics)
        self._mlflow.log_step_to_parent(baseline_metrics, step=0)

        # STEP 1+: Evolve and evaluate ontologies using incremental mutation
        # num_steps=0 means only base, num_steps=1 means base + 1 evolution step, etc.
        best_candidate = base_ontology
        best_auc: Optional[float] = None  # AUC of the last accepted step's winner
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
            step_winner = self._select_best_candidate_from_current_step(step)
            winner_metrics = self.results.get(step_winner.candidate_tag)

            accepted = self._should_accept_step(
                winner_auc=winner_metrics.auc if winner_metrics else None,
                best_auc=best_auc,
                drop_regressing_steps=self.config.experiment.drop_regressing_steps,
                tolerance=self.config.experiment.auc_drop_tolerance,
            )

            if accepted:
                best_candidate = step_winner
                if winner_metrics is not None:
                    best_auc = winner_metrics.auc
                    self._last_accepted_auc = winner_metrics.auc
                logger.info(
                    "Step %d accepted: %s becomes evolution parent (best AUC=%s)",
                    step,
                    step_winner.candidate_tag,
                    f"{best_auc:.4f}" if best_auc is not None else "n/a",
                )
            else:
                logger.info(
                    "Step %d REJECTED: winner %s AUC=%.4f is below last accepted "
                    "AUC=%.4f (tolerance=%.4f) — dropping step, keeping %s",
                    step,
                    step_winner.candidate_tag,
                    winner_metrics.auc,
                    best_auc,
                    self.config.experiment.auc_drop_tolerance,
                    best_candidate.candidate_tag,
                )
                # Losers were pruned in _run_step_incremental_evolution;
                # prune the winner too so the rejected step leaves no structure.
                self._prune_candidate_tags(step_winner.candidate_tag)

            self._finalize_step_addition_status(step, step_winner.candidate_tag, accepted)
            self._mlflow.tag_step_acceptance(self.candidates_per_step.get(step, []), accepted)
            self._mlflow.log_step_to_parent(
                self.results.get(best_candidate.candidate_tag), step=step
            )

        if self.addition_history:
            self._mlflow.log_addition_history(self.addition_history)

        logger.info("Experiment completed")

        # Save all ontologies and summary
        self._save_ontologies()

        return self.results

    async def _write_article_embeddings(self, articles_df: pd.DataFrame) -> None:
        """Embed all article texts and store on Article nodes for the baseline.

        Uses the same embedding type and model as the rest of the pipeline so
        dimensions are consistent. Matches Article nodes by text hash (same key
        used in the incremental mutator).

        Args:
            articles_df: Articles DataFrame with a 'text' column.
        """
        import os

        import numpy as np

        from kg_builder_llm.ml.embeddings import embed_text_deterministic

        api_key = os.getenv("OPENAI_API_KEY")
        embedding_type = self.config.experiment.feature.embedding_type
        local_model = self.config.experiment.feature.local_model_name

        if embedding_type == "openai" and not api_key:
            logger.warning(
                "OPENAI_API_KEY not set — skipping article embedding, baseline will be 0"
            )
            return

        logger.info("Embedding %d articles for baseline (%s)…", len(articles_df), embedding_type)
        rows = []
        for _, row in articles_df.iterrows():
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                emb = embed_text_deterministic(
                    text,
                    api_key=api_key,
                    embedding_type=embedding_type,
                    local_model=local_model,
                )
                emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
                rows.append({"id": article_text_id(text), "emb": emb_list})
            except Exception as e:
                logger.warning("Failed to embed article: %s", e)

        if rows:
            self.driver.run_query(
                """
                UNWIND $rows AS row
                MATCH (a:Article {id: row.id})
                SET a.text_embedding = row.emb
                """,
                parameters={"rows": rows},
            )

        logger.info("✓ Wrote text embeddings for %d / %d articles", len(rows), len(articles_df))

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
        best_metrics = self.results.get(best_candidate.candidate_tag, ModelMetrics.empty())

        # Generate evolved candidates
        candidates = []
        for variant_idx in range(self.config.experiment.max_candidates_per_step):
            evolved = await self.evolution_agent.propose_new_candidate(
                previous=best_candidate,
                metrics=best_metrics,
                step_index=step,
                variant_index=variant_idx,
                addition_history=self.addition_history,
            )
            candidates.append(evolved)
            if evolved.addition is not None:
                self.addition_history.append(
                    {
                        "step": step,
                        "variant": variant_idx,
                        "candidate_tag": evolved.candidate_tag,
                        "node": evolved.addition.get("node"),
                        "relationship": evolved.addition.get("relationship"),
                        "patterns": evolved.addition.get("patterns", []),
                        "auc": None,
                        "delta_auc": None,
                        "delta_reference": None,
                        "status": "pending",
                    }
                )
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
            self._deduplicate_graph(candidate.candidate_tag)

            # Build allowed tags for evaluation
            # Step 1+: evaluate with base + current + remaining tagged candidates from previous steps
            allowed_tags_for_eval = ["base_structure", candidate.candidate_tag] + [
                c.candidate_tag
                for prev_step in range(step)
                if prev_step in self.candidates_per_step
                for c in self.candidates_per_step[prev_step]
            ]

            logger.info("Evaluating with allowed tags: %s", allowed_tags_for_eval)

            # Build day labels for evaluation
            day_labels = self._extract_day_labels(price_df)

            metrics = evaluate_candidate(
                self.driver,
                self.embedder,
                [candidate.candidate_tag],
                day_labels,
                price_df,
                allowed_tags=allowed_tags_for_eval,
                embedding_type=self.config.experiment.feature.embedding_type,
                local_model=self.config.experiment.feature.local_model_name,
                lookback_days=self.config.experiment.feature.lookback_days,
                min_chain_hops=self.config.experiment.feature.min_chain_hops,
                max_chain_hops=self.config.experiment.feature.max_chain_hops,
                path_uniqueness=self.config.experiment.feature.path_uniqueness,
                feature_mode=self.config.experiment.feature.feature_mode,
                max_metapath_hops=self.config.experiment.feature.max_metapath_hops,
                train_ratio=self.config.experiment.feature.train_ratio,
            )

            self.results[candidate.candidate_tag] = metrics
            logger.info(
                "Candidate %s: AUC=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
                candidate.candidate_tag,
                metrics.auc,
                metrics.f1,
                getattr(metrics, "precision", 0.0),
                getattr(metrics, "recall", 0.0),
            )
            delta_auc, delta_reference = self._compute_delta_auc(metrics.auc)
            if candidate.addition is not None:
                self._record_addition_result(
                    candidate.candidate_tag, metrics.auc, delta_auc, delta_reference
                )
                logger.info(
                    "Addition %s + %s: ΔAUC=%s (vs %s)",
                    candidate.addition.get("node"),
                    candidate.addition.get("relationship"),
                    f"{delta_auc:+.4f}" if delta_auc is not None else "n/a",
                    delta_reference or "n/a",
                )
            self._mlflow.log_candidate(
                candidate,
                metrics,
                step,
                idx,
                nodes_added=nodes_added,
                rels_added=rels_added,
                nodes_total=nodes_after,
                rels_total=rels_after,
                delta_auc=delta_auc,
                delta_reference=delta_reference,
            )

        # Prune tags of losing candidates
        if self.results and self.candidates_per_step[step]:
            best_step = max(
                self.candidates_per_step[step],
                key=lambda c: self.results.get(c.candidate_tag, ModelMetrics.empty()).auc,
            )
            logger.info("Winner of step %d: %s", step, best_step.candidate_tag)
            self._mlflow.tag_step_winner(self.candidates_per_step[step], best_step.candidate_tag)

            # Prune tags from losing candidates (don't delete nodes/rels, just remove their tags)
            for candidate in self.candidates_per_step[step]:
                if candidate.candidate_tag != best_step.candidate_tag:
                    self._prune_candidate_tags(candidate.candidate_tag)
                    logger.info("Pruned tags from losing candidate: %s", candidate.candidate_tag)

    def _compute_delta_auc(self, candidate_auc: float) -> tuple[Optional[float], str]:
        """AUC change of a candidate vs the current reference graph.

        The reference is the last accepted step winner's AUC; before any step
        has been accepted it falls back to the article-text baseline. Returns
        (delta, reference_name), with delta None when no valid reference exists
        or either AUC is NaN.
        """
        if candidate_auc is None or math.isnan(candidate_auc):
            return None, ""
        if self._last_accepted_auc is not None and not math.isnan(self._last_accepted_auc):
            return candidate_auc - self._last_accepted_auc, "last_accepted_winner"
        baseline = self.results.get("baseline_article_embedding")
        if baseline is not None and not math.isnan(baseline.auc):
            return candidate_auc - baseline.auc, "baseline_article_embedding"
        return None, ""

    def _record_addition_result(
        self,
        candidate_tag: str,
        auc: float,
        delta_auc: Optional[float],
        delta_reference: str,
    ) -> None:
        """Fill in the evaluation outcome on a pending addition-history entry."""
        for entry in self.addition_history:
            if entry["candidate_tag"] == candidate_tag:
                entry["auc"] = auc
                entry["delta_auc"] = delta_auc
                entry["delta_reference"] = delta_reference or None
                return

    def _finalize_step_addition_status(self, step: int, winner_tag: str, accepted: bool) -> None:
        """Mark a step's additions as accepted (winner of an accepted step) or rejected."""
        for entry in self.addition_history:
            if entry["step"] == step:
                is_kept = accepted and entry["candidate_tag"] == winner_tag
                entry["status"] = "accepted" if is_kept else "rejected"

    @staticmethod
    def _should_accept_step(
        winner_auc: Optional[float],
        best_auc: Optional[float],
        drop_regressing_steps: bool,
        tolerance: float,
    ) -> bool:
        """Decide whether a step's winner replaces the current best candidate.

        A step is rejected only when the gate is enabled, a previous step has
        already been accepted, and the winner's AUC fell more than `tolerance`
        below the last accepted AUC. NaN AUCs (e.g. a single-class validation
        window) carry no evidence of regression, so they never reject a step.

        Args:
            winner_auc: AUC of this step's best candidate (None if unevaluated)
            best_auc: AUC of the last accepted step's winner (None before step 1)
            drop_regressing_steps: Whether the acceptance gate is enabled
            tolerance: Allowed AUC regression before a step is dropped

        Returns:
            True if the step's winner should become the evolution parent
        """
        if not drop_regressing_steps or best_auc is None or winner_auc is None:
            return True
        if math.isnan(winner_auc) or math.isnan(best_auc):
            return True
        return winner_auc >= best_auc - tolerance

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
            key=lambda c: self.results.get(c.candidate_tag, ModelMetrics.empty()).auc,
        )

        best_metrics = self.results.get(best_candidate.candidate_tag, ModelMetrics.empty())
        logger.info(
            "Selected best candidate from step %d: %s (AUC=%.4f)",
            step,
            best_candidate.candidate_tag,
            best_metrics.auc,
        )

        return best_candidate

    def _deduplicate_graph(self, candidate_tag: str) -> None:
        """Merge duplicate nodes created during candidate building.

        Runs two passes:
        1. Name-based merge: same label + same lowercased name → one node.
        2. Key-based merge: same label + same uppercased key → one node.

        Both passes use APOC refactor.mergeNodes to combine properties and
        relationships rather than deleting them.

        Args:
            candidate_tag: Tag of the just-built candidate (used only for logging).
        """
        logger.info("Running post-build deduplication for candidate %s …", candidate_tag)

        try:
            name_results = self.driver.run_query(MERGE_DUPLICATES_BY_NAME_CYPHER)
            merged_by_name = sum(r.get("merged", 0) for r in (name_results or []))
            logger.info("  Name-based merge: %d duplicate groups collapsed", merged_by_name)
        except Exception as e:
            logger.warning("Name-based deduplication failed (APOC required): %s", e)

        try:
            key_results = self.driver.run_query(MERGE_DUPLICATES_BY_KEY_CYPHER)
            merged_by_key = sum(r.get("merged", 0) for r in (key_results or []))
            logger.info("  Key-based merge: %d duplicate groups collapsed", merged_by_key)
        except Exception as e:
            logger.warning("Key-based deduplication failed (APOC required): %s", e)

    def _extract_day_labels(self, price_df: pd.DataFrame) -> Dict[str, int]:
        """Extract day-level labels from price data.

        Args:
            price_df: DataFrame with index 'day' and 'direction' column (or 'date' column)

        Returns:
            Dict mapping date to label (0 or 1)
        """
        labels = {}

        # Check if price_df is indexed by day or has a date column
        if price_df.index.name == "day":
            # Indexed by day
            for day, row in price_df.iterrows():
                day_str = str(day)
                # direction column should be 0 or 1
                label = int(row["direction"]) if "direction" in price_df.columns else 0
                labels[day_str] = label
        else:
            # Has date/day column
            for _, row in price_df.iterrows():
                day = str(row.get("date") or row.get("day", ""))
                label = int(row["direction"]) if "direction" in row else 0
                labels[day] = label

        logger.info("Extracted labels for %d days", len(labels))
        return labels

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
        RETURN true AS found
        LIMIT 1
        """
        result = self.driver.run_query(query, parameters={"tag": tag})
        return bool(result)

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
                metrics = self.results.get(candidate.candidate_tag, ModelMetrics.empty())
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

        if self.addition_history:
            history_path = self.ontologies_dir / "addition_history.json"
            history_path.write_text(json.dumps(self.addition_history, indent=2, default=str))
            logger.info("✓ Saved addition history: %s", history_path)

        logger.info("✓ Saved ontologies to %s", self.ontologies_dir)

        try:
            mlflow.log_artifacts(str(self.ontologies_dir), artifact_path="ontologies")
            logger.info("✓ Logged ontology artifacts to MLflow")
        except Exception as e:
            logger.warning("MLflow artifact logging failed: %s", e)
