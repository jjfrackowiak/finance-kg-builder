"""MLflow logging for the experiment orchestrator.

All methods swallow MLflow errors: tracking failures must never abort an
experiment run.
"""

import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from kg_builder_llm.core.ontology import OntologyCandidate

logger = logging.getLogger(__name__)


class MlflowExperimentLogger:
    """Logs baselines, candidates, and step outcomes as nested MLflow runs.

    Instantiate while the parent MLflow run is active: the parent run id is
    captured at construction time so step-level metrics can still be logged to
    it from inside nested child runs.
    """

    def __init__(self) -> None:
        parent_run = mlflow.active_run()
        self._parent_run_id = parent_run.info.run_id if parent_run else None
        self._client = MlflowClient() if self._parent_run_id else None
        self._run_ids: Dict[str, str] = {}

    def log_step_to_parent(self, metrics, step: int) -> None:
        """Log a step's headline metrics onto the parent run's timeline."""
        if not (self._client and self._parent_run_id) or metrics is None:
            return
        try:
            ts = int(time.time() * 1000)
            # Per-block feature coverage rides along on the parent timeline so a
            # block that silently went to all zeros is visible per step, not
            # only buried in pod logs that get garbage-collected after an hour.
            block_stats = getattr(metrics, "block_stats", None) or {}
            for key, val in {
                **block_stats,
                "auc": metrics.auc,
                "f1": metrics.f1,
                "precision": getattr(metrics, "precision", 0.0),
                "recall": getattr(metrics, "recall", 0.0),
                "brier_score": getattr(metrics, "brier_score", 0.0),
                "train_auc": getattr(metrics, "train_auc", 0.0),
                "train_brier": getattr(metrics, "train_brier", 0.0),
                "train_val_gap": getattr(metrics, "train_auc", 0.0) - metrics.auc,
                "max_hops_train": getattr(metrics, "max_hops_train", 0),
                "max_hops_val": getattr(metrics, "max_hops_val", 0),
                "n_train_days": getattr(metrics, "n_train_days", 0),
                "graph/n_nodes": getattr(metrics, "n_nodes_total", 0),
                "graph/n_edges": getattr(metrics, "n_rels_total", 0),
            }.items():
                self._client.log_metric(
                    self._parent_run_id, key, float(val), timestamp=ts, step=step
                )
        except Exception as e:
            logger.warning("MLflow parent step logging failed at step %d: %s", step, e)

    def log_baseline(self, metrics) -> None:
        """Log the article-text baseline as a nested MLflow run."""
        try:
            with mlflow.start_run(run_name="baseline_article_embedding", nested=True):
                mlflow.log_params({"candidate_tag": "baseline_article_embedding", "step": 0})
                mlflow.log_metrics(
                    {
                        "auc": metrics.auc,
                        "f1": metrics.f1,
                        "precision": getattr(metrics, "precision", 0.0),
                        "recall": getattr(metrics, "recall", 0.0),
                        "brier_score": getattr(metrics, "brier_score", 0.0),
                        "train_auc": getattr(metrics, "train_auc", 0.0),
                        "train_brier": getattr(metrics, "train_brier", 0.0),
                        "train_val_gap": getattr(metrics, "train_auc", 0.0) - metrics.auc,
                        "n_train_days": getattr(metrics, "n_train_days", 0),
                        "n_val_days": getattr(metrics, "n_val_days", 0),
                    },
                    step=0,
                )
                mlflow.set_tag("winner", "false")
        except Exception as e:
            logger.warning("MLflow baseline logging failed: %s", e)

    def log_candidate(
        self,
        candidate: OntologyCandidate,
        metrics,
        step: int,
        variant_idx: int,
        nodes_added: int = 0,
        rels_added: int = 0,
        nodes_total: int = 0,
        rels_total: int = 0,
        delta_auc: Optional[float] = None,
        delta_reference: str = "",
    ) -> None:
        """Log a single candidate's metrics and ontology params as a nested MLflow run."""
        try:
            node_labels = [
                n.get("label", "") if isinstance(n, dict) else str(n)
                for n in candidate.schema.get("node_types", [])
            ]
            rel_labels = [
                r.get("label", "") if isinstance(r, dict) else str(r)
                for r in candidate.schema.get("relationship_types", [])
            ]
            with mlflow.start_run(run_name=candidate.candidate_tag, nested=True) as run:
                mlflow.log_params(
                    {
                        "candidate_tag": candidate.candidate_tag,
                        "step": step,
                        "variant_idx": variant_idx,
                        "parent_tag": candidate.parent_tag or "base",
                        "node_types": ",".join(node_labels),
                        "rel_types": ",".join(rel_labels),
                        "n_node_types": len(node_labels),
                        "n_rel_types": len(rel_labels),
                        "description": candidate.description[:250],
                    }
                )
                if candidate.addition is not None:
                    mlflow.log_params(
                        {
                            "addition_node": candidate.addition.get("node") or "none",
                            "addition_rel": candidate.addition.get("relationship") or "none",
                        }
                    )
                if delta_auc is not None:
                    mlflow.log_metric("delta_auc", delta_auc, step=step)
                    mlflow.set_tag("delta_reference", delta_reference)
                mlflow.log_metrics(
                    {
                        "auc": metrics.auc,
                        "f1": metrics.f1,
                        "precision": getattr(metrics, "precision", 0.0),
                        "recall": getattr(metrics, "recall", 0.0),
                        "brier_score": getattr(metrics, "brier_score", 0.0),
                        "train_auc": getattr(metrics, "train_auc", 0.0),
                        "train_brier": getattr(metrics, "train_brier", 0.0),
                        "train_val_gap": getattr(metrics, "train_auc", 0.0) - metrics.auc,
                        "max_hops_train": metrics.max_hops_train,
                        "max_hops_val": metrics.max_hops_val,
                        "n_train_days": getattr(metrics, "n_train_days", 0),
                        "n_val_days": getattr(metrics, "n_val_days", 0),
                        "graph/n_nodes": nodes_total,
                        "graph/n_edges": rels_total,
                        "graph/nodes_added": nodes_added,
                        "graph/rels_added": rels_added,
                    },
                    step=step,
                )

                # Feature importances artifact
                fi = getattr(metrics, "feature_importances", None)
                if fi:
                    self._log_json_artifact(
                        {"feature_importances": fi},
                        prefix=f"fi_step{step}_v{variant_idx}_",
                        artifact_path="feature_importances",
                    )

                # Per-candidate ontology snapshot (survives mid-run failure)
                self._log_json_artifact(
                    {
                        "candidate_tag": candidate.candidate_tag,
                        "step": step,
                        "schema": candidate.schema,
                        "description": candidate.description,
                        "metrics": {
                            "auc": metrics.auc,
                            "f1": metrics.f1,
                            "precision": getattr(metrics, "precision", 0.0),
                        },
                    },
                    prefix=f"ontology_step{step}_v{variant_idx}_",
                    artifact_path="ontologies",
                )

                self._run_ids[candidate.candidate_tag] = run.info.run_id
        except Exception as e:
            logger.warning("MLflow logging failed for %s: %s", candidate.candidate_tag, e)

    def tag_step_winner(self, candidates: List[OntologyCandidate], winner_tag: str) -> None:
        """Tag the winner and losers of a step on their (already closed) nested runs."""
        try:
            client = MlflowClient()
            for candidate in candidates:
                run_id = self._run_ids.get(candidate.candidate_tag)
                if run_id:
                    is_winner = candidate.candidate_tag == winner_tag
                    client.set_tag(run_id, "winner", str(is_winner).lower())
                    client.set_tag(run_id, "step_winner", winner_tag)
        except Exception as e:
            logger.warning("MLflow winner tagging failed: %s", e)

    def tag_step_acceptance(self, candidates: List[OntologyCandidate], accepted: bool) -> None:
        """Tag all of a step's nested runs with whether the step was accepted."""
        try:
            client = MlflowClient()
            for candidate in candidates:
                run_id = self._run_ids.get(candidate.candidate_tag)
                if run_id:
                    client.set_tag(run_id, "step_accepted", str(accepted).lower())
        except Exception as e:
            logger.warning("MLflow step-acceptance tagging failed: %s", e)

    def log_addition_history(self, history: List[dict]) -> None:
        """Log the single-addition history as an artifact on the parent run."""
        if not (self._client and self._parent_run_id):
            return
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="addition_history_"
            ) as f:
                json.dump(history, f, indent=2, default=str)
                tmp_path = f.name
            self._client.log_artifact(self._parent_run_id, tmp_path, artifact_path="additions")
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning("MLflow addition-history logging failed: %s", e)

    @staticmethod
    def _log_json_artifact(payload: dict, prefix: str, artifact_path: str) -> None:
        """Write a dict to a temp JSON file and log it as an MLflow artifact."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=prefix
        ) as f:
            json.dump(payload, f, indent=2, default=str)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
        os.unlink(tmp_path)
