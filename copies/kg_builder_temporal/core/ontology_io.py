"""Ontology I/O and serialization."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from copies.kg_builder_temporal.core.ontology import OntologyCandidate
from copies.kg_builder_temporal.ml.modeling import ModelMetrics

logger = logging.getLogger(__name__)


def save_ontology_candidate(
    candidate: OntologyCandidate,
    metrics: ModelMetrics,
    output_dir: Path,
) -> Path:
    """Save ontology candidate and metrics to JSON.

    Args:
        candidate: OntologyCandidate to save
        metrics: ModelMetrics for this candidate
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from candidate tag
    filename = f"{candidate.candidate_tag}.json"
    filepath = output_dir / filename

    # Prepare data
    data = {
        "candidate_tag": candidate.candidate_tag,
        "description": candidate.description,
        "step_index": candidate.step_index,
        "parent_tag": candidate.parent_tag,
        "schema": candidate.schema,
        "metrics": {
            "auc": float(metrics.auc),
            "f1": float(metrics.f1),
            "max_hops_train": int(metrics.max_hops_train),
            "max_hops_val": int(metrics.max_hops_val),
        },
    }

    # Write to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved ontology: %s", filepath)
    return filepath


def save_ontology_summary(
    candidates_by_step: Dict[int, List[OntologyCandidate]],
    metrics_by_tag: Dict[str, ModelMetrics],
    best_candidate_tag: str,
    output_dir: Path,
) -> Path:
    """Save summary of all ontology candidates and results.

    Args:
        candidates_by_step: Dict mapping step → list of candidates
        metrics_by_tag: Dict mapping candidate_tag → metrics
        best_candidate_tag: Tag of best candidate overall
        output_dir: Directory to save to

    Returns:
        Path to saved summary file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_candidate": best_candidate_tag,
        "steps": {},
    }

    # Build step-by-step summary
    for step, candidates in candidates_by_step.items():
        step_data = {"candidates": []}

        for candidate in candidates:
            metrics = metrics_by_tag.get(candidate.candidate_tag)
            if metrics:
                step_data["candidates"].append(
                    {
                        "tag": candidate.candidate_tag,
                        "description": candidate.description,
                        "auc": float(metrics.auc),
                        "f1": float(metrics.f1),
                        "max_hops_train": int(metrics.max_hops_train),
                        "max_hops_val": int(metrics.max_hops_val),
                    }
                )

        # Find best in this step
        if step_data["candidates"]:
            best_in_step = max(step_data["candidates"], key=lambda x: x["auc"])
            step_data["best"] = best_in_step["tag"]

        summary["steps"][f"step_{step}"] = step_data

    # Write summary
    summary_path = output_dir / "ontology_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved ontology summary: %s", summary_path)
    return summary_path


def load_ontology_candidate(filepath: Path) -> tuple:
    """Load ontology candidate from JSON.

    Args:
        filepath: Path to saved ontology JSON

    Returns:
        Tuple of (OntologyCandidate, ModelMetrics)
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract metrics
    metrics_data = data.pop("metrics")
    metrics = ModelMetrics(
        auc=metrics_data["auc"],
        f1=metrics_data["f1"],
        max_hops_train=metrics_data.get("max_hops_train", 0),
        max_hops_val=metrics_data.get("max_hops_val", 0),
    )

    # Reconstruct candidate
    candidate = OntologyCandidate(
        candidate_tag=data["candidate_tag"],
        schema=data["schema"],
        description=data["description"],
        parent_tag=data.get("parent_tag"),
        step_index=data["step_index"],
    )

    return candidate, metrics
