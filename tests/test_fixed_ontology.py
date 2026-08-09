"""Tests for the fixed-ontology (out-of-sample) run mode."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from kg_builder_llm.config import Config, ExperimentConfig, Neo4jConfig, OpenAIConfig
from kg_builder_llm.core.ontology_io import load_ontology_candidate
from kg_builder_llm.ml.modeling import ModelMetrics
from kg_builder_llm.pipeline.orchestrator import Orchestrator

SWEEP_BEST = Path("resources/ontologies/tsla_best_step_1_candidate_1.json")


def _make_config(fixed_ontology: str) -> Config:
    return Config(
        neo4j=Neo4jConfig(uri="neo4j://localhost:7687", user="neo4j", password="password"),
        openai=OpenAIConfig(api_key="sk-test"),
        experiment=ExperimentConfig(num_steps=3, fixed_ontology=fixed_ontology),
    )


def _make_orchestrator(fixed_ontology: str) -> Orchestrator:
    driver = Mock()
    driver.run_query = Mock(return_value=[])
    driver.get_count = Mock(return_value=10)
    driver.get_relationship_count = Mock(return_value=5)
    orch = Orchestrator(_make_config(fixed_ontology), driver, Mock(), Mock())
    orch._mlflow = Mock()
    return orch


class TestSweepBestOntologyArtifact:
    """The committed sweep winner must stay loadable — the OOS job depends on it."""

    def test_loads_and_carries_the_evolved_addition(self):
        candidate, metrics = load_ontology_candidate(SWEEP_BEST)

        labels = [n["label"] for n in candidate.schema["node_types"]]
        assert "EarningsReport" in labels, "evolved node type missing from the schema"
        assert "HAS_EARNINGS" in candidate.schema["relationship_types"]
        assert ["Company", "HAS_EARNINGS", "EarningsReport"] in candidate.schema["patterns"]
        assert metrics.auc == pytest.approx(0.7871853546910755)

    def test_provenance_is_recorded(self):
        data = json.loads(SWEEP_BEST.read_text())
        prov = data["provenance"]
        assert prov["mlflow_run_id"]
        assert prov["source_candidate_tag"] == "step_1_candidate_1"


class TestFixedOntologyMode:
    """`--fixed-ontology` must build exactly one graph and never evolve."""

    def test_missing_file_fails_loudly(self):
        orch = _make_orchestrator("resources/ontologies/does_not_exist.json")
        with pytest.raises(FileNotFoundError):
            asyncio.run(orch._run_fixed_ontology(pd.DataFrame(), pd.DataFrame()))

    def test_builds_once_and_retags_the_candidate(self):
        orch = _make_orchestrator(str(SWEEP_BEST))
        metrics = ModelMetrics(auc=0.61, f1=0.5, max_hops_train=3, max_hops_val=3)

        with (
            patch(
                "kg_builder_llm.pipeline.orchestrator.build_kg_incremental_candidate",
                new=AsyncMock(),
            ) as build,
            patch("kg_builder_llm.pipeline.orchestrator.tag_candidate_entities"),
            patch("kg_builder_llm.pipeline.orchestrator.evaluate_candidate", return_value=metrics),
            patch.object(Orchestrator, "_deduplicate_graph"),
            patch.object(Orchestrator, "_extract_day_labels", return_value={}),
        ):
            candidate = asyncio.run(orch._run_fixed_ontology(pd.DataFrame(), pd.DataFrame()))

        assert build.await_count == 1, "fixed-ontology mode must build the graph exactly once"
        # Only base_structure is linkable: there are no sibling candidates to isolate from.
        assert build.await_args.args[-1] == ["base_structure"]
        assert candidate.candidate_tag == "fixed_ontology"
        assert candidate.parent_tag == "base_structure"
        assert orch.results["fixed_ontology"] is metrics
        assert orch.candidates_per_step == {1: [candidate]}

    def test_no_evolution_agent_call(self):
        """num_steps is set to 3 in the fixture; the mode must still not evolve."""
        orch = _make_orchestrator(str(SWEEP_BEST))
        orch.evolution_agent.propose_new_candidate = AsyncMock()
        metrics = ModelMetrics(auc=0.61, f1=0.5, max_hops_train=3, max_hops_val=3)

        with (
            patch(
                "kg_builder_llm.pipeline.orchestrator.build_kg_incremental_candidate",
                new=AsyncMock(),
            ),
            patch("kg_builder_llm.pipeline.orchestrator.tag_candidate_entities"),
            patch("kg_builder_llm.pipeline.orchestrator.evaluate_candidate", return_value=metrics),
            patch.object(Orchestrator, "_deduplicate_graph"),
            patch.object(Orchestrator, "_extract_day_labels", return_value={}),
        ):
            asyncio.run(orch._run_fixed_ontology(pd.DataFrame(), pd.DataFrame()))

        orch.evolution_agent.propose_new_candidate.assert_not_awaited()
        assert orch.addition_history == []
