"""Tests for pipeline modules."""

import pytest

from kg_builder_llm.pipeline.orchestrator import Orchestrator


class TestOrchestrator:
    """Tests for orchestrator."""
    
    def test_orchestrator_init(self, neo4j_config, openai_config, experiment_config, mock_llm):
        """Test orchestrator initialization."""
        from kg_builder_llm.config import Config
        from kg_builder_llm.core.graph import GraphDriver
        
        config = Config(
            neo4j=neo4j_config,
            openai=openai_config,
            experiment=experiment_config,
        )
        
        # Create a mock driver
        from unittest.mock import Mock
        mock_driver = Mock(spec=GraphDriver)
        
        orchestrator = Orchestrator(config, mock_driver, mock_llm, mock_llm)
        
        assert orchestrator.config == config
        assert orchestrator.results == {}


class TestCountGraphByTags:
    """Tests for the KG-size counting helper used by the KG-growth figure."""

    def test_counts_scoped_to_tags(self):
        from unittest.mock import Mock

        from kg_builder_llm.core.graph import GraphDriver
        from kg_builder_llm.pipeline.evaluator import count_graph_by_tags

        driver = Mock(spec=GraphDriver)
        driver.run_query.side_effect = [[{"c": 42}], [{"c": 99}]]

        n_nodes, n_rels = count_graph_by_tags(driver, ["base_structure", "step_1_candidate_0"])

        assert (n_nodes, n_rels) == (42, 99)
        # tags are passed as a bound parameter, not string-interpolated
        _, kwargs_or_params = driver.run_query.call_args_list[0][0]
        assert kwargs_or_params["tags"] == ["base_structure", "step_1_candidate_0"]

    def test_returns_zero_on_failure(self):
        from unittest.mock import Mock

        from kg_builder_llm.core.graph import GraphDriver
        from kg_builder_llm.pipeline.evaluator import count_graph_by_tags

        driver = Mock(spec=GraphDriver)
        driver.run_query.side_effect = RuntimeError("neo4j down")

        assert count_graph_by_tags(driver, ["base_structure"]) == (0, 0)
