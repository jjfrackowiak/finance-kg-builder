"""Tests for single-addition ontology evolution mode."""

import json
import math

from kg_builder_llm.core.ontology import OntologyCandidate
from kg_builder_llm.ml.modeling import ModelMetrics
from kg_builder_llm.pipeline.ontology_evolution import OntologyEvolutionAgent
from kg_builder_llm.pipeline.orchestrator import Orchestrator

PARENT_SCHEMA = {
    "node_types": [
        {"label": "Company", "properties": [{"name": "symbol", "type": "STRING"}]},
        {"label": "Article", "properties": [{"name": "headline", "type": "STRING"}]},
    ],
    "relationship_types": ["MENTIONS"],
    "patterns": [["Article", "MENTIONS", "Company"]],
}


def _evolved(node_types, rel_types, patterns):
    return {
        "node_types": PARENT_SCHEMA["node_types"] + node_types,
        "relationship_types": PARENT_SCHEMA["relationship_types"] + rel_types,
        "patterns": PARENT_SCHEMA["patterns"] + patterns,
    }


class TestEnforceSingleAddition:
    def test_single_addition_passes_through(self):
        evolved = _evolved(
            [{"label": "CreditRating", "properties": [{"name": "grade", "type": "STRING"}]}],
            ["RATED_BY"],
            [["Company", "RATED_BY", "CreditRating"]],
        )
        schema, addition = OntologyEvolutionAgent.enforce_single_addition(evolved, PARENT_SCHEMA)

        assert addition == {
            "node": "CreditRating",
            "relationship": "RATED_BY",
            "patterns": [["Company", "RATED_BY", "CreditRating"]],
        }
        assert [n["label"] for n in schema["node_types"]] == ["Company", "Article", "CreditRating"]
        assert schema["relationship_types"] == ["MENTIONS", "RATED_BY"]
        assert ["Company", "RATED_BY", "CreditRating"] in schema["patterns"]

    def test_extra_additions_are_trimmed_to_first(self):
        evolved = _evolved(
            [
                {"label": "CreditRating", "properties": []},
                {"label": "MacroEvent", "properties": []},
            ],
            ["RATED_BY", "AFFECTED_BY"],
            [
                ["Company", "RATED_BY", "CreditRating"],
                ["Company", "AFFECTED_BY", "MacroEvent"],
            ],
        )
        schema, addition = OntologyEvolutionAgent.enforce_single_addition(evolved, PARENT_SCHEMA)

        assert addition["node"] == "CreditRating"
        assert addition["relationship"] == "RATED_BY"
        labels = [n["label"] for n in schema["node_types"]]
        assert "MacroEvent" not in labels
        assert "AFFECTED_BY" not in schema["relationship_types"]
        # Pattern referencing the dropped node/rel must not survive
        assert ["Company", "AFFECTED_BY", "MacroEvent"] not in schema["patterns"]

    def test_rel_only_addition(self):
        evolved = _evolved([], ["COMPETES_WITH"], [["Company", "COMPETES_WITH", "Company"]])
        schema, addition = OntologyEvolutionAgent.enforce_single_addition(evolved, PARENT_SCHEMA)

        assert addition["node"] is None
        assert addition["relationship"] == "COMPETES_WITH"
        assert addition["patterns"] == [["Company", "COMPETES_WITH", "Company"]]

    def test_parent_schema_is_never_mutated_or_shrunk(self):
        evolved = {
            "node_types": [{"label": "CreditRating", "properties": []}],  # dropped parent types
            "relationship_types": ["RATED_BY"],
            "patterns": [["Company", "RATED_BY", "CreditRating"]],
        }
        schema, _ = OntologyEvolutionAgent.enforce_single_addition(evolved, PARENT_SCHEMA)

        labels = [n["label"] for n in schema["node_types"]]
        assert "Company" in labels and "Article" in labels
        assert "MENTIONS" in schema["relationship_types"]
        assert ["Article", "MENTIONS", "Company"] in schema["patterns"]
        assert PARENT_SCHEMA["relationship_types"] == ["MENTIONS"]

    def test_pattern_with_unknown_endpoint_is_dropped(self):
        evolved = _evolved(
            [{"label": "CreditRating", "properties": []}],
            ["RATED_BY"],
            [["Hallucinated", "RATED_BY", "CreditRating"]],
        )
        _, addition = OntologyEvolutionAgent.enforce_single_addition(evolved, PARENT_SCHEMA)
        assert addition["patterns"] == []

    def test_no_addition_at_all(self):
        schema, addition = OntologyEvolutionAgent.enforce_single_addition(
            PARENT_SCHEMA, PARENT_SCHEMA
        )
        assert addition == {"node": None, "relationship": None, "patterns": []}
        assert schema["patterns"] == PARENT_SCHEMA["patterns"]


class TestAdditionHistoryFormatting:
    def test_empty_history(self):
        assert OntologyEvolutionAgent._format_addition_history([]) == "  (none yet)"

    def test_entries_with_and_without_delta(self):
        history = [
            {
                "node": "CreditRating",
                "relationship": "RATED_BY",
                "delta_auc": 0.05,
                "status": "accepted",
            },
            {
                "node": "MacroEvent",
                "relationship": "AFFECTED_BY",
                "delta_auc": None,
                "status": "pending",
            },
        ]
        text = OntologyEvolutionAgent._format_addition_history(history)
        assert "CreditRating + RATED_BY: ΔAUC=+0.0500 (accepted)" in text
        assert "MacroEvent + AFFECTED_BY: result pending (pending)" in text

    def test_rel_only_entry(self):
        history = [
            {
                "node": None,
                "relationship": "COMPETES_WITH",
                "delta_auc": -0.02,
                "status": "rejected",
            },
        ]
        text = OntologyEvolutionAgent._format_addition_history(history)
        assert "COMPETES_WITH (rel-only, between existing types): ΔAUC=-0.0200 (rejected)" in text


class TestSingleAdditionPrompt:
    def test_prompt_offers_rel_only_option(self):
        agent = OntologyEvolutionAgent(llm=None, single_addition=True)
        prompt = agent._create_single_addition_prompt(
            _parent_candidate(),
            ModelMetrics.empty(),
            step_index=1,
            variant_index=0,
            addition_history=[],
        )
        assert "one new relationship type between EXISTING node types" in prompt
        assert "exactly 1 new relationship type and NO new node type" in prompt
        # The multi-addition default prompt must remain unchanged
        default_prompt = agent._create_evolution_prompt(
            _parent_candidate(), ModelMetrics.empty(), step_index=1, variant_index=0
        )
        assert "propose 3–5 new node types" in default_prompt


class _FakeLLM:
    """Returns queued schema JSON responses, recording call count."""

    def __init__(self, schemas):
        self._schemas = list(schemas)
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        # Serve the last schema repeatedly once the queue is exhausted
        schema = self._schemas[min(self.calls - 1, len(self._schemas) - 1)]
        return json.dumps(schema)


def _parent_candidate():
    return OntologyCandidate(candidate_tag="step_1_candidate_0", schema=PARENT_SCHEMA)


class TestDuplicateAdditionRetry:
    async def test_retries_until_fresh_combo(self):
        duplicate = _evolved(
            [{"label": "CreditRating", "properties": []}],
            ["RATED_BY"],
            [["Company", "RATED_BY", "CreditRating"]],
        )
        fresh = _evolved(
            [{"label": "MacroEvent", "properties": []}],
            ["AFFECTED_BY"],
            [["Company", "AFFECTED_BY", "MacroEvent"]],
        )
        llm = _FakeLLM([duplicate, fresh])
        agent = OntologyEvolutionAgent(llm, single_addition=True)
        history = [{"node": "CreditRating", "relationship": "RATED_BY", "status": "pending"}]

        candidate = await agent.propose_new_candidate(
            previous=_parent_candidate(),
            metrics=ModelMetrics.empty(),
            step_index=2,
            variant_index=1,
            addition_history=history,
        )

        assert llm.calls == 2
        assert candidate.addition["node"] == "MacroEvent"
        assert candidate.addition["relationship"] == "AFFECTED_BY"

    async def test_gives_up_after_max_attempts(self):
        duplicate = _evolved(
            [{"label": "CreditRating", "properties": []}],
            ["RATED_BY"],
            [["Company", "RATED_BY", "CreditRating"]],
        )
        llm = _FakeLLM([duplicate])
        agent = OntologyEvolutionAgent(llm, single_addition=True)
        history = [{"node": "CreditRating", "relationship": "RATED_BY", "status": "pending"}]

        candidate = await agent.propose_new_candidate(
            previous=_parent_candidate(),
            metrics=ModelMetrics.empty(),
            step_index=2,
            variant_index=1,
            addition_history=history,
        )

        assert llm.calls == 3
        # Keeps the duplicate rather than failing the run
        assert candidate.addition["node"] == "CreditRating"

    async def test_no_retry_when_first_combo_is_new(self):
        fresh = _evolved(
            [{"label": "MacroEvent", "properties": []}],
            ["AFFECTED_BY"],
            [["Company", "AFFECTED_BY", "MacroEvent"]],
        )
        llm = _FakeLLM([fresh])
        agent = OntologyEvolutionAgent(llm, single_addition=True)

        candidate = await agent.propose_new_candidate(
            previous=_parent_candidate(),
            metrics=ModelMetrics.empty(),
            step_index=1,
            variant_index=0,
            addition_history=[],
        )

        assert llm.calls == 1
        assert candidate.addition["node"] == "MacroEvent"


class _Metrics:
    def __init__(self, auc):
        self.auc = auc


def _bare_orchestrator(last_accepted=None, baseline=None):
    orch = object.__new__(Orchestrator)
    orch._last_accepted_auc = last_accepted
    orch.results = {}
    if baseline is not None:
        orch.results["baseline_article_embedding"] = _Metrics(baseline)
    orch.addition_history = []
    return orch


class TestDeltaAuc:
    def test_delta_vs_last_accepted(self):
        orch = _bare_orchestrator(last_accepted=0.5, baseline=0.2)
        delta, ref = orch._compute_delta_auc(0.6)
        assert math.isclose(delta, 0.1)
        assert ref == "last_accepted_winner"

    def test_delta_falls_back_to_baseline(self):
        orch = _bare_orchestrator(last_accepted=None, baseline=0.2)
        delta, ref = orch._compute_delta_auc(0.5)
        assert math.isclose(delta, 0.3)
        assert ref == "baseline_article_embedding"

    def test_no_reference(self):
        orch = _bare_orchestrator()
        assert orch._compute_delta_auc(0.5) == (None, "")

    def test_nan_candidate(self):
        orch = _bare_orchestrator(last_accepted=0.5)
        assert orch._compute_delta_auc(float("nan")) == (None, "")


class TestAdditionStatus:
    def test_winner_of_accepted_step_is_accepted_others_rejected(self):
        orch = _bare_orchestrator()
        orch.addition_history = [
            {"step": 1, "candidate_tag": "step_1_candidate_0", "status": "pending"},
            {"step": 1, "candidate_tag": "step_1_candidate_1", "status": "pending"},
            {"step": 2, "candidate_tag": "step_2_candidate_0", "status": "pending"},
        ]
        orch._finalize_step_addition_status(1, "step_1_candidate_0", accepted=True)
        assert orch.addition_history[0]["status"] == "accepted"
        assert orch.addition_history[1]["status"] == "rejected"
        assert orch.addition_history[2]["status"] == "pending"

    def test_rejected_step_marks_winner_rejected(self):
        orch = _bare_orchestrator()
        orch.addition_history = [
            {"step": 2, "candidate_tag": "step_2_candidate_0", "status": "pending"},
        ]
        orch._finalize_step_addition_status(2, "step_2_candidate_0", accepted=False)
        assert orch.addition_history[0]["status"] == "rejected"

    def test_record_addition_result(self):
        orch = _bare_orchestrator()
        orch.addition_history = [
            {
                "step": 1,
                "candidate_tag": "step_1_candidate_0",
                "auc": None,
                "delta_auc": None,
                "delta_reference": None,
                "status": "pending",
            },
        ]
        orch._record_addition_result("step_1_candidate_0", 0.6, 0.1, "last_accepted_winner")
        entry = orch.addition_history[0]
        assert entry["auc"] == 0.6
        assert entry["delta_auc"] == 0.1
        assert entry["delta_reference"] == "last_accepted_winner"
