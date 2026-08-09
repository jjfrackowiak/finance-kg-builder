"""Tests for _make_run_name distinguishing gate/single-addition configs."""

import argparse

from kg_builder_llm.main import _make_run_name


class _Config:
    class experiment:
        target_ticker = "TSLA"


def _args(**overrides):
    defaults = dict(
        day_start=None,
        day_end=None,
        time_window_days=30,
        steps=2,
        candidates=2,
        feature_mode="path",
        evolution_prompt="default",
        single_addition=False,
        keep_regressing_steps=False,
        fixed_ontology="",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMakeRunName:
    def test_default_config_has_no_suffix(self):
        name = _make_run_name(_args(), _Config())
        assert name == "TSLA-steps2-cands2-path-default-w30d"

    def test_single_addition_adds_suffix(self):
        name = _make_run_name(_args(single_addition=True), _Config())
        assert name.endswith("-singleadd")

    def test_keep_regressing_steps_adds_suffix(self):
        name = _make_run_name(_args(keep_regressing_steps=True), _Config())
        assert name.endswith("-keepregress")

    def test_both_flags_combine_and_are_distinguishable_from_default(self):
        default_name = _make_run_name(_args(), _Config())
        both_name = _make_run_name(
            _args(single_addition=True, keep_regressing_steps=True), _Config()
        )
        assert both_name != default_name
        assert "-singleadd" in both_name
        assert "-keepregress" in both_name

    def test_fixed_ontology_names_the_run_oos(self):
        """OOS runs must not be mistaken for sweep runs when pulling MLflow."""
        name = _make_run_name(
            _args(
                fixed_ontology="resources/ontologies/tsla_best_step_1_candidate_1.json",
                day_start="2023-08-16",
                feature_mode="hybrid",
            ),
            _Config(),
        )
        assert name == "OOS-TSLA-tsla_best_step_1_candidate_1-hybrid-2023-08-16"
        # steps/candidates/prompt are meaningless without evolution — keep them out.
        assert "steps" not in name and "cands" not in name
