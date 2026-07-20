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
