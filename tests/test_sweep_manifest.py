"""Tests for sweep.py's per-config CLI passthrough (no k8s/AWS required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kg_builder_llm" / "scripts"))

from sweep import build_job_manifest, build_optional_args  # noqa: E402


def _main_container_args(cfg, **kwargs):
    job = build_job_manifest(
        sweep_id="abc123",
        run_index=0,
        cfg=cfg,
        parent_run_id="parent-run",
        image="kg-orchestrator:latest",
        cpu_request="500m",
        memory_request="2Gi",
        stub=False,
        **kwargs,
    )
    kg_builder = next(c for c in job.spec.template.spec.containers if c.name == "kg-builder")
    return kg_builder.args[0]


class TestBuildOptionalArgs:
    def test_empty_cfg_produces_no_args(self):
        assert build_optional_args({}) == ""

    def test_single_addition_flag(self):
        assert " --single-addition" in build_optional_args({"single_addition": True})

    def test_single_addition_false_omitted(self):
        assert "--single-addition" not in build_optional_args({"single_addition": False})

    def test_keep_regressing_steps_flag(self):
        assert " --keep-regressing-steps" in build_optional_args({"keep_regressing_steps": True})

    def test_auc_drop_tolerance_value(self):
        assert " --auc-drop-tolerance 0.02" in build_optional_args({"auc_drop_tolerance": 0.02})

    def test_multiple_options_combine(self):
        args = build_optional_args(
            {"single_addition": True, "embedding_type": "openai", "train_ratio": 0.8}
        )
        assert " --single-addition" in args
        assert " --embedding-type openai" in args
        assert " --train-ratio 0.8" in args


class TestBuildJobManifestPassthrough:
    def test_single_addition_reaches_container_command(self):
        cmd = _main_container_args({"steps": 2, "candidates": 2, "single_addition": True})
        assert "--single-addition" in cmd

    def test_auc_drop_tolerance_reaches_container_command(self):
        cmd = _main_container_args({"auc_drop_tolerance": 0.03})
        assert "--auc-drop-tolerance 0.03" in cmd

    def test_default_config_has_neither_flag(self):
        cmd = _main_container_args({"steps": 1, "candidates": 1})
        assert "--single-addition" not in cmd
        assert "--keep-regressing-steps" not in cmd

    def test_base_args_still_present(self):
        cmd = _main_container_args({"steps": 3, "candidates": 2, "feature_mode": "subgraph"})
        assert "--steps 3" in cmd
        assert "--candidates 2" in cmd
        assert "--feature-mode subgraph" in cmd
