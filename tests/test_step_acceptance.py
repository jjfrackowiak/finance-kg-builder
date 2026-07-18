"""Tests for the step-acceptance gate (drop steps whose AUC regresses)."""

from kg_builder_llm.pipeline.orchestrator import Orchestrator


class TestShouldAcceptStep:
    """Tests for Orchestrator._should_accept_step."""

    def test_first_step_always_accepted(self):
        """With no previously accepted AUC, the step establishes the reference."""
        assert Orchestrator._should_accept_step(
            winner_auc=0.4, best_auc=None, drop_regressing_steps=True, tolerance=0.0
        )

    def test_improving_step_accepted(self):
        assert Orchestrator._should_accept_step(
            winner_auc=0.62, best_auc=0.55, drop_regressing_steps=True, tolerance=0.0
        )

    def test_equal_auc_accepted(self):
        assert Orchestrator._should_accept_step(
            winner_auc=0.55, best_auc=0.55, drop_regressing_steps=True, tolerance=0.0
        )

    def test_regressing_step_rejected(self):
        assert not Orchestrator._should_accept_step(
            winner_auc=0.50, best_auc=0.55, drop_regressing_steps=True, tolerance=0.0
        )

    def test_tolerance_allows_small_regression(self):
        assert Orchestrator._should_accept_step(
            winner_auc=0.53, best_auc=0.55, drop_regressing_steps=True, tolerance=0.02
        )
        assert not Orchestrator._should_accept_step(
            winner_auc=0.52, best_auc=0.55, drop_regressing_steps=True, tolerance=0.02
        )

    def test_gate_disabled_accepts_regression(self):
        assert Orchestrator._should_accept_step(
            winner_auc=0.30, best_auc=0.70, drop_regressing_steps=False, tolerance=0.0
        )

    def test_nan_winner_auc_accepted(self):
        """NaN AUC (e.g. single-class validation window) is not evidence of regression."""
        assert Orchestrator._should_accept_step(
            winner_auc=float("nan"), best_auc=0.55, drop_regressing_steps=True, tolerance=0.0
        )

    def test_nan_best_auc_accepted(self):
        assert Orchestrator._should_accept_step(
            winner_auc=0.55, best_auc=float("nan"), drop_regressing_steps=True, tolerance=0.0
        )
        assert Orchestrator._should_accept_step(
            winner_auc=float("nan"), best_auc=float("nan"), drop_regressing_steps=True, tolerance=0.0
        )

    def test_unevaluated_winner_accepted(self):
        """A winner with no metrics (e.g. fallback base ontology) is not rejected."""
        assert Orchestrator._should_accept_step(
            winner_auc=None, best_auc=0.55, drop_regressing_steps=True, tolerance=0.0
        )
