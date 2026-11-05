"""Milestone 16.2-B integration regression checks."""
from pathlib import Path


def test_holdout_uses_evidence_producing_validator():
    from zenerestimation.evaluation.evaluator import ForecastEvaluator
    import inspect
    source = inspect.getsource(ForecastEvaluator._evaluate)
    assert "validate_forecast_alignment_with_evidence" in source
    assert '"forecast_alignment": alignment_evidence.to_dict()' in source


def test_optimization_records_per_fold_evidence():
    from zenerestimation.optimization.evaluator import OptimizationEvaluator
    import inspect
    source = inspect.getsource(OptimizationEvaluator.evaluate)
    assert "validate_forecast_alignment_with_evidence" in source
    assert '"fold_alignment": fold_alignment' in source
    assert '"fold_index": fold.index' in source
