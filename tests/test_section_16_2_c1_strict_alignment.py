"""Focused tests for opt-in strict temporal alignment in optimization."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.optimization.evaluator import (
    ForecastAlignmentError,
    OptimizationEvaluator,
)
from zenerestimation.optimization.validation import ExpandingWindowSplitter


class _Model:
    def __init__(self, dates=None):
        self.dates = dates

    def fit(self, dataset):
        return self

    def predict(self, steps):
        result = SimpleNamespace(forecast=np.array([1.0, 2.0]))
        if self.dates is not None:
            result.dates = self.dates
        return result


def _evaluator(monkeypatch, *, strict, date_mode):
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2025-03-01", "2025-06-01"]),
        "microVolt": [1.0, 2.0],
    })
    folds = [SimpleNamespace(index=i, train=object(), validation=frame,
                             validation_points=2) for i in range(2)]
    monkeypatch.setattr(splitter, "split", lambda dataset: folds)
    monkeypatch.setattr(OptimizationEvaluator, "_validate_inputs", staticmethod(
        lambda dataset, model_factory, params: None
    ))
    evaluator = OptimizationEvaluator(splitter, require_verified_alignment=strict)
    calls = iter(range(2))

    def factory(**params):
        i = next(calls)
        dates = frame["ds"] if date_mode == "verified" or (date_mode == "mixed" and i == 0) else None
        return _Model(dates)

    return evaluator, factory


def test_default_accepts_positional_only(monkeypatch):
    evaluator, factory = _evaluator(monkeypatch, strict=False, date_mode="missing")
    result = evaluator.evaluate(object(), factory, {"window": 4})
    assert result.status == "evaluated"
    assert result.score == pytest.approx(0)
    assert [x["status"] for x in result.metadata["fold_alignment"]] == [
        "positional_only", "positional_only"
    ]


def test_strict_accepts_verified_folds(monkeypatch):
    evaluator, factory = _evaluator(monkeypatch, strict=True, date_mode="verified")
    result = evaluator.evaluate(object(), factory, {"window": 4})
    assert result.status == "evaluated"
    assert result.score == pytest.approx(0)
    assert all(x["timestamp_match"] is True for x in result.metadata["fold_alignment"])


def test_strict_rejects_unverified_second_fold(monkeypatch):
    evaluator, factory = _evaluator(monkeypatch, strict=True, date_mode="mixed")
    with pytest.raises(ForecastAlignmentError, match=r"fold 1.*positional_only"):
        evaluator.evaluate(object(), factory, {"window": 4})


def test_strict_rejects_unverified_first_fold(monkeypatch):
    evaluator, factory = _evaluator(monkeypatch, strict=True, date_mode="missing")
    with pytest.raises(ForecastAlignmentError, match=r"fold 0.*positional_only"):
        evaluator.evaluate(object(), factory, {"window": 4})


def test_policy_type_is_validated():
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    with pytest.raises(TypeError, match="require_verified_alignment must be a boolean"):
        OptimizationEvaluator(splitter, require_verified_alignment="yes")
