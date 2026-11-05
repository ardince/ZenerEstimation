"""Milestone 16.2-C3: strict alignment through the full optimizer path.

These tests use deterministic forecasting doubles and real optimization objects;
only dataset/split construction is isolated to keep the tests fast.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.optimization.artifact import OptimizationArtifact
from zenerestimation.optimization.evaluator import OptimizationEvaluator
from zenerestimation.optimization.search import GenericOptimizer
from zenerestimation.optimization.validation import ExpandingWindowSplitter


class _ForecastModel:
    def __init__(self, *, dates_mode, bias, fold_index, frame):
        self.dates_mode = dates_mode
        self.bias = bias
        self.fold_index = fold_index
        self.frame = frame

    def fit(self, dataset):
        return self

    def predict(self, steps):
        result = SimpleNamespace(forecast=np.asarray([1.0, 2.0]) + self.bias)
        if self.dates_mode == "verified" or (
            self.dates_mode == "second_missing" and self.fold_index == 0
        ):
            result.dates = self.frame["ds"]
        elif self.dates_mode == "shifted":
            result.dates = self.frame["ds"] + pd.Timedelta(days=1)
        return result


def _run(monkeypatch, *, strict, candidates):
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2025-03-01", "2025-06-01"]),
        "microVolt": [1.0, 2.0],
    })
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    folds = [SimpleNamespace(index=i, train=object(), validation=frame,
                             validation_points=2) for i in range(2)]
    monkeypatch.setattr(splitter, "split", lambda dataset: folds)
    monkeypatch.setattr(OptimizationEvaluator, "_validate_inputs", staticmethod(
        lambda dataset, model_factory, params: None
    ))
    monkeypatch.setattr(GenericOptimizer, "_validate_dataset", staticmethod(
        lambda dataset: None
    ))
    evaluator = OptimizationEvaluator(splitter, require_verified_alignment=strict)
    optimizer = GenericOptimizer(evaluator, model="DeterministicTestModel")
    fold_calls = {}

    def factory(*, name, dates_mode, bias):
        index = fold_calls.get(name, 0)
        fold_calls[name] = index + 1
        return _ForecastModel(dates_mode=dates_mode, bias=bias,
                              fold_index=index % 2, frame=frame)

    return optimizer.optimize(object(), factory, candidates)


def _candidate(name, dates_mode, bias):
    return {"name": name, "dates_mode": dates_mode, "bias": bias}


def test_strict_rejects_better_positional_candidate_and_ranks_verified(monkeypatch):
    result = _run(monkeypatch, strict=True, candidates=[
        _candidate("unverified", "missing", 0.0),
        _candidate("verified_worse", "verified", 0.4),
        _candidate("verified_best", "verified", 0.2),
    ])
    failed, worse, best = result.candidates
    assert failed.status == "failed" and failed.score is None
    assert failed.metadata["error_type"] == "ForecastAlignmentError"
    assert failed.metadata["alignment_failure"] == {
        "reason": "missing_forecast_dates", "fold_index": 0,
        "required_status": "verified", "observed_status": "positional_only",
    }
    assert worse.status == best.status == "evaluated"
    assert result.best_params["name"] == "verified_best"
    assert result.best_score == pytest.approx(0.2)


def test_strict_rejects_when_only_second_fold_is_unverified(monkeypatch):
    result = _run(monkeypatch, strict=True, candidates=[
        _candidate("mixed", "second_missing", 0.0),
        _candidate("verified", "verified", 0.5),
    ])
    failed = result.candidates[0]
    assert failed.status == "failed"
    assert failed.metadata["alignment_failure"]["fold_index"] == 1
    assert result.best_params["name"] == "verified"


def test_default_mode_preserves_positional_candidate_eligibility(monkeypatch):
    result = _run(monkeypatch, strict=False, candidates=[
        _candidate("positional", "missing", 0.0),
        _candidate("verified", "verified", 0.3),
    ])
    assert all(candidate.status == "evaluated" for candidate in result.candidates)
    assert result.best_params["name"] == "positional"
    assert result.best_score == pytest.approx(0.0)


def test_shifted_dates_rejected_even_without_strict_mode(monkeypatch):
    result = _run(monkeypatch, strict=False, candidates=[
        _candidate("shifted", "shifted", 0.0),
        _candidate("verified", "verified", 0.3),
    ])
    assert result.candidates[0].status == "failed"
    assert result.best_params["name"] == "verified"


def test_artifact_excludes_raw_dates_but_keeps_failure_diagnostics(monkeypatch):
    result = _run(monkeypatch, strict=True, candidates=[
        _candidate("missing", "missing", 0.0),
        _candidate("verified", "verified", 0.3),
    ])
    payload = OptimizationArtifact(result).to_dict()
    failure = payload["candidates"][0]["metadata"]["alignment_failure"]
    assert failure["fold_index"] == 0
    assert failure["reason"] == "missing_forecast_dates"
    assert "forecast_dates" not in failure
    assert "validation_dates" not in failure
    for record in payload["candidates"][1]["metadata"]["fold_alignment"]:
        assert "forecast_dates" not in record
        assert "validation_dates" not in record
        assert record["status"] == "verified"


def test_deterministic_repeated_search_has_same_scores_and_statuses(monkeypatch):
    params = [
        _candidate("missing", "missing", 0.0),
        _candidate("a", "verified", 0.4),
        _candidate("b", "verified", 0.2),
    ]
    first = _run(monkeypatch, strict=True, candidates=params)
    second = _run(monkeypatch, strict=True, candidates=params)
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)
    assert [c.status for c in first.candidates] == [c.status for c in second.candidates]
    assert [c.score for c in first.candidates] == [c.score for c in second.candidates]
    assert [c.metadata.get("alignment_failure") for c in first.candidates] == [
        c.metadata.get("alignment_failure") for c in second.candidates
    ]
