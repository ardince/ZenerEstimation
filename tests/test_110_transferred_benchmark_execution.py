"""G.3.2 benchmark orchestration tests: no neural training."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from examples.optimization import run_110_transferred_benchmark as runner


def fake_evaluations():
    results, forecasts = {}, {}
    for i, name in enumerate(runner.EXPECTED_MODELS):
        values = runner.HOLDOUT_VALUES + (i + 1) * 0.1
        results[name] = SimpleNamespace(
            model=name, evaluation_steps=5,
            dates=tuple(runner.HOLDOUT_DATES),
            actual=tuple(runner.HOLDOUT_VALUES),
            predicted=tuple(values),
            rmse=float(np.sqrt(np.mean((values - runner.HOLDOUT_VALUES) ** 2))),
            mae=float(np.mean(abs(values - runner.HOLDOUT_VALUES))),
            mape=float(np.mean(abs((values - runner.HOLDOUT_VALUES) / runner.HOLDOUT_VALUES)) * 100),
        )
        forecasts[name] = SimpleNamespace(
            forecast=values.copy(), dates=tuple(runner.HOLDOUT_DATES)
        )
    return results, forecasts


def test_result_validation_accepts_aligned_frozen_holdout():
    runner.validate_execution_results(*fake_evaluations())


def test_result_validation_rejects_wrong_date():
    evaluations, forecasts = fake_evaluations()
    evaluations["ARIMA"].dates = tuple(pd.date_range("2024-03-01", periods=5, freq="QS-MAR"))
    with pytest.raises(ValueError, match="incorrect holdout dates"):
        runner.validate_execution_results(evaluations, forecasts)


def test_result_validation_rejects_changed_actuals():
    evaluations, forecasts = fake_evaluations()
    evaluations["ARIMA"].actual = (999.,) + evaluations["ARIMA"].actual[1:]
    with pytest.raises(AssertionError, match="altered holdout"):
        runner.validate_execution_results(evaluations, forecasts)


def test_result_validation_rejects_missing_model():
    evaluations, forecasts = fake_evaluations()
    del evaluations["GRU"]
    with pytest.raises(ValueError, match="model set"):
        runner.validate_execution_results(evaluations, forecasts)


def test_result_validation_rejects_nonfinite_prediction():
    evaluations, forecasts = fake_evaluations()
    values = list(evaluations["GRU"].predicted)
    values[0] = float("nan")
    evaluations["GRU"].predicted = tuple(values)
    with pytest.raises(ValueError, match="invalid holdout predictions"):
        runner.validate_execution_results(evaluations, forecasts)


def test_result_validation_rejects_forecast_mismatch():
    evaluations, forecasts = fake_evaluations()
    forecasts["LSTM"].forecast[0] += 1.
    with pytest.raises(AssertionError, match="forecast/evaluation disagreement"):
        runner.validate_execution_results(evaluations, forecasts)


def test_runner_calls_benchmark_once_and_saves_existing_objects(monkeypatch, tmp_path):
    evaluations, forecasts = fake_evaluations()
    dataset = object()
    specs = tuple(SimpleNamespace(name=name) for name in runner.EXPECTED_MODELS)
    calls = {"preflight": 0, "evaluate": 0, "save": 0}
    def preflight(d):
        assert d is dataset
        calls["preflight"] += 1
    class Benchmark:
        def __init__(self, evaluator):
            assert evaluator == "evaluator"
        def evaluate_with_forecasts(self, *, dataset, specs):
            calls["evaluate"] += 1
            return evaluations, forecasts
    def save(e, f, s, *, output_dir):
        assert e is evaluations and f is forecasts and s is specs
        calls["save"] += 1
        return {"benchmark_json": tmp_path / "benchmark.json"}
    monkeypatch.setattr(runner, "verify_preflight", preflight)
    monkeypatch.setattr(runner, "make_evaluator", lambda: "evaluator")
    monkeypatch.setattr(runner, "OptimizedBenchmark", Benchmark)
    monkeypatch.setattr(runner, "save_execution_artifacts", save)
    result = runner.run_benchmark(dataset=dataset, specs=specs, output_dir=tmp_path)
    assert result.evaluations is evaluations
    assert result.forecasts is forecasts
    assert calls == {"preflight": 1, "evaluate": 1, "save": 1}
