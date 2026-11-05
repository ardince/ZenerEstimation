"""F.6.2: reusable runner results without additional forecasting calls."""
from types import SimpleNamespace

from examples.optimization import run_410_optimized_benchmark as benchmark
from examples.optimization import run_410_operational_extrapolation as operational

import pandas as pd

def test_benchmark_run_returns_exact_computed_objects(monkeypatch):
    dataset = object()
    specs = (object(),)
    evaluations = {'model': object()}
    forecasts = {'model': object()}

    dates = pd.Series(
        pd.to_datetime([
            "2022-03-01",
            "2022-06-01",
            "2022-09-01",
            "2022-12-01",
        ]),
        name="ds",
    )

    paths = {'report': object()}
    calls = {'evaluate': 0, 'save': 0}

    class Preprocessor:
        def fit_transform(self, training):
            return SimpleNamespace(data={'ds': dates})

    class Evaluator:
        def __init__(self, **kwargs):
            pass

        def split_dataset(self, dataset_arg):
            assert dataset_arg is dataset
            return object(), object()

    class Runner:
        def __init__(self, evaluator):
            pass

        def evaluate_with_forecasts(self, *, dataset, specs):
            calls['evaluate'] += 1
            return evaluations, forecasts

    def save(*args):
        calls['save'] += 1
        assert args[1] is evaluations
        assert args[2] is forecasts

        pd.testing.assert_series_equal(
            args[3],
            dates,
        )

        return paths

    monkeypatch.setattr(benchmark, 'load_dataset', lambda: dataset)
    monkeypatch.setattr(benchmark, 'create_model_specs', lambda: specs)
    monkeypatch.setattr(benchmark, 'TemporalPreprocessor', Preprocessor)
    monkeypatch.setattr(benchmark, 'ForecastEvaluator', Evaluator)
    monkeypatch.setattr(benchmark, 'OptimizedBenchmark', Runner)
    monkeypatch.setattr(benchmark, 'save_benchmark_artifacts', save)
    monkeypatch.setattr(benchmark, 'print_dataset_summary', lambda *a: None)
    monkeypatch.setattr(benchmark, 'print_frozen_specs', lambda *a: None)
    monkeypatch.setattr(benchmark, 'print_results', lambda *a: None)
    monkeypatch.setattr(benchmark, 'print_forecasts', lambda *a: None)

    result = benchmark.run_benchmark()
    assert isinstance(result, benchmark.BenchmarkRun)
    assert result.dataset is dataset
    assert result.evaluations is evaluations
    assert result.forecast_results is forecasts

    pd.testing.assert_series_equal(
        result.training_dates,
        dates,
    )

    assert result.training_dates is not dates

    assert result.artifact_paths is paths
    assert calls == {'evaluate': 1, 'save': 1}


def test_operational_run_returns_exact_computed_objects(monkeypatch):
    dataset = object()
    training = object()
    specs = (SimpleNamespace(name='ARIMA'),)
    forecasts = {'ARIMA': object()}
    paths = {'report': SimpleNamespace(read_text=lambda **kwargs: 'report')}
    calls = {'compute': 0, 'save': 0}

    def compute(t, s):
        calls['compute'] += 1
        assert t is training and s is specs
        return forecasts

    def save(t, f, s, **kwargs):
        calls['save'] += 1
        assert t is training and f is forecasts and s is specs
        assert kwargs['require_expected_models'] is True
        return paths

    monkeypatch.setattr(operational, 'load_dataset', lambda: dataset)
    monkeypatch.setattr(operational, 'prepare_operational_training', lambda d: training)
    monkeypatch.setattr(operational, 'create_model_specs', lambda: specs)
    monkeypatch.setattr(operational, 'compute_forecasts', compute)
    monkeypatch.setattr(operational, 'save_outputs', save)

    result = operational.run_operational_extrapolation()
    assert isinstance(result, operational.OperationalRun)
    assert result.training is training
    assert result.forecasts is forecasts
    assert result.specs is specs
    assert result.artifact_paths is paths
    assert calls == {'compute': 1, 'save': 1}


def test_main_delegates_without_rerunning(monkeypatch):
    calls = {'benchmark': 0, 'operational': 0}

    def benchmark_run():
        calls['benchmark'] += 1

    def operational_run():
        calls['operational'] += 1

    monkeypatch.setattr(benchmark, 'run_benchmark', benchmark_run)
    monkeypatch.setattr(operational, 'run_operational_extrapolation', operational_run)
    benchmark.main()
    operational.main()
    assert calls == {'benchmark': 1, 'operational': 1}
