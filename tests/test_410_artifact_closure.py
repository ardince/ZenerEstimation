"""Closure tests for dataset-410 benchmark and operational evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.optimization import run_410_operational_extrapolation as operational
from examples.optimization import run_410_optimized_benchmark as benchmark


EXPECTED_MODELS = (
    "ARIMA",
    "Kalman",
    "LSTM",
    "GRU",
    "LinearTrendLSTM",
    "KalmanLSTM",
)

EXPECTED_METRICS = {
    "ARIMA": {
        "rmse": 0.849441641452755,
        "mae": 0.7625092812293447,
        "mape": 2.427532722653989,
    },
    "Kalman": {
        "rmse": 0.9398293583288397,
        "mae": 0.8521409061757996,
        "mape": 2.71411335229159,
    },
    "LSTM": {
        "rmse": 1.3070269621826662,
        "mae": 1.1769761505126948,
        "mape": 3.746587349341602,
    },
    "GRU": {
        "rmse": 1.3954127141231591,
        "mae": 1.2603444519042963,
        "mape": 4.012699765601528,
    },
    "LinearTrendLSTM": {
        "rmse": 0.7881730653000166,
        "mae": 0.7272312048572029,
        "mape": 2.3199160712729974,
    },
    "KalmanLSTM": {
        "rmse": 0.6721611750049541,
        "mae": 0.5700114302412848,
        "mape": 1.8602854625097138,
    },
}


def _load_json(path: Path) -> dict:
    assert path.exists(), f"Required closure artifact is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_benchmark_boundaries_remain_closed():
    assert benchmark.BENCHMARK_STEPS == 6
    assert benchmark.DEVELOPMENT_END == "2023-12-01"
    assert benchmark.BENCHMARK_START == "2024-03-01"
    assert benchmark.BENCHMARK_END == "2025-06-01"
    assert benchmark.SEED == 42


def test_frozen_model_specifications_match_operational_model_set():
    specs = benchmark.create_model_specs()
    assert tuple(spec.name for spec in specs) == EXPECTED_MODELS
    assert operational.EXPECTED_MODELS == EXPECTED_MODELS
    for spec in specs:
        if "seed" in spec.params:
            assert spec.params["seed"] == 42


def test_saved_benchmark_metrics_match_accepted_reference():
    data = _load_json(benchmark.RESULT_DIRECTORY / "benchmark.json")
    assert data["status"] == "benchmarked"
    assert data["battery"] == benchmark.BATTERY
    assert tuple(data["models"]) == EXPECTED_MODELS

    for model, metrics in EXPECTED_METRICS.items():
        for metric, expected in metrics.items():
            assert data["metrics"][model][metric] == pytest.approx(
                expected, rel=0.0, abs=1e-12
            )

    assert data["best_models"] == {
        "rmse": "KalmanLSTM",
        "mae": "KalmanLSTM",
        "mape": "KalmanLSTM",
    }


def test_saved_benchmark_provenance_preserves_holdout_contract():
    data = _load_json(benchmark.RESULT_DIRECTORY / "provenance.json")
    metadata = data["metadata"]

    assert data["status"] == "linked"
    assert data["battery"] == benchmark.BATTERY
    assert metadata["benchmark_steps"] == 6
    assert metadata["development_end"] == "2023-12-01"
    assert metadata["benchmark_start"] == "2024-03-01"
    assert metadata["benchmark_end"] == "2025-06-01"
    assert metadata["neural_seed"] == 42
    assert metadata["seed_policy"] == "fixed_precommitted_seed"
    assert metadata["optimization_evidence"] == "copied_unchanged"


def test_operational_evidence_is_temporally_separate_from_benchmark():
    data = _load_json(operational.OUTPUT_DIR / "future_extrapolation.json")

    assert data["training_cutoff"] == benchmark.BENCHMARK_END
    assert data["forecast_start"] == "2025-09-01"
    assert data["forecast_end"] == "2026-12-01"
    assert data["horizon_quarters"] == 6
    assert tuple(data["models"]) == EXPECTED_MODELS
    assert data["prediction_count"] == 36

    benchmark_end = np.datetime64(benchmark.BENCHMARK_END)
    forecast_dates = np.asarray(data["forecast_dates"], dtype="datetime64[D]")
    assert np.all(forecast_dates > benchmark_end)


def test_operational_evidence_cannot_be_used_as_holdout_accuracy():
    data = _load_json(operational.OUTPUT_DIR / "future_extrapolation.json")

    assert data["kind"] == "operational_extrapolation_not_holdout_evaluation"
    assert data["scientific_status"] == (
        "unvalidated_future_operational_extrapolation"
    )
    assert data["configuration_source"] == "frozen_410_optimized_benchmark"
    assert data["model_update_policy"] == (
        "refit_through_training_cutoff_without_reoptimization"
    )
    assert data["measured_actuals"] is None
    assert data["forecast_errors"] is None
    assert data["evaluation_metrics"] is None
    assert data["uncertainty_intervals"] is None

    raw = json.dumps(data).lower()
    assert '"rmse"' not in raw
    assert '"mae"' not in raw
    assert '"mape"' not in raw


def test_operational_configurations_equal_frozen_benchmark_specs():
    data = _load_json(operational.OUTPUT_DIR / "future_extrapolation.json")
    frozen_specs = {
        spec.name: spec.params
        for spec in benchmark.create_model_specs()
    }

    for model in EXPECTED_MODELS:
        stored = data["model_configurations"][model]
        expected = frozen_specs[model]

        assert set(stored) == set(expected)
        for key, expected_value in expected.items():
            stored_value = stored[key]
            if isinstance(expected_value, tuple):
                assert tuple(stored_value) == expected_value
            else:
                assert stored_value == expected_value


def test_operational_predictions_are_complete_and_finite():
    data = _load_json(operational.OUTPUT_DIR / "future_extrapolation.json")

    assert set(data["predictions"]) == set(EXPECTED_MODELS)
    count = 0
    for model in EXPECTED_MODELS:
        values = np.asarray(data["predictions"][model], dtype=float)
        assert values.shape == (6,)
        assert np.isfinite(values).all()
        count += values.size

    assert count == 36
    assert count == data["prediction_count"]


def test_benchmark_and_operational_artifact_directories_are_distinct():
    benchmark_dir = benchmark.RESULT_DIRECTORY.resolve()
    operational_dir = operational.OUTPUT_DIR.resolve()

    assert benchmark_dir != operational_dir
    assert operational_dir not in benchmark_dir.parents
    assert benchmark_dir not in operational_dir.parents
