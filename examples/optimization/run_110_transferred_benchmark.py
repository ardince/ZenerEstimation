"""Sprint 14.9G.3.1: preflight for the frozen 410-to-110 transfer benchmark.

Run from the repository root:
    python -m examples.optimization.run_110_transferred_benchmark

Default mode is preflight-only. Use --execute to run the frozen benchmark.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.evaluator import ForecastEvaluator
from examples.optimization.run_410_optimized_benchmark import create_model_specs
from zenerestimation.evaluation.benchmark import OptimizedBenchmark
from zenerestimation.evaluation.benchmark_integration import comparison_from_evaluations
from zenerestimation.optimization import BenchmarkArtifact, BenchmarkReport
from zenerestimation.visualization import BenchmarkPlotter

BATTERY = "732B-5610110"
SOURCE_BATTERY = "732B-5610410"
DATA_PATH = Path("datasets/processed") / f"{BATTERY}.csv"
OUTPUT_DIR = Path("results") / BATTERY / "transferred_benchmark"
TRAINING_END = pd.Timestamp("2022-12-01")
EVALUATION_END = pd.Timestamp("2024-03-01")
HOLDOUT_DATES = pd.date_range("2023-03-01", periods=5, freq="QS-MAR")
HOLDOUT_VALUES = np.array([167.17, 167.73, 168.30, 168.86, 169.90])
EXPECTED_MODELS = (
    "ARIMA", "Kalman", "LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM"
)


def load_dataset(path=DATA_PATH):
    """Read the unmodified canonical processed dataset."""
    return BatteryDataset.from_processed_csv(Path(path), battery=BATTERY)


def create_transferred_specs():
    """Use the exact 410 factories and frozen parameters, without retuning."""
    specs = create_model_specs()
    if tuple(spec.name for spec in specs) != EXPECTED_MODELS:
        raise ValueError("410 frozen model specification names have changed")
    return specs


def make_evaluator():
    return ForecastEvaluator(
        evaluation_steps=len(HOLDOUT_DATES),
        evaluation_end=EVALUATION_END,
        preprocessor=TemporalPreprocessor(),
    )


def verify_preflight(dataset, *, evaluator=None):
    """Validate the exact approved split before any model fit or prediction.

    The returned training set is raw; preprocessing is explicitly training-only.
    """
    if evaluator is None:
        evaluator = make_evaluator()
    frame = dataset.data
    required = {"ds", "microVolt", "is_observed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["ds"], errors="raise"))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Canonical timestamps must be sorted and unique")
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="QS-MAR")):
        raise ValueError("Canonical timeline is not complete and quarterly")
    if frame["is_observed"].isna().any() or not frame["is_observed"].isin([True, False]).all():
        raise ValueError("Invalid observation flags")
    observed = frame["is_observed"].astype(bool).to_numpy()
    finite = np.isfinite(pd.to_numeric(frame["microVolt"], errors="coerce").to_numpy(dtype=float))
    if not np.array_equal(observed, finite):
        raise ValueError("Measured/unmeasured provenance disagrees with target values")

    training, holdout = evaluator.split_dataset(dataset)
    train_dates = pd.DatetimeIndex(training.data["ds"])
    held_dates = pd.DatetimeIndex(holdout["ds"])
    if train_dates[-1] != TRAINING_END or held_dates.tolist() != HOLDOUT_DATES.tolist():
        raise ValueError("Frozen 110 training/holdout dates do not match")
    if (train_dates > TRAINING_END).any() or (held_dates > EVALUATION_END).any():
        raise ValueError("Future observation leaked into benchmark")
    if not holdout["is_observed"].astype(bool).all():
        raise ValueError("Benchmark holdout includes unmeasured targets")
    np.testing.assert_allclose(
        holdout["microVolt"].to_numpy(dtype=float), HOLDOUT_VALUES,
        rtol=0, atol=1e-6, err_msg="Frozen measured holdout values changed",
    )
    prepared = evaluator.preprocessor.fit_transform(training)
    if not pd.DatetimeIndex(prepared.data["ds"]).equals(train_dates):
        raise ValueError("Preprocessing changed the training timeline")
    if prepared.data["microVolt"].isna().any():
        raise ValueError("Training preprocessing left missing values")
    if not np.array_equal(
        prepared.data["is_observed"].to_numpy(),
        training.data["is_observed"].to_numpy(),
    ):
        raise ValueError("Preprocessing changed observation provenance")
    return training, holdout, prepared


def preflight_manifest(specs):
    """Document the transfer decision; no 110 optimization is claimed."""
    return {
        "battery": BATTERY,
        "configuration_source_battery": SOURCE_BATTERY,
        "configuration_source_module": "examples.optimization.run_410_optimized_benchmark",
        "experiment_type": "frozen_cross_battery_parameter_transfer",
        "optimized_for_dataset_110": False,
        "benchmark_executed": False,
        "training_end": TRAINING_END.date().isoformat(),
        "holdout_dates": [d.date().isoformat() for d in HOLDOUT_DATES],
        "holdout_steps": len(HOLDOUT_DATES),
        "neural_benchmark_seed": 42,
        "stability_seeds": [0, 1, 2, 3, 4],
        "models": [
            {"name": spec.name, "params": dict(spec.params)} for spec in specs
        ],
        "note": "Preflight only: no fit/predict and no holdout-based selection.",
    }


@dataclass(frozen=True)
class TransferredBenchmarkRun:
    evaluations: dict
    forecasts: dict
    paths: dict


def validate_execution_results(evaluations, forecasts):
    """Reject incomplete, misaligned, or nonfinite model results."""
    if set(evaluations) != set(EXPECTED_MODELS):
        raise ValueError("Benchmark evaluation model set differs from frozen six")
    if set(forecasts) != set(EXPECTED_MODELS):
        raise ValueError("Forecast result model set differs from frozen six")
    for name in EXPECTED_MODELS:
        result = evaluations[name]
        if result.model != name or result.evaluation_steps != len(HOLDOUT_DATES):
            raise ValueError(f"{name}: invalid evaluation identity or horizon")
        if pd.DatetimeIndex(result.dates).tolist() != HOLDOUT_DATES.tolist():
            raise ValueError(f"{name}: incorrect holdout dates")
        np.testing.assert_allclose(
            np.asarray(result.actual, dtype=float),
            HOLDOUT_VALUES, rtol=0, atol=1e-6,
            err_msg=f"{name}: altered holdout measurements",
        )
        predicted = np.asarray(result.predicted, dtype=float)
        if len(predicted) != len(HOLDOUT_DATES) or not np.isfinite(predicted).all():
            raise ValueError(f"{name}: invalid holdout predictions")
        for metric in ("rmse", "mae", "mape"):
            if not np.isfinite(float(getattr(result, metric))):
                raise ValueError(f"{name}: nonfinite {metric}")
        forecast = forecasts[name]
        if len(np.asarray(forecast.forecast).reshape(-1)) != len(HOLDOUT_DATES):
            raise ValueError(f"{name}: forecast length mismatch")
        if pd.DatetimeIndex(forecast.dates).tolist() != HOLDOUT_DATES.tolist():
            raise ValueError(f"{name}: forecast date mismatch")
        np.testing.assert_allclose(
            np.asarray(forecast.forecast, dtype=float),
            predicted, rtol=0, atol=1e-8,
            err_msg=f"{name}: forecast/evaluation disagreement",
        )


def save_execution_artifacts(evaluations, forecasts, specs, output_dir=OUTPUT_DIR):
    """Persist the already-computed results; never fit or predict again."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = comparison_from_evaluations(BATTERY, evaluations)
    standard = BenchmarkArtifact(comparison).save(output_dir)
    report_path = BenchmarkReport(comparison, evaluations=evaluations).save(
        output_dir / "benchmark.txt"
    )
    figure_path = BenchmarkPlotter(evaluations).save(
        output_dir / "benchmark_forecast.png"
    )
    rows = []
    for name in EXPECTED_MODELS:
        result = evaluations[name]
        errors = np.asarray(result.predicted, dtype=float) - HOLDOUT_VALUES
        rows.append({
            "model": name,
            "rmse": float(result.rmse),
            "mae": float(result.mae),
            "mape": float(result.mape),
            "first_point_gap_pred_minus_actual": float(errors[0]),
            "bias_pred_minus_actual": float(errors.mean()),
        })
    metrics_path = output_dir / "extended_metrics.csv"
    import pandas as _pd
    _pd.DataFrame(rows).to_csv(metrics_path, index=False)
    predictions_path = output_dir / "holdout_predictions.csv"
    table = _pd.DataFrame({
        "date": HOLDOUT_DATES.strftime("%Y-%m-%d"),
        "measured_microVolt": HOLDOUT_VALUES,
    })
    for name in EXPECTED_MODELS:
        table[name] = np.asarray(evaluations[name].predicted, dtype=float)
    table.to_csv(predictions_path, index=False, float_format="%.12g")
    provenance = preflight_manifest(specs)
    provenance["benchmark_executed"] = True
    provenance["note"] = (
        "Frozen cross-battery transfer; no dataset-110 parameter or seed selection. "
        "Holdout evaluation is distinct from operational extrapolation."
    )
    provenance["artifacts"] = {
        "benchmark": "benchmark.json",
        "extended_metrics": metrics_path.name,
        "holdout_predictions": predictions_path.name,
    }
    provenance_path = output_dir / "transfer_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, default=str) + "\\n",
        encoding="utf-8",
    )
    return {
        "benchmark_json": standard["json"],
        "benchmark_csv": standard["csv"],
        "report": report_path,
        "figure": figure_path,
        "extended_metrics": metrics_path,
        "predictions": predictions_path,
        "provenance": provenance_path,
    }


def run_benchmark(*, dataset=None, specs=None, output_dir=OUTPUT_DIR):
    """Preflight, evaluate the frozen six once each, then save existing results."""
    if dataset is None:
        dataset = load_dataset()
    if specs is None:
        specs = create_transferred_specs()
    if tuple(spec.name for spec in specs) != EXPECTED_MODELS:
        raise ValueError("Frozen model specifications changed")
    verify_preflight(dataset)
    evaluator = make_evaluator()
    evaluations, forecasts = OptimizedBenchmark(
        evaluator=evaluator
    ).evaluate_with_forecasts(dataset=dataset, specs=specs)
    validate_execution_results(evaluations, forecasts)
    paths = save_execution_artifacts(
        evaluations, forecasts, specs, output_dir=output_dir
    )
    return TransferredBenchmarkRun(
        evaluations=evaluations, forecasts=forecasts, paths=paths
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="Fit the six frozen models and save measured-holdout benchmark evidence",
    )
    args = parser.parse_args()
    if args.execute:
        result = run_benchmark()
        print("Dataset-110 transferred benchmark: COMPLETE")
        for key, path in result.paths.items():
            print(f"{key}: {path}")
        return
    dataset = load_dataset()
    specs = create_transferred_specs()
    training, holdout, prepared = verify_preflight(dataset)
    manifest = preflight_manifest(specs)
    print("Dataset-110 transferred benchmark preflight: PASS")
    print(f"Training: {len(training)} canonical quarters, through {TRAINING_END.date()}")
    print(f"Holdout: {len(holdout)} measured quarters, through {EVALUATION_END.date()}")
    print(f"Training-only preprocessing: {len(prepared)} quarters")
    print(json.dumps(manifest, indent=2, default=str))
    print("No models were fitted or used to predict.")


if __name__ == "__main__":
    main()
