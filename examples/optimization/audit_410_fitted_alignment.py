
"""Audit historical fitted-value alignment for dataset 410.

Run from the repository root:

    python -m examples.optimization.audit_410_fitted_alignment

Diagnostic only:
- No optimization or parameter selection.
- One fit and forecast per frozen model.
- No official benchmark artifacts are modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.benchmark import OptimizedBenchmark
from zenerestimation.evaluation.evaluator import ForecastEvaluator

from examples.optimization.run_410_optimized_benchmark import (
    BENCHMARK_STEPS,
    DEVELOPMENT_END,
    BENCHMARK_START,
    BENCHMARK_END,
    create_model_specs,
    load_dataset,
)


EXPECTED_WARMUP = {
    "ARIMA": None,  # Initialization-dependent
    "Kalman": 0,
    "LSTM": 4,
    "GRU": 4,
    "LinearTrendLSTM": 12,
    "KalmanLSTM": 8,
}


def audit_alignment(
    original,
    prepared_training,
    evaluations,
    forecasts,
):
    """Check historical fitted evidence and forecast boundaries."""

    if set(evaluations) != set(EXPECTED_WARMUP):
        raise ValueError("Unexpected evaluation model set")

    if set(forecasts) != set(evaluations):
        raise ValueError("Forecast model set mismatch")

    raw = original.data.copy()
    raw_dates = pd.DatetimeIndex(pd.to_datetime(raw["ds"]))

    training = prepared_training.data.copy()
    training_dates = pd.DatetimeIndex(
        pd.to_datetime(training["ds"])
    )

    if (
        training_dates.hasnans
        or not training_dates.is_unique
        or not training_dates.is_monotonic_increasing
    ):
        raise ValueError("Invalid training timeline")

    expected_end = pd.Timestamp(DEVELOPMENT_END)
    expected_start = pd.Timestamp(BENCHMARK_START)
    expected_benchmark_end = pd.Timestamp(BENCHMARK_END)

    if training_dates[-1] != expected_end:
        raise ValueError("Unexpected training end")

    if not training_dates.equals(
        raw_dates[raw_dates <= expected_end]
    ):
        raise ValueError(
            "Prepared training dates differ from raw development dates"
        )

    records = []

    for name, expected_warmup in EXPECTED_WARMUP.items():
        evaluation = evaluations[name]
        result = forecasts[name]

        fitted = result.fitted

        if fitted is None:
            raise ValueError(f"{name}: fitted is unavailable")

        values = np.asarray(fitted, dtype=float)

        if values.ndim != 1:
            raise ValueError(f"{name}: fitted must be 1D")

        length_ok = len(values) == len(training_dates)

        leading_nan = 0
        for value in values:
            if not np.isnan(value):
                break
            leading_nan += 1

        warmup_ok = (
            True
            if expected_warmup is None
            else leading_nan == expected_warmup
        )

        # ARIMA's initialized observations are examined
        # separately rather than treated as neural warm-up.
        interior_nan = np.isnan(
            values[leading_nan:]
        ).any()

        finite_after_warmup = (
            not interior_nan
            and not np.isinf(values).any()
        )

        index_mode = "array"
        index_ok = False

        if isinstance(fitted, pd.Series):
            if isinstance(fitted.index, pd.DatetimeIndex):
                index_mode = "calendar"
                index_ok = fitted.index.equals(training_dates)

            elif fitted.index.equals(training.index):
                index_mode = "training-positional"
                index_ok = True

            elif fitted.index.equals(
                pd.RangeIndex(len(values))
            ):
                index_mode = "zero-based-positional"
                index_ok = True

            else:
                index_mode = "unrecognized-index"

        else:
            # An ndarray carries no explicit index.
            # Position is its only available alignment.
            index_mode = "implicit-positional"
            index_ok = length_ok

        evaluation_dates = pd.DatetimeIndex(
            pd.to_datetime(evaluation.dates)
        )
        result_dates = pd.DatetimeIndex(
            pd.to_datetime(result.dates)
        )

        dates_ok = (
            result_dates.equals(evaluation_dates)
            and len(result_dates) == BENCHMARK_STEPS
            and result_dates[0] == expected_start
            and result_dates[-1] == expected_benchmark_end
            and training_dates[-1] < result_dates[0]
        )

        predictions_ok = np.array_equal(
            np.asarray(result.forecast, dtype=float),
            np.asarray(evaluation.predicted, dtype=float),
        )

        passed = all((
            length_ok,
            warmup_ok,
            finite_after_warmup,
            index_ok,
            dates_ok,
            predictions_ok,
        ))

        records.append({
            "Model": name,
            "Train": len(training_dates),
            "Fitted": len(values),
            "Leading NaNs": leading_nan,
            "Index mode": index_mode,
            "Length OK": length_ok,
            "Index OK": index_ok,
            "Warm-up OK": warmup_ok,
            "Finite OK": finite_after_warmup,
            "Dates OK": dates_ok,
            "Predictions OK": predictions_ok,
            "PASS": passed,
        })

    report = pd.DataFrame.from_records(records)

    print("\n410 FITTED-ALIGNMENT AUDIT")
    print("=" * 100)
    print(report.to_string(index=False))

    print("\nTraining:")
    print("First date:", training_dates[0].date())
    print("Last date :", training_dates[-1].date())
    print("Rows      :", len(training_dates))

    print("\nOverall:", (
        "PASS" if report["PASS"].all()
        else "REQUIRES INVESTIGATION"
    ))

    return report


def main():
    dataset = load_dataset()

    evaluator = ForecastEvaluator(
        evaluation_steps=BENCHMARK_STEPS,
        preprocessor=TemporalPreprocessor(),
    )

    # Reconstruct the exact training-only preparation
    # contract, independently of benchmark execution.
    training, validation = evaluator.split_dataset(dataset)

    prepared_training = evaluator.preprocessor.fit_transform(
        training
    )

    # Verify that preprocessing did not change the dates.
    before_dates = pd.DatetimeIndex(
        pd.to_datetime(training.data["ds"])
    )
    after_dates = pd.DatetimeIndex(
        pd.to_datetime(prepared_training.data["ds"])
    )

    if not before_dates.equals(after_dates):
        raise ValueError(
            "Temporal preprocessing changed the timeline"
        )

    if len(validation) != BENCHMARK_STEPS:
        raise ValueError("Unexpected holdout length")

    benchmark = OptimizedBenchmark(evaluator)

    evaluations, forecasts = benchmark.evaluate_with_forecasts(
        dataset=dataset,
        specs=create_model_specs(),
    )

    report = audit_alignment(
        dataset,
        prepared_training,
        evaluations,
        forecasts,
    )

    if not report["PASS"].all():
        raise SystemExit(
            "Fitted-alignment audit requires investigation"
        )


if __name__ == "__main__":
    main()
