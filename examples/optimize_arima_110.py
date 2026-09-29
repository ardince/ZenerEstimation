"""
Real-data ARIMA optimization demonstration.

Battery:
    732B-5610110

Workflow:
    processed dataset
        -> final benchmark boundary
        -> development-only temporal optimization
        -> selected ARIMA order
        -> untouched final benchmark evaluation
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Non-invertible starting MA "
        "parameters found.*"
    ),
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "Non-stationary starting "
        "autoregressive parameters found.*"
    ),
    category=UserWarning,
)

from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal.preprocessor import (
    TemporalPreprocessor,
)
from zenerestimation.evaluation.evaluator import (
    ForecastEvaluator,
)
from zenerestimation.forecasting.arima import (
    ARIMAForecaster,
)
from zenerestimation.optimization import (
    ARIMAParameterSpace,
    BenchmarkBoundary,
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
    OptimizationArtifact,
)

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATTERY = "732B-5610110"

DATA_PATH = Path(
    "datasets/processed/732B-5610110.csv"
)

EVALUATION_STEPS = 5
EVALUATION_END = "2024-03-01"

INTERNAL_FOLDS = 3
INTERNAL_VALIDATION_STEPS = EVALUATION_STEPS

SELECTION_METRIC = "rmse"


# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

dataset = BatteryDataset.from_processed_csv(
    DATA_PATH,
    battery=BATTERY,
)


# ---------------------------------------------------------
# Dataset identity guard
# ---------------------------------------------------------

KNOWN_110_CHECKPOINTS = {
    "2022-12-01": 166.61,
    "2023-03-01": 167.17,
    "2024-03-01": 169.90,
}

for date, expected in (
    KNOWN_110_CHECKPOINTS.items()
):

    timestamp = pd.Timestamp(
        date
    )

    matches = dataset.data.loc[
        dataset.data["ds"] == timestamp,
        "microVolt",
    ]

    assert len(matches) == 1, (
        f"Expected exactly one observation "
        f"for {date}, found {len(matches)}."
    )

    actual = float(
        matches.iloc[0]
    )

    assert np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-9,
    ), (
        "Dataset identity check failed for "
        f"{BATTERY}: {date} expected "
        f"{expected:.6f}, found {actual:.6f}. "
        f"Check DATA_PATH: {DATA_PATH}"
    )


print()
print("=" * 70)
print("ARIMA OPTIMIZATION — 732B-5610110")
print("=" * 70)

print()
print("Complete processed dataset")
print("-" * 70)

print(
    f"Rows             : {len(dataset)}"
)

print(
    f"Start            : "
    f"{dataset.data['ds'].iloc[0].date()}"
)

print(
    f"End              : "
    f"{dataset.data['ds'].iloc[-1].date()}"
)


# ---------------------------------------------------------
# Final benchmark boundary
# ---------------------------------------------------------

boundary = BenchmarkBoundary(
    evaluation_steps=EVALUATION_STEPS,
    evaluation_end=EVALUATION_END,
)

development = (
    boundary.development_dataset(
        dataset
    )
)

benchmark = (
    boundary.benchmark_data(
        dataset
    )
)

print()
print("Benchmark boundary")
print("-" * 70)

print(
    f"Development rows : {len(development)}"
)

print(
    f"Development end  : "
    f"{development.data['ds'].iloc[-1].date()}"
)

print(
    f"Benchmark rows   : {len(benchmark)}"
)

print(
    f"Benchmark start  : "
    f"{benchmark['ds'].iloc[0].date()}"
)

print(
    f"Benchmark end    : "
    f"{benchmark['ds'].iloc[-1].date()}"
)


# ---------------------------------------------------------
# Leakage-safe training preprocessing
# ---------------------------------------------------------

preprocessor = TemporalPreprocessor(
    method="linear",
    fill_edges=True,
)


# ---------------------------------------------------------
# Internal temporal validation
# ---------------------------------------------------------

splitter = ExpandingWindowSplitter(
    folds=INTERNAL_FOLDS,
    validation_steps=INTERNAL_VALIDATION_STEPS,
)

optimization_evaluator = (
    OptimizationEvaluator(
        splitter,
        metric=SELECTION_METRIC,
        preprocessor=preprocessor,
    )
)


# ---------------------------------------------------------
# ARIMA search space
# ---------------------------------------------------------

space = ARIMAParameterSpace(
    p=(
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        12,
    ),
    d=(
        0,
        1,
    ),
    q=(
        0,
        1,
        2,
        3,
    ),
)

print()
print("Optimization configuration")
print("-" * 70)

print(
    f"Candidates       : "
    f"{space.candidate_count}"
)

print(
    f"Internal folds   : "
    f"{INTERNAL_FOLDS}"
)

print(
    f"Validation steps : "
    f"{INTERNAL_VALIDATION_STEPS}"
)

print(
    f"Selection metric : "
    f"{SELECTION_METRIC.upper()}"
)


# ---------------------------------------------------------
# Optimize on development data ONLY
# ---------------------------------------------------------

optimizer = GenericOptimizer(
    optimization_evaluator,
    model="ARIMA",
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning,
    )

    optimization = optimizer.optimize(
        dataset=development,
        model_factory=ARIMAForecaster,
        candidates=space.candidates(),
    )


    optimization_artifact = OptimizationArtifact(
        optimization,
        benchmark_steps=EVALUATION_STEPS,
    )


print()
print("Candidate execution summary")
print("-" * 70)

print(
    f"Attempted        : "
    f"{optimization.candidate_count}"
)

print(
    f"Evaluated        : "
    f"{optimization.successful_count}"
)

print(
    f"Failed           : "
    f"{optimization.failed_count}"
)


# ---------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------

evaluated_candidates = [
    candidate
    for candidate in optimization.candidates
    if candidate.status == "evaluated"
]

failed_candidates = [
    candidate
    for candidate in optimization.candidates
    if candidate.status == "failed"
]

if failed_candidates:

    print()
    print("Failed candidates")
    print("-" * 70)

    for candidate in failed_candidates:

        print(
            f"{candidate.params}  "
            f"{candidate.metadata.get('error_type', 'UnknownError')}: "
            f"{candidate.metadata.get('error_message', '')}"
        )


assert evaluated_candidates, (
    "ARIMA optimization produced no "
    "successfully evaluated candidates."
)

ranked = sorted(
    evaluated_candidates,
    key=lambda candidate: candidate.score,
)

print()
print("Top ARIMA candidates")
print("-" * 70)

print(
    f"{'Rank':<6}"
    f"{'Order':<18}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'MAPE':>12}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):

    order = candidate.params[
        "order"
    ]

    print(
        f"{rank:<6}"
        f"{str(order):<18}"
        f"{candidate.metrics['rmse']:>12.6f}"
        f"{candidate.metrics['mae']:>12.6f}"
        f"{candidate.metrics['mape']:>12.6f}"
    )


print()
print("Selected ARIMA configuration")
print("-" * 70)

print(
    f"Best parameters  : "
    f"{optimization.best_params}"
)

print(
    f"Internal RMSE    : "
    f"{optimization.best_score:.6f}"
)


# ---------------------------------------------------------
# Final benchmark evaluation
# ---------------------------------------------------------

selected_model = ARIMAForecaster(
    **optimization.best_params
)

benchmark_evaluator = ForecastEvaluator(
    evaluation_steps=EVALUATION_STEPS,
    preprocessor=preprocessor,
    evaluation_end=EVALUATION_END,
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning,
    )

    evaluation = benchmark_evaluator.evaluate(
        dataset=dataset,
        model=selected_model,
    )


print()
print("Final untouched benchmark")
print("-" * 70)

print(
    f"RMSE             : "
    f"{evaluation.rmse:.6f}"
)

print(
    f"MAE              : "
    f"{evaluation.mae:.6f}"
)

print(
    f"MAPE             : "
    f"{evaluation.mape:.6f}%"
)


print()
print("Benchmark predictions")
print("-" * 70)

print(
    f"{'Date':<14}"
    f"{'Actual':>12}"
    f"{'Predicted':>14}"
    f"{'Error':>14}"
)

for (
    date,
    actual,
    predicted,
) in zip(
    evaluation.dates,
    evaluation.actual,
    evaluation.predicted,
):

    error = (
        predicted
        - actual
    )

    print(
        f"{str(date)[:10]:<14}"
        f"{actual:>12.6f}"
        f"{predicted:>14.6f}"
        f"{error:>14.6f}"
    )

print()
print("=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)


expected_dates = [
    "2023-03-01",
    "2023-06-01",
    "2023-09-01",
    "2023-12-01",
    "2024-03-01",
]

expected_values = [
    167.17,
    167.73,
    168.30,
    168.86,
    169.90,
]

actual_dates = (
    benchmark["ds"]
    .dt.strftime("%Y-%m-%d")
    .tolist()
)

actual_values = (
    benchmark["microVolt"]
    .astype(float)
    .tolist()
)

assert actual_dates == expected_dates

for actual, expected in zip(
    actual_values,
    expected_values,
):
    assert abs(actual - expected) < 1e-9

    assert development.data["ds"].max() < pd.Timestamp(
        "2023-03-01"
    )

    assert benchmark["ds"].max() == pd.Timestamp(
        "2024-03-01"
    )

    assert not (
        development.data["ds"]
        > pd.Timestamp(EVALUATION_END)
    ).any()

    assert not (
        benchmark["ds"]
        > pd.Timestamp(EVALUATION_END)
    ).any()


print()
print("Selected candidate fold metrics")
print("-" * 70)

selected_candidate = next(
    candidate
    for candidate in optimization.candidates
    if candidate.params == optimization.best_params
)

for index, metrics in enumerate(
    selected_candidate.metadata["fold_metrics"],
    start=1,
):
    print(
        f"Fold {index}: "
        f"RMSE={metrics['rmse']:.6f}  "
        f"MAE={metrics['mae']:.6f}  "
        f"MAPE={metrics['mape']:.6f}%"
    )


print()
print("Top candidate fold RMSE")
print("-" * 90)

print(
    f"{'Rank':<6}"
    f"{'Order':<18}"
    f"{'Fold 1':>12}"
    f"{'Fold 2':>12}"
    f"{'Fold 3':>12}"
    f"{'Mean':>12}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):
    fold_rmse = [
        fold["rmse"]
        for fold
        in candidate.metadata["fold_metrics"]
    ]

    print(
        f"{rank:<6}"
        f"{str(candidate.params['order']):<18}"
        f"{fold_rmse[0]:>12.6f}"
        f"{fold_rmse[1]:>12.6f}"
        f"{fold_rmse[2]:>12.6f}"
        f"{candidate.score:>12.6f}"
    )


# ---------------------------------------------------------
# Standardized result files
# ---------------------------------------------------------

result_files = create_result_files(
    battery=BATTERY,
    model="arima",
    # include any other arguments required by the
    # EXISTING create_result_files() contract
)


# ---------------------------------------------------------
# Write optimization artifact
# ---------------------------------------------------------

optimization_path = (
    result_files.directory
    / "optimization.json"
)

save_metadata(
    optimization_path,
    optimization_artifact.to_dict(),
)

print(
    f"Optimization artifact: "
    f"{optimization_path}"
)